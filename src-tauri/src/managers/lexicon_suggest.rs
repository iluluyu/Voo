//! 自适应词库：从最近历史里本地提取「易混淆专名/术语」候选（Candidate）。
//!
//! 四态模型中的 Candidate 只出现在建议里，**不自动激活**——本模块
//! 只读历史与设置，不写任何状态。提升为热词由前端调用现有的
//! 「记住这条修正」命令（`shortcut::suggest_correction_to_hotword`）
//! 完成，本模块不重复实现写入。
//!
//! 算法（全本地，无网络）：
//! 1. 取最近 N 天历史的 `post_processed_text`（空则 `transcription_text`）；
//! 2. 抽取三类片段（[`extract_fragments`]）：
//!    ① ASCII 词：≥3 字符、含字母、非纯数字，保留原始大小写；
//!    ② 中英混排片段：同一汉字+拉丁连续段，长度 ≤20；
//!    ③ 纯汉字 n-gram（n=2..8），只在最长纯汉字子段内滑动；
//! 3. 统计总出现次数 `count` 与覆盖记录条数 `docs`（同一条记录内重复
//!    只计 1 次 docs，防单条刷量）；
//! 4. 过滤：命中内置常用词/虚词表（[`COMMON_HAN_WORDS`]/
//!    [`COMMON_ENGLISH_WORDS`]）直接排除；`docs / 总记录数 > 10%`
//!    视为高频常用排除；`count < 3` 或 `docs < 2` 排除；与既有热词
//!    （含别名）/Blocked 相等或互为子串/超串的候选排除；
//! 5. 按可混淆度打分（[`score_term`]，见常量）而非 count/docs 排序，
//!    输出 score 降序、同分 docs 升序（越少越特殊），最多 50 条。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::AppHandle;

use crate::audio_toolkit::chinese_hotwords::{is_han, parse_hotword_line};

/// 纯汉字 n-gram 最短长度
pub const MIN_NGRAM_LEN: usize = 2;
/// 纯汉字 n-gram 最长长度
pub const MAX_NGRAM_LEN: usize = 8;
/// ASCII 词最短长度（≥3 且必须含字母、非纯数字）
pub const MIN_ASCII_WORD_LEN: usize = 3;
/// 片段合法长度上限（也是中英混排片段上限）
pub const MAX_TERM_LEN: usize = 20;
/// 候选最小总出现次数
pub const MIN_COUNT: usize = 3;
/// 候选最小覆盖记录条数
pub const MIN_DOCS: usize = 2;
/// docs 占比上限（百分比）：docs/总记录数 > 10% 视为高频常用，排除
pub const MAX_DOCS_RATIO_PERCENT: u64 = 10;
/// 最多返回候选条数
pub const MAX_CANDIDATES: usize = 50;
/// 默认回看天数
pub const DEFAULT_DAYS: u32 = 30;
/// 回看天数上限（防御性钳制）
pub const MAX_DAYS: u32 = 3650;
/// 样例上下文最大字符数
pub const SAMPLE_MAX_CHARS: usize = 20;

// ===== 评分表（可混淆度） =====

/// 含拉丁字母（ASCII 词 / 中英混排）：+40
pub const SCORE_HAS_LATIN: i32 = 40;
/// 中英混排片段：+35
pub const SCORE_MIXED_HAN_LATIN: i32 = 35;
/// 纯汉字长度 3..=8 且不在常用表：+10
pub const SCORE_HAN_UNCOMMON: i32 = 10;
/// 纯汉字长度 ==2 或 >8：减分而非排除
pub const PENALTY_HAN_EDGE_LEN: i32 = -5;
/// 出现次数在 3..=15：+10（频次适中 = 真术语而非刷量/万能词）
pub const SCORE_COUNT_WINDOW: i32 = 10;
/// 出现记录条数 ≤5：+5（越少越特殊）
pub const SCORE_DOCS_RARE: i32 = 5;
/// `SCORE_COUNT_WINDOW` 的频次窗口下限
pub const COUNT_WINDOW_MIN: usize = 3;
/// `SCORE_COUNT_WINDOW` 的频次窗口上限
pub const COUNT_WINDOW_MAX: usize = 15;
/// `SCORE_DOCS_RARE` 的 docs 上限
pub const RARE_DOCS_MAX: usize = 5;

/// 内置常用词/虚词表（汉字部分，全部 ≥2 字）。候选与其任一词相等或
/// 互为子串/超串 → 直接排除。目的：把「的/了/我们/时候」这类高频
/// 通用词挡在建议外，把位置留给低频专名与术语。
pub const COMMON_HAN_WORDS: &[&str] = &[
    "的时候",
    "时候",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "自己",
    "大家",
    "这个",
    "那个",
    "这些",
    "那些",
    "这里",
    "那里",
    "哪里",
    "什么",
    "怎么",
    "为什么",
    "怎么样",
    "这样",
    "那样",
    "一样",
    "一些",
    "一下",
    "一个",
    "两个",
    "每个",
    "各种",
    "所有",
    "全部",
    "可以",
    "应该",
    "可能",
    "必须",
    "需要",
    "还是",
    "就是",
    "但是",
    "可是",
    "不过",
    "而且",
    "然后",
    "所以",
    "因为",
    "如果",
    "虽然",
    "或者",
    "以及",
    "对于",
    "关于",
    "通过",
    "根据",
    "按照",
    "已经",
    "曾经",
    "正在",
    "开始",
    "继续",
    "结束",
    "完成",
    "觉得",
    "感觉",
    "知道",
    "了解",
    "认为",
    "以为",
    "希望",
    "想要",
    "喜欢",
    "愿意",
    "能够",
    "直接",
    "简单",
    "非常",
    "特别",
    "真的",
    "确实",
    "其实",
    "当然",
    "肯定",
    "一定",
    "大概",
    "问题",
    "东西",
    "地方",
    "事情",
    "情况",
    "方面",
    "内容",
    "方法",
    "部分",
    "时间",
    "今天",
    "明天",
    "昨天",
    "现在",
    "以后",
    "之前",
    "上面",
    "下面",
    "前面",
    "后面",
    "里面",
    "外面",
    "一起",
    "很多",
    "许多",
    "而已",
    "的话",
    "如何",
    "任何",
    "聊天",
    "说话",
    "电话",
    "手机",
    "电脑",
];

/// 内置常用词表（英文部分，全小写）。ASCII 候选与其大小写不敏感相等
/// → 排除（用相等而非子串，避免误伤 Ising/makefile 这类真词）。
pub const COMMON_ENGLISH_WORDS: &[&str] = &[
    "the", "and", "but", "for", "not", "you", "your", "are", "was", "were", "been", "this", "that",
    "these", "those", "with", "from", "into", "about", "there", "here", "they", "them", "then",
    "than", "when", "what", "which", "where", "who", "how", "why", "will", "would", "could",
    "should", "just", "very", "really", "maybe", "please", "thanks", "thank", "sorry", "hello",
    "yeah", "okay", "also", "some", "more", "most", "much", "many", "like", "want", "need", "know",
    "think", "make", "made", "take", "get", "got", "going", "because", "since", "while", "after",
    "before", "again", "still", "only",
];

/// 一条候选词（四态模型中的 Candidate：仅展示，不参与纠错）。
#[derive(Clone, Debug, Serialize, Deserialize, Type)]
pub struct Candidate {
    /// 候选词（ASCII 词 / 中英混排片段 / 纯汉字 2..=8 字 n-gram）
    pub term: String,
    /// 窗口内总出现次数
    pub count: usize,
    /// 覆盖的不同历史记录条数
    pub docs: usize,
    /// 样例上下文（≤20 字，连续数字已脱敏）
    pub sample: String,
    /// 可混淆度得分（越高越值得加入热词），见模块头评分表
    pub score: i32,
}

/// 参与统计的一条历史文本（id 仅用于语义说明；docs 去重按记录天然完成）。
pub struct HistoryText {
    #[allow(dead_code)]
    pub id: i64,
    pub text: String,
}

/// 片段的三种类型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermKind {
    /// 纯 ASCII 词（含字母、非纯数字、保留大小写）
    Ascii,
    /// 中英混排（汉字+拉丁同段）
    Mixed,
    /// 纯汉字
    Han,
}

/// 词频中间态
struct TermStats {
    count: usize,
    docs: usize,
    sample: String,
}

/// 判定片段类型并校验合法性：总长 2..=20；ASCII 词另需 ≥3 且含字母；
/// 纯汉字需 2..=8；混排需同时含汉字与拉丁字母。纯数字/含其他字符 → None。
pub fn classify_term(term: &str) -> Option<TermKind> {
    let mut han = false;
    let mut latin = false;
    for c in term.chars() {
        if is_han(c) {
            han = true;
        } else if c.is_ascii_alphabetic() {
            latin = true;
        } else if !c.is_ascii_alphanumeric() {
            return None;
        }
    }
    let len = term.chars().count();
    match (han, latin) {
        (true, true) if len <= MAX_TERM_LEN => Some(TermKind::Mixed),
        (false, true) if (MIN_ASCII_WORD_LEN..=MAX_TERM_LEN).contains(&len) => {
            Some(TermKind::Ascii)
        }
        (true, false) if (MIN_NGRAM_LEN..=MAX_NGRAM_LEN).contains(&len) => Some(TermKind::Han),
        _ => None,
    }
}

/// 纯汉字子段内滑 n-gram（n=2..=8）。每个出现位置各产出一次
/// （count 按出现次数累计）。
fn push_han_ngrams(run: &[char], out: &mut Vec<String>) {
    let len = run.len();
    if len < MIN_NGRAM_LEN {
        return;
    }
    let max_n = MAX_NGRAM_LEN.min(len);
    for n in MIN_NGRAM_LEN..=max_n {
        for start in 0..=(len - n) {
            out.push(run[start..start + n].iter().collect());
        }
    }
}

/// ASCII 子词：≥3 字符且含字母（非纯数字），保留原始大小写。
fn push_ascii_word(word: &[char], out: &mut Vec<String>) {
    if word.len() >= MIN_ASCII_WORD_LEN && word.iter().any(|c| c.is_ascii_alphabetic()) {
        out.push(word.iter().collect());
    }
}

/// 把一个「汉字|ASCII 字母数字」连续段拆成候选片段：
/// - 段内同时有汉字与拉丁 → 整段作为中英混排片段（≤20 才收）；
/// - 汉字子段 → 纯汉字 n-gram（2..=8）；
/// - ASCII 子词 → ASCII 词（≥3、含字母、保留大小写）。
fn flush_run(run: &[char], out: &mut Vec<String>) {
    if run.is_empty() {
        return;
    }
    let has_han = run.iter().any(|c| is_han(*c));
    let has_latin = run.iter().any(|c| c.is_ascii_alphabetic());
    if has_han && has_latin && run.len() <= MAX_TERM_LEN {
        out.push(run.iter().collect());
    }
    let mut han_run: Vec<char> = Vec::new();
    let mut ascii_run: Vec<char> = Vec::new();
    for &c in run {
        if is_han(c) {
            push_ascii_word(&ascii_run, out);
            ascii_run.clear();
            han_run.push(c);
        } else if c.is_ascii_alphanumeric() {
            push_han_ngrams(&han_run, out);
            han_run.clear();
            ascii_run.push(c);
        }
    }
    push_han_ngrams(&han_run, out);
    push_ascii_word(&ascii_run, out);
}

/// 抽取三类片段：①ASCII 词 ②中英混排片段 ③纯汉字 n-gram（2..=8）。
/// 汉字与 ASCII 字母数字之外的任何字符（标点/空白/全角符号）都是分段边界。
pub fn extract_fragments(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut run: Vec<char> = Vec::new();
    for c in text.chars() {
        if is_han(c) || c.is_ascii_alphanumeric() {
            run.push(c);
        } else {
            flush_run(&run, &mut out);
            run.clear();
        }
    }
    flush_run(&run, &mut out);
    out
}

/// 构造样例上下文：以候选词为中心截 ≤20 字窗口，控制字符转空格、
/// 连续 ≥2 位数字合并脱敏为 `*`（单数字保留）。
pub fn make_sample(text: &str, term: &str) -> String {
    let Some(byte_pos) = text.find(term) else {
        return sanitize_sample(&term.chars().take(SAMPLE_MAX_CHARS).collect::<String>());
    };
    let chars: Vec<char> = text.chars().collect();
    let term_start = text[..byte_pos].chars().count();
    let term_len = term.chars().count();
    let lead = SAMPLE_MAX_CHARS.saturating_sub(term_len) / 2;
    let start = term_start.saturating_sub(lead);
    let end = (start + SAMPLE_MAX_CHARS).min(chars.len());
    let window: String = chars[start..end].iter().collect();
    sanitize_sample(&window)
}

/// 样例脱敏：控制字符 → 空格；连续 ≥2 位数字 → `*`。
fn sanitize_sample(s: &str) -> String {
    fn flush_digits(out: &mut String, digits: &mut String) {
        if digits.chars().count() >= 2 {
            out.push('*');
        } else {
            out.push_str(digits);
        }
        digits.clear();
    }

    let mut out = String::new();
    let mut digits = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
            continue;
        }
        flush_digits(&mut out, &mut digits);
        if c.is_control() {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    flush_digits(&mut out, &mut digits);
    out
}

/// 候选与词表任一词相等、或互为子串/超串 → 排除（避免噪声与重复建议）。
fn excluded_by(term: &str, words: &[String]) -> bool {
    words.iter().any(|w| {
        let w = w.trim();
        !w.is_empty() && (w.contains(term) || term.contains(w))
    })
}

/// 命中内置常用词/虚词表 → 排除。汉字走双向子串（「的时候」拦
/// 「时候」，「怎么样」拦「么样」）；纯 ASCII 走大小写不敏感相等。
pub fn is_common_term(term: &str) -> bool {
    let has_han = term.chars().any(is_han);
    let has_latin = term.chars().any(|c| c.is_ascii_alphabetic());
    if has_latin && !has_han {
        let lower = term.to_ascii_lowercase();
        COMMON_ENGLISH_WORDS.iter().any(|w| *w == lower.as_str())
    } else {
        COMMON_HAN_WORDS
            .iter()
            .any(|w| term.contains(w) || w.contains(term))
    }
}

/// 可混淆度打分（越高越值得建议为热词）：
/// - 含拉丁 +40；中英混排再 +35；
/// - 纯汉字 3..=8 且不在常用表 +10；长度 ==2 或 >8 减 5 分（不排除）；
/// - 出现次数 3..=15 +10；docs ≤5 +5。
fn score_term(term: &str, kind: TermKind, count: usize, docs: usize) -> i32 {
    let mut score = 0;
    if kind != TermKind::Han {
        score += SCORE_HAS_LATIN;
    }
    if kind == TermKind::Mixed {
        score += SCORE_MIXED_HAN_LATIN;
    }
    if kind == TermKind::Han {
        let len = term.chars().count();
        if (MIN_NGRAM_LEN + 1..=MAX_NGRAM_LEN).contains(&len) && !is_common_term(term) {
            score += SCORE_HAN_UNCOMMON;
        } else {
            score += PENALTY_HAN_EDGE_LEN;
        }
    }
    if (COUNT_WINDOW_MIN..=COUNT_WINDOW_MAX).contains(&count) {
        score += SCORE_COUNT_WINDOW;
    }
    if docs <= RARE_DOCS_MAX {
        score += SCORE_DOCS_RARE;
    }
    score
}

/// 纯函数核心：从历史文本统计、过滤并打分候选词。
///
/// `existing_hotwords` 应传入**已拆平**的热词表（每行 `目标 | 别名` 的
/// 目标与全部别名逐个展开，见 [`flatten_hotword_words`]）；`blocked` 为
/// 四态模型中用户明确屏蔽的词（该状态尚未落地时传空切片）。
pub fn extract_candidates(
    records: &[HistoryText],
    existing_hotwords: &[String],
    blocked: &[String],
) -> Vec<Candidate> {
    let mut stats: HashMap<String, TermStats> = HashMap::new();
    let mut total_docs: usize = 0;
    for record in records {
        let fragments = extract_fragments(&record.text);
        if fragments.is_empty() {
            continue;
        }
        total_docs += 1;
        // 同一条记录内重复只计 1 次 docs（防单条刷量）
        let mut seen_in_record: HashSet<&str> = HashSet::with_capacity(fragments.len());
        for fragment in &fragments {
            let first_in_record = seen_in_record.insert(fragment.as_str());
            let entry = stats.entry(fragment.clone()).or_insert_with(|| TermStats {
                count: 0,
                docs: 0,
                sample: make_sample(&record.text, fragment),
            });
            entry.count += 1;
            if first_in_record {
                entry.docs += 1;
            }
        }
    }

    let mut candidates: Vec<Candidate> = stats
        .into_iter()
        .filter_map(|(term, s)| {
            let kind = classify_term(&term)?;
            if s.count < MIN_COUNT || s.docs < MIN_DOCS {
                return None;
            }
            // docs/总记录数 > 10% → 高频常用词，排除
            if (s.docs as u64) * 100 > (total_docs as u64) * MAX_DOCS_RATIO_PERCENT {
                return None;
            }
            if is_common_term(&term) {
                return None;
            }
            if excluded_by(&term, existing_hotwords) || excluded_by(&term, blocked) {
                return None;
            }
            let score = score_term(&term, kind, s.count, s.docs);
            Some(Candidate {
                term,
                count: s.count,
                docs: s.docs,
                sample: s.sample,
                score,
            })
        })
        .collect();

    // score 降序 → docs 升序（越少越特殊）→ count 降序 → 词升序（稳定 tie-break）
    candidates.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then(a.docs.cmp(&b.docs))
            .then(b.count.cmp(&a.count))
            .then(a.term.cmp(&b.term))
    });
    candidates.truncate(MAX_CANDIDATES);
    candidates
}

/// 把生效热词表（每行 `目标 | 别名 ~~~ 黑名单`）拆平为逐个词的列表。
/// 黑名单是热词内部的上下文排除条件，不是独立热词，不参与候选排除。
pub fn flatten_hotword_words(lines: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for line in lines {
        let parsed = parse_hotword_line(line);
        if !parsed.primary.is_empty() {
            out.push(parsed.primary);
        }
        out.extend(parsed.aliases);
    }
    out
}

/// 只读加载最近 `days` 天历史文本（与 HistoryManager 同一路径的
/// history.db，只读连接，不执行迁移）。
fn load_recent_history(app: &AppHandle, days: u32) -> Result<Vec<HistoryText>> {
    let db_path: PathBuf = crate::portable::app_data_dir(app)?.join("history.db");
    if !db_path.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    // 历史写入与候选读取可能并发；短暂等待而不是立即 SQLITE_BUSY。
    let _ = conn.busy_timeout(Duration::from_millis(2000));

    let cutoff = chrono::Utc::now().timestamp() - (days as i64) * 86_400;
    let mut stmt = conn.prepare(
        "SELECT id, transcription_text, post_processed_text
         FROM transcription_history
         WHERE timestamp >= ?1
         ORDER BY id DESC",
    )?;
    let rows = stmt.query_map(rusqlite::params![cutoff], |row| {
        let id: i64 = row.get(0)?;
        let transcription: String = row.get(1)?;
        let processed: Option<String> = row.get(2)?;
        // 后处理文本是同一语音的整理版：非空则优先，一条记录只算一份文本
        let text = match processed {
            Some(p) if !p.trim().is_empty() => p,
            _ => transcription,
        };
        Ok(HistoryText { id, text })
    })?;
    Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
}

/// 只读命令：最近 `days` 天历史的候选词（Candidate，按可混淆度降序）。
///
/// - 纯本地统计，无网络；不写 settings，不写历史；
/// - SQLite 读与片段统计都在 `spawn_blocking` 中执行，不阻塞异步运行时；
/// - `days == 0` 归一到默认 30 天，上限 3650 天（防御性钳制）；
/// - Blocked 状态尚未落地（设置里没有对应字段），当前传空集；
///   落地后在 [`load_recent_history`] 侧并入即可，纯函数已支持。
#[tauri::command]
#[specta::specta]
pub async fn get_recent_term_candidates(
    app: AppHandle,
    days: u32,
) -> Result<Vec<Candidate>, String> {
    // 先在异步侧取设置（store 读可能触发一次性迁移，属既有路径）
    let existing =
        flatten_hotword_words(&crate::settings::get_settings(&app).effective_custom_words());
    let days = if days == 0 {
        DEFAULT_DAYS
    } else {
        days.min(MAX_DAYS)
    };

    tauri::async_runtime::spawn_blocking(move || {
        let records = load_recent_history(&app, days).map_err(|e| e.to_string())?;
        Ok(extract_candidates(&records, &existing, &[]))
    })
    .await
    .map_err(|e| format!("term candidate task failed: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: i64, text: &str) -> HistoryText {
        HistoryText {
            id,
            text: text.to_string(),
        }
    }

    /// n 条互相不同的填充记录（数字打断汉字段，其 n-gram 全部命中
    /// 「docs 占比 > 10%」排除规则，只用于垫高总记录数）。
    fn fillers(n: usize) -> Vec<HistoryText> {
        (0..n)
            .map(|i| rec(i as i64, &format!("第{i}天日常流水账随便记录一下")))
            .collect()
    }

    fn find<'a>(cands: &'a [Candidate], term: &str) -> Option<&'a Candidate> {
        cands.iter().find(|c| c.term == term)
    }

    #[test]
    fn common_table_sizes_and_hits() {
        // 常用表至少 60 条（汉字部分）
        assert!(COMMON_HAN_WORDS.len() >= 60);
        assert!(COMMON_ENGLISH_WORDS.len() >= 30);
        // 汉字：双向子串命中
        for w in [
            "我们",
            "的时候",
            "时候",
            "什么",
            "么样",
            "可以",
            "因为",
            "所以",
        ] {
            assert!(is_common_term(w), "{w} should be common");
        }
        // 英文：大小写不敏感相等命中
        for w in ["the", "The", "okay", "PLEASE"] {
            assert!(is_common_term(w), "{w} should be common");
        }
        // 不该命中的专名/术语
        for w in ["DeepSeek", "量子纠缠", "Ising", "makefile", "GPT"] {
            assert!(!is_common_term(w), "{w} should not be common");
        }
    }

    #[test]
    fn classify_term_kinds_and_legality() {
        assert_eq!(classify_term("DeepSeek"), Some(TermKind::Ascii));
        assert_eq!(classify_term("GPT4"), Some(TermKind::Ascii));
        assert_eq!(classify_term("用C语言"), Some(TermKind::Mixed));
        assert_eq!(classify_term("量子纠缠"), Some(TermKind::Han));
        // 长度/字符合法性
        assert_eq!(classify_term("ab"), None, "ascii < 3");
        assert_eq!(classify_term("12345"), None, "纯数字");
        assert_eq!(classify_term("！！"), None, "标点");
        assert_eq!(
            classify_term("一二三四五六七八九"),
            None,
            "汉字片段 > 8 不合法"
        );
        let long_han: String = "一二三四五六七八九".repeat(3);
        assert_eq!(classify_term(&long_han), None, "超 20 字");
    }

    #[test]
    fn fragments_ascii_mixed_and_han() {
        // ① ASCII 词：保留原始大小写；过短（I）不收
        let frags = extract_fragments("I love DeepSeek and GPT-4o");
        let set: HashSet<&str> = frags.iter().map(|s| s.as_str()).collect();
        assert!(set.contains("DeepSeek"));
        assert!(set.contains("love"));
        assert!(set.contains("GPT"));
        assert!(!set.contains("deepseek"), "大小写应保留");
        assert!(!set.contains("I"), "长度 < 3 的 ASCII 不收");

        // ② 中英混排：整段成为片段；段内 ASCII 词与汉字 n-gram 照常产出
        let frags = extract_fragments("深度学习GPT真不错");
        let set: HashSet<&str> = frags.iter().map(|s| s.as_str()).collect();
        assert!(set.contains("深度学习GPT真不错"), "整段混排片段");
        assert!(set.contains("GPT"));
        assert!(set.contains("深度学习"));
        assert!(set.contains("真不错"));

        // ③ 纯汉字 n-gram：2..=8，不跨标点/ASCII
        let frags = extract_fragments("深度学习abc，很好用");
        let set: HashSet<&str> = frags.iter().map(|s| s.as_str()).collect();
        assert!(set.contains("深度"));
        assert!(set.contains("深度学习"));
        assert!(set.contains("很好用"));
        assert!(!set.contains("习很"), "不跨 ASCII/标点");
        assert!(frags.iter().all(
            |g| (MIN_NGRAM_LEN..=MAX_NGRAM_LEN).contains(&g.chars().count())
                || g.chars().any(|c| c.is_ascii_alphabetic())
        ));
    }

    #[test]
    fn fragments_reject_digits_punct_and_short_ascii() {
        assert!(extract_fragments("123 456 789!!!。。。、，;").is_empty());
        assert!(extract_fragments("go ok no").is_empty());
        // 卡号里的纯数字永远不会成为候选（无字母）
        let mut records = fillers(40);
        for i in 0..3 {
            records.push(rec(100 + i, "卡号12345678记住"));
        }
        let cands = extract_candidates(&records, &[], &[]);
        assert!(
            cands.iter().all(|c| {
                c.term
                    .chars()
                    .any(|ch| is_han(ch) || ch.is_ascii_alphabetic())
            }),
            "纯数字片段不得入选: {:?}",
            cands
        );
    }

    #[test]
    fn deepseek_ranks_top_above_han_terms() {
        let mut records = fillers(40);
        records.push(rec(100, "DeepSeek 很强"));
        records.push(rec(101, "DeepSeek 很强"));
        records.push(rec(102, "DeepSeek和DeepSeek"));
        for i in 0..3 {
            records.push(rec(110 + i, "量子纠缠"));
        }
        let cands = extract_candidates(&records, &[], &[]);
        // DeepSeek：count=4、docs=3 → 40+10+5=55，必须排第一
        let d = find(&cands, "DeepSeek").expect("DeepSeek 应成为候选");
        assert_eq!((d.count, d.docs, d.score), (4, 3, 55));
        assert_eq!(cands[0].term, "DeepSeek", "DeepSeek 应排最前");
        // 纯汉字术语得分 25，排在后面
        let h = find(&cands, "量子纠缠").expect("量子纠缠 应成为候选");
        assert_eq!(h.score, 25);
        let id = cands.iter().position(|c| c.term == "量子纠缠").unwrap();
        assert!(id > 0);
        assert!(!cands[..id].iter().any(|c| c.score > 55));
    }

    #[test]
    fn common_words_are_excluded() {
        let mut records = fillers(40);
        for i in 0..3 {
            records.push(rec(100 + i, "我们，的时候，可以，因为，所以"));
            records.push(rec(200 + i, "the best"));
        }
        let cands = extract_candidates(&records, &[], &[]);
        for w in [
            "我们",
            "的时候",
            "时候",
            "的时",
            "可以",
            "因为",
            "所以",
            "聊天",
            "the",
        ] {
            assert!(find(&cands, w).is_none(), "常用词 {w} 不应成为候选");
        }
        // 非常用英文词照常入选（大小写不敏感相等只拦虚词）
        let b = find(&cands, "best").expect("best 应成为候选");
        assert_eq!(b.score, 40 + 10 + 5);
    }

    #[test]
    fn thresholds_count_at_least_3_and_docs_at_least_2() {
        // count=2（2 条记录各 1 次）→ 排除
        let mut low = fillers(40);
        low.push(rec(100, "星际航行日志"));
        low.push(rec(101, "星际航行日志"));
        assert!(find(&extract_candidates(&low, &[], &[]), "星际航行").is_none());

        // count=3、docs=2（一条记录内 2 次 + 一条 1 次）→ 保留
        let mut ok = fillers(40);
        ok.push(rec(100, "星际航行日志与星际航行日志"));
        ok.push(rec(101, "只剩星际航行日志"));
        let cands = extract_candidates(&ok, &[], &[]);
        let c = find(&cands, "星际航行").expect("应通过阈值");
        assert_eq!((c.count, c.docs), (3, 2));
        assert!(!c.sample.is_empty());
    }

    #[test]
    fn same_record_repeats_count_once_for_docs() {
        // 单条记录刷 5 次：count=5 但 docs=1 < 2 → 排除
        let mut spam = fillers(40);
        spam.push(rec(1, "重复词重复词重复词重复词重复词"));
        assert!(find(&extract_candidates(&spam, &[], &[]), "重复词").is_none());
    }

    #[test]
    fn docs_coverage_over_ratio_excluded() {
        // 4/44 ≈ 9.1% ≤ 10% → 保留
        let mut ok = fillers(40);
        for i in 0..4 {
            ok.push(rec(100 + i, "星际航行日志"));
        }
        assert!(find(&extract_candidates(&ok, &[], &[]), "星际航行").is_some());

        // 5/45 ≈ 11.1% > 10% → 视为高频常用排除
        let mut hot = fillers(40);
        for i in 0..5 {
            hot.push(rec(100 + i, "星际航行日志"));
        }
        assert!(find(&extract_candidates(&hot, &[], &[]), "星际航行").is_none());
    }

    #[test]
    fn excludes_existing_hotwords_substring_and_superstring() {
        let mut base = fillers(40);
        for i in 0..3 {
            base.push(rec(100 + i, "项目计划讨论"));
        }
        // 无既有热词 → 候选在
        assert!(find(&extract_candidates(&base, &[], &[]), "项目计划").is_some());

        let w = |s: &str| vec![s.to_string()];
        // 相等
        assert!(find(&extract_candidates(&base, &w("项目计划"), &[]), "项目计划").is_none());
        // 候选是热词的子串
        assert!(find(
            &extract_candidates(&base, &w("项目计划书"), &[]),
            "项目计划"
        )
        .is_none());
        // 候选是热词的超串
        assert!(find(&extract_candidates(&base, &w("项目"), &[]), "项目计划").is_none());
    }

    #[test]
    fn excludes_blocked_terms() {
        let mut base = fillers(40);
        for i in 0..3 {
            base.push(rec(100 + i, "项目计划讨论"));
        }
        let blocked = vec!["项目计划".to_string()];
        assert!(find(&extract_candidates(&base, &[], &blocked), "项目计划").is_none());
    }

    #[test]
    fn flatten_hotword_words_expands_primary_and_aliases_not_blacklist() {
        let lines = vec!["深度学习 | 深度学习班 ~~~ 世纪".to_string()];
        assert_eq!(
            flatten_hotword_words(&lines),
            vec!["深度学习".to_string(), "深度学习班".to_string()]
        );
    }

    #[test]
    fn typical_output_top5_fixed() {
        // 固定排序样例：混排 > ASCII 专名 > 纯汉字术语 > 纯汉字噪声
        let mut records = fillers(40);
        for i in 0..3 {
            records.push(rec(100 + i, "用C语言写脚本，DeepSeek 也会用"));
            records.push(rec(200 + i, "量子纠缠"));
        }
        let cands = extract_candidates(&records, &[], &[]);
        let top5: Vec<(&str, i32)> = cands
            .iter()
            .take(5)
            .map(|c| (c.term.as_str(), c.score))
            .collect();
        assert_eq!(
            top5,
            vec![
                ("用C语言写脚本", 90), // 40+35+10+5
                ("DeepSeek", 55),      // 40+10+5
                ("也会用", 25),        // 10+10+5
                ("写脚本", 25),
                ("子纠缠", 25),
            ]
        );
        // 长度 2 的纯汉字减分而非排除：仍在列表里，但排在同 docs 的 3 字词后
        let two = find(&cands, "量子").expect("2 字 n-gram 不应被排除");
        assert_eq!(
            two.score,
            PENALTY_HAN_EDGE_LEN + SCORE_COUNT_WINDOW + SCORE_DOCS_RARE
        );
        // 全列表满足排序不变式：score 降序 → docs 升序 → count 降序 → 词升序
        for w in cands.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let ok = a.score > b.score
                || (a.score == b.score
                    && (a.docs < b.docs
                        || (a.docs == b.docs
                            && (a.count > b.count || (a.count == b.count && a.term <= b.term)))));
            assert!(ok, "order violated: {a:?} before {b:?}");
        }
    }

    #[test]
    fn tie_broken_by_docs_ascending() {
        // 两个 25 分纯汉字术语：docs=2 者排在 docs=3 者之前
        let mut records = fillers(40);
        records.push(rec(100, "星际漫游与星际漫游"));
        records.push(rec(101, "又是星际漫游"));
        for i in 0..3 {
            records.push(rec(110 + i, "深空探测"));
        }
        let cands = extract_candidates(&records, &[], &[]);
        let a = find(&cands, "星际漫游").unwrap();
        let b = find(&cands, "深空探测").unwrap();
        assert_eq!((a.score, b.score), (25, 25));
        assert_eq!((a.docs, b.docs), (2, 3));
        let ia = cands.iter().position(|c| c.term == "星际漫游").unwrap();
        let ib = cands.iter().position(|c| c.term == "深空探测").unwrap();
        assert!(ia < ib, "同分时 docs 小者（更特殊）在前");
    }

    #[test]
    fn caps_output_at_50() {
        // 60 个五不相连 3 字词（任何 gram 只属于一个词）
        let mut records = Vec::new();
        for i in 0..60usize {
            let term: String = (0..3)
                .map(|k| char::from_u32(0x4E00 + (i * 3 + k) as u32).unwrap())
                .collect();
            for j in 0..5 {
                records.push(rec((i * 10 + j) as i64, &term));
            }
        }
        let cands = extract_candidates(&records, &[], &[]);
        assert_eq!(cands.len(), MAX_CANDIDATES);
        // 3 字词（25 分）全部排在 2 字 gram（20 分）之前
        assert!(cands.iter().all(|c| c.term.chars().count() == 3));
    }

    #[test]
    fn sample_is_bounded_and_digits_masked() {
        let records = vec![rec(1, "今天我们讨论了项目计划以及明年的整体安排情况说明")];
        let cands = extract_candidates(&records, &[], &[]);
        // 单条记录 docs=1 < 2，不应成为候选
        assert!(find(&cands, "项目计划").is_none());

        let s = make_sample(
            "今天我们讨论了项目计划以及明年的整体安排情况说明",
            "项目计划",
        );
        assert!(s.chars().count() <= SAMPLE_MAX_CHARS);
        assert!(s.contains("项目计划"));

        let masked = make_sample("电话13800001111说项目计划", "项目计划");
        assert!(masked.chars().count() <= SAMPLE_MAX_CHARS);
        assert!(masked.contains("项目计划"));
        assert!(!masked.contains("138"));
        assert!(masked.contains('*'));
    }
}
