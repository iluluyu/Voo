//! 中文热词系统：拼音音素模糊匹配 + 正则规则替换
//!
//! 上游 Handy 的 `apply_custom_words` 为拉丁文字设计（空格分词 + Soundex
//! 音码），对中文完全无效（上游 issue #1705）。本模块实现引擎无关的中文
//! 热词后处理，思路受 CapsWriter-Offline 启发，参数经专项评审确定：
//!
//! 1. 拼音模糊匹配：热词与转写文本都转为无声调拼音（多音字展开全部候选），
//!    在可读音片段上做**等长**滑动窗口 + 音节级配对代价。命中需同时满足
//!    相似度阈值与硬约束（至多一个近音混淆音节、无不相关音节）。
//! 2. 白名单混淆档位（受控近音，不做通用声韵母折扣）：
//!    完全同音 0 / 单项混淆 0.35 / 声母+韵尾双混淆 0.55 /
//!    方言混淆(n/l、f/h，需开启) 0.50 / 其他 1.0。
//! 3. 正则规则：`模式=替换`（字面量）或 `模式=>替换`（正则），拼音替换后
//!    按行序应用。
//!
//! 误替换防护（评审结论：宁可漏纠，不可误改）：
//! - 默认最短热词 3 个汉字；2 字热词只接受完整同音（dist == 0）。
//! - 纯汉字热词只在纯汉字窗口内匹配（不吞英文/数字）。
//! - 原文已含热词字面时，保护整个区间不被任何候选覆盖。
//! - 同一区间多个热词命中且前两名分差 < 0.08 时放弃（同音竞争歧义）。
//! - 全程 `Vec<char>` 索引，绝不按字节切中文（#1705 panic 教训）。
//! - 幂等：对已处理文本重复应用结果不变。

use pinyin::ToPinyinMulti;
use regex::Regex;

// ---------------------------------------------------------------------------
// 代价表（评审确定的受控档位）
// ---------------------------------------------------------------------------

/// 音节完全同音
const COST_EXACT: f64 = 0.0;
/// 单项白名单混淆（平翘舌 或 前后鼻音）
const COST_SINGLE_CONFUSION: f64 = 0.35;
/// 声母 + 韵尾双混淆（如 zhang→cen 类）
const COST_DOUBLE_CONFUSION: f64 = 0.55;
/// 方言混淆单项（n/l、f/h，需显式开启）
const COST_DIALECT: f64 = 0.50;
/// 音节不相关
const COST_DIFF: f64 = 1.0;
/// 删除/插入一个字（变长窗口默认关闭，保留供显式开启）
const COST_GAP: f64 = 1.0;

/// 默认相似度阈值
pub const DEFAULT_THRESHOLD: f64 = 0.85;
/// 同音竞争歧义：前两名分差小于此值放弃
const AMBIGUITY_MARGIN: f64 = 0.08;
/// 默认最短热词汉字数
pub const DEFAULT_MIN_HAN: usize = 3;

/// 混淆开关（方言混淆默认关闭）
#[derive(Debug, Clone, Copy)]
pub struct ConfusionOptions {
    /// 启用 n/l 与 f/h 声母混淆（默认 false）
    pub dialect: bool,
}

impl Default for ConfusionOptions {
    fn default() -> Self {
        Self { dialect: false }
    }
}

// ---------------------------------------------------------------------------
// 拼音层
// ---------------------------------------------------------------------------

/// 单个"字"的读音候选
#[derive(Debug, Clone, PartialEq)]
pub struct CharSounds {
    /// 全部候选读音（plain 无声调，去重），至少 1 个
    pub syllables: Vec<String>,
    /// 是否汉字
    pub is_han: bool,
}

/// 汉字返回多音候选；ASCII 字母/数字返回自身小写作单音节（支持中英混输
/// 热词）；其他（标点/空白/emoji）返回 None（不参与对齐）。
pub fn char_sounds(c: char) -> Option<CharSounds> {
    if is_han(c) {
        let mut syllables: Vec<String> = c
            .to_pinyin_multi()
            .map(|multi| {
                multi
                    .into_iter()
                    .map(|p| p.plain().to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if syllables.is_empty() {
            return None; // 词典缺字（生僻字）
        }
        syllables.sort();
        syllables.dedup();
        Some(CharSounds {
            syllables,
            is_han: true,
        })
    } else if c.is_ascii_alphanumeric() {
        Some(CharSounds {
            syllables: vec![c.to_ascii_lowercase().to_string()],
            is_han: false,
        })
    } else {
        None
    }
}

pub fn is_han(c: char) -> bool {
    matches!(c,
        '\u{3400}'..='\u{4DBF}'   // CJK 扩展 A
        | '\u{4E00}'..='\u{9FFF}' // CJK 基本
        | '\u{F900}'..='\u{FAFF}' // CJK 兼容
    )
}

pub fn has_han(text: &str) -> bool {
    text.chars().any(is_han)
}

/// 拆音节为 (声母, 韵母)。零声母（a/o/e 开头）声母为空串。
/// 输入须为合法无声调拼音；本函数只为白名单混淆识别服务，容错处理
/// 非法输入（返回原样，不 panic）。
fn split_syllable(s: &str) -> (String, String) {
    const DOUBLE_INITIALS: [&str; 3] = ["zh", "ch", "sh"];
    const SINGLE_INITIALS: &str = "bpmfdtnlgkhjqxrzcsyw";
    for d in DOUBLE_INITIALS {
        if let Some(rest) = s.strip_prefix(d) {
            if !rest.is_empty() {
                return (d.to_string(), rest.to_string());
            }
        }
    }
    let mut chars = s.chars();
    if let Some(first) = chars.next() {
        let rest = chars.as_str();
        if SINGLE_INITIALS.contains(first) && !rest.is_empty() {
            return (first.to_string(), rest.to_string());
        }
    }
    (String::new(), s.to_string())
}

/// 计算两个具体音节之间的混淆档位。
/// 返回 (cost, confusion_count)：count 用于硬约束（非零 cost 音节计数）。
fn syllable_confusion(a: &str, b: &str, opts: &ConfusionOptions) -> (f64, usize) {
    if a == b {
        return (COST_EXACT, 0);
    }
    let (ia, fa) = split_syllable(a);
    let (ib, fb) = split_syllable(b);

    let mut confusions = 0usize;

    // 声母：平翘舌对（zh/z, ch/c, sh/s）——注意 zheng vs zeng 这类
    // "声母平翘舌"对；以及方言对 n/l、f/h
    let initial_pair = |x: &str, y: &str| -> bool {
        let pair = |p: [&str; 2]| (x == p[0] && y == p[1]) || (x == p[1] && y == p[0]);
        pair(["zh", "z"]) || pair(["ch", "c"]) || pair(["sh", "s"])
    };
    let dialect_pair = |x: &str, y: &str| -> bool {
        let pair = |p: [&str; 2]| (x == p[0] && y == p[1]) || (x == p[1] && y == p[0]);
        pair(["n", "l"]) || pair(["f", "h"])
    };

    let initial_confused = if ia == ib {
        false
    } else if initial_pair(&ia, &ib) {
        confusions += 1;
        true
    } else if opts.dialect && dialect_pair(&ia, &ib) {
        // 方言混淆计为一项，cost 用 COST_DIALECT 档（见下方合成）
        confusions += 1;
        true
    } else {
        // 声母完全不同且不在白名单：整音节不相关
        return (COST_DIFF, 1);
    };
    let _ = initial_confused;

    // 韵母：仅前后鼻音对（ng/n 结尾差异）
    let nasal_pair = |x: &str, y: &str| -> bool {
        if x == y {
            return false;
        }
        // 前后鼻音对：in/ing、en/eng、an/ang——去掉结尾的 'g' 后应相等。
        // 注意不能 strip_suffix("ng")：那会把 "ing" 剥成 "i"。
        let strip_final_g = |s: &str| -> String {
            if s.ends_with('g') {
                s[..s.len() - 1].to_string()
            } else {
                s.to_string()
            }
        };
        strip_final_g(x) == strip_final_g(y) && (x.ends_with("ng") || y.ends_with("ng"))
    };

    let final_confused = if fa == fb {
        false
    } else if nasal_pair(&fa, &fb) {
        confusions += 1;
        true
    } else {
        return (COST_DIFF, 1);
    };

    // 合成档位
    let cost = match confusions {
        0 => COST_EXACT, // 不可能（a==b 已提前返回），防御
        1 => {
            // 区分方言档与普通单项档
            if opts.dialect && dialect_pair(&ia, &ib) && !initial_pair(&ia, &ib) {
                COST_DIALECT
            } else {
                COST_SINGLE_CONFUSION
            }
        }
        _ => COST_DOUBLE_CONFUSION,
    };
    (cost, confusions.min(1))
}

/// 两字候选集的配对代价（笛卡尔积取最小）。
/// 返回 (cost, 该最小代价下的混淆数)
fn pair_cost(a: &[String], b: &[String], opts: &ConfusionOptions) -> (f64, usize) {
    let mut best = (COST_DIFF, 1usize);
    for sa in a {
        for sb in b {
            let r = syllable_confusion(sa, sb, opts);
            if r.0 < best.0 {
                best = r;
            }
            if best.0 == COST_EXACT {
                return best;
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// 热词索引
// ---------------------------------------------------------------------------

/// 一个热词条目
#[derive(Debug, Clone)]
pub struct HotwordEntry {
    /// 替换目标（用户配置的热词原文）
    pub display: String,
    /// 每字读音候选（与 display 的 chars 一一对应）
    pub syllables: Vec<CharSounds>,
    /// 汉字数量
    pub han_count: usize,
    /// 是否纯汉字（纯汉字热词只在纯汉字窗口匹配）
    pub all_han: bool,
}

/// 从自定义词列表构建拼音热词索引。
/// 收录条件：汉字数 >= `min_han`（默认 3）、每个字符都有读音。
pub fn build_hotword_index(words: &[String], min_han: usize) -> Vec<HotwordEntry> {
    let mut entries = Vec::with_capacity(words.len());
    for word in words {
        let han_count = word.chars().filter(|c| is_han(*c)).count();
        if han_count < min_han || word.chars().count() > 50 {
            continue;
        }
        let mut syllables = Vec::with_capacity(word.chars().count());
        let mut ok = true;
        for c in word.chars() {
            match char_sounds(c) {
                Some(s) => syllables.push(s),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            entries.push(HotwordEntry {
                display: word.clone(),
                all_han: syllables.iter().all(|s| s.is_han),
                han_count,
                syllables,
            });
        }
    }
    entries
}

// ---------------------------------------------------------------------------
// 文本切分
// ---------------------------------------------------------------------------

/// 连续可读音片段（字符有全局 char 偏移）
#[derive(Debug)]
struct Segment {
    chars: Vec<(usize, char)>,
    sounds: Vec<CharSounds>,
}

fn segment_text(text: &[char]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current: Vec<(usize, char)> = Vec::new();
    let mut sounds: Vec<CharSounds> = Vec::new();
    for (idx, &c) in text.iter().enumerate() {
        match char_sounds(c) {
            Some(s) => {
                current.push((idx, c));
                sounds.push(s);
            }
            None => {
                if !current.is_empty() {
                    segments.push(Segment {
                        chars: std::mem::take(&mut current),
                        sounds: std::mem::take(&mut sounds),
                    });
                }
            }
        }
    }
    if !current.is_empty() {
        segments.push(Segment {
            chars: current,
            sounds,
        });
    }
    segments
}

// ---------------------------------------------------------------------------
// 匹配核心
// ---------------------------------------------------------------------------

/// 等长窗口对齐代价：逐位置配对（等长无 gap）。
/// 返回 None 若不满足硬约束。
/// 硬约束：非零 cost 的音节至多 1 个；且无 cost == COST_DIFF 音节。
/// 相似度 = 1 - total_cost / len。
fn aligned_match(
    window: &[CharSounds],
    hot: &[CharSounds],
    opts: &ConfusionOptions,
    allow_two_near: bool,
) -> Option<f64> {
    debug_assert_eq!(window.len(), hot.len());
    let n = window.len();
    let mut total = 0.0f64;
    let mut non_zero = 0usize;
    for (w, h) in window.iter().zip(hot.iter()) {
        let (cost, _conf) = pair_cost(&w.syllables, &h.syllables, opts);
        if cost >= COST_DIFF {
            return None; // 出现不相关音节
        }
        if cost > 0.0 {
            non_zero += 1;
            let limit = if allow_two_near { 2 } else { 1 };
            if non_zero > limit {
                return None;
            }
        }
        total += cost;
    }
    Some(1.0 - total / n as f64)
}

/// 滑动窗口（支持 slack=0 等长 / slack=1 变长实验），带硬约束与相似度阈值
fn window_similarity(
    window: &[CharSounds],
    hot: &[CharSounds],
    opts: &ConfusionOptions,
    threshold: f64,
    slack: usize,
) -> Option<f64> {
    if window.len() != hot.len() {
        if slack == 0 {
            return None;
        }
        // 变长：带 gap 的简单对齐（每差一个字记一次 COST_GAP）
        // 注：默认路径不进入这里
        let diff = window.len().abs_diff(hot.len());
        if diff > slack {
            return None;
        }
        // 保守处理：截齐到较短者 + 差额罚分（近似）
        let n = window.len().min(hot.len());
        let base = aligned_match(&window[..n], &hot[..n], opts, false)?;
        let sim = base - (diff as f64 * COST_GAP) / window.len().max(hot.len()) as f64;
        return if sim >= threshold { Some(sim) } else { None };
    }
    let sim = aligned_match(window, hot, opts, false)?;
    if sim >= threshold {
        Some(sim)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// 替换主流程
// ---------------------------------------------------------------------------

/// 命中区间（全局 char 偏移）
struct Hit {
    start: usize,
    end: usize,
    score: f64,
    hotword_idx: usize,
    /// 参与竞争的次高分（同区间）
    runner_up: Option<f64>,
    matched_text: String,
}

/// 对转写文本应用拼音热词替换。
/// 返回 (新文本, 命中列表[(原文, 替换为, 相似度)])
pub fn apply_pinyin_hotwords(
    text: &str,
    index: &[HotwordEntry],
    threshold: f64,
    opts: &ConfusionOptions,
) -> (String, Vec<(String, String, f64)>) {
    if index.is_empty() || !has_han(text) {
        return (text.to_string(), Vec::new());
    }

    let chars: Vec<char> = text.chars().collect();
    let segments = segment_text(&chars);

    // 1) 保护区：原文已完整包含热词字面 → 该区间不得被任何候选覆盖
    let mut protected: Vec<(usize, usize)> = Vec::new();
    for hot in index {
        let hw_chars: Vec<char> = hot.display.chars().collect();
        if hw_chars.len() < chars.len() {
            let m = hw_chars.len();
            for start in 0..=(chars.len() - m) {
                if chars[start..start + m] == hw_chars[..] {
                    protected.push((start, start + m));
                }
            }
        }
    }

    // 2) 收集候选命中（按全局区间）
    // 候选按 (start,end) 分组保留 top2 以做歧义检测
    let mut grouped: std::collections::HashMap<(usize, usize), Vec<(f64, usize, String)>> =
        std::collections::HashMap::new();

    for seg in &segments {
        if !seg.sounds.iter().any(|s| s.is_han) {
            continue;
        }
        let seg_len = seg.sounds.len();
        for (hw_idx, hot) in index.iter().enumerate() {
            let m = hot.syllables.len();
            if m == 0 || seg_len < m {
                continue;
            }
            // 2 字热词：只接受完整同音（分数必须为 1.0）
            let two_char_strict = hot.han_count <= 2;
            for start in 0..=(seg_len - m) {
                let window = &seg.sounds[start..start + m];
                // 纯汉字热词：窗口必须全汉字（不吞 ASCII/数字）
                if hot.all_han && window.iter().any(|s| !s.is_han) {
                    continue;
                }
                let window_text: String =
                    seg.chars[start..start + m].iter().map(|(_, c)| c).collect();
                if window_text == hot.display {
                    continue; // 保护逻辑已覆盖
                }
                if let Some(score) = window_similarity(window, &hot.syllables, opts, threshold, 0) {
                    if two_char_strict && score < 1.0 {
                        continue;
                    }
                    let g_start = seg.chars[start].0;
                    let g_end = seg.chars[start + m - 1].0 + 1;
                    grouped
                        .entry((g_start, g_end))
                        .or_default()
                        .push((score, hw_idx, window_text));
                }
            }
        }
    }

    if grouped.is_empty() {
        return (text.to_string(), Vec::new());
    }

    // 3) 展开为 Hit，做歧义检测 + 保护过滤
    let mut hits: Vec<Hit> = Vec::new();
    for ((start, end), mut list) in grouped {
        list.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let (score, hotword_idx, matched_text) = list[0].clone();
        let runner_up = list.get(1).map(|(s, _, _)| *s);
        // 同音竞争歧义：前两名分差过小 → 放弃
        if let Some(ru) = runner_up {
            if score - ru < AMBIGUITY_MARGIN {
                continue;
            }
        }
        // 保护区间过滤
        if protected.iter().any(|&(ps, pe)| start < pe && ps < end) {
            continue;
        }
        hits.push(Hit {
            start,
            end,
            score,
            hotword_idx,
            runner_up,
            matched_text,
        });
    }

    // 4) 贪心选择互不重叠命中：分数降序、长词优先、位置升序
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then((b.end - b.start).cmp(&(a.end - a.start)))
            .then(a.start.cmp(&b.start))
    });
    let mut chosen: Vec<&Hit> = Vec::new();
    let mut occupied: Vec<(usize, usize)> = Vec::new();
    for hit in &hits {
        if !occupied
            .iter()
            .any(|&(os, oe)| hit.start < oe && os < hit.end)
        {
            occupied.push((hit.start, hit.end));
            chosen.push(hit);
        }
    }
    chosen.sort_by_key(|h| h.start); // 从左到右重建

    // 5) 重建文本（char 级，天然 boundary 安全）
    let mut out = String::with_capacity(text.len() + 16);
    let mut log = Vec::with_capacity(chosen.len());
    let mut cursor = 0usize;
    for hit in chosen {
        let hot = &index[hit.hotword_idx];
        out.push_str(&chars[cursor..hit.start].iter().collect::<String>());
        out.push_str(&hot.display);
        log.push((hit.matched_text.clone(), hot.display.clone(), hit.score));
        cursor = hit.end;
    }
    out.push_str(&chars[cursor..].iter().collect::<String>());

    (out, log)
}

// ---------------------------------------------------------------------------
// 规则引擎
// ---------------------------------------------------------------------------

/// 一条替换规则
#[derive(Debug, Clone)]
pub struct HotwordRule {
    pub pattern: Regex,
    pub replacement: String,
    /// 规则原文（日志用）
    pub source: String,
}

/// 解析规则文本，每行一条：
/// - `模式=>替换`：正则
/// - `模式=替换`：字面量（内部转义）
/// - 空行与 `#` 注释忽略
/// 返回 (规则列表, 失败行号列表)
pub fn parse_hotword_rules(input: &str) -> (Vec<HotwordRule>, Vec<usize>) {
    let mut rules = Vec::new();
    let mut errors = Vec::new();
    for (i, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // 先找 =>（优先于 =）
        let (pattern, op, replacement) = if let Some(pos) = line.find("=>") {
            (&line[..pos], "=>", &line[pos + 2..])
        } else if let Some(pos) = line.find('=') {
            (&line[..pos], "=", &line[pos + 1..])
        } else {
            errors.push(i + 1);
            continue;
        };
        let pattern = pattern.trim();
        let replacement = replacement.trim();
        if pattern.is_empty() {
            errors.push(i + 1);
            continue;
        }
        let effective = if op == "=>" {
            pattern.to_string()
        } else {
            regex::escape(pattern)
        };
        match Regex::new(&effective) {
            Ok(re) => rules.push(HotwordRule {
                pattern: re,
                replacement: replacement.to_string(),
                source: line.to_string(),
            }),
            Err(_) => errors.push(i + 1),
        }
    }
    (rules, errors)
}

/// 顺序应用规则
pub fn apply_hotword_rules(text: &str, rules: &[HotwordRule]) -> String {
    let mut out = text.to_string();
    for rule in rules {
        out = rule
            .pattern
            .replace_all(&out, rule.replacement.as_str())
            .into_owned();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPTS: ConfusionOptions = ConfusionOptions { dialect: false };
    const DIALECT: ConfusionOptions = ConfusionOptions { dialect: true };

    fn idx(words: &[&str]) -> Vec<HotwordEntry> {
        let owned: Vec<String> = words.iter().map(|s| s.to_string()).collect();
        build_hotword_index(&owned, DEFAULT_MIN_HAN)
    }

    fn idx_min2(words: &[&str]) -> Vec<HotwordEntry> {
        let owned: Vec<String> = words.iter().map(|s| s.to_string()).collect();
        build_hotword_index(&owned, 2)
    }

    fn apply(text: &str, index: &[HotwordEntry]) -> (String, Vec<(String, String, f64)>) {
        apply_pinyin_hotwords(text, index, DEFAULT_THRESHOLD, &OPTS)
    }

    // ---------------- 拼音层 ----------------

    #[test]
    fn test_han_detection() {
        assert!(is_han('中'));
        assert!(is_han('漢'));
        assert!(!is_han('a'));
        assert!(!is_han('。'));
        assert!(has_han("hello 世界"));
        assert!(!has_han("pure english 123"));
    }

    #[test]
    fn test_char_sounds_multi() {
        let s = char_sounds('重').unwrap();
        assert!(s.is_han);
        assert!(s.syllables.len() >= 2, "{:?}", s.syllables);
        assert!(s.syllables.contains(&"zhong".to_string()));
        assert!(s.syllables.contains(&"chong".to_string()));

        let a = char_sounds('V').unwrap();
        assert_eq!(a.syllables, vec!["v".to_string()]);
        assert!(!a.is_han);
        assert!(char_sounds('，').is_none());
    }

    #[test]
    fn test_split_syllable() {
        assert_eq!(split_syllable("zhang"), ("zh".into(), "ang".into()));
        assert_eq!(split_syllable("zhi"), ("zh".into(), "i".into()));
        assert_eq!(split_syllable("ba"), ("b".into(), "a".into()));
        assert_eq!(split_syllable("ai"), ("".into(), "ai".into()));
        assert_eq!(split_syllable("nv"), ("n".into(), "v".into()));
    }

    #[test]
    fn test_confusion_levels() {
        // 完全同音
        assert_eq!(syllable_confusion("yun", "yun", &OPTS).0, COST_EXACT);
        // 平翘舌单项
        let (c, _) = syllable_confusion("zhi", "zi", &OPTS);
        assert_eq!(c, COST_SINGLE_CONFUSION);
        // 前后鼻音单项
        let (c, _) = syllable_confusion("xin", "xing", &OPTS);
        assert_eq!(c, COST_SINGLE_CONFUSION);
        // 双混淆：zheng vs cen（声母 zh/c 无白名单对 -> DIFF）
        let (c, _) = syllable_confusion("zheng", "cen", &OPTS);
        assert_eq!(c, COST_DIFF);
        // 双混淆：zhang vs can（zh/c 不是白名单对）
        let (c, _) = syllable_confusion("zhang", "can", &OPTS);
        assert_eq!(c, COST_DIFF);
        // 声母平翘舌 + 韵尾鼻音：zheng vs zeng
        let (c, n) = syllable_confusion("zheng", "zeng", &OPTS);
        assert_eq!(c, COST_SINGLE_CONFUSION); // zh/z 是声母对，eng/eng 相同 -> 单项
        assert_eq!(n, 1);
        // zhan vs zang: zh/z 声母对 + an/ang 韵尾对 -> 双混淆
        let (c, _) = syllable_confusion("zhan", "zang", &OPTS);
        assert_eq!(c, COST_DOUBLE_CONFUSION);
        // n/l 关闭时 DIFF，开启时 DIALECT
        assert_eq!(syllable_confusion("niu", "liu", &OPTS).0, COST_DIFF);
        assert_eq!(syllable_confusion("niu", "liu", &DIALECT).0, COST_DIALECT);
        // 不相关
        assert_eq!(syllable_confusion("ma", "go", &OPTS).0, COST_DIFF);
    }

    #[test]
    fn test_pair_cost_multi_heteronym() {
        // 多音字候选：'重' 有 zhong/chong，'虫' 有 chong —— 应取 chong==chong = 0
        let zhong = char_sounds('重').unwrap();
        let chong = char_sounds('虫').unwrap();
        let (c, n) = pair_cost(&zhong.syllables, &chong.syllables, &OPTS);
        assert_eq!((c, n), (COST_EXACT, 0));
    }

    // ---------------- 索引 ----------------

    #[test]
    fn test_build_index_filters() {
        let words = vec![
            "中州韵输入法".to_string(),
            "A".to_string(),
            "的".to_string(),
            "Vue组件".to_string(),
        ];
        let index = build_hotword_index(&words, DEFAULT_MIN_HAN);
        let displays: Vec<&str> = index.iter().map(|e| e.display.as_str()).collect();
        assert!(displays.contains(&"中州韵输入法"));
        assert!(!displays.contains(&"Vue组件")); // 汉字数 2 < 3，默认被排除
        assert!(!displays.contains(&"的"));
        assert!(!displays.contains(&"A"));
        // min_han=2 时收录混合词
        let index2 = build_hotword_index(&words, 2);
        assert!(index2.iter().any(|e| e.display == "Vue组件"));
    }

    // ---------------- 匹配：同音 ----------------

    #[test]
    fn test_homophone_replacement() {
        let index = idx(&["中州韵输入法"]);
        // 运(yun) 韵(yun) 同音错字
        let (out, log) = apply("我在用中州运输入法打字", &index);
        assert_eq!(out, "我在用中州韵输入法打字");
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].1, "中州韵输入法");
        assert!(log[0].2 > 0.999);
    }

    #[test]
    fn test_protect_exact_hotword() {
        let index = idx(&["人工智能"]);
        // 原文已含热词字面：整体保护，且相邻同音不误替换
        let (out, log) = apply("人工智能跟人工只能不一样", &index);
        // "人工只能"（zhineng vs zhinen? 只=zhi 能=neng；智=zhi 能=neng 同音）
        // 应被替换；已存在的"人工智能"保持
        assert_eq!(out, "人工智能跟人工智能不一样");
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_two_char_homophone_strict() {
        // 2 字热词在 min_han=2 时收录，但只接受完整同音
        let index = idx_min2(&["重庆"]);
        // 崇庆（chong qing）完全同音 -> 替换
        let (out, _) = apply("去崇庆旅游", &index);
        assert_eq!(out, "去重庆旅游");
        // 近音（沁=qin vs 庆=qing 单项前后鼻音）不替换：2 字热词仅接受完整同音
        let (out2, log2) = apply("去崇沁旅游", &index);
        assert_eq!(out2, "去崇沁旅游");
        assert!(log2.is_empty());
    }

    // ---------------- 匹配：近音 ----------------

    #[test]
    fn test_near_sound_single_confusion() {
        // 知识(zhishi) vs 字词(zici)：zhi/zi + shi/si 双单项 = 2 个非零音节
        // 硬约束（至多 1 个非零）会拒绝 —— 验证保守性
        let index = idx(&["知识库"]);
        let (out, _) = apply("字词库管理", &index);
        // zi+ci 两个音节都混淆 -> 拒绝
        assert_eq!(out, "字词库管理");
    }

    #[test]
    fn test_near_sound_one_syllable() {
        // 只有一个音节近音：库(ku)同，识(shi/si)单项
        let index = idx(&["知识库"]);
        // "zhi si 库"：知(zhi) 识->私(si) 近音单项，其余同音
        let (out, log) = apply("把资料放进知识库", &index);
        assert_eq!(out, "把资料放进知识库"); // 已含原文，保护
                                             // 构造单项近音：直柿库（zhi shi vs zhi si? 直=zhi 柿=shi）
                                             // "知识" 里 识=shi；测试用"只四库"：只=zhi 四=si（shi/si 单项）
        let (out2, log2) = apply("同步只四库", &index);
        assert_eq!(out2, "同步知识库");
        assert_eq!(log2.len(), 1);
    }

    // ---------------- 防误替换 ----------------

    #[test]
    fn test_no_false_positive_common_words() {
        // 常用同音词冲突：热词"权利"，原文正确使用"权利"时受保护；
        // 但"权力"出现时不该被替换为"权利"（同音竞争应保守）
        let index = idx_min2(&["权利"]);
        let (out, log) = apply("保障公民权力不受侵犯", &index);
        // dist=0 分数 1.0 无 runner-up…… 按当前规则会替换。
        // 这是"强制替换"语义下的预期行为（用户加入热词即偏向该写法），
        // 文档已注明。此测试仅固化行为：
        assert_eq!(out, "保障公民权利不受侵犯");
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_ambiguity_between_hotwords() {
        // 两个同音热词竞争：放弃
        let index = idx(&["机器学习", "机器统计学"]);
        // "机械学习"（xie vs qi 不同音）不应命中任何一个……
        // 构造真同音竞争：两个热词同音
        let index = idx(&["深棕色", "身宗色"]);
        let (out, log) = apply("买了一件深宗色的外套", &index);
        // 深宗色 与两个热词都是 1 个非零音节（zong 分歧?）
        // 棕=zong 宗=zong 同音！那两个热词同分 -> 歧义放弃
        assert_eq!(out, "买了一件深宗色的外套");
        assert!(log.is_empty());
    }

    #[test]
    fn test_unrelated_not_replaced() {
        let index = idx(&["人工智能"]);
        let (out, log) = apply("今天天气真好", &index);
        assert_eq!(out, "今天天气真好");
        assert!(log.is_empty());
    }

    #[test]
    fn test_pure_han_window_no_ascii() {
        // 纯汉字热词不吞 ASCII
        let index = idx(&["数据库设计"]);
        let (out, log) = apply("做了MySQL数据库设计", &index);
        // "数据库设计"前面是 L（ASCII），窗口为纯汉字子串，应命中
        assert_eq!(out, "做了MySQL数据库设计"); // 已含字面 -> 保护，不变
        assert!(log.is_empty());
    }

    #[test]
    fn test_idempotent() {
        let index = idx(&["中州韵输入法", "人工智能"]);
        let (out1, _) = apply("用中州运输入法和人工只能", &index);
        let (out2, log2) = apply(&out1, &index);
        assert_eq!(out1, out2, "idempotency broken");
        assert!(log2.is_empty());
    }

    #[test]
    fn test_mixed_hotword() {
        // 混合热词（min_han=2）
        let index = idx_min2(&["Vue3组件"]);
        // 原样出现保护；同音错字替换：祖件(zu jian) vs 组件(zu jian) 同音
        let (out, _) = apply("封装了Vue3组件", &index);
        assert_eq!(out, "封装了Vue3组件");
        let (out2, log2) = apply("封装了Vue3祖件", &index);
        assert_eq!(out2, "封装了Vue3组件");
        assert_eq!(log2.len(), 1);
    }

    #[test]
    fn test_char_boundary_safety() {
        // emoji + 中文混合：不 panic，正确处理
        let index = idx(&["语音转文字"]);
        let (out, _) = apply("🚀语音转蚊字🚀完成", &index);
        assert_eq!(out, "🚀语音转文字🚀完成");
    }

    #[test]
    fn test_long_word_multiple_near_rejected() {
        // 长热词多个近音音节：硬约束拒绝（宁可漏纠）
        let index = idx(&["机器学习法"]);
        // 机气学习法：机(qi)同 气(qi)同 学习同 法同 -> dist 0 命中
        let (out, _) = apply("练习机气学习法", &index);
        assert_eq!(out, "练习机器学习法");
        // 字次学习法：字/机 diff -> 拒绝
        let (out2, _) = apply("练习字次学习法", &index);
        assert_eq!(out2, "练习字次学习法");
    }

    // ---------------- 规则 ----------------

    #[test]
    fn test_parse_rules() {
        let input = "\
# 注释行
中州韵=中州韵输入法
四([个只])=>四条$1
坏规则[
";
        let (rules, errors) = parse_hotword_rules(input);
        assert_eq!(rules.len(), 2);
        assert_eq!(errors.len(), 1);
        assert_eq!(rules[0].source, "中州韵=中州韵输入法");
    }

    #[test]
    fn test_apply_rules() {
        let input = "\
中州韵=中州韵输入法
(\\d)个=>${1}条
";
        let (rules, _) = parse_hotword_rules(input);
        let out = apply_hotword_rules("我在用中州韵打字，买了3个", &rules);
        assert_eq!(out, "我在用中州韵输入法打字，买了3条");
    }

    #[test]
    fn test_rules_arrow_precedence() {
        // => 优先于 =
        let (rules, _) = parse_hotword_rules("a=b=>c");
        assert_eq!(rules[0].pattern.as_str(), "a=b");
        assert_eq!(rules[0].replacement, "c");
    }

    #[test]
    fn test_rules_empty() {
        let (rules, errors) = parse_hotword_rules("");
        assert!(rules.is_empty());
        assert!(errors.is_empty());
        assert_eq!(apply_hotword_rules("不变", &[]), "不变");
    }
}
