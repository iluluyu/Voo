//! 粘贴后修改学习（Candidate 态）：从「贴出去的文本」与「用户随后改成的
//! 文本」中提取词级纠错候选。
//!
//! 本模块只做纯规则提取 + JSON 持久化：**绝不写 `custom_words`、不给引擎
//! 授予近音、不触碰任何平台/输入框 API**（输入侧采集是后续接入的事）。
//! 产出全部停留在 Candidate，落盘在 `<data_dir>/learned_corrections.json`：
//!
//! - `candidates`：学到的候选（`wrong → corrected`，重复出现累加 `count`）；
//! - `blocked`：被 Undo 拉黑的词对，同一对之后永远不再进入候选。
//!
//! 学习链路的入口是 [`observe_paste_correction`]，它先过 [`should_learn`]
//! （所有阈值收口在 [`diff_word_correction`]），再查 blocked、做去重累加。

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::Manager;

/// 一条学到的纠错候选（Candidate 态，未激活、不参与任何纠错）。
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
pub struct LearnedCandidate {
    /// 原词（贴出去后被改掉的中间差异段）
    pub wrong: String,
    /// 新词（用户改成的中间差异段）
    pub corrected: String,
    /// 同一对被观察到的次数（重复出现累加）
    pub count: u32,
    /// 首次学到该对的时间（Unix 毫秒）
    pub first_seen: u64,
    /// 最近一次观察到该对的时间（Unix 毫秒）
    pub last_seen: u64,
}

/// 一对被 Undo 拉黑的纠错（键名 `blocked`）。命中即永不再学习。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockedPair {
    pub wrong: String,
    pub corrected: String,
    /// 拉黑时间（Unix 毫秒）
    pub blocked_at: u64,
}

/// JSON 持久化结构：`{"candidates": [...], "blocked": [...]}`。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct LearnStore {
    #[serde(default)]
    candidates: Vec<LearnedCandidate>,
    #[serde(default)]
    blocked: Vec<BlockedPair>,
}

/// 每侧中间差异段允许的最大 token 数（按空白切分）。
const MAX_TOKENS: usize = 4;
/// 每侧中间差异段允许的最大字符数（Unicode 标量）。
const MAX_CHARS: usize = 24;
/// 否定词字符：仅这些字符的增删/互换不构成可学的「词形纠错」。
const NEGATION_CHARS: [char; 6] = ['不', '没', '未', '别', '无', '非'];

/// 串行化对 store 文件的读-改-写（命令与学习入口共用；命令可能并发进来）。
static STORE_IO_LOCK: Mutex<()> = Mutex::new(());

fn is_hanzi(c: char) -> bool {
    matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}')
}

fn is_allowed_char(c: char) -> bool {
    is_hanzi(c) || c.is_ascii_alphabetic() || c == ' '
}

/// 差异段必须：非空；只含汉字/拉丁字母/空白（数字、标点、其他文字一律拒绝，
/// 「无句内标点」「数字变化」由此白名单保证）；至少含一个汉字（纯拉丁不接受，
/// 只接受纯汉字或汉字+拉丁）；≤ MAX_TOKENS 个 token；≤ MAX_CHARS 个字符。
fn valid_segment(seg: &str) -> bool {
    if seg.is_empty() {
        return false;
    }
    if seg.chars().any(|c| !is_allowed_char(c)) {
        return false;
    }
    if !seg.chars().any(is_hanzi) {
        return false;
    }
    if seg.chars().count() > MAX_CHARS {
        return false;
    }
    seg.split_whitespace().count() <= MAX_TOKENS
}

/// 去掉否定词字符后的剩余部分，用于识别「仅否定语义翻转」的伪纠错。
fn strip_negation(seg: &str) -> String {
    seg.chars()
        .filter(|c| !NEGATION_CHARS.contains(c))
        .collect()
}

/// 归一化中间差异段：trim + 连续空白折叠成单个空格。
fn normalize_segment(seg: &str) -> String {
    seg.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 提取一次粘贴后修改中的词级纠错：前后缀对齐，取中间差异段 `(原词, 新词)`。
///
/// 规则（任一不满足即 `None`）：
///
/// | 规则 | 判定 |
/// |---|---|
/// | 前后缀锚点 | 前缀与后缀都为空（整段替换，无局部锚点）→ `None` |
/// | 词级替换 | 任一侧中间段为空（纯插入/纯删除）→ `None` |
/// | 纯空白 | 归一化后两侧相同 → `None` |
/// | 纯大小写 | 小写化后两侧相同 → `None` |
/// | 否定词 | 去掉否定字符后两侧相同 → `None` |
/// | 体积 | 每侧 ≤4 token 且 ≤24 字符 |
/// | 字符白名单 | 每侧只含汉字/拉丁字母/空白（拒数字、句内标点、其他文字）|
/// | 脚本 | 每侧至少含一个汉字（纯拉丁拒绝；接受纯汉字或汉字+拉丁）|
pub fn diff_word_correction(inserted: &str, later: &str) -> Option<(String, String)> {
    let a: Vec<char> = inserted.chars().collect();
    let b: Vec<char> = later.chars().collect();
    if a.is_empty() || b.is_empty() || a == b {
        return None;
    }

    let prefix = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    // 前后缀不得重叠（如 "哈哈" vs "哈哈哈"）。
    let max_suffix = (a.len() - prefix).min(b.len() - prefix);
    let suffix = a
        .iter()
        .rev()
        .zip(b.iter().rev())
        .take_while(|(x, y)| x == y)
        .count()
        .min(max_suffix);

    // 整段替换：没有任何共同前后缀锚点，无法认定是局部词级修改。
    if prefix == 0 && suffix == 0 {
        return None;
    }

    let mid_a: String = a[prefix..a.len() - suffix].iter().collect();
    let mid_b: String = b[prefix..b.len() - suffix].iter().collect();

    // 纯插入/纯删除：一侧中间段（去掉空白后）为空，不是“改词”。
    if mid_a.trim().is_empty() || mid_b.trim().is_empty() {
        return None;
    }

    let wrong = normalize_segment(&mid_a);
    let corrected = normalize_segment(&mid_b);

    if wrong == corrected {
        return None; // 纯空白差异
    }
    if wrong.to_lowercase() == corrected.to_lowercase() {
        return None; // 纯大小写
    }
    if strip_negation(&wrong) == strip_negation(&corrected) {
        return None; // 否定词变化
    }
    if !valid_segment(&wrong) || !valid_segment(&corrected) {
        return None;
    }
    Some((wrong, corrected))
}

/// 学习门槛的统一包装：粘贴-修改学习链路只调它，不直接调
/// [`diff_word_correction`]，保证阈值只有一处收口。
pub fn should_learn(inserted: &str, later: &str) -> bool {
    !inserted.trim().is_empty()
        && !later.trim().is_empty()
        && inserted != later
        && diff_word_correction(inserted, later).is_some()
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// store 文件路径（`<data_dir>/learned_corrections.json`）。
pub fn learn_store_path(data_dir: &Path) -> PathBuf {
    data_dir.join("learned_corrections.json")
}

/// 读取 store；文件不存在或损坏时返回空 store（损坏只告警，不阻塞学习）。
pub fn load_learn_store(data_dir: &Path) -> LearnStore {
    let path = learn_store_path(data_dir);
    match std::fs::read_to_string(&path) {
        Ok(text) if !text.trim().is_empty() => match serde_json::from_str::<LearnStore>(&text) {
            Ok(store) => store,
            Err(err) => {
                log::warn!(
                    "learned-corrections store corrupt ({}), starting fresh: {}",
                    path.display(),
                    err
                );
                LearnStore::default()
            }
        },
        _ => LearnStore::default(),
    }
}

/// 写 store：先写临时文件再原子改名，避免半截写坏 JSON。
pub fn save_learn_store(data_dir: &Path, store: &LearnStore) -> std::io::Result<()> {
    let path = learn_store_path(data_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(store).map_err(std::io::Error::from)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, &path)
}

/// 学习入口（后续由粘贴-修改采集接线调用）：过 [`should_learn`] 阈值、
/// 查 blocked、去重累加 `count`，返回写入/更新后的候选。
/// 命中 blocked 或未过阈值返回 `None`；落盘失败只告警（学习是尽力而为，
/// 不能阻塞输入路径）。
pub fn observe_paste_correction(
    data_dir: &Path,
    inserted: &str,
    later: &str,
) -> Option<LearnedCandidate> {
    if !should_learn(inserted, later) {
        return None;
    }
    let (wrong, corrected) = diff_word_correction(inserted, later)?;
    let _guard = STORE_IO_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut store = load_learn_store(data_dir);
    if store
        .blocked
        .iter()
        .any(|p| p.wrong == wrong && p.corrected == corrected)
    {
        return None;
    }
    let now = now_unix_ms();
    let candidate = if let Some(existing) = store
        .candidates
        .iter_mut()
        .find(|c| c.wrong == wrong && c.corrected == corrected)
    {
        existing.count += 1;
        existing.last_seen = now;
        existing.clone()
    } else {
        let candidate = LearnedCandidate {
            wrong,
            corrected,
            count: 1,
            first_seen: now,
            last_seen: now,
        };
        store.candidates.push(candidate.clone());
        candidate
    };
    if let Err(err) = save_learn_store(data_dir, &store) {
        log::warn!(
            "failed to persist learned-corrections store ({}): {}",
            learn_store_path(data_dir).display(),
            err
        );
    }
    Some(candidate)
}

/// 撤销：删除所有原词匹配的候选，并把对应词对加入 `blocked`（持久化），
/// 之后同一对不再被学习。返回被撤销的条数；0 表示没有匹配的候选。
pub fn undo_learned_pairs(data_dir: &Path, wrong: &str) -> usize {
    let _guard = STORE_IO_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut store = load_learn_store(data_dir);
    let (matched, rest): (Vec<_>, Vec<_>) = store
        .candidates
        .iter()
        .cloned()
        .partition(|c| c.wrong == wrong);
    if matched.is_empty() {
        return 0;
    }
    let now = now_unix_ms();
    for candidate in &matched {
        if !store
            .blocked
            .iter()
            .any(|p| p.wrong == candidate.wrong && p.corrected == candidate.corrected)
        {
            store.blocked.push(BlockedPair {
                wrong: candidate.wrong.clone(),
                corrected: candidate.corrected.clone(),
                blocked_at: now,
            });
        }
    }
    store.candidates = rest;
    if let Err(err) = save_learn_store(data_dir, &store) {
        log::warn!(
            "failed to persist learned-corrections store ({}): {}",
            learn_store_path(data_dir).display(),
            err
        );
    }
    matched.len()
}

/// 只读预览当前学到的纠错候选（按最近出现时间倒序）。只读 store，
/// 不写 `custom_words`、不影响任何纠错行为。
#[tauri::command]
#[specta::specta]
pub fn preview_learned_candidates(app: tauri::AppHandle) -> Result<Vec<LearnedCandidate>, String> {
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let _guard = STORE_IO_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let store = load_learn_store(&data_dir);
    let mut candidates = store.candidates;
    candidates.sort_by(|a, b| {
        b.last_seen
            .cmp(&a.last_seen)
            .then_with(|| a.wrong.cmp(&b.wrong))
    });
    Ok(candidates)
}

/// 撤销一条已学习的候选：删除所有原词为 `wrong` 的候选，并把对应
/// `(wrong → corrected)` 词对加入 `blocked` 持久化，之后同一对不再被学习。
/// 返回被撤销的候选条数；0 表示没有匹配的候选（不新增 blocked 条目）。
#[tauri::command]
#[specta::specta]
pub fn undo_learned_candidate(app: tauri::AppHandle, wrong: String) -> Result<usize, String> {
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let wrong = wrong.trim().to_string();
    if wrong.is_empty() {
        return Err("wrong 不能为空".to_string());
    }
    Ok(undo_learned_pairs(&data_dir, &wrong))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "voo-learn-corrections-{}-{}",
            std::process::id(),
            seq
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // ===== 正例 =====

    #[test]
    fn accepts_word_fix_with_shared_context() {
        assert_eq!(
            diff_word_correction("帮我查一下王晓明", "帮我查一下王小明"),
            Some(("晓".to_string(), "小".to_string()))
        );
    }

    #[test]
    fn accepts_hanzi_plus_latin_segment() {
        assert_eq!(
            diff_word_correction("起Ray服务", "起服务Redis"),
            Some(("Ray服务".to_string(), "服务Redis".to_string()))
        );
    }

    #[test]
    fn accepts_exactly_24_chars_and_4_tokens() {
        let wrong = "甲".repeat(24);
        let corrected = "乙".repeat(24);
        assert_eq!(
            diff_word_correction(&format!("看{wrong}完"), &format!("看{corrected}完")),
            Some((wrong, corrected))
        );
        // 4 个 token 恰好在限内
        assert!(diff_word_correction("他说 甲 乙 丙 好", "他说 甲 乙 丁 好").is_some());
    }

    // ===== 反例：对齐/形态 =====

    #[test]
    fn rejects_whole_replacement_without_anchor() {
        assert_eq!(diff_word_correction("苹果", "香蕉"), None);
        assert_eq!(diff_word_correction("今天开会", "明天放假"), None);
    }

    #[test]
    fn rejects_pure_insertion_or_deletion() {
        assert_eq!(diff_word_correction("开会", "开全会"), None); // 纯插入
        assert_eq!(diff_word_correction("开全会", "开会"), None); // 纯删除
    }

    #[test]
    fn rejects_identical_or_empty_inputs() {
        assert_eq!(diff_word_correction("一样", "一样"), None);
        assert_eq!(diff_word_correction("", "词"), None);
        assert_eq!(diff_word_correction("词", ""), None);
    }

    #[test]
    fn rejects_case_only_change() {
        assert_eq!(diff_word_correction("买V版", "买v版"), None);
    }

    #[test]
    fn rejects_whitespace_only_change() {
        assert_eq!(diff_word_correction("王 小 明", "王小明"), None);
    }

    // ===== 反例：内容白名单 =====

    #[test]
    fn rejects_digit_change() {
        assert_eq!(diff_word_correction("用了3天", "用了三天"), None);
        assert_eq!(diff_word_correction("第2版", "第贰版"), None);
    }

    #[test]
    fn rejects_intra_sentence_punctuation() {
        assert_eq!(diff_word_correction("他说，好的", "他说：好的"), None);
        assert_eq!(diff_word_correction("发张三、李四", "发张三/李四"), None);
    }

    #[test]
    fn rejects_latin_only_segment() {
        // 两侧中间段都无汉字（纯拉丁）：只接受纯汉字或汉字+拉丁
        assert_eq!(
            diff_word_correction("部署Redis集群", "部署Valkey集群"),
            None
        );
    }

    #[test]
    fn rejects_negation_flip() {
        assert_eq!(diff_word_correction("他不来", "他没来"), None); // 不 ↔ 没
        assert_eq!(diff_word_correction("今天没去成", "今天非去成"), None); // 没 ↔ 非
    }

    #[test]
    fn rejects_over_token_or_char_limit() {
        // 5 个 token
        assert_eq!(
            diff_word_correction("他说 甲 乙 丙 丁 戊 好", "他说 子 丑 寅 好"),
            None
        );
        // 25 个字符
        let wrong = "甲".repeat(25);
        assert_eq!(
            diff_word_correction(&format!("看{wrong}完"), "看乙完"),
            None
        );
    }

    // ===== should_learn 包装 =====

    #[test]
    fn should_learn_wraps_thresholds() {
        assert!(should_learn("帮我查一下王晓明", "帮我查一下王小明"));
        assert!(!should_learn("一样", "一样")); // 相同
        assert!(!should_learn("", "")); // 空
        assert!(!should_learn("买V版", "买v版")); // 纯大小写
        assert!(!should_learn("苹果", "香蕉")); // 整段替换
    }

    // ===== 持久化 / blocked / undo =====

    #[test]
    fn observe_upserts_count_and_persists() {
        let dir = test_dir();
        let first = observe_paste_correction(&dir, "帮我查一下王晓明", "帮我查一下王小明").unwrap();
        assert_eq!(first.count, 1);
        let second =
            observe_paste_correction(&dir, "帮我查一下王晓明", "帮我查一下王小明").unwrap();
        assert_eq!(second.count, 2);

        let store = load_learn_store(&dir);
        assert_eq!(store.candidates.len(), 1);
        assert_eq!(store.candidates[0].count, 2);
        assert_eq!(store.candidates[0].wrong, "晓");
        assert_eq!(store.candidates[0].corrected, "小");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn observe_rejects_threshold_failures() {
        let dir = test_dir();
        assert!(observe_paste_correction(&dir, "苹果", "香蕉").is_none());
        assert!(observe_paste_correction(&dir, "用了3天", "用了三天").is_none());
        assert!(load_learn_store(&dir).candidates.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn undo_moves_pair_to_blocked_and_blocks_future_learning() {
        let dir = test_dir();
        observe_paste_correction(&dir, "帮我查一下王晓明", "帮我查一下王小明").unwrap();
        assert_eq!(undo_learned_pairs(&dir, "晓"), 1);

        let store = load_learn_store(&dir);
        assert!(store.candidates.is_empty());
        assert_eq!(store.blocked.len(), 1);
        assert_eq!(store.blocked[0].wrong, "晓");
        assert_eq!(store.blocked[0].corrected, "小");

        // 同一对之后不再被学习
        assert!(observe_paste_correction(&dir, "帮我查一下王晓明", "帮我查一下王小明").is_none());
        assert!(load_learn_store(&dir).candidates.is_empty());

        // 重复 undo / 不存在的 wrong：不新增 blocked、返回 0
        assert_eq!(undo_learned_pairs(&dir, "晓"), 0);
        assert_eq!(load_learn_store(&dir).blocked.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn undo_only_touches_matching_wrong() {
        let dir = test_dir();
        observe_paste_correction(&dir, "帮我查一下王晓明", "帮我查一下王小明").unwrap();
        observe_paste_correction(&dir, "明天去上海", "明天去海上").unwrap();
        assert_eq!(undo_learned_pairs(&dir, "晓"), 1);

        let store = load_learn_store(&dir);
        assert_eq!(store.candidates.len(), 1);
        assert_eq!(store.candidates[0].wrong, "上海");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn persisted_json_uses_candidates_and_blocked_keys() {
        let dir = test_dir();
        observe_paste_correction(&dir, "帮我查一下王晓明", "帮我查一下王小明").unwrap();
        undo_learned_pairs(&dir, "晓");
        let text = std::fs::read_to_string(learn_store_path(&dir)).unwrap();
        assert!(text.contains("\"candidates\""));
        assert!(text.contains("\"blocked\""));
        assert!(text.contains("\"wrong\""));
        assert!(text.contains("\"corrected\""));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_store_starts_fresh() {
        let dir = test_dir();
        std::fs::write(learn_store_path(&dir), "{ not json").unwrap();
        assert!(load_learn_store(&dir).candidates.is_empty());
        // 且能继续写入
        observe_paste_correction(&dir, "帮我查一下王晓明", "帮我查一下王小明").unwrap();
        assert_eq!(load_learn_store(&dir).candidates.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
