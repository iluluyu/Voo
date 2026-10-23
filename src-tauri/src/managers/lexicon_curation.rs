//! LLM 辅助词库整理（直接基于历史转写文本提炼术语）。
//!
//! 触发：历史条数距上次整理新增 ≥ [`HISTORY_TRIGGER_COUNT`]（默认 200）
//! 或距上次整理 ≥ [`MIN_INTERVAL_SECS`]（默认 7 天）。
//!
//! 输入：**最近 [`SAMPLE_WINDOW_SECS`]（7 天）窗口内的全部历史转写文本**
//! （不做条数抽样），逐条优先后处理文本、否则原始转写，trim、条内换行
//! 折叠为空格、跳过空与纯数字、截断到 [`MAX_ENTRY_CHARS`]（200 字符）、
//! 去重后按「1. …」编号拼接，总字符上限 [`MAX_TOTAL_CHARS`]（16000）。
//! 超出上限时**按最近优先保留**、丢弃更早条目，并在日志与返回值里记录
//! sent_entries / dropped_entries / chars（绝不静默截断）。发送内容只有
//! 这些截断后的转写文本，不含 URL、窗口标题、文件名。
//!
//! 幻觉防护（代码层强制，不依赖 prompt）：LLM 返回的 term 必须**逐字
//! 出现**在送出文本中（[`filter_hallucinated`]），否则丢弃该条。
//!
//! 解析严格：非法 JSON / 多余字段 / term 超长或全数字 / 空词一律丢弃
//! **该条**并记 debug 日志（不 warn），失败不影响主流程。
//!
//! 权限模型（docs/product-strategy.md 四态）：LLM 结果**只写 Candidate
//! 存储**（app 数据目录 `lexicon_curation.json`，内存 + 磁盘），绝不进入
//! `custom_words` / `imported_lexicons` 等纠错索引。前端经
//! [`preview_lexicon_curation`] 只读预览；提升/忽略交互后续版本实现。
//! 状态文件同时记录上次整理的来源条目数、丢弃数、字符数与来源条目 id，
//! 便于追溯每次候选出自哪些历史会话。
//!
//! 节流：一次整理最多一次 LLM 请求；进程内同时至多一个整理任务
//! （[`RUNNING`]）；请求失败后 [`RETRY_BACKOFF_SECS`]（24h）内不再重试。
//! 上次时间与结果计数记录在同一状态文件。
//!
//! 开关：settings `post_process_lexicon_curation_enabled`（默认开启）；
//! 环境变量 `VOO_LEXICON_CURATION` 显式设置时优先生效
//! （`0/false/off/no/disabled` 关闭，其余任何值开启），保留紧急关闭/
//! 测试通道，由 [`curation_enabled`] 统一判定。

use crate::managers::history::{HistoryEntry, HistoryManager};
use crate::settings::get_settings;
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use tauri::{AppHandle, Manager};

// ---------------------------------------------------------------------------
// 参数
// ---------------------------------------------------------------------------

/// 开关环境变量：显式设置时覆盖 settings（紧急关闭/测试通道）。
const ENV_ENABLED: &str = "VOO_LEXICON_CURATION";
/// 历史条数触发阈值：距上次整理新增 ≥ 此值。
const HISTORY_TRIGGER_COUNT: i64 = 200;
/// 时间触发阈值：距上次整理 ≥ 7 天。
const MIN_INTERVAL_SECS: i64 = 7 * 24 * 60 * 60;
/// 取样窗口：只看最近 7 天的转写文本（边界含当天整 7 天前的条目）。
const SAMPLE_WINDOW_SECS: i64 = 7 * 24 * 60 * 60;
/// 请求失败后的重试退避：24h 内不再重试。
const RETRY_BACKOFF_SECS: i64 = 24 * 60 * 60;
/// 单条历史文本截断长度（Unicode 标量数；条内换行折叠为空格）。
const MAX_ENTRY_CHARS: usize = 200;
/// 拼接后送出文本的总字符上限；超出时按最近优先保留，更早条目丢弃并计数。
const MAX_TOTAL_CHARS: usize = 16_000;
/// 词条（term 与每个 alias）的长度区间（Unicode 标量数）。
const MIN_TERM_CHARS: usize = 2;
const MAX_TERM_CHARS: usize = 50;
/// 每条候选最多接受的别名数。
const MAX_ALIASES: usize = 3;
/// evidence / reason 截断长度。
const MAX_EVIDENCE_CHARS: usize = 200;
/// 单次整理最多接受的候选数（防御 LLM 超量输出）。
const MAX_ACCEPTED_PER_RUN: usize = 20;
/// Candidate 存储上限（超出按 created_at 淘汰最旧）。
const MAX_STORED_CANDIDATES: usize = 100;

/// 整理请求的输出 token 预算：输出是候选数组，比听写后处理的统一
/// 512 上限更宽，避免候选列表被截断（旧版聊天端点不设上限，这是最接
/// 近的保守值）。
const CURATION_MAX_OUTPUT_TOKENS: u32 = 4096;

/// 整理请求的 system prompt：只允许从给定编号文本里提取术语。
const CURATION_SYSTEM_PROMPT: &str = "你是中文语音输入的个人词库整理助手。用户会提供按序号编号的历史转写文本片段（每条已截断）。请只从这些给定文本中提取频繁出现且容易被语音识别写错的专有名词、领域术语、产品名、品牌、人名与缩写（含中英混排，如软件名、型号、项目代号）；通用日常用语、语法词、单字、纯数字一律不要。每个入选词输出：term（必须逐字来自给定文本，不得改写、翻译、合并或编造）、aliases（同一事物的其他常见写法，最多 3 个，可为空数组）、evidence（从给定文本中逐字摘录的、包含该词的短片段）、reason（一句话中文说明它为什么值得加入个人热词库）。只输出 JSON 数组，不要任何其他文本。";

// ---------------------------------------------------------------------------
// 数据结构
// ---------------------------------------------------------------------------

/// 一个词库整理候选（Candidate 态：只供预览与确认，绝不进纠错索引）。
#[derive(Debug, Clone, Serialize, Deserialize, specta::Type, PartialEq)]
pub struct LexiconCandidate {
    /// 术语本体（2–50 字符，允许中英混排；必须逐字出现在送出文本中）。
    pub term: String,
    /// 同指别名（同样校验过的写法，最多 3 个）。
    pub aliases: Vec<String>,
    /// 一句话入选理由（LLM reason，截断至 200 字）。
    pub evidence: String,
    /// term 在本次送出文本中的出现次数。
    pub count: u64,
    /// 写入 Candidate 存储的时间（unix 秒）。
    pub created_at: i64,
}

/// 整理状态：上次时间与结果计数（与候选一起持久化到同一 JSON 文件）。
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct CurationState {
    /// 上次完成整理（含"无候选"的成功空跑）的时间。
    #[serde(default)]
    last_run_at: Option<i64>,
    /// 上次请求失败的时间（24h 退避用）。
    #[serde(default)]
    last_failure_at: Option<i64>,
    /// 上次整理时的历史总条数（条数阈值按增量算）。
    #[serde(default)]
    history_count_at_run: i64,
    /// 上次整理接受的候选数。
    #[serde(default)]
    last_accepted: usize,
    /// 上次整理因严格校验（含幻觉过滤）丢弃的条数。
    #[serde(default)]
    last_rejected: usize,
    /// 上次整理实际送出的历史条目数。
    #[serde(default)]
    last_sent_entries: usize,
    /// 上次整理因总字符上限被丢弃（更早）的条目数。
    #[serde(default)]
    last_dropped_entries: usize,
    /// 上次整理送出文本的总字符数。
    #[serde(default)]
    last_sent_chars: usize,
    /// 上次整理送出的历史条目 id（追溯候选来源会话）。
    #[serde(default)]
    last_source_entry_ids: Vec<i64>,
    /// Candidate 存储。
    candidates: Vec<LexiconCandidate>,
}

/// [`should_run`] 的判定结果（仅用于日志与测试）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurationTrigger {
    /// 不触发。
    No,
    /// 条数阈值触发。
    Count,
    /// 时间阈值触发。
    Interval,
}

// ---------------------------------------------------------------------------
// 进程内状态（内存 + 磁盘）
// ---------------------------------------------------------------------------

/// 进程内同时至多一个整理任务。
static RUNNING: AtomicBool = AtomicBool::new(false);

/// Candidate 存储 + 整理状态（内存镜像；磁盘为唯一持久层）。
static STATE: OnceLock<Mutex<CurationState>> = OnceLock::new();

/// 状态文件路径（首次 [`ensure_store`] 时固定）。
static STORE_PATH: OnceLock<PathBuf> = OnceLock::new();

fn state_cell() -> &'static Mutex<CurationState> {
    STATE.get_or_init(|| Mutex::new(CurationState::default()))
}

fn lock_state() -> std::sync::MutexGuard<'static, CurationState> {
    // 中毒说明持有者 panic 过；状态文件才是权威，恢复继续用即可。
    state_cell()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// 初始化存储：固定状态文件路径并（仅首次）从磁盘加载。
/// 并发调用时两边各自加载同一文件，后写者胜——命令只读、整理任务由
/// [`RUNNING`] 串行，实际写入方唯一，竞态无害。
fn ensure_store(app: &AppHandle) -> Option<PathBuf> {
    if let Some(path) = STORE_PATH.get() {
        return Some(path.clone());
    }
    let dir = match crate::portable::app_data_dir(app) {
        Ok(dir) => dir,
        Err(e) => {
            debug!("lexicon curation: no app data dir ({e:?}); memory-only");
            return None;
        }
    };
    let path = dir.join("lexicon_curation.json");
    match std::fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<CurationState>(&text) {
            Ok(loaded) => *lock_state() = loaded,
            Err(e) => debug!("lexicon curation: state file corrupt, starting fresh: {e}"),
        },
        Err(_) => debug!("lexicon curation: no state file yet at {path:?}"),
    }
    let _ = STORE_PATH.set(path.clone());
    Some(path)
}

/// 把内存状态写回磁盘（失败只记 debug，不影响主流程）。
fn persist_state(path: &PathBuf, state: &CurationState) {
    match serde_json::to_string_pretty(state) {
        Ok(text) => {
            if let Err(e) = std::fs::write(path, text) {
                debug!("lexicon curation: failed to persist state: {e}");
            }
        }
        Err(e) => debug!("lexicon curation: failed to serialize state: {e}"),
    }
}

// ---------------------------------------------------------------------------
// 开关
// ---------------------------------------------------------------------------

/// 开关：settings `post_process_lexicon_curation_enabled`（默认开启）；
/// 环境变量 `VOO_LEXICON_CURATION` 显式设置时优先（`0/false/off/no/disabled`
/// 大小写不敏感 = 关闭，其余任何值 = 开启）。
fn curation_enabled(app: &AppHandle) -> bool {
    if let Ok(value) = std::env::var(ENV_ENABLED) {
        return env_disables(&value);
    }
    get_settings(app).post_process_lexicon_curation_enabled
}

/// 解析开关取值：只有显式的 false 类词才关闭，其余任何值（含未设置）开启。
fn env_disables(value: &str) -> bool {
    !matches!(
        value.trim().to_lowercase().as_str(),
        "0" | "false" | "off" | "no" | "disabled"
    )
}

// ---------------------------------------------------------------------------
// 触发判定（纯函数，单测覆盖）
// ---------------------------------------------------------------------------

/// 是否应触发一次整理：失败退避优先，其次条数阈值，最后时间阈值。
/// 从未整理过（`last_run_at == None`）时只有条数阈值能触发
/// （`history_count_at_run` 默认 0，即历史累积到 200 条首次触发）。
fn should_run(now: i64, state: &CurationState, history_count: i64) -> CurationTrigger {
    if let Some(failed) = state.last_failure_at {
        if now.saturating_sub(failed) < RETRY_BACKOFF_SECS {
            return CurationTrigger::No;
        }
    }
    let new_entries = history_count.saturating_sub(state.history_count_at_run);
    if new_entries >= HISTORY_TRIGGER_COUNT {
        return CurationTrigger::Count;
    }
    if let Some(last_run) = state.last_run_at {
        if now.saturating_sub(last_run) >= MIN_INTERVAL_SECS {
            return CurationTrigger::Interval;
        }
    }
    CurationTrigger::No
}

// ---------------------------------------------------------------------------
// 历史文本取样（纯函数，单测覆盖）
// ---------------------------------------------------------------------------

/// 7 天窗口取样结果：送出文本 + 来源条目 id + 因上限丢弃数。
#[derive(Debug, Default, Clone, PartialEq)]
struct SampledCorpus {
    /// 按序号拼接的送出文本（即发给 LLM 的用户消息全文）。
    text: String,
    /// 送出条目的历史 id（按新到旧）。
    entry_ids: Vec<i64>,
    /// 本可送出但因总字符上限被丢弃的（更早）条目数。
    dropped_entries: usize,
}

/// 从历史条目构建送出文本：取 [`SAMPLE_WINDOW_SECS`] 窗口内**全部**条目
/// （不做条数抽样），按新到旧稳定排序后逐条：优先后处理文本、否则原始
/// 转写；trim、条内换行折叠为空格、跳过空与纯数字、截断
/// [`MAX_ENTRY_CHARS`]、与已送出条目去重；以「n. 」编号拼接。累计超过
/// [`MAX_TOTAL_CHARS`] 即停止追加（最近优先），其后所有本可送出的条目
/// 计入 `dropped_entries`（空/纯数字/重复不计——它们本就不送）。
fn build_sample_corpus(entries: &[HistoryEntry], now: i64) -> SampledCorpus {
    let mut windowed: Vec<&HistoryEntry> = entries
        .iter()
        .filter(|e| now.saturating_sub(e.timestamp) <= SAMPLE_WINDOW_SECS)
        .collect();
    // 输入顺序不应影响结果：稳定排序让「最近优先」只由时间（并列看 id）决定。
    windowed.sort_by(|a, b| b.timestamp.cmp(&a.timestamp).then(b.id.cmp(&a.id)));

    let mut corpus = SampledCorpus::default();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut total = 0usize;
    let mut capped = false;
    for entry in windowed {
        let raw = entry
            .post_processed_text
            .as_deref()
            .filter(|t| !t.trim().is_empty())
            .unwrap_or(&entry.transcription_text);
        let line: String = raw
            .trim()
            .replace(['\n', '\r'], " ")
            .chars()
            .take(MAX_ENTRY_CHARS)
            .collect();
        if line.trim().is_empty() || line.chars().all(|c| c.is_numeric()) {
            continue; // 空或纯数字：本就不在候选之列，不计入丢弃。
        }
        if capped {
            // 已超总字符上限：此后（更早的）条目按最近优先全部丢弃并计数。
            corpus.dropped_entries += 1;
            continue;
        }
        if !seen.insert(line.clone()) {
            continue; // 与已送出条目重复。
        }
        let numbered = format!("{}. {}\n", corpus.entry_ids.len() + 1, line);
        if total + numbered.chars().count() > MAX_TOTAL_CHARS {
            capped = true;
            corpus.dropped_entries += 1;
            seen.remove(&line); // 未送出，不占用去重槽。
            continue;
        }
        total += numbered.chars().count();
        corpus.text.push_str(&numbered);
        corpus.entry_ids.push(entry.id);
    }
    if corpus.text.ends_with('\n') {
        corpus.text.pop();
    }
    corpus
}

// ---------------------------------------------------------------------------
/// 严格解析（纯函数，单测覆盖）
// ---------------------------------------------------------------------------

/// LLM 单条原始输出的反序列化形状。`deny_unknown_fields`：多余字段
/// 让**该条**解析失败（丢弃该条，而不是整个响应）。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawItem {
    term: String,
    aliases: Vec<String>,
    evidence: String,
    reason: String,
}

/// 词（term / alias）校验：trim 后 2–50 个字符，且至少含一个字母或汉字
/// （`is_alphabetic`，中英混排/产品名/缩写都放行；纯数字、纯标点、
/// 纯空白不合格）。逐字出现与否由 [`filter_hallucinated`] 另行强制。
fn valid_term(term: &str) -> bool {
    let trimmed = term.trim();
    let len = trimmed.chars().count();
    (MIN_TERM_CHARS..=MAX_TERM_CHARS).contains(&len) && trimmed.chars().any(char::is_alphabetic)
}

/// 严格解析 LLM 输出。期望顶层 JSON 数组 `[{term, aliases, evidence,
/// reason}]`（兼容旧包装 `{"candidates": [...]}`），逐条按 [`RawItem`]
/// （禁多余字段）解析并校验：非法 JSON、顶层形状不符 → 全部丢弃；单条
/// 字段缺失/多余/类型错、term 或 alias 不合格、term 为空 → 丢弃该条并
/// 计数。`evidence`（逐字摘录）只用于 debug 日志（release 下脱敏），
/// 持久化存 `reason`（入选理由，与前端 `evidence` 字段语义一致）。
/// 所有丢弃只记 debug。返回 (合格候选, 丢弃数)。
fn parse_llm_candidates(content: &str, now: i64) -> (Vec<LexiconCandidate>, usize) {
    let content = strip_think_block(content.trim());
    let parsed: serde_json::Value = match serde_json::from_str(content) {
        Ok(value) => value,
        Err(e) => {
            debug!("lexicon curation: response is not valid JSON ({e}); all dropped");
            return (Vec::new(), 0);
        }
    };
    let items: &[serde_json::Value] = if let Some(array) = parsed.as_array() {
        array
    } else if let Some(array) = parsed.get("candidates").and_then(|v| v.as_array()) {
        array
    } else {
        debug!("lexicon curation: response is not a JSON array; all dropped");
        return (Vec::new(), 0);
    };

    let mut accepted: Vec<LexiconCandidate> = Vec::new();
    let mut rejected = 0usize;
    let mut seen_terms: std::collections::HashSet<String> = std::collections::HashSet::new();
    for item in items {
        let raw = match serde_json::from_value::<RawItem>(item.clone()) {
            Ok(raw) => raw,
            Err(e) => {
                debug!("lexicon curation: dropped one entry ({e})");
                rejected += 1;
                continue;
            }
        };
        let term = raw.term.trim().to_string();
        if !valid_term(&term) {
            debug!("lexicon curation: dropped entry with invalid term (len/chars)");
            rejected += 1;
            continue;
        }
        if !seen_terms.insert(term.clone()) {
            debug!("lexicon curation: dropped duplicate term in response");
            rejected += 1;
            continue;
        }
        let mut aliases: Vec<String> = Vec::new();
        for alias in &raw.aliases {
            let alias = alias.trim().to_string();
            if alias == term || aliases.contains(&alias) {
                continue;
            }
            if valid_term(&alias) {
                aliases.push(alias);
            } else {
                debug!("lexicon curation: dropped one invalid alias");
            }
            if aliases.len() >= MAX_ALIASES {
                break;
            }
        }
        // LLM 的 evidence（逐字摘录）仅用于排查，不持久化；release 脱敏。
        let quote: String = raw
            .evidence
            .trim()
            .chars()
            .take(MAX_EVIDENCE_CHARS)
            .collect();
        debug!(
            "lexicon curation: term '{}' quote: {}",
            term,
            crate::utils::redact_text(&quote)
        );
        let evidence: String = raw.reason.trim().chars().take(MAX_EVIDENCE_CHARS).collect();
        accepted.push(LexiconCandidate {
            term,
            aliases,
            evidence,
            count: 0,
            created_at: now,
        });
    }
    (accepted, rejected)
}

/// 幻觉防护（代码层强制，不依赖 prompt）：term 必须**逐字出现**在送出
/// 文本中，否则丢弃该条（trim 后大小写敏感的逐字符 `contains`）。
/// 返回丢弃数。
fn filter_hallucinated(candidates: &mut Vec<LexiconCandidate>, sent_text: &str) -> usize {
    let before = candidates.len();
    candidates.retain(|c| sent_text.contains(c.term.trim()));
    before - candidates.len()
}

/// 去掉部分无法关掉思考的端点塞在 content 开头的 `<think>...</think>`。
fn strip_think_block(s: &str) -> &str {
    if let Some(rest) = s.strip_prefix("<think>") {
        if let Some(end) = rest.find("</think>") {
            return rest[end + "</think>".len()..].trim_start();
        }
    }
    s
}

// ---------------------------------------------------------------------------
/// 合并入 Candidate 存储（纯函数，单测覆盖）
// ---------------------------------------------------------------------------

/// 把通过校验的候选合并进 Candidate 存储：
/// - 已在生效热词表（`custom_words`/导入词库合并结果）里的词**跳过**，
///   不重复建议（本模块从不写 `custom_words`，这里只是避免冗余候选）；
/// - 已存在的候选只更新 aliases/evidence/count，保留原 created_at；
/// - 新候选追加；总量超过 [`MAX_STORED_CANDIDATES`] 时淘汰最旧。
fn merge_candidates(
    existing: &mut Vec<LexiconCandidate>,
    incoming: Vec<LexiconCandidate>,
    effective_words: &[String],
    count_by_term: &HashMap<String, u64>,
) {
    let active: std::collections::HashSet<&str> =
        effective_words.iter().map(|word| word.trim()).collect();
    for mut candidate in incoming {
        if active.contains(candidate.term.trim()) {
            debug!(
                "lexicon curation: '{}' already an active word; not re-suggested",
                candidate.term
            );
            continue;
        }
        candidate.count = count_by_term.get(&candidate.term).copied().unwrap_or(0);
        match existing.iter_mut().find(|c| c.term == candidate.term) {
            Some(stored) => {
                stored.aliases = candidate.aliases;
                stored.evidence = candidate.evidence;
                stored.count = candidate.count;
            }
            None => existing.push(candidate),
        }
    }
    if existing.len() > MAX_STORED_CANDIDATES {
        // 稳定排序后截掉最旧的（created_at 相同时保后来者）。
        existing.sort_by_key(|c| c.created_at);
        let overflow = existing.len() - MAX_STORED_CANDIDATES;
        existing.drain(0..overflow);
    }
}

// ---------------------------------------------------------------------------
/// LLM 请求
// ---------------------------------------------------------------------------

/// 结构化输出 schema：顶层数组 `[{term, aliases[], evidence, reason}]`，
/// 全字段必填、禁多余属性（与解析端的严格策略一致）。
fn output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "term": { "type": "string" },
                "aliases": { "type": "array", "items": { "type": "string" } },
                "evidence": { "type": "string" },
                "reason": { "type": "string" }
            },
            "required": ["term", "aliases", "evidence", "reason"],
            "additionalProperties": false
        }
    })
}

// ---------------------------------------------------------------------------
/// 整理执行
// ---------------------------------------------------------------------------

/// 入口（应用启动后与每次历史保存后调用）：开关关闭或已有整理在跑则
/// 直接返回，否则后台执行一次整理。
/// 绝不阻塞调用方，任何失败只影响整理自身。
pub fn maybe_start_curation(app: &AppHandle) {
    if !curation_enabled(app) {
        return;
    }
    if RUNNING.swap(true, Ordering::SeqCst) {
        debug!("lexicon curation: already running; skipping");
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let started = std::time::Instant::now();
        match run_curation(&app).await {
            Ok(outcome) => {
                info!(
                    "lexicon curation finished: {outcome} ({:?})",
                    started.elapsed()
                );
            }
            Err(e) => debug!("lexicon curation failed: {e}"),
        }
        RUNNING.store(false, Ordering::SeqCst);
    });
}

/// 执行一次整理（至多一次 LLM 请求）。返回 Ok(outcome 描述，含送出/
/// 丢弃/字符计数) 或 Err(原因)；配置不全/未到触发点不算失败（不记 24h
/// 退避），只有 LLM 请求失败/空响应才记 `last_failure_at`。
async fn run_curation(app: &AppHandle) -> Result<String, String> {
    let settings = get_settings(app);
    let provider = settings
        .active_post_process_provider()
        .cloned()
        .ok_or_else(|| "post-process provider not selected".to_string())?;
    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();
    if model.trim().is_empty() {
        return Err(format!(
            "post-process provider '{}' has no model configured",
            provider.id
        ));
    }

    let Some(store_path) = ensure_store(app) else {
        return Err("no app data dir for curation state".to_string());
    };

    let hm = std::sync::Arc::clone(&app.state::<std::sync::Arc<HistoryManager>>());
    let paginated = hm
        .get_history_entries(None, None)
        .await
        .map_err(|e| format!("failed to read history: {e}"))?;
    let history_count = paginated.entries.len() as i64;

    let now = chrono::Utc::now().timestamp();
    let trigger = {
        let state = lock_state();
        should_run(now, &state, history_count)
    };
    if trigger == CurationTrigger::No {
        return Ok("not due".to_string());
    }

    // 取样：7 天窗口内全部历史转写文本（不抽样），超上限按最近优先丢弃。
    let corpus = build_sample_corpus(&paginated.entries, now);
    let sent_entries = corpus.entry_ids.len();
    let sent_chars = corpus.text.chars().count();
    if corpus.text.is_empty() {
        let mut state = lock_state();
        state.last_run_at = Some(now);
        state.last_failure_at = None;
        state.history_count_at_run = history_count;
        state.last_accepted = 0;
        state.last_rejected = 0;
        state.last_sent_entries = 0;
        state.last_dropped_entries = 0;
        state.last_sent_chars = 0;
        state.last_source_entry_ids = Vec::new();
        persist_state(&store_path, &state);
        return Ok("no usable history in 7d window (empty run recorded)".to_string());
    }
    // 明示少发了多少：绝不静默截断（sent / dropped / chars 同时入日志、
    // 返回值与状态文件）。
    info!(
        "lexicon curation sample: sent_entries={sent_entries} dropped_entries={} chars={sent_chars} (cap {MAX_TOTAL_CHARS})",
        corpus.dropped_entries
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();
    // 思考档位与后处理共用同一设置（默认 none：整理是后台任务，不值得
    // 为它付出推理延迟）。
    let thinking = settings.post_process_thinking;

    let outcome = crate::llm_client::send_chat_completion_with_schema(
        &provider,
        api_key,
        &model,
        corpus.text.clone(),
        Some(CURATION_SYSTEM_PROMPT.to_string()),
        Some(output_schema()),
        thinking,
        CURATION_MAX_OUTPUT_TOKENS,
    )
    .await;

    let content = match outcome {
        Ok(Some(content)) => content,
        Ok(None) => {
            record_failure(&store_path, now);
            return Err("LLM response had no content".to_string());
        }
        Err(e) => {
            record_failure(&store_path, now);
            return Err(format!("LLM request failed: {e}"));
        }
    };

    let (mut accepted, mut rejected) = parse_llm_candidates(&content, now);
    // Blocked 档（含跨重导入持久、暂时不在生效词表里的词）不进建议；
    // 与 lexicon_suggest 的 excluded_by 同语义（双向包含），别名同样拦截。
    let blocked_terms = settings.blocked_suggest_terms();
    accepted.retain(|c| {
        let term = c.term.trim();
        !blocked_terms.iter().any(|b| {
            let b = b.trim();
            !b.is_empty() && (b.contains(term) || term.contains(b))
        })
    });
    // 幻觉防护：term 必须逐字出现在送出文本中，否则丢弃并计数。
    let hallucinated = filter_hallucinated(&mut accepted, &corpus.text);
    if hallucinated > 0 {
        debug!("lexicon curation: dropped {hallucinated} terms absent from sent text");
    }
    rejected += hallucinated;
    accepted.truncate(MAX_ACCEPTED_PER_RUN);
    // count = term 在送出文本中的出现次数（UI 频次展示沿用）。
    let count_by_term: HashMap<String, u64> = accepted
        .iter()
        .map(|c| {
            (
                c.term.clone(),
                corpus.text.matches(c.term.as_str()).count() as u64,
            )
        })
        .collect();

    let accepted_count = accepted.len();
    {
        let mut state = lock_state();
        merge_candidates(
            &mut state.candidates,
            accepted,
            &settings.effective_custom_words(),
            &count_by_term,
        );
        state.last_run_at = Some(now);
        state.last_failure_at = None;
        state.history_count_at_run = history_count;
        state.last_accepted = accepted_count;
        state.last_rejected = rejected;
        state.last_sent_entries = sent_entries;
        state.last_dropped_entries = corpus.dropped_entries;
        state.last_sent_chars = sent_chars;
        state.last_source_entry_ids = corpus.entry_ids;
        persist_state(&store_path, &state);
    }
    Ok(format!(
        "curation run recorded: sent_entries={sent_entries} dropped_entries={} chars={sent_chars}, accepted {accepted_count}, rejected {rejected}",
        corpus.dropped_entries
    ))
}

/// 记一次失败（24h 退避）：不动候选，不把 `last_run_at` 当成功推进。
fn record_failure(store_path: &PathBuf, now: i64) {
    let mut state = lock_state();
    state.last_failure_at = Some(now);
    persist_state(store_path, &state);
}

// ---------------------------------------------------------------------------
/// 只读命令（前端预览用）
// ---------------------------------------------------------------------------

/// 只读预览当前词库整理候选（Candidate 态，未激活、不影响纠错）。
#[tauri::command]
#[specta::specta]
pub fn preview_lexicon_curation(app: AppHandle) -> Result<Vec<LexiconCandidate>, String> {
    let _ = ensure_store(&app);
    let state = lock_state();
    Ok(state.candidates.clone())
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_700_000_000;

    fn history_entry(id: i64, timestamp: i64, text: &str) -> HistoryEntry {
        HistoryEntry {
            id,
            file_name: format!("file{id}.wav"),
            timestamp,
            saved: true,
            title: String::new(),
            transcription_text: text.to_string(),
            post_processed_text: None,
            post_process_prompt: None,
            post_process_requested: false,
        }
    }

    fn candidate(term: &str, created_at: i64) -> LexiconCandidate {
        LexiconCandidate {
            term: term.to_string(),
            aliases: Vec::new(),
            evidence: "测试理由".to_string(),
            count: 3,
            created_at,
        }
    }

    // ---- 触发与节流 ----

    #[test]
    fn threshold_triggers_at_200_new_entries() {
        let state = CurationState {
            last_run_at: Some(1_000_000),
            history_count_at_run: 100,
            ..Default::default()
        };
        // 差一条未到阈值不触发（300-100=200 已达标，改用 299）。
        assert_eq!(
            should_run(1_000_000 + 100, &state, 299),
            CurationTrigger::No
        );
        assert_eq!(
            should_run(1_000_000, &state, 100 + HISTORY_TRIGGER_COUNT),
            CurationTrigger::Count
        );
    }

    #[test]
    fn never_ran_requires_full_threshold() {
        let fresh = CurationState::default();
        assert_eq!(
            should_run(9_999_999, &fresh, HISTORY_TRIGGER_COUNT - 1),
            CurationTrigger::No
        );
        assert_eq!(
            should_run(9_999_999, &fresh, HISTORY_TRIGGER_COUNT),
            CurationTrigger::Count
        );
    }

    #[test]
    fn time_triggers_after_seven_days() {
        let now = 2_000_000_000;
        let state = CurationState {
            last_run_at: Some(now - MIN_INTERVAL_SECS),
            history_count_at_run: 500,
            ..Default::default()
        };
        assert_eq!(should_run(now, &state, 500), CurationTrigger::Interval);
        let recent = CurationState {
            last_run_at: Some(now - MIN_INTERVAL_SECS + 1),
            history_count_at_run: 500,
            ..Default::default()
        };
        assert_eq!(should_run(now, &recent, 500), CurationTrigger::No);
    }

    #[test]
    fn failure_blocks_for_24h_even_at_threshold() {
        let now = 2_000_000_000;
        let state = CurationState {
            last_run_at: Some(now - 100 * MIN_INTERVAL_SECS),
            last_failure_at: Some(now - RETRY_BACKOFF_SECS + 1),
            history_count_at_run: 0,
            ..Default::default()
        };
        // 条数阈值已满足，但失败退避优先。
        assert_eq!(
            should_run(now, &state, HISTORY_TRIGGER_COUNT * 5),
            CurationTrigger::No
        );
        // 退避过期后条数阈值放行。
        let expired = CurationState {
            last_failure_at: Some(now - RETRY_BACKOFF_SECS),
            history_count_at_run: 0,
            ..Default::default()
        };
        assert_eq!(
            should_run(now, &expired, HISTORY_TRIGGER_COUNT),
            CurationTrigger::Count
        );
    }

    // ---- 取样：7 天窗口 / 全量 / 去重 / 截断 / 上限 ----

    #[test]
    fn sampling_window_is_seven_days_inclusive() {
        let entries = vec![
            history_entry(1, NOW - SAMPLE_WINDOW_SECS, "整七天前的边界条目"),
            history_entry(2, NOW - SAMPLE_WINDOW_SECS - 1, "过期一秒的条目"),
            history_entry(3, NOW, "今天的条目"),
        ];
        let corpus = build_sample_corpus(&entries, NOW);
        assert_eq!(corpus.entry_ids, vec![3, 1], "按新到旧，边界含整 7 天");
        assert!(corpus.text.contains("边界条目"));
        assert!(!corpus.text.contains("过期一秒"));
        assert_eq!(corpus.dropped_entries, 0);
    }

    #[test]
    fn sampling_keeps_every_window_entry_without_count_cap() {
        let entries: Vec<HistoryEntry> = (0..120)
            .map(|i| history_entry(i, NOW - i, &format!("第{i}条内容不同的转写")))
            .collect();
        let corpus = build_sample_corpus(&entries, NOW);
        assert_eq!(corpus.entry_ids.len(), 120, "7 天内全部纳入，不做条数抽样");
        assert_eq!(corpus.dropped_entries, 0);
        assert!(corpus.text.starts_with("1. "));
        assert!(corpus.text.contains("\n120. "));
    }

    #[test]
    fn sampling_dedupes_and_skips_empty_and_digit_only() {
        let entries = vec![
            history_entry(1, NOW, "量子芯片良率提升"),
            history_entry(2, NOW - 10, "量子芯片良率提升"), // 与已送出重复
            history_entry(3, NOW - 20, "   \n\t "),         // 空白
            history_entry(4, NOW - 30, "12345"),            // 纯数字
            history_entry(5, NOW - 40, "２０２４"),         // 全角纯数字
            history_entry(6, NOW - 50, "第3章要点"),        // 含汉字：保留
        ];
        let corpus = build_sample_corpus(&entries, NOW);
        assert_eq!(corpus.entry_ids, vec![1, 6]);
        assert_eq!(corpus.dropped_entries, 0, "跳过≠丢弃，不计数");
        assert!(corpus.text.contains("第3章要点"));
    }

    #[test]
    fn sampling_truncates_each_entry_to_limit() {
        let long = "长".repeat(MAX_ENTRY_CHARS + 30);
        let corpus = build_sample_corpus(&[history_entry(1, NOW, &long)], NOW);
        let line = corpus.text.strip_prefix("1. ").expect("numbered line");
        assert_eq!(line.chars().count(), MAX_ENTRY_CHARS);
    }

    #[test]
    fn sampling_folds_newlines_and_prefers_post_processed_text() {
        let mut entry = history_entry(1, NOW, "原始\n转写");
        entry.post_processed_text = Some("后处理\n文本".to_string());
        let corpus = build_sample_corpus(&[entry], NOW);
        assert_eq!(corpus.text, "1. 后处理 文本");
        assert!(!corpus.text.contains("原始"));
    }

    #[test]
    fn sampling_drops_oldest_beyond_char_cap_and_counts() {
        // 每条 ~MAX_ENTRY_CHARS 字符且首 4 字互不相同，容量只装得下前若干条。
        let entries: Vec<HistoryEntry> = (0..100)
            .map(|i| {
                let text = format!("{i:03}词") + &"字".repeat(MAX_ENTRY_CHARS);
                history_entry(i, NOW - i, &text)
            })
            .collect();
        let mut shuffled = entries.clone();
        shuffled.reverse(); // 打乱输入顺序也无妨：builder 内部按时间排序。
        let corpus = build_sample_corpus(&shuffled, NOW);
        let sent = corpus.entry_ids.len();
        assert!(sent >= 1);
        assert!(corpus.text.chars().count() <= MAX_TOTAL_CHARS);
        assert_eq!(
            sent + corpus.dropped_entries,
            100,
            "送出 + 因上限丢弃 = 窗口内全部可送条目"
        );
        // 最近优先：id 0/1（最新）在、99（最旧）不在，且顺序新到旧。
        assert!(corpus.entry_ids.contains(&0));
        assert!(corpus.entry_ids.contains(&1));
        assert!(!corpus.entry_ids.contains(&99));
        assert_eq!(corpus.entry_ids, (0..sent as i64).collect::<Vec<_>>());
    }

    #[test]
    fn sampling_empty_when_nothing_usable() {
        let entries = vec![
            history_entry(1, NOW - SAMPLE_WINDOW_SECS - 5, "太老"),
            history_entry(2, NOW, "  "),
        ];
        let corpus = build_sample_corpus(&entries, NOW);
        assert!(corpus.text.is_empty());
        assert!(corpus.entry_ids.is_empty());
        assert_eq!(corpus.dropped_entries, 0);
    }

    // ---- 严格解析 ----

    #[test]
    fn parse_accepts_valid_response() {
        let content = r#"[
            {"term":"量子芯片","aliases":["量子晶片"],"evidence":"提到量子芯片良率","reason":"高频领域术语，易被写成量子芯品"}
        ]"#;
        let (accepted, rejected) = parse_llm_candidates(content, NOW);
        assert_eq!(rejected, 0);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].term, "量子芯片");
        assert_eq!(accepted[0].aliases, vec!["量子晶片".to_string()]);
        // 持久化存 reason（入选理由），evidence 摘录只进日志。
        assert_eq!(accepted[0].evidence, "高频领域术语，易被写成量子芯品");
        assert_eq!(accepted[0].created_at, NOW);
    }

    #[test]
    fn parse_tolerates_candidates_wrapper() {
        let content =
            r#"{"candidates":[{"term":"量子芯片","aliases":[],"evidence":"q","reason":"r"}]}"#;
        let (accepted, rejected) = parse_llm_candidates(content, NOW);
        assert_eq!(rejected, 0);
        assert_eq!(accepted.len(), 1);
    }

    #[test]
    fn parse_invalid_json_drops_everything() {
        for bad in ["这不是JSON", "[}", "", "{\"candidates\": [}"] {
            let (accepted, rejected) = parse_llm_candidates(bad, NOW);
            assert!(accepted.is_empty(), "no candidate should survive {bad:?}");
            assert_eq!(rejected, 0);
        }
    }

    #[test]
    fn parse_wrong_top_level_shape_drops_everything() {
        let (accepted, _) = parse_llm_candidates(r#"{"terms":[]}"#, NOW);
        assert!(accepted.is_empty());
        let (accepted, _) = parse_llm_candidates(r#""一个字符串""#, NOW);
        assert!(accepted.is_empty());
    }

    #[test]
    fn parse_extra_field_drops_only_that_entry() {
        let content = r#"[
            {"term":"甲术语","aliases":[],"evidence":"a","reason":"ra","extra":1},
            {"term":"乙术语","aliases":[],"evidence":"b","reason":"rb"}
        ]"#;
        let (accepted, rejected) = parse_llm_candidates(content, NOW);
        assert_eq!(rejected, 1);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].term, "乙术语");
    }

    #[test]
    fn parse_missing_field_drops_entry() {
        let content = r#"[{"term":"甲术语","aliases":[],"evidence":"a"}]"#;
        let (accepted, rejected) = parse_llm_candidates(content, NOW);
        assert!(accepted.is_empty());
        assert_eq!(rejected, 1);
    }

    #[test]
    fn parse_overlong_term_dropped() {
        let long = "长".repeat(MAX_TERM_CHARS + 1);
        let content =
            format!(r#"[{{"term":"{long}","aliases":[],"evidence":"q","reason":"太长"}}]"#);
        let (accepted, rejected) = parse_llm_candidates(&content, NOW);
        assert!(accepted.is_empty());
        assert_eq!(rejected, 1);
    }

    #[test]
    fn parse_accepts_mixed_terms_but_rejects_digits_and_junk() {
        let content = r#"[
            {"term":"ChatGPT","aliases":[],"evidence":"q","reason":"工具名"},
            {"term":"C++","aliases":[],"evidence":"q","reason":"语言名"},
            {"term":"2024","aliases":[],"evidence":"q","reason":"纯数字"},
            {"term":"  ","aliases":[],"evidence":"q","reason":"空白"}
        ]"#;
        let (accepted, rejected) = parse_llm_candidates(content, NOW);
        assert_eq!(rejected, 2);
        assert_eq!(accepted.len(), 2);
        assert_eq!(accepted[0].term, "ChatGPT");
        assert_eq!(accepted[1].term, "C++");
    }

    #[test]
    fn parse_duplicate_term_dropped() {
        let content = r#"[
            {"term":"词库","aliases":[],"evidence":"q","reason":"一"},
            {"term":"词库","aliases":[],"evidence":"q","reason":"二"}
        ]"#;
        let (accepted, rejected) = parse_llm_candidates(content, NOW);
        assert_eq!(accepted.len(), 1);
        assert_eq!(rejected, 1);
    }

    #[test]
    fn parse_aliases_sanitized_and_capped() {
        let content = r#"[
            {"term":"词库","aliases":["词库","2024","乙词","丙词","丁词","戊词"],"evidence":"q","reason":"e"}
        ]"#;
        let (accepted, rejected) = parse_llm_candidates(content, NOW);
        assert_eq!(rejected, 0);
        // 去掉与 term 相同的和不合格的（纯数字），再按上限截断为 3 个。
        assert_eq!(
            accepted[0].aliases,
            vec!["乙词".to_string(), "丙词".to_string(), "丁词".to_string()]
        );
    }

    #[test]
    fn parse_reason_truncated() {
        let reason = "长".repeat(MAX_EVIDENCE_CHARS + 50);
        let content =
            format!(r#"[{{"term":"词库","aliases":[],"evidence":"q","reason":"{reason}"}}]"#);
        let (accepted, _) = parse_llm_candidates(&content, NOW);
        assert_eq!(accepted[0].evidence.chars().count(), MAX_EVIDENCE_CHARS);
    }

    #[test]
    fn parse_tolerates_leading_think_block() {
        let content = format!(
            "<think>推理过程</think>[{{\"term\":\"词库\",\"aliases\":[],\"evidence\":\"q\",\"reason\":\"e\"}}]"
        );
        let (accepted, _) = parse_llm_candidates(&content, NOW);
        assert_eq!(accepted.len(), 1);
    }

    // ---- 幻觉过滤：term 必须逐字出现在送出文本中 ----

    #[test]
    fn filter_drops_terms_not_present_in_sent_text() {
        let corpus = build_sample_corpus(
            &[history_entry(1, NOW, "把 VSCode 配置同步到量子芯片平台")],
            NOW,
        );
        let content = r#"[
            {"term":"VSCode","aliases":[],"evidence":"q","reason":"工具名"},
            {"term":"幻觉词表","aliases":[],"evidence":"q","reason":"模型编的"},
            {"term":"量子芯片","aliases":[],"evidence":"q","reason":"领域术语"}
        ]"#;
        let (mut accepted, _) = parse_llm_candidates(content, NOW);
        let dropped = filter_hallucinated(&mut accepted, &corpus.text);
        assert_eq!(dropped, 1, "不在送出文本中的幻觉词被代码过滤");
        assert_eq!(accepted.len(), 2);
        assert!(accepted.iter().all(|c| c.term != "幻觉词表"));
    }

    #[test]
    fn filter_is_verbatim_not_case_folded() {
        let corpus = build_sample_corpus(&[history_entry(1, NOW, "小写 chatgpt 出现过")], NOW);
        let content = r#"[{"term":"ChatGPT","aliases":[],"evidence":"q","reason":"r"}]"#;
        let (mut accepted, _) = parse_llm_candidates(content, NOW);
        let dropped = filter_hallucinated(&mut accepted, &corpus.text);
        assert_eq!(dropped, 1, "逐字匹配：大小写不同也算幻觉");
        assert!(accepted.is_empty());
    }

    // ---- 合并（只写 Candidate，不动 custom_words）----

    #[test]
    fn merge_skips_terms_already_active_in_custom_words() {
        let mut stored = Vec::new();
        let effective = vec!["已有词".to_string(), "手工词".to_string()];
        merge_candidates(
            &mut stored,
            vec![candidate("已有词", NOW), candidate("新术语", NOW)],
            &effective,
            &HashMap::from([("新术语".to_string(), 7u64)]),
        );
        // 已激活的词不进候选；新词进候选并带上送出文本计数。
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].term, "新术语");
        assert_eq!(stored[0].count, 7);
    }

    #[test]
    fn merge_updates_existing_candidate_in_place() {
        let mut stored = vec![candidate("术语", 100)];
        let mut updated = candidate("术语", NOW);
        updated.aliases = vec!["别名".to_string()];
        merge_candidates(&mut stored, vec![updated], &[], &HashMap::new());
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].aliases, vec!["别名".to_string()]);
        assert_eq!(stored[0].created_at, 100, "created_at keeps first sight");
    }

    #[test]
    fn merge_evicts_oldest_beyond_cap() {
        let mut stored: Vec<LexiconCandidate> = (0..MAX_STORED_CANDIDATES)
            .map(|i| candidate(&format!("旧{i}"), i as i64))
            .collect();
        merge_candidates(
            &mut stored,
            vec![candidate("新术语", NOW)],
            &[],
            &HashMap::new(),
        );
        assert_eq!(stored.len(), MAX_STORED_CANDIDATES);
        assert!(stored.iter().any(|c| c.term == "新术语"));
        assert!(!stored.iter().any(|c| c.term == "旧0"), "oldest evicted");
    }

    #[test]
    fn merge_never_writes_custom_words_structurally() {
        // 模块没有任何对 settings 的写路径；用签名约束佐证：
        // merge_candidates 拿到的 effective_words 只是只读引用。
        let effective = vec!["词甲".to_string()];
        let mut stored = Vec::new();
        merge_candidates(
            &mut stored,
            vec![candidate("词乙", NOW)],
            &effective,
            &HashMap::new(),
        );
        assert_eq!(
            effective,
            vec!["词甲".to_string()],
            "read-only input untouched"
        );
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].term, "词乙");
    }

    // ---- 状态持久化形状 ----

    #[test]
    fn state_roundtrips_through_json() {
        let state = CurationState {
            last_run_at: Some(123),
            last_failure_at: Some(456),
            history_count_at_run: 789,
            last_accepted: 2,
            last_rejected: 1,
            last_sent_entries: 33,
            last_dropped_entries: 2,
            last_sent_chars: 15_998,
            last_source_entry_ids: vec![10, 9, 8],
            candidates: vec![candidate("词库", 42)],
        };
        let text = serde_json::to_string(&state).expect("serialize");
        let back: CurationState = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.last_run_at, Some(123));
        assert_eq!(back.last_failure_at, Some(456));
        assert_eq!(back.history_count_at_run, 789);
        assert_eq!(back.last_sent_entries, 33);
        assert_eq!(back.last_dropped_entries, 2);
        assert_eq!(back.last_sent_chars, 15_998);
        assert_eq!(back.last_source_entry_ids, vec![10, 9, 8]);
        assert_eq!(back.candidates.len(), 1);
        assert_eq!(back.candidates[0].term, "词库");
    }

    #[test]
    fn state_loads_old_files_without_sampling_fields() {
        // 旧版状态文件没有 last_sent_* 字段：serde(default) 兜底。
        let old = r#"{"last_run_at":1,"last_failure_at":null,"history_count_at_run":5,"last_accepted":0,"last_rejected":0,"candidates":[]}"#;
        let state: CurationState = serde_json::from_str(old).expect("old state loads");
        assert_eq!(state.last_run_at, Some(1));
        assert_eq!(state.last_sent_entries, 0);
        assert_eq!(state.last_dropped_entries, 0);
        assert_eq!(state.last_sent_chars, 0);
        assert!(state.last_source_entry_ids.is_empty());
    }

    #[test]
    fn enabled_by_default_and_disabled_by_false_values() {
        assert!(env_disables(""));
        assert!(env_disables("yes"));
        assert!(env_disables("1"));
        assert!(env_disables("随便什么值"));
        assert!(!env_disables("0"));
        assert!(!env_disables("FALSE"));
        assert!(!env_disables(" off "));
        assert!(!env_disables("No"));
        assert!(!env_disables("disabled"));
    }
}
