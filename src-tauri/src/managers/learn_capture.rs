//! 粘贴后修改学习的输入侧采集：把一次性粘贴的完成点（`actions.rs`
//! 粘贴线程的 `Ok` 分支）接到 [`crate::managers::learn_corrections`]。
//!
//! 时序（全部在独立后台线程，绝不阻塞粘贴线程与音频回调）：
//! 1. 起点身份 + 基线读取（粘贴刚落地；读取未命中粘贴文本只宽限重试
//!    一次，部分应用异步提交粘贴）；
//! 2. 粘贴完成后 **2s** 第一次读取；
//! 3. 粘贴完成后 **8s** 第二次读取；
//! 4. 仅当两次读取**相同**（用户改完、内容稳定）且从读取中截出的
//!    「粘贴区」与粘贴文本**不同**时，才交给
//!    [`learn_corrections::observe_paste_correction`] 做 diff 与落盘
//!    （阈值、blocked、去重累加全部在那里收口）。
//!
//! 读取复用 [`crate::managers::llm_context`] 的 UIA（Windows）/ AX
//! （macOS）光标前文本读取（同一条平台链，窗口按粘贴文本长度放开）；
//! 身份校验复用 [`crate::managers::paste_target::capture`]：两次读取
//! 之前都必须复核前台仍是粘贴时的同一目标（拿不到按已变化处理，
//! fail closed）。Linux/Wayland 等读不到的环境恒为 `None`，整条链路
//! 自然退化为不学。
//!
//! 静默策略（与 `llm_context` 同口径）：任何一步拿不到——读取失败、
//! 身份变化、基线不含粘贴文本、读取区前缀漂移、粘贴区未被改动——
//! 都直接放弃本次学习，不 warn、不打日志、不影响粘贴结果；学习是
//! 尽力而为。
//!
//! 增量上屏（一贴多段）不接入：多段粘贴的词对归因不可靠，调用方以
//! `whole_paste == false` 跳过（见 `actions.rs` 的窥视判断）。

use crate::managers::learn_corrections::{observe_paste_correction, LearnedCandidate};
use crate::managers::{llm_context, paste_target};
use std::path::Path;
use std::time::{Duration, Instant};
use tauri::Manager;

/// 第一次读取距粘贴完成的延迟。
const FIRST_READ_DELAY: Duration = Duration::from_secs(2);
/// 第二次读取距粘贴完成的延迟。
const SECOND_READ_DELAY: Duration = Duration::from_secs(8);
/// 基线读取未命中粘贴文本时的宽限重试延迟（部分应用异步提交粘贴）。
const BASELINE_SETTLE: Duration = Duration::from_millis(250);
/// 读取窗口在粘贴文本之外额外多读的前缀锚点字符数：粘贴前的已有文本
/// 至少留出这些字符做前缀锚，才能把「粘贴区」从读取窗口里截出来。
const PREFIX_ANCHOR_CHARS: usize = 24;
/// 读取窗口上限（超过的整段粘贴不学；绝不读整篇文档）。
const MAX_READ_CHARS: usize = 1024;

/// 可注入采集环境：生产为平台实现（[`production_env`]），测试注入假
/// 实现以覆盖时序与失败路径。
pub(crate) struct CaptureEnv {
    /// 光标前 ≤`n` 字符的原始文本读取（平台隔离；拿不到为 `None`）。
    pub(crate) read: Box<dyn Fn(usize) -> Option<String> + Send>,
    /// 当前前台粘贴目标的不透明身份 key（`paste_target` 语义）。
    pub(crate) identity: Box<dyn Fn() -> Option<String> + Send>,
    /// 等待（生产为 `thread::sleep`；测试记录时序不真睡）。
    pub(crate) sleep: Box<dyn Fn(Duration) + Send>,
}

fn production_env() -> CaptureEnv {
    CaptureEnv {
        read: Box::new(llm_context::read_text_before_cursor_raw),
        identity: Box::new(|| paste_target::capture().map(|t| t.key().to_string())),
        sleep: Box::new(std::thread::sleep),
    }
}

/// 粘贴成功后由粘贴线程调用：`whole_paste == false`（增量上屏会话）、
/// 设置 `learn_from_corrections_enabled` 关闭或文本不可学时直接返回
/// （不起线程、不触碰任何平台 API）；否则起后台线程采集学习。线程
/// 起不来也静默放弃。
pub fn spawn_after_paste(app: &tauri::AppHandle, inserted: &str, whole_paste: bool) {
    let enabled = crate::settings::get_settings(app).learn_from_corrections_enabled;
    if !capture_allowed(enabled, whole_paste, inserted) {
        return;
    }
    let app = app.clone();
    let inserted = inserted.to_string();
    let _ = std::thread::Builder::new()
        .name("learn-corrections".into())
        .spawn(move || {
            let Ok(data_dir) = app.path().app_data_dir() else {
                return;
            };
            let _ = learn_from_paste(
                enabled,
                whole_paste,
                &data_dir,
                &inserted,
                &production_env(),
            );
        });
}

/// 学习主流程（后台线程执行；测试同步执行）：`whole_paste == false`、
/// 开关关闭或文本不可学时一步不执行；采集稳定且粘贴区被修改时，把
/// `(粘贴文本, 修改后文本)` 交给 [`observe_paste_correction`]。
/// 返回学到/更新的候选；任何一步拿不到返回 `None`。
pub(crate) fn learn_from_paste(
    enabled: bool,
    whole_paste: bool,
    data_dir: &Path,
    pasted: &str,
    env: &CaptureEnv,
) -> Option<LearnedCandidate> {
    if !capture_allowed(enabled, whole_paste, pasted) {
        return None;
    }
    let later = observe_stable(pasted, env)?;
    observe_paste_correction(data_dir, pasted, &later)
}

/// 采集门槛的单一收口（粘贴线程的 early return 与学习主流程共用）：
/// 开关开、一次性整段粘贴、文本可学，三者同时成立才起线程/读输入框。
fn capture_allowed(enabled: bool, whole_paste: bool, pasted: &str) -> bool {
    enabled && whole_paste && viable(pasted)
}

/// 文本可学：非空白，且读取窗口（粘贴文本 + 前缀锚）不超上限。
fn viable(pasted: &str) -> bool {
    !pasted.trim().is_empty() && pasted.chars().count() + PREFIX_ANCHOR_CHARS <= MAX_READ_CHARS
}

/// 采集并判定：返回「粘贴区被修改后的稳定文本」。任一步拿不到 →
/// `None`（静默放弃，不告警）。
fn observe_stable(pasted: &str, env: &CaptureEnv) -> Option<String> {
    let window = pasted.chars().count() + PREFIX_ANCHOR_CHARS;
    let started = Instant::now();
    // 起点身份拿不到（平台不支持/查询失败）即放弃：不读无法核实身份
    // 的输入框。
    let target = (env.identity)()?;
    let baseline = read_baseline(pasted, window, env)?;
    wait_until(&started, FIRST_READ_DELAY, env);
    let first = read_same_target(&target, window, env)?;
    wait_until(&started, SECOND_READ_DELAY, env);
    let second = read_same_target(&target, window, env)?;
    if first != second {
        return None; // 两次读取不同：内容还在变，不学
    }
    // 从读取窗口截出粘贴区：基线以粘贴文本结尾，其头部即粘贴前已有
    // 的前缀；两次读取须仍以该前缀开头，否则窗口漂移/改动越界（如删
    // 了粘贴区之前的字符），放弃。
    let prefix_len = baseline.len() - pasted.len();
    if !first.starts_with(&baseline[..prefix_len]) {
        return None;
    }
    let later = first[prefix_len..].to_string();
    (later != pasted).then_some(later)
}

/// 基线读取：粘贴刚落地时读一次，窗口必须以粘贴文本结尾（确认读到的
/// 就是目标输入框且粘贴已提交）；失败或未命中都宽限重试一次（部分
/// 应用异步提交粘贴，期间读取可能拿到旧内容或直接失败），仍不命中
/// 放弃。
fn read_baseline(pasted: &str, window: usize, env: &CaptureEnv) -> Option<String> {
    let mut baseline = (env.read)(window);
    if !baseline.as_deref().is_some_and(|b| b.ends_with(pasted)) {
        (env.sleep)(BASELINE_SETTLE);
        baseline = (env.read)(window);
    }
    baseline.filter(|b| b.ends_with(pasted))
}

/// 同一目标的读取：先复核前台身份仍是粘贴目标，再读。身份拿不到或
/// 变化都放弃（fail closed：宁可少学，不读别的窗口）。
fn read_same_target(target: &str, window: usize, env: &CaptureEnv) -> Option<String> {
    if (env.identity)()? != target {
        return None;
    }
    (env.read)(window)
}

/// 睡到距 `started` 满 `target`（已过则立即返回）：两次读取的延迟都从
/// 粘贴完成时刻起算，宽限重试不会推迟整体时序。
fn wait_until(started: &Instant, target: Duration, env: &CaptureEnv) {
    let elapsed = started.elapsed();
    if elapsed < target {
        (env.sleep)(target - elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::managers::learn_corrections::learn_store_path;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    fn test_dir() -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("voo-learn-capture-{}-{}", std::process::id(), seq));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 脚本化环境：`read`/`identity` 按脚本回放（read 耗尽后 `None`，
    /// identity 耗尽后停留最后一档），`sleep` 只记录时长不真睡；read
    /// 同时模拟平台读取的窗口截断（只保留最后 `window` 个字符）。
    struct Script {
        reads: Mutex<Vec<Option<String>>>,
        identities: Mutex<Vec<Option<String>>>,
        sleeps: Mutex<Vec<Duration>>,
        read_calls: Mutex<usize>,
        identity_calls: Mutex<usize>,
    }

    impl Script {
        fn new(reads: Vec<Option<&str>>, identities: Vec<Option<&str>>) -> Arc<Self> {
            Arc::new(Self {
                reads: Mutex::new(reads.into_iter().map(|r| r.map(str::to_string)).collect()),
                identities: Mutex::new(
                    identities
                        .into_iter()
                        .map(|i| i.map(str::to_string))
                        .collect(),
                ),
                sleeps: Mutex::new(Vec::new()),
                read_calls: Mutex::new(0),
                identity_calls: Mutex::new(0),
            })
        }

        fn env(self: &Arc<Self>) -> CaptureEnv {
            let for_read = Arc::clone(self);
            let for_identity = Arc::clone(self);
            let for_sleep = Arc::clone(self);
            CaptureEnv {
                read: Box::new(move |window| {
                    let mut calls = for_read.read_calls.lock().unwrap();
                    *calls += 1;
                    let text = for_read
                        .reads
                        .lock()
                        .unwrap()
                        .get(*calls - 1)
                        .cloned()
                        .flatten()?;
                    let total = text.chars().count();
                    if total <= window {
                        Some(text)
                    } else {
                        Some(text.chars().skip(total - window).collect())
                    }
                }),
                identity: Box::new(move || {
                    let mut calls = for_identity.identity_calls.lock().unwrap();
                    *calls += 1;
                    let ids = for_identity.identities.lock().unwrap();
                    let idx = (*calls - 1).min(ids.len().saturating_sub(1));
                    ids.get(idx).cloned().flatten()
                }),
                sleep: Box::new(move |d| {
                    for_sleep.sleeps.lock().unwrap().push(d);
                }),
            }
        }

        fn read_calls(&self) -> usize {
            *self.read_calls.lock().unwrap()
        }

        fn identity_calls(&self) -> usize {
            *self.identity_calls.lock().unwrap()
        }

        fn sleeps(&self) -> Vec<Duration> {
            self.sleeps.lock().unwrap().clone()
        }
    }

    /// 常规正例：粘贴前输入框已有前缀，用户把「晓」改成「小」，2s 与
    /// 8s 两次读取相同。
    #[test]
    fn learns_stable_correction() {
        let dir = test_dir();
        let script = Script::new(
            vec![
                Some("任务：帮我查一下王晓明"),
                Some("任务：帮我查一下王小明"),
                Some("任务：帮我查一下王小明"),
            ],
            vec![Some("pid:1"), Some("pid:1"), Some("pid:1")],
        );
        let learned = learn_from_paste(true, true, &dir, "帮我查一下王晓明", &script.env())
            .expect("稳定且被修改：应学到");
        assert_eq!(
            (learned.wrong.as_str(), learned.corrected.as_str()),
            ("晓", "小")
        );
        assert_eq!(learned.count, 1);
        // 读取时序：基线立即命中（无宽限重试）→ 睡到 2s → 睡到 8s
        // （两次延迟都从粘贴完成时刻起算；假 sleep 不耗时，误差仅为
        // 执行耗时）。
        let sleeps = script.sleeps();
        assert_eq!(sleeps.len(), 2);
        assert!(sleeps[0] > Duration::from_secs(2) - Duration::from_millis(50));
        assert!(sleeps[0] <= Duration::from_secs(2));
        assert!(sleeps[1] > Duration::from_secs(8) - Duration::from_millis(50));
        assert!(sleeps[1] <= Duration::from_secs(8));
        assert_eq!(script.read_calls(), 3);
        // 落盘到 learned store（不写 custom_words/设置）。
        let store = std::fs::read_to_string(learn_store_path(&dir)).unwrap();
        assert!(store.contains("\"wrong\": \"晓\""));
        assert!(store.contains("\"corrected\": \"小\""));
    }

    /// 空输入框（无前缀）同样可学：窗口就是粘贴区本身。
    #[test]
    fn learns_with_empty_prefix() {
        let dir = test_dir();
        let script = Script::new(
            vec![
                Some("帮我查一下王晓明"),
                Some("帮我查一下王小明"),
                Some("帮我查一下王小明"),
            ],
            vec![Some("pid:1")],
        );
        let learned =
            learn_from_paste(true, true, &dir, "帮我查一下王晓明", &script.env()).unwrap();
        assert_eq!(
            (learned.wrong.as_str(), learned.corrected.as_str()),
            ("晓", "小")
        );
    }

    /// 长前缀超出前缀锚：读取窗口截掉前缀头部，锚定仍成立（粘贴区截取
    /// 只依赖窗口尾部的粘贴文本）。
    #[test]
    fn long_prefix_anchors_on_window_tail() {
        let dir = test_dir();
        let pasted = "开完会发报告";
        let prefix: String = "前".repeat(40);
        let field_after_edit = format!("{prefix}开完会发报表");
        let script = Script::new(
            vec![
                Some(format!("{prefix}{pasted}").as_str()),
                Some(field_after_edit.as_str()),
                Some(field_after_edit.as_str()),
            ],
            vec![Some("pid:1")],
        );
        let learned = learn_from_paste(true, true, &dir, pasted, &script.env()).unwrap();
        assert_eq!(
            (learned.wrong.as_str(), learned.corrected.as_str()),
            ("告", "表")
        );
    }

    /// 两次读取不同（还在改）：不学、不落盘。
    #[test]
    fn unstable_reads_skip_learning() {
        let dir = test_dir();
        let script = Script::new(
            vec![
                Some("任务：帮我查一下王晓明"),
                Some("任务：帮我查一下王小明"),
                Some("任务：帮我查一下王大明"),
            ],
            vec![Some("pid:1")],
        );
        assert!(learn_from_paste(true, true, &dir, "帮我查一下王晓明", &script.env()).is_none());
        assert!(!learn_store_path(&dir).exists(), "不稳定不得落盘");
        assert_eq!(script.read_calls(), 3);
    }

    /// 开关关闭：一次读取/身份探测都不做，直接返回。
    #[test]
    fn disabled_switch_never_reads() {
        let dir = test_dir();
        let script = Script::new(vec![Some("任务：帮我查一下王晓明")], vec![Some("pid:1")]);
        assert!(learn_from_paste(false, true, &dir, "帮我查一下王晓明", &script.env()).is_none());
        assert_eq!(script.read_calls(), 0, "开关关闭不得读取");
        assert_eq!(script.identity_calls(), 0, "开关关闭不得探测身份");
        assert!(!learn_store_path(&dir).exists());
    }

    /// 增量上屏会话（`whole_paste == false`）：与开关关闭同口径，一次
    /// 读取/身份探测都不做，直接返回、不落盘。
    #[test]
    fn incremental_paste_session_skips() {
        let dir = test_dir();
        let script = Script::new(vec![Some("任务：帮我查一下王小明")], vec![Some("pid:1")]);
        assert!(learn_from_paste(true, false, &dir, "帮我查一下王晓明", &script.env()).is_none());
        assert_eq!(script.read_calls(), 0, "增量上屏不得读取");
        assert_eq!(script.identity_calls(), 0, "增量上屏不得探测身份");
        assert!(!learn_store_path(&dir).exists());
    }

    /// 读取失败（平台读不到/读取出错）：静默返回 None，不 panic。
    #[test]
    fn read_failure_is_silent_and_safe() {
        let dir = test_dir();
        // 基线就失败：宽限重试一次后放弃。
        let script = Script::new(vec![None, None], vec![Some("pid:1")]);
        assert!(learn_from_paste(true, true, &dir, "帮我查一下王晓明", &script.env()).is_none());
        assert_eq!(script.read_calls(), 2);
        assert!(script.sleeps().contains(&BASELINE_SETTLE));
        // 第一次观察读取失败：同样静默放弃。
        let script2 = Script::new(vec![Some("帮我查一下王晓明"), None], vec![Some("pid:1")]);
        assert!(learn_from_paste(true, true, &dir, "帮我查一下王晓明", &script2.env()).is_none());
        // 第二次观察读取失败：同样静默放弃。
        let script3 = Script::new(
            vec![Some("帮我查一下王晓明"), Some("帮我查一下王小明"), None],
            vec![Some("pid:1")],
        );
        assert!(learn_from_paste(true, true, &dir, "帮我查一下王晓明", &script3.env()).is_none());
        assert!(!learn_store_path(&dir).exists());
    }

    /// 基线不含粘贴文本（读错窗口/粘贴被应用改写）：重试一次仍不命中
    /// 即放弃，不学。
    #[test]
    fn baseline_without_pasted_text_skips() {
        let dir = test_dir();
        let script = Script::new(
            vec![Some("完全无关的窗口内容"), Some("还是无关内容")],
            vec![Some("pid:1")],
        );
        assert!(learn_from_paste(true, true, &dir, "帮我查一下王晓明", &script.env()).is_none());
        assert_eq!(script.read_calls(), 2, "基线未命中只宽限重试一次");
    }

    /// 身份拿不到 / 身份变化：不读，不学。
    #[test]
    fn identity_missing_or_changed_skips() {
        let dir = test_dir();
        // 起点就拿不到身份。
        let no_identity = Script::new(vec![Some("帮我查一下王晓明")], vec![None]);
        assert!(
            learn_from_paste(true, true, &dir, "帮我查一下王晓明", &no_identity.env()).is_none()
        );
        assert_eq!(no_identity.read_calls(), 0, "没有起点身份不得读取");
        // 2s 复核时身份已变（切走/读不回）：读到第一次为止。
        let changed = Script::new(
            vec![
                Some("任务：帮我查一下王晓明"),
                Some("任务：帮我查一下王小明"),
            ],
            vec![Some("pid:1"), Some("pid:2")],
        );
        assert!(learn_from_paste(true, true, &dir, "帮我查一下王晓明", &changed.env()).is_none());
        assert_eq!(changed.read_calls(), 1);
        assert!(!learn_store_path(&dir).exists());
    }

    /// 粘贴区没被改（读取与基线一致）：与粘贴文本相同，不学。
    #[test]
    fn unchanged_pasted_text_skips() {
        let dir = test_dir();
        let script = Script::new(
            vec![
                Some("任务：帮我查一下王晓明"),
                Some("任务：帮我查一下王晓明"),
                Some("任务：帮我查一下王晓明"),
            ],
            vec![Some("pid:1")],
        );
        assert!(learn_from_paste(true, true, &dir, "帮我查一下王晓明", &script.env()).is_none());
        assert!(!learn_store_path(&dir).exists());
    }

    /// 前缀漂移（用户删了粘贴区之前的字符，窗口内容左移）：锚定失败，
    /// 不学——避免把粘贴前文本误认成修改。
    #[test]
    fn prefix_drift_skips() {
        let dir = test_dir();
        let script = Script::new(
            vec![
                Some("提醒：开完会发报告"),
                Some("开完会发报告"),
                Some("开完会发报告"),
            ],
            vec![Some("pid:1")],
        );
        assert!(learn_from_paste(true, true, &dir, "开完会发报告", &script.env()).is_none());
        assert!(!learn_store_path(&dir).exists());
    }

    /// 超长粘贴（窗口超上限）不可学：不读取。
    #[test]
    fn oversized_paste_is_not_viable() {
        let dir = test_dir();
        let script = Script::new(vec![], vec![]);
        let pasted: String = "甲".repeat(MAX_READ_CHARS);
        assert!(learn_from_paste(true, true, &dir, &pasted, &script.env()).is_none());
        assert_eq!(script.read_calls(), 0);
    }

    /// 学成后只写 learned store：目录里没有 settings store，JSON 顶层
    /// 只有 candidates/blocked，不出现 custom_words。
    #[test]
    fn never_writes_custom_words() {
        let dir = test_dir();
        let script = Script::new(
            vec![
                Some("任务：帮我查一下王晓明"),
                Some("任务：帮我查一下王小明"),
                Some("任务：帮我查一下王小明"),
            ],
            vec![Some("pid:1")],
        );
        learn_from_paste(true, true, &dir, "帮我查一下王晓明", &script.env()).unwrap();
        let entries: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(entries, vec!["learned_corrections.json".to_string()]);
        let text = std::fs::read_to_string(learn_store_path(&dir)).unwrap();
        assert!(!text.contains("custom_words"));
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, vec!["blocked", "candidates"]);
    }
}
