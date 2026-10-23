//! 流式模型的增量上屏：把 committed 前缀的新增部分边说边贴进目标
//! 应用，finalize 只补最后差额。纯逻辑（[`Committer`]/接缝拼接/最终
//! 扣减）与粘贴线程（[`StreamCommitter`]）分离，前者全部可单测；后
//! 者复用 `clipboard::paste_delta`（既有 paste/enigo 通道），不新写
//! 键盘注入。
//!
//! 安全策略（任何异常都退回"本次会话不再增量"）：
//! - 新 committed 必须以已上屏前缀为前缀，否则降级并 warn，绝不重贴；
//! - tentative 永不提交（只送 committed 转换结果）；
//! - 粘贴失败（焦点丢失/注入失败等）→ 降级；
//! - 取消（句柄 drop）→ 粘贴线程立刻放弃队列中未提交的增量；
//! - LLM 润色可能开启或粘贴方式为 None 时整场禁用，退回一次性粘贴。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use log::{debug, warn};
use tauri::AppHandle;

use crate::managers::paste_target::{CaptureFn, TargetGuard};
use crate::settings::{AppSettings, PasteMethod};

/// Commands for the paste thread. Targets carry the whole converted
/// committed text (not the delta) so the thread — the only writer of the
/// pasted bookkeeping — can diff safely whatever the callback timing.
enum CommitCmd {
    Target(String),
    /// Drain every queued target, paste what is due, reply with the final
    /// pasted prefix, and exit the thread.
    Flush(mpsc::Sender<String>),
}

/// One live stream's incremental committer. `Clone` copies are handed to the
/// engine's progress callback; the original lives on the stream session.
#[derive(Clone)]
pub(super) struct StreamCommitter {
    tx: Option<mpsc::Sender<CommitCmd>>,
    /// Set when the session is dropped (cancelled): the paste thread skips
    /// everything still queued and exits without pasting it.
    abandoned: Arc<AtomicBool>,
    enabled: bool,
}

impl StreamCommitter {
    /// Begin committing for one stream when the settings allow it, pasting
    /// through the existing clipboard channel. Otherwise returns a no-op
    /// handle and the session keeps today's one-paste-at-the-end behavior.
    pub(super) fn start(app_handle: AppHandle, settings: &AppSettings) -> Self {
        match eligibility(settings) {
            Ok(()) => Self::spawn(move |delta| crate::clipboard::paste_delta(delta, &app_handle)),
            Err(reason) => {
                debug!("Incremental commit inactive: {}", reason);
                Self::disabled()
            }
        }
    }

    /// Test/embedding entry point with an injected paste action. The target
    /// guard probes the real platform identity.
    fn spawn<F>(paste: F) -> Self
    where
        F: FnMut(&str) -> Result<(), String> + Send + 'static,
    {
        Self::spawn_gated(paste, Box::new(crate::managers::paste_target::capture))
    }

    /// Test/embedding entry point with an injected paste action **and**
    /// identity probe: the guard binds the session to the window that had
    /// focus when the thread starts and blocks every further increment once
    /// focus leaves it (blocked text is kept for the final paste).
    fn spawn_gated<F>(mut paste: F, capture: CaptureFn) -> Self
    where
        F: FnMut(&str) -> Result<(), String> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<CommitCmd>();
        let abandoned = Arc::new(AtomicBool::new(false));
        let abandoned_for_thread = Arc::clone(&abandoned);
        let spawned = std::thread::Builder::new()
            .name("stream-commit-paste".to_string())
            .spawn(move || {
                let mut committer = Committer::new();
                // Session start: bind every paste to the window that has
                // focus now (best-effort; silently unchecked when the
                // platform cannot provide an identity).
                let mut guard = TargetGuard::with_capture(capture);
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        CommitCmd::Target(target) => {
                            if abandoned_for_thread.load(Ordering::Acquire) {
                                break;
                            }
                            process(&mut committer, &mut guard, &mut paste, &target);
                        }
                        CommitCmd::Flush(reply) => {
                            // Drain first: every delta already committed by
                            // the model must be pasted (and bookkept) before
                            // the final paste is diffed against them.
                            while let Ok(CommitCmd::Target(target)) = rx.try_recv() {
                                if abandoned_for_thread.load(Ordering::Acquire) {
                                    break;
                                }
                                process(&mut committer, &mut guard, &mut paste, &target);
                            }
                            let _ = reply.send(committer.pasted.clone());
                            return;
                        }
                    }
                }
                // The channel closed (session dropped): exit without pasting
                // anything still queued.
            });
        match spawned {
            Ok(_join) => Self {
                tx: Some(tx),
                abandoned,
                enabled: true,
            },
            Err(e) => {
                warn!("Incremental commit could not start its paste thread: {}", e);
                Self::disabled()
            }
        }
    }

    fn disabled() -> Self {
        Self {
            tx: None,
            abandoned: Arc::new(AtomicBool::new(false)),
            enabled: false,
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Offer the latest converted committed text. Cheap and non-blocking:
    /// called on the engine's progress thread, never the audio callback.
    pub(super) fn submit(&self, target: &str) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(CommitCmd::Target(target.to_string()));
        }
    }

    /// Wait until every queued delta has been pasted, and return what was
    /// actually pasted. Empty when nothing landed (feature off/failed).
    pub(super) fn flush(&mut self) -> String {
        let Some(tx) = self.tx.take() else {
            return String::new();
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        if tx.send(CommitCmd::Flush(reply_tx)).is_err() {
            warn!("Incremental commit paste thread exited before finalize");
            return String::new();
        }
        match reply_rx.recv() {
            Ok(pasted) => pasted,
            Err(_) => {
                warn!("Incremental commit paste thread did not answer finalize");
                String::new()
            }
        }
    }
}

impl Drop for StreamCommitter {
    fn drop(&mut self) {
        // Session over without finalize (cancel): abandon the queue. The
        // sender halves drop with `self`, closing the thread's channel.
        self.abandoned.store(true, Ordering::Release);
    }
}

/// Paste one target's due delta, or degrade the committer. The target guard
/// runs only when a paste is actually due; a guard block degrades quietly
/// (the blocked text stays un-pasted in the bookkeeping, so the final paste
/// outputs the remainder) and never pastes into the new focus.
fn process(
    committer: &mut Committer,
    guard: &mut TargetGuard,
    paste: &mut dyn FnMut(&str) -> Result<(), String>,
    target: &str,
) {
    match committer.plan(target) {
        Plan::Paste(delta) => {
            if !guard.allow_paste() {
                committer.degrade_quiet(&format!(
                    "target guard blocked further incremental pastes: {}",
                    guard.block_reason().unwrap_or("unknown")
                ));
                return;
            }
            match paste(&delta) {
                Ok(()) => committer.confirm(&delta),
                Err(e) => committer.degrade(&format!("pasting a committed segment failed: {}", e)),
            }
        }
        Plan::Skip => {}
        Plan::Degrade(reason) => committer.degrade(reason),
    }
}

/// What to do with one new committed target.
#[derive(Debug, PartialEq, Eq)]
enum Plan {
    /// Paste this segment, then record it as pasted.
    Paste(String),
    /// Nothing new worth pasting.
    Skip,
    /// The target does not extend what was pasted; stop committing.
    Degrade(&'static str),
}

/// Pure bookkeeping of what has actually been pasted, in the converted
/// committed text's space. All decisions (prefix check, seams, finalize
/// remainder) hang off this one string, so they are unit-testable.
struct Committer {
    pasted: String,
    degraded: bool,
}

impl Committer {
    fn new() -> Self {
        Self {
            pasted: String::new(),
            degraded: false,
        }
    }

    /// The delta to paste for `target`, the latest converted committed text.
    /// `target` must extend [`Self::pasted`]; anything else is a rewrite of
    /// already-pasted text, which incremental commit cannot express — the
    /// caller degrades instead of appending garbage.
    fn plan(&self, target: &str) -> Plan {
        if self.degraded {
            return Plan::Skip;
        }
        let delta = match target.strip_prefix(&self.pasted) {
            Some(delta) => delta,
            None => return Plan::Degrade("committed text does not extend the pasted prefix"),
        };
        // The final text is trimmed, so never commit leading whitespace:
        // the prefix relationship would not survive post-processing.
        let delta = if self.pasted.is_empty() {
            delta.trim_start()
        } else {
            delta
        };
        // Whitespace-only growth buys nothing and pastes badly; the next
        // real delta still carries the whitespace.
        if delta.trim().is_empty() {
            return Plan::Skip;
        }
        Plan::Paste(delta.to_string())
    }

    /// Record a segment that was just pasted successfully.
    fn confirm(&mut self, delta: &str) {
        self.pasted = join_segments(&self.pasted, delta);
    }

    /// Stop committing for the rest of the session. Whatever was already
    /// pasted stays (it cannot be un-pasted); nothing more is appended.
    fn degrade(&mut self, reason: &str) {
        if !self.degraded {
            warn!("Incremental commit stopped for this recording: {}", reason);
            self.degraded = true;
        }
    }

    /// Stop committing like [`Self::degrade`], but log at debug level only:
    /// the target-guard block must not warn (policy) — the event stays
    /// visible at debug and the text is kept for the final paste.
    fn degrade_quiet(&mut self, reason: &str) {
        if !self.degraded {
            debug!("Incremental commit stopped for this recording: {}", reason);
            self.degraded = true;
        }
    }
}

/// Join two committed segments at their seam, by language: 中文（CJK）任一侧
/// 都不插空格；两侧都是拉丁字母时补一个空格；标点/已有空白直接相接。
fn join_segments(prev: &str, next: &str) -> String {
    let mut out = String::with_capacity(prev.len() + next.len() + 1);
    out.push_str(prev);
    if needs_space(prev, next) {
        out.push(' ');
    }
    out.push_str(next);
    out
}

/// Whether a synthetic space belongs between `prev`'s last and `next`'s
/// first character.
fn needs_space(prev: &str, next: &str) -> bool {
    let (Some(last), Some(first)) = (prev.chars().next_back(), next.chars().next()) else {
        return false;
    };
    if last.is_whitespace() || first.is_whitespace() || is_glue_punct(first) {
        return false;
    }
    if is_cjk(last) || is_cjk(first) {
        return false;
    }
    last.is_ascii_alphabetic() && first.is_ascii_alphabetic()
}

/// CJK ideographs, kana, hangul, CJK punctuation and fullwidth forms —
/// none of these take spaces at a seam.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'      // CJK 统一汉字
        | '\u{3400}'..='\u{4DBF}'    // 扩展 A
        | '\u{F900}'..='\u{FAFF}'    // 兼容汉字
        | '\u{20000}'..='\u{2A6DF}'  // 扩展 B
        | '\u{3000}'..='\u{303F}'    // CJK 标点（、。「」）
        | '\u{3040}'..='\u{30FF}'    // 假名
        | '\u{AC00}'..='\u{D7AF}'    // 谚文
        | '\u{FF00}'..='\u{FFEF}'    // 全角形式（！？）
    )
}

/// Punctuation that never takes a leading space after a previous segment.
fn is_glue_punct(c: char) -> bool {
    matches!(
        c,
        ',' | '.'
            | '!'
            | '?'
            | ';'
            | ':'
            | ')'
            | ']'
            | '}'
            | '>'
            | '%'
            | '"'
            | '\''
            | '…'
            | '—'
            | '、'
            | '。'
            | '，'
            | '！'
            | '？'
            | '；'
            | '：'
            | '）'
            | '】'
            | '》'
            | '」'
            | '』'
            | '％'
    )
}

/// 已增量上屏、等待最终粘贴扣减的前缀。flush 时写入；被下一次最终
/// `paste()` 消费（只消费一次），或被下一场转写开始时清除。
static FINAL_PASTE_PREFIX: Mutex<Option<String>> = Mutex::new(None);

fn lock_prefix() -> std::sync::MutexGuard<'static, Option<String>> {
    FINAL_PASTE_PREFIX.lock().unwrap_or_else(|e| e.into_inner())
}

/// Record what a finished stream already pasted, so the final paste only
/// outputs the remainder.
pub(crate) fn arm_final_paste_prefix(pasted: String) {
    *lock_prefix() = Some(pasted);
}

/// Drop any armed remainder (stale from a cancelled or unpasted session).
/// Called when a new transcription begins.
pub(crate) fn clear_final_paste_prefix() {
    *lock_prefix() = None;
}

/// 是否有增量前缀等待最终粘贴扣减（只窥视，不消费）。粘贴后修改学习
/// 用它在最终 `paste()` 之前判断本次粘贴是否整段输出：增量上屏的会话
/// （一贴多段）不学，避免把多段粘贴归因成一次修改。
pub(crate) fn has_armed_final_paste_prefix() -> bool {
    matches!(lock_prefix().as_deref(), Some(prefix) if !prefix.is_empty())
}

/// The text the final paste should output: the armed diff of `text`, or
/// `text` itself when no incremental prefix is armed. Consumes the arm.
///
/// Never returns a longer string than `text`: a mismatch (post-processing
/// rewrote already-pasted text) keeps what was pasted and pastes nothing
/// more — 重贴整段只会制造重复。
pub(crate) fn adjust_final_paste(text: &str) -> String {
    let Some(pasted) = lock_prefix().take() else {
        return text.to_string();
    };
    if pasted.is_empty() {
        return text.to_string();
    }
    if let Some(rest) = text.strip_prefix(&pasted) {
        return rest.to_string();
    }
    // Post-processing trimmed a trailing whitespace we had committed.
    if text.trim_end() == pasted.trim_end() {
        return String::new();
    }
    warn!(
        "Incremental commit: final text does not extend the pasted prefix; \
         keeping the streamed text and pasting no remainder"
    );
    String::new()
}

/// Whether incremental commit may run for a recording with these settings.
fn eligibility(settings: &AppSettings) -> Result<(), &'static str> {
    if !settings.incremental_commit {
        return Err("the incremental commit setting is off");
    }
    if settings.paste_method == PasteMethod::None {
        return Err("paste is disabled (paste method set to none)");
    }
    if polish_possible(settings) {
        return Err("LLM post-processing is configured and would rewrite the text");
    }
    Ok(())
}

/// Mirrors `actions.rs`'s post-processing preconditions: when an LLM could
/// polish the final text, incremental commit stays off so the polished
/// result keeps today's one-paste behavior.
fn polish_possible(settings: &AppSettings) -> bool {
    if !settings.post_process_enabled {
        return false;
    }
    let Some(provider) = settings.active_post_process_provider() else {
        return false;
    };
    let model_ready = settings
        .post_process_models
        .get(&provider.id)
        .is_some_and(|model| !model.trim().is_empty());
    let prompt_ready = settings
        .post_process_selected_prompt_id
        .as_ref()
        .and_then(|id| settings.post_process_prompts.iter().find(|p| &p.id == id))
        .is_some_and(|prompt| !prompt.prompt.trim().is_empty());
    model_ready && prompt_ready
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// The armed prefix is process-global; serialize the tests that touch it.
    static ARM_LOCK: StdMutex<()> = StdMutex::new(());

    fn confirm_seq(committer: &mut Committer, targets: &[&str]) {
        for target in targets {
            if let Plan::Paste(delta) = committer.plan(target) {
                committer.confirm(&delta);
            }
        }
    }

    #[test]
    fn prefix_increments_paste_each_delta() {
        let mut c = Committer::new();
        assert_eq!(c.plan("你好"), Plan::Paste("你好".to_string()));
        c.confirm("你好");
        assert_eq!(c.plan("你好，世界"), Plan::Paste("，世界".to_string()));
        c.confirm("，世界");
        assert_eq!(c.plan("你好，世界。"), Plan::Paste("。".to_string()));
        c.confirm("。");
        // No growth: skip, never a re-paste.
        assert_eq!(c.plan("你好，世界。"), Plan::Skip);
        assert_eq!(c.pasted, "你好，世界。");
    }

    #[test]
    fn chinese_joins_without_spaces() {
        assert_eq!(join_segments("你好", "世界"), "你好世界");
        assert_eq!(join_segments("你好", "，世界"), "你好，世界");
        assert_eq!(join_segments("", "世界"), "世界");
        assert_eq!(join_segments("句尾", "。"), "句尾。");
        // 全角标点两侧也不插空格
        assert_eq!(join_segments("测试", "！继续"), "测试！继续");
    }

    #[test]
    fn latin_joins_with_space_and_keeps_own_whitespace() {
        assert_eq!(join_segments("hello", "world"), "hello world");
        // 模型自己带的空格不叠加
        assert_eq!(join_segments("hello", " world"), "hello world");
        assert_eq!(join_segments("hello ", "world"), "hello world");
        // 标点直接相接
        assert_eq!(join_segments("Hello", "world!"), "Hello world!");
        // 中英混排：CJK 一侧即不插空格
        assert_eq!(join_segments("你好", "world"), "你好world");
        assert_eq!(join_segments("hello", "世界"), "hello世界");
        // 数字不强行加空格
        assert_eq!(join_segments("voo", "2026"), "voo2026");
    }

    #[test]
    fn non_prefix_target_degrades_and_never_repastes() {
        let mut c = Committer::new();
        confirm_seq(&mut c, &["你好", "你好世界"]);
        assert_eq!(c.pasted, "你好世界");
        // 回退重塑：不是已上屏文本的延伸
        assert_eq!(
            c.plan("再见世界"),
            Plan::Degrade("committed text does not extend the pasted prefix")
        );
        c.degrade("committed text does not extend the pasted prefix");
        // 降级后一切目标都不再粘贴
        assert_eq!(c.plan("再见世界更多"), Plan::Skip);
        assert_eq!(c.plan("你好世界！"), Plan::Skip);
        assert_eq!(c.pasted, "你好世界");
    }

    #[test]
    fn whitespace_only_and_leading_whitespace_are_skipped() {
        let mut c = Committer::new();
        // 首段前导空白不提交（最终文本会 trim，保住前缀关系）
        assert_eq!(c.plan("  你好"), Plan::Paste("你好".to_string()));
        c.confirm("你好");
        // 纯空白增长不粘贴
        assert_eq!(c.plan("你好 "), Plan::Skip);
        assert_eq!(c.plan("你好  "), Plan::Skip);
        // 后续增量带着空白照常提交
        assert_eq!(c.plan("你好 world"), Plan::Paste(" world".to_string()));
        c.confirm(" world");
        assert_eq!(c.pasted, "你好 world");
    }

    #[test]
    fn finalize_remainder_only_pastes_the_tail() {
        let _guard = ARM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_final_paste_prefix();
        // 无前缀：原文照贴
        assert_eq!(adjust_final_paste("整段文本"), "整段文本");
        // 前缀延伸：只补差额，绝不整段重贴
        arm_final_paste_prefix("你好，".to_string());
        assert_eq!(adjust_final_paste("你好，世界。"), "世界。");
        // 消费一次后不再影响后续粘贴
        assert_eq!(adjust_final_paste("另一段"), "另一段");
        // 全部已上屏（最终文本尾部被 trim）：不再粘贴
        arm_final_paste_prefix("你好 ".to_string());
        assert_eq!(adjust_final_paste("你好"), "");
        // 非前缀（后处理改写了已上屏部分）：保留已上屏，不补不重贴
        arm_final_paste_prefix("你好世界".to_string());
        assert_eq!(adjust_final_paste("再见世界"), "");
        // 清除开关：armed 也会被新一场转写清除
        arm_final_paste_prefix("残留".to_string());
        clear_final_paste_prefix();
        assert_eq!(adjust_final_paste("新录音文本"), "新录音文本");
    }

    #[test]
    fn eligibility_gates() {
        let mut settings = AppSettings::default();
        assert!(eligibility(&settings).is_err());
        settings.incremental_commit = true;
        assert!(eligibility(&settings).is_ok());
        settings.paste_method = PasteMethod::None;
        assert!(eligibility(&settings).is_err());
        settings.paste_method = PasteMethod::CtrlV;
        // 配好可用的 LLM 润色后禁用（保持一次性粘贴路径）
        settings.post_process_enabled = true;
        settings.post_process_provider_id = "custom".to_string();
        settings
            .post_process_providers
            .push(crate::settings::PostProcessProvider {
                id: "custom".to_string(),
                label: "Custom".to_string(),
                base_url: "http://localhost:1/v1".to_string(),
                allow_base_url_edit: false,
                models_endpoint: None,
                supports_structured_output: false,
                api_format: Default::default(),
            });
        settings
            .post_process_models
            .insert("custom".to_string(), "gpt-test".to_string());
        settings.post_process_selected_prompt_id = Some("p1".to_string());
        settings
            .post_process_prompts
            .push(crate::settings::LLMPrompt {
                id: "p1".to_string(),
                name: "P".to_string(),
                prompt: "polish ${output}".to_string(),
            });
        assert!(polish_possible(&settings));
        assert!(eligibility(&settings).is_err());
        // 关闭润色开关后恢复
        settings.post_process_enabled = false;
        assert!(eligibility(&settings).is_ok());
    }

    #[test]
    fn disabled_committer_is_a_no_op() {
        let c = StreamCommitter::disabled();
        assert!(!c.enabled());
        c.submit("你好");
        let mut c = c;
        assert_eq!(c.flush(), "");
    }

    /// A recording paste action that counts calls and can fail on demand.
    struct RecorderPaste {
        pasted: Arc<StdMutex<Vec<String>>>,
        fail_next: StdMutex<bool>,
    }

    impl RecorderPaste {
        fn run(&self, delta: &str) -> Result<(), String> {
            if *self.fail_next.lock().unwrap() {
                *self.fail_next.lock().unwrap() = false;
                return Err("simulated focus loss".to_string());
            }
            self.pasted.lock().unwrap().push(delta.to_string());
            Ok(())
        }
    }

    fn wait_for(predicate: impl Fn() -> bool) {
        for _ in 0..200 {
            if predicate() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("condition not met in time");
    }

    /// Deterministic spawn for the existing committer tests: the identity
    /// probe reports "unavailable", so the guard silently allows
    /// (best-effort) and pastes exercise the committer, not the host
    /// desktop.capture() 返回 None 时必须零告警、零行为差异。
    fn spawn_probe_unavailable<F>(paste: F) -> StreamCommitter
    where
        F: FnMut(&str) -> Result<(), String> + Send + 'static,
    {
        StreamCommitter::spawn_gated(paste, Box::new(|| None))
    }

    /// 身份探测脚本：第 1 次调用（会话起点）与之后每次校验按序回放。
    fn scripted_probe(steps: Vec<&str>) -> CaptureFn {
        use crate::managers::paste_target::PasteTarget;
        let steps: Vec<String> = steps.into_iter().map(String::from).collect();
        let calls = Arc::new(Mutex::new(0usize));
        Box::new(move || {
            let mut called = calls.lock().unwrap();
            let step = (*called).min(steps.len().saturating_sub(1));
            *called += 1;
            steps.get(step).map(|k| PasteTarget::new(k, k))
        })
    }

    #[test]
    fn target_guard_blocks_when_focus_changes() {
        // 起点 pid:1；第一次校验仍 pid:1（放行），随后焦点换成 pid:2。
        let recorder = Arc::new(RecorderPaste {
            pasted: Arc::new(Mutex::new(Vec::new())),
            fail_next: Mutex::new(false),
        });
        let for_thread = {
            let recorder = Arc::clone(&recorder);
            move |delta: &str| recorder.run(delta)
        };
        let mut committer = StreamCommitter::spawn_gated(
            for_thread,
            scripted_probe(vec!["pid:1", "pid:1", "pid:2"]),
        );
        committer.submit("你好");
        committer.submit("你好，世界");
        let pasted = committer.flush();
        // 只上屏焦点变化前的第一段；后续增量被拦下，绝不贴进新焦点。
        assert_eq!(*recorder.pasted.lock().unwrap(), vec!["你好".to_string()]);
        // 被拦下的文本保留在记账之外：flush 只报告真正上屏的前缀，
        // 剩余部分留给最终一次性粘贴补齐（adjust_final_paste 语义）。
        assert_eq!(pasted, "你好");
    }

    #[test]
    fn thread_pastes_increments_in_order() {
        let recorder = Arc::new(RecorderPaste {
            pasted: Arc::new(StdMutex::new(Vec::new())),
            fail_next: StdMutex::new(false),
        });
        let for_thread = {
            let recorder = Arc::clone(&recorder);
            move |delta: &str| recorder.run(delta)
        };
        let mut committer = spawn_probe_unavailable(for_thread);
        assert!(committer.enabled());
        committer.submit("你好");
        committer.submit("你好，世界");
        committer.submit("你好，世界。");
        let pasted = committer.flush();
        let calls = recorder.pasted.lock().unwrap().clone();
        assert_eq!(calls, vec!["你好", "，世界", "。"]);
        assert_eq!(pasted, "你好，世界。");
    }

    #[test]
    fn paste_failure_degrades_and_keeps_bookkeeping() {
        let recorder = Arc::new(RecorderPaste {
            pasted: Arc::new(StdMutex::new(Vec::new())),
            fail_next: StdMutex::new(true),
        });
        let for_thread = {
            let recorder = Arc::clone(&recorder);
            move |delta: &str| recorder.run(delta)
        };
        let mut committer = spawn_probe_unavailable(for_thread);
        committer.submit("你好");
        // 首段粘贴尝试发生且失败（fail 标志被消耗）
        wait_for(|| !*recorder.fail_next.lock().unwrap());
        // 首段粘贴失败 → 降级：后续增量全部跳过，已上屏记账为空
        committer.submit("你好世界");
        assert_eq!(committer.flush(), "");
        assert!(recorder.pasted.lock().unwrap().is_empty());
    }

    #[test]
    fn drop_abandons_queued_deltas() {
        let recorder = Arc::new(RecorderPaste {
            pasted: Arc::new(StdMutex::new(Vec::new())),
            fail_next: StdMutex::new(false),
        });
        let committer = {
            let recorder = Arc::clone(&recorder);
            spawn_probe_unavailable(move |delta: &str| recorder.run(delta))
        };
        committer.submit("第一段");
        wait_for(|| !recorder.pasted.lock().unwrap().is_empty());
        // 取消：句柄 drop 后，队列里未粘贴的增量不得再上屏
        drop(committer);
        let count_after_drop = recorder.pasted.lock().unwrap().len();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(recorder.pasted.lock().unwrap().len(), count_after_drop);
    }
}
