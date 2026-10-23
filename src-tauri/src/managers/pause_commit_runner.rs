//! 停顿提交运行器（pause-commit runner）：把 [`PauseCommitSegmenter`] 接到
//! 真实采集、转写与粘贴链路上，为批处理模型实现"停顿即出字"。
//!
//! # 链路
//!
//! ```text
//! CaptureSink --每块一个 VAD 判定(含被丢弃的静音)--> verdict_sink
//!   -> PauseCommitEngine（缓冲 + 分段器 + 绝对区间切片 + 静音回收）
//!   -> mpsc 队列 -> pause-commit worker 线程
//!        -> TranscriptionManager::transcribe_segment（批处理转写 + 后处理）
//!        -> utils::paste（既有粘贴通道，逐段追加）
//! ```
//!
//! 判定是 `SmoothedVad` 平滑后的输出（语音结束后 hangover 期间仍判语
//! 音）：分段器在语音→静音跳变处把 hangover 记入已流逝的真实静音
//! （`PauseCommitConfig::hangover_ms`，由采集侧按实际帧长传入），因此
//! 默认 400 ms 确认静音在第一个非语音判定到达（≈512 ms 真实静音）时
//! 即触发，而不是叠加成 ~900 ms。
//!
//! # 记账（不重叠、不漏样本）
//!
//! - 采集回调把**每一个**条件化采样块（含静音）按原顺序追加进引擎缓冲，
//!   缓冲下标 0 对应绝对样本位置 `origin`；分段器的 `position` 与缓冲严格
//!   同步：整帧块逐帧推进，非整帧的块拆成整帧 + 实际长度的尾巴——sink
//!   只把 `valid_frames` 交给 tap，录音尾块常短于一帧，恒按整帧推进会让
//!   尾段区间越过缓冲末端、切片失败而**整段丢失**（W1）。
//! - 段边界是分段器给出的 16 kHz **绝对样本区间** `[start, end)`，切片即
//!   `buffer[start - origin .. end - origin]`：相邻段区间单调不重叠，每个
//!   有声样本恰好属于一段（段间静音按 VAD 语义裁掉，不属于任何段）。
//! - 已提交边界之前的样本、以及连续静音超过 3 s 的前缀（此时分段器必然已
//!   回到 Idle）会被回收，长口述内存有界。
//!
//! # 失败隔离
//!
//! 每段独立转写、独立粘贴；一段失败（或 panic）只报错并跳过该段，已提交
//! 的文字不受影响，后续段照常处理。取消（cancel generation 变化）后未处
//! 理的段直接丢弃，不再粘贴。
//!
//! # 与既有路径的关系
//!
//! 默认关闭（`incremental_commit=false`）时本模块完全不参与：不装 tap、
//! 不起线程，采集与整段转写路径逐字节不变。

use crate::audio_toolkit::audio::VadVerdictCallback;
use crate::audio_toolkit::pause_commit::{
    CommittedSegment, PauseCommitConfig, PauseCommitSegmenter,
};
use crate::managers::paste_target::{CaptureFn, TargetGuard};
use crate::managers::transcription::TranscriptionManager;
use log::{debug, error, info, warn};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter};

/// 连续静音超过 3 s 即回收其前缀：最长的提交等待是短句确认静音
/// 800 ms **真实静音**（hangover 已折算在内），远小于 3 s，因此 3 s 连续
/// "非语音"判定足以证明分段器已回到 Idle；而未来任何一段的 `start` 都
/// 不早于当前 `position()`，回收不会丢掉仍会被切片引用的样本。
const SILENCE_DRAIN_THRESHOLD: usize = 3 * 16_000;

/// 一个待转写、待粘贴的段。`start`/`end` 是 16 kHz 绝对样本位置。
#[derive(Debug, Clone)]
pub struct SegmentJob {
    pub index: u64,
    pub start: usize,
    pub end: usize,
    /// 达到单段硬上限被迫切分：边界可能落在词中间，此段文字不可靠。
    pub forced: bool,
    pub samples: Vec<f32>,
}

/// 段与段之间的拼接规则：两边都是 ASCII（拉丁文字/数字/符号）补一个
/// 空格；任一边是 CJK（中文等）则不补——中文段间绝不插空格；混合脚本
/// 不补（贴近中文排版）。不自动加句号。
fn needs_join_space(previous_last: Option<char>, next_first: char) -> bool {
    match previous_last {
        Some(last) => last.is_ascii() && next_first.is_ascii(),
        // 没有上文（首段）：与既有整段粘贴一致，不加前缀。
        None => false,
    }
}

/// `text` 作为新一段要粘贴的增量（含按语言补的空格前缀）。
fn segment_piece(committed: &str, text: &str) -> String {
    let previous_last = committed.chars().last();
    let next_first = text.chars().next();
    let space = next_first.is_some_and(|first| needs_join_space(previous_last, first));
    if space {
        format!(" {text}")
    } else {
        text.to_string()
    }
}

/// 把一段文字并入已提交全文（诊断用副本），与 [`segment_piece`] 同一规则。
fn append_segment_text(committed: &mut String, next: &str) {
    committed.push_str(&segment_piece(committed, next));
}

/// 分段记账核心：连续缓冲 + 分段器 + 绝对区间切片 + 静音回收。纯逻辑，
/// 不碰线程与 Tauri，可独立单测。
struct PauseCommitEngine {
    segmenter: PauseCommitSegmenter,
    frame_samples: usize,
    /// 自绝对位置 `origin` 起的连续采样（含静音）。
    buffer: Vec<f32>,
    origin: usize,
    /// 连续非语音样本数。
    silent_run: usize,
    next_index: u64,
    /// 只警告一次的块长异常（正常每个采集块恰好一个 VAD 帧）。
    warned_misaligned: bool,
}

impl PauseCommitEngine {
    fn new(cfg: PauseCommitConfig, frame_samples: usize) -> Self {
        Self {
            segmenter: PauseCommitSegmenter::new(cfg, frame_samples),
            frame_samples,
            buffer: Vec::new(),
            origin: 0,
            silent_run: 0,
            next_index: 0,
            warned_misaligned: false,
        }
    }

    /// 喂入一个采集块及其 VAD 判定，返回本块提交出的段（按提交顺序）。
    ///
    /// 正常每个采集块恰好一个 VAD 帧；尾块的 `valid_frames` 可能小于
    /// 一帧（输入缓冲零填充已被 sink 裁掉），防御路径也可能见到多帧块。
    /// 无论哪种情况都按**实际样本数**记账（W1）：整帧部分逐帧推进判定，
    /// 非整帧的尾巴按实际长度推进（`push_frame_len`），分段器的
    /// `position` 与缓冲始终严格同步——恒按整帧推进会让尾段区间越过
    /// 缓冲末端，切片失败而整段丢失。
    fn push_chunk(&mut self, voiced: bool, samples: &[f32]) -> Vec<SegmentJob> {
        let full_frames = samples.len() / self.frame_samples;
        let tail = samples.len() % self.frame_samples;
        if tail != 0 && !self.warned_misaligned {
            self.warned_misaligned = true;
            warn!(
                "pause-commit: chunk of {} samples is not whole {}-sample VAD \
                 frames; accounting stays sample-exact ({} whole frame(s) + a \
                 {}-sample tail)",
                samples.len(),
                self.frame_samples,
                full_frames,
                tail
            );
        }

        self.buffer.extend_from_slice(samples);
        let mut jobs = Vec::new();
        for _ in 0..full_frames {
            if let Some(segment) = self.segmenter.push_frame(voiced) {
                if let Some(job) = self.job_for(segment) {
                    jobs.push(job);
                }
            }
        }
        if tail > 0 {
            if let Some(segment) = self.segmenter.push_frame_len(voiced, tail) {
                if let Some(job) = self.job_for(segment) {
                    jobs.push(job);
                }
            }
        }
        if voiced {
            self.silent_run = 0;
        } else {
            self.silent_run += samples.len();
        }
        self.drain();
        jobs
    }

    /// 录音结束：把进行中的段（若有语音）作为最后一段提交。
    fn finish(&mut self) -> Option<SegmentJob> {
        let segment = self.segmenter.finish()?;
        let job = self.job_for(segment);
        // 之后不会再有音频，全部回收。
        let keep = self.segmenter.position().saturating_sub(self.origin);
        if keep <= self.buffer.len() {
            self.buffer.drain(..keep);
            self.origin += keep;
        }
        job
    }

    /// 已缓冲到的绝对样本位置。
    fn position(&self) -> usize {
        self.origin + self.buffer.len()
    }

    /// 绝对区间 → 缓冲切片。区间必须仍在缓冲内；越界说明记账被破坏，
    /// 报错并跳过该段（不 panic、不影响其它段）。
    fn job_for(&mut self, segment: CommittedSegment) -> Option<SegmentJob> {
        let Some(start) = segment.start.checked_sub(self.origin) else {
            error!(
                "pause-commit: segment [{}, {}) starts before the buffer origin {}",
                segment.start, segment.end, self.origin
            );
            return None;
        };
        let Some(end) = segment.end.checked_sub(self.origin) else {
            error!(
                "pause-commit: segment [{}, {}) ends before the buffer origin {}",
                segment.start, segment.end, self.origin
            );
            return None;
        };
        let samples = match self.buffer.get(start..end) {
            Some(slice) => slice.to_vec(),
            None => {
                error!(
                    "pause-commit: slice [{}, {}) outside buffered range [{}, {})",
                    segment.start,
                    segment.end,
                    self.origin,
                    self.position()
                );
                return None;
            }
        };
        let index = self.next_index;
        self.next_index += 1;
        Some(SegmentJob {
            index,
            start: segment.start,
            end: segment.end,
            forced: segment.forced,
            samples,
        })
    }

    /// 回收不会再被引用的静音前缀：已提交边界之前永远安全；连续静音超过
    /// 阈值时，`position()` 之前也不会再被引用（见 `SILENCE_DRAIN_THRESHOLD`）。
    fn drain(&mut self) {
        let mut floor = self.segmenter.committed_until();
        if self.silent_run >= SILENCE_DRAIN_THRESHOLD {
            floor = floor.max(self.segmenter.position());
        }
        let keep = floor.saturating_sub(self.origin);
        if keep > 0 && keep <= self.buffer.len() {
            self.buffer.drain(..keep);
            self.origin += keep;
        }
    }
}

/// Worker 命令。段与终止信号走同一条 FIFO 通道，保证粘贴按提交顺序。
enum Cmd {
    Segment(SegmentJob),
    Abort,
}

/// 取消判定：录音管理器的 cancel generation 已离开本次录音。
pub type CancelCheck = Box<dyn Fn() -> bool + Send + Sync>;

/// 一段的转写与粘贴。生产实现走 `TranscriptionManager` 与既有粘贴通道；
/// 测试用假实现验证隔离与顺序。
pub trait SegmentCommitter: Send {
    /// 批处理转写该段音频并返回文本（失败时向用户报错并返回 `Err`）。
    fn transcribe(&mut self, job: &SegmentJob) -> Result<String, String>;
    /// 通过既有粘贴通道粘贴拼接好的增量文本。
    fn paste(&mut self, piece: &str) -> Result<(), String>;
    /// 向用户报告该段失败（toast 等）。消息不含音频，只含错误说明。
    fn report_failure(&mut self, job: &SegmentJob, error: &str);
}

struct RunnerShared {
    engine: Mutex<PauseCommitEngine>,
    /// `None` = 已终结（finalize/abort 过），通道关闭。
    tx: Mutex<Option<Sender<Cmd>>>,
    /// 已粘贴文本的诊断副本（用于日志与收尾统计）。
    committed: Mutex<String>,
    cancelled: CancelCheck,
    done: Mutex<bool>,
    done_cv: Condvar,
    committed_segments: AtomicU64,
    /// 被目标守卫拦下、未上屏的文本（诊断副本：文本保留可见，不丢弃）。
    held: Mutex<String>,
    held_segments: AtomicU64,
}

fn lock_or_recover<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// 一次停顿提交会话的句柄。`Arc` 共享：采集回调、录音管理器与转写管理器
/// 各持一份；worker 退出后 `wait()` 返回。
pub struct PauseCommitRunner {
    shared: Arc<RunnerShared>,
}

impl PauseCommitRunner {
    /// 生产构造：转写走 `TranscriptionManager::transcribe_segment`，粘贴走
    /// `utils::paste`，取消跟随录音管理器的 cancel generation。
    /// `hangover_ms` 是采集 VAD 对本次策略实际施加的语音尾部 hangover
    /// （`CaptureVad::hangover_frames_for` × 帧长），分段器用它把确认
    /// 静音折回真实时间轴。
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        app: AppHandle,
        transcription: Arc<TranscriptionManager>,
        cancel_flag: Arc<AtomicU64>,
        cancel_generation: u64,
        frame_samples: usize,
        hangover_ms: u32,
    ) -> Arc<Self> {
        let committer = TauriSegmentCommitter { app, transcription };
        let cancelled: CancelCheck =
            Box::new(move || cancel_flag.load(Ordering::Acquire) != cancel_generation);
        Self::spawn(
            Box::new(committer),
            cancelled,
            Box::new(crate::managers::paste_target::capture),
            frame_samples,
            hangover_ms,
        )
    }

    /// 通用构造（生产与测试共用）。worker 线程立刻开始接收段。
    pub fn spawn(
        committer: Box<dyn SegmentCommitter>,
        cancelled: CancelCheck,
        capture: CaptureFn,
        frame_samples: usize,
        hangover_ms: u32,
    ) -> Arc<Self> {
        let cfg = PauseCommitConfig {
            hangover_ms,
            ..PauseCommitConfig::default()
        };
        // 一次性启动日志：把各档“触发的静音阈值 ms”钉在真实时间轴上，
        // 用户从日志就能确认多长的停顿会出字（修复前旧实现实际需 ~928/
        // ~1472 ms 真实静音，日志里看不出来）。
        info!(
            "pause-commit armed: a pause commits after {}ms of real silence \
             (segments shorter than {}ms wait {}ms; past {}ms of speech a \
             {}ms pause commits); verdicts carry a {}ms hangover credited at \
             each speech-to-silence transition, frame {} samples",
            cfg.real_silence_ms(cfg.silence_ms, frame_samples),
            cfg.short_utterance_ms,
            cfg.real_silence_ms(cfg.short_utterance_silence_ms, frame_samples),
            cfg.seek_pause_ms,
            cfg.real_silence_ms(cfg.seek_pause_silence_ms, frame_samples),
            hangover_ms,
            frame_samples
        );
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(RunnerShared {
            engine: Mutex::new(PauseCommitEngine::new(cfg, frame_samples)),
            tx: Mutex::new(Some(tx)),
            committed: Mutex::new(String::new()),
            cancelled,
            done: Mutex::new(false),
            done_cv: Condvar::new(),
            committed_segments: AtomicU64::new(0),
            held: Mutex::new(String::new()),
            held_segments: AtomicU64::new(0),
        });

        let worker_shared = Arc::clone(&shared);
        let spawned = std::thread::Builder::new()
            .name("pause-commit".into())
            .spawn(move || run_worker(worker_shared, rx, committer, capture));

        if let Err(spawn_error) = spawned {
            // 起线程失败：关闭通道并标记完成，录音自动退回整段路径的
            // 收尾（已提交的段保持，未提交的丢弃）。
            error!("pause-commit: failed to spawn worker thread: {spawn_error}");
            if let Some(tx) = lock_or_recover(&shared.tx).take() {
                drop(tx);
            }
            *lock_or_recover(&shared.done) = true;
            shared.done_cv.notify_all();
        }

        Arc::new(Self { shared })
    }

    /// 装到 `CaptureSink` 上的逐帧判定 tap。只做缓冲追加、分段器推进与
    /// 非阻塞入队，在采集投递线程上运行，绝不转写、绝不粘贴。
    pub fn verdict_sink(self: &Arc<Self>) -> VadVerdictCallback {
        let shared = Arc::clone(&self.shared);
        Arc::new(move |voiced: bool, samples: &[f32]| {
            let jobs = lock_or_recover(&shared.engine).push_chunk(voiced, samples);
            if jobs.is_empty() {
                return;
            }
            if let Some(tx) = lock_or_recover(&shared.tx).as_ref() {
                for job in jobs {
                    // 队列无人消费时（worker 已死）发送失败即丢弃，
                    // 不能阻塞采集线程。
                    let _ = tx.send(Cmd::Segment(job));
                }
            }
        })
    }

    /// 录音正常停止：提交尾段（若有）并让 worker 清空队列后自行退出。
    /// 幂等；与 `abort` 先到者生效。
    pub fn finalize(&self) {
        let tail = lock_or_recover(&self.shared.engine).finish();
        if let Some(tx) = lock_or_recover(&self.shared.tx).take() {
            if let Some(job) = tail {
                let _ = tx.send(Cmd::Segment(job));
            }
            // 丢掉最后一个 Sender：worker 清完队列后 recv 失败即退出。
            drop(tx);
        }
    }

    /// 放弃会话：未处理的段直接丢弃，不再转写、不再粘贴。幂等。
    pub fn abort(&self) {
        if let Some(tx) = lock_or_recover(&self.shared.tx).take() {
            let _ = tx.send(Cmd::Abort);
            drop(tx);
        }
    }

    /// 等 worker 结束。幂等，可多 waiter。
    pub fn wait(&self) {
        let mut done = lock_or_recover(&self.shared.done);
        while !*done {
            done = match self.shared.done_cv.wait(done) {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
    }

    /// 已粘贴全文（诊断副本）。
    pub fn committed_text(&self) -> String {
        lock_or_recover(&self.shared.committed).clone()
    }

    /// 已成功粘贴的段数。
    pub fn committed_segments(&self) -> u64 {
        self.shared.committed_segments.load(Ordering::Acquire)
    }

    /// 被目标守卫拦下、未上屏的文本（诊断副本）。
    pub fn held_text(&self) -> String {
        lock_or_recover(&self.shared.held).clone()
    }

    /// 被目标守卫拦下的段数。
    pub fn held_segments(&self) -> u64 {
        self.shared.held_segments.load(Ordering::Acquire)
    }
}

/// 生产用 [`SegmentCommitter`]：每段一次批处理转写（与整段路径同一套
/// setup 与后处理），随后走既有粘贴通道追加。
struct TauriSegmentCommitter {
    app: AppHandle,
    transcription: Arc<TranscriptionManager>,
}

impl SegmentCommitter for TauriSegmentCommitter {
    fn transcribe(&mut self, job: &SegmentJob) -> Result<String, String> {
        if job.forced {
            // 硬上限forced边界：可能切在词中间，必须可诊断。
            warn!(
                "pause-commit segment #{}: forced boundary (max segment length) — \
                 the cut may split a word; text around this seam is unreliable",
                job.index
            );
        }
        let started = Instant::now();
        let text = self
            .transcription
            .transcribe_segment(job.samples.clone())
            .map_err(|e| e.to_string())?;
        debug!(
            "pause-commit segment #{} decoded {} sample(s) in {:?}: {} char(s)",
            job.index,
            job.samples.len(),
            started.elapsed(),
            text.chars().count()
        );
        Ok(text)
    }

    fn paste(&mut self, piece: &str) -> Result<(), String> {
        crate::utils::paste(piece.to_string(), self.app.clone())
    }

    fn report_failure(&mut self, job: &SegmentJob, error: &str) {
        let _ = self.app.emit(
            "transcription-error",
            format!("Segment {}: {error}", job.index + 1),
        );
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn run_worker(
    shared: Arc<RunnerShared>,
    rx: Receiver<Cmd>,
    mut committer: Box<dyn SegmentCommitter>,
    capture: CaptureFn,
) {
    let started = Instant::now();
    // 相同错误只 toast 一次：坏模型会让每一段都失败，不能刷屏。
    let mut last_reported_error: Option<String> = None;
    // 会话起点：绑定前台窗口身份；每段粘贴前校验。取不到身份的平台
    // 静默 best-effort（不告警、不记日志），身份变化则拦下且只记 debug。
    let mut guard = TargetGuard::with_capture(capture);
    // 守卫拦截只 toast 一次（第一段带文本交还用户），其余仅记日志。
    let mut guard_reported = false;

    while let Ok(cmd) = rx.recv() {
        let Cmd::Segment(job) = cmd else {
            debug!("pause-commit worker aborted; dropping everything not yet pasted");
            break;
        };
        if (shared.cancelled)() {
            info!(
                "pause-commit worker: recording cancelled; dropping segment #{} unpasted",
                job.index
            );
            break;
        }

        // 1) 转写（隔离 panic 与错误：只影响本段）。
        let transcribed = catch_unwind(AssertUnwindSafe(|| committer.transcribe(&job)));
        let text = match transcribed {
            Ok(Ok(text)) => text.trim().to_string(),
            Ok(Err(error)) => {
                error!(
                    "pause-commit segment #{} failed; committed text is untouched: {}",
                    job.index, error
                );
                if last_reported_error.as_deref() != Some(error.as_str()) {
                    committer.report_failure(&job, &error);
                    last_reported_error = Some(error);
                }
                continue;
            }
            Err(payload) => {
                let message = panic_message(&*payload);
                error!(
                    "pause-commit segment #{} panicked; committed text is untouched: {}",
                    job.index, message
                );
                committer.report_failure(&job, &message);
                last_reported_error = Some(message);
                continue;
            }
        };
        if text.is_empty() {
            debug!(
                "pause-commit segment #{} produced no text; nothing pasted",
                job.index
            );
            continue;
        }

        // 2) 按语言拼接出本段增量（中文不插空格）。
        let piece = segment_piece(&lock_or_recover(&shared.committed), &text);

        // 3) 粘贴前再查一次取消：取消后宁可丢这一段也不追加。
        if (shared.cancelled)() {
            info!(
                "pause-commit segment #{}: cancelled before paste; dropped",
                job.index
            );
            break;
        }
        // 3a) 粘贴目标守卫：焦点离开起点窗口后绝不再粘贴；文本保留在
        //     held 记账里（不丢弃），绝不贴进新焦点、不把焦点抢回来。
        if !guard.allow_paste() {
            let reason = guard.block_reason().unwrap_or("unknown").to_string();
            debug!(
                "pause-commit segment #{} held back ({} char(s)): {}",
                job.index,
                piece.trim().chars().count(),
                reason
            );
            {
                let mut held = lock_or_recover(&shared.held);
                let joined = segment_piece(&held, &text);
                held.push_str(&joined);
            }
            shared.held_segments.fetch_add(1, Ordering::AcqRel);
            if !guard_reported {
                guard_reported = true;
                // 既有错误 toast：只报第一段（带被拦文本），其余段仅记
                // 日志，避免长口述刷屏。
                committer
                    .report_failure(&job, &format!("held unpasted ({reason}): {}", piece.trim()));
            }
            continue;
        }
        match catch_unwind(AssertUnwindSafe(|| committer.paste(&piece))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                error!(
                    "pause-commit segment #{} paste failed; committed text is untouched: {}",
                    job.index, error
                );
                if last_reported_error.as_deref() != Some(error.as_str()) {
                    committer.report_failure(&job, &error);
                    last_reported_error = Some(error);
                }
                continue;
            }
            Err(payload) => {
                let message = panic_message(&*payload);
                error!(
                    "pause-commit segment #{} paste panicked: {}",
                    job.index, message
                );
                committer.report_failure(&job, &message);
                last_reported_error = Some(message);
                continue;
            }
        }

        // 4) 粘贴成功才计入已提交全文。
        append_segment_text(&mut lock_or_recover(&shared.committed), &text);
        shared.committed_segments.fetch_add(1, Ordering::AcqRel);
        last_reported_error = None;
        info!(
            "pause-commit segment #{} committed: samples [{}, {}) ({} ms), forced={}, {} char(s) pasted",
            job.index,
            job.start,
            job.end,
            (job.end - job.start) * 1000 / 16_000,
            job.forced,
            piece.trim_start().chars().count()
        );
        debug!(
            "pause-commit committed text so far: {}",
            crate::utils::redact_text(&shared.committed.lock().unwrap_or_else(|p| p.into_inner()))
        );
    }

    let held_segments = shared.held_segments.load(Ordering::Acquire);
    if held_segments > 0 {
        debug!(
            "pause-commit worker finished: {} segment(s) held back by the target guard: {}",
            held_segments,
            crate::utils::redact_text(&lock_or_recover(&shared.held))
        );
    }
    *lock_or_recover(&shared.done) = true;
    shared.done_cv.notify_all();
    debug!(
        "pause-commit worker finished: {} segment(s) committed in {:?}",
        shared.committed_segments.load(Ordering::Acquire),
        started.elapsed()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const FRAME: usize = 512; // 32 ms @ 16 kHz，与分段器单测一致

    /// 每帧样本值 = 帧序号（有声）或 10_000+序号（静音），便于核对切片。
    fn frame(marker: f32) -> Vec<f32> {
        vec![marker; FRAME]
    }

    /// 按毫秒构造帧流（有声段用递增标记，静音段用 10_000+ 标记）。
    /// 返回 (帧列表, 完整镜像流)。
    fn stream(
        voiced_ms: u32,
        silent_ms: u32,
        first_marker: f32,
    ) -> (Vec<(bool, Vec<f32>)>, Vec<f32>) {
        let mut chunks = Vec::new();
        let mut mirror = Vec::new();
        let mut marker = first_marker;
        for _ in 0..(voiced_ms / 32) {
            let f = frame(marker);
            mirror.extend_from_slice(&f);
            chunks.push((true, f));
            marker += 1.0;
        }
        for _ in 0..(silent_ms / 32) {
            let f = frame(10_000.0 + marker);
            mirror.extend_from_slice(&f);
            chunks.push((false, f));
            marker += 1.0;
        }
        (chunks, mirror)
    }

    /// 平滑判定的帧流（tap 实际收到的那种）：真实语音判 true；其后
    /// hangover_ms 的真实静音仍判 true；再往后才判 false。标记逐帧递增，
    /// 镜像流照记全部样本。
    fn stream_smoothed(
        voiced_ms: u32,
        hangover_ms: u32,
        silent_ms: u32,
        first_marker: f32,
    ) -> (Vec<(bool, Vec<f32>)>, Vec<f32>) {
        let mut chunks = Vec::new();
        let mut mirror = Vec::new();
        let mut marker = first_marker;
        for _ in 0..(voiced_ms / 32) {
            let f = frame(marker);
            mirror.extend_from_slice(&f);
            chunks.push((true, f));
            marker += 1.0;
        }
        for _ in 0..(hangover_ms / 32) {
            let f = frame(marker);
            mirror.extend_from_slice(&f);
            chunks.push((true, f));
            marker += 1.0;
        }
        for _ in 0..(silent_ms / 32) {
            let f = frame(10_000.0 + marker);
            mirror.extend_from_slice(&f);
            chunks.push((false, f));
            marker += 1.0;
        }
        (chunks, mirror)
    }

    fn engine() -> PauseCommitEngine {
        PauseCommitEngine::new(PauseCommitConfig::default(), FRAME)
    }

    #[test]
    fn segments_slice_exact_absolute_ranges_without_overlap_or_gap() {
        let mut eng = engine();
        let mut mirror = Vec::new();
        let mut jobs = Vec::new();

        let mut feed = |voiced_ms: u32,
                        silent_ms: u32,
                        marker: f32,
                        eng: &mut PauseCommitEngine,
                        mirror: &mut Vec<f32>,
                        jobs: &mut Vec<SegmentJob>| {
            let (chunks, m) = stream(voiced_ms, silent_ms, marker);
            mirror.extend_from_slice(&m);
            for (voiced, samples) in chunks {
                jobs.extend(eng.push_chunk(voiced, &samples));
            }
        };

        feed(4_000, 700, 1.0, &mut eng, &mut mirror, &mut jobs);
        feed(6_000, 1_200, 100.0, &mut eng, &mut mirror, &mut jobs);
        feed(2_000, 800, 300.0, &mut eng, &mut mirror, &mut jobs);
        jobs.extend(eng.finish().into_iter());

        assert_eq!(jobs.len(), 3, "三段口述各提交一次：{jobs:?}");
        let mut prev_end = 0;
        for job in &jobs {
            assert!(
                job.start >= prev_end,
                "段不得重叠: {job:?} after {prev_end}"
            );
            assert!(job.end > job.start);
            prev_end = job.end;
            // 切片必须逐字节等于采集缓冲的绝对区间。
            assert_eq!(
                job.samples,
                mirror[job.start..job.end],
                "段 #{} 切片不精确",
                job.index
            );
        }
        // 每个有声样本恰好被一段覆盖（静音不属于任何段）。
        let voiced_total: usize = mirror.chunks(FRAME).filter(|c| c[0] < 10_000.0).count() * FRAME;
        let covered: usize = jobs.iter().map(|j| j.end - j.start).sum();
        assert_eq!(covered, voiced_total, "有声样本不得漏、不得重复计入");
    }

    #[test]
    fn silence_drain_keeps_slices_exact_after_long_pause() {
        let mut eng = engine();
        let mut mirror = Vec::new();
        let mut jobs = Vec::new();

        // 段一 + 超过回收阈值的静音（4 s > 3 s → 触发 drain）。
        let (chunks, m) = stream(2_000, 4_000, 1.0);
        mirror.extend_from_slice(&m);
        for (voiced, samples) in chunks {
            jobs.extend(eng.push_chunk(voiced, &samples));
        }
        assert_eq!(jobs.len(), 1);
        assert!(eng.origin > 0, "长静音应推进缓冲 origin（回收已发生）");

        // 段二在回收之后开始：切片仍必须精确。
        let (chunks, m) = stream(3_500, 700, 100.0);
        mirror.extend_from_slice(&m);
        for (voiced, samples) in chunks {
            jobs.extend(eng.push_chunk(voiced, &samples));
        }
        jobs.extend(eng.finish().into_iter());

        assert_eq!(jobs.len(), 2);
        assert!(jobs[1].start >= jobs[0].end);
        for job in &jobs {
            assert_eq!(
                job.samples,
                mirror[job.start..job.end],
                "回收后切片不精确: {job:?}"
            );
        }
    }

    #[test]
    fn forced_cap_marks_the_job_unreliable() {
        let mut eng = engine();
        let mut jobs = Vec::new();
        // 40 s 连续有声：第一刀在 30 s 硬上限处，forced=true。
        let (chunks, _) = stream(40_000, 0, 1.0);
        for (voiced, samples) in chunks {
            jobs.extend(eng.push_chunk(voiced, &samples));
        }
        assert!(!jobs.is_empty(), "无停顿时必须强制切分");
        assert!(jobs[0].forced, "第一刀是硬上限 forced 边界");
        let ms = (jobs[0].end - jobs[0].start) * 1000 / 16_000;
        assert!(
            (29_900..=30_200).contains(&ms),
            "forced 边界贴着 30 s 上限，实际 {ms} ms"
        );
    }

    #[test]
    fn finish_flushes_an_uncommitted_tail() {
        let mut eng = engine();
        // 5 s 语音 + 480 ms 仍判 Speech 的 hangover（真实静音未过阈值）。
        let (chunks, mirror) = stream_smoothed(5_000, 480, 0, 1.0);
        for (voiced, samples) in chunks {
            assert!(
                eng.push_chunk(voiced, &samples).is_empty(),
                "停顿未确认前不应提交"
            );
        }
        let job = eng.finish().expect("结束时应补齐尾段");
        assert!(!job.forced, "自然收尾不是 forced 边界");
        assert_eq!(job.samples, mirror[job.start..job.end]);
    }

    #[test]
    fn short_tail_chunk_still_commits_and_loses_no_samples() {
        // W1 回归（引擎层）：sink 只把 valid_frames 交给 tap——录音停止时
        // 尾块可能不足一个完整 VAD 帧。修复前记账恒按整帧推进，尾段区间
        // 越过缓冲末端 → 切片失败 → 尾段整段丢失；现在按实际样本数记账，
        // finish 的尾段切片逐样本精确、一个样本不丢。
        let mut eng = engine();
        let mut mirror = Vec::new();
        let mut jobs = Vec::new();
        // 5 s 判语音的整帧。
        for _ in 0..(5_000 / 32) {
            let f = frame(7.0);
            mirror.extend_from_slice(&f);
            jobs.extend(eng.push_chunk(true, &f));
        }
        // 300 样本判语音的尾块（< FRAME = 512）。
        let tail = vec![9.0; 300];
        mirror.extend_from_slice(&tail);
        jobs.extend(eng.push_chunk(true, &tail));
        assert!(jobs.is_empty(), "停顿未确认前不应提交");
        assert_eq!(eng.position(), mirror.len(), "position 与缓冲必须同步");
        let job = eng.finish().expect("短尾块后尾段必须照常提交");
        assert!(!job.forced);
        assert_eq!(job.end, mirror.len(), "段尾必须覆盖全部有效样本");
        assert_eq!(
            job.samples,
            mirror[job.start..job.end],
            "切片必须逐样本精确"
        );
        assert_eq!(job.samples.len(), 156 * FRAME + 300);
    }

    #[test]
    fn misaligned_mid_stream_chunks_keep_slices_exact() {
        // W1 回归（防御路径）：中递出现非整帧块时拆成整帧 + 实际长度
        // 尾巴，position 不漂移，后续切片仍逐样本精确。
        let mut eng = engine();
        let mut mirror = Vec::new();
        let mut jobs = Vec::new();
        // 一个 2.5 帧长的块（2×512 + 256）：2 整帧 + 256 样本尾巴，判语音。
        let chunk_v: Vec<f32> = vec![5.0; FRAME * 2 + 256];
        mirror.extend_from_slice(&chunk_v);
        jobs.extend(eng.push_chunk(true, &chunk_v));
        assert!(jobs.is_empty());
        // 后续整帧静音直到提交（默认档：短句等 800 ms 真实静音）。
        for i in 0..40 {
            let f = frame(10_000.0 + i as f32);
            mirror.extend_from_slice(&f);
            jobs.extend(eng.push_chunk(false, &f));
        }
        assert!(!jobs.is_empty(), "静音足够长后应提交");
        let mut prev_end = 0;
        for job in &jobs {
            assert!(job.start >= prev_end, "段不得重叠");
            assert_eq!(
                job.samples,
                mirror[job.start..job.end],
                "切片必须逐样本精确"
            );
            prev_end = job.end;
        }
        assert_eq!(prev_end, FRAME * 2 + 256, "段尾应停在最后一个有声样本");
        assert_eq!(eng.position(), mirror.len());
    }

    #[test]
    fn half_second_real_pause_commits_on_the_first_noise_verdict() {
        // 修复目标（引擎层）：tap 送来平滑判定——480 ms hangover 仍判
        // Speech，第 512 ms 真实静音才出现第一帧 Noise；hangover 信用让
        // 这一帧立即提交（旧实现需 ~928 ms 真实静音，用户停顿够不着，
        // 于是整段只剩停止后的尾段）。
        let cfg = PauseCommitConfig {
            hangover_ms: 480,
            ..PauseCommitConfig::default()
        };
        let mut eng = PauseCommitEngine::new(cfg, FRAME);
        let (chunks, _) = stream_smoothed(4_000, 480, 0, 1.0);
        for (voiced, samples) in chunks {
            assert!(eng.push_chunk(voiced, &samples).is_empty());
        }
        let jobs = eng.push_chunk(false, &frame(10_500.0));
        assert_eq!(jobs.len(), 1, "0.5 s 真实停顿必须提交");
        assert!(!jobs[0].forced);
        assert_eq!((jobs[0].end - jobs[0].start) * 1000 / 16_000, 4_480);
    }

    #[test]
    fn short_tail_chunk_through_the_tap_still_commits_the_tail() {
        // W1 回归（全链路）：真实 tap 收到的尾块短于一帧（sink 只交
        // valid_frames）。第一段正常提交后，第二段以 300 样本的短尾块
        // 结束——finalize 必须把含短尾块的尾段提交出来（修复前记账漂移，
        // 切片越界，尾段整段丢失）。
        let (committer, _transcribed, pastes, _failures) = FakeCommitter::new(None);
        let runner = PauseCommitRunner::spawn(
            Box::new(committer),
            Box::new(|| false),
            Box::new(|| None),
            FRAME,
            480,
        );
        let sink = runner.verdict_sink();
        // 段一：4 s 判 Speech（语音 + hangover）+ 1 帧 Noise（真实静音
        // 512 ms ≥ 400 ms）→ 录音中就提交。
        let (chunks, _) = stream_smoothed(4_000, 480, 32, 1.0);
        for (voiced, samples) in chunks {
            sink(voiced, &samples);
        }
        // 段二：1 s 判 Speech 的整帧 + 300 样本判 Speech 的短尾块。
        for _ in 0..(1_000 / 32) {
            sink(true, &frame(50.0));
        }
        sink(true, &vec![60.0; 300]);
        runner.finalize();
        runner.wait();
        assert_eq!(runner.committed_segments(), 2, "短尾块的尾段必须提交");
        assert_eq!(
            *pastes.lock().unwrap(),
            vec!["t0".to_string(), " t1".to_string()]
        );
    }

    #[test]
    fn chinese_segments_join_without_space() {
        let mut committed = String::new();
        append_segment_text(&mut committed, "你好");
        append_segment_text(&mut committed, "世界");
        assert_eq!(committed, "你好世界");

        append_segment_text(&mut committed, "价格是100元");
        assert_eq!(committed, "你好世界价格是100元", "混合脚本不插空格");
    }

    #[test]
    fn latin_segments_join_with_one_space() {
        let mut committed = String::new();
        append_segment_text(&mut committed, "hello");
        append_segment_text(&mut committed, "world");
        assert_eq!(committed, "hello world");
        append_segment_text(&mut committed, "again.");
        assert_eq!(committed, "hello world again.");
    }

    #[test]
    fn empty_pieces_are_ignored() {
        let mut committed = String::new();
        append_segment_text(&mut committed, "");
        append_segment_text(&mut committed, "hi");
        append_segment_text(&mut committed, "");
        assert_eq!(committed, "hi");
    }

    /// 记录每步调用的假提交器，可指定某段转写失败。
    struct FakeCommitter {
        fail_index: Option<u64>,
        transcribed: Arc<Mutex<Vec<u64>>>,
        pastes: Arc<Mutex<Vec<String>>>,
        failures: Arc<Mutex<Vec<(u64, String)>>>,
        texts: Arc<Mutex<HashMap<u64, String>>>,
    }

    impl FakeCommitter {
        fn new(
            fail_index: Option<u64>,
        ) -> (
            Self,
            Arc<Mutex<Vec<u64>>>,
            Arc<Mutex<Vec<String>>>,
            Arc<Mutex<Vec<(u64, String)>>>,
        ) {
            let transcribed = Arc::new(Mutex::new(Vec::new()));
            let pastes = Arc::new(Mutex::new(Vec::new()));
            let failures = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    fail_index,
                    texts: Arc::new(Mutex::new(HashMap::new())),
                    transcribed: Arc::clone(&transcribed),
                    pastes: Arc::clone(&pastes),
                    failures: Arc::clone(&failures),
                },
                transcribed,
                pastes,
                failures,
            )
        }
    }

    impl SegmentCommitter for FakeCommitter {
        fn transcribe(&mut self, job: &SegmentJob) -> Result<String, String> {
            self.transcribed.lock().unwrap().push(job.index);
            if Some(job.index) == self.fail_index {
                return Err("simulated segment failure".to_string());
            }
            let text = format!("t{}", job.index);
            self.texts.lock().unwrap().insert(job.index, text.clone());
            Ok(text)
        }

        fn paste(&mut self, piece: &str) -> Result<(), String> {
            self.pastes.lock().unwrap().push(piece.to_string());
            Ok(())
        }

        fn report_failure(&mut self, job: &SegmentJob, error: &str) {
            self.failures
                .lock()
                .unwrap()
                .push((job.index, error.to_string()));
        }
    }

    /// 通过 verdict tap 喂帧流（有声+静音），驱动真实的分段与入队。
    fn feed(runner: &Arc<PauseCommitRunner>, voiced_ms: u32, silent_ms: u32, marker: f32) {
        let (chunks, _) = stream(voiced_ms, silent_ms, marker);
        let sink = runner.verdict_sink();
        for (voiced, samples) in chunks {
            sink(voiced, &samples);
        }
    }

    #[test]
    fn runner_pastes_segments_in_order_with_language_join() {
        let (committer, _transcribed, pastes, failures) = FakeCommitter::new(None);
        let runner = PauseCommitRunner::spawn(
            Box::new(committer),
            Box::new(|| false),
            Box::new(|| None),
            FRAME,
            480,
        );

        feed(&runner, 4_000, 700, 1.0);
        feed(&runner, 4_000, 700, 100.0);
        runner.finalize();
        runner.wait();

        assert!(failures.lock().unwrap().is_empty());
        let pastes = pastes.lock().unwrap();
        assert_eq!(
            *pastes,
            vec!["t0".to_string(), " t1".to_string()],
            "拉丁段间补一个空格"
        );
        assert_eq!(runner.committed_text(), "t0 t1");
        assert_eq!(runner.committed_segments(), 2);
    }

    #[test]
    fn half_second_pause_pastes_during_recording() {
        // 用户报的缺陷：录音中完全不出字，只在停止后给一个总结果。
        // 回归：0.5 s 真实停顿（15 帧 hangover 判 Speech + 1 帧 Noise）
        // 必须在录音**进行中**就粘贴，而不是等 finalize 的尾段。
        let (committer, _transcribed, pastes, _failures) = FakeCommitter::new(None);
        let runner = PauseCommitRunner::spawn(
            Box::new(committer),
            Box::new(|| false),
            Box::new(|| None),
            FRAME,
            480,
        );

        let (chunks, _) = stream_smoothed(4_000, 480, 32, 1.0);
        let sink = runner.verdict_sink();
        for (voiced, samples) in chunks {
            sink(voiced, &samples);
        }
        // 不 finalize：段必须在停止前提交并被 worker 粘贴。
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while runner.committed_segments() == 0 && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            runner.committed_segments(),
            1,
            "0.5 s 停顿必须在停止前提交（而非只等停止后的总结果）"
        );
        runner.abort(); // 收尾：丢弃可能存在的尾段（此时应为空）
        runner.wait();
        assert_eq!(*pastes.lock().unwrap(), vec!["t0".to_string()]);
    }

    #[test]
    fn segment_failure_is_isolated_and_reported() {
        let (committer, transcribed, pastes, failures) = FakeCommitter::new(Some(1));
        let runner = PauseCommitRunner::spawn(
            Box::new(committer),
            Box::new(|| false),
            Box::new(|| None),
            FRAME,
            480,
        );

        feed(&runner, 4_000, 700, 1.0); // 段 0 成功
        feed(&runner, 4_000, 700, 100.0); // 段 1 转写失败
        feed(&runner, 4_000, 700, 200.0); // 段 2 必须照常提交
        runner.finalize();
        runner.wait();

        assert_eq!(
            *transcribed.lock().unwrap(),
            vec![0, 1, 2],
            "失败的段也要尝试转写"
        );
        let pastes = pastes.lock().unwrap();
        assert_eq!(
            *pastes,
            vec!["t0".to_string(), " t2".to_string()],
            "只有失败的段被跳过"
        );
        assert_eq!(runner.committed_text(), "t0 t2", "已提交内容不受失败影响");
        assert_eq!(
            *failures.lock().unwrap(),
            vec![(1, "simulated segment failure".to_string())],
            "失败要报错（一次）"
        );
    }

    #[test]
    fn cancelled_recordings_paste_nothing() {
        let (committer, _transcribed, pastes, _failures) = FakeCommitter::new(None);
        let runner = PauseCommitRunner::spawn(
            Box::new(committer),
            Box::new(|| true), // 从一开始就处于取消状态
            Box::new(|| None),
            FRAME,
            480,
        );

        feed(&runner, 4_000, 700, 1.0);
        feed(&runner, 4_000, 700, 100.0);
        runner.finalize();
        runner.wait();

        assert!(pastes.lock().unwrap().is_empty(), "取消后不得粘贴");
        assert_eq!(runner.committed_text(), "");
        assert_eq!(runner.committed_segments(), 0);
    }

    #[test]
    fn abort_discards_queued_segments() {
        let (committer, _transcribed, pastes, _failures) = FakeCommitter::new(None);
        let runner = PauseCommitRunner::spawn(
            Box::new(committer),
            Box::new(|| false),
            Box::new(|| None),
            FRAME,
            480,
        );

        runner.abort(); // 先放弃：之后 tap 仍可安全喂（不入队）
        feed(&runner, 4_000, 700, 1.0);
        runner.finalize(); // 幂等：通道已关，无尾段
        runner.wait();

        assert!(pastes.lock().unwrap().is_empty());
        assert_eq!(runner.committed_segments(), 0);
    }

    #[test]
    fn target_guard_holds_segments_when_focus_changes() {
        // 起点记录 pid:1；段 0 校验仍 pid:1（放行粘贴），随后焦点换成
        // pid:2：之后的段必须被拦下——保留文本、不贴新焦点、不再恢复。
        let calls = Arc::new(Mutex::new(0usize));
        let calls_for_capture = Arc::clone(&calls);
        let capture: CaptureFn = Box::new(move || {
            use crate::managers::paste_target::PasteTarget;
            let mut called = calls_for_capture.lock().unwrap();
            *called += 1;
            if *called <= 2 {
                Some(PasteTarget::new("pid:1", "pid:1"))
            } else {
                Some(PasteTarget::new("pid:2", "pid:2"))
            }
        });
        let (committer, _transcribed, pastes, failures) = FakeCommitter::new(None);
        let runner =
            PauseCommitRunner::spawn(Box::new(committer), Box::new(|| false), capture, FRAME, 480);

        feed(&runner, 4_000, 700, 1.0); // 段 0：身份一致，照常粘贴
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while runner.committed_segments() == 0 && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        feed(&runner, 4_000, 700, 100.0); // 段 1：焦点已变，必须拦下
        runner.finalize();
        runner.wait();

        assert_eq!(*pastes.lock().unwrap(), vec!["t0".to_string()]);
        assert_eq!(runner.committed_text(), "t0");
        assert_eq!(runner.held_segments(), 1);
        assert_eq!(runner.held_text(), "t1", "被拦下的文本必须保留");
        let failures = failures.lock().unwrap();
        assert_eq!(failures.len(), 1, "守卫拦截只 toast 一次");
        assert!(failures[0].1.contains("held unpasted"));
    }
}
