use std::sync::{mpsc, Arc};

use handy_recorder::{AudioChunk, Sink};

use crate::audio_toolkit::{
    audio::AudioVisualiser,
    constants,
    level::InputLeveler,
    vad::{VadFrame, VoiceActivityDetector},
};
use log::debug;

mod utterance_span;

use self::utterance_span::{OfflineUtteranceSpan, SpanDecision};

/// How 16 kHz mono frames should be filtered for one recording session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VadPolicy {
    /// Bypass VAD and forward every frame.
    Disabled,
    /// Batch profile: the recording keeps continuous utterance spans —
    /// pre-roll, intra-span pauses and tails included — rather than
    /// per-frame speech only (see `utterance_span.rs` and the
    /// `OFFLINE_SPAN_*` constants below).
    Offline,
    /// VAD profile with a longer post-speech tail for streaming-capable models.
    Streaming,
}

/// A single VAD engine plus the two hangover-tail lengths its smoothing wrapper
/// should use. The offline and streaming policies are never active
/// concurrently, so one detector is reconfigured per recording rather than
/// kept as two resident engines.
pub struct CaptureVad {
    detector: Box<dyn VoiceActivityDetector>,
    offline_hangover_frames: usize,
    streaming_hangover_frames: usize,
}

impl CaptureVad {
    pub fn new(
        detector: Box<dyn VoiceActivityDetector>,
        offline_hangover_frames: usize,
        streaming_hangover_frames: usize,
    ) -> Self {
        assert!(
            detector.frame_samples() > 0,
            "VAD frame size must be non-zero"
        );
        Self {
            detector,
            offline_hangover_frames,
            streaming_hangover_frames,
        }
    }

    /// Samples per detector frame; the recorder delivers chunks of this size.
    pub fn frame_samples(&self) -> usize {
        self.detector.frame_samples()
    }

    /// Post-speech hangover tail (in detector frames) for the given policy —
    /// what [`CaptureSink::new`] applies to the detector, and therefore how
    /// long the verdict tap keeps reporting speech after real speech stops.
    /// `Disabled` never reaches the detector, so it maps to the offline value.
    pub fn hangover_frames_for(&self, policy: VadPolicy) -> usize {
        match policy {
            VadPolicy::Streaming => self.streaming_hangover_frames,
            VadPolicy::Offline | VadPolicy::Disabled => self.offline_hangover_frames,
        }
    }
}

/// Receives spectrum levels for the recording overlay.
pub type LevelCallback = Arc<dyn Fn(Vec<f32>) + Send + Sync + 'static>;
/// Receives each 16 kHz mono frame that passes the recording's VAD policy, in
/// order. Used to feed live streaming transcription as audio arrives. Under
/// `Offline` the passed stream is the continuous utterance spans — pre-roll,
/// intra-span pauses and tails included — so a live consumer sees exactly the
/// bytes the recording buffer keeps.
pub type AudioFrameCallback = Arc<dyn Fn(&[f32]) + Send + Sync + 'static>;
/// Receives every conditioned 16 kHz mono chunk with the recording's VAD
/// verdict for it (`true` = speech), **including the chunks the policy drops
/// from the recording**, once per chunk in capture order. Pause-commit
/// segmentation uses it: it needs silence timing the recording buffer no
/// longer carries. The tap sees the *current* chunk, never a `Speech`
/// aggregate's pre-roll replay, so every conditioned sample passes it exactly
/// once and absolute sample positions stay exact. Runs on the recorder's
/// delivery thread: it must be cheap and never block.
///
/// The chunk carries exactly the block's **valid** samples: a final chunk may
/// be shorter than one VAD frame (`valid_frames < frame_samples`; the input
/// buffer's zero padding is cropped, and the detector's length check then
/// fails over to a `Speech` verdict so the tail is never dropped). Consumers
/// must account sample positions and silence by the actual chunk length,
/// never a constant frame size (see `PauseCommitEngine::push_chunk`).
///
/// The verdict is the detector's **smoothed** output: during the post-speech
/// hangover tail (`CaptureVad::hangover_frames_for`) real silence is still
/// reported as speech. Consumers that need real-silence timing must credit
/// that hangover (see `PauseCommitConfig::hangover_ms`) instead of counting
/// silence from scratch, or every pause costs hangover + their own threshold.
pub type VadVerdictCallback = Arc<dyn Fn(bool, &[f32]) + Send + Sync + 'static>;

/// What a finished recording hands back.
pub struct CapturedAudio {
    /// 16 kHz mono samples that passed the VAD policy.
    pub samples: Vec<f32>,
    /// The detector the recording borrowed, for the next one.
    pub vad: CaptureVad,
}

/// One recording's consumer of 16 kHz mono audio from handy-recorder. Runs on
/// the recorder's delivery thread, never the real-time audio thread, so it
/// can run VAD inline. Chunks are exactly one VAD frame long.
pub struct CaptureSink {
    vad: CaptureVad,
    policy: VadPolicy,
    visualizer: AudioVisualiser,
    on_levels: LevelCallback,
    on_audio: AudioFrameCallback,
    /// Optional per-chunk VAD verdict tap (pause-commit segmentation).
    /// `None` — the default — keeps `process_chunk` byte-identical to the
    /// pre-tap path.
    on_vad_verdict: Option<VadVerdictCallback>,
    /// Signalled on the first chunk: the device is delivering audio. Silence
    /// counts; readiness is not speech.
    ready: Option<mpsc::Sender<()>>,
    samples: Vec<f32>,
    /// Offline 区间保留状态机。其它策略下从不推进，只占着空结构。
    span: OfflineUtteranceSpan,
    /// 输入电平调理：**必须作用在送进 VAD 的同一份样本上**，否则轻声
    /// 会先被 VAD 当噪声丢掉，增益再大也救不回来。
    leveler: Option<InputLeveler>,
}

/// Spectrum window at 16 kHz: 32 ms, near the overlay's ~30 Hz refresh.
const VISUALIZER_WINDOW: usize = 512;
const VISUALIZER_BUCKETS: usize = 16;

/* ───────── 批处理（Offline）连续语句保留策略 ─────────
 *
 * sol 结论的实验起点（不是模型要求）：逐帧拼接对批处理 ASR 是破坏性
 * 裁剪——语句内短停顿被抽走、词首/词尾可能被截断；应保留连续语句
 * 跨度。Streaming 依赖逐帧语义、Disabled 全保留，都不使用这些常量。
 * 实现在 `utterance_span.rs`；毫秒均指真实静音（hangover 已折算）。
 */

/// 区间开始前额外保留的真实前导音频。词首的弱起音/气息发生在 VAD
/// 确认语音（onset 需要连续有声帧）之前，200 ms 足以补上被截断的
/// 词首，又不引入大片环境噪声。Silero 平滑层自带的 ~450 ms prefill
/// 不再重复保留：前导由区间状态机统一管理，行为与 VAD 后端无关。
const OFFLINE_SPAN_PREROLL_MS: u64 = 200;

/// 区间结束时额外保留的真实尾随音频。词尾送气/弱尾音在 VAD 判静后
/// 仍有能量，300 ms 避免截断词尾，也不至于拖着长静音稀释 ASR 上下文。
const OFFLINE_SPAN_TAIL_MS: u64 = 300;

/// 连续静音达到该长度才判定语句结束、关闭区间。自然口述中句内停顿
/// （换气、逗号级）多在 200–600 ms；≥800 ms 基本是句间停顿，保留它
/// 只增加无信息静音。该值同时必须 ≥ 前导 + 尾随，否则相邻两个区间
/// 的保留会在静音里重叠（构造处有断言）。
const OFFLINE_SPAN_BREAK_SILENCE_MS: u64 = 800;

/// 单个区间的保留上限，与长音频切块策略（chunking.rs 的
/// `MAX_CHUNK_SAMPLES` = 40 s，"most models' accuracy degrades past
/// 30-40 s"）同一预算：一个区间拖入的音频不超过现有长音频上限。
/// 触顶只重置区间记账并继续无缝保留，绝不丢连续语音；整个录音缓冲
/// 的增长仍以 Disabled 策略为上界（连续口述本来就是全保留）。
const OFFLINE_SPAN_MAX_MS: u64 = 40_000;

/// 毫秒 → 16 kHz 样本数。这些都是“约”值，向下取整足够。
const fn offline_span_samples(ms: u64) -> usize {
    (ms * constants::WHISPER_SAMPLE_RATE as u64 / 1000) as usize
}

impl CaptureSink {
    /// A sink for one recording. Resets the detector and applies the policy's
    /// hangover tail, so no state carries over from the previous recording.
    pub fn new(
        mut vad: CaptureVad,
        policy: VadPolicy,
        on_levels: LevelCallback,
        on_audio: AudioFrameCallback,
        ready: mpsc::Sender<()>,
        level_normalization: bool,
    ) -> Self {
        // 平滑判定的 hangover 折算成真实静音信用，交给区间状态机，
        // 让 OFFLINE_SPAN_* 常量始终指真实毫秒（与 PauseCommitConfig
        // ::hangover_ms 同一约定，见 utterance_span.rs 模块文档）。
        let span = OfflineUtteranceSpan::new(
            offline_span_samples(OFFLINE_SPAN_PREROLL_MS),
            offline_span_samples(OFFLINE_SPAN_TAIL_MS),
            offline_span_samples(OFFLINE_SPAN_BREAK_SILENCE_MS),
            offline_span_samples(OFFLINE_SPAN_MAX_MS),
            vad.hangover_frames_for(VadPolicy::Offline) * vad.frame_samples(),
        );
        if policy != VadPolicy::Disabled {
            let hangover = vad.hangover_frames_for(policy);
            vad.detector.set_hangover_frames(hangover);
            vad.detector.reset();
        }
        Self {
            vad,
            policy,
            visualizer: AudioVisualiser::new(
                constants::WHISPER_SAMPLE_RATE,
                VISUALIZER_WINDOW,
                VISUALIZER_BUCKETS,
                400.0,
                4000.0,
            ),
            on_levels,
            on_audio,
            on_vad_verdict: None,
            ready: Some(ready),
            samples: Vec::new(),
            span,
            leveler: if level_normalization {
                Some(InputLeveler::with_capture_rate(
                    constants::WHISPER_SAMPLE_RATE,
                ))
            } else {
                None
            },
        }
    }

    /// The detector, from a sink that never recorded (its start failed).
    pub fn into_vad(self) -> CaptureVad {
        self.vad
    }

    /// Installs the per-chunk VAD verdict tap ([`VadVerdictCallback`]).
    /// Call before the sink is handed to the recorder; the tap runs on the
    /// recorder's delivery thread and must not block. With no tap installed
    /// the sink behaves exactly as before this hook existed.
    pub fn set_vad_verdict_callback(&mut self, on_vad_verdict: VadVerdictCallback) {
        self.on_vad_verdict = Some(on_vad_verdict);
    }

    /// Hands back the recording and the detector. Logs what VAD still held
    /// back when the recording stopped.
    pub fn finish(mut self) -> CapturedAudio {
        // Diagnostic for audio still withheld when capture stopped; it is not
        // conclusive in either direction.
        if self.policy != VadPolicy::Disabled {
            if let Some(report) = self.vad.detector.tail_report() {
                log::debug!(
                    "VAD at stop: withheld tail {} frames (~{}ms, {} voiced), in_speech={}, onset_counter={}, hangover_counter={}",
                    report.withheld_frames,
                    report.withheld_frames * self.vad.frame_samples() * 1000
                        / constants::WHISPER_SAMPLE_RATE as usize,
                    report.withheld_voiced_frames,
                    report.in_speech,
                    report.onset_counter,
                    report.hangover_counter
                );
            }
        }
        if let Some(leveler) = self.leveler.as_ref() {
            debug!(
                "Input leveler at stop: gain {:.1} dB, floor {:.1} dBFS, clipped samples {}, bad samples {}",
                leveler.gain_db(),
                leveler.noise_floor_dbfs(),
                leveler.clipped_samples(),
                leveler.bad_samples()
            );
        }
        // 批处理区间在停止时收尾：未决停顿截成尾随保留，不丢已进区间
        // 的音频；语句仍在进行时缓冲本就以最后一块结尾，无可补尾。
        if self.policy == VadPolicy::Offline {
            let tail = self.span.finish();
            if !tail.is_empty() {
                Self::emit(&mut self.samples, &self.on_audio, &tail);
            }
        }
        CapturedAudio {
            samples: self.samples,
            vad: self.vad,
        }
    }

    fn emit(samples: &mut Vec<f32>, on_audio: &AudioFrameCallback, frame: &[f32]) {
        samples.extend_from_slice(frame);
        on_audio(frame);
    }
}

impl Sink for CaptureSink {
    fn process_chunk(&mut self, chunk: AudioChunk<'_>) {
        if let Some(ready) = self.ready.take() {
            let _ = ready.send(());
        }

        // Mono by construction: the recorder mixes down or picks one channel.
        let real = &chunk.samples[..chunk.valid_frames];
        if let Some(levels) = self.visualizer.feed(real) {
            (self.on_levels)(levels);
        }

        // 条件化（电平调理 + 限幅）后再送 VAD 与录音缓冲：两处必须看到
        // 完全相同的样本，包括 prefill 帧。
        // 只复制真实样本：尾部零填充不能进入录音（会被当成真实静音）。
        let mut conditioned: Vec<f32> = Vec::with_capacity(real.len());
        conditioned.extend_from_slice(real);
        if let Some(leveler) = self.leveler.as_mut() {
            leveler.process_in_place(&mut conditioned);
        }

        if self.policy == VadPolicy::Disabled {
            Self::emit(&mut self.samples, &self.on_audio, &conditioned);
            return;
        }

        // The detector needs whole frames; a final chunk shorter than one
        // frame fails its length check and falls back to `Speech`, so the tail
        // is never dropped (the tap accounts it by its real length — see
        // `VadVerdictCallback`). The tap gets the current chunk and its
        // verdict *before* the `Speech` aggregate replays pre-roll, so it sees
        // each conditioned sample exactly once.
        let verdict = self
            .vad
            .detector
            .push_frame(&conditioned)
            .unwrap_or(VadFrame::Speech(&conditioned));
        if let Some(on_vad_verdict) = self.on_vad_verdict.as_ref() {
            on_vad_verdict(verdict.is_speech(), &conditioned);
        }
        match self.policy {
            // 流式：逐帧语义一字不改，流式模型依赖现有时序。
            VadPolicy::Streaming => match verdict {
                VadFrame::Speech(frame) => Self::emit(&mut self.samples, &self.on_audio, frame),
                VadFrame::Noise => {}
            },
            // 批处理：保留连续语句跨度（前导 + 短停顿 + 尾随，见
            // utterance_span.rs）。检测器 Speech 聚合帧自带的 prefill
            // 回放不再进缓冲——前导由区间状态机统一管理，避免同一
            // 音频保留两遍；进入缓冲/回调的始终是本块的真实样本。
            VadPolicy::Offline => match self.span.push(verdict.is_speech(), &conditioned) {
                SpanDecision::Drop => {}
                SpanDecision::Keep => Self::emit(&mut self.samples, &self.on_audio, &conditioned),
                SpanDecision::Flush {
                    samples: pre,
                    keep_chunk,
                } => {
                    if !pre.is_empty() {
                        Self::emit(&mut self.samples, &self.on_audio, &pre);
                    }
                    if keep_chunk {
                        Self::emit(&mut self.samples, &self.on_audio, &conditioned);
                    }
                }
            },
            // Disabled 在上方已原样放行；不可达，防御性丢弃而非 panic
            // （录音回调线程绝不能 panic）。
            VadPolicy::Disabled => {}
        }
    }
}

#[cfg(test)]
mod tests;
