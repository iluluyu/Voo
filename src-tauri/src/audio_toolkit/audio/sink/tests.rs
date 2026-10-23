use super::{AudioFrameCallback, CaptureSink, CaptureVad, LevelCallback, VadPolicy};
use crate::audio_toolkit::vad::{VadFrame, VoiceActivityDetector};
use handy_recorder::{AudioChunk, Sink};
use std::sync::{mpsc, Arc, Mutex};

/// Keeps frames whose first sample is non-zero, and records how the sink
/// configured it.
#[derive(Default)]
struct ScriptedVad {
    frame_samples: usize,
    hangover: Arc<Mutex<Option<usize>>>,
    resets: Arc<Mutex<usize>>,
}

impl VoiceActivityDetector for ScriptedVad {
    fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> anyhow::Result<VadFrame<'a>> {
        Ok(if frame[0] != 0.0 {
            VadFrame::Speech(frame)
        } else {
            VadFrame::Noise
        })
    }

    fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    fn set_hangover_frames(&mut self, frames: usize) {
        *self.hangover.lock().unwrap() = Some(frames);
    }

    fn reset(&mut self) {
        *self.resets.lock().unwrap() += 1;
    }
}

const FRAME: usize = 4;

struct Harness {
    sink: CaptureSink,
    ready: mpsc::Receiver<()>,
    fed: Arc<Mutex<Vec<f32>>>,
    hangover: Arc<Mutex<Option<usize>>>,
    resets: Arc<Mutex<usize>>,
}

fn harness(policy: VadPolicy) -> Harness {
    harness_with_leveling(policy, false)
}

fn harness_with_leveling(policy: VadPolicy, level_normalization: bool) -> Harness {
    let vad = ScriptedVad {
        frame_samples: FRAME,
        ..Default::default()
    };
    let hangover = Arc::clone(&vad.hangover);
    let resets = Arc::clone(&vad.resets);
    let fed = Arc::new(Mutex::new(Vec::new()));
    let on_audio: AudioFrameCallback = {
        let fed = Arc::clone(&fed);
        Arc::new(move |frame: &[f32]| fed.lock().unwrap().extend_from_slice(frame))
    };
    let on_levels: LevelCallback = Arc::new(|_| {});
    let (ready_tx, ready) = mpsc::channel();
    let sink = CaptureSink::new(
        CaptureVad::new(Box::new(vad), 10, 20),
        policy,
        on_levels,
        on_audio,
        ready_tx,
        level_normalization,
    );
    Harness {
        sink,
        ready,
        fed,
        hangover,
        resets,
    }
}

fn chunk(samples: &[f32], valid_frames: usize) -> AudioChunk<'_> {
    AudioChunk {
        samples,
        sample_rate: 16_000,
        channels: 1,
        valid_frames,
    }
}

/// 16 kHz 下毫秒 → 样本数。
fn ms(millis: usize) -> usize {
    millis * 16
}

/// 逐块（FRAME 样本）哦入「值 × 毫秒」脚本；非 0 值被 ScriptedVad 判为
/// 语音，0 值判为静音。
fn feed_script(sink: &mut CaptureSink, phases: &[(f32, usize)]) {
    for &(value, millis) in phases {
        let samples = vec![value; ms(millis)];
        for piece in samples.chunks(FRAME) {
            sink.process_chunk(chunk(piece, piece.len()));
        }
    }
}

/// 相邻同值样本合并成 (值, 长度) 游程，方便断言区间的精确边界。
fn runs(samples: &[f32]) -> Vec<(f32, usize)> {
    let mut out: Vec<(f32, usize)> = Vec::new();
    for &s in samples {
        match out.last_mut() {
            Some(last) if last.0 == s => last.1 += 1,
            _ => out.push((s, 1)),
        }
    }
    out
}

#[test]
fn readiness_fires_once_on_the_first_chunk() {
    let mut h = harness(VadPolicy::Disabled);
    assert!(h.ready.try_recv().is_err());

    h.sink.process_chunk(chunk(&[0.0; FRAME], FRAME));
    h.sink.process_chunk(chunk(&[0.0; FRAME], FRAME));

    assert!(h.ready.try_recv().is_ok());
    // The sender is gone after the first chunk, so a second signal can't come.
    assert_eq!(h.ready.try_recv(), Err(mpsc::TryRecvError::Disconnected));
}

#[test]
fn readiness_waiter_wakes_when_a_recording_ends_without_audio() {
    let h = harness(VadPolicy::Offline);
    let _ = h.sink.finish();
    assert!(h.ready.recv().is_err());
}

#[test]
fn disabled_vad_keeps_every_real_sample_and_drops_padding() {
    let mut h = harness(VadPolicy::Disabled);
    h.sink.process_chunk(chunk(&[1.0, 0.0, 2.0, 0.0], FRAME));
    // Final chunk: one real frame, three of zero padding.
    h.sink.process_chunk(chunk(&[3.0, 0.0, 0.0, 0.0], 1));

    let expected = vec![1.0, 0.0, 2.0, 0.0, 3.0];
    assert_eq!(*h.fed.lock().unwrap(), expected);
    assert_eq!(h.sink.finish().samples, expected);
    assert_eq!(
        *h.resets.lock().unwrap(),
        0,
        "a disabled policy leaves VAD alone"
    );
}

#[test]
fn streaming_keeps_speech_frames_only() {
    let mut h = harness(VadPolicy::Streaming);
    h.sink.process_chunk(chunk(&[1.0; FRAME], FRAME));
    h.sink.process_chunk(chunk(&[0.0; FRAME], FRAME));
    h.sink.process_chunk(chunk(&[2.0; FRAME], FRAME));

    // 逐字节等于旧逐帧行为：只有 Speech 帧，静音帧丢弃。流式模型依赖
    // 这套逐帧语义，Offline 改为区间保留后本测试钉住它不变。
    let expected = [[1.0; FRAME], [2.0; FRAME]].concat();
    assert_eq!(*h.fed.lock().unwrap(), expected);
    assert_eq!(h.sink.finish().samples, expected);
}

#[test]
fn offline_keeps_a_within_span_pause_between_two_frames() {
    let mut h = harness(VadPolicy::Offline);
    h.sink.process_chunk(chunk(&[1.0; FRAME], FRAME));
    h.sink.process_chunk(chunk(&[0.0; FRAME], FRAME));
    h.sink.process_chunk(chunk(&[2.0; FRAME], FRAME));

    // 4 样本停顿远未达断开长度：作为语句内短停顿整个保留，不再被抽走。
    let captured = h.sink.finish();
    assert_eq!(runs(&captured.samples), vec![(1.0, 4), (0.0, 4), (2.0, 4)]);
    assert_eq!(*h.fed.lock().unwrap(), captured.samples);
}

#[test]
fn offline_span_keeps_pre_roll_pause_and_tail_as_one_continuous_span() {
    let mut h = harness(VadPolicy::Offline);
    feed_script(
        &mut h.sink,
        &[
            (0.0, 600),   // 前导只能留最后 200 ms
            (1.0, 100),   // 语句开始
            (0.0, 400),   // 语句内短停顿（< 800 ms，整个保留）
            (2.0, 100),   // 语句继续
            (0.0, 1_000), // 静音 ≥ 800 ms：断开，留 300 ms 尾随
        ],
    );
    let captured = h.sink.finish();
    // 连续区间 = 前导 200 + 语句 + 短停顿 + 语句 + 尾随 300（毫秒 ×16）。
    assert_eq!(
        runs(&captured.samples),
        vec![
            (0.0, 3_200),
            (1.0, 1_600),
            (0.0, 6_400),
            (2.0, 1_600),
            (0.0, 4_800),
        ]
    );
    // on_audio 与录音缓冲看到同一批字节、同一顺序。
    assert_eq!(*h.fed.lock().unwrap(), captured.samples);
}

#[test]
fn offline_long_silence_splits_two_spans_and_drops_the_deep_middle() {
    let mut h = harness(VadPolicy::Offline);
    feed_script(&mut h.sink, &[(1.0, 100), (0.0, 1_500), (2.0, 100)]);
    let captured = h.sink.finish();
    // 1500 ms 句间静音只保留尾随 300 + 前导 200 = 500 ms；深处静音
    // 丢弃，两个区间各自独立、不重叠。
    assert_eq!(
        runs(&captured.samples),
        vec![(1.0, 1_600), (0.0, 8_000), (2.0, 1_600)]
    );
    assert_eq!(*h.fed.lock().unwrap(), captured.samples);
}

#[test]
fn offline_finish_keeps_the_pending_pause_up_to_the_tail() {
    let mut h = harness(VadPolicy::Offline);
    feed_script(&mut h.sink, &[(1.0, 100), (0.0, 200)]);
    let captured = h.sink.finish();
    // 停止时停顿未达断开长度：按尾随规则收尾（≤ 300 ms → 全留），
    // 已进入区间的音频一个不丢。
    assert_eq!(runs(&captured.samples), vec![(1.0, 1_600), (0.0, 3_200)]);
    assert_eq!(*h.fed.lock().unwrap(), captured.samples);
}

#[test]
fn offline_finish_during_speech_adds_no_tail() {
    let mut h = harness(VadPolicy::Offline);
    feed_script(&mut h.sink, &[(1.0, 100)]);
    let captured = h.sink.finish();
    assert_eq!(runs(&captured.samples), vec![(1.0, 1_600)]);
}

#[test]
fn offline_span_cap_never_drops_continuous_speech() {
    let mut h = harness(VadPolicy::Offline);
    // 40 s 连续语音（触及区间上限）：全部保留，上限只重置记账。
    feed_script(&mut h.sink, &[(1.0, 40_000)]);
    let captured = h.sink.finish();
    assert_eq!(captured.samples.len(), ms(40_000));
    assert!(captured.samples.iter().all(|&s| s == 1.0));
    assert_eq!(h.fed.lock().unwrap().len(), ms(40_000));
}

#[test]
fn offline_span_flushes_do_not_disturb_the_verdict_tap() {
    let mut h = harness(VadPolicy::Offline);
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(bool, usize, f32)>::new()));
    {
        let seen = std::sync::Arc::clone(&seen);
        h.sink
            .set_vad_verdict_callback(std::sync::Arc::new(move |speech, frame| {
                seen.lock().unwrap().push((speech, frame.len(), frame[0]));
            }));
    }
    feed_script(
        &mut h.sink,
        &[(0.0, 100), (1.0, 100), (0.0, 100), (2.0, 100)],
    );
    // 每个条件化块恰好一次、按序、带当时的判定——与区间状态机的前导
    // 回放/停顿补回互不影响（pause-commit 依赖这一契约）。
    let expected: Vec<(bool, usize, f32)> = [(false, 0.0), (true, 1.0), (false, 0.0), (true, 2.0)]
        .iter()
        .flat_map(|&(speech, value)| {
            std::iter::repeat((speech, FRAME, value)).take(ms(100) / FRAME)
        })
        .collect();
    assert_eq!(*seen.lock().unwrap(), expected);
}

#[test]
fn each_recording_resets_the_detector_with_its_policys_hangover() {
    let offline = harness(VadPolicy::Offline);
    assert_eq!(*offline.hangover.lock().unwrap(), Some(10));
    assert_eq!(*offline.resets.lock().unwrap(), 1);

    let streaming = harness(VadPolicy::Streaming);
    assert_eq!(*streaming.hangover.lock().unwrap(), Some(20));
    assert_eq!(*streaming.resets.lock().unwrap(), 1);
}

#[test]
fn finish_returns_the_detector_for_the_next_recording() {
    let h = harness(VadPolicy::Offline);
    let captured = h.sink.finish();
    assert_eq!(captured.vad.frame_samples(), FRAME);
    assert_eq!(captured.vad.hangover_frames_for(VadPolicy::Streaming), 20);
}

#[test]
fn leveling_runs_before_vad_and_reaches_the_recording() {
    // 注意：电平调理的时间常数是毫秒级（预热 300 ms、提升 ~1 s），所以这里
    // 用 100 ms 的真实块喂，而不是 VAD 测试用的 4 样本小块。
    let mut h = harness_with_leveling(VadPolicy::Disabled, true);
    let block = 1_600usize; // 100 ms @16k
    let speech: Vec<f32> = (0..block)
        .map(|i| 0.01 * ((i as f32) * 0.3).sin())
        .collect();
    let silence = vec![0.0f32; block];
    for _ in 0..20 {
        // 语音 100 ms + 停顿 300 ms：min-stat 在停顿里看到底噪。
        h.sink.process_chunk(AudioChunk {
            samples: &speech,
            sample_rate: 16_000,
            channels: 1,
            valid_frames: block,
        });
        for _ in 0..3 {
            h.sink.process_chunk(AudioChunk {
                samples: &silence,
                sample_rate: 16_000,
                channels: 1,
                valid_frames: block,
            });
        }
    }
    let fed = h.fed.lock().unwrap().clone();
    let last_voiced = fed
        .chunks(block)
        .filter(|c| c.iter().any(|s| s.abs() > 0.0))
        .last()
        .map(|c| c.to_vec())
        .expect("应至少有一帧非静音输出");
    let peak_in = speech.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    let peak_out = last_voiced.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    assert!(
        peak_out > peak_in * 1.5,
        "轻声应被电平调理放大：in={peak_in:.4} out={peak_out:.4}"
    );
    assert!(last_voiced.iter().all(|s| s.is_finite()));
}

#[test]
fn leveling_disabled_is_bit_exact_passthrough() {
    let mut h = harness(policy_disabled());
    let raw = [0.25f32, -0.5, 0.125, 0.0];
    h.sink.process_chunk(chunk(&raw, FRAME));
    assert_eq!(*h.fed.lock().unwrap(), raw.to_vec());
}

fn policy_disabled() -> VadPolicy {
    VadPolicy::Disabled
}
