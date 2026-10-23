//! nnnoiseless 真实降噪后端（48 kHz、480 样本帧）。
//!
//! # 幅度缩放（显式、勿隐式）
//!
//! nnnoiseless 的公开 API 要求 f32 样本处于 **16-bit PCM 量程**
//! （`[-32768, 32767]`，见其 `process_frame` 文档），而本 crate 的降噪
//! 链路统一使用 `[-1,1]` 归一化信号。本后端在进出后端时做显式缩放：
//! `in × 32768 → nnnoiseless → out ÷ 32768`，见 [`PCM_SCALE`]。
//!
//! # 延迟（干湿路对齐的依据）
//!
//! nnnoiseless 的分析窗覆盖 `[上一帧, 当前帧]` 960 样本，`process_frame`
//! 第 k 次调用的输出对应第 k-1 帧输入（第一帧输出为淡入伪影，上游按
//! [`DenoiseBackend::latency_frames`] 丢弃）。即恒定 **1 帧 = 480 样本
//! = 10 ms** 算法延迟，湿路比干路晚一帧，由 stage 的干路 FIFO 对齐。
//!
//! # pvad
//!
//! `process_frame` 的返回值即该帧语音概率（0..1，模型 VAD 输出），
//! 原样透传给 FSM 特征（G1 门 / speech_like）。
//!
//! # 故障安全
//!
//! 任何一次输出帧或 pvad 含非有限值即置 [`NnnoiselessBackend::broken`]，
//! 此后该后端恒等直通（帧内容不动、pvad 为中性 0.5），stage 侧另有
//! 二次防护；绝不 panic、绝不把 NaN 放进输出流。

use super::{DenoiseBackend, FrameOutput, PLACEHOLDER_PVAD};

/// [-1,1] 归一化 ↔ 16-bit PCM 量程的显式缩放因子（2^15）。
pub const PCM_SCALE: f32 = 32_768.0;

/// nnnoiseless 后端（内置标准模型）。nnnoiseless::DenoiseState 未实现
/// Debug，这里手写实现以暴露诊断字段。
pub struct NnnoiselessBackend {
    state: Box<nnnoiseless::DenoiseState<'static>>,
    broken: bool,
    /// 已处理帧数（遥测/测试用）。
    frames: u64,
}

impl NnnoiselessBackend {
    /// 使用内置标准模型构造。
    pub fn new() -> Self {
        Self {
            state: nnnoiseless::DenoiseState::new(),
            broken: false,
            frames: 0,
        }
    }

    /// 后端是否已因非有限输出被旁路。
    pub fn broken(&self) -> bool {
        self.broken
    }

    /// 已处理帧数。
    pub fn frames_processed(&self) -> u64 {
        self.frames
    }
}

impl Default for NnnoiselessBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for NnnoiselessBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnnoiselessBackend")
            .field("broken", &self.broken)
            .field("frames", &self.frames)
            .finish()
    }
}

impl DenoiseBackend for NnnoiselessBackend {
    fn latency_frames(&self) -> usize {
        // 分析窗 960 样本（前帧+本帧），本帧输出对应上一帧输入。
        1
    }

    fn process_frame(&mut self, frame: &mut [f32; super::FRAME_SAMPLES]) -> FrameOutput {
        self.frames += 1;
        if self.broken {
            // 已旁路：恒等直通，pvad 中性。
            return FrameOutput::identity(PLACEHOLDER_PVAD as f32);
        }
        let mut input = [0.0f32; super::FRAME_SAMPLES];
        for (dst, &src) in input.iter_mut().zip(frame.iter()) {
            *dst = src * PCM_SCALE;
        }
        let mut output = [0.0f32; super::FRAME_SAMPLES];
        let pvad = self.state.process_frame(&mut output, &input);
        if !pvad.is_finite() || output.iter().any(|s| !s.is_finite()) {
            // 故障安全：置 broken、本帧起恒等直通（frame 内容保持 HP 原样）。
            self.broken = true;
            return FrameOutput::identity(PLACEHOLDER_PVAD as f32);
        }
        for (dst, &src) in frame.iter_mut().zip(output.iter()) {
            *dst = src / PCM_SCALE;
        }
        FrameOutput {
            pvad: pvad.clamp(0.0, 1.0),
            usable: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_toolkit::denoise::PassthroughBackend;

    /// 固定种子 LCG 白噪声（与 mod.rs 测试同实现，确定性）。
    struct Lcg(u32);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as f32 / 8_388_608.0) - 1.0
        }
    }

    #[test]
    fn backend_reports_one_frame_latency() {
        assert_eq!(NnnoiselessBackend::new().latency_frames(), 1);
        assert_eq!(PassthroughBackend.latency_frames(), 0);
    }

    /// 缩放往返：恒等输入 → 输出仍在合理量程且有限；pvad ∈ [0,1]。
    #[test]
    fn backend_output_finite_and_pvad_in_range() {
        let mut backend = NnnoiselessBackend::new();
        let mut frame = [0.0f32; super::super::FRAME_SAMPLES];
        let mut phase = 0.0f64;
        let mut rng = Lcg(42);
        for _ in 0..50 {
            for s in frame.iter_mut() {
                phase += 2.0 * std::f64::consts::PI * 440.0 / 48_000.0;
                *s = (0.3 * phase.sin()) as f32 + 0.05 * rng.next_f32();
            }
            let out = backend.process_frame(&mut frame);
            assert!(out.usable);
            assert!((0.0..=1.0).contains(&out.pvad));
            assert!(frame.iter().all(|s| s.is_finite()));
            assert!(frame.iter().all(|s| s.abs() <= 2.0));
        }
        assert!(!backend.broken());
    }

    /// 纯噪声多帧后，denoised 能量应显著低于输入（模型真正在降噪）。
    ///
    /// 契约依据探针 `probe_matrix`（可重现：`cargo test -- --ignored
    /// probe_matrix --nocapture`）：模型对**中幅度带限平稳噪声**抑制最强
    /// （0.05 带限 → 实测 -39dB）；对高幅度宽带噪声不保证抑制（0.2 白噪
    /// 噪实测仅 -1dB，pvad>0.5 被判语音）——那是模型属性，不是接线 bug。
    /// 这里测前者，断言 -12dB（实测 -39dB，留平台余量）。
    #[test]
    fn backend_attenuates_stationary_bandlimited_noise() {
        let mut backend = NnnoiselessBackend::new();
        let mut rng = Lcg(7);
        let mut lp = 0.0f32;
        let mut frame = [0.0f32; super::super::FRAME_SAMPLES];
        let mut in_energy = 0.0f64;
        let mut out_energy = 0.0f64;
        // 跳过前 50 帧（模型 warmup / 延迟帧）。
        for i in 0..200 {
            for s in frame.iter_mut() {
                let x = 0.05 * rng.next_f32();
                lp += 0.316 * (x - lp); // one-pole ~3.5kHz：带限平稳噪声
                *s = lp;
            }
            let keep: Vec<f32> = frame.to_vec();
            let out = backend.process_frame(&mut frame);
            assert!(out.usable);
            if i >= 50 {
                in_energy += keep.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>();
                out_energy += frame.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>();
            }
        }
        let ratio_db = 10.0 * (out_energy / in_energy).log10();
        assert!(
            ratio_db < -12.0,
            "带限平稳噪声应有明显衰减，实际 {ratio_db:.1} dB"
        );
    }
}

#[cfg(test)]
mod probe {
    use super::*;
    struct Lcg(u32);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as f32 / 8_388_608.0) - 1.0
        }
    }
    /// one-pole LP ~3.5kHz @48k：把白噪声整成“语音带内”噪声。
    fn bandlimited(rng: &mut Lcg, lp: &mut f32) -> f32 {
        let x = rng.next_f32();
        *lp += 0.316 * (x - *lp); // a≈0.316 → fc≈3.5k
        *lp
    }
    #[test]
    #[ignore = "诊断探针：cargo test -- --ignored probe_matrix --nocapture"]
    fn probe_matrix() {
        let names = ["white", "bandlim", "sine300", "sine+wn"];
        for amp in [0.02f32, 0.05, 0.2, 0.5] {
            for (kind, name) in names.iter().enumerate() {
                let mut backend = NnnoiselessBackend::new();
                let mut rng = Lcg(1234);
                let mut lp = 0.0f32;
                let mut phase = 0.0f64;
                let mut frame = [0.0f32; super::super::FRAME_SAMPLES];
                let (mut ie, mut oe, mut pv) = (0.0f64, 0.0f64, 0.0f64);
                let mut n = 0usize;
                for f in 0..200 {
                    for s in frame.iter_mut() {
                        *s = match kind {
                            0 => amp * rng.next_f32(),
                            1 => amp * bandlimited(&mut rng, &mut lp),
                            2 => {
                                phase += 2.0 * std::f64::consts::PI * 300.0 / 48_000.0;
                                (amp as f64 * phase.sin()) as f32
                            }
                            _ => {
                                phase += 2.0 * std::f64::consts::PI * 300.0 / 48_000.0;
                                (amp as f64 * 0.5 * phase.sin()) as f32 + amp * 0.5 * rng.next_f32()
                            }
                        };
                    }
                    let keep = frame;
                    let out = backend.process_frame(&mut frame);
                    if f >= 50 {
                        ie += keep.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
                        oe += frame.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
                        pv += out.pvad as f64;
                        n += 1;
                    }
                }
                let db = 10.0 * (oe / ie).log10();
                println!(
                    "amp={amp:<5} {name:<8} post50frames: {db:6.1} dB  mean_pvad={:.3}  frames={}",
                    pv / n as f64,
                    backend.frames_processed()
                );
            }
        }
    }
}
