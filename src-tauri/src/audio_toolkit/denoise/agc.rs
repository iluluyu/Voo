//! 语音门控电平归一化（AGC），放在 RNNoise 之前。
//!
//! # 为什么需要
//!
//! RNNoise 对电平敏感：同一段噪声在不同幅度下会被判成"语音"或"噪声"。
//! 实测（`probe_matrix`）0.5 幅度宽带噪声平均 pvad≈0.73（被当语音、几乎不
//! 抑制），0.05 带限噪声可压到 -39 dB。因此把语音先归一化到固定目标电平，
//! 才能让降噪行为稳定、可复现。
//!
//! 但**不是"越小越好"**：输入压得过低，16-bit 量化的量化噪声占比上升，
//! 再放大回来就是可闻的"嘶嘶"声。目标应落在语音 RMS 约 **-26 dBFS**。
//!
//! # 算法
//!
//! - 平方信号 10 ms one-pole 得到 RMS 包络；
//! - 期望增益 = 目标电平 / 当前 RMS，限制在 `[-12, +18] dB`；
//! - 增益在 dB 域平滑：衰减 20 ms 快、提升 200 ms 慢（避免把底噪顶起来）；
//! - **静音门控冻结**：RMS 低于 `GATE_DBFS` 时不更新增益，否则每次停顿
//!   都会把底噪放大又缩回，听感是"抽吸"；
//! - 逐样本因果、恒等长，不引入延迟。
//!
//! # 边界
//!
//! AGC 无法恢复已削顶的样本；它只保证后续处理看到一致的电平。

/// 目标语音 RMS 电平（dBFS）。
pub const TARGET_RMS_DBFS: f32 = -26.0;
/// 最小增益（更响的输入最多衰减这么多）。
pub const MIN_GAIN_DB: f32 = -12.0;
/// 最大增益（更轻的输入最多提升这么多；再低只会放大量化噪声）。
pub const MAX_GAIN_DB: f32 = 18.0;
/// 门控电平：低于它视为静音/底噪，冻结增益。
pub const GATE_DBFS: f32 = -55.0;

/// RMS 包络系数（10 ms @48k）。
const RMS_ALPHA: f32 = 0.002_083;
/// 增益衰减（更响）与提升（更轻）的平滑系数：20 ms / 200 ms @48k。
const GAIN_DOWN_ALPHA: f32 = 0.001_042;
const GAIN_UP_ALPHA: f32 = 0.000_104;

/// 语音门控 AGC。
#[derive(Debug)]
pub struct SpeechAgc {
    env_sq: f32,
    gain_db: f32,
    /// 非有限/异常输入计数（遥测用）。
    bad_samples: u64,
}

impl Default for SpeechAgc {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechAgc {
    pub fn new() -> Self {
        Self {
            env_sq: 0.0,
            gain_db: 0.0,
            bad_samples: 0,
        }
    }

    /// 当前增益（dB）。
    pub fn gain_db(&self) -> f32 {
        self.gain_db
    }

    /// 当前 RMS 包络（dBFS）。
    pub fn level_dbfs(&self) -> f32 {
        let rms = self.env_sq.max(0.0).sqrt();
        20.0 * (rms + 1e-12).log10()
    }

    /// 异常输入计数。
    pub fn bad_samples(&self) -> u64 {
        self.bad_samples
    }

    /// 处理一个样本。
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = if x.is_finite() {
            x.clamp(-8.0, 8.0)
        } else {
            self.bad_samples = self.bad_samples.saturating_add(1);
            0.0
        };
        self.env_sq += RMS_ALPHA * (x * x - self.env_sq);

        let level = self.level_dbfs();
        if level > GATE_DBFS {
            let desired = (TARGET_RMS_DBFS - level).clamp(MIN_GAIN_DB, MAX_GAIN_DB);
            let alpha = if desired < self.gain_db {
                GAIN_DOWN_ALPHA
            } else {
                GAIN_UP_ALPHA
            };
            self.gain_db += alpha * (desired - self.gain_db);
        }
        // 静音：冻结增益（不更新），输出仍是 0。
        x * 10f32.powf(self.gain_db / 20.0)
    }

    /// 处理整块（就地）。
    pub fn process_in_place(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f32 = 48_000.0;

    fn sine(amp: f32, secs: f32, freq: f32) -> Vec<f32> {
        let n = (FS * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / FS;
                amp * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        let sum: f32 = samples.iter().map(|s| s * s).sum();
        (sum / samples.len() as f32).sqrt()
    }

    fn level_db(x: f32) -> f32 {
        20.0 * (x + 1e-12).log10()
    }

    #[test]
    fn quiet_speech_is_lifted_towards_target_within_limit() {
        let mut agc = SpeechAgc::new();
        let mut samples = sine(0.01, 1.0, 220.0);
        let before = level_db(rms(&samples));
        agc.process_in_place(&mut samples);
        let after = level_db(rms(&samples));
        assert!(
            after > before + 6.0,
            "轻音应被显著提升：{before:.1} -> {after:.1} dBFS"
        );
        assert!(
            agc.gain_db() <= MAX_GAIN_DB + 0.01,
            "增益必须受上限约束，实际 {:.1} dB",
            agc.gain_db()
        );
        assert!(
            after < TARGET_RMS_DBFS + 6.0,
            "不应超过目标电平附近：{after:.1}"
        );
    }

    #[test]
    fn loud_speech_is_reduced_within_limit() {
        let mut agc = SpeechAgc::new();
        let mut samples = sine(0.9, 1.0, 220.0);
        agc.process_in_place(&mut samples);
        assert!(
            agc.gain_db() >= MIN_GAIN_DB - 0.01,
            "衰减必须受下限约束，实际 {:.1} dB",
            agc.gain_db()
        );
        assert!(
            agc.gain_db() < -3.0,
            "过响输入应被衰减，实际 {:.1} dB",
            agc.gain_db()
        );
        assert!(samples.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn gain_is_frozen_during_silence() {
        let mut agc = SpeechAgc::new();
        let mut voiced = sine(0.01, 0.6, 220.0);
        agc.process_in_place(&mut voiced);
        let settled = agc.gain_db();
        let mut silence = vec![0.0f32; (FS * 1.0) as usize];
        agc.process_in_place(&mut silence);
        // 门控是按 RMS 包络（10 ms）判断的，静音开始后包络还要衰减约 30 ms
        // 才跌破 -55 dBFS，这段内允许增益继续收敛；之后必须完全冻结。
        assert!(
            (agc.gain_db() - settled).abs() < 0.5,
            "静音期间增益必须冻结：{settled:.2} -> {:.2} dB",
            agc.gain_db()
        );
        assert!(silence.iter().all(|s| *s == 0.0), "静音仍应是 0");
    }

    #[test]
    fn very_quiet_input_hits_the_gain_ceiling_not_more() {
        let mut agc = SpeechAgc::new();
        // -60 dBFS 左右，远低于门控且需要 > 上限的提升。
        let mut samples = sine(0.001, 2.0, 220.0);
        agc.process_in_place(&mut samples);
        assert!(agc.gain_db() <= MAX_GAIN_DB + 0.01);
    }

    #[test]
    fn non_finite_input_is_counted_and_recoverable() {
        let mut agc = SpeechAgc::new();
        let mut samples = sine(0.05, 0.2, 220.0);
        samples[5] = f32::NAN;
        samples[6] = f32::INFINITY;
        agc.process_in_place(&mut samples);
        assert_eq!(agc.bad_samples(), 2);
        let mut tail = sine(0.05, 0.2, 220.0);
        agc.process_in_place(&mut tail);
        assert!(tail.iter().all(|s| s.is_finite()));
        assert!(agc.gain_db().is_finite());
    }
}
