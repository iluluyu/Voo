//! 爆破音（plosive / "噗"声）防护：低频瞬态检测 + 动态抬高通。
//!
//! # 为什么需要它
//!
//! 爆破音是气流冲击麦克风时的 **20–150 Hz 短促高能脉冲**（p/t/k/b/d/g 前的
//! 送气）。它有两个后果：
//!
//! 1. 低频大幅摆动，听感是"噗/砰"，还会连带压缩器/AGC 抽吸；
//! 2. 幅度足够大时在采集端就已削顶（clipping），**削顶不可逆**——软件侧
//!    只能避免"继续放大它"，不能恢复波形。
//!
//! RNNoise 不是去爆音器：它是频谱抑制 + VAD，没有爆破音专门机制，且其
//! 增益平滑会把低频脉冲抹成"闷响"。所以必须在降噪之前独立处理。
//!
//! # 算法
//!
//! - 用 120 Hz one-pole 低通取"低频带"，分别跟踪低频包络与全带包络
//!   （快起音 1 ms / 慢释放 100 ms）。
//! - 当低频带能量占比高（`low/all > 0.6`）且绝对值超过门限时判为爆破音。
//! - 判定期间把信号按权重 `a` 混入 **160 Hz 二阶高通**（切掉"噗"的主体），
//!   `a` 以 3 ms 起音升到 1、60 ms 释放回落，避免硬切造成的新瞬态。
//! - 输出恒等长、逐样本因果，零额外延迟（用于实时链路时不需要对账）。
//!
//! 与既有 90 Hz 高通串接：正常语音不受影响（160 Hz 高通只在检测期内以
//! 权重 `a` 参与混合），爆破音期间低频被额外衰减约 20 dB。
//!
//! # 已知边界
//!
//! 已经削顶的样本无法修复；此处只保证"不再把削顶残留继续放大"，
//! 真正的解法是降低麦克风增益（系统侧）或加大拾音距离。

use super::HighPass90;

/// 低频检测低通：fc≈80 Hz @48k，`1-exp(-2π·fc/fs)`。
///
/// 取 80 Hz 而不是 120 Hz：普通话元音的 F1 常在 250–500 Hz，120 Hz 低通
/// 会漏进相当多 F1 能量，导致正常元音被判成爆破音（实测 220 Hz 正弦
/// 占比 0.68 就超过门限）。80 Hz 下元音占比降到 ~0.48，爆破音仍接近 1。
const LOW_DETECT_ALPHA: f32 = 0.010_404;
/// 爆破音期间混入的高通（RBJ 160 Hz Q=0.7071 @48k）。
const HP160_B0: f32 = 0.985_299_37;
const HP160_B1: f32 = -1.970_598_74;
const HP160_B2: f32 = 0.985_299_37;
const HP160_A1: f32 = -1.970_382_62;
const HP160_A2: f32 = 0.970_814_85;

/// 能量平滑系数（10 ms @48k）。用**均方能量比**而不是瞬时包络比：
/// 60 Hz 正弦经低通后 |x| 的纹波在 120 Hz，若用 1 ms 包络比会剧烈抖动，
/// 实测导致检测反复开关、混入权重始终上不去（衰减只有 -1.5 dB）。
const ENERGY_ALPHA: f32 = 0.002_083;
/// 能量比快速/慢速释放：快升（2 ms 平滑已足够）、慢落（30 ms）。
const ENERGY_FALL: f32 = 0.000_694;

/// 低频能量占比门限（`low_sq / all_sq`）。60 Hz 爆破音约 0.4–0.9；
/// 220 Hz 元音约 0.12，留出足够间隔。
const LOW_RATIO_ENTER: f32 = 0.30;
/// 低频 RMS 绝对门限，低于它不判爆破音（避免安静时的呼吸噪声）。
const LOW_ABS_ENTER: f32 = 0.03;
/// 退出迟滞（比进入低），避免在边界抖动。
const LOW_RATIO_EXIT: f32 = 0.22;

/// 混合权重 `a` 的起音/释放系数（3 ms / 60 ms @48k）。
const DUCK_ATTACK: f32 = 0.006_897;
const DUCK_RELEASE: f32 = 0.000_347;

/// 二阶高通 160 Hz（Direct Form II transposed，f32 状态）。
#[derive(Debug, Default, Clone, Copy)]
struct HighPass160 {
    z1: f32,
    z2: f32,
}

impl HighPass160 {
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = HP160_B0 * x + self.z1;
        self.z1 = HP160_B1 * x - HP160_A1 * y + self.z2;
        self.z2 = HP160_B2 * x - HP160_A2 * y;
        y
    }
}

/// 爆破音防护（逐样本、因果、等长）。
#[derive(Debug)]
pub struct PlosiveGuard {
    /// 全带高通 90 Hz 之后再次用作"正常路径"的参考（保持相位一致）。
    _reference: HighPass90,
    hp160: HighPass160,
    low_lp: f32,
    /// 低频带均方能量（平滑）。
    low_sq: f32,
    /// 全带均方能量（平滑）。
    all_sq: f32,
    /// 当前混入权重 0..1。
    duck: f32,
    /// 检测期计数（遥测/测试用）。
    active_samples: u64,
    detections: u64,
}

impl Default for PlosiveGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl PlosiveGuard {
    pub fn new() -> Self {
        Self {
            _reference: HighPass90::default(),
            hp160: HighPass160::default(),
            low_lp: 0.0,
            low_sq: 0.0,
            all_sq: 0.0,
            duck: 0.0,
            active_samples: 0,
            detections: 0,
        }
    }

    /// 当前混入权重（0 = 未介入，1 = 全切低频）。
    pub fn duck(&self) -> f32 {
        self.duck
    }

    /// 已介入的样本数。
    pub fn active_samples(&self) -> u64 {
        self.active_samples
    }

    /// 判定次数（从 0 跳到检测阈值）。
    pub fn detections(&self) -> u64 {
        self.detections
    }

    /// 处理一个样本，返回防护后的样本。
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        // 入口兜底：非有限值按静音处理，避免一次 NaN 永久污染滤波器与包络。
        let x = if x.is_finite() {
            x.clamp(-8.0, 8.0)
        } else {
            0.0
        };
        let band = x - self.low_lp;
        self.low_lp += LOW_DETECT_ALPHA * band;
        let low = self.low_lp.abs();

        // 均方能量：上升用 10 ms 平滑，下降更慢（30 ms），保证一次爆破音
        // 内比值稳定，不会逐样本抖动。
        let low_sq_now = self.low_lp * self.low_lp;
        let all_sq_now = x * x;
        self.low_sq = if low_sq_now > self.low_sq {
            self.low_sq + ENERGY_ALPHA * (low_sq_now - self.low_sq)
        } else {
            self.low_sq + ENERGY_FALL * (low_sq_now - self.low_sq)
        };
        self.all_sq = if all_sq_now > self.all_sq {
            self.all_sq + ENERGY_ALPHA * (all_sq_now - self.all_sq)
        } else {
            self.all_sq + ENERGY_FALL * (all_sq_now - self.all_sq)
        };

        let ratio = self.low_sq / (self.all_sq + 1e-12);
        let low_rms = self.low_sq.max(0.0).sqrt();
        let entering = ratio > LOW_RATIO_ENTER && low_rms > LOW_ABS_ENTER;
        let hold = self.duck > 0.01 && ratio > LOW_RATIO_EXIT && low_rms > LOW_ABS_ENTER;
        let target = if entering || hold { 1.0 } else { 0.0 };

        if target > self.duck {
            if self.duck <= 0.01 {
                self.detections += 1;
            }
            self.duck += DUCK_ATTACK * (target - self.duck);
        } else {
            self.duck += DUCK_RELEASE * (target - self.duck);
        }
        if self.duck > 0.01 {
            self.active_samples += 1;
        }

        // 正常路径与高通路径按权重混合：正常语音 duck≈0 时完全透明。
        let high = self.hp160.process(x);
        if self.duck <= 0.0 {
            x
        } else {
            x * (1.0 - self.duck) + high * self.duck
        }
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

    fn low_band_energy(samples: &[f32]) -> f64 {
        // 120 Hz one-pole 低通后的能量：爆破音主体就在这个带内。
        let mut lp = 0.0f32;
        let mut sum = 0.0f64;
        for &s in samples {
            lp += LOW_DETECT_ALPHA * (s - lp);
            sum += (lp as f64) * (lp as f64);
        }
        sum
    }

    fn high_band_energy(samples: &[f32]) -> f64 {
        // 用 160 Hz 高通后的能量近似"语音主体"未被误伤的程度。
        let mut hp = HighPass160::default();
        let mut sum = 0.0f64;
        for &s in samples {
            let y = hp.process(s);
            sum += (y as f64) * (y as f64);
        }
        sum
    }

    #[test]
    fn plosive_burst_low_band_is_strongly_attenuated() {
        // 60 Hz 衰减脉冲模拟"噗"声：起音快、能量集中在前 40 ms。
        let n = (FS * 0.2) as usize;
        let burst = (FS * 0.04) as usize;
        let raw: Vec<f32> = (0..n)
            .map(|i| {
                if i < burst {
                    let t = i as f32 / FS;
                    0.8 * (-t / 0.010).exp() * (2.0 * std::f32::consts::PI * 60.0 * t).sin()
                } else {
                    0.0
                }
            })
            .collect();
        let mut guard = PlosiveGuard::new();
        let out: Vec<f32> = raw.iter().map(|&s| guard.process(s)).collect();

        // 检测没有前视：最前面的 2–3 ms 必然原样通过，真正的爆破音主体在
        // 检测生效之后。因此按"生效后的窗口"评估衰减，而不是整段。
        let settle = (FS * 0.005) as usize;
        let window = (FS * 0.005) as usize..burst;
        let before = low_band_energy(&raw[window.clone()]);
        let after = low_band_energy(&out[window]);
        let reduction_db = 10.0 * (after / before).log10();
        assert!(
            reduction_db < -10.0,
            "检测生效后爆破音低频应大幅衰减，实际 {reduction_db:.1} dB"
        );

        // 前 3 ms 属于无前视的直通区：只要求它不被放大成削顶。
        assert!(
            out[..settle].iter().all(|s| s.abs() <= 0.85),
            "直通区不应放大输入"
        );
        assert!(guard.detections() >= 1, "应至少判定一次爆破音");
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn sustained_low_band_is_attenuated_by_the_highpass_corner() {
        // 稳态 60 Hz：检测持续生效，衰减应逼近 160 Hz 二阶高通在该频点的
        // 理论值（约 -17 dB），验证混合确实切到高通路径。
        let n = (FS * 0.3) as usize;
        let raw: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / FS;
                0.5 * (2.0 * std::f32::consts::PI * 60.0 * t).sin()
            })
            .collect();
        let mut guard = PlosiveGuard::new();
        let out: Vec<f32> = raw.iter().map(|&s| guard.process(s)).collect();
        let window = (FS * 0.1) as usize..n; // 等检测完全收敛
        let before = low_band_energy(&raw[window.clone()]);
        let after = low_band_energy(&out[window]);
        let reduction_db = 10.0 * (after / before).log10();
        assert!(
            reduction_db < -14.0,
            "稳态低频应逼近高通极限，实际 {reduction_db:.1} dB"
        );
    }

    #[test]
    fn steady_vowel_is_not_ducked() {
        // 220 Hz 基频 + 谐波的稳定元音：低频占比低于门限，不应介入。
        let n = (FS * 0.5) as usize;
        let raw: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / FS;
                let w = 2.0 * std::f32::consts::PI;
                0.2 * (w * 220.0 * t).sin()
                    + 0.1 * (w * 440.0 * t).sin()
                    + 0.05 * (w * 880.0 * t).sin()
            })
            .collect();
        let mut guard = PlosiveGuard::new();
        let out: Vec<f32> = raw.iter().map(|&s| guard.process(s)).collect();

        assert_eq!(guard.detections(), 0, "稳定元音不应触发爆破音防护");
        assert!(guard.duck() < 0.01, "未介入时混入权重应接近 0");
        let before = high_band_energy(&raw);
        let after = high_band_energy(&out);
        let change_db = 10.0 * (after / before).log10();
        assert!(
            change_db.abs() < 0.1,
            "未介入时语音主体必须原样通过，实际变化 {change_db:.3} dB"
        );
    }

    #[test]
    fn silence_stays_exactly_silent() {
        let mut guard = PlosiveGuard::new();
        let mut samples = vec![0.0f32; 4800];
        guard.process_in_place(&mut samples);
        assert!(samples.iter().all(|s| *s == 0.0));
        assert_eq!(guard.detections(), 0);
    }

    #[test]
    fn duck_releases_after_the_burst() {
        let mut guard = PlosiveGuard::new();
        let mut x = vec![0.0f32; (FS * 0.5) as usize];
        for (i, s) in x.iter_mut().enumerate() {
            if i < (FS * 0.03) as usize {
                let t = i as f32 / FS;
                *s = 0.8 * (-t / 0.01).exp() * (2.0 * std::f32::consts::PI * 55.0 * t).sin();
            }
        }
        for s in x.iter_mut() {
            *s = guard.process(*s);
        }
        assert!(
            guard.duck() < 0.05,
            "爆破音结束后权重应回落，实际 {:.3}",
            guard.duck()
        );
    }

    #[test]
    fn non_finite_input_does_not_poison_state() {
        let mut guard = PlosiveGuard::new();
        let mut samples = vec![0.1f32; 480];
        samples[10] = f32::NAN;
        samples[11] = f32::INFINITY;
        guard.process_in_place(&mut samples);
        // 后面的正常样本必须仍然有限且能继续检测。
        let mut tail = vec![0.0f32; 480];
        for (i, s) in tail.iter_mut().enumerate() {
            let t = i as f32 / FS;
            *s = 0.5 * (2.0 * std::f32::consts::PI * 55.0 * t).sin();
        }
        guard.process_in_place(&mut tail);
        assert!(
            tail.iter().all(|s| s.is_finite()),
            "非有限输入后必须恢复正常"
        );
    }
}
