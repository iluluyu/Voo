//! 自适应降噪（算法提纲为评审定稿，参数照抄，勿自行改动）。
//!
//! # 信号链
//!
//! ```text
//! 输入 48k f32 任意长块
//!   → 高通 biquad 90 Hz / Q=0.707（恒开）
//!   → [特征提取 + DenoiseBackend 降噪（如 nnnoiseless，见 nn.rs）]
//!   → v1 直通模式（process）：输出即高通结果，零延迟；
//!     真实混合模式（process_mix/finish_mix）：输出 g·denoised+(1-g)·HP，
//!     干路由 FIFO 对齐后端延迟（latency_frames）
//! ```
//!
//! 真实后端见 [`nn::NnnoiselessBackend`]（[-1,1]↔16bit 显式缩放、pvad 返回、
//! 非有限安全旁路）。混合模式下：任意长块进 `process_mix`（返回当前可用
//! 输出，内容相对输入延迟 = 后端延迟），流结束调 `finish_mix` 取尾帧
//! （零 padding 上屏，总输出长度 == 总输入长度）。
//!
//! # 帧结构
//!
//! - 480 样本 = 10 ms = 一个 nnnoiseless 帧对齐（S1 能量、S2 噪声底按此更新）；
//! - 960 样本 = 20 ms = 一个 FSM/遥调 tick（S4 周期性、S5 R_lf、S7 σ_Rlf、
//!   状态机与 CSV 遥调按此更新）。
//!
//! 特征全部在高通后的直通信号上计算。
//!
//! # 参数总表（定稿）
//!
//! | 环节 | 参数 |
//! |------|------|
//! | 高通 | RBJ 90 Hz Q=0.707 @48k，b=[0.99171,-1.98342,0.99171]，a=[1,-1.98333,0.98347]（直接使用系数，不现场计算） |
//! | S1 | L=20·log10(sqrt(mean(x²))+1e-10)，Ls=0.2·L+0.8·Ls' |
//! | S2 | 200 帧环形缓冲，m=min+3dB；非语音帧 floor 每帧 clamp 在 [floor-0.10, floor+0.02]；语音帧只允许 +0.02 且禁止下调；speech_like = pvad 门 && SNR>3dB |
//! | S4 | 20ms 窗归一化自相关峰值，lag∈[96,768]，num/(sqrt(e0·eτ)+1e-12) |
//! | S5 | one-pole LP fc≈1.5k（y[i]=y[i-1]+0.1783·(x[i]-y[i-1])，0.1783 为归一化系数），20ms 窗 Σy²/Σx² |
//! | S7 | 最近 50 次 R_lf 的样本标准差（n-1） |
//! | FSM | S0 CALIB(g=0.45) / S1 DENOISE_LOW(g=0.45) / S2 DENOISE_HIGH(g=0.90) / S3 BREATHY_BYPASS(g=0) |
//! | 门  | G1 SNR>6dB && pvad>0.5；G2 p<0.25 进 / >0.35 出；G3 R_lf<0.45 进 / >0.55 出；G4 σ_Rlf>0.05 |
//! | 迟滞 | 进 10 tick(200ms) / 出 25 tick(500ms) / 驻留 300ms / 增益 150ms 线性 crossfade |
//! | S1↔S2 | SNR<6dB 持续 500ms 升 S2；SNR>12dB 持续 1s 降 S1 |
//! | 校准 | 累计非语音≥0.5s 或 10s 超时取 min+3dB；初始 floor=-45 dBFS |

use std::collections::VecDeque;

/// nnnoiseless 真实降噪后端（48k、[-1,1]↔16bit 缩放、pvad、安全旁路）。
pub mod agc;
pub mod nn;
pub mod plosive;

/// 模块工作采样率（48 kHz）。
pub const SAMPLE_RATE: u32 = 48_000;
/// nnnoiseless 帧长（10 ms @48k）。
pub const FRAME_SAMPLES: usize = 480;
/// FSM/遥调 tick 长度（20 ms @48k）。
pub const TICK_SAMPLES: usize = 960;
/// tick 周期（毫秒）。
pub const TICK_MS: u32 = 20;

// ---- S1 帧能量 ----
/// 能量公式底数保护：`20·log10(sqrt(mean(x²)) + 1e-10)`。
const ENERGY_EPSILON: f64 = 1e-10;
/// 平滑系数：Ls = 0.2·L + 0.8·Ls'。
const LS_SMOOTH_NEW: f64 = 0.2;

// ---- S2 噪声底 min-stat ----
/// Ls 环形缓冲长度（200 帧 × 10ms = 2s）。
pub const MINSTAT_FRAMES: usize = 200;
/// 噪声底估计 = 窗内最小 Ls + 3dB。
pub const MINSTAT_MARGIN_DB: f64 = 3.0;
/// 非语音帧每帧最大下调量（dB/帧）。
const FLOOR_DOWN_MAX_DB: f64 = 0.10;
/// 每帧最大上调量（dB/帧）。
const FLOOR_UP_MAX_DB: f64 = 0.02;
/// 初始噪声底（dBFS）。
pub const FLOOR_INITIAL_DB: f64 = -45.0;
/// speech_like 的 SNR 门（dB）。
pub const SPEECH_SNR_DB: f64 = 3.0;

// ---- S4 周期性 ----
/// 归一化自相关 lag 搜索范围（48Hz..500Hz 基频 @48k）。
pub const LAG_MIN: usize = 96;
pub const LAG_MAX: usize = 768;

// ---- S5 低频占比 ----
/// one-pole 低通系数（y[i]=y[i-1]+0.1783·(x[i]-y[i-1])，fc≈1.5 kHz @48k）。
const LP_ALPHA: f64 = 0.1783;
/// S7：σ_Rlf 历史长度（最近 50 次更新）。
pub const RLF_HISTORY: usize = 50;

// ---- 高通 biquad（RBJ 90 Hz / Q=0.707 @48k，提纲给定系数，勿现场计算） ----
const HP_B0: f64 = 0.99171;
const HP_B1: f64 = -1.98342;
const HP_B2: f64 = 0.99171;
const HP_A1: f64 = -1.98333;
const HP_A2: f64 = 0.98347;

// ---- FSM 门限（定稿参数） ----
/// G1：语音门 SNR>6dB（pvad 门见 [`pvad_gate`]）。
pub const G1_SNR_DB: f64 = 6.0;
/// G2：周期性进门（p<0.25 进 BREATHY_BYPASS）。
pub const G2_P_ENTER: f64 = 0.25;
/// G2：周期性出门（p>0.35 出）。
pub const G2_P_EXIT: f64 = 0.35;
/// G3：R_lf 进门（<0.45 进）。
pub const G3_RLF_ENTER: f64 = 0.45;
/// G3：R_lf 出门（>0.55 出）。
pub const G3_RLF_EXIT: f64 = 0.55;
/// G4：σ_Rlf>0.05。
pub const G4_SIGMA: f64 = 0.05;
/// 进入 BREATHY_BYPASS 迟滞：10 tick = 200ms。
pub const ENTER_TICKS: u32 = 10;
/// 退出 BREATHY_BYPASS 迟滞：25 tick = 500ms。
pub const EXIT_TICKS: u32 = 25;
/// 状态间最小驻留：300ms = 15 tick。
pub const DWELL_TICKS: u32 = 15;
/// 增益线性 crossfade 时长 150ms（= 7.5 tick，按 tick 线性步进）。
pub const CROSSFADE_MS: f64 = 150.0;
/// SNR<6dB 持续 500ms（25 tick）从 DENOISE_LOW 升 DENOISE_HIGH。
pub const RISE_HIGH_TICKS: u32 = 25;
/// SNR>12dB 持续 1s（50 tick）从 DENOISE_HIGH 降 DENOISE_LOW。
pub const FALL_LOW_TICKS: u32 = 50;
/// DENOISE_HIGH 降回 DENOISE_LOW 的 SNR 门（dB）。
pub const SNR_HIGH_EXIT_DB: f64 = 12.0;
/// 校准完成：累计非语音 ≥ 0.5s（25 tick）。
pub const CAL_NON_SPEECH_TICKS: u32 = 25;
/// 校准超时：10s（500 tick）。
pub const CAL_TIMEOUT_TICKS: u32 = 500;
/// 各状态目标增益 g（0=旁路，1=全抑制）。
pub const GAIN_CALIB: f64 = 0.45;
pub const GAIN_DENOISE_LOW: f64 = 0.45;
pub const GAIN_DENOISE_HIGH: f64 = 0.90;
pub const GAIN_BYPASS: f64 = 0.0;
/// v1 占位 pvad（真实后端接入前恒 0.5）。
pub const PLACEHOLDER_PVAD: f64 = 0.5;

/// 遥测行环形缓冲上限（调用方应定期 `drain_telemetry`；超限丢最旧行）。
const TELEMETRY_CAP: usize = 4096;

/// pvad 语音门。
///
/// 提纲门限为 `pvad > 0.5`；这里实现为 `pvad >= 0.5`。对真实后端
/// （pvad 为 (0,1) 连续值）二者行为一致（边界测度为零）；取 `>=` 是为了
/// 让 v1 的占位 pvad=0.5 恰好处于中性位——按提纲 "v1 无 nnnoiseless 时
/// pvad 恒 0.5 占位，SNR 门生效"，占位期间 speech_like / G1 仅由 SNR 门
/// 决定，而不是被 pvad 恒假锁死。
#[inline]
pub fn pvad_gate(pvad: f64) -> bool {
    pvad >= PLACEHOLDER_PVAD
}

/// speech_like = pvad 门 && SNR>3dB（S2 噪声底更新与校准计数共用）。
#[inline]
pub fn speech_like(pvad: f64, snr_db: f64) -> bool {
    pvad_gate(pvad) && snr_db > SPEECH_SNR_DB
}

// ============================================================================
// 降噪后端 trait
// ============================================================================

/// 后端单帧处理结果。
#[derive(Debug, Clone, Copy)]
pub struct FrameOutput {
    /// 本帧语音概率（喂 FSM 特征的 pvad；真实后端为模型 VAD 输出）。
    pub pvad: f32,
    /// 输出是否可用：`false` 表示后端故障（`frame` 内容无效），stage 将
    /// 干路直通旁路（绝不 panic、绝不让非有限值进入输出流）。
    pub usable: bool,
}

impl FrameOutput {
    /// 正常结果。
    pub fn ok(pvad: f32) -> Self {
        Self { pvad, usable: true }
    }

    /// 故障/恒等直通结果（stage 旁路湿路）。
    pub fn identity(pvad: f32) -> Self {
        Self {
            pvad,
            usable: false,
        }
    }
}

/// 降噪后端 trait：[`AdaptiveDenoiseStage`] 每个 480 样本帧调用一次。
pub trait DenoiseBackend: Send {
    /// 降噪输出相对输入帧的额外延迟（480 样本帧数）。nnnoiseless 的分析
    /// 窗覆盖 [前帧,本帧]，第 k 次调用的输出对应第 k-1 帧输入，故为 1。
    /// stage 据此对齐干湿路并决定丢弃的初始淡入帧数。
    fn latency_frames(&self) -> usize {
        0
    }

    /// 处理一个 480 样本帧（48 kHz mono，[-1,1]）：原位写入“本次调用产出
    /// 的降噪帧”（相对输入滞后 `latency_frames` 帧），返回 pvad。
    fn process_frame(&mut self, frame: &mut [f32; FRAME_SAMPLES]) -> FrameOutput;
}

/// v1 占位后端：恒等直通，pvad 恒 0.5。
#[derive(Debug, Default, Clone, Copy)]
pub struct PassthroughBackend;

impl DenoiseBackend for PassthroughBackend {
    fn process_frame(&mut self, _frame: &mut [f32; FRAME_SAMPLES]) -> FrameOutput {
        FrameOutput::ok(PLACEHOLDER_PVAD as f32)
    }
}

/// 实验链路总开关：`VOO_EXPERIMENTAL_DENOISE=1` 时启用真实降噪（默认关）。
/// 仅接受精确值 `1`，其余任何值/未设置均为关。
/// 【offline-only】当前仅离线工具链路存在；16k 录音接线落地时再随接线
/// 一起消费（默认关闭 = 生产路径零引用）。
pub fn experimental_denoise_enabled() -> bool {
    std::env::var("VOO_EXPERIMENTAL_DENOISE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

// ============================================================================
// 高通 biquad（90 Hz / Q=0.707，Direct Form II transposed，f64 状态）
// ============================================================================

/// 恒开高通滤波器。系数为提纲给定的 RBJ 90 Hz Q=0.707 @48k 值。
#[derive(Debug, Default, Clone, Copy)]
pub struct HighPass90 {
    z1: f64,
    z2: f64,
}

impl HighPass90 {
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let y = HP_B0 * x + self.z1;
        self.z1 = HP_B1 * x - HP_A1 * y + self.z2;
        self.z2 = HP_B2 * x - HP_A2 * y;
        y as f32
    }
}

// ============================================================================
// 特征提取
// ============================================================================

/// S1：帧能量 dBFS，`L = 20·log10(sqrt(mean(x²)) + 1e-10)`。
pub fn frame_energy_dbfs(frame: &[f32]) -> f64 {
    let sum: f64 = frame
        .iter()
        .map(|&s| {
            let s = s as f64;
            s * s
        })
        .sum();
    let mean = sum / frame.len() as f64;
    20.0 * (mean.sqrt() + ENERGY_EPSILON).log10()
}

/// S4：20ms 窗归一化自相关峰值。
///
/// `r(τ) = num / (sqrt(e0·eτ) + 1e-12)`，其中
/// `num = Σ_{i<W-τ} x[i]·x[i+τ]`，`e0 = Σ_{i<W-τ} x[i]²`，`eτ = Σ_{i≥τ} x[i]²`，
/// 取 lag∈[96,768] 内的最大值（ energies 用前缀和计算）。
pub fn periodicity_peak(window: &[f32]) -> f64 {
    let n = window.len();
    if n < LAG_MIN + 1 {
        return 0.0;
    }
    // 前缀能量：e[k] = Σ_{i<k} x[i]²
    let mut energy = vec![0.0f64; n + 1];
    for i in 0..n {
        let s = window[i] as f64;
        energy[i + 1] = energy[i] + s * s;
    }
    let total = energy[n];
    let mut best = 0.0f64;
    let lag_hi = LAG_MAX.min(n - 1);
    for lag in LAG_MIN..=lag_hi {
        let span = n - lag;
        let num: f64 = (0..span)
            .map(|i| window[i] as f64 * window[i + lag] as f64)
            .sum();
        let e0 = energy[span];
        let et = total - energy[lag];
        let r = num / ((e0 * et).sqrt() + 1e-12);
        if r > best {
            best = r;
        }
    }
    best.clamp(0.0, 1.0)
}

/// one-pole 低通（fc≈1.5 kHz @48k）。
///
/// 提纲写法 `y[i]=y[i-1]+0.1783·x[i]` 为简写；直流增益为 1 的标准形式是
/// `y[i] = y[i-1] + 0.1783·(x[i] - y[i-1])`（否则是积分器），按后者实现。
#[derive(Debug, Default, Clone, Copy)]
struct OnePoleLp {
    y: f64,
}

impl OnePoleLp {
    #[inline]
    fn process(&mut self, x: f32) -> f64 {
        self.y += LP_ALPHA * (x as f64 - self.y);
        self.y
    }
}

/// 样本标准差（n-1，Bessel 修正；不足 2 个样本时为 0）。
fn sample_std(values: &VecDeque<f64>) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// S5 + S7：低频占比 R_lf 与其波动性 σ_Rlf。
///
/// 逐样本 `push`（内部跑 one-pole 低通并累加 20ms 窗能量），每 960 样本
/// `tick` 一次得到 `(R_lf, σ_Rlf)`；σ_Rlf 是最近 [`RLF_HISTORY`] 次更新
/// 的样本标准差。
#[derive(Debug, Default)]
pub struct LowFreqRatio {
    lp: OnePoleLp,
    sum_x2: f64,
    sum_y2: f64,
    history: VecDeque<f64>,
}

impl LowFreqRatio {
    #[inline]
    pub fn push(&mut self, x: f32) {
        let y = self.lp.process(x);
        let xf = x as f64;
        self.sum_x2 += xf * xf;
        self.sum_y2 += y * y;
    }

    /// 结束当前 20ms 窗：返回 `(R_lf, σ_Rlf)` 并重置累加器。
    pub fn tick(&mut self) -> (f64, f64) {
        let r = (self.sum_y2 / (self.sum_x2 + 1e-12)).clamp(0.0, 1.0);
        self.sum_x2 = 0.0;
        self.sum_y2 = 0.0;
        self.history.push_back(r);
        while self.history.len() > RLF_HISTORY {
            self.history.pop_front();
        }
        let sigma = sample_std(&self.history);
        (r, sigma)
    }

    /// 最近一次窗口的 R_lf 值（无更新前为 0）。
    pub fn last_r_lf(&self) -> f64 {
        self.history.back().copied().unwrap_or(0.0)
    }
}

// ============================================================================
// S2：噪声底 min-stat 简化版
// ============================================================================

/// 噪声底跟踪器：Ls 的 200 帧环形缓冲，`m = min + 3dB`。
///
/// - 非语音帧：floor 每帧 clamp 在 `[floor-0.10, floor+0.02]`（即每帧最多
///   下调 0.10 dB、上调 0.02 dB，向 m 逼近）；
/// - 语音帧：只允许 +0.02 且禁止下调（delta clamp 在 `[0, +0.02]`）。
#[derive(Debug, Clone)]
pub struct NoiseFloorTracker {
    ring: Vec<f64>,
    head: usize,
    count: usize,
    floor: f64,
}

impl Default for NoiseFloorTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseFloorTracker {
    pub fn new() -> Self {
        Self {
            ring: vec![0.0; MINSTAT_FRAMES],
            head: 0,
            count: 0,
            floor: FLOOR_INITIAL_DB,
        }
    }

    fn min_ls(&self) -> f64 {
        if self.count == 0 {
            return FLOOR_INITIAL_DB;
        }
        let valid = if self.count < MINSTAT_FRAMES {
            &self.ring[..self.count]
        } else {
            &self.ring[..]
        };
        valid.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// 推入一帧 Ls 并按 speech_like 更新噪声底，返回更新后的 floor。
    pub fn update(&mut self, ls: f64, speech_like: bool) -> f64 {
        self.ring[self.head] = ls;
        self.head = (self.head + 1) % MINSTAT_FRAMES;
        self.count = (self.count + 1).min(MINSTAT_FRAMES);
        let m = self.min_ls() + MINSTAT_MARGIN_DB;
        let delta = m - self.floor;
        self.floor += if speech_like {
            delta.clamp(0.0, FLOOR_UP_MAX_DB)
        } else {
            delta.clamp(-FLOOR_DOWN_MAX_DB, FLOOR_UP_MAX_DB)
        };
        self.floor
    }

    /// 当前窗内 min+3dB（校准完成时直接取该值作为 floor）。
    pub fn min_plus3_db(&self) -> f64 {
        self.min_ls() + MINSTAT_MARGIN_DB
    }

    pub fn floor_db(&self) -> f64 {
        self.floor
    }

    pub fn set_floor(&mut self, floor: f64) {
        self.floor = floor;
    }
}

// ============================================================================
// 4 状态 FSM 骨架（v1 输出恒直通，状态机只驱动遥调里的 state/gain）
// ============================================================================

/// FSM 状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiseState {
    /// S0 校准：累计非语音≥0.5s 或 10s 超时后取 min+3dB 转 DENOISE_LOW。
    Calib,
    /// S1 常规降噪（g=0.45）。
    DenoiseLow,
    /// S2 强降噪（g=0.90）。
    DenoiseHigh,
    /// S3 气声旁路（g=0）。
    BreathyBypass,
}

impl DenoiseState {
    pub fn as_str(&self) -> &'static str {
        match self {
            DenoiseState::Calib => "calib",
            DenoiseState::DenoiseLow => "denoise_low",
            DenoiseState::DenoiseHigh => "denoise_high",
            DenoiseState::BreathyBypass => "breathy_bypass",
        }
    }

    /// 该状态的目标增益（0=旁路，1=全抑制）。
    pub fn target_gain(&self) -> f64 {
        match self {
            DenoiseState::Calib => GAIN_CALIB,
            DenoiseState::DenoiseLow => GAIN_DENOISE_LOW,
            DenoiseState::DenoiseHigh => GAIN_DENOISE_HIGH,
            DenoiseState::BreathyBypass => GAIN_BYPASS,
        }
    }
}

/// 一个 20ms tick 的特征输入（全部由音频侧计算好喂入，便于独立测试）。
#[derive(Debug, Clone, Copy)]
pub struct TickFeatures {
    pub snr_db: f64,
    pub pvad: f64,
    pub periodicity: f64,
    pub r_lf: f64,
    pub sigma_rlf: f64,
}

/// FSM tick 结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsmEvent {
    None,
    /// 发生状态迁移（含 CALIB→DENOISE_LOW；上层据 from==Calib 把噪声底
    /// 设为 min+3dB）。
    Transitioned {
        from: DenoiseState,
        to: DenoiseState,
    },
}

/// 4 状态 FSM。
///
/// 门条件（BREATHY_BYPASS 进）：G1（SNR>6dB && pvad 门）**同时** G2（p<0.25）
/// 同时 G3（R_lf<0.45）同时 G4（σ_Rlf>0.05），连续 [`ENTER_TICKS`] tick
/// 且已驻留 ≥[`DWELL_TICKS`]。出：!G1 或 p>0.35 或 R_lf>0.55，连续
/// [`EXIT_TICKS`] tick。S1↔S2 由 SNR 驱动（见 [`RISE_HIGH_TICKS`] /
/// [`FALL_LOW_TICKS`]）。任何迁移清零全部迟滞计数器并重新武装 150ms
/// 线性增益 crossfade。
#[derive(Debug, Clone)]
pub struct DenoiseFsm {
    state: DenoiseState,
    gain: f64,
    /// 每 tick 增益线性步进量（0 表示已到目标）。
    ramp_step: f64,
    dwell_ticks: u32,
    bypass_entry_ticks: u32,
    bypass_exit_ticks: u32,
    snr_low_ticks: u32,
    snr_high_ticks: u32,
    cal_ticks: u32,
    cal_non_speech_ticks: u32,
}

impl Default for DenoiseFsm {
    fn default() -> Self {
        Self::new()
    }
}

impl DenoiseFsm {
    pub fn new() -> Self {
        Self {
            state: DenoiseState::Calib,
            gain: GAIN_CALIB,
            ramp_step: 0.0,
            dwell_ticks: 0,
            bypass_entry_ticks: 0,
            bypass_exit_ticks: 0,
            snr_low_ticks: 0,
            snr_high_ticks: 0,
            cal_ticks: 0,
            cal_non_speech_ticks: 0,
        }
    }

    pub fn state(&self) -> DenoiseState {
        self.state
    }

    /// 当前应用增益（crossfade 中间值）。
    pub fn gain(&self) -> f64 {
        self.gain
    }

    pub fn dwell_ticks(&self) -> u32 {
        self.dwell_ticks
    }

    pub fn bypass_entry_ticks(&self) -> u32 {
        self.bypass_entry_ticks
    }

    pub fn bypass_exit_ticks(&self) -> u32 {
        self.bypass_exit_ticks
    }

    pub fn snr_low_ticks(&self) -> u32 {
        self.snr_low_ticks
    }

    pub fn snr_high_ticks(&self) -> u32 {
        self.snr_high_ticks
    }

    /// 推进一个 20ms tick。
    pub fn tick(&mut self, f: &TickFeatures) -> FsmEvent {
        self.dwell_ticks = self.dwell_ticks.saturating_add(1);
        let from = self.state;
        let speech = speech_like(f.pvad, f.snr_db);

        match self.state {
            DenoiseState::Calib => {
                self.cal_ticks = self.cal_ticks.saturating_add(1);
                if !speech {
                    self.cal_non_speech_ticks = self.cal_non_speech_ticks.saturating_add(1);
                }
                // 校准出口不受驻留门约束（0.5s/10s 本身已 ≥ 300ms）。
                if self.cal_non_speech_ticks >= CAL_NON_SPEECH_TICKS
                    || self.cal_ticks >= CAL_TIMEOUT_TICKS
                {
                    self.enter(DenoiseState::DenoiseLow);
                }
            }
            DenoiseState::DenoiseLow | DenoiseState::DenoiseHigh => {
                // 气声旁路进门（优先于 SNR 档位切换：旁路是防伪影安全门）。
                let g1 = f.snr_db > G1_SNR_DB && pvad_gate(f.pvad);
                let breathy = g1
                    && f.periodicity < G2_P_ENTER
                    && f.r_lf < G3_RLF_ENTER
                    && f.sigma_rlf > G4_SIGMA;
                self.bypass_entry_ticks = if breathy {
                    self.bypass_entry_ticks.saturating_add(1)
                } else {
                    0
                };
                if self.bypass_entry_ticks >= ENTER_TICKS && self.dwell_ticks >= DWELL_TICKS {
                    self.enter(DenoiseState::BreathyBypass);
                } else {
                    // SNR 档位（S1↔S2）。
                    match self.state {
                        DenoiseState::DenoiseLow => {
                            if f.snr_db < G1_SNR_DB {
                                self.snr_low_ticks += 1;
                            } else {
                                self.snr_low_ticks = 0;
                            }
                            if self.snr_low_ticks >= RISE_HIGH_TICKS
                                && self.dwell_ticks >= DWELL_TICKS
                            {
                                self.enter(DenoiseState::DenoiseHigh);
                            }
                        }
                        DenoiseState::DenoiseHigh => {
                            if f.snr_db > SNR_HIGH_EXIT_DB {
                                self.snr_high_ticks += 1;
                            } else {
                                self.snr_high_ticks = 0;
                            }
                            if self.snr_high_ticks >= FALL_LOW_TICKS
                                && self.dwell_ticks >= DWELL_TICKS
                            {
                                self.enter(DenoiseState::DenoiseLow);
                            }
                        }
                        _ => unreachable!("matched DenoiseLow | DenoiseHigh"),
                    }
                }
            }
            DenoiseState::BreathyBypass => {
                let g1 = f.snr_db > G1_SNR_DB && pvad_gate(f.pvad);
                let exit_breathy = !g1 || f.periodicity > G2_P_EXIT || f.r_lf > G3_RLF_EXIT;
                self.bypass_exit_ticks = if exit_breathy {
                    self.bypass_exit_ticks.saturating_add(1)
                } else {
                    0
                };
                if self.bypass_exit_ticks >= EXIT_TICKS && self.dwell_ticks >= DWELL_TICKS {
                    // 统一回 DENOISE_LOW；若 SNR 仍低，S1↔S2 逻辑会在
                    // 500ms 内重新升到 DENOISE_HIGH。
                    self.enter(DenoiseState::DenoiseLow);
                }
            }
        }

        // 增益 150ms 线性 crossfade：迁移当 tick 起按 |Δg|/7.5 线性步进。
        let target = self.state.target_gain();
        if self.ramp_step != 0.0 {
            let next = self.gain + self.ramp_step;
            let reached = (self.ramp_step > 0.0 && next >= target)
                || (self.ramp_step < 0.0 && next <= target);
            self.gain = if reached { target } else { next };
            if reached {
                self.ramp_step = 0.0;
            }
        }

        if from != self.state {
            FsmEvent::Transitioned {
                from,
                to: self.state,
            }
        } else {
            FsmEvent::None
        }
    }

    /// 执行状态迁移：清零全部迟滞计数器、重置驻留、武装线性 crossfade。
    fn enter(&mut self, to: DenoiseState) {
        self.state = to;
        self.dwell_ticks = 0;
        self.bypass_entry_ticks = 0;
        self.bypass_exit_ticks = 0;
        self.snr_low_ticks = 0;
        self.snr_high_ticks = 0;
        let target = to.target_gain();
        if (self.gain - target).abs() < 1e-12 {
            self.ramp_step = 0.0;
        } else {
            self.ramp_step = (target - self.gain) / (CROSSFADE_MS / TICK_MS as f64);
        }
    }

    /// 测试辅助：直接置状态（视为已驻留足够久，增益贴目标值）。
    #[cfg(test)]
    fn force_state(&mut self, state: DenoiseState) {
        self.state = state;
        self.gain = state.target_gain();
        self.ramp_step = 0.0;
        self.dwell_ticks = DWELL_TICKS;
        self.bypass_entry_ticks = 0;
        self.bypass_exit_ticks = 0;
        self.snr_low_ticks = 0;
        self.snr_high_ticks = 0;
    }

    /// 测试辅助：置状态但驻留计数归零（验证 300ms 驻留门）。
    #[cfg(test)]
    fn force_state_fresh(&mut self, state: DenoiseState) {
        self.force_state(state);
        self.dwell_ticks = 0;
    }
}

// ============================================================================
// 自适应降噪级
// ============================================================================

/// 自适应降噪级 v1。
///
/// 任意长度 48k f32 块进、等长直通块出（内部 480 样本环形对齐 nnnoiseless
/// 帧长，只影响特征/遥调时序，不给输出引入延迟）。v1 混合级恒直通：
/// 输出仅是高通结果；`g` 只出现在遥调里，供离线回放调阈值。
/// 送进降噪前的实验性预处理（电平归一化 + 爆破音防护）。
#[derive(Debug, Default)]
pub struct Preprocess {
    agc: agc::SpeechAgc,
    plosive: plosive::PlosiveGuard,
}

pub struct AdaptiveDenoiseStage {
    hp: HighPass90,
    rlf: LowFreqRatio,
    /// 当前 480 样本帧的 HP 输出累加缓冲（帧满即处理）。
    frame: Vec<f32>,
    frame_fill: usize,
    /// tick 内前一帧（20ms 窗 = 前一帧 + 当前帧）。
    prev_frame: Vec<f32>,
    tick_frame_count: u8,
    ls_smoothed: Option<f64>,
    floor: NoiseFloorTracker,
    fsm: DenoiseFsm,
    backend: Box<dyn DenoiseBackend>,
    /// 送进后端/干路的处理链（AGC + 爆破音防护）后的当前帧；
    /// 与 `frame`（HP 原始，供特征/FSM 分析）分离，保证既有阈值语义不变。
    frame_proc: Vec<f32>,
    /// 预处理链（AGC + 爆破音防护）。`None` = 仅原 HP/降噪链（既有语义，
    /// 老测试与 A/B 对照都用这条）；`Some` = 实验链。
    preprocess: Option<Preprocess>,
    pvad: f64,
    tick_index: u64,
    telemetry: VecDeque<String>,
    /// 已进入帧管线的帧数（on_frame 调用计数，混合路径对齐用）。
    frames_in: u64,
    /// 真实混合路径状态；`None` = v1 直通模式（`process` 语义）。
    mix: Option<MixState>,
}

/// 混合路径（真实后端）的流式状态。
#[derive(Debug, Default)]
struct MixState {
    /// 干路（HP 输出帧）FIFO：领先湿路 `latency_frames` 帧，逐帧对齐。
    dry: VecDeque<f32>,
    /// process_mix 累计输入样本数（finish 对账用）。
    pushed: usize,
    /// 已返回给调用方的样本数。
    emitted: usize,
    /// 混合输出 FIFO（已产出未返回）。
    out: Vec<f32>,
    /// 后端非有限/不可用次数。
    backend_failures: u32,
    /// 劣化后永久干路旁路（输出 = 延迟对齐后的 HP）。
    degraded: bool,
    /// 入口 sanitize 掉的非有限输入样本数（NaN/Inf 会被高通状态永久
    /// 污染，混合入口一律置 0；仅统计+标记，不旁路后端）。
    bad_input_samples: u64,
    /// 是否曾出现非有限输入（诊断标记）。
    input_sanitized: bool,
    /// finish_mix 已调用（防二次 finish 重放 padding / 后续 process 破坏计数）。
    finished: bool,
}

impl Default for AdaptiveDenoiseStage {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveDenoiseStage {
    /// 使用占位直通后端构造（v1 默认）。
    pub fn new() -> Self {
        Self::with_backend(Box::new(PassthroughBackend))
    }

    /// 指定降噪后端构造，v1 直通模式（`process` 语义，零延迟等长输出）。
    pub fn with_backend(backend: Box<dyn DenoiseBackend>) -> Self {
        Self {
            hp: HighPass90::default(),
            rlf: LowFreqRatio::default(),
            frame: vec![0.0; FRAME_SAMPLES],
            frame_fill: 0,
            prev_frame: vec![0.0; FRAME_SAMPLES],
            tick_frame_count: 0,
            ls_smoothed: None,
            floor: NoiseFloorTracker::new(),
            fsm: DenoiseFsm::new(),
            backend,
            frame_proc: vec![0.0; FRAME_SAMPLES],
            preprocess: None,
            pvad: PLACEHOLDER_PVAD,
            tick_index: 0,
            telemetry: VecDeque::new(),
            frames_in: 0,
            mix: None,
        }
    }

    /// 指定降噪后端并启用真实混合路径（`process_mix`/`finish_mix`，
    /// 输出 g·denoised+(1-g)·HP，干湿路按后端延迟对齐）。每条流新建
    /// 一个 stage（流结束后不可复用）。
    pub fn with_backend_and_mix(backend: Box<dyn DenoiseBackend>) -> Self {
        Self {
            mix: Some(MixState::default()),
            ..Self::with_backend(backend)
        }
    }

    /// 混合路径 + 实验性预处理链（语音门控 AGC → 爆破音防护 → 降噪）。
    ///
    /// 与 [`Self::with_backend_and_mix`] 分开构造，是为了让"原链路"仍是
    /// 默认语义（既有测试与 A/B 对照都依赖它），实验链必须显式选用。
    pub fn with_backend_and_mix_preprocessed(backend: Box<dyn DenoiseBackend>) -> Self {
        Self {
            preprocess: Some(Preprocess::default()),
            ..Self::with_backend_and_mix(backend)
        }
    }

    /// 处理任意长 48k f32 块，返回等长输出（v1 = 高通直通）。
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        for &s in input {
            let y = self.hp.process(s);
            self.rlf.push(y);
            out.push(y);
            self.frame[self.frame_fill] = y;
            self.frame_fill += 1;
            if self.frame_fill == FRAME_SAMPLES {
                self.frame_fill = 0;
                self.on_frame();
            }
        }
        out
    }

    /// 真实混合路径：处理任意长 48k f32 块，返回当前已确定的全部输出
    /// （可能多于/少于本次输入长度；内容相对输入延迟 = 后端延迟）。
    ///
    /// - 需 [`Self::with_backend_and_mix`] 构造；`finish_mix` 之后不可再调
    ///   （assert，二次处理会破坏长度对账）；
    /// - **入口 sanitize**：非有限输入样本（NaN/±Inf）被置 0 并计数
    ///   （[`Self::bad_input_samples`]），防止高通状态被永久污染；默认
    ///   关闭的 v1 `process()` 路径不做此事；
    /// - 输出与分块方式无关（见测试 `mix_chunk_invariance`）。
    pub fn process_mix(&mut self, input: &[f32]) -> Vec<f32> {
        {
            let mix = self
                .mix
                .as_mut()
                .expect("process_mix 需 with_backend_and_mix 构造");
            assert!(!mix.finished, "process_mix 不可在 finish_mix 之后调用");
            mix.pushed += input.len();
        }
        // 入口 sanitize：非有限输入直接置 0（hp 状态一旦被 NaN 污染
        // 将永久输出 NaN，干湿两路都无法自愈）。局部计数，循环后回写，
        // 避免 mix 借用跨越 on_frame。
        let mut bad: u64 = 0;
        for &s in input {
            let s = if s.is_finite() {
                s
            } else {
                bad += 1;
                0.0
            };
            let y = self.hp.process(s);
            self.rlf.push(y);
            self.frame[self.frame_fill] = y;
            // 处理链只作用于上屏/降噪通路：AGC 先把语音归一化到目标电平，
            // 爆破音防护再截掉低频脉冲；分析通路（frame）保持 HP 原始
            // 信号，FSM/噪声底阈值语义不变。
            let z = match self.preprocess.as_mut() {
                Some(pre) => pre.plosive.process(pre.agc.process(y)),
                None => y,
            };
            self.frame_proc[self.frame_fill] = z;
            self.frame_fill += 1;
            if self.frame_fill == FRAME_SAMPLES {
                self.frame_fill = 0;
                self.on_frame();
            }
        }
        let mix = self.mix.as_mut().unwrap();
        if bad > 0 {
            mix.bad_input_samples += bad;
            mix.input_sanitized = true;
        }
        let out = std::mem::take(&mut mix.out);
        mix.emitted += out.len();
        out
    }

    /// 结束流：推出尾部（不满帧零填充驱动但不让它出屏，后端延迟帧以零
    /// 输入推进），返回剩余输出。**总输出长度 == 总输入长度**（零丢失、
    /// 零 padding 上屏），与分块方式无关。消耗流：此后 stage 不可再用；
    /// 二次调用安全幂等（返回空，不重放 padding）。
    pub fn finish_mix(&mut self) -> Vec<f32> {
        let mix = self
            .mix
            .as_mut()
            .expect("finish_mix 需 with_backend_and_mix 构造");
        if mix.finished {
            return Vec::new();
        }
        // 先置 finished：即使后续断言/panic 中断，也不会重放 padding。
        mix.finished = true;
        // 1) 尾部不满帧：零填充成整帧驱动 on_frame（pad 派生样本由末尾
        //    truncate 丢弃，不上屏）。
        if self.frame_fill > 0 {
            for v in self.frame[self.frame_fill..].iter_mut() {
                *v = 0.0;
            }
            self.frame_fill = 0;
            self.on_frame();
        }
        // 2) 后端延迟尾帧：零输入推进 latency 帧，推出最后 L 个真实帧。
        let latency = self.backend.latency_frames();
        for _ in 0..latency {
            for v in self.frame.iter_mut() {
                *v = 0.0;
            }
            self.on_frame();
        }
        // 3) 对账：截到与输入等长。
        let mix = self.mix.as_mut().unwrap();
        let target = mix.pushed - mix.emitted;
        debug_assert!(
            mix.out.len() >= target,
            "混合输出欠产: out={} target={}",
            mix.out.len(),
            target
        );
        mix.out.truncate(target);
        let out = std::mem::take(&mut mix.out);
        mix.emitted += out.len();
        out
    }

    /// 混合路径是否已劣化旁路（后端非有限/故障；直通模式恒 false）。
    pub fn backend_degraded(&self) -> bool {
        self.mix.as_ref().map(|m| m.degraded).unwrap_or(false)
    }

    /// 混合路径后端故障计数（诊断/遥测）。
    pub fn backend_failures(&self) -> u32 {
        self.mix.as_ref().map(|m| m.backend_failures).unwrap_or(0)
    }

    /// 混合路径入口 sanitize 掉的非有限输入样本数（诊断/遥测）。
    pub fn bad_input_samples(&self) -> u64 {
        self.mix.as_ref().map(|m| m.bad_input_samples).unwrap_or(0)
    }

    /// 混合路径是否曾出现非有限输入。
    pub fn input_sanitized(&self) -> bool {
        self.mix
            .as_ref()
            .map(|m| m.input_sanitized)
            .unwrap_or(false)
    }

    /// 一个 480 样本帧到点：S1 能量、S2 噪声底；每 2 帧一个 20ms tick。
    fn on_frame(&mut self) {
        // S1 帧能量 + 平滑。
        let level = frame_energy_dbfs(&self.frame);
        let ls = match self.ls_smoothed {
            Some(prev) => LS_SMOOTH_NEW * level + (1.0 - LS_SMOOTH_NEW) * prev,
            None => level,
        };
        self.ls_smoothed = Some(ls);

        // 降噪后端调用：v1 直通模式丢弃降噪输出只取 pvad；混合模式下
        // scratch 为本次产出的降噪帧（滞后输入 latency_frames 帧）。
        let mut scratch: [f32; FRAME_SAMPLES] = [0.0; FRAME_SAMPLES];
        // 混合路径喂后端的必须是处理链输出（AGC + 爆破音防护）；v1 直通
        // 路径保持原语义，仍用 HP 帧。
        scratch.copy_from_slice(if self.mix.is_some() {
            &self.frame_proc
        } else {
            &self.frame
        });
        let mut out_info = self.backend.process_frame(&mut scratch);
        // 二次防护（后端自报 usable 也不信任非有限样本/pvad）。
        if !scratch.iter().all(|s| s.is_finite()) || !out_info.pvad.is_finite() {
            out_info.usable = false;
        }
        self.pvad = if out_info.pvad.is_finite() {
            out_info.pvad.clamp(0.0, 1.0) as f64
        } else {
            PLACEHOLDER_PVAD
        };

        // S2/S3 噪声底 + SNR（SNR 取最新 10ms 帧的 Ls）。
        let snr = ls - self.floor.floor_db();
        let speech = speech_like(self.pvad, snr);
        self.floor.update(ls, speech);

        // ---- 混合路径输出：g·denoised + (1-g)·HP ----
        // 帧序号 k = frames_in（从 0 计）；本次 scratch 对应输入帧 k-L，
        // 当 k >= L 才有真实湿路帧可出（首 L 帧为后端淡入伪影，丢弃）。
        // g 取输出时刻 FSM 增益（相对音频超前 ≤1 tick；150ms crossfade
        // 下可忽略，且按帧管线确定、与分块方式无关）。
        self.frames_in += 1;
        if let Some(mix) = self.mix.as_mut() {
            if !out_info.usable {
                mix.backend_failures = mix.backend_failures.saturating_add(1);
                mix.degraded = true;
            }
            // 干路入队：本帧（k）经处理链的输出（未降噪），与湿路同一电平域。
            mix.dry.extend(self.frame_proc.iter().copied());
            let latency = self.backend.latency_frames() as u64;
            if self.frames_in > latency {
                let mut dry_frame = [0.0f32; FRAME_SAMPLES];
                for v in dry_frame.iter_mut() {
                    *v = mix.dry.pop_front().unwrap_or(0.0);
                }
                if mix.degraded {
                    // 安全旁路：输出 = 干路（延迟对齐后的 HP）。
                    mix.out.extend_from_slice(&dry_frame);
                } else {
                    let g = self.fsm.gain();
                    for i in 0..FRAME_SAMPLES {
                        let wet = scratch[i] as f64;
                        let dry = dry_frame[i] as f64;
                        let m = g * wet + (1.0 - g) * dry;
                        // 单点非有限兜底（理论上不可达）：回落干路样本。
                        mix.out.push(if m.is_finite() {
                            m as f32
                        } else {
                            dry_frame[i]
                        });
                    }
                }
            }
        }

        // 20ms tick 的前半帧：只保存窗口。
        self.tick_frame_count += 1;
        if self.tick_frame_count == 1 {
            self.prev_frame.copy_from_slice(&self.frame);
            return;
        }
        self.tick_frame_count = 0;

        // S4 周期性（20ms 窗）+ S5/S7 低频占比。
        let mut window = Vec::with_capacity(TICK_SAMPLES);
        window.extend_from_slice(&self.prev_frame);
        window.extend_from_slice(&self.frame);
        let periodicity = periodicity_peak(&window);
        let (r_lf, sigma_rlf) = self.rlf.tick();

        let features = TickFeatures {
            snr_db: snr,
            pvad: self.pvad,
            periodicity,
            r_lf,
            sigma_rlf,
        };
        let event = self.fsm.tick(&features);
        // 校准完成：floor 直接取当前窗 min+3dB。
        if let FsmEvent::Transitioned {
            from: DenoiseState::Calib,
            ..
        } = event
        {
            self.floor.set_floor(self.floor.min_plus3_db());
        }

        self.tick_index += 1;
        let t = self.tick_index as f64 * (TICK_MS as f64 / 1000.0);
        self.telemetry.push_back(format!(
            "{t:.3},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.3}",
            ls,
            self.floor.floor_db(),
            snr,
            periodicity,
            r_lf,
            sigma_rlf,
            self.fsm.state().as_str(),
            self.fsm.gain(),
        ));
        while self.telemetry.len() > TELEMETRY_CAP {
            self.telemetry.pop_front();
        }
    }

    /// CSV 表头：`(t, Ls, floor, snr, p, r_lf, sigma_rlf, state, g)`。
    pub fn telemetry_header() -> &'static str {
        "t,ls,floor,snr,p,r_lf,sigma_rlf,state,gain"
    }

    /// 最近一个 20ms tick 的遥调行（尚无完整 tick 时为 `None`）。
    pub fn telemetry_row(&self) -> Option<String> {
        self.telemetry.back().cloned()
    }

    /// 取走（drain）自上次调用以来的全部遥调行，按 tick 顺序返回。
    pub fn drain_telemetry(&mut self) -> Vec<String> {
        self.telemetry.drain(..).collect()
    }

    /// 当前 FSM 状态。
    pub fn state(&self) -> DenoiseState {
        self.fsm.state()
    }

    /// 当前应用增益（crossfade 中间值）。
    pub fn gain(&self) -> f64 {
        self.fsm.gain()
    }

    /// 当前噪声底（dBFS）。
    pub fn noise_floor_db(&self) -> f64 {
        self.floor.floor_db()
    }

    /// 最近一次后端 pvad。
    pub fn pvad(&self) -> f64 {
        self.pvad
    }

    /// 当前语音门控 AGC 增益（dB）；未启用预处理链时为 0。
    pub fn agc_gain_db(&self) -> f32 {
        self.preprocess.as_ref().map_or(0.0, |p| p.agc.gain_db())
    }

    /// 当前爆破音防护混入权重（0..1）；未启用时为 0。
    pub fn plosive_duck(&self) -> f32 {
        self.preprocess.as_ref().map_or(0.0, |p| p.plosive.duck())
    }

    /// 已判定的爆破音次数。
    pub fn plosive_detections(&self) -> u64 {
        self.preprocess
            .as_ref()
            .map_or(0, |p| p.plosive.detections())
    }
}

// ============================================================================
// 测试（合成信号全部确定性：固定正弦/方波 + 固定种子 LCG，无随机性）
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 固定种子 LCG“白噪声”（确定性，跨运行逐位一致）。
    struct Lcg(u32);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            // Numerical Recipes 常数；输出 ∈ [-1, 1)。
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.0 >> 8) as f32 / 8_388_608.0) - 1.0
        }
    }

    /// 固定频率正弦（相位累积，避免 i 增大时的精度漂移）。
    fn sine(freq: f64, n: usize, amp: f64, phase: Option<f64>) -> Vec<f32> {
        let mut phase = phase.unwrap_or(0.0);
        let step = 2.0 * std::f64::consts::PI * freq / SAMPLE_RATE as f64;
        (0..n)
            .map(|_| {
                phase += step;
                (amp * phase.sin()) as f32
            })
            .collect()
    }

    /// 稳态 RMS（跳过 warmup 样本）。
    fn steady_rms(x: &[f32], warmup: usize) -> f64 {
        let s: f64 = x[warmup..]
            .iter()
            .map(|&v| {
                let v = v as f64;
                v * v
            })
            .sum();
        (s / (x.len() - warmup) as f64).sqrt()
    }

    // ---- 高通频响 ----

    /// 50 Hz 正弦衰减。理论值：给定系数（90 Hz/Q=0.707）在 50 Hz 处
    /// |H| ≈ 0.299，即 ~10.5 dB。提纲的单测描述写的是 “>12dB”，但那需要
    /// fc≈98Hz 才能达到；评审定稿的系数优先，因此按理论值 ±余量断言
    /// （下界 10 dB 抓回归，上界 11.5 dB 防止有人把 fc 改大凑 12dB）。
    #[test]
    fn hp_attenuates_50hz_about_10db() {
        let input = sine(50.0, SAMPLE_RATE as usize * 2, 0.5, None);
        let mut hp = HighPass90::default();
        let out: Vec<f32> = input.iter().map(|&s| hp.process(s)).collect();
        let ratio = steady_rms(&out, 4800) / steady_rms(&input, 4800);
        let attn_db = -20.0 * ratio.log10();
        assert!(
            attn_db > 10.0 && attn_db < 11.5,
            "50Hz 衰减 {attn_db:.3} dB（理论 ~10.5 dB）"
        );
    }

    #[test]
    fn hp_flat_at_1khz() {
        let input = sine(1000.0, SAMPLE_RATE as usize, 0.5, None);
        let mut hp = HighPass90::default();
        let out: Vec<f32> = input.iter().map(|&s| hp.process(s)).collect();
        let ratio = steady_rms(&out, 4800) / steady_rms(&input, 4800);
        let db = 20.0 * ratio.log10();
        assert!(db.abs() < 0.5, "1kHz 通带偏差 {db:.4} dB");
    }

    // ---- S2 min-stat 噪声底 ----

    #[test]
    fn floor_tracker_rise_and_descent_rate_limits() {
        let mut t = NoiseFloorTracker::new();
        // 非语音：floor=-45，m=-27 → 每帧 +0.02 封顶。
        for _ in 0..10 {
            t.update(-30.0, false);
        }
        assert!((t.floor_db() - (-44.80)).abs() < 1e-9, "{}", t.floor_db());
        // 换成更低的环境：m=-57 → 每帧 -0.10 封顶。
        for _ in 0..50 {
            t.update(-60.0, false);
        }
        assert!((t.floor_db() - (-49.80)).abs() < 1e-9, "{}", t.floor_db());
    }

    #[test]
    fn floor_tracker_speech_never_decreases() {
        let mut t = NoiseFloorTracker::new();
        // 语音帧 m=-17 高于 floor → 只允许每帧 +0.02。
        for _ in 0..3 {
            t.update(-20.0, true);
        }
        assert!((t.floor_db() - (-44.94)).abs() < 1e-9, "{}", t.floor_db());
        // 语音帧：环内残留更低 Ls → m=-57 低于 floor，禁止下调。
        for _ in 0..5 {
            t.update(-60.0, true);
        }
        assert!((t.floor_db() - (-44.94)).abs() < 1e-9, "{}", t.floor_db());
    }

    #[test]
    fn floor_tracker_ring_holds_last_200_frames() {
        let mut t = NoiseFloorTracker::new();
        for _ in 0..MINSTAT_FRAMES {
            t.update(-30.0, false);
        }
        assert!((t.min_plus3_db() - (-27.0)).abs() < 1e-9);
        // 再推 200 帧更低值，旧帧全部被逐出。
        for _ in 0..MINSTAT_FRAMES {
            t.update(-60.0, false);
        }
        assert!((t.min_plus3_db() - (-57.0)).abs() < 1e-9);
    }

    #[test]
    fn floor_tracker_min_before_full() {
        let mut t = NoiseFloorTracker::new();
        for ls in [-30.0, -40.0, -35.0] {
            t.update(ls, false);
        }
        assert!((t.min_plus3_db() - (-37.0)).abs() < 1e-9);
    }

    // ---- S4 周期性 ----

    #[test]
    fn periodicity_voiced_sine_is_high() {
        // 220 Hz 正弦（基频周期 218 样本 ∈ [96,768]）。
        let window = sine(220.0, TICK_SAMPLES, 0.5, None);
        let p = periodicity_peak(&window);
        assert!(p > 0.6, "p = {p:.4}");
    }

    #[test]
    fn periodicity_white_noise_is_low() {
        let mut lcg = Lcg(0x1234_5678);
        let window: Vec<f32> = (0..TICK_SAMPLES).map(|_| lcg.next_f32()).collect();
        let p = periodicity_peak(&window);
        assert!(p < 0.3, "p = {p:.4}");
    }

    // ---- S5/S7 低频占比 ----

    #[test]
    fn rlf_white_noise_is_low() {
        let mut t = LowFreqRatio::default();
        let mut lcg = Lcg(0xDEAD_BEEF);
        // 预热一个窗口再取值，避开滤波器起始暂态。
        for _ in 0..TICK_SAMPLES {
            t.push(lcg.next_f32());
        }
        let _ = t.tick();
        for _ in 0..TICK_SAMPLES {
            t.push(lcg.next_f32());
        }
        let (r, _) = t.tick();
        assert!(r < 0.5, "r_lf = {r:.4}（理论 ~0.09）");
    }

    #[test]
    fn rlf_sine_500hz_is_high() {
        // 预热一个窗口避开起始暂态，再取稳态窗口。
        let mut t = LowFreqRatio::default();
        for s in sine(500.0, TICK_SAMPLES * 2, 0.5, None) {
            t.push(s);
        }
        let (r, _) = t.tick();
        assert!(r > 0.6, "r_lf = {r:.4}（理论 ~0.90）");
    }

    #[test]
    fn rlf_sigma_detects_fluctuation() {
        // 稳定正弦 → σ 很小（逐窗 push/tick 交错）。
        let mut t = LowFreqRatio::default();
        let mut sigma_steady = f64::MAX;
        for _ in 0..10 {
            for s in sine(500.0, TICK_SAMPLES, 0.5, None) {
                t.push(s);
            }
            let (_, sigma) = t.tick();
            sigma_steady = sigma;
        }
        assert!(sigma_steady < 0.01, "σ = {sigma_steady:.4}");
        // 交替 正弦窗/噪声窗 → σ 超过 G4 门。
        let mut lcg = Lcg(7);
        for round in 0..20 {
            if round % 2 == 0 {
                for s in sine(500.0, TICK_SAMPLES, 0.5, None) {
                    t.push(s);
                }
            } else {
                for _ in 0..TICK_SAMPLES {
                    t.push(lcg.next_f32());
                }
            }
            t.tick();
        }
        let sigma_mixed = sample_std(&t.history);
        assert!(sigma_mixed > 0.05, "σ = {sigma_mixed:.4}");
    }

    // ---- FSM ----

    fn feats(snr: f64, pvad: f64, p: f64, r_lf: f64, sigma: f64) -> TickFeatures {
        TickFeatures {
            snr_db: snr,
            pvad,
            periodicity: p,
            r_lf,
            sigma_rlf: sigma,
        }
    }

    /// 语音 + 高 SNR + 周期性强（非气声）。
    fn clean_speech() -> TickFeatures {
        feats(10.0, 0.9, 0.80, 0.70, 0.01)
    }

    /// 气声：G1✓（SNR 10>6、pvad 0.9）+ G2✓(p 0.10<0.25) + G3✓(R_lf 0.30<0.45)
    /// + G4✓(σ 0.10>0.05)。
    fn breathy() -> TickFeatures {
        feats(10.0, 0.9, 0.10, 0.30, 0.10)
    }

    /// 低 SNR（<6dB）常规语音。
    fn noisy_speech() -> TickFeatures {
        feats(3.0, 0.9, 0.80, 0.70, 0.01)
    }

    /// 高 SNR（>12dB）。
    fn quiet_speech() -> TickFeatures {
        feats(15.0, 0.9, 0.80, 0.70, 0.01)
    }

    /// 非语音（SNR 低 + pvad 低）。
    fn non_speech() -> TickFeatures {
        feats(0.0, 0.1, 0.10, 0.30, 0.01)
    }

    #[test]
    fn fsm_calibration_exits_after_half_second_non_speech() {
        let mut fsm = DenoiseFsm::new();
        // 先喂 24 tick 语音：非语音累计为 0，同时验证语音不累计。
        for _ in 0..24 {
            assert_eq!(fsm.tick(&clean_speech()), FsmEvent::None);
        }
        assert_eq!(fsm.state(), DenoiseState::Calib);
        for _ in 0..24 {
            fsm.tick(&non_speech());
        }
        // 24 tick 非语音还不够 0.5s（25 tick）。
        assert_eq!(fsm.state(), DenoiseState::Calib);
        let ev = fsm.tick(&non_speech());
        assert_eq!(
            ev,
            FsmEvent::Transitioned {
                from: DenoiseState::Calib,
                to: DenoiseState::DenoiseLow,
            }
        );
    }

    #[test]
    fn fsm_calibration_timeout_at_10s() {
        let mut fsm = DenoiseFsm::new();
        // 一直语音：靠 10s 超时出口。
        for _ in 0..499 {
            fsm.tick(&clean_speech());
        }
        assert_eq!(fsm.state(), DenoiseState::Calib);
        let ev = fsm.tick(&clean_speech());
        assert!(matches!(
            ev,
            FsmEvent::Transitioned {
                from: DenoiseState::Calib,
                ..
            }
        ));
    }

    #[test]
    fn fsm_breathy_entry_needs_10_sustained_ticks() {
        let mut fsm = DenoiseFsm::new();
        fsm.force_state(DenoiseState::DenoiseLow);
        for _ in 0..9 {
            fsm.tick(&breathy());
        }
        assert_eq!(fsm.state(), DenoiseState::DenoiseLow);
        assert_eq!(fsm.bypass_entry_ticks(), 9);
        let ev = fsm.tick(&breathy());
        assert_eq!(fsm.state(), DenoiseState::BreathyBypass);
        assert!(matches!(ev, FsmEvent::Transitioned { .. }));
    }

    #[test]
    fn fsm_breathy_entry_blocked_by_dwell() {
        let mut fsm = DenoiseFsm::new();
        fsm.force_state_fresh(DenoiseState::DenoiseLow);
        // 迟滞计数 14 tick 时已达 10，但驻留只有 14 < 15。
        for _ in 0..14 {
            fsm.tick(&breathy());
        }
        assert_eq!(fsm.state(), DenoiseState::DenoiseLow);
        assert_eq!(fsm.bypass_entry_ticks(), 14);
        fsm.tick(&breathy()); // 第 15 tick：驻留达标 → 进入。
        assert_eq!(fsm.state(), DenoiseState::BreathyBypass);
    }

    #[test]
    fn fsm_breathy_entry_streak_resets() {
        let mut fsm = DenoiseFsm::new();
        fsm.force_state(DenoiseState::DenoiseLow);
        for _ in 0..9 {
            fsm.tick(&breathy());
        }
        fsm.tick(&clean_speech()); // 断 streak。
        assert_eq!(fsm.bypass_entry_ticks(), 0);
        for _ in 0..9 {
            fsm.tick(&breathy());
        }
        assert_eq!(fsm.state(), DenoiseState::DenoiseLow);
        fsm.tick(&breathy());
        assert_eq!(fsm.state(), DenoiseState::BreathyBypass);
    }

    #[test]
    fn fsm_breathy_exit_needs_25_sustained_ticks() {
        let mut fsm = DenoiseFsm::new();
        fsm.force_state(DenoiseState::BreathyBypass);
        // 出条件：p>0.35。
        for _ in 0..24 {
            fsm.tick(&feats(10.0, 0.9, 0.50, 0.30, 0.10));
        }
        assert_eq!(fsm.state(), DenoiseState::BreathyBypass);
        fsm.tick(&feats(10.0, 0.9, 0.50, 0.30, 0.10));
        assert_eq!(fsm.state(), DenoiseState::DenoiseLow);
    }

    #[test]
    fn fsm_breathy_exit_when_speech_ends() {
        // G1 失效（pvad 掉到 0.2）也算出条件。
        let mut fsm = DenoiseFsm::new();
        fsm.force_state(DenoiseState::BreathyBypass);
        for _ in 0..25 {
            fsm.tick(&feats(10.0, 0.2, 0.10, 0.30, 0.10));
        }
        assert_eq!(fsm.state(), DenoiseState::DenoiseLow);
    }

    #[test]
    fn fsm_snr_low_sustained_rises_to_high() {
        let mut fsm = DenoiseFsm::new();
        fsm.force_state(DenoiseState::DenoiseLow);
        for _ in 0..24 {
            fsm.tick(&noisy_speech());
        }
        assert_eq!(fsm.state(), DenoiseState::DenoiseLow);
        let ev = fsm.tick(&noisy_speech()); // 500ms 到点。
        assert_eq!(
            ev,
            FsmEvent::Transitioned {
                from: DenoiseState::DenoiseLow,
                to: DenoiseState::DenoiseHigh,
            }
        );
    }

    #[test]
    fn fsm_snr_high_sustained_falls_to_low() {
        let mut fsm = DenoiseFsm::new();
        fsm.force_state(DenoiseState::DenoiseHigh);
        for _ in 0..49 {
            fsm.tick(&quiet_speech());
        }
        assert_eq!(fsm.state(), DenoiseState::DenoiseHigh);
        fsm.tick(&quiet_speech()); // 1s 到点。
        assert_eq!(fsm.state(), DenoiseState::DenoiseLow);
    }

    #[test]
    fn fsm_snr_streak_resets_on_single_clean_tick() {
        let mut fsm = DenoiseFsm::new();
        fsm.force_state(DenoiseState::DenoiseLow);
        for _ in 0..24 {
            fsm.tick(&noisy_speech());
        }
        fsm.tick(&clean_speech()); // SNR 回到 >6dB，断 streak。
        for _ in 0..24 {
            fsm.tick(&noisy_speech());
        }
        assert_eq!(fsm.state(), DenoiseState::DenoiseLow);
        fsm.tick(&noisy_speech());
        assert_eq!(fsm.state(), DenoiseState::DenoiseHigh);
    }

    #[test]
    fn fsm_gain_crossfade_is_linear() {
        let mut fsm = DenoiseFsm::new();
        fsm.force_state(DenoiseState::DenoiseLow); // gain 0.45
        assert!((fsm.gain() - 0.45).abs() < 1e-12);
        // 触发进入 BREATHY_BYPASS（目标 0）：步长 = -0.45/7.5 = -0.06。
        for _ in 0..10 {
            fsm.tick(&breathy());
        }
        assert_eq!(fsm.state(), DenoiseState::BreathyBypass);
        let mut gains = vec![fsm.gain()];
        for _ in 0..10 {
            fsm.tick(&breathy());
            gains.push(fsm.gain());
        }
        let expected = [0.39, 0.33, 0.27, 0.21, 0.15, 0.09, 0.03, 0.0, 0.0, 0.0, 0.0];
        for (got, want) in gains.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "{gains:?}");
        }
        // 反向：出旁路回 0.45（步长 +0.06）。
        fsm.force_state(DenoiseState::BreathyBypass);
        fsm.tick(&feats(10.0, 0.9, 0.50, 0.70, 0.10)); // 出条件 tick 1
        assert!((fsm.gain() - 0.0).abs() < 1e-12); // 目标没变，gain 保持 0
        for _ in 0..25 {
            fsm.tick(&clean_speech());
        }
        // 已回 DENOISE_LOW 并开始向 0.45 爬升。
        assert!(fsm.gain() > 0.0 && fsm.gain() <= 0.45);
    }

    // ---- Stage 级 ----

    /// 确定性混合信号：440Hz 正弦 + 220Hz 方波 + 固定 LCG 噪声。
    fn deterministic_mix(n: usize) -> Vec<f32> {
        let sine440 = sine(440.0, n, 0.3, None);
        let mut phase = 0.0f64;
        let step = 2.0 * std::f64::consts::PI * 220.0 / SAMPLE_RATE as f64;
        let mut lcg = Lcg(42);
        (0..n)
            .map(|i| {
                phase += step;
                let square = if phase.sin() >= 0.0 { 0.2 } else { -0.2 };
                sine440[i] + square as f32 + 0.1 * lcg.next_f32()
            })
            .collect()
    }

    #[test]
    fn stage_output_is_chunking_invariant_and_equal_length() {
        let input = deterministic_mix(20_003);
        let whole = AdaptiveDenoiseStage::new().process(&input);
        assert_eq!(whole.len(), input.len());

        let mut stage = AdaptiveDenoiseStage::new();
        let mut pieced = Vec::with_capacity(input.len());
        let mut offsets = [1usize, 479, 960, 1234, 5000, 9999].into_iter().peekable();
        let mut pos = 0;
        while let Some(len) = offsets.next() {
            pieced.extend_from_slice(&stage.process(&input[pos..pos + len]));
            pos += len;
        }
        pieced.extend_from_slice(&stage.process(&input[pos..]));
        assert_eq!(pieced.len(), input.len());
        assert_eq!(whole, pieced, "任意分块不得改变逐样本输出");
    }

    #[test]
    fn stage_telemetry_cadence_and_columns() {
        let mut stage = AdaptiveDenoiseStage::new();
        let _ = stage.process(&vec![0.0; SAMPLE_RATE as usize]);
        let rows = stage.drain_telemetry();
        assert_eq!(rows.len(), 50, "1s 音频 = 50 个 20ms tick");
        assert_eq!(
            AdaptiveDenoiseStage::telemetry_header(),
            "t,ls,floor,snr,p,r_lf,sigma_rlf,state,gain"
        );
        assert!(rows[0].starts_with("0.020,"));
        assert!(rows[49].starts_with("1.000,"));
        for row in &rows {
            assert_eq!(row.split(',').count(), 9, "row = {row}");
        }
        // 第 25 tick（索引 24）静音完成校准 → denoise_low。
        assert!(rows[23].contains(",calib,"));
        assert!(rows[24].contains(",denoise_low,"));
        // telemetry_row 给最新一行。
        assert_eq!(stage.telemetry_row(), None); // 已被 drain 清空
    }

    #[test]
    fn stage_silence_completes_calibration() {
        let mut stage = AdaptiveDenoiseStage::new();
        // 占位 pvad=0.5 处于中性位：静音 SNR<3dB → 非语音，0.5s 完成校准。
        let _ = stage.process(&vec![0.0; 24_000]);
        assert_eq!(stage.state(), DenoiseState::DenoiseLow);
        // min+3dB：静音 Ls=-200dBFS → floor = -197。
        assert!(stage.noise_floor_db() < -100.0);
    }

    #[test]
    fn stage_short_inputs_do_not_tick() {
        let mut stage = AdaptiveDenoiseStage::new();
        assert!(stage.process(&[]).is_empty());
        assert_eq!(stage.process(&[0.1; 100]).len(), 100);
        assert_eq!(stage.process(&[0.1; 859]).len(), 859);
        assert!(stage.drain_telemetry().is_empty());
        assert_eq!(stage.process(&[0.1; 1]).len(), 1); // 累计 960 → 1 tick。
        assert_eq!(stage.drain_telemetry().len(), 1);
    }

    #[test]
    fn stage_output_is_highpassed() {
        // 50Hz 经级联输出衰减 ~10.5dB（与高通单测一致）。
        let input = sine(50.0, SAMPLE_RATE as usize * 2, 0.5, None);
        let out = AdaptiveDenoiseStage::new().process(&input);
        let ratio = steady_rms(&out, 4800) / steady_rms(&input, 4800);
        assert!(ratio < 0.32, "ratio = {ratio:.4}");
        // 1kHz 通带不受影响。
        let input = sine(1000.0, SAMPLE_RATE as usize, 0.5, None);
        let out = AdaptiveDenoiseStage::new().process(&input);
        let ratio = steady_rms(&out, 4800) / steady_rms(&input, 4800);
        assert!((20.0 * ratio.log10()).abs() < 0.5, "ratio = {ratio:.4}");
    }

    #[test]
    fn stage_uses_custom_backend_pvad() {
        /// pvad 恒 1.0 的占位后端：静音也应被判定为“语音”。
        #[derive(Default)]
        struct AlwaysVoice;
        impl DenoiseBackend for AlwaysVoice {
            fn process_frame(&mut self, _f: &mut [f32; FRAME_SAMPLES]) -> FrameOutput {
                FrameOutput::ok(1.0)
            }
        }
        let mut stage = AdaptiveDenoiseStage::with_backend(Box::new(AlwaysVoice));
        let _ = stage.process(&vec![0.0; SAMPLE_RATE as usize]);
        // 静音（SNR<3dB）→ speech_like 仍为假 → 校准仍能完成；但校准后
        // SNR<6dB 持续 500ms → 恰好再升到 DENOISE_HIGH（1s = 50 tick）。
        assert_eq!(stage.state(), DenoiseState::DenoiseHigh);
        assert!((stage.pvad() - 1.0).abs() < 1e-9);
    }

    // ====================================================================
    // 真实混合路径（process_mix / finish_mix）
    // ====================================================================

    /// 带 LCG 噪声的确定性测试信号。
    fn noisy_sine(n: usize, sine_amp: f64, noise_amp: f64, seed: u32) -> Vec<f32> {
        let mut rng = Lcg(seed);
        let tone = sine(440.0, n, sine_amp, None);
        (0..n)
            .map(|i| tone[i] + noise_amp as f32 * rng.next_f32())
            .collect()
    }

    #[test]
    fn env_switch_defaults_off_and_reads_exact_value() {
        // 环境变量为进程全局：本测试是本测试二进制内唯一读取方，安全。
        std::env::remove_var("VOO_EXPERIMENTAL_DENOISE");
        assert!(!experimental_denoise_enabled());
        std::env::set_var("VOO_EXPERIMENTAL_DENOISE", "0");
        assert!(!experimental_denoise_enabled());
        std::env::set_var("VOO_EXPERIMENTAL_DENOISE", "true");
        assert!(!experimental_denoise_enabled());
        std::env::set_var("VOO_EXPERIMENTAL_DENOISE", "1");
        assert!(experimental_denoise_enabled());
        std::env::remove_var("VOO_EXPERIMENTAL_DENOISE");
    }

    /// 总输出长度 == 总输入长度（含非 480 倍数长度、空流、短于延迟的流）。
    #[test]
    fn mix_finish_preserves_exact_length() {
        let mut lens = vec![0usize, 1, 479, 480, 481, 960, 961, 4800];
        lens.push(FRAME_SAMPLES + 123);
        lens.push(FRAME_SAMPLES * 3 + 7);
        for len in lens {
            let input = noisy_sine(len, 0.3, 0.05, len as u32 + 1);
            let mut stage =
                AdaptiveDenoiseStage::with_backend_and_mix(Box::new(nn::NnnoiselessBackend::new()));
            let mut got = stage.process_mix(&input);
            got.extend(stage.finish_mix());
            assert_eq!(got.len(), len, "len={len}");
            assert!(got.iter().all(|s| s.is_finite()), "len={len}");
            assert!(got.iter().all(|s| s.abs() <= 1.1), "len={len}");
            assert!(!stage.backend_degraded(), "len={len}");
        }
    }

    /// 分块不变性：同一输入流，任意分块 → 逐位一致输出。
    #[test]
    fn mix_chunk_invariance() {
        let input = noisy_sine(SAMPLE_RATE as usize * 2, 0.25, 0.08, 99);
        let run = |chunks: Vec<usize>| -> Vec<f32> {
            let mut stage =
                AdaptiveDenoiseStage::with_backend_and_mix(Box::new(nn::NnnoiselessBackend::new()));
            let mut out = Vec::new();
            let mut pos = 0usize;
            for &c in &chunks {
                let end = (pos + c).min(input.len());
                out.extend(stage.process_mix(&input[pos..end]));
                pos = end;
            }
            out.extend(stage.process_mix(&input[pos..]));
            out.extend(stage.finish_mix());
            out
        };
        let whole = run(vec![usize::MAX]);
        assert_eq!(whole.len(), input.len());
        let blocky = run(vec![4800; 20]);
        assert_eq!(blocky, whole);
        let ragged = run(vec![1, 479, 961, 12345, 7, 480]);
        assert_eq!(ragged, whole);
    }

    /// 真实降噪链路应显著衰减平稳带限噪声（不验证音质，只验证“真的在降”）。
    ///
    /// 契约同 nn.rs 探针 `probe_matrix`：模型对中幅度带限平稳噪声抑制
    /// 最强；高幅度宽带噪声不保证（模型属性）。6s 时长保证 FSM 走完
    /// 校准（floor 从 -45 dBFS 爬到 min+3dB 需 ~1.8s）并升到
    /// DENOISE_HIGH（g=0.9），跳过 3.5s warmup 后测稳态。
    #[test]
    fn mix_suppresses_stationary_bandlimited_noise() {
        let n = SAMPLE_RATE as usize * 6;
        let mut rng = Lcg(5);
        let mut lp = 0.0f32;
        let input: Vec<f32> = (0..n)
            .map(|_| {
                let x = 0.06 * rng.next_f32();
                lp += 0.316 * (x - lp);
                lp
            })
            .collect();
        // 基准：v1 直通（仅 HP）。
        let base = AdaptiveDenoiseStage::new().process(&input);
        let mut stage =
            AdaptiveDenoiseStage::with_backend_and_mix(Box::new(nn::NnnoiselessBackend::new()));
        let mut got = stage.process_mix(&input);
        got.extend(stage.finish_mix());
        let warm = SAMPLE_RATE as usize / 2 * 7; // 3.5s
        let base_rms = steady_rms(&base, warm);
        let got_rms = steady_rms(&got, warm);
        let reduction_db = 20.0 * (base_rms / got_rms).log10();
        assert!(
            reduction_db > 8.0,
            "平稳带限噪声衰减不足: {reduction_db:.1} dB (base={base_rms:.4} got={got_rms:.4})"
        );
        assert!(!stage.backend_degraded());
        assert!(got.iter().all(|s| s.is_finite()));
    }

    /// 后端非有限输出 → 安全旁路（干路 HP），不 panic、不产生 NaN，
    /// 劣化后输出与纯 HP 逐位一致（latency=0 下对齐）。
    #[test]
    fn mix_backend_nan_latches_bypass_no_panic() {
        struct NanOnce {
            calls: u32,
        }
        impl DenoiseBackend for NanOnce {
            fn process_frame(&mut self, frame: &mut [f32; FRAME_SAMPLES]) -> FrameOutput {
                self.calls += 1;
                if self.calls == 3 {
                    // 后端自报可用但输出含 NaN：stage 二次防护必须兜住。
                    frame[10] = f32::NAN;
                }
                FrameOutput::ok(0.5)
            }
        }
        let input = sine(440.0, SAMPLE_RATE as usize, 0.4, None);
        let mut stage = AdaptiveDenoiseStage::with_backend_and_mix(Box::new(NanOnce { calls: 0 }));
        let mut got = stage.process_mix(&input);
        got.extend(stage.finish_mix());
        assert_eq!(got.len(), input.len());
        assert!(got.iter().all(|s| s.is_finite()));
        assert!(stage.backend_degraded());
        assert_eq!(stage.backend_failures(), 1);
        // 基准 HP：劣化前（帧 0-2）湿路 == 干路，混合 ≈ HP（浮点舍入级
        // 差异）；劣化帧起应为纯 HP 直通。整体逐位比对差值 < 1e-6，且
        // 帧 3 起逐位相等。
        let hp = AdaptiveDenoiseStage::new().process(&input);
        let mut max_diff = 0.0f64;
        for (a, b) in got.iter().zip(hp.iter()) {
            max_diff = max_diff.max((*a as f64 - *b as f64).abs());
        }
        assert!(max_diff < 1e-6, "max_diff={max_diff}");
        for i in (3 * FRAME_SAMPLES)..got.len() {
            assert_eq!(got[i], hp[i], "i={i}");
        }
    }

    /// 纯静音：输出有限、有界，且长度守恒（不因后端而并发出声）。
    #[test]
    fn mix_silence_stays_silent_and_finite() {
        let input = vec![0.0f32; SAMPLE_RATE as usize];
        let mut stage =
            AdaptiveDenoiseStage::with_backend_and_mix(Box::new(nn::NnnoiselessBackend::new()));
        let mut got = stage.process_mix(&input);
        got.extend(stage.finish_mix());
        assert_eq!(got.len(), input.len());
        assert!(got.iter().all(|s| s.is_finite()));
        assert!(got.iter().all(|s| s.abs() < 1e-3));
    }

    /// 非有限输入（NaN/±Inf）在混合入口被置 0：高通状态不被永久污染，
    /// 坏样本之后输出恢复 finite，且全程无 NaN 进入输出。
    #[test]
    fn mix_input_nonfinite_sanitized_and_recovers() {
        let n = SAMPLE_RATE as usize / 2;
        let mut input = noisy_sine(n, 0.3, 0.05, 21);
        // 中段插入坏样本（含跨帧边界，确保污染路径被真实验证）。
        input[1000] = f32::NAN;
        input[1001] = f32::INFINITY;
        input[5000] = f32::NEG_INFINITY;
        let mut stage =
            AdaptiveDenoiseStage::with_backend_and_mix(Box::new(nn::NnnoiselessBackend::new()));
        // 分三段喂入，坏样本夹在中间。
        let mut got = stage.process_mix(&input[..700]);
        got.extend(stage.process_mix(&input[700..1200]));
        got.extend(stage.process_mix(&input[1200..]));
        got.extend(stage.finish_mix());
        assert_eq!(got.len(), input.len());
        assert!(got.iter().all(|s| s.is_finite()), "输出仍含非有限值");
        assert_eq!(stage.bad_input_samples(), 3);
        assert!(stage.input_sanitized());
        assert!(!stage.backend_degraded(), "输入坏样本不应旁路后端");
        // 坏样本之后恢复正常：尾部有能量且有限（未被置零/污染）。
        let tail_rms = steady_rms(&got[n - 4800..], 0);
        assert!(tail_rms > 1e-3, "尾部应恢复正常信号，rms={tail_rms}");
    }

    /// finish_mix 幂等：二次调用返回空，不重放 padding（总长不因二次
    /// finish 变化）；finish 后再 process_mix 属误用，必须 assert 拦截。
    #[test]
    fn finish_mix_twice_idempotent_and_post_finish_guards() {
        let input = noisy_sine(FRAME_SAMPLES * 5 + 123, 0.3, 0.05, 33);
        let mut stage =
            AdaptiveDenoiseStage::with_backend_and_mix(Box::new(nn::NnnoiselessBackend::new()));
        let mut got = stage.process_mix(&input);
        got.extend(stage.finish_mix());
        assert_eq!(got.len(), input.len());
        // 二次 finish：空、不重放。
        assert!(stage.finish_mix().is_empty());
        assert!(stage.finish_mix().is_empty());
        // 误用拦截。
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = stage.process_mix(&input[..16]);
        }));
        assert!(result.is_err(), "finish 后 process_mix 应 assert");
    }
}
