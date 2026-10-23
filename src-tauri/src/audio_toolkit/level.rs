//! 输入电平调理（leveler + 前视限幅）：让"低声"和"大声"都落进模型能正常解码的区间。
//!
//! # 为什么必须放在 VAD 之前
//!
//! 轻声说话首先死在 VAD：能量不够门限，帧被当噪声丢掉，后面再怎么归一化
//! 都救不回来。所以增益调理必须作用在**送进 VAD 的同一份样本**上，并让
//! VAD 与录音缓冲看到完全相同的条件化音频（含 prefill 帧）。
//!
//! # 与"固定 RMS 归一化"的区别
//!
//! 不能用"每一小段都拉到 -26 dBFS"：安静办公室里那会把底噪一起顶到
//! 语音电平，VAD 反复误触发。这里做的是**有界、噪声感知**的电平调理：
//!
//! - 只提升、不衰减：增益下限是单位增益（`max_cut_db = 0`），绝不为了
//!   凑目标电平把信号往下压；唯一的向下处理是末端限幅器的必要保护；
//! - 提升受 SNR 迟滞门控：估计 SNR > 10 dB 才允许开始提升，< 6 dB 停止
//!   提升（期间冻结当前增益，不撤、不抽吸）；
//! - 噪声预算：底噪估计 + 增益 ≤ -50 dBFS（例：底噪 -55 dBFS 时最多
//!   +5 dB），保证提升不会把底噪顶进模型的有效解码区间；
//! - 底噪未就绪（预热未结束，或 min-stat 窗口尚未测满一个完整窗口）
//!   时一律**单位增益**：没有可靠底噪就提升等于盲目放大噪声——用 0 或
//!   极小值"充当"底噪会把平稳噪声整段当语音放大（实测 +9.2 dB）；
//! - 增益向上慢（~500 ms，避免抽吸）；响度突然回升时撤得快（~20 ms），
//!   避免 quiet→loud 过渡期过 boost；
//! - 末端峰值限幅带 5 ms 前视（16 kHz ≈ 80 样本）：增益包络在峰值到达
//!   输出前就开始收拢，输出峰值绝不超过 -3 dBFS 天花板。
//!
//! # min-stat 底噪是向下偏置估计
//!
//! 底噪取"窗口内分块最小 RMS 的最小值"。最小值统计天然**向下偏置**：
//! 它逼近的是底噪的下包络，而不是平均电平。因此由它算出的 SNR 偏乐观、
//! 噪声预算也偏宽松一点。这里接受该偏置（宁可少放噪声进来），但读
//! [`InputLeveler::noise_floor_dbfs`] 的人要知道它是下界，不是均值。
//!
//! # 明确不做的事
//!
//! - 不做噪声闸门（noise gate）：底噪应当以低电平随语音一起进模型，
//!   被切掉的头尾反而伤解码；
//! - 不做逐块归一化：增益按慢时间常数演化，绝不按块重置、重拉；
//! - 不用提升后的 VAD 结果门控增益：VAD 看到的是条件化后的信号，用它
//!   反过来控增益会形成反馈环（越响 → 越提 → VAD 越活跃 → 再提）。
//!
//! # 边界
//!
//! - 已被 ADC 削顶的大声输入**无法恢复**，只能在采集/系统增益处降低；
//! - 语音埋在噪声里时增益救不了 SNR；
//! - 不要与系统级 AGC 叠加使用（两层自适应会互相追逐）；
//! - 前视限幅让输出整体延迟 5 ms（16 kHz ≈ 80 样本）：开头补等长零
//!   样本，每个输入样本恰有一个输出，**样本数守恒不变**。
//!
//! 采样率无关：所有时间常数按毫秒换算，16 kHz 采集与 48 kHz 离线链共用。

/// 目标语音 RMS 电平（dBFS）。调参起点，不是模型的硬性要求。
pub const TARGET_ACTIVE_RMS_DBFS: f32 = -26.0;
/// 向上增益上限。
pub const DEFAULT_MAX_BOOST_DB: f32 = 12.0;
/// 限幅器峰值上限（dBFS）。
pub const LIMITER_CEILING_DBFS: f32 = -3.0;
/// SNR 迟滞上沿：估计 SNR 超过该值才**允许开始**提升。
pub const SNR_BOOST_ENTER_DB: f32 = 10.0;
/// SNR 迟滞下沿：估计 SNR 低于该值**停止**提升（冻结，不撤已建立的增益）。
pub const SNR_BOOST_EXIT_DB: f32 = 6.0;
/// 提升后噪声预算（dBFS）：底噪估计 + 增益不得超过该电平。
/// 例：底噪 -55 dBFS 时最多 +5 dB。
pub const POST_BOOST_NOISE_BUDGET_DBFS: f32 = -50.0;
/// 限幅器前视（毫秒）：增益包络提前这么多时间对即将输出的峰值收拢。
pub const LIMITER_LOOKAHEAD_MS: f32 = 5.0;

/// 电平调理参数（时间单位毫秒）。
#[derive(Debug, Clone, Copy)]
pub struct LevelerConfig {
    pub sample_rate: u32,
    pub target_rms_dbfs: f32,
    /// 最大提升（dB，正数）。
    pub max_boost_db: f32,
    /// 最大衰减（dB，正数）。默认 0：绝不为了凑目标而低于单位增益，
    /// 向下处理只保留末端限幅器的必要保护。
    pub max_cut_db: f32,
    /// 增益向上时间常数（~500 ms）：慢到不抽吸，快到句内来得及生效。
    pub boost_time_ms: f32,
    /// 撤增益时间常数（~20 ms）：quiet→loud 过渡期快速让路，避免过 boost。
    pub cut_time_ms: f32,
    /// RMS 包络时间常数。
    pub rms_time_ms: f32,
    /// 底噪 min-stat 窗口（毫秒）。窗口内最小 RMS 近似底噪：
    /// 平稳噪声的"最小值"就是它自己，因此不会被判成语音而放大。
    /// 注意这是**向下偏置**估计（见模块文档）。
    pub floor_window_ms: f32,
    /// min-stat 分块长度（毫秒），控制内存与时间分辨率权衡。
    pub floor_block_ms: f32,
    /// 预热期（毫秒）：**仅用于排除启动瞬态**（包络从 0 起升、设备刚
    /// 打开时的电平抖动），期间不采样底噪、不调整增益。预热结束后若
    /// 底噪窗口尚未测满，仍保持单位增益。
    pub warmup_ms: f32,
    /// 限幅器起音/释放（毫秒）。
    pub limiter_attack_ms: f32,
    pub limiter_release_ms: f32,
    /// 限幅器前视（毫秒）。
    pub limiter_lookahead_ms: f32,
    pub limiter_ceiling_dbfs: f32,
}

impl LevelerConfig {
    /// 16 kHz 采集路径默认参数。
    pub fn for_capture(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            target_rms_dbfs: TARGET_ACTIVE_RMS_DBFS,
            max_boost_db: DEFAULT_MAX_BOOST_DB,
            max_cut_db: 0.0,
            boost_time_ms: 500.0,
            cut_time_ms: 20.0,
            rms_time_ms: 50.0,
            floor_window_ms: 2_000.0,
            floor_block_ms: 100.0,
            warmup_ms: 300.0,
            limiter_attack_ms: 1.0,
            limiter_release_ms: 120.0,
            limiter_lookahead_ms: LIMITER_LOOKAHEAD_MS,
            limiter_ceiling_dbfs: LIMITER_CEILING_DBFS,
        }
    }

    fn coef(&self, ms: f32) -> f32 {
        let samples = (ms / 1000.0) * self.sample_rate as f32;
        if samples <= 1.0 {
            1.0
        } else {
            1.0 - (-1.0 / samples).exp()
        }
    }
}

impl Default for LevelerConfig {
    fn default() -> Self {
        Self::for_capture(16_000)
    }
}

/// 有界、噪声感知的输入电平调理 + 前视峰值限幅。
#[derive(Debug)]
pub struct InputLeveler {
    cfg: LevelerConfig,
    rms_alpha: f32,
    boost_alpha: f32,
    cut_alpha: f32,
    limiter_attack: f32,
    limiter_release: f32,
    /// RMS 包络（平方）。
    env_sq: f32,
    /// 底噪 min-stat：每个分块内的最小 env_sq 与最近若干分块的环形缓冲。
    block_min_sq: f32,
    block_samples_left: u32,
    block_len: u32,
    floor_blocks: Vec<f32>,
    floor_cursor: usize,
    /// 已提交的分块数（封顶于窗口槽数）：测满一个窗口后底噪才算就绪。
    blocks_committed: usize,
    warmup_left: u32,
    /// SNR 迟滞状态：当前是否允许提升。
    boost_allowed: bool,
    /// 当前增益（dB）。
    gain_db: f32,
    /// 限幅器当前衰减（线性 0..1，1 = 不衰减）。
    limiter_gain: f32,
    /// 前视环形缓冲：始终保存接下来要输出的全部样本（含即将离开的那个），
    /// 供限幅边链提前看到峰值。输出 = 输入延迟前视长度（开头补零）。
    lookahead_buf: Vec<f32>,
    lookahead_pos: usize,
    bad_samples: u64,
    clipped_samples: u64,
}

impl InputLeveler {
    pub fn new(cfg: LevelerConfig) -> Self {
        let lookahead_len = (((cfg.limiter_lookahead_ms / 1000.0) * cfg.sample_rate as f32).round()
            as u32)
            .max(1) as usize;
        Self {
            rms_alpha: cfg.coef(cfg.rms_time_ms),
            boost_alpha: cfg.coef(cfg.boost_time_ms),
            cut_alpha: cfg.coef(cfg.cut_time_ms),
            limiter_attack: cfg.coef(cfg.limiter_attack_ms),
            limiter_release: cfg.coef(cfg.limiter_release_ms),
            cfg,
            env_sq: 0.0,
            block_min_sq: f32::INFINITY,
            block_samples_left: {
                let len = ((cfg.floor_block_ms / 1000.0) * cfg.sample_rate as f32).round() as u32;
                len.max(1)
            },
            block_len: {
                let len = ((cfg.floor_block_ms / 1000.0) * cfg.sample_rate as f32).round() as u32;
                len.max(1)
            },
            floor_blocks: vec![
                f32::INFINITY;
                ((cfg.floor_window_ms / cfg.floor_block_ms).round() as usize).max(1)
            ],
            floor_cursor: 0,
            blocks_committed: 0,
            warmup_left: ((cfg.warmup_ms / 1000.0) * cfg.sample_rate as f32).round() as u32,
            boost_allowed: false,
            gain_db: 0.0,
            limiter_gain: 1.0,
            lookahead_buf: vec![0.0; lookahead_len],
            lookahead_pos: 0,
            bad_samples: 0,
            clipped_samples: 0,
        }
    }

    pub fn with_capture_rate(sample_rate: u32) -> Self {
        Self::new(LevelerConfig::for_capture(sample_rate))
    }

    /// 当前增益（dB）。
    pub fn gain_db(&self) -> f32 {
        self.gain_db
    }

    /// 底噪估计是否就绪：预热结束且 min-stat 窗口测满了至少一个完整窗口。
    /// 未就绪时调理器保持单位增益。
    fn floor_ready(&self) -> bool {
        self.blocks_committed >= self.floor_blocks.len()
    }

    /// 当前底噪估计（dBFS）：最近窗口内分块最小值。**向下偏置**估计
    /// （下界，不是均值）；未就绪时返回 NaN（此时增益必为单位）。
    pub fn noise_floor_dbfs(&self) -> f32 {
        if !self.floor_ready() {
            return f32::NAN;
        }
        10.0 * (self.floor_sq() + 1e-12).log10()
    }

    fn floor_sq(&self) -> f32 {
        self.floor_blocks
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .min(self.block_min_sq)
            .min(1.0)
    }

    /// 非有限输入计数。
    pub fn bad_samples(&self) -> u64 {
        self.bad_samples
    }

    /// 输入达到满刻度的样本数（削顶风险提示）。
    pub fn clipped_samples(&self) -> u64 {
        self.clipped_samples
    }

    /// 处理一个样本（就地返回）。每个输入样本恰有一个输出：前视缓冲
    /// 只把信号整体延迟前视长度（开头补等长零样本），不改变样本数守恒。
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = if x.is_finite() {
            x.clamp(-8.0, 8.0)
        } else {
            self.bad_samples += 1;
            0.0
        };
        if x.abs() >= 0.999 {
            self.clipped_samples += 1;
        }

        self.env_sq += self.rms_alpha * (x * x - self.env_sq);

        // min-stat 底噪：分块内取最小 RMS，再在窗口内取最小。平稳噪声的
        // 最小值就是它自己（比值≈0 dB，不会被判成语音放大）；语音爆发会
        // 明显高于前一段静音/底噪的块最小值，从而被判定为活动。
        // 注意：最小值统计是**向下偏置**的（见模块文档）。
        //
        // 预热期不采样、不调整：它仅用于排除启动瞬态——包络从 0 起升，
        // 最初几毫秒的最小值接近 0，若把它存进窗口，会把底噪估成
        // -120 dBFS，于是平稳噪声被整段当成语音放大（实测 +9.2 dB）。
        // 预热结束后窗口未测满时同样保持单位增益（floor_ready == false）。
        if self.warmup_left > 0 {
            self.warmup_left -= 1;
        } else {
            self.block_min_sq = self.block_min_sq.min(self.env_sq);
            self.block_samples_left -= 1;
            if self.block_samples_left == 0 {
                self.floor_blocks[self.floor_cursor] = self.block_min_sq;
                self.floor_cursor = (self.floor_cursor + 1) % self.floor_blocks.len();
                self.block_min_sq = f32::INFINITY;
                self.block_samples_left = self.block_len;
                if self.blocks_committed < self.floor_blocks.len() {
                    self.blocks_committed += 1;
                }
            }
        }

        let level_db = 10.0 * (self.env_sq + 1e-12).log10();
        let floor_ready = self.floor_ready();
        let floor_db = if floor_ready {
            10.0 * (self.floor_sq() + 1e-12).log10()
        } else {
            f32::NAN
        };

        // SNR 迟滞门控：>10 dB 才允许开始提升，<6 dB 停止提升。
        // 迟滞避免信号在门限附近抖动时增益反复起停。底噪未就绪时
        // 一律不允许（单位增益）。
        if floor_ready {
            let snr_db = level_db - floor_db;
            if self.boost_allowed {
                if snr_db < SNR_BOOST_EXIT_DB {
                    self.boost_allowed = false;
                }
            } else if snr_db > SNR_BOOST_ENTER_DB {
                self.boost_allowed = true;
            }
        } else {
            self.boost_allowed = false;
        }

        // 提升后噪声预算：floor + gain ≤ -50 dBFS（例：底噪 -55 → 最多 +5）。
        let budget_cap_db = if floor_ready {
            (POST_BOOST_NOISE_BUDGET_DBFS - floor_db).max(0.0)
        } else {
            0.0
        };

        if self.boost_allowed {
            // 只升不降：desired 下限是单位增益（max_cut 默认 0），绝不
            // 为了凑目标衰减；再受噪声预算封顶。提升慢（500 ms）、
            // 撤快（20 ms）：响度回升让路，轻声缓慢托起。
            let desired = (self.cfg.target_rms_dbfs - level_db)
                .clamp(-self.cfg.max_cut_db, self.cfg.max_boost_db)
                .min(budget_cap_db);
            let alpha = if desired < self.gain_db {
                self.cut_alpha
            } else {
                self.boost_alpha
            };
            self.gain_db += alpha * (desired - self.gain_db);
        } else if self.gain_db > budget_cap_db {
            // 冻结期间的必要保护：环境变吵（底噪估计上移）导致已冻结的
            // 增益超出噪声预算时，只向下收回超出的部分，绝不再提升。
            self.gain_db += self.cut_alpha * (budget_cap_db - self.gain_db);
        }
        // 其余情况冻结当前增益：静音/低 SNR 期既不提升也不撤，避免
        // 字间停顿把增益抽干（下一个词的开头又被 VAD 丢掉）。

        let leveled = x * 10f32.powf(self.gain_db / 20.0);

        // 峰值限幅（带前视）：信号路径延迟前视长度。环形缓冲始终保存
        // "接下来要输出的全部样本"（含即将离开的当前样本与刚进入的样本），
        // 边链在其上取峰值——大样本进入缓冲后，包络有整整一个前视周期
        // （5 ms ≫ 1 ms 起音）把增益收拢到位，因此平滑包络几乎总是提前
        // 就位；指数逼近的残差由下面的出样守卫兜底。
        let ceiling = 10f32.powf(self.cfg.limiter_ceiling_dbfs / 20.0);
        let slot = self.lookahead_pos;
        let delayed = self.lookahead_buf[slot];
        self.lookahead_buf[slot] = leveled;
        self.lookahead_pos = (slot + 1) % self.lookahead_buf.len();

        let mut peak = delayed.abs();
        for v in self.lookahead_buf.iter() {
            let a = v.abs();
            if a > peak {
                peak = a;
            }
        }
        let target_gain = if peak > ceiling { ceiling / peak } else { 1.0 };
        let l_alpha = if target_gain < self.limiter_gain {
            self.limiter_attack
        } else {
            self.limiter_release
        };
        self.limiter_gain += l_alpha * (target_gain - self.limiter_gain);

        // 出样守卫（无过冲）：包络若尚未完全到位，按即将输出的样本硬性
        // 封顶增益，保证输出峰值绝不超过天花板。
        let out_gain = if delayed.abs() > ceiling {
            self.limiter_gain.min(ceiling / delayed.abs())
        } else {
            self.limiter_gain
        };
        let mut out = delayed * out_gain;
        if out.abs() > ceiling {
            // 浮点末位保护：clamp 到精确上限（超出的部分本来就是要削掉的）。
            out = out.clamp(-ceiling, ceiling);
        }
        if out.is_finite() {
            out
        } else {
            0.0
        }
    }

    /// 处理整块（就地）。
    pub fn process_in_place(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process(*s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: u32 = 16_000;
    /// 5 ms 前视 @16 kHz。前 L 个输出是补零，之后 out[n] == in[n-L]。
    const LOOKAHEAD: usize = 80;

    fn sine(amp: f32, secs: f32, freq: f32) -> Vec<f32> {
        let n = (FS as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / FS as f32;
                amp * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    fn rms_db(samples: &[f32]) -> f32 {
        let sum: f32 = samples.iter().map(|s| s * s).sum();
        10.0 * ((sum / samples.len() as f32) + 1e-12).log10()
    }

    /// 语音般的"说话 0.4 s / 停顿 0.4 s"序列：min-stat 只有在停顿里才看得到
    /// 真实底噪，这也是它区分语音与平稳噪声的前提。
    fn bursty_speech(amp: f32, seconds: f32, freq: f32) -> Vec<f32> {
        let n = (FS as f32 * seconds) as usize;
        let on = (FS as f32 * 0.4) as usize;
        let cycle = (FS as f32 * 0.8) as usize;
        (0..n)
            .map(|i| {
                if i % cycle < on {
                    let t = i as f32 / FS as f32;
                    amp * (2.0 * std::f32::consts::PI * freq * t).sin()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// 精确的"延迟透传"校验：单位增益 + 限幅空闲时，输出必须是输入
    /// 逐比特延迟 LOOKAHEAD 个样本（前视缓冲只延迟、不增减样本）。
    fn assert_delayed_passthrough(out: &[f32], input: &[f32]) {
        assert_eq!(out.len(), input.len(), "样本数守恒");
        assert!(
            out[..LOOKAHEAD].iter().all(|s| *s == 0.0),
            "前视冷启动应补 {LOOKAHEAD} 个零样本"
        );
        assert_eq!(
            out[LOOKAHEAD..],
            input[..input.len() - LOOKAHEAD],
            "单位增益期必须逐比特透传（仅整体延迟）"
        );
    }

    #[test]
    fn quiet_speech_is_lifted_but_capped() {
        let mut leveler = InputLeveler::with_capture_rate(FS);
        // 底噪 2.3 s 才就绪，需给提升留足时间（就绪后按 500 ms 时间常数爬）。
        let mut samples = bursty_speech(0.01, 6.0, 220.0);
        let before = rms_db(&samples);
        leveler.process_in_place(&mut samples);
        let after = rms_db(&samples);
        assert!(
            after > before + 6.0,
            "轻声应被明显提升：{before:.1} -> {after:.1}"
        );
        assert!(
            leveler.gain_db() <= DEFAULT_MAX_BOOST_DB + 0.2,
            "提升必须受 +12 dB 上限约束，实际 {:.1}",
            leveler.gain_db()
        );
    }

    #[test]
    fn loud_speech_is_not_amplified_and_respects_limiter() {
        let mut leveler = InputLeveler::with_capture_rate(FS);
        let mut samples = sine(0.95, 2.0, 220.0);
        leveler.process_in_place(&mut samples);
        assert!(leveler.gain_db() <= 0.1, "大声输入不应被提升");
        let ceiling = 10f32.powf(LIMITER_CEILING_DBFS / 20.0);
        let peak = samples.iter().fold(0.0f32, |a, s| a.max(s.abs()));
        assert!(
            peak <= ceiling * 1.05,
            "峰值应被限制在 {ceiling:.3} 附近，实际 {peak:.3}"
        );
    }

    #[test]
    fn silence_does_not_raise_gain() {
        let mut leveler = InputLeveler::with_capture_rate(FS);
        let mut silence = vec![0.0f32; FS as usize * 2];
        leveler.process_in_place(&mut silence);
        assert!(
            leveler.gain_db().abs() < 0.01,
            "静音期间不应提升增益，实际 {:.3} dB",
            leveler.gain_db()
        );
        assert!(silence.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn noise_floor_is_tracked_so_noise_is_not_boosted() {
        let mut leveler = InputLeveler::with_capture_rate(FS);
        // 稳态 4 s 平底噪（不含语音）：SNR≈0，不应被当成语音持续放大。
        let mut noise: Vec<f32> = (0..FS as usize * 4)
            .map(|i| {
                let t = i as f32 / FS as f32;
                0.02 * (2.0 * std::f32::consts::PI * 100.0 * t).sin()
            })
            .collect();
        leveler.process_in_place(&mut noise);
        assert!(
            leveler.gain_db() < 3.0,
            "纯底噪不应触发大幅提升，实际 {:.1} dB（floor {:.1} dBFS）",
            leveler.gain_db(),
            leveler.noise_floor_dbfs()
        );
    }

    #[test]
    fn gain_moves_slowly_so_it_does_not_pump() {
        let mut leveler = InputLeveler::with_capture_rate(FS);
        // 交替的响/轻段（0.6 s < 底噪就绪 2.3 s，全程单位增益）：增益必须
        // 纹丝不动，绝不按块重拉。
        let mut loud = sine(0.5, 0.3, 220.0);
        let mut quiet = sine(0.01, 0.3, 220.0);
        leveler.process_in_place(&mut loud);
        let g1 = leveler.gain_db();
        leveler.process_in_place(&mut quiet);
        let g2 = leveler.gain_db();
        assert!(
            (g2 - g1).abs() < 6.0,
            "增益跳变过大（{g1:.1} -> {g2:.1} dB），会抽吸"
        );
    }

    #[test]
    fn clipping_is_counted_and_output_stays_finite() {
        let mut leveler = InputLeveler::with_capture_rate(FS);
        let mut samples = sine(1.4, 0.2, 220.0);
        samples[3] = f32::NAN;
        leveler.process_in_place(&mut samples);
        assert!(leveler.bad_samples() >= 1);
        assert!(leveler.clipped_samples() > 0, "应记录削顶样本");
        assert!(samples.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn floor_not_ready_keeps_unit_gain_bit_exact() {
        // 底噪未就绪（1.5 s < 预热 300 ms + 窗口 2 s）：必须精确单位增益，
        // 逐比特透传（仅前视延迟）；底噪报告 NaN（不得用 0/极小值充当）。
        let mut leveler = InputLeveler::with_capture_rate(FS);
        let input = sine(0.01, 1.5, 220.0);
        let mut out = input.clone();
        leveler.process_in_place(&mut out);
        assert_eq!(leveler.gain_db(), 0.0, "未就绪必须是精确单位增益");
        assert!(
            leveler.noise_floor_dbfs().is_nan(),
            "未就绪时底噪估计应报告无效（NaN）"
        );
        assert!(!leveler.floor_ready());
        assert_delayed_passthrough(&out, &input);
    }

    #[test]
    fn loud_input_stays_at_unit_gain_before_and_after_floor_ready() {
        // 预热刚结束（350 ms）、窗口将满（2.2 s）以及就绪后（3 s）三个探针：
        // 大声输入（-9 dBFS，远高于目标）全程单位增益——绝不为了凑目标
        // 衰减（max_cut = 0）；连续大声自身成为 min-stat 底噪，SNR≈0 也
        // 不允许提升。
        let mut leveler = InputLeveler::with_capture_rate(FS);
        let input = sine(0.5, 3.0, 220.0);
        let probe_at = |ms: usize| (FS as f32 * ms as f32 / 1000.0) as usize;
        let probes = [probe_at(350), probe_at(2_200), probe_at(2_990)];
        let mut hit = 0usize;
        let mut out = Vec::with_capacity(input.len());
        for (i, s) in input.iter().enumerate() {
            out.push(leveler.process(*s));
            if hit < probes.len() && i + 1 == probes[hit] {
                assert_eq!(
                    leveler.gain_db(),
                    0.0,
                    "探针 {} 处必须仍单位增益",
                    ["350ms", "2.2s", "2.99s"][hit]
                );
                hit += 1;
            }
        }
        assert_eq!(hit, probes.len(), "所有探针都应被触发");
        assert!(leveler.floor_ready(), "3 s 后窗口应已测满");
        assert_eq!(leveler.gain_db(), 0.0);
        assert_delayed_passthrough(&out, &input);
    }

    #[test]
    fn quiet_to_loud_withdraws_gain_fast_without_overshoot() {
        let mut leveler = InputLeveler::with_capture_rate(FS);
        // 先建立 ~+12 dB 增益（quiet 语音，停顿为数字零 → 底噪极低）。
        let mut quiet = bursty_speech(0.01, 6.0, 220.0);
        leveler.process_in_place(&mut quiet);
        let g0 = leveler.gain_db();
        assert!(g0 > 9.0, "前置条件：quiet 段应已建立高增益，实际 {g0:.1}");

        // 突然大声：增益应以 ~20 ms 时间常数撤回，过渡期不得过冲天花板。
        let ceiling = 10f32.powf(LIMITER_CEILING_DBFS / 20.0);
        let loud = sine(0.9, 0.15, 220.0);
        let mut t_drop6: Option<usize> = None;
        let mut t_to_1db: Option<usize> = None;
        let mut peak = 0.0f32;
        for (i, s) in loud.iter().enumerate() {
            let y = leveler.process(*s);
            peak = peak.max(y.abs());
            let g = leveler.gain_db();
            if t_drop6.is_none() && g0 - g >= 6.0 {
                t_drop6 = Some(i);
            }
            if t_to_1db.is_none() && g <= 1.0 {
                t_to_1db = Some(i);
            }
        }
        let t20 = (FS as f32 * 0.020) as usize;
        let t60 = (FS as f32 * 0.060) as usize;
        assert!(
            t_drop6.unwrap_or(usize::MAX) <= t20,
            "20 ms 内应已撤 ≥6 dB（τ=20 ms → 1τ 内撤 63%）"
        );
        assert!(
            t_to_1db.unwrap_or(usize::MAX) <= t60,
            "60 ms 内应撤回到 ≤1 dB（τ=20 ms → 3τ 内撤 95%）"
        );
        assert!(
            peak <= ceiling,
            "撤增益过渡期输出不得过冲：{peak:.4} > {ceiling:.4}"
        );
    }

    #[test]
    fn lookahead_limiter_conserves_samples_and_holds_ceiling() {
        let mut leveler = InputLeveler::with_capture_rate(FS);
        // 大幅突发（超天花板 2 倍）+ 短间隙，按非整帧分块喂入：前视缓冲
        // 不得改变样本数守恒，输出峰值绝不超过天花板（无过冲）。
        let mut input: Vec<f32> = Vec::new();
        for _ in 0..6 {
            input.extend(sine(1.5, 0.05, 300.0));
            input.extend(vec![0.001f32; (FS as f32 * 0.02) as usize]);
        }
        let n = input.len();
        let mut out: Vec<f32> = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let end = (i + 997).min(n);
            let mut chunk = input[i..end].to_vec();
            leveler.process_in_place(&mut chunk);
            out.extend_from_slice(&chunk);
            i = end;
        }
        assert_eq!(out.len(), n, "前视缓冲不得改变样本数守恒");
        let ceiling = 10f32.powf(LIMITER_CEILING_DBFS / 20.0);
        let peak = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
        assert!(
            peak <= ceiling,
            "前视限幅输出峰值 {peak:.6} 超过天花板 {ceiling:.6}"
        );
        assert!(out.iter().all(|s| s.is_finite()));
    }

    /// 构造已知 SNR 的确定性信号：底噪正弦（env=1e-8 → -80 dBFS），
    /// 测试信号比它高 snr_db dB（正弦 env 精确收敛于 amp²/2）。
    const FLOOR_AMP: f32 = 1.414e-4;

    fn amp_at(snr_db: f32) -> f32 {
        FLOOR_AMP * 10f32.powf(snr_db / 20.0)
    }

    #[test]
    fn snr_hysteresis_gates_boosting() {
        let mut leveler = InputLeveler::with_capture_rate(FS);
        let mut priming = sine(FLOOR_AMP, 2.4, 100.0);
        leveler.process_in_place(&mut priming);
        assert!(leveler.floor_ready(), "2.4 s 后底噪窗口应已测满");
        assert_eq!(leveler.gain_db(), 0.0, "SNR≈0 不得提升");

        // A：SNR = +10.5 dB（>10）→ 允许提升。
        let mut a = sine(amp_at(10.5), 0.7, 220.0);
        leveler.process_in_place(&mut a);
        let g_a = leveler.gain_db();
        assert!(g_a > 3.0, "SNR>10 dB 应允许提升，实际 {g_a:.2}");

        // B：SNR = +8 dB（6..10 迟滞区间）→ 保持提升（不退出）。
        let g_b_start = g_a;
        let mut b = sine(amp_at(8.0), 0.35, 220.0);
        leveler.process_in_place(&mut b);
        let g_b = leveler.gain_db();
        assert!(
            g_b > g_b_start + 0.3,
            "迟滞区间应保持提升：{g_b_start:.2} -> {g_b:.2}"
        );

        // C：SNR = +5 dB（<6）→ 停止提升（冻结，不撤也不升）。
        let c = sine(amp_at(5.0), 0.35, 220.0);
        let mut g_c_ref = None;
        for (i, s) in c.iter().enumerate() {
            leveler.process(*s);
            if i == (FS as f32 * 0.1) as usize {
                g_c_ref = Some(leveler.gain_db());
            }
        }
        let g_c_ref = g_c_ref.expect("C 段内应已采样参考增益");
        let g_c = leveler.gain_db();
        assert!(
            g_c <= g_c_ref + 0.05,
            "SNR<6 dB 退出后不得继续提升：ref {g_c_ref:.2} end {g_c:.2}"
        );
        assert!(
            g_c >= g_c_ref - 0.2,
            "退出是冻结而非撤回：ref {g_c_ref:.2} end {g_c:.2}"
        );

        // D：回到 SNR = +8 dB（未超 10）→ 不得重新进入提升。
        let g_d_start = leveler.gain_db();
        let mut d = sine(amp_at(8.0), 0.35, 220.0);
        leveler.process_in_place(&mut d);
        let g_d = leveler.gain_db();
        assert!(
            g_d <= g_d_start + 0.05,
            "8 dB 不应重新触发提升：{g_d_start:.2} -> {g_d:.2}"
        );
    }

    #[test]
    fn snr_below_enter_threshold_never_boosts() {
        // SNR = +8 dB（介于 6 与 10 之间）：从未进入提升态，增益纹丝不动。
        let mut leveler = InputLeveler::with_capture_rate(FS);
        let mut priming = sine(FLOOR_AMP, 2.4, 100.0);
        leveler.process_in_place(&mut priming);
        let mut sig = sine(amp_at(8.0), 1.0, 220.0);
        leveler.process_in_place(&mut sig);
        assert!(
            leveler.gain_db().abs() < 1e-6,
            "SNR 8 dB（<10）不得开始提升，实际 {:.3}",
            leveler.gain_db()
        );
    }
}
