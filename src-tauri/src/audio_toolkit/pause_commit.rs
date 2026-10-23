//! 停顿提交（pause-commit）分段器：长口述按自然停顿切成可独立转写的段。
//!
//! # 设计依据
//!
//! - 只在**确认静音**处切分（默认 400 ms **真实静音**），比流式路径的
//!   1650 ms 更早，用户能更快看到文字。分段器的输入是 `SmoothedVad`
//!   **平滑后**的判定：语音结束后，离线策略的 hangover（450 ms 折算
//!   整帧后 480 ms）里真实静音仍被判成 Speech。若分段器再在自己的时钟
//!   上从 0 数静音，用户要停 ~900 ms（480 hangover + 400 确认）才会提交
//!   一次——正常停顿根本够不着，于是整段只在录音结束时作为尾段提交。
//!   因此分段器在每个语音→静音跳变处把 hangover 一次性记入**已流逝的
//!   真实静音**（[`PauseCommitConfig::hangover_ms`]），让所有阈值都回到
//!   "从最后一个真实有声帧起算"的时间轴；
//! - 短于 ~1 s 的片段不立刻提交，等 800 ms 静音再说——避免把"嗯…"这类
//!   停顿当成一句话切碎；
//! - 连续说话超过 seek 阈值（20 s）后，遇到 ≥200 ms 的停顿就提交
//!   （[`PauseCommitConfig::seek_pause_silence_ms`]），不必等满常规确认
//!   静音；到 30 s 仍无停顿则强制切一刀（`forced`），保住单次解码上限；
//! - 每个采样点只属于一个段：段边界记录绝对样本位置，不做重叠、
//!   不做文本层去重（"非常非常"这类合法重复不能被吞）；
//! - 记账按**实际有效样本数**推进（[`PauseCommitSegmenter::push_frame_len`]）：
//!   采集尾块的 `valid_frames` 可能小于一个完整 VAD 帧（输入缓冲的零
//!   填充已被 sink 裁掉），若恒按整帧推进，`position` 会越过缓冲末端，
//!   finish 的尾段切片失败而**整段丢失**。
//!
//! # 它解决什么、不解决什么
//!
//! 解决的是**分段记账**：不会漏字、不会因重叠重复出字、不会把上一段
//! 的尾巴再次送给模型。**不解决**跨停顿的语义损失：断句后的专名、
//! 未说完的从句、标点与大小写会变差。要量化这种损失必须用同一批
//! 录音做整段 vs 分段的 A/B（见 `docs/chinese-evaluation.md`）。
//!
//! 因此本模块只提供分段事实，是否启用、如何上屏由上层决定；生产默认
//! 仍是整段转写。
//!
//! # 为什么扣除 hangover，而不是喂"未平滑的原始判定"
//!
//! 拿到原始判定只有两条来路，都不划算：把内层 VAD 再跑一遍——Silero
//! 带 LSTM 状态，同一帧推两次会破坏状态；或给 `VoiceActivityDetector`
//! trait 增加第二个逐帧回调——牵动所有后端与长录音打分路径。而
//! hangover 是常量且精确已知（`CaptureVad::hangover_frames_for`，sink
//! 启动时按策略装上），"平滑判定 + 跳变处记入 hangover"逐帧等价于
//! 原始判定的时间轴，还保留了平滑判定自带的 onset 确认（连续有声才
//! 算开口），比逐帧原始判定更稳。

/// 分段策略（默认值来自 sol 的建议起点，不是实验最优值）。
#[derive(Debug, Clone, Copy)]
pub struct PauseCommitConfig {
    /// 确认静音多久算一次停顿（毫秒，指**真实静音**：hangover 信用已
    /// 折算，见 [`PauseCommitConfig::hangover_ms`]）。
    pub silence_ms: u32,
    /// 短句阈值：短于它的片段要等更长的静音才提交。
    pub short_utterance_ms: u32,
    /// 短片段等待的更长静音（毫秒，真实静音）。
    pub short_utterance_silence_ms: u32,
    /// 单段硬上限；到点后即使没停顿也强制切分。
    pub max_segment_ms: u32,
    /// 超过它之后遇到停顿就提交，不必等满上限。
    pub seek_pause_ms: u32,
    /// 已超过 [`PauseCommitConfig::seek_pause_ms`] 的长段遇到停顿所需的
    /// **更短**确认静音（毫秒，真实静音）：说满 20 s 后用户稍作停顿
    /// （≥200 ms）就该出字，不必再等满 `silence_ms`。
    pub seek_pause_silence_ms: u32,
    /// 输入判定中已含的 VAD hangover（毫秒）。`SmoothedVad` 在语音
    /// 结束后的 hangover 期间仍判 Speech，所以每个语音→静音跳变发生
    /// 时，真实静音其实已经流逝了 hangover + 1 帧；分段器把这 hangover
    /// 一次性记入静音计数（按样本数精确折算），`silence_ms` 等阈值从而
    /// 始终指**真实静音**。0 = 输入是未平滑的原始判定，不记信用。生产
    /// 路径按实际帧长折算后传入（`CaptureVad::hangover_frames_for` ×
    /// 帧长），默认值仅约等于 `VAD_OFFLINE_HANGOVER_MS`。
    pub hangover_ms: u32,
}

impl Default for PauseCommitConfig {
    fn default() -> Self {
        Self {
            silence_ms: 400,
            short_utterance_ms: 1_000,
            short_utterance_silence_ms: 800,
            max_segment_ms: 30_000,
            seek_pause_ms: 20_000,
            seek_pause_silence_ms: 200,
            hangover_ms: 450,
        }
    }
}

impl PauseCommitConfig {
    /// `required_ms` 档的确认静音在**真实时间轴**上是多少毫秒：静音只随
    /// 非语音判定的到达而累计——第一帧非语音判定到达时，真实静音已流逝
    /// `hangover + 1` 帧，此后每帧累加实际样本数；提交点是第一个达到
    /// 阈值的累计值。与 [`PauseCommitSegmenter::push_frame_len`] 的记账
    /// 一致，供启动日志与测试断言（"触发的静音阈值 ms"）。
    pub fn real_silence_ms(&self, required_ms: u32, frame_samples: usize) -> u32 {
        let required = ms_to_samples(required_ms);
        let hangover = ms_to_samples(self.hangover_ms);
        let credited = hangover + frame_samples;
        if credited >= required {
            return samples_to_ms(credited);
        }
        // 之后每帧累加：提交点 = hangover + ceil((required − hangover)/帧长) 帧。
        let frames = (required - hangover + frame_samples - 1) / frame_samples;
        samples_to_ms(hangover + frames * frame_samples)
    }
}

/// 毫秒 → 16 kHz 样本数（精确：16_000/1_000 = 16，无舍入）。
const fn ms_to_samples(ms: u32) -> usize {
    ms as usize * 16
}

/// 样本数 → 毫秒（16 kHz，向下取整）。
const fn samples_to_ms(samples: usize) -> u32 {
    (samples / 16) as u32
}

/// 一个已确认可以提交的片段（样本索引均为 16 kHz 绝对位置）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommittedSegment {
    /// 段首样本（含）。
    pub start: usize,
    /// 段尾样本（不含）：等于最后一个**被判为语音**的帧的结束位置
    /// （平滑判定下含 hangover 尾部，转写时多一点尾部静音无害）。
    pub end: usize,
    /// 是否因达到硬上限被强制切分（边界可能落在词中间）。
    pub forced: bool,
}

impl CommittedSegment {
    pub fn samples(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// 分段器的当前状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// 还没开始说话。
    Idle,
    /// 段进行中；`silence_samples` 为当前连续**真实静音**的样本数
    /// （含语音→静音跳变处一次性记入的 hangover 信用）。
    InSegment { silence_samples: usize },
}

/// 按 VAD 判定逐帧推进的分段器。
///
/// 调用方只需提供"这一帧是否为语音"和帧长（样本数），样本位置与
/// 静音计时都由本结构维护，因此与具体 VAD 后端无关、可精确单测。
/// 静音按**实际样本数**累计（不是帧数）：尾块可以短于一帧，帧数计数
/// 会让"真实静音"口径漂移。
#[derive(Debug)]
pub struct PauseCommitSegmenter {
    cfg: PauseCommitConfig,
    frame_samples: usize,
    /// 语音→静音跳变时一次性记入的 hangover 样本数（`cfg.hangover_ms`）。
    hangover_samples: usize,
    state: State,
    /// 当前段起点（绝对样本位置）。
    segment_start: usize,
    /// 最后一个有声样本的结束位置（绝对）。
    last_voiced_end: usize,
    /// 已喂入的样本总数。
    position: usize,
    /// 已提交到的位置，用于诊断断言。
    committed_until: usize,
}

impl PauseCommitSegmenter {
    pub fn new(cfg: PauseCommitConfig, frame_samples: usize) -> Self {
        assert!(frame_samples > 0, "frame size must be non-zero");
        Self {
            cfg,
            frame_samples,
            hangover_samples: ms_to_samples(cfg.hangover_ms),
            state: State::Idle,
            segment_start: 0,
            last_voiced_end: 0,
            position: 0,
            committed_until: 0,
        }
    }

    /// 当前已喂入样本数。
    pub fn position(&self) -> usize {
        self.position
    }

    /// 已提交到的样本位置（诊断/测试用）。
    pub fn committed_until(&self) -> usize {
        self.committed_until
    }

    /// 推进一个完整 VAD 帧（构造时的 `frame_samples` 个样本）。
    pub fn push_frame(&mut self, voiced: bool) -> Option<CommittedSegment> {
        self.push_frame_len(voiced, self.frame_samples)
    }

    /// 推进一帧，帧长为 `sample_len` 个**实际有效**样本。
    ///
    /// 采集尾块的 `valid_frames` 可能小于 `frame_samples`（输入缓冲的
    /// 零填充已被 sink 裁掉）：记账必须按实际长度推进——恒按整帧推进会
    /// 让 `position` 越过缓冲末端，finish 的尾段切片失败而**整段丢失**。
    /// 静音同样只累计 `sample_len` 个真实样本。`sample_len` 为 0 时忽略
    /// （空块不推进、不判定）。返回 `Some(segment)` 表示该段已可提交。
    pub fn push_frame_len(&mut self, voiced: bool, sample_len: usize) -> Option<CommittedSegment> {
        if sample_len == 0 {
            return None;
        }
        let frame_start = self.position;
        let frame_end = frame_start + sample_len;

        if voiced {
            if matches!(self.state, State::Idle) {
                self.segment_start = frame_start;
            }
            self.state = State::InSegment { silence_samples: 0 };
            self.last_voiced_end = frame_end;
            self.position = frame_end;

            // 硬上限兜底：即使一直没有停顿也要切，边界落在词中间
            // （forced=true，上层应记为不可靠边界）。
            let segment_samples = self.last_voiced_end - self.segment_start;
            if segment_samples >= ms_to_samples(self.cfg.max_segment_ms) {
                return Some(self.commit(true));
            }
            return None;
        }

        // 非语音帧：语音尚未开始，什么都不用做。
        let State::InSegment { silence_samples } = self.state else {
            self.position = frame_end;
            return None;
        };

        // 语音→静音跳变（silence_samples == 0 ⇒ 上一帧判语音）：输入判定
        // 在 hangover 期间仍报语音，真实静音其实已流逝 hangover 样本，
        // 一次性记入（hangover_ms = 0 → 无信用，等价于原始判定时间轴）。
        let silence_samples = if silence_samples == 0 {
            self.hangover_samples + sample_len
        } else {
            silence_samples + sample_len
        };
        self.state = State::InSegment { silence_samples };
        self.position = frame_end;

        let segment_samples = self.last_voiced_end - self.segment_start;
        let segment_ms = samples_to_ms(segment_samples);
        // 三档确认静音（均指真实静音）：说满 seek 阈值的长段遇到短停顿
        // 即提交；短句要多等；其余按 silence_ms。
        let required_ms = if segment_ms >= self.cfg.seek_pause_ms {
            self.cfg.seek_pause_silence_ms
        } else if segment_ms < self.cfg.short_utterance_ms {
            self.cfg.short_utterance_silence_ms
        } else {
            self.cfg.silence_ms
        };
        if silence_samples >= ms_to_samples(required_ms) {
            return Some(self.commit(false));
        }
        None
    }

    /// 录制结束：把剩下的部分作为一段提交（若还有语音）。最后一个块
    /// 是短于整帧的尾块时，其全部有效样本已计入 `last_voiced_end`，
    /// 尾段照常提交、不丢样本。
    pub fn finish(&mut self) -> Option<CommittedSegment> {
        if matches!(self.state, State::Idle) || self.last_voiced_end <= self.segment_start {
            self.state = State::Idle;
            return None;
        }
        Some(self.commit(false))
    }

    fn commit(&mut self, forced: bool) -> CommittedSegment {
        let segment = CommittedSegment {
            start: self.segment_start,
            end: self.last_voiced_end,
            forced,
        };
        debug_assert!(segment.start >= self.committed_until, "段不得重叠");
        debug_assert!(segment.end > segment.start, "段必须非空");
        self.committed_until = segment.end;
        self.state = State::Idle;
        self.segment_start = self.last_voiced_end;
        segment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: usize = 512; // 常见 VAD 帧长（32 ms）

    /// 生产档配置：Silero 512 样本帧，hangover = ceil(450 ms) = 15 帧
    /// = 480 ms（`frames_for_duration_ms` 向上取整， managers/audio.rs
    /// 按实际帧长折算后传入）。
    fn production_cfg() -> PauseCommitConfig {
        PauseCommitConfig {
            hangover_ms: 480,
            ..PauseCommitConfig::default()
        }
    }

    /// 按毫秒生成 voiced 帧序列（**原始判定**时间轴，无 hangover），
    ///方便构造场景。
    fn frames(voiced_ms: u32, silent_ms: u32) -> Vec<bool> {
        let mut out = Vec::new();
        out.extend(std::iter::repeat(true).take((voiced_ms as usize / 32).max(0)));
        out.extend(std::iter::repeat(false).take((silent_ms as usize / 32).max(0)));
        out
    }

    /// 生成**平滑后**判定的帧流（tap 实际收到的序列）：真实语音判
    /// Speech；其后 hangover_ms 的真实静音仍判 Speech；再之后判 Noise。
    fn smoothed(voiced_ms: u32, hangover_ms: u32, silent_ms: u32) -> Vec<bool> {
        let mut out = Vec::new();
        out.extend(std::iter::repeat(true).take((voiced_ms as usize / 32).max(0)));
        out.extend(std::iter::repeat(true).take((hangover_ms as usize / 32).max(0)));
        out.extend(std::iter::repeat(false).take((silent_ms as usize / 32).max(0)));
        out
    }

    fn run(segmenter: &mut PauseCommitSegmenter, script: &[bool]) -> Vec<CommittedSegment> {
        let mut out = Vec::new();
        for &voiced in script {
            if let Some(seg) = segmenter.push_frame(voiced) {
                out.push(seg);
            }
        }
        if let Some(seg) = segmenter.finish() {
            out.push(seg);
        }
        out
    }

    #[test]
    fn ordinary_utterance_commits_after_default_silence() {
        let mut seg = PauseCommitSegmenter::new(production_cfg(), FRAME);
        // 5 s 语音 + 480 ms 仍判 Speech 的 hangover + 600 ms 判 Noise 的
        // 静音：第一帧 Noise（真实静音 512 ms ≥ 400 ms）就应提交一次。
        let script = smoothed(5_000, 480, 600);
        let out = run(&mut seg, &script);
        assert_eq!(out.len(), 1, "应只提交一段：{out:?}");
        assert!(!out[0].forced);
        let ms = samples_to_ms(out[0].samples());
        // 4992 ms 语音（156 帧）+ 480 ms hangover 尾 = 5472 ms。
        assert!(
            (5_350..=5_600).contains(&ms),
            "段长应≈语音+hangover 尾，实际 {ms} ms"
        );
        assert_eq!(out[0].start, 0, "段首应从第一个有声帧开始");
    }

    #[test]
    fn half_second_pause_commits_promptly() {
        // 修复目标：~0.5 s 的真实停顿就要提交。tap 送来的是平滑判定
        // ——480 ms hangover 仍判 Speech，第 512 ms 才出现第一帧 Noise；
        // hangover 信用让这一帧立即满足 400 ms 阈值（旧实现要再等
        // 450 ms 判 Noise，共 ~928 ms 真实静音，用户停顿根本够不着）。
        let cfg = production_cfg();
        assert_eq!(
            cfg.real_silence_ms(cfg.silence_ms, FRAME),
            512,
            "新阈值：512 ms 真实静音（512 样本帧）"
        );

        let mut seg = PauseCommitSegmenter::new(cfg, FRAME);
        // 4 s 语音 + 480 ms 判 Speech 的 hangover：真实静音 480 ms，
        // 不提交。
        for &v in &smoothed(4_000, 480, 0) {
            assert!(seg.push_frame(v).is_none(), "hangover 期间不应提交");
        }
        // 第 512 ms 真实静音：第一帧 Noise 判定 → 立即提交。
        let out = seg.push_frame(false).expect("0.5 s 真实停顿必须提交");
        assert!(!out.forced);
        let ms = samples_to_ms(out.samples());
        // 3997 ms 语音（125 帧）+ 480 ms hangover 尾 = 4480 ms。
        assert!((4_350..=4_600).contains(&ms), "实际 {ms} ms");
        assert_eq!(out.start, 0);
    }

    #[test]
    fn short_fragment_waits_for_longer_silence() {
        let cfg = production_cfg();
        let mut seg = PauseCommitSegmenter::new(cfg, FRAME);
        // 短句确认静音 800 ms（12800 样本）− hangover 信用 480 ms
        // （7680 样本）= 还需 5120 样本判 Noise = 10 帧。
        let need = (ms_to_samples(cfg.short_utterance_silence_ms) - ms_to_samples(cfg.hangover_ms))
            / FRAME;
        assert_eq!(need, 10, "测试前提：800 − 480 = 320 ms = 10 帧");
        // 0.5 s 语音 + hangover + 到阈值前 1 帧的 Noise 静音：短片段要等
        // ~800 ms 真实静音，先不提交。
        for &v in &smoothed(500, 480, ((need - 1) * 32) as u32) {
            assert!(
                seg.push_frame(v).is_none(),
                "短片段在 800 ms 真实静音前不应提交"
            );
        }
        // 再补 1 帧 Noise → 累计 800 ms 真实静音，达到短句等待阈值。
        let mut committed = None;
        for &v in &frames(0, 32) {
            if let Some(s) = seg.push_frame(v) {
                committed = Some(s);
            }
        }
        let seg_out = committed.expect("短片段应在更长真实静音后提交");
        assert!(!seg_out.forced);
    }

    #[test]
    fn long_dictation_commits_at_the_first_pause_after_seek_threshold() {
        let mut seg = PauseCommitSegmenter::new(production_cfg(), FRAME);
        let out = run(&mut seg, &smoothed(22_000, 480, 600));
        assert_eq!(out.len(), 1);
        assert!(!out[0].forced, "有自然停顿时不应强制切分");
        let ms = samples_to_ms(out[0].samples());
        // 22 s 语音（687 帧）+ 480 ms hangover 尾 = 22464 ms。
        assert!((22_300..=22_600).contains(&ms), "实际 {ms} ms");
    }

    #[test]
    fn seek_pause_tier_needs_only_a_short_pause() {
        // W2：说满 seek 阈值（20 s）后，≥200 ms 真实静音即可提交，不必
        // 等满常规 silence_ms（400 ms）。用原始判定时间轴（hangover 0）
        // 单独验证这一档。
        let cfg = PauseCommitConfig {
            hangover_ms: 0,
            ..PauseCommitConfig::default()
        };
        // 常规段（5 s）：224 ms 静音 < 400 ms，不提交。
        let mut seg = PauseCommitSegmenter::new(cfg, FRAME);
        for &v in &frames(5_000, 224) {
            assert!(seg.push_frame(v).is_none(), "常规段仍需 400 ms 静音");
        }
        // 长段（21 s）：192 ms 静音（6 帧 = 3072 < 3200）仍不提交，
        // 第 7 帧（224 ms = 3584 ≥ 3200）提交。
        let mut seg = PauseCommitSegmenter::new(cfg, FRAME);
        let mut committed = None;
        for &v in &frames(21_000, 224) {
            if let Some(s) = seg.push_frame(v) {
                committed = Some(s);
            }
        }
        let out = committed.expect("长段的 224 ms 停顿必须提交");
        assert!(!out.forced);
        assert_eq!(out.end, (21_000 / 32) * FRAME, "段尾应停在最后一个有声样本");
    }

    #[test]
    fn continuous_speech_is_force_cut_at_the_cap() {
        let mut seg = PauseCommitSegmenter::new(production_cfg(), FRAME);
        let script = vec![true; 40_000 / 32];
        let out = run(&mut seg, &script);
        assert!(!out.is_empty(), "无停顿时必须强制切分");
        assert!(out[0].forced, "第一次切分应是 forced");
        let ms = samples_to_ms(out[0].samples());
        assert!(
            (29_900..=30_200).contains(&ms),
            "强制切分应贴着 30 s 上限，实际 {ms} ms"
        );
    }

    #[test]
    fn real_silence_ms_matches_the_actual_commit_point() {
        // 阈值声明（real_silence_ms，进启动日志）必须与真实帧流上的
        // 首次提交点一致：512 样本帧（Silero）与 256 样本帧（Earshot）
        // 两种生产帧长都验证。
        for (frame, hangover_ms) in [(512, 480), (256, 464)] {
            let frame_ms = (frame * 1000 / 16_000) as u32;
            let cfg = PauseCommitConfig {
                hangover_ms,
                ..PauseCommitConfig::default()
            };
            let expected = cfg.real_silence_ms(cfg.silence_ms, frame);
            let mut seg = PauseCommitSegmenter::new(cfg, frame);

            // 4 s 语音（帧对齐）。
            for _ in 0..(4_000 / frame_ms) {
                seg.push_frame(true);
            }
            // hangover 帧：真实静音已流逝，但仍判 Speech（两个生产档的
            // hangover 样本数都恰好是整帧）。
            let hangover_frames = ms_to_samples(hangover_ms) / frame;
            for _ in 0..hangover_frames {
                assert!(seg.push_frame(true).is_none(), "frame={frame}");
            }
            // 逐帧 Noise，记录首次提交时的真实静音时长。
            let mut real_silence_ms = hangover_ms;
            let mut committed_at = None;
            for _ in 0..64 {
                real_silence_ms += frame_ms;
                if seg.push_frame(false).is_some() {
                    committed_at = Some(real_silence_ms);
                    break;
                }
            }
            assert_eq!(
                committed_at,
                Some(expected),
                "frame={frame}: 声明阈值应等于实际首帧提交点"
            );
        }
    }

    #[test]
    fn segments_never_overlap_and_cover_every_voiced_sample() {
        let mut seg = PauseCommitSegmenter::new(production_cfg(), FRAME);
        let mut script = Vec::new();
        script.extend(frames(4_000, 700));
        script.extend(frames(6_000, 1_200));
        script.extend(frames(2_000, 800));
        let out = run(&mut seg, &script);
        assert!(out.len() >= 3, "三段口述应各提交一次：{out:?}");
        let mut prev_end = 0;
        for s in &out {
            assert!(s.start >= prev_end, "段之间不得重叠");
            assert!(s.end > s.start);
            prev_end = s.end;
        }
        let voiced_samples: usize = script.iter().filter(|v| **v).count() * FRAME;
        let covered: usize = out.iter().map(|s| s.samples()).sum();
        assert_eq!(
            covered, voiced_samples,
            "每个有声样本必须恰好属于一个段（无遗漏也无重复）"
        );
    }

    #[test]
    fn silence_only_produces_no_segments() {
        let mut seg = PauseCommitSegmenter::new(PauseCommitConfig::default(), FRAME);
        let out = run(&mut seg, &vec![false; 200]);
        assert!(out.is_empty());
        assert_eq!(seg.committed_until(), 0);
    }

    #[test]
    fn configurable_thresholds_are_honoured() {
        let cfg = PauseCommitConfig {
            silence_ms: 200,
            short_utterance_ms: 1_000,
            short_utterance_silence_ms: 400,
            max_segment_ms: 10_000,
            seek_pause_ms: 5_000,
            seek_pause_silence_ms: 100,
            // 原始判定时间轴（无 hangover 信用）：行为与旧分段器一致。
            hangover_ms: 0,
        };
        // 200 ms 阈值按样本精确记账：6 帧（192 ms）不够、7 帧（224 ms）提交。
        assert_eq!(cfg.real_silence_ms(cfg.silence_ms, FRAME), 224);
        let mut seg = PauseCommitSegmenter::new(cfg, FRAME);
        let out = run(&mut seg, &frames(2_000, 300));
        assert_eq!(out.len(), 1, "自定义 200 ms 停顿应更早提交");
        assert!(!out[0].forced);
    }

    #[test]
    fn partial_tail_frame_keeps_accounting_exact_and_commits_the_tail() {
        // W1 回归：采集尾块 valid_frames 可能小于一个完整 VAD 帧（输入
        // 缓冲的零填充被裁掉）。分段器必须按实际样本数推进 position，
        // finish 时把剩余有效样本全部计入尾段——多记（恒按整帧）会让
        // 尾段区间越过缓冲末端而整段丢失，少记会丢样本。
        let mut seg = PauseCommitSegmenter::new(production_cfg(), FRAME);
        // 2 s 语音（整帧）+ 300 样本的有效尾块（< 512）。
        for _ in 0..(2_000 / 32) {
            assert!(seg.push_frame(true).is_none());
        }
        assert!(seg.push_frame_len(true, 300).is_none());
        assert_eq!(
            seg.position(),
            62 * FRAME + 300,
            "position 必须按实际有效样本数推进"
        );
        let out = seg.finish().expect("含短尾块的尾段必须提交");
        assert!(!out.forced);
        assert_eq!(out.start, 0);
        assert_eq!(out.end, 62 * FRAME + 300, "尾段必须包含全部有效样本");
        assert_eq!(seg.committed_until(), out.end);
    }

    #[test]
    fn partial_silent_frame_counts_only_its_real_samples() {
        // 静音按实际样本数累计：12 整帧（6144）+ 200 样本短帧 = 6344 <
        // 6400（400 ms）不提交；若错误地按整帧计（6144 + 512 = 6656 ≥
        // 6400）就会提前提交。
        let cfg = PauseCommitConfig {
            hangover_ms: 0,
            ..PauseCommitConfig::default()
        };
        let mut seg = PauseCommitSegmenter::new(cfg, FRAME);
        for _ in 0..(2_000 / 32) {
            assert!(seg.push_frame(true).is_none());
        }
        for _ in 0..12 {
            assert!(seg.push_frame(false).is_none(), "6144 样本静音 < 400 ms");
        }
        assert!(
            seg.push_frame_len(false, 200).is_none(),
            "短帧只按 200 样本计：6344 < 6400，不得提交"
        );
        let out = seg.push_frame(false).expect("再一整帧即 6856 ≥ 6400，提交");
        assert!(!out.forced);
        assert_eq!(out.end, 62 * FRAME, "段尾应停在最后一个有声样本");
        // 空块：不推进、不判定。
        let before = seg.position();
        assert!(seg.push_frame_len(true, 0).is_none());
        assert_eq!(seg.position(), before);
    }
}
