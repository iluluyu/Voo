//! 批处理（`VadPolicy::Offline`）录音缓冲的「连续语句跨度」保留状态机。
//!
//! # 为什么存在
//!
//! 旧 Offline 路径只把 VAD 判为语音的帧写进录音缓冲：语句内的短停顿被
//! 抽走，词首/词尾可能被截断——对批处理 ASR 是破坏性裁剪（sol 结论）。
//! 本模块把保留单位从「帧」升级为「语句区间」：
//!
//! * VAD 一旦确认进入语句，就从「语句起点 − 前导」开始保留连续原始
//!   样本（语句内短停顿原样保留），直到「语句结束 + 尾随」；
//! * 只有连续静音 ≥ 断开长度才结束区间：结束时把停顿开头的尾随一段
//!   保留下来，其余静音丢弃，下一个语句重新从前导开始；
//! * 相邻两个区间的保留在静音里不重叠：前一段最多多留 `tail`、后一段
//!   最多回看 `preroll`，构造时断言 `preroll + tail ≤ break`。
//!
//! # 时间轴与 hangover 折算
//!
//! 输入判定是 `SmoothedVad` 的**平滑后**输出：语音结束后 hangover 期间
//! 的真实静音仍判 Speech。为了让断开长度等常量始终指**真实静音**（与
//! `PauseCommitConfig::hangover_ms` 同一约定），每个语音→静音跳变处把
//! hangover 一次性记入已流逝静音（`silence_credit`）。输入是未平滑的
//! 原始判定时传 0。
//!
//! # 纯度与上界
//!
//! 纯样本流逻辑：不看 VAD 内部状态、不抛错、不阻塞。所有辅助缓冲有上
//! 界——前导环 ≤ 前导长度，未决停顿 ≤ 断开长度，单区间记账 ≤ 区间上限
//! （触顶只重置记账并继续无缝保留，绝不丢连续语音）。输出顺序与到达
//! 顺序一致；常量与依据见 `sink.rs` 的 `OFFLINE_SPAN_*`。

/// 一个块的保留决定。
#[derive(Debug, PartialEq)]
pub(super) enum SpanDecision {
    /// 本块不进录音缓冲：区间尚未开始（只刷新前导环），或区间内尚未
    /// 定性的停顿（挂在未决缓冲里，语音恢复时整个补回）。
    Drop,
    /// 本块直接保留，之前没有要补的音频（语句进行中）。
    Keep,
    /// 先把这段真实音频补进录音缓冲，再按 `keep_chunk` 处理本块：
    /// 区间开启时是前导；语句恢复时是刚补定的停顿；区间关闭时是
    /// ≤ 尾随长度的停顿开头（此时本块本身属于被丢弃的深静音）。
    Flush { samples: Vec<f32>, keep_chunk: bool },
}

/// Offline 录音缓冲的保留状态机。样本数全部按 16 kHz 计。
pub(super) struct OfflineUtteranceSpan {
    preroll_cap: usize,
    tail_cap: usize,
    break_cap: usize,
    span_cap: usize,
    /// 语音→静音跳变处一次性记入的已流逝真实静音（hangover 折算）。
    silence_credit: usize,
    open: bool,
    /// 最近 ≤ `preroll_cap` 的真实样本，只在区间关闭期间维护。
    preroll: Vec<f32>,
    /// 区间内语音之后尚未定性的停顿样本，≤ `break_cap` + 一块。
    hold: Vec<f32>,
    /// 当前区间已保留的样本数（含前导与已补回的停顿），触顶记账用。
    span_samples: usize,
}

impl OfflineUtteranceSpan {
    /// `silence_credit` 见模块文档；生产路径是
    /// `CaptureVad::hangover_frames_for(Offline) × frame_samples`。
    pub(super) fn new(
        preroll_cap: usize,
        tail_cap: usize,
        break_cap: usize,
        span_cap: usize,
        silence_credit: usize,
    ) -> Self {
        // 相邻两段不能在句间静音里重叠：断开静音至少要容纳前一段的
        // 尾随 + 后一段的前导。
        assert!(
            preroll_cap + tail_cap <= break_cap,
            "span tail + pre-roll must not exceed the break silence"
        );
        Self {
            preroll_cap,
            tail_cap,
            break_cap,
            span_cap,
            silence_credit,
            open: false,
            preroll: Vec::new(),
            hold: Vec::new(),
            span_samples: 0,
        }
    }

    /// 用「本块是否判为语音」推进状态机。见 [`SpanDecision`]。
    pub(super) fn push(&mut self, speech: bool, chunk: &[f32]) -> SpanDecision {
        if !self.open {
            if !speech {
                // 区间还没开始：只留最近的前导，等语音确认后回放。
                self.preroll.extend_from_slice(chunk);
                let excess = self.preroll.len().saturating_sub(self.preroll_cap);
                if excess > 0 {
                    self.preroll.drain(..excess);
                }
                return SpanDecision::Drop;
            }
            // 区间开启：回放严格在本块之前的前导，再收本块。检测器
            // Speech 聚合帧自带的 prefill 回放不重复保留（前导由此处
            // 统一管理，行为与 VAD 后端无关）。
            self.open = true;
            let pre = std::mem::take(&mut self.preroll);
            self.span_samples = pre.len() + chunk.len();
            if self.span_samples >= self.span_cap {
                self.span_samples = 0;
            }
            return SpanDecision::Flush {
                samples: pre,
                keep_chunk: true,
            };
        }

        if speech {
            // 语句继续；若停顿被语音打断，先把整个停顿补回去（连续跨度）。
            let pre = std::mem::take(&mut self.hold);
            self.span_samples += pre.len() + chunk.len();
            if self.span_samples >= self.span_cap {
                // 触顶：只重置区间记账，保留不中断、不丢样本。一个
                // 区间因此至多比上限多出一个未决停顿（< 断开长度），
                // 仍落在长音频切块预算的同一量级。
                self.span_samples = 0;
            }
            if pre.is_empty() {
                return SpanDecision::Keep;
            }
            return SpanDecision::Flush {
                samples: pre,
                keep_chunk: true,
            };
        }

        // 区间内的静音：先挂起，等它要么被语音打断（整个保留），要么
        // 达到断开长度（只留开头一段尾随）。跳变处记入 hangover 折算，
        // 让 break_cap 始终指真实静音（见模块文档）。
        if self.hold.is_empty() {
            self.hold.reserve(self.break_cap.min(64 * 1024));
        }
        self.hold.extend_from_slice(chunk);
        if self.silence_credit + self.hold.len() >= self.break_cap {
            self.hold.truncate(self.tail_cap);
            let tail = std::mem::take(&mut self.hold);
            self.open = false;
            self.span_samples = 0;
            self.preroll.clear();
            return SpanDecision::Flush {
                samples: tail,
                keep_chunk: false,
            };
        }
        SpanDecision::Drop
    }

    /// 录音停止时收尾：未决停顿截成 ≤ 尾随长度保留，区间就此结束。
    /// 已保留的音频不受影响——语句仍在进行时缓冲本来就以最后一块
    /// 结尾，没有可补的尾。
    pub(super) fn finish(&mut self) -> Vec<f32> {
        self.preroll.clear();
        if !self.open {
            return Vec::new();
        }
        self.open = false;
        self.span_samples = 0;
        self.hold.truncate(self.tail_cap);
        std::mem::take(&mut self.hold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 前导 8、尾随 12、断开 32、区间上限 40（样本数），块长 4。
    fn machine() -> OfflineUtteranceSpan {
        OfflineUtteranceSpan::new(8, 12, 32, 40, 0)
    }

    fn v(value: f32) -> Vec<f32> {
        vec![value; 4]
    }

    #[test]
    fn opening_replays_only_the_capped_pre_roll() {
        let mut m = machine();
        // 12 样本静音 → 前导环只留最近 8 个。
        assert_eq!(m.push(false, &v(1.0)), SpanDecision::Drop);
        assert_eq!(m.push(false, &v(2.0)), SpanDecision::Drop);
        assert_eq!(m.push(false, &v(3.0)), SpanDecision::Drop);
        assert_eq!(
            m.push(true, &v(7.0)),
            SpanDecision::Flush {
                samples: [v(2.0), v(3.0)].concat(),
                keep_chunk: true,
            }
        );
    }

    #[test]
    fn opening_with_no_prior_audio_flushes_nothing() {
        let mut m = machine();
        assert_eq!(
            m.push(true, &v(7.0)),
            SpanDecision::Flush {
                samples: Vec::new(),
                keep_chunk: true,
            }
        );
        assert_eq!(m.push(true, &v(7.0)), SpanDecision::Keep);
    }

    #[test]
    fn a_short_pause_is_flushed_whole_when_speech_resumes() {
        let mut m = machine();
        m.push(true, &v(7.0));
        // 停顿 12 样本 < 断开 32：挂起，不丢。
        for i in 0..3 {
            assert_eq!(m.push(false, &v(5.0 + i as f32)), SpanDecision::Drop);
        }
        assert_eq!(
            m.push(true, &v(6.0)),
            SpanDecision::Flush {
                samples: [v(5.0), v(6.0), v(7.0)].concat(),
                keep_chunk: true,
            }
        );
    }

    #[test]
    fn reaching_the_break_keeps_only_the_tail_and_closes() {
        let mut m = machine();
        m.push(true, &v(7.0));
        for _ in 0..7 {
            assert_eq!(m.push(false, &v(5.0)), SpanDecision::Drop, "28 < 32 未断开");
        }
        // 第 8 块到达断开长度：保留开头 12 个尾随样本，本块丢弃。
        assert_eq!(
            m.push(false, &v(5.0)),
            SpanDecision::Flush {
                samples: vec![5.0; 12],
                keep_chunk: false,
            }
        );
        // 区间已关：静音只进前导环。
        assert_eq!(m.push(false, &v(5.0)), SpanDecision::Drop);
        assert_eq!(
            m.push(true, &v(9.0)),
            SpanDecision::Flush {
                samples: v(5.0),
                keep_chunk: true,
            }
        );
    }

    #[test]
    fn silence_credit_counts_as_elapsed_real_silence() {
        // hangover 折算 8：真实静音 32 = 已流逝 8 + Noise 判定 24。
        let mut m = OfflineUtteranceSpan::new(8, 12, 32, 40, 8);
        m.push(true, &v(7.0));
        for _ in 0..5 {
            assert_eq!(m.push(false, &v(5.0)), SpanDecision::Drop, "8+20 < 32");
        }
        assert_eq!(
            m.push(false, &v(5.0)),
            SpanDecision::Flush {
                samples: vec![5.0; 12],
                keep_chunk: false,
            }
        );
    }

    #[test]
    fn the_span_cap_never_drops_speech() {
        let mut m = machine();
        assert!(matches!(m.push(true, &v(1.0)), SpanDecision::Flush { .. }));
        // 远超区间上限 40 的连续语音：全部保留，只是记账触顶重置。
        for _ in 0..100 {
            assert_eq!(
                m.push(true, &v(2.0)),
                SpanDecision::Keep,
                "触顶绝不丢连续语音"
            );
        }
    }

    #[test]
    fn finish_keeps_at_most_the_tail_of_a_pending_pause() {
        let mut m = machine();
        m.push(true, &v(7.0));
        for _ in 0..4 {
            m.push(false, &v(5.0)); // 16 样本未决停顿
        }
        assert_eq!(m.finish(), vec![5.0; 12], "未决停顿截成尾随长度");
        assert_eq!(m.finish(), Vec::<f32>::new(), "重复 finish 无副作用");
    }

    #[test]
    fn finish_during_speech_adds_nothing() {
        let mut m = machine();
        m.push(true, &v(7.0));
        assert_eq!(m.finish(), Vec::<f32>::new());
    }

    #[test]
    fn finish_after_a_break_adds_nothing() {
        let mut m = machine();
        m.push(true, &v(7.0));
        for _ in 0..8 {
            m.push(false, &v(5.0));
        }
        let closed_tail = m.finish();
        assert!(closed_tail.is_empty(), "区间已关，无未决音频");
    }

    #[test]
    #[should_panic(expected = "pre-roll must not exceed")]
    fn overlapping_margins_are_rejected() {
        // 前导 20 + 尾随 20 > 断开 32：相邻两段会在静音里重叠。
        OfflineUtteranceSpan::new(20, 20, 32, 40, 0);
    }

    #[test]
    fn touching_margins_are_accepted() {
        // 前导 + 尾随 == 断开：两段保留恰好相接，不重叠。
        let mut m = OfflineUtteranceSpan::new(8, 12, 20, 40, 0);
        m.push(true, &v(7.0));
        for _ in 0..5 {
            m.push(false, &v(5.0)); // 第 5 块（20 样本）恰好断开
        }
        m.push(false, &v(5.0)); // 断开后的静音进前导环
        assert_eq!(
            m.push(true, &v(9.0)),
            SpanDecision::Flush {
                samples: v(5.0), // 只剩 4 样本静音可作前导
                keep_chunk: true,
            }
        );
    }
}
