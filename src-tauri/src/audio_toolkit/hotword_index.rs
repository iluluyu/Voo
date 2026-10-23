//! 拼音热词候选检索层：词长 × 首字候选音节的倒排索引。
//!
//! [`super::chinese_hotwords::apply_pinyin_hotwords`] 的旧内层循环对
//! 每个词条在全文逐起点验证，复杂度 O(词表 × 文本 × 词长)，5 万词级
//! 导入词库下单句匹配两百毫秒级（见 `examples/lexicon_bench`）。本模块
//! 把「哪些词条可能命中某个起点」收敛为首字音节查桶：
//!
//! - **构建**（词表编译期，随 `HotwordConfig` 缓存，不随转写重建）：
//!   每个词条按其**首字全部候选音节**（多音字全部入桶）的**混淆闭包**
//!   入桶——闭包 = 与该音节配对代价 < COST_DIFF 的全部音节，即
//!   声母白名单对（zh/z、ch/c、sh/s；方言档另有 n/l、f/h）× 韵母鼻音
//!   对（an/ang 类去/补尾 g）的全部组合（完全同音 / 单项 / 双混淆，
//!   与 `syllable_confusion` 的档位一一对应）。方言对单独入副表，
//!   查询时按 [`ConfusionOptions`] 叠加，方言关闭时零浪费。
//! - **检索**（转写期）：对文本每个起点，取该字 plain 候选音节做
//!   **精确**桶查找（零字符串构造），取出候选词条下标，交回调用方用
//!   既有打分函数逐一验证。
//!
//! **正确性（超集性）**：任何通过旧路径 `window_similarity` 的窗口，
//! 其首字音节对代价必然 < COST_DIFF（`aligned_match` 对任一位置代价
//! ≥ COST_DIFF 直接判 `None`）。混淆代价关于两个音节对称（白名单对
//! 与鼻音对都是双向匹配），因此「热词首字候选 h 与窗口首字候选 s 代价
//! < COST_DIFF」⟺「s ∈ 闭包(h)」⟺「查询 s 命中 h 所在桶」。检索层
//! 是旧候选枚举的**保守超集**：只做预筛，打分、严格档双侧主读音校验、
//! 2 字档、Candidate（no_rewrite）跳过、黑名单、保护区与歧义判定
//! 全部留在调用方原逻辑，分毫不动。闭包生成与
//! `chinese_hotwords::syllable_confusion` 的一致性由
//! `closure_matches_syllable_confusion` 对真实音节全集交叉验证。

use super::chinese_hotwords::{CharSounds, ConfusionOptions, HotwordEntry};
use std::collections::HashMap;

/// 按词长分桶的词条下标：(词长, 该词长下的词条下标组)。
type LengthBuckets = Vec<(u16, Vec<u32>)>;

/// 拼音热词倒排索引：首字候选音节的混淆闭包 → 各词长桶。
///
/// 构建见 [`HotwordSearchIndex::build`]；词条下标指向调用方传入的
/// `&[HotwordEntry]`，两者必须同源同生命周期（缓存路径下同处一个
/// `HotwordConfig`，随编译缓存一起失效重建）。
#[derive(Debug, Default)]
pub struct HotwordSearchIndex {
    /// 常规混淆闭包桶（声母 zh/z、ch/c、sh/s + 韵尾鼻音，单/双组合）
    plain: HashMap<Box<str>, LengthBuckets>,
    /// 方言混淆额外桶（声母 n/l、f/h；`opts.dialect` 开启时叠加查询）。
    /// 单独成表：方言关闭时不查、不浪费验证，方言开启时不漏取。
    dialect_extra: HashMap<Box<str>, LengthBuckets>,
}

/// 声母白名单互换（双向，镜像 `syllable_confusion` 的 initial_pair /
/// dialect_pair）。返回 None 表示该声母无白名单对。
fn swap_initial(initial: &str, dialect: bool) -> Option<&'static str> {
    match initial {
        "zh" => Some("z"),
        "z" => Some("zh"),
        "ch" => Some("c"),
        "c" => Some("ch"),
        "sh" => Some("s"),
        "s" => Some("sh"),
        "n" if dialect => Some("l"),
        "l" if dialect => Some("n"),
        "f" if dialect => Some("h"),
        "h" if dialect => Some("f"),
        _ => None,
    }
}

/// 拆音节为 (声母, 韵母)，零声母（a/o/e 等开头）声母为空串。
/// 镜像 `chinese_hotwords::split_syllable` 的规则（zh/ch/sh 双字母
/// 声母 + 单声母表 bpmfdtnlgkhjqxrzcsyw）；与权威实现的一致性由
/// `closure_matches_syllable_confusion` 间接验证（拆错必然导致闭包
/// 与真实代价不符而被测试抓住）。
fn split_initial_final(s: &str) -> (&str, &str) {
    for d in ["zh", "ch", "sh"] {
        if let Some(rest) = s.strip_prefix(d) {
            if !rest.is_empty() {
                return (d, rest);
            }
        }
    }
    let mut chars = s.chars();
    if let Some(first) = chars.next() {
        let rest = chars.as_str();
        if "bpmfdtnlgkhjqxrzcsyw".contains(first) && !rest.is_empty() {
            return (&s[..first.len_utf8()], rest);
        }
    }
    ("", s)
}

/// 一个音节的混淆闭包键。
///
/// - `dialect == false`：{原声母, 常规白名单互换} × {原韵母, 鼻音变体}
///   （≤ 2×2 = 4 个键，含音节自身）；
/// - `dialect == true`：**只生成**方言互换声母（n/l、f/h）× 全部韵母
///   变体，供副表使用（常规键已在主表，不重复）。
///
/// 韵母鼻音变体镜像 `syllable_confusion` 的 nasal_pair（strip 尾 g 后
/// 相等且至少一方以 "ng" 结尾）：以 "ng" 结尾去 g、以 n 结尾补 g，
/// 其余无变体。
fn closure_keys(syllable: &str, dialect: bool) -> Vec<String> {
    let (initial, final_) = split_initial_final(syllable);
    let mut initials: Vec<&str> = Vec::with_capacity(2);
    if dialect {
        if let Some(sw) = swap_initial(initial, true) {
            initials.push(sw);
        }
    } else {
        initials.push(initial);
        if let Some(sw) = swap_initial(initial, false) {
            initials.push(sw);
        }
    }
    let mut finals: Vec<String> = Vec::with_capacity(2);
    if !initials.is_empty() {
        finals.push(final_.to_string());
        if final_.ends_with("ng") {
            finals.push(final_[..final_.len() - 1].to_string());
        } else if final_.ends_with('n') {
            finals.push(format!("{final_}g"));
        }
    }
    let mut keys = Vec::with_capacity(initials.len() * finals.len());
    for i in &initials {
        for f in &finals {
            let mut k = String::with_capacity(i.len() + f.len());
            k.push_str(i);
            k.push_str(f);
            // 变体必须能按原 (声母, 韵母) 读回：如零声母韵尾 n 补 g 得
            // "ng"，但权威拆分把 "ng" 读成声母 n + 韵母 g，并不在真实
            // 闭包内（syllable_confusion("n","ng") == COST_DIFF）。
            // 读不回去的变体直接丢弃（真实音节不会因丢弃而漏取，
            // 由 closure_matches_syllable_confusion 全集交叉验证）。
            if !keys.contains(&k) && split_initial_final(&k) == (*i, f.as_str()) {
                keys.push(k);
            }
        }
    }
    keys
}

/// 向指定表插入 (键, 词长, 词条下标)。
fn insert_bucket(map: &mut HashMap<Box<str>, LengthBuckets>, key: String, len: u16, idx: u32) {
    let buckets = map.entry(key.into_boxed_str()).or_default();
    match buckets.iter_mut().find(|(l, _)| *l == len) {
        Some((_, ids)) => ids.push(idx),
        None => buckets.push((len, vec![idx])),
    }
}

impl HotwordSearchIndex {
    /// 从拼音热词索引构建倒排检索层。空音节词条（防御，正常不存在）
    /// 与超长词条跳过。Candidate（`no_rewrite`）词条照常入桶：档位
    /// 过滤是匹配决策，检索层保持对全部词条可查。
    pub fn build(index: &[HotwordEntry]) -> Self {
        let mut plain: HashMap<Box<str>, LengthBuckets> = HashMap::new();
        let mut dialect_extra: HashMap<Box<str>, LengthBuckets> = HashMap::new();
        for (i, hot) in index.iter().enumerate() {
            let m = hot.syllables.len();
            if m == 0 || m > u16::MAX as usize {
                continue;
            }
            let idx = i as u32;
            // 多音字：首字全部候选音节各自入桶
            for s in &hot.syllables[0].syllables {
                for key in closure_keys(s, false) {
                    insert_bucket(&mut plain, key, m as u16, idx);
                }
                for key in closure_keys(s, true) {
                    insert_bucket(&mut dialect_extra, key, m as u16, idx);
                }
            }
        }
        Self {
            plain,
            dialect_extra,
        }
    }

    /// 对一个匹配起点发射候选词条下标（回调按发射序消费）。
    ///
    /// - `first`：起点字的读音候选（窗口首字）；
    /// - `avail`：起点到段尾的可用窗口长（= 段长 - 起点），只发射
    ///   词长 ≤ avail 的词条；
    /// - `opts`：方言混淆开关决定是否叠加方言副表。
    ///
    /// 同一词条可能经多个键重复发射（首字多候选的闭包相交）：重复
    /// 候选对调用方无害（同一 (score, 词条) 二次入组会被按替换目标
    /// 合并，最终输出不变），故不做去重，换取查询零状态、零分配。
    pub fn for_each_candidate<F: FnMut(usize)>(
        &self,
        first: &CharSounds,
        avail: usize,
        opts: &ConfusionOptions,
        mut f: F,
    ) {
        for s in &first.syllables {
            if let Some(buckets) = self.plain.get(s.as_str()) {
                emit_buckets(buckets, avail, &mut f);
            }
            if opts.dialect {
                if let Some(buckets) = self.dialect_extra.get(s.as_str()) {
                    emit_buckets(buckets, avail, &mut f);
                }
            }
        }
    }
}

/// 发射一个键下全部词长 ≤ avail 的词条下标。
fn emit_buckets<F: FnMut(usize)>(buckets: &LengthBuckets, avail: usize, f: &mut F) {
    for (len, ids) in buckets {
        if (*len as usize) <= avail {
            for &id in ids {
                f(id as usize);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_toolkit::chinese_hotwords::{
        build_hotword_index, char_sounds, syllable_confusion, DEFAULT_MIN_HAN,
    };

    const NO_DIALECT: ConfusionOptions = ConfusionOptions { dialect: false };
    const DIALECT: ConfusionOptions = ConfusionOptions { dialect: true };

    /// 真实音节全集：CJK 基本区全部汉字的 plain 候选 + ASCII 单字符，
    /// 去重排序（闭包交叉验证的论域）。
    fn syllable_universe() -> Vec<String> {
        let mut set = std::collections::BTreeSet::new();
        for cp in 0x4E00u32..=0x9FA5 {
            if let Some(c) = char::from_u32(cp) {
                if let Some(cs) = char_sounds(c) {
                    if cs.is_han {
                        set.extend(cs.syllables);
                    }
                }
            }
        }
        for c in 'a'..='z' {
            set.insert(c.to_string());
        }
        for c in '0'..='9' {
            set.insert(c.to_string());
        }
        set.into_iter().collect()
    }

    /// 闭包生成必须与权威代价函数逐对一致：
    /// 对论域内每个 x（采样 + 人工挑选的边界音节）与**全部** y，
    /// `syllable_confusion(x, y).0 < 1.0`（非 COST_DIFF）
    /// ⟺ y ∈ closure(x)（方言开 / 关分别验证）。
    /// 拆音节或白名单表任何一处与权威实现漂移都会在此失败。
    #[test]
    fn closure_matches_syllable_confusion() {
        let universe = syllable_universe();
        let mut probe: Vec<String> = vec![
            "zhang".into(),
            "zang".into(),
            "zhan".into(),
            "zan".into(),
            "zhi".into(),
            "zi".into(),
            "chi".into(),
            "ci".into(),
            "shi".into(),
            "si".into(),
            "liu".into(),
            "niu".into(),
            "fan".into(),
            "huan".into(),
            "an".into(),
            "ang".into(),
            "in".into(),
            "ing".into(),
            "en".into(),
            "eng".into(),
            "n".into(),
            "ng".into(),
            "g".into(),
            "m".into(),
            "hm".into(),
            "hng".into(),
            "e".into(),
            "er".into(),
            "yun".into(),
            "yong".into(),
            "nv".into(),
            "v".into(),
            "wu".into(),
            "yi".into(),
            "a".into(),
            "ai".into(),
        ];
        // 全集等距采样（覆盖全部声母/韵母组合形态）
        let step = (universe.len() / 120).max(1);
        probe.extend(universe.iter().step_by(step).cloned());

        for x in &probe {
            let plain_set: Vec<String> = closure_keys(x, false);
            let full_set: Vec<String> = plain_set
                .iter()
                .cloned()
                .chain(closure_keys(x, true))
                .collect();
            for y in &universe {
                let cost_plain = syllable_confusion(x, y, &NO_DIALECT).0;
                let cost_dialect = syllable_confusion(x, y, &DIALECT).0;
                assert_eq!(
                    cost_plain < 1.0,
                    plain_set.iter().any(|k| k == y),
                    "非方言闭包不一致：x={x} y={y} cost={cost_plain}"
                );
                assert_eq!(
                    cost_dialect < 1.0,
                    full_set.iter().any(|k| k == y),
                    "方言闭包不一致：x={x} y={y} cost={cost_dialect}"
                );
            }
        }
    }

    /// 检索基础：同音/近音/多音字候选都能取到，无关首字不取；
    /// 词长 > 可用窗口长的词条不发射。
    #[test]
    fn retrieves_homophone_near_sound_and_respects_length() {
        let words: Vec<String> = ["知识库", "中州韵输入法", "重庆"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let index = build_hotword_index(&words, 2);
        let search = HotwordSearchIndex::build(&index);

        let collect = |c: char, avail: usize, opts: &ConfusionOptions| {
            let mut out = Vec::new();
            search.for_each_candidate(&char_sounds(c).unwrap(), avail, opts, |i| out.push(i));
            out
        };

        // 「只」(zhi) 经平翘舌闭包应命中「知识库」；「中」命中「中州韵输入法」
        assert!(collect('只', 3, &NO_DIALECT).contains(&0));
        assert!(collect('中', 6, &NO_DIALECT).contains(&1));
        // 可用窗口不足：词长 3 的词条在 avail=2 时不发射
        assert!(!collect('只', 2, &NO_DIALECT).contains(&0));
        // 无关首字（如「天」）不命中任何桶
        assert!(collect('天', 8, &NO_DIALECT).is_empty());
        // 多音字查询：「重」(zhong/chong) 的全部候选音节都可查
        let zhong = char_sounds('重').unwrap();
        let mut out = Vec::new();
        search.for_each_candidate(&zhong, 2, &NO_DIALECT, |i| out.push(i));
        assert!(out.contains(&2), "多音字候选未覆盖：{out:?}");
    }

    /// 方言副表：n/l、f/h 声母对只在 opts.dialect 开启时命中。
    #[test]
    fn dialect_pairs_only_retrieved_when_enabled() {
        let words: Vec<String> = ["流传检验"].iter().map(|s| s.to_string()).collect();
        let index = build_hotword_index(&words, DEFAULT_MIN_HAN);
        let search = HotwordSearchIndex::build(&index);
        let niu = char_sounds('牛').unwrap(); // niu：方言对 liu
        let mut out = Vec::new();
        search.for_each_candidate(&niu, 4, &NO_DIALECT, |i| out.push(i));
        assert!(out.is_empty(), "方言关闭时不应命中：{out:?}");
        search.for_each_candidate(&niu, 4, &DIALECT, |i| out.push(i));
        assert_eq!(out, vec![0]);
    }

    /// 空索引零发射；Candidate（no_rewrite）词条照常可检索（档位
    /// 过滤属于匹配决策）。
    #[test]
    fn empty_index_emits_nothing() {
        let search = HotwordSearchIndex::build(&[]);
        let zhi = char_sounds('知').unwrap();
        let mut n = 0;
        search.for_each_candidate(&zhi, 10, &DIALECT, |_| n += 1);
        assert_eq!(n, 0);
    }
}
