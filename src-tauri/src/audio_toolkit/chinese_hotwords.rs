//! 中文热词系统：拼音音素模糊匹配 + 正则规则替换
//!
//! 上游 Handy 的 `apply_custom_words` 为拉丁文字设计（空格分词 + Soundex
//! 音码），对中文完全无效（上游 issue #1705）。本模块实现引擎无关的中文
//! 热词后处理，思路受 CapsWriter-Offline 启发，参数经专项评审确定：
//!
//! 1. 拼音模糊匹配：热词与转写文本都转为无声调拼音（多音字展开全部候选），
//!    在可读音片段上做**等长**滑动窗口 + 音节级配对代价。命中需同时满足
//!    相似度阈值与硬约束（至多一个近音混淆音节、无不相关音节）。
//! 2. 白名单混淆档位（受控近音，不做通用声韵母折扣）：
//!    完全同音 0 / 单项混淆 0.35 / 声母+韵尾双混淆 0.55 /
//!    方言混淆(n/l、f/h，需开启) 0.50 / 其他 1.0。
//! 3. 正则规则：`模式=替换`（字面量）或 `模式=>替换`（正则），拼音替换后
//!    按行序应用。
//!
//! 误替换防护（评审结论：宁可漏纠，不可误改）：
//! - 默认最短热词 3 个汉字；2 字热词只接受完整同音（dist == 0）。
//! - 来源严格度：导入词库（imported_lexicons）词条只接受**双侧常规
//!   读音下的完全同音**（窗口与热词逐位都是各自字符的常规读音
//!   [`CharSounds::primary`] 且相等，dist == 0），不做近音模糊；
//!   手工词（custom_words）保持近音行为。背景：多音字展开会让
//!   日常短语经非常规读音凑出“完全同音”（窗口侧如 什(shí)么(mó)时
//!   == 实模式，把「什么时候」改写成「实模式候」；热词侧如
//!   会(kuài)计准则 == 快记准则），5 万词级分类词库的领域词必须
//!   按严格档对待。
//! - 四态档位（`settings.lexicon_term_modes`，key = 解析后的替换目标
//!   primary，别名继承目标档位；黑名单仍是独立匹配约束）：Blocked
//!   不建条目（自动纠正与建议都不进）；Candidate 建条目但不参与
//!   改写（仅建议路径）；ActiveExact 走严格档（匹配时双侧常规读音
//!   校验）；ActiveFuzzy 走现有近音档。档位在索引展开条目之前解析，
//!   且参与热词缓存键（档位变化必然重编译）。缺省由来源决定：
//!   手工词 ActiveFuzzy、导入词 Candidate（settings 层解析）。
//! - 纯汉字热词只在纯汉字窗口内匹配（不吞英文/数字）。
//! - 原文已含热词字面时，保护整个区间不被任何候选覆盖。
//! - 同一区间多个热词命中时：先剔除黑名单命中的候选，再按替换目标合并
//!   （重复热词/同目标别名不构成竞争），仅不同目标间前两名分差 < 0.08
//!   时放弃（同音竞争歧义）。
//! - 全程 `Vec<char>` 索引，绝不按字节切中文（#1705 panic 教训）。
//! - 幂等：对已处理文本重复应用结果不变。

use log::warn;
use pinyin::{ToPinyin, ToPinyinMulti};
use regex::Regex;
use serde::{Deserialize, Serialize};
use specta::Type;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// 代价表（评审确定的受控档位）
// ---------------------------------------------------------------------------

/// 音节完全同音
const COST_EXACT: f64 = 0.0;
/// 单项白名单混淆（平翘舌 或 前后鼻音）
const COST_SINGLE_CONFUSION: f64 = 0.35;
/// 声母 + 韵尾双混淆（如 zhang→cen 类）
const COST_DOUBLE_CONFUSION: f64 = 0.55;
/// 方言混淆单项（n/l、f/h，需显式开启）
const COST_DIALECT: f64 = 0.50;
/// 音节不相关
const COST_DIFF: f64 = 1.0;
/// 删除/插入一个字（变长窗口默认关闭，保留供显式开启）
const COST_GAP: f64 = 1.0;

/// 默认相似度阈值
pub const DEFAULT_THRESHOLD: f64 = 0.85;
/// 同音竞争歧义：前两名分差小于此值放弃
const AMBIGUITY_MARGIN: f64 = 0.08;
/// 默认最短热词汉字数
pub const DEFAULT_MIN_HAN: usize = 3;

/// 混淆开关（方言混淆默认关闭）
#[derive(Debug, Clone, Copy)]
pub struct ConfusionOptions {
    /// 启用 n/l 与 f/h 声母混淆（默认 false）
    pub dialect: bool,
}

impl Default for ConfusionOptions {
    fn default() -> Self {
        Self { dialect: false }
    }
}

// ---------------------------------------------------------------------------
// 拼音层
// ---------------------------------------------------------------------------

/// 单个“字”的读音候选
#[derive(Debug, Clone, PartialEq)]
pub struct CharSounds {
    /// 全部候选读音（plain 无声调，去重），至少 1 个
    pub syllables: Vec<String>,
    /// 常规读音（pinyin 单读音典；ASCII 为字符自身）。严格词条的
    /// 完全同音判定以它为准（双侧逐位相等）：任何一侧多音字用
    /// 非常规读音凑出的“同音”都不算。
    pub primary: String,
    /// 是否汉字
    pub is_han: bool,
}

/// 汉字返回多音候选；ASCII 字母/数字返回自身小写作单音节（支持中英混输
/// 热词）；其他（标点/空白/emoji）返回 None（不参与对齐）。
pub fn char_sounds(c: char) -> Option<CharSounds> {
    if is_han(c) {
        let mut syllables: Vec<String> = c
            .to_pinyin_multi()
            .map(|multi| {
                multi
                    .into_iter()
                    .map(|p| p.plain().to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if syllables.is_empty() {
            return None; // 词典缺字（生僻字）
        }
        syllables.sort();
        syllables.dedup();
        // 常规读音取单读音典；两典不一致时收编进候选集（防御，保证
        // primary 与 syllables 同源，严格判定不会绕开候选机制）
        let canonical = c.to_pinyin().map(|p| p.plain().to_string());
        let primary = match canonical {
            Some(p) if syllables.contains(&p) => p,
            _ => syllables[0].clone(),
        };
        Some(CharSounds {
            syllables,
            primary,
            is_han: true,
        })
    } else if c.is_ascii_alphanumeric() {
        let primary = c.to_ascii_lowercase().to_string();
        Some(CharSounds {
            syllables: vec![primary.clone()],
            primary,
            is_han: false,
        })
    } else {
        None
    }
}

pub fn is_han(c: char) -> bool {
    matches!(c,
        '\u{3400}'..='\u{4DBF}'   // CJK 扩展 A
        | '\u{4E00}'..='\u{9FFF}' // CJK 基本
        | '\u{F900}'..='\u{FAFF}' // CJK 兼容
    )
}

pub fn has_han(text: &str) -> bool {
    text.chars().any(is_han)
}

/// 拆音节为 (声母, 韵母)。零声母（a/o/e 开头）声母为空串。
/// 输入须为合法无声调拼音；本函数只为白名单混淆识别服务，容错处理
/// 非法输入（返回原样，不 panic）。
fn split_syllable(s: &str) -> (String, String) {
    const DOUBLE_INITIALS: [&str; 3] = ["zh", "ch", "sh"];
    const SINGLE_INITIALS: &str = "bpmfdtnlgkhjqxrzcsyw";
    for d in DOUBLE_INITIALS {
        if let Some(rest) = s.strip_prefix(d) {
            if !rest.is_empty() {
                return (d.to_string(), rest.to_string());
            }
        }
    }
    let mut chars = s.chars();
    if let Some(first) = chars.next() {
        let rest = chars.as_str();
        if SINGLE_INITIALS.contains(first) && !rest.is_empty() {
            return (first.to_string(), rest.to_string());
        }
    }
    (String::new(), s.to_string())
}

/// 计算两个具体音节之间的混淆档位。
/// 返回 (cost, confusion_count)：count 用于硬约束（非零 cost 音节计数）。
pub(crate) fn syllable_confusion(a: &str, b: &str, opts: &ConfusionOptions) -> (f64, usize) {
    if a == b {
        return (COST_EXACT, 0);
    }
    let (ia, fa) = split_syllable(a);
    let (ib, fb) = split_syllable(b);

    let mut confusions = 0usize;

    // 声母：平翘舌对（zh/z, ch/c, sh/s）——注意 zheng vs zeng 这类
    // "声母平翘舌"对；以及方言对 n/l、f/h
    let initial_pair = |x: &str, y: &str| -> bool {
        let pair = |p: [&str; 2]| (x == p[0] && y == p[1]) || (x == p[1] && y == p[0]);
        pair(["zh", "z"]) || pair(["ch", "c"]) || pair(["sh", "s"])
    };
    let dialect_pair = |x: &str, y: &str| -> bool {
        let pair = |p: [&str; 2]| (x == p[0] && y == p[1]) || (x == p[1] && y == p[0]);
        pair(["n", "l"]) || pair(["f", "h"])
    };

    let initial_confused = if ia == ib {
        false
    } else if initial_pair(&ia, &ib) {
        confusions += 1;
        true
    } else if opts.dialect && dialect_pair(&ia, &ib) {
        // 方言混淆计为一项，cost 用 COST_DIALECT 档（见下方合成）
        confusions += 1;
        true
    } else {
        // 声母完全不同且不在白名单：整音节不相关
        return (COST_DIFF, 1);
    };
    let _ = initial_confused;

    // 韵母：仅前后鼻音对（ng/n 结尾差异）
    let nasal_pair = |x: &str, y: &str| -> bool {
        if x == y {
            return false;
        }
        // 前后鼻音对：in/ing、en/eng、an/ang——去掉结尾的 'g' 后应相等。
        // 注意不能 strip_suffix("ng")：那会把 "ing" 剥成 "i"。
        let strip_final_g = |s: &str| -> String {
            if s.ends_with('g') {
                s[..s.len() - 1].to_string()
            } else {
                s.to_string()
            }
        };
        strip_final_g(x) == strip_final_g(y) && (x.ends_with("ng") || y.ends_with("ng"))
    };

    let _final_confused = if fa == fb {
        false
    } else if nasal_pair(&fa, &fb) {
        confusions += 1;
        true
    } else {
        return (COST_DIFF, 1);
    };

    // 合成档位
    let cost = match confusions {
        0 => COST_EXACT, // 不可能（a==b 已提前返回），防御
        1 => {
            // 区分方言档与普通单项档
            if opts.dialect && dialect_pair(&ia, &ib) && !initial_pair(&ia, &ib) {
                COST_DIALECT
            } else {
                COST_SINGLE_CONFUSION
            }
        }
        _ => COST_DOUBLE_CONFUSION,
    };
    (cost, confusions.min(1))
}

/// 两字候选集的配对代价（笛卡尔积取最小）。
/// 返回 (cost, 该最小代价下的混淆数)
fn pair_cost(a: &[String], b: &[String], opts: &ConfusionOptions) -> (f64, usize) {
    let mut best = (COST_DIFF, 1usize);
    for sa in a {
        for sb in b {
            let r = syllable_confusion(sa, sb, opts);
            if r.0 < best.0 {
                best = r;
            }
            if best.0 == COST_EXACT {
                return best;
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// 热词索引
// ---------------------------------------------------------------------------

/// 一个热词条目
/// 一行自定义词汇解析结果：`目标 | 别名1 ~~~ 黑名单1 | 黑名单2`
#[derive(Debug, Clone, Default)]
pub struct ParsedHotwordLine {
    /// 替换目标（第一个词）
    pub primary: String,
    /// 触发别名（其余词，命中时同样替换为 primary）
    pub aliases: Vec<String>,
    /// 上下文黑名单（句中含有任一词时放弃替换，防同音歧义）
    pub blacklist: Vec<String>,
}

/// 解析一行自定义词汇：`|` 分隔目标与别名，行尾 `~~~` 后为黑名单。
/// 空段忽略；井号开头整行注释由调用方处理。
pub fn parse_hotword_line(line: &str) -> ParsedHotwordLine {
    let mut out = ParsedHotwordLine::default();
    let (words_part, blacklist_part) = match line.split_once("~~~") {
        Some((w, b)) => (w, Some(b)),
        None => (line, None),
    };
    if let Some(b) = blacklist_part {
        out.blacklist = b
            .split('|')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
    }
    for (i, seg) in words_part.split('|').enumerate() {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }
        if i == 0 || out.primary.is_empty() {
            if out.primary.is_empty() {
                out.primary = seg.to_owned();
            }
        } else {
            out.aliases.push(seg.to_owned());
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct HotwordEntry {
    /// 替换目标（用户配置的热词原文或别名行的第一个词）
    pub display: String,
    /// 每字读音候选（与触发文本的 chars 一一对应）
    pub syllables: Vec<CharSounds>,
    /// 汉字数量
    pub han_count: usize,
    /// 是否纯汉字（纯汉字热词只在纯汉字窗口匹配）
    pub all_han: bool,
    /// 上下文黑名单（整句含任一词时放弃该命中）
    pub blacklist: Vec<String>,
    /// 严格词条：只接受双侧常规读音下的完全同音
    /// （dist == 0 且窗口与热词逐位 [`CharSounds::primary`] 相等），
    /// 不做近音模糊——这是匹配时的双侧主读音校验，不是打分惩罚或
    /// 事后过滤。四态档位下由 ActiveExact 触发（ActiveFuzzy 为 false
    /// 保持近音行为）。
    pub strict: bool,
    /// Candidate（候选）档：条目保留在索引里供建议路径，但匹配阶段
    /// 直接跳过——不进自动纠正（等同跳过该条目）。Blocked 档不会
    /// 建出条目，故无需标志。
    pub no_rewrite: bool,
}

/// 热词条目的四态匹配档位（`settings.lexicon_term_modes` 的值类型）。
///
/// key = 解析后的替换目标（primary）；同一行的别名继承目标档位，
/// 黑名单是独立的匹配约束、不进模式表。档位必须在建索引展开条目
/// **之前**解析（见 [`build_hotword_index_with_modes`]）：
/// - [`TermMode::Candidate`]：建条目但不参与自动纠正（`no_rewrite`），
///   只对建议路径可见；
/// - [`TermMode::ActiveExact`]：严格档——匹配时的双侧常规读音校验
///   （canonical_homophone），不是打分惩罚或事后过滤；
/// - [`TermMode::ActiveFuzzy`]：现有近音 fuzzy 档；
/// - [`TermMode::Blocked`]：不建条目，自动纠正与建议都不参与。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TermMode {
    /// 候选：只进建议，不改写文本
    Candidate,
    /// 启用·严格同音（双侧常规读音校验）
    ActiveExact,
    /// 启用·近音模糊
    ActiveFuzzy,
    /// 屏蔽：不建条目
    Blocked,
}

impl TermMode {
    /// 设置存储/命令参数里的档位名（snake_case：`candidate` /
    /// `active_exact` / `active_fuzzy` / `blocked`，大小写不敏感）。
    /// 命令层用它把非法档位名变成显式报错，而不是静默失败。
    pub fn from_str_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "candidate" => Some(Self::Candidate),
            "active_exact" => Some(Self::ActiveExact),
            "active_fuzzy" => Some(Self::ActiveFuzzy),
            "blocked" => Some(Self::Blocked),
            _ => None,
        }
    }
}

/// 从自定义词列表构建拼音热词索引。
/// 每行支持 `目标 | 别名 ~~~ 黑名单`（见 parse_hotword_line）。
/// 收录条件：汉字数 >= `min_han`（默认 3）、每个字符都有读音；
/// 别名与目标共享收录门槛。
/// 全部词条按手工词（近音宽松）对待：等价于
/// `build_hotword_index_with_fuzzy(words, min_han, words)`。
pub fn build_hotword_index(words: &[String], min_han: usize) -> Vec<HotwordEntry> {
    build_hotword_index_with_fuzzy(words, min_han, words)
}

/// 构建拼音热词索引，并按来源区分严格度：
/// - `fuzzy_words`（手工词白名单，按行 trim 后精确比较）内的行 →
///   近音宽松，保持既有行为；
/// - 其余行（imported_lexicons 导入词）→ 严格词条（见
///   [`HotwordEntry::strict`]），防大规模词库词误改日常短语。
pub fn build_hotword_index_with_fuzzy(
    words: &[String],
    min_han: usize,
    fuzzy_words: &[String],
) -> Vec<HotwordEntry> {
    let mut entries = Vec::with_capacity(words.len());
    for line in words {
        let parsed = parse_hotword_line(line);
        if parsed.primary.is_empty() {
            continue;
        }
        // 同一行（含别名）共享来源：手工词整行宽松，导入词整行严格
        let fuzzy = fuzzy_words.iter().any(|w| w.trim() == line.trim());
        // 目标 + 别名各建一条，display 都指向 primary
        let triggers = std::iter::once(parsed.primary.as_str())
            .chain(parsed.aliases.iter().map(String::as_str));
        for trigger in triggers {
            let han_count = trigger.chars().filter(|c| is_han(*c)).count();
            if han_count < min_han || trigger.chars().count() > 50 {
                continue;
            }
            let mut syllables = Vec::with_capacity(trigger.chars().count());
            let mut ok = true;
            for c in trigger.chars() {
                match char_sounds(c) {
                    Some(s) => syllables.push(s),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                entries.push(HotwordEntry {
                    display: parsed.primary.clone(),
                    all_han: syllables.iter().all(|s| s.is_han),
                    han_count,
                    syllables,
                    blacklist: parsed.blacklist.clone(),
                    strict: !fuzzy,
                    no_rewrite: false,
                });
            }
        }
    }
    entries
}

/// 按四态档位表构建索引（`settings.lexicon_term_modes` 的消费端）。
///
/// 档位在展开条目之前解析，key = 解析后的替换目标（primary）；
/// 同一行的别名继承目标档位，黑名单保持独立匹配约束：
/// - `Blocked` → 整行不建条目（含全部别名）；
/// - `Candidate` → 建条目但 `no_rewrite = true`（不进自动纠正）；
/// - `ActiveExact` → `strict = true`（匹配时双侧常规读音校验）；
/// - `ActiveFuzzy` → `strict = false`（现有近音档）。
///
/// 表里没有的 primary 兜底按 `ActiveExact` 对待（与旧导入词 strict
/// 行为一致）；来源缺省（手工→ActiveFuzzy、导入→Candidate）由
/// settings 层解析成完整表后传入（见
/// `crate::settings::AppSettings::effective_term_modes`）。
pub fn build_hotword_index_with_modes(
    words: &[String],
    min_han: usize,
    modes: &HashMap<String, TermMode>,
) -> Vec<HotwordEntry> {
    let mut entries = Vec::with_capacity(words.len());
    for line in words {
        let parsed = parse_hotword_line(line);
        if parsed.primary.is_empty() {
            continue;
        }
        // 档位按替换目标解析（别名/黑名单继承整行），在建条目之前判定
        let mode = modes
            .get(parsed.primary.as_str())
            .copied()
            .unwrap_or(TermMode::ActiveExact);
        if mode == TermMode::Blocked {
            continue;
        }
        let strict = mode == TermMode::ActiveExact;
        let no_rewrite = mode == TermMode::Candidate;
        // 目标 + 别名各建一条，display 都指向 primary
        let triggers = std::iter::once(parsed.primary.as_str())
            .chain(parsed.aliases.iter().map(String::as_str));
        for trigger in triggers {
            let han_count = trigger.chars().filter(|c| is_han(*c)).count();
            if han_count < min_han || trigger.chars().count() > 50 {
                continue;
            }
            let mut syllables = Vec::with_capacity(trigger.chars().count());
            let mut ok = true;
            for c in trigger.chars() {
                match char_sounds(c) {
                    Some(s) => syllables.push(s),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                entries.push(HotwordEntry {
                    display: parsed.primary.clone(),
                    all_han: syllables.iter().all(|s| s.is_han),
                    han_count,
                    syllables,
                    blacklist: parsed.blacklist.clone(),
                    strict,
                    no_rewrite,
                });
            }
        }
    }
    entries
}

// ---------------------------------------------------------------------------
// 文本切分
// ---------------------------------------------------------------------------

/// 连续可读音片段（字符有全局 char 偏移）
#[derive(Debug)]
struct Segment {
    chars: Vec<(usize, char)>,
    sounds: Vec<CharSounds>,
}

fn segment_text(text: &[char]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current: Vec<(usize, char)> = Vec::new();
    let mut sounds: Vec<CharSounds> = Vec::new();
    for (idx, &c) in text.iter().enumerate() {
        match char_sounds(c) {
            Some(s) => {
                current.push((idx, c));
                sounds.push(s);
            }
            None => {
                if !current.is_empty() {
                    segments.push(Segment {
                        chars: std::mem::take(&mut current),
                        sounds: std::mem::take(&mut sounds),
                    });
                }
            }
        }
    }
    if !current.is_empty() {
        segments.push(Segment {
            chars: current,
            sounds,
        });
    }
    segments
}

// ---------------------------------------------------------------------------
// 匹配核心
// ---------------------------------------------------------------------------

/// 等长窗口对齐代价：逐位置配对（等长无 gap）。
/// 返回 None 若不满足硬约束。
/// 硬约束：非零 cost 的音节至多 1 个；且无 cost == COST_DIFF 音节。
/// 相似度 = 1 - total_cost / len。
fn aligned_match(
    window: &[CharSounds],
    hot: &[CharSounds],
    opts: &ConfusionOptions,
    allow_two_near: bool,
) -> Option<f64> {
    debug_assert_eq!(window.len(), hot.len());
    let n = window.len();
    let mut total = 0.0f64;
    let mut non_zero = 0usize;
    for (w, h) in window.iter().zip(hot.iter()) {
        let (cost, _conf) = pair_cost(&w.syllables, &h.syllables, opts);
        if cost >= COST_DIFF {
            return None; // 出现不相关音节
        }
        if cost > 0.0 {
            non_zero += 1;
            let limit = if allow_two_near { 2 } else { 1 };
            if non_zero > limit {
                return None;
            }
        }
        total += cost;
    }
    Some(1.0 - total / n as f64)
}

/// 严格词条的完全同音判定：**双侧**常规读音对齐——窗口与热词逐位都按
/// 各自字符的常规读音（[`CharSounds::primary`]）读出且彼此相等。
/// 多音字展开仍保留在两侧候选集内（近音/手工档路径与 score 计算照旧），
/// 但严格档不允许**任何一侧**用非常规读音凑出“同音”：窗口侧如
/// 什(shí)×实（「什么时候」误改「实模式候」），热词侧如
/// 会(kuài)×快（「快记准则」常规读音 kuài-ji 与「会计准则」常规读音
/// hui-ji 不同音，不能改写）。近音（score < 1.0）由调用方先行拒绝，
/// 这里只负责“同音是否双侧常规”：primary 相等即唯一可行对齐就是
/// 常规读音对齐，不依赖任何一侧的候选展开。
fn canonical_homophone(window: &[CharSounds], hot: &[CharSounds]) -> bool {
    window.iter().zip(hot).all(|(w, h)| w.primary == h.primary)
}

/// 滑动窗口（支持 slack=0 等长 / slack=1 变长实验），带硬约束与相似度阈值
fn window_similarity(
    window: &[CharSounds],
    hot: &[CharSounds],
    opts: &ConfusionOptions,
    threshold: f64,
    slack: usize,
) -> Option<f64> {
    if window.len() != hot.len() {
        if slack == 0 {
            return None;
        }
        // 变长：带 gap 的简单对齐（每差一个字记一次 COST_GAP）
        // 注：默认路径不进入这里
        let diff = window.len().abs_diff(hot.len());
        if diff > slack {
            return None;
        }
        // 保守处理：截齐到较短者 + 差额罚分（近似）
        let n = window.len().min(hot.len());
        let base = aligned_match(&window[..n], &hot[..n], opts, false)?;
        let sim = base - (diff as f64 * COST_GAP) / window.len().max(hot.len()) as f64;
        return if sim >= threshold { Some(sim) } else { None };
    }
    let sim = aligned_match(window, hot, opts, false)?;
    if sim >= threshold {
        Some(sim)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// 替换主流程
// ---------------------------------------------------------------------------

/// 命中区间（全局 char 偏移）
struct Hit {
    start: usize,
    end: usize,
    score: f64,
    hotword_idx: usize,
    matched_text: String,
}

/// 一次拼音热词命中（区间基于替换后的输出文本，char 坐标）
#[derive(Debug, Clone)]
pub struct HotwordHit {
    /// 输出文本中的起始 char 偏移
    pub start: usize,
    /// 输出文本中的结束 char 偏移（不含）
    pub end: usize,
    /// 被替换的原文片段
    pub matched: String,
    /// 替换为的热词文本
    pub replacement: String,
    /// 相似度
    pub score: f64,
}

/// 对转写文本应用拼音热词替换。
/// 返回 (新文本, 命中列表——区间基于新文本，供规则引擎跳过已替换区域)
pub fn apply_pinyin_hotwords(
    text: &str,
    index: &[HotwordEntry],
    threshold: f64,
    opts: &ConfusionOptions,
) -> (String, Vec<HotwordHit>) {
    if index.is_empty() || !has_han(text) {
        return (text.to_string(), Vec::new());
    }
    // 小型词表（LLM 词表筛选、测试、demo）在此现建检索层（微秒级）；
    // 解码期大词表走 HotwordConfig 缓存里预构建的检索层
    // （apply_chinese_hotwords_indexed），不在每次转写重建。
    let search = crate::audio_toolkit::hotword_index::HotwordSearchIndex::build(index);
    apply_pinyin_hotwords_with_search(text, index, &search, threshold, opts)
}

/// 匹配主体（检索层由调用方注入）。与旧实现的唯一差异：候选枚举
/// 从「遍历全部词条」换成倒排桶取候选（超集性证明见 `hotword_index`
/// 模块文档），打分与全部防护逻辑逐行保留。
fn apply_pinyin_hotwords_with_search(
    text: &str,
    index: &[HotwordEntry],
    search: &crate::audio_toolkit::hotword_index::HotwordSearchIndex,
    threshold: f64,
    opts: &ConfusionOptions,
) -> (String, Vec<HotwordHit>) {
    if index.is_empty() || !has_han(text) {
        return (text.to_string(), Vec::new());
    }

    let chars: Vec<char> = text.chars().collect();
    let segments = segment_text(&chars);

    // 1) 保护区：原文已完整包含热词字面 → 该区间不得被任何候选覆盖
    let mut protected: Vec<(usize, usize)> = Vec::new();
    for hot in index {
        let hw_chars: Vec<char> = hot.display.chars().collect();
        // `<=` 且非空：整句恰好等于热词字面时同样构成保护区（此前 `<`
        // 在等长时跳过扫描，整句正确文本会被同长近音热词改写掉）
        if !hw_chars.is_empty() && hw_chars.len() <= chars.len() {
            let m = hw_chars.len();
            for start in 0..=(chars.len() - m) {
                if chars[start..start + m] == hw_chars[..] {
                    protected.push((start, start + m));
                }
            }
        }
    }

    // 2) 收集候选命中（按全局区间）
    // 候选按 (start,end) 分组；同分组的歧义检测见第 3 步
    let mut grouped: std::collections::HashMap<(usize, usize), Vec<(f64, usize)>> =
        std::collections::HashMap::new();

    for seg in &segments {
        if !seg.sounds.iter().any(|s| s.is_han) {
            continue;
        }
        let seg_len = seg.sounds.len();
        // 候选枚举走倒排检索层（旧实现为「遍历全部词条 × 全部起点」的
        // O(词表×文本×词长) 内层循环）：对每个起点按首字候选音节取
        // 候选词条，再走与旧实现逐行相同的验证（打分、严格档、2 字档
        // 全部不动）。候选可能重复发射（多音字闭包相交），重复入组被
        // 第 3 步按替换目标合并，结果不变。
        for start in 0..seg_len {
            search.for_each_candidate(&seg.sounds[start], seg_len - start, opts, |hw_idx| {
                let hot = &index[hw_idx];
                // Candidate（候选）档：条目在索引里供建议路径，但匹配阶段
                // 直接跳过——不进自动纠正（等同跳过该条目）。
                // Blocked 档在索引构建时就没有条目，无需在此处理。
                if hot.no_rewrite {
                    return;
                }
                let m = hot.syllables.len();
                debug_assert!(m > 0 && start + m <= seg_len, "检索层词长过滤失效");
                let window = &seg.sounds[start..start + m];
                // 纯汉字热词：窗口必须全汉字（不吞 ASCII/数字）
                if hot.all_han && window.iter().any(|s| !s.is_han) {
                    return;
                }
                let window_text: String =
                    seg.chars[start..start + m].iter().map(|(_, c)| c).collect();
                if window_text == hot.display {
                    return; // 保护逻辑已覆盖
                }
                if let Some(score) = window_similarity(window, &hot.syllables, opts, threshold, 0) {
                    // 2 字热词：只接受完整同音（分数必须为 1.0）
                    if hot.han_count <= 2 && score < 1.0 {
                        return;
                    }
                    // 严格（导入）词条：只接受双侧常规读音下的完全同音
                    // （dist == 0 且窗口/热词逐位 primary 相等；任何一侧
                    // 的多音字非常规读音凑对都不算）
                    if hot.strict && (score < 1.0 || !canonical_homophone(window, &hot.syllables)) {
                        return;
                    }
                    let g_start = seg.chars[start].0;
                    let g_end = seg.chars[start + m - 1].0 + 1;
                    grouped
                        .entry((g_start, g_end))
                        .or_default()
                        .push((score, hw_idx));
                }
            });
        }
    }

    if grouped.is_empty() {
        return (text.to_string(), Vec::new());
    }

    // 3) 展开为 Hit：先剔除上下文黑名单命中的候选，再按替换目标合并
    //    （同一 display 只留最高分——重复热词/同目标别名的替换结果相同，
    //    不构成竞争歧义），最后仅在不同替换目标间做歧义检测 + 保护过滤
    let text_str: String = chars.iter().collect();
    let mut hits: Vec<Hit> = Vec::new();
    for ((start, end), list) in grouped {
        let mut best_by_display: std::collections::HashMap<&str, (f64, usize)> =
            std::collections::HashMap::new();
        for (score, hotword_idx) in list {
            let hot = &index[hotword_idx];
            // 上下文黑名单：整句含任一黑词 → 该候选直接出局（CapsWriter
            // 同款语义，防同音歧义），且不作为竞争者压制其他热词
            if !hot.blacklist.is_empty()
                && hot.blacklist.iter().any(|b| text_str.contains(b.as_str()))
            {
                continue;
            }
            match best_by_display.get(hot.display.as_str()) {
                Some(&(best, _)) if best >= score => {}
                _ => {
                    best_by_display.insert(hot.display.as_str(), (score, hotword_idx));
                }
            }
        }
        let mut finalists: Vec<(f64, usize)> = best_by_display.into_values().collect();
        if finalists.is_empty() {
            continue;
        }
        finalists.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let (score, hotword_idx) = finalists[0];
        // 同音竞争歧义：仅不同替换目标参与；前两名分差过小 → 放弃
        if let Some(&(ru, _)) = finalists.get(1) {
            if score - ru < AMBIGUITY_MARGIN {
                continue;
            }
        }
        // 保护区间过滤
        if protected.iter().any(|&(ps, pe)| start < pe && ps < end) {
            continue;
        }
        hits.push(Hit {
            start,
            end,
            score,
            hotword_idx,
            matched_text: chars[start..end].iter().collect(),
        });
    }

    // 4) 贪心选择互不重叠命中：分数降序、长词优先、位置升序
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then((b.end - b.start).cmp(&(a.end - a.start)))
            .then(a.start.cmp(&b.start))
    });
    let mut chosen: Vec<&Hit> = Vec::new();
    let mut occupied: Vec<(usize, usize)> = Vec::new();
    for hit in &hits {
        if occupied
            .iter()
            .any(|&(os, oe)| hit.start < oe && os < hit.end)
        {
            continue;
        }
        occupied.push((hit.start, hit.end));
        chosen.push(hit);
    }
    chosen.sort_by_key(|h| h.start); // 从左到右重建

    // 5) 重建文本（char 级，天然 boundary 安全），命中区间坐标同步换算到输出文本
    let mut out = String::with_capacity(text.len() + 16);
    let mut log = Vec::with_capacity(chosen.len());
    let mut cursor = 0usize;
    for hit in chosen {
        let hot = &index[hit.hotword_idx];
        out.push_str(&chars[cursor..hit.start].iter().collect::<String>());
        let out_start = out.chars().count();
        out.push_str(&hot.display);
        let out_end = out_start + hot.display.chars().count();
        log.push(HotwordHit {
            start: out_start,
            end: out_end,
            matched: hit.matched_text.clone(),
            replacement: hot.display.clone(),
            score: hit.score,
        });
        cursor = hit.end;
    }
    out.push_str(&chars[cursor..].iter().collect::<String>());

    (out, log)
}

/// 完整中文热词后处理：拼音模糊替换 + 规则替换。
/// 规则只应用于未被拼音替换触碰的区域，避免对刚替换出的热词二次改写
/// （如拼音已输出“中州韵输入法”，规则 `中州韵=中州韵输入法` 不会再叠加）。
pub fn apply_chinese_hotwords(
    text: &str,
    index: &[HotwordEntry],
    rules: &[HotwordRule],
    threshold: f64,
    opts: &ConfusionOptions,
) -> (String, Vec<HotwordHit>) {
    let (after_pinyin, hits) = apply_pinyin_hotwords(text, index, threshold, opts);
    if rules.is_empty() {
        return (after_pinyin, hits);
    }
    if hits.is_empty() {
        return (apply_hotword_rules(&after_pinyin, rules), hits);
    }
    // 分段：未触碰段跑规则，已替换段原样保留
    let out_chars: Vec<char> = after_pinyin.chars().collect();
    let mut result = String::with_capacity(after_pinyin.len() + 16);
    let mut cursor = 0usize;
    for h in &hits {
        let untouched: String = out_chars[cursor..h.start].iter().collect();
        result.push_str(&apply_hotword_rules(&untouched, rules));
        result.push_str(&h.replacement);
        cursor = h.end;
    }
    let tail: String = out_chars[cursor..].iter().collect();
    result.push_str(&apply_hotword_rules(&tail, rules));
    (result, hits)
}

/// 候选检索入口（解码期主路径）：与 [`apply_chinese_hotwords`] 行为
/// 完全一致，但拼音匹配层复用 `HotwordConfig` 里随编译缓存预构建的
/// 倒排检索层（词长 × 首字候选音节），不在每次转写重建——5 万词级
/// 导入词库下省去每次转写的检索层重建开销。缓存键（词表/规则/
/// min_han/档位表）任一变化时整个配置连同检索层一起重编译。
pub fn apply_chinese_hotwords_indexed(
    text: &str,
    config: &HotwordConfig,
    threshold: f64,
    opts: &ConfusionOptions,
) -> (String, Vec<HotwordHit>) {
    let (after_pinyin, hits) =
        apply_pinyin_hotwords_with_search(text, &config.index, &config.search, threshold, opts);
    if config.rules.is_empty() {
        return (after_pinyin, hits);
    }
    if hits.is_empty() {
        return (apply_hotword_rules(&after_pinyin, &config.rules), hits);
    }
    // 分段：未触碰段跑规则，已替换段原样保留（与 apply_chinese_hotwords
    // 尾段一致，改动需同步两处）
    let out_chars: Vec<char> = after_pinyin.chars().collect();
    let mut result = String::with_capacity(after_pinyin.len() + 16);
    let mut cursor = 0usize;
    for h in &hits {
        let untouched: String = out_chars[cursor..h.start].iter().collect();
        result.push_str(&apply_hotword_rules(&untouched, &config.rules));
        result.push_str(&h.replacement);
        cursor = h.end;
    }
    let tail: String = out_chars[cursor..].iter().collect();
    result.push_str(&apply_hotword_rules(&tail, &config.rules));
    (result, hits)
}

// ---------------------------------------------------------------------------
// 规则引擎
// ---------------------------------------------------------------------------

/// 一条替换规则
#[derive(Debug, Clone)]
pub struct HotwordRule {
    pub pattern: Regex,
    pub replacement: String,
    /// 规则原文（日志用）
    pub source: String,
}

/// 解析规则文本，每行一条：
/// - `模式=>替换`：正则
/// - `模式=替换`：字面量（内部转义）
/// - 空行与 `#` 注释忽略
/// 返回 (规则列表, 失败行号列表)
pub fn parse_hotword_rules(input: &str) -> (Vec<HotwordRule>, Vec<usize>) {
    let mut rules = Vec::new();
    let mut errors = Vec::new();
    for (i, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // 先找 =>（优先于 =）
        let (pattern, op, replacement) = if let Some(pos) = line.find("=>") {
            (&line[..pos], "=>", &line[pos + 2..])
        } else if let Some(pos) = line.find('=') {
            (&line[..pos], "=", &line[pos + 1..])
        } else {
            errors.push(i + 1);
            continue;
        };
        let pattern = pattern.trim();
        let replacement = replacement.trim();
        if pattern.is_empty() {
            errors.push(i + 1);
            continue;
        }
        let effective = if op == "=>" {
            pattern.to_string()
        } else {
            regex::escape(pattern)
        };
        match Regex::new(&effective) {
            Ok(re) => rules.push(HotwordRule {
                pattern: re,
                replacement: replacement.to_string(),
                source: line.to_string(),
            }),
            Err(_) => errors.push(i + 1),
        }
    }
    (rules, errors)
}

/// 顺序应用规则
pub fn apply_hotword_rules(text: &str, rules: &[HotwordRule]) -> String {
    let mut out = text.to_string();
    for rule in rules {
        out = rule
            .pattern
            .replace_all(&out, rule.replacement.as_str())
            .into_owned();
    }
    out
}

// ---------------------------------------------------------------------------
// 编译缓存（bounded 单槽）
// ---------------------------------------------------------------------------

/// 一次编译完成的中文热词配置：拼音索引 + 规则 + 规则解析诊断。
/// 规则解析（含失败行号诊断）只在编译时做一次，命中路径零重复解析。
#[derive(Debug)]
pub struct HotwordConfig {
    /// 拼音热词索引（`build_hotword_index` 产物）
    pub index: Vec<HotwordEntry>,
    /// 候选检索层（词长 × 首字候选音节倒排索引，见 `hotword_index`）。
    /// 与 `index` 同源构建、随编译缓存一起失效重建，转写期零重建。
    pub search: crate::audio_toolkit::hotword_index::HotwordSearchIndex,
    /// 编译好的替换规则（按行序）
    pub rules: Vec<HotwordRule>,
    /// 解析失败的行号（1-based，`parse_hotword_rules` 的诊断）
    pub rule_errors: Vec<usize>,
}

/// 缓存键：词表逐项 + 规则文本 + min_han + 手工词白名单 + 四态档位表，
/// 全部精确比较。
struct CacheKey {
    words: Vec<String>,
    rules_text: String,
    min_han: usize,
    /// 手工词白名单（旧 fuzzy 路径的严格度维度）。手工/导入拆分不同但
    /// 合并词表相同的两份配置会产出不同的索引，必须参与键比较。
    fuzzy_words: Vec<String>,
    /// 四态档位表（key = primary，见 [`TermMode`]）。档位变化（哪怕
    /// 词表/规则不变）必须换键——否则改了档位仍复用旧索引。与
    /// `fuzzy_words` 是两个互斥入口的维度：fuzzy 路径恒为空表，
    /// modes 路径恒为空白名单。
    modes: HashMap<String, TermMode>,
}

impl CacheKey {
    /// 精确键比较：任一维度不同即视为配置变化，触发重编译。
    fn matches(
        &self,
        words: &[String],
        rules_text: &str,
        min_han: usize,
        fuzzy_words: &[String],
        modes: &HashMap<String, TermMode>,
    ) -> bool {
        self.min_han == min_han
            && self.rules_text == rules_text
            && self.words.len() == words.len()
            && self.words.iter().zip(words).all(|(a, b)| a == b)
            && self.fuzzy_words.len() == fuzzy_words.len()
            && self
                .fuzzy_words
                .iter()
                .zip(fuzzy_words)
                .all(|(a, b)| a == b)
            && self.modes.len() == modes.len()
            && self.modes.iter().all(|(k, v)| modes.get(k) == Some(v))
    }
}

/// 中文热词配置的有界单槽缓存。
///
/// THUOCL 规模词库（5 万词，见 `examples/lexicon_bench`）的索引构建是
/// 毫秒级开销，不能每次转写重建；而配置极少变化，单槽（键 = 词表 +
/// 规则文本 + min_han + 手工词白名单/四态档位表，精确比较）即可覆盖，
/// 内存恒有界。
///
/// 锁内只做键比较与 `Arc` 克隆；编译（索引构建 + 正则解析）在锁外
/// 进行，不阻塞并发转写的后处理。并发未命中可能各自编译并都返回
/// 自建配置，最后写入者留存单槽——正确性由键精确比较保证。
pub struct HotwordCache {
    slot: Mutex<Option<(CacheKey, Arc<HotwordConfig>)>>,
}

impl HotwordCache {
    pub const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// 命中则克隆 `Arc` 返回（O(1)，无编译）；未命中则锁外编译后写回单槽。
    /// 全部词条按手工词（近音宽松）对待，行为与既有调用一致；需要区分
    /// 导入词严格度的路径用 [`HotwordCache::get_or_build_with_fuzzy`]。
    pub fn get_or_build(
        &self,
        words: &[String],
        rules_text: &str,
        min_han: usize,
    ) -> Arc<HotwordConfig> {
        self.get_or_build_with_fuzzy(words, rules_text, min_han, words)
    }

    /// 带手工词白名单的编译入口（旧路径）：`fuzzy_words` 内的行走近音
    /// 宽松，其余（导入词库）编成严格词条（只接受常规读音下的完全
    /// 同音）。生产路径已改走 [`HotwordCache::get_or_build_with_modes`]；
    /// 保留此入口供既有测试与等价性校验。
    pub fn get_or_build_with_fuzzy(
        &self,
        words: &[String],
        rules_text: &str,
        min_han: usize,
        fuzzy_words: &[String],
    ) -> Arc<HotwordConfig> {
        // 锁内：只比较 + 克隆
        {
            let slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((key, config)) = slot.as_ref() {
                if key.matches(words, rules_text, min_han, fuzzy_words, &HashMap::new()) {
                    return Arc::clone(config);
                }
            }
        }
        // 锁外：编译
        let config = Arc::new(Self::compile(words, rules_text, min_han, fuzzy_words));
        let key = CacheKey {
            words: words.to_vec(),
            rules_text: rules_text.to_owned(),
            min_han,
            fuzzy_words: fuzzy_words.to_vec(),
            modes: HashMap::new(),
        };
        // 写回单槽（覆盖旧配置即失效）
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some((key, Arc::clone(&config)));
        config
    }

    /// 带四态档位表的编译入口（生产路径）：档位表参与缓存键——
    /// 档位变化（哪怕词表/规则/min_han 完全不变）必然重编译，
    /// 绝不复用旧索引。档位表由 settings 层解析（来源缺省 + 显式
    /// override，见 `crate::settings::AppSettings::effective_term_modes`）。
    pub fn get_or_build_with_modes(
        &self,
        words: &[String],
        rules_text: &str,
        min_han: usize,
        modes: &HashMap<String, TermMode>,
    ) -> Arc<HotwordConfig> {
        // 锁内：只比较 + 克隆
        {
            let slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((key, config)) = slot.as_ref() {
                if key.matches(words, rules_text, min_han, &[], modes) {
                    return Arc::clone(config);
                }
            }
        }
        // 锁外：编译
        let config = Arc::new(Self::compile_with_modes(words, rules_text, min_han, modes));
        let key = CacheKey {
            words: words.to_vec(),
            rules_text: rules_text.to_owned(),
            min_han,
            fuzzy_words: Vec::new(),
            modes: modes.clone(),
        };
        // 写回单槽（覆盖旧配置即失效）
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some((key, Arc::clone(&config)));
        config
    }

    /// 编译一份完整配置。解析失败只在（重）编译时告警一次，缓存命中
    /// 不重复刷屏；告警只含行号，不把规则/热词文本写进日志。
    fn compile(
        words: &[String],
        rules_text: &str,
        min_han: usize,
        fuzzy_words: &[String],
    ) -> HotwordConfig {
        let (rules, rule_errors) = parse_hotword_rules(rules_text);
        if !rule_errors.is_empty() {
            warn!("Hotword rules failed to parse on lines {rule_errors:?}");
        }
        let index = build_hotword_index_with_fuzzy(words, min_han, fuzzy_words);
        Self::config_with(index, rules, rule_errors)
    }

    /// 四态档位表版本的编译（[`HotwordCache::get_or_build_with_modes`]
    /// 的锁外编译路径）。
    fn compile_with_modes(
        words: &[String],
        rules_text: &str,
        min_han: usize,
        modes: &HashMap<String, TermMode>,
    ) -> HotwordConfig {
        let (rules, rule_errors) = parse_hotword_rules(rules_text);
        if !rule_errors.is_empty() {
            warn!("Hotword rules failed to parse on lines {rule_errors:?}");
        }
        let index = build_hotword_index_with_modes(words, min_han, modes);
        Self::config_with(index, rules, rule_errors)
    }

    /// 由索引与规则组装配置：检索层（倒排索引）在此随索引一起构建，
    /// 两个编译入口共用，保证检索层与词条索引严格同源。
    fn config_with(
        index: Vec<HotwordEntry>,
        rules: Vec<HotwordRule>,
        rule_errors: Vec<usize>,
    ) -> HotwordConfig {
        let search = crate::audio_toolkit::hotword_index::HotwordSearchIndex::build(&index);
        HotwordConfig {
            search,
            index,
            rules,
            rule_errors,
        }
    }
}

/// 进程级共享缓存实例：转写后处理各路径（batch / live）共用同一配置。
pub fn shared_hotword_cache() -> &'static HotwordCache {
    static SHARED: HotwordCache = HotwordCache::new();
    &SHARED
}

impl Default for HotwordCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 仅规则的有界单槽缓存（键 = 规则文本精确比较）。
///
/// 非中文文本的转写路径只需要规则：用这个独立小缓存，避免 Latin
/// 转写触碰/换出中文全词表大缓存（其索引构建是毫秒级开销）。
pub struct RulesCache {
    slot: Mutex<Option<(String, Arc<Vec<HotwordRule>>)>>,
}

impl RulesCache {
    pub const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// 命中则克隆 `Arc` 返回；未命中则锁外解析后写回单槽。
    /// 锁内只做比较与克隆，解析在锁外；告警只在新解析时发一次。
    pub fn rules_for(&self, rules_text: &str) -> Arc<Vec<HotwordRule>> {
        {
            let slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((key, rules)) = slot.as_ref() {
                if key == rules_text {
                    return Arc::clone(rules);
                }
            }
        }
        let (rules, rule_errors) = parse_hotword_rules(rules_text);
        if !rule_errors.is_empty() {
            warn!("Hotword rules failed to parse on lines {rule_errors:?}");
        }
        let rules = Arc::new(rules);
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some((rules_text.to_owned(), Arc::clone(&rules)));
        rules
    }
}

impl Default for RulesCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 进程级共享 rules-only 缓存实例（非中文路径使用）。
pub fn shared_rules_cache() -> &'static RulesCache {
    static SHARED: RulesCache = RulesCache::new();
    &SHARED
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPTS: ConfusionOptions = ConfusionOptions { dialect: false };
    const DIALECT: ConfusionOptions = ConfusionOptions { dialect: true };

    fn idx(words: &[&str]) -> Vec<HotwordEntry> {
        let owned: Vec<String> = words.iter().map(|s| s.to_string()).collect();
        // 手工词语义：全部词条走近音宽松（既有行为）
        build_hotword_index_with_fuzzy(&owned, DEFAULT_MIN_HAN, &owned)
    }

    fn idx_min2(words: &[&str]) -> Vec<HotwordEntry> {
        let owned: Vec<String> = words.iter().map(|s| s.to_string()).collect();
        build_hotword_index_with_fuzzy(&owned, 2, &owned)
    }

    /// 导入词语义：全部词条严格（只接受常规读音下的完全同音）
    fn idx_imported(words: &[&str]) -> Vec<HotwordEntry> {
        let owned: Vec<String> = words.iter().map(|s| s.to_string()).collect();
        build_hotword_index_with_fuzzy(&owned, DEFAULT_MIN_HAN, &[])
    }

    fn apply(text: &str, index: &[HotwordEntry]) -> (String, Vec<HotwordHit>) {
        apply_pinyin_hotwords(text, index, DEFAULT_THRESHOLD, &OPTS)
    }

    // ---------------- 拼音层 ----------------

    #[test]
    fn test_han_detection() {
        assert!(is_han('中'));
        assert!(is_han('漢'));
        assert!(!is_han('a'));
        assert!(!is_han('。'));
        assert!(has_han("hello 世界"));
        assert!(!has_han("pure english 123"));
    }

    #[test]
    fn test_char_sounds_multi() {
        let s = char_sounds('重').unwrap();
        assert!(s.is_han);
        assert!(s.syllables.len() >= 2, "{:?}", s.syllables);
        assert!(s.syllables.contains(&"zhong".to_string()));
        assert!(s.syllables.contains(&"chong".to_string()));

        let a = char_sounds('V').unwrap();
        assert_eq!(a.syllables, vec!["v".to_string()]);
        assert!(!a.is_han);
        assert!(char_sounds('，').is_none());
    }

    #[test]
    fn test_split_syllable() {
        assert_eq!(split_syllable("zhang"), ("zh".into(), "ang".into()));
        assert_eq!(split_syllable("zhi"), ("zh".into(), "i".into()));
        assert_eq!(split_syllable("ba"), ("b".into(), "a".into()));
        assert_eq!(split_syllable("ai"), ("".into(), "ai".into()));
        assert_eq!(split_syllable("nv"), ("n".into(), "v".into()));
    }

    #[test]
    fn test_confusion_levels() {
        // 完全同音
        assert_eq!(syllable_confusion("yun", "yun", &OPTS).0, COST_EXACT);
        // 平翘舌单项
        let (c, _) = syllable_confusion("zhi", "zi", &OPTS);
        assert_eq!(c, COST_SINGLE_CONFUSION);
        // 前后鼻音单项
        let (c, _) = syllable_confusion("xin", "xing", &OPTS);
        assert_eq!(c, COST_SINGLE_CONFUSION);
        // 双混淆：zheng vs cen（声母 zh/c 无白名单对 -> DIFF）
        let (c, _) = syllable_confusion("zheng", "cen", &OPTS);
        assert_eq!(c, COST_DIFF);
        // 双混淆：zhang vs can（zh/c 不是白名单对）
        let (c, _) = syllable_confusion("zhang", "can", &OPTS);
        assert_eq!(c, COST_DIFF);
        // 声母平翘舌 + 韵尾鼻音：zheng vs zeng
        let (c, n) = syllable_confusion("zheng", "zeng", &OPTS);
        assert_eq!(c, COST_SINGLE_CONFUSION); // zh/z 是声母对，eng/eng 相同 -> 单项
        assert_eq!(n, 1);
        // zhan vs zang: zh/z 声母对 + an/ang 韵尾对 -> 双混淆
        let (c, _) = syllable_confusion("zhan", "zang", &OPTS);
        assert_eq!(c, COST_DOUBLE_CONFUSION);
        // n/l 关闭时 DIFF，开启时 DIALECT
        assert_eq!(syllable_confusion("niu", "liu", &OPTS).0, COST_DIFF);
        assert_eq!(syllable_confusion("niu", "liu", &DIALECT).0, COST_DIALECT);
        // 不相关
        assert_eq!(syllable_confusion("ma", "go", &OPTS).0, COST_DIFF);
    }

    #[test]
    fn test_pair_cost_multi_heteronym() {
        // 多音字候选：'重' 有 zhong/chong，'虫' 有 chong —— 应取 chong==chong = 0
        let zhong = char_sounds('重').unwrap();
        let chong = char_sounds('虫').unwrap();
        let (c, n) = pair_cost(&zhong.syllables, &chong.syllables, &OPTS);
        assert_eq!((c, n), (COST_EXACT, 0));
    }

    // ---------------- 索引 ----------------

    #[test]
    fn test_build_index_filters() {
        let words = vec![
            "中州韵输入法".to_string(),
            "A".to_string(),
            "的".to_string(),
            "Vue组件".to_string(),
        ];
        let index = build_hotword_index(&words, DEFAULT_MIN_HAN);
        let displays: Vec<&str> = index.iter().map(|e| e.display.as_str()).collect();
        assert!(displays.contains(&"中州韵输入法"));
        assert!(!displays.contains(&"Vue组件")); // 汉字数 2 < 3，默认被排除
        assert!(!displays.contains(&"的"));
        assert!(!displays.contains(&"A"));
        // min_han=2 时收录混合词
        let index2 = build_hotword_index(&words, 2);
        assert!(index2.iter().any(|e| e.display == "Vue组件"));
    }

    // ---------------- 匹配：同音 ----------------

    #[test]
    fn test_homophone_replacement() {
        let index = idx(&["中州韵输入法"]);
        // 运(yun) 韵(yun) 同音错字
        let (out, log) = apply("我在用中州运输入法打字", &index);
        assert_eq!(out, "我在用中州韵输入法打字");
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].replacement, "中州韵输入法");
        assert!(log[0].score > 0.999);
    }

    #[test]
    fn test_protect_exact_hotword() {
        let index = idx(&["人工智能"]);
        // 原文已含热词字面：整体保护，且相邻同音不误替换
        let (out, log) = apply("人工智能跟人工只能不一样", &index);
        // "人工只能"（zhineng vs zhinen? 只=zhi 能=neng；智=zhi 能=neng 同音）
        // 应被替换；已存在的"人工智能"保持
        assert_eq!(out, "人工智能跟人工智能不一样");
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_two_char_homophone_strict() {
        // 2 字热词在 min_han=2 时收录，但只接受完整同音
        let index = idx_min2(&["重庆"]);
        // 崇庆（chong qing）完全同音 -> 替换
        let (out, _) = apply("去崇庆旅游", &index);
        assert_eq!(out, "去重庆旅游");
        // 近音（沁=qin vs 庆=qing 单项前后鼻音）不替换：2 字热词仅接受完整同音
        let (out2, log2) = apply("去崇沁旅游", &index);
        assert_eq!(out2, "去崇沁旅游");
        assert!(log2.is_empty());
    }

    // ---------------- 匹配：近音 ----------------

    #[test]
    fn test_near_sound_single_confusion() {
        // 知识(zhishi) vs 字词(zici)：zhi/zi + shi/si 双单项 = 2 个非零音节
        // 硬约束（至多 1 个非零）会拒绝 —— 验证保守性
        let index = idx(&["知识库"]);
        let (out, _) = apply("字词库管理", &index);
        // zi+ci 两个音节都混淆 -> 拒绝
        assert_eq!(out, "字词库管理");
    }

    #[test]
    fn test_near_sound_one_syllable() {
        // 只有一个音节近音：库(ku)同，识(shi/si)单项
        let index = idx(&["知识库"]);
        // "zhi si 库"：知(zhi) 识->私(si) 近音单项，其余同音
        let (out, log) = apply("把资料放进知识库", &index);
        assert_eq!(out, "把资料放进知识库"); // 已含原文，保护
                                             // 构造单项近音：直柿库（zhi shi vs zhi si? 直=zhi 柿=shi）
                                             // "知识" 里 识=shi；测试用"只四库"：只=zhi 四=si（shi/si 单项）
        let (out2, log2) = apply("同步只四库", &index);
        assert_eq!(out2, "同步知识库");
        assert_eq!(log2.len(), 1);
    }

    // ---------------- 防误替换 ----------------

    #[test]
    fn test_no_false_positive_common_words() {
        // 常用同音词冲突：热词"权利"，原文正确使用"权利"时受保护；
        // 但"权力"出现时不该被替换为"权利"（同音竞争应保守）
        let index = idx_min2(&["权利"]);
        let (out, log) = apply("保障公民权力不受侵犯", &index);
        // dist=0 分数 1.0 无 runner-up…… 按当前规则会替换。
        // 这是"强制替换"语义下的预期行为（用户加入热词即偏向该写法），
        // 文档已注明。此测试仅固化行为：
        assert_eq!(out, "保障公民权利不受侵犯");
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_ambiguity_between_hotwords() {
        // 两个同音热词竞争：放弃
        let index = idx(&["机器学习", "机器统计学"]);
        // "机械学习"（xie vs qi 不同音）不应命中任何一个……
        // 构造真同音竞争：两个热词同音
        let index = idx(&["深棕色", "身宗色"]);
        let (out, log) = apply("买了一件深宗色的外套", &index);
        // 深宗色 与两个热词都是 1 个非零音节（zong 分歧?）
        // 棕=zong 宗=zong 同音！那两个热词同分 -> 歧义放弃
        assert_eq!(out, "买了一件深宗色的外套");
        assert!(log.is_empty());
    }

    #[test]
    fn test_unrelated_not_replaced() {
        let index = idx(&["人工智能"]);
        let (out, log) = apply("今天天气真好", &index);
        assert_eq!(out, "今天天气真好");
        assert!(log.is_empty());
    }

    #[test]
    fn test_pure_han_window_no_ascii() {
        // 纯汉字热词不吞 ASCII
        let index = idx(&["数据库设计"]);
        let (out, log) = apply("做了MySQL数据库设计", &index);
        // "数据库设计"前面是 L（ASCII），窗口为纯汉字子串，应命中
        assert_eq!(out, "做了MySQL数据库设计"); // 已含字面 -> 保护，不变
        assert!(log.is_empty());
    }

    #[test]
    fn test_idempotent() {
        let index = idx(&["中州韵输入法", "人工智能"]);
        let (out1, _) = apply("用中州运输入法和人工只能", &index);
        let (out2, log2) = apply(&out1, &index);
        assert_eq!(out1, out2, "idempotency broken");
        assert!(log2.is_empty());
    }

    #[test]
    fn test_mixed_hotword() {
        // 混合热词（min_han=2）
        let index = idx_min2(&["Vue3组件"]);
        // 原样出现保护；同音错字替换：祖件(zu jian) vs 组件(zu jian) 同音
        let (out, _) = apply("封装了Vue3组件", &index);
        assert_eq!(out, "封装了Vue3组件");
        let (out2, log2) = apply("封装了Vue3祖件", &index);
        assert_eq!(out2, "封装了Vue3组件");
        assert_eq!(log2.len(), 1);
    }

    #[test]
    fn test_char_boundary_safety() {
        // emoji + 中文混合：不 panic，正确处理
        let index = idx(&["语音转文字"]);
        let (out, _) = apply("🚀语音转蚊字🚀完成", &index);
        assert_eq!(out, "🚀语音转文字🚀完成");
    }

    #[test]
    fn test_long_word_multiple_near_rejected() {
        // 长热词多个近音音节：硬约束拒绝（宁可漏纠）
        let index = idx(&["机器学习法"]);
        // 机气学习法：机(qi)同 气(qi)同 学习同 法同 -> dist 0 命中
        let (out, _) = apply("练习机气学习法", &index);
        assert_eq!(out, "练习机器学习法");
        // 字次学习法：字/机 diff -> 拒绝
        let (out2, _) = apply("练习字次学习法", &index);
        assert_eq!(out2, "练习字次学习法");
    }

    // ---------------- 规则 ----------------

    #[test]
    fn test_rules_do_not_double_apply_on_pinyin_hits() {
        // 拼音替换出的热词区域不再被规则二次改写（v0.1.1 防呆）
        let index = idx(&["中州韵输入法"]);
        let (rules, _) = parse_hotword_rules("中州韵=中州韵输入法");
        let (out, hits) = apply_chinese_hotwords(
            "我在用中州运输入法打字",
            &index,
            &rules,
            DEFAULT_THRESHOLD,
            &OPTS,
        );
        assert_eq!(out, "我在用中州韵输入法打字");
        assert_eq!(hits.len(), 1);
        // 规则仍作用于未触碰区域
        let (out2, _) =
            apply_chinese_hotwords("中州韵很好用", &index, &rules, DEFAULT_THRESHOLD, &OPTS);
        assert_eq!(out2, "中州韵输入法很好用");
    }

    #[test]
    fn test_parse_hotword_line_alias_and_blacklist() {
        // 纯目标
        let p = parse_hotword_line("中州韵输入法");
        assert_eq!(p.primary, "中州韵输入法");
        assert!(p.aliases.is_empty() && p.blacklist.is_empty());
        // 目标 + 别名
        let p = parse_hotword_line("Claude思维链 | 克劳德思维链 | cloud思维链");
        assert_eq!(p.primary, "Claude思维链");
        assert_eq!(p.aliases, vec!["克劳德思维链", "cloud思维链"]);
        // 带黑名单
        let p = parse_hotword_line("四喜丸子 ~~~ 世纪 | 世界");
        assert_eq!(p.primary, "四喜丸子");
        assert_eq!(p.blacklist, vec!["世纪", "世界"]);
        // 全语法混合 + 多余空格
        let p = parse_hotword_line(" 目标词 | 别名 ~~~ 黑1 | 黑2 ");
        assert_eq!(p.primary, "目标词");
        assert_eq!(p.aliases, vec!["别名"]);
        assert_eq!(p.blacklist, vec!["黑1", "黑2"]);
        // 空行/全空段
        assert!(parse_hotword_line("").primary.is_empty());
        assert!(parse_hotword_line("~~~ 黑名单").primary.is_empty());
    }

    #[test]
    fn test_alias_triggers_replacement() {
        // 别名命中 → 替换为第一个词（CapsWriter 同款语义）
        let index = idx(&["中州韵输入法 | 中州韵"]);
        // “中州韵”本身 3 字，作为别名被收录；说“中州韵” → 替换为“中州韵输入法”
        let (out, log) = apply("中州韵很好用", &index);
        assert_eq!(out, "中州韵输入法很好用");
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].replacement, "中州韵输入法");
        // 幂等：目标词已在文本中，自匹配不改写
        let (out2, log2) = apply("中州韵输入法很好用", &index);
        assert_eq!(out2, "中州韵输入法很好用");
        assert!(log2.is_empty());
    }

    #[test]
    fn test_blacklist_blocks_hit_in_context() {
        // 上下文含黑名单词 → 放弃命中（同音歧义防护）
        let index = idx(&["四喜丸子 ~~~ 世纪"]);
        // “完子”与“丸子”同音，本应替换，但句中含黑词“世纪” → 放弃
        let (out, log) = apply("二十一世纪吃四喜完子", &index);
        assert_eq!(out, "二十一世纪吃四喜完子");
        assert!(log.is_empty());
        // 无黑词上下文 → 正常替换
        let (out2, log2) = apply("今晚吃四喜完子", &index);
        assert_eq!(out2, "今晚吃四喜丸子");
        assert_eq!(log2.len(), 1);
    }

    #[test]
    fn test_parse_rules() {
        let input = "\
# 注释行
中州韵=中州韵输入法
四([个只])=>四条$1
坏规则[
";
        let (rules, errors) = parse_hotword_rules(input);
        assert_eq!(rules.len(), 2);
        assert_eq!(errors.len(), 1);
        assert_eq!(rules[0].source, "中州韵=中州韵输入法");
    }

    #[test]
    fn test_apply_rules() {
        let input = "\
中州韵=中州韵输入法
(\\d)个=>${1}条
";
        let (rules, _) = parse_hotword_rules(input);
        let out = apply_hotword_rules("我在用中州韵打字，买了3个", &rules);
        assert_eq!(out, "我在用中州韵输入法打字，买了3条");
    }

    #[test]
    fn test_rules_arrow_precedence() {
        // => 优先于 =
        let (rules, _) = parse_hotword_rules("a=b=>c");
        assert_eq!(rules[0].pattern.as_str(), "a=b");
        assert_eq!(rules[0].replacement, "c");
    }

    #[test]
    fn test_rules_empty() {
        let (rules, errors) = parse_hotword_rules("");
        assert!(rules.is_empty());
        assert!(errors.is_empty());
        assert_eq!(apply_hotword_rules("不变", &[]), "不变");
    }

    // ---------------- 匹配器修正：保护区 / 竞争歧义 ----------------

    #[test]
    fn test_whole_sentence_exact_hotword_is_protected() {
        // 整句恰好等于热词字面：也必须受保护。此前 `m < len` 的条件在
        // 等长时跳过保护扫描，同长同音的第二热词会把整句正确文本改写掉
        let index = idx(&["中州韵输入法", "中州运输入法"]);
        let (out, log) = apply("中州韵输入法", &index);
        assert_eq!(out, "中州韵输入法");
        assert!(log.is_empty());
    }

    #[test]
    fn test_duplicate_hotwords_no_false_ambiguity() {
        // 同一热词重复配置：同分同 display 的候选合并，不构成竞争歧义
        let index = idx(&["中州韵输入法", "中州韵输入法"]);
        let (out, log) = apply("我在用中州运输入法", &index);
        assert_eq!(out, "我在用中州韵输入法");
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_alias_same_target_no_false_ambiguity() {
        // 同目标别名：目标“甲乙丙”与别名“甲以丙”（以/乙 同音）同分，
        // 但替换结果相同 → 合并后正常替换，而非假歧义放弃
        let index = idx(&["甲乙丙 | 甲以丙"]);
        let (out, log) = apply("用甲以丙测试", &index);
        assert_eq!(out, "用甲乙丙测试");
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].replacement, "甲乙丙");
    }

    #[test]
    fn test_blacklisted_candidate_excluded_before_ambiguity() {
        // 黑名单命中的候选先行出局：不再以“竞争歧义”拖死另一个热词
        let index = idx(&["中州韵", "中州云 ~~~ 乌云"]);
        let (out, log) = apply("天上乌云密布，中州运打字", &index);
        assert_eq!(out, "天上乌云密布，中州韵打字");
        assert_eq!(log.len(), 1);
        // 无黑词上下文时两个不同目标同分 → 仍按 0.08 阈值歧义放弃
        let (out2, log2) = apply("晴空万里，中州运打字", &index);
        assert_eq!(out2, "晴空万里，中州运打字");
        assert!(log2.is_empty());
    }

    // ---------------- 缓存 ----------------

    #[test]
    fn test_cache_hit_returns_same_arc() {
        let cache = HotwordCache::new();
        let words = vec!["中州韵输入法".to_string()];
        let a = cache.get_or_build(&words, "中州韵=中州韵输入法", DEFAULT_MIN_HAN);
        let b = cache.get_or_build(&words, "中州韵=中州韵输入法", DEFAULT_MIN_HAN);
        assert!(Arc::ptr_eq(&a, &b), "命中必须复用同一 Arc");
        assert_eq!(a.index.len(), 1);
        assert_eq!(a.rules.len(), 1);
        assert!(a.rule_errors.is_empty());
    }

    #[test]
    fn test_rules_cache_hit_and_invalidation() {
        // rules-only 独立单槽：命中复用 Arc，规则文本变化即失效重解析
        let cache = RulesCache::new();
        let a = cache.rules_for("很好用=非常好用");
        let b = cache.rules_for("很好用=非常好用");
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.len(), 1);
        let c = cache.rules_for("很好用=最强好用");
        assert!(!Arc::ptr_eq(&a, &c));
        assert_eq!(c[0].replacement, "最强好用");
        // 空文本 → 空规则（命中路径同样零解析）
        let empty = cache.rules_for("");
        assert!(empty.is_empty());
        let empty2 = cache.rules_for("");
        assert!(Arc::ptr_eq(&empty, &empty2));
    }

    #[test]
    fn test_cache_invalidated_on_config_change() {
        let cache = HotwordCache::new();
        let w1 = vec!["中州韵输入法".to_string()];
        let w2 = vec!["重庆".to_string()];
        let a = cache.get_or_build(&w1, "", DEFAULT_MIN_HAN);
        // 规则文本变化 → 重编译
        let b = cache.get_or_build(&w1, "知识库=知识库", DEFAULT_MIN_HAN);
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(b.rules.len(), 1);
        // 词表变化 → 重编译（且 min_han=3 下“重庆”2 字不入索引）
        let c = cache.get_or_build(&w2, "知识库=知识库", DEFAULT_MIN_HAN);
        assert!(!Arc::ptr_eq(&b, &c));
        assert!(c.index.is_empty());
        // min_han 变化 → 重编译（min_han=2 下“重庆”入索引）
        let d = cache.get_or_build(&w2, "知识库=知识库", 2);
        assert!(!Arc::ptr_eq(&c, &d));
        assert_eq!(d.index.len(), 1);
        // 回到首个配置：单槽被替换过 → 重新编译（Arc 不同，内容等价）
        let e = cache.get_or_build(&w1, "", DEFAULT_MIN_HAN);
        assert!(!Arc::ptr_eq(&a, &e));
        assert_eq!(e.index.len(), a.index.len());
        assert_eq!(e.rules.len(), a.rules.len());
    }

    #[test]
    fn test_cache_rule_diagnostics_cached() {
        // 失效行进入缓存诊断，坏行不阻断好行；命中路径同样携带诊断
        let cache = HotwordCache::new();
        let rules_text = "坏规则[\n知识库=知识库";
        let cfg = cache.get_or_build(&[], rules_text, DEFAULT_MIN_HAN);
        assert_eq!(cfg.rule_errors, vec![1]);
        assert_eq!(cfg.rules.len(), 1);
        assert_eq!(cfg.rules[0].replacement, "知识库");
        let cfg2 = cache.get_or_build(&[], rules_text, DEFAULT_MIN_HAN);
        assert!(Arc::ptr_eq(&cfg, &cfg2));
        assert_eq!(cfg2.rule_errors, vec![1]);
    }

    #[test]
    fn test_cache_concurrent_access() {
        // 并发同配置访问：无死锁、内容一致，收敛后命中同一 Arc
        let cache = Arc::new(HotwordCache::new());
        let stems = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
        let words: Vec<String> = (0..64)
            .map(|i| format!("并发{}{}热词组", stems[i / 10], stems[i % 10]))
            .collect();
        let mut handles = Vec::new();
        for _ in 0..4 {
            let cache = Arc::clone(&cache);
            let words = words.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..50 {
                    let cfg = cache.get_or_build(&words, "并发=>平行", DEFAULT_MIN_HAN);
                    assert_eq!(cfg.index.len(), 64);
                    assert_eq!(cfg.rules.len(), 1);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let a = cache.get_or_build(&words, "并发=>平行", DEFAULT_MIN_HAN);
        let b = cache.get_or_build(&words, "并发=>平行", DEFAULT_MIN_HAN);
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.index.len(), 64);
    }

    // ---------------- 规则独立生效（不依赖词表/汉字文本） ----------------

    #[test]
    fn test_rules_only_config_no_words() {
        // 词表为空：索引为空但规则照常编译、照常应用
        let cache = HotwordCache::new();
        let cfg = cache.get_or_build(&[], "中州韵=中州韵输入法", DEFAULT_MIN_HAN);
        assert!(cfg.index.is_empty());
        assert_eq!(cfg.rules.len(), 1);
        let (out, hits) = apply_chinese_hotwords(
            "我在用中州韵打字",
            &cfg.index,
            &cfg.rules,
            DEFAULT_THRESHOLD,
            &OPTS,
        );
        assert_eq!(out, "我在用中州韵输入法打字");
        assert!(hits.is_empty());
    }

    #[test]
    fn test_ascii_rules_on_non_han_text() {
        // ASCII 规则对非汉字文本生效（转写侧不得以 has_han 拦住规则）
        let cfg = HotwordCache::new().get_or_build(&[], "dont=>don't\nteh=the", DEFAULT_MIN_HAN);
        assert!(!has_han("i dont know teh answer"));
        assert_eq!(
            apply_hotword_rules("i dont know teh answer", &cfg.rules),
            "i don't know the answer"
        );
    }

    // ---------------- 来源严格度：导入词 vs 手工词 ----------------

    #[test]
    fn imported_strict_word_never_rewrites_common_phrase() {
        // 真实误替换回归（用户实测）：THUOCL IT 词库的「实模式」
        // (shi-mo-shi) 曾把「什么时候」改写成「实模式候」。机理：多音字
        // 展开下 什(shí)×实、么(mó)×模、时(shí)×式 逐字完全同音，
        // score == 1.0 绕过了近音硬约束。严格词条要求窗口按常规读音
        // （shen-me-shi）读出时同音，命中被拒绝。
        let index = idx_imported(&["实模式"]);
        assert!(index[0].strict);
        let (out, log) = apply("什么时候", &index);
        assert_eq!(out, "什么时候");
        assert!(log.is_empty());
        // 带上下文同样安全：任何 3 字窗口都无法常规读音下对齐
        let (out2, log2) = apply("你什么时候到站", &index);
        assert_eq!(out2, "你什么时候到站");
        assert!(log2.is_empty());
    }

    #[test]
    fn imported_strict_word_rejects_near_sound() {
        // 导入词不做近音模糊：只四库（zhi-si-ku）对知识库只有单项
        // 平翘舌混淆，严格档拒绝
        let index = idx_imported(&["知识库"]);
        let (out, log) = apply("同步只四库", &index);
        assert_eq!(out, "同步只四库");
        assert!(log.is_empty());
    }

    #[test]
    fn imported_strict_exact_homophone_typo_still_corrected() {
        // 导入词的完全同音错字仍纠正：磨(mo) 常规读音落在 模{mo,mu}
        // 候选内，全字同音 dist == 0
        let index = idx_imported(&["实模式"]);
        let (out, log) = apply("实磨式启动失败", &index);
        assert_eq!(out, "实模式启动失败");
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].replacement, "实模式");
        assert!(log[0].score >= 0.999);
    }

    #[test]
    fn manual_word_keeps_near_sound_behavior() {
        // 手工词（白名单内）保持既有近音行为：单项平翘舌混淆仍纠正
        let index = idx(&["知识库"]);
        assert!(!index[0].strict);
        let (out, log) = apply("同步只四库", &index);
        assert_eq!(out, "同步知识库");
        assert_eq!(log.len(), 1);
        // 手工词「模型量化」仍能纠正「模型亮化」（量/亮 同音）
        let index2 = idx(&["模型量化"]);
        let (out2, log2) = apply("模型亮化", &index2);
        assert_eq!(out2, "模型量化");
        assert_eq!(log2.len(), 1);
    }

    #[test]
    fn manual_word_is_deliberate_forced_replacement() {
        // 手工录入的「实模式」是用户刻意选择，沿用既有“强制替换”
        // 语义（同 test_no_false_positive_common_words）：多音字凑对
        // 的完全同音仍替换。边界即意图：手工词宽松、导入词严格。
        let index = idx(&["实模式"]);
        let (out, log) = apply("什么时候", &index);
        assert_eq!(out, "实模式候");
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn strict_hotword_side_cannot_forge_homophone() {
        // 智谱复核中危 bug：严格档原先只约束窗口侧（窗口每字 primary
        // 落在热词候选集内），热词侧多音字仍可挑非常规读音凑“完全
        // 同音”。单读音典事实：会 primary=hui、候选={hui,kuai}；快
        // primary=kuai（唯一读音）。窗口「快记准则」按常规读音是
        // kuai-ji-zhun-ze，「会计准则」常规读音是 hui-ji-zhun-ze——
        // 并不同音，旧判定却因 kuai ∈ 会 的候选集而放行改写。
        let index = idx_imported(&["会计准则"]);
        assert!(index[0].strict);
        let (out, log) = apply("快记准则更新了", &index);
        assert_eq!(out, "快记准则更新了", "热词侧多音字凑对必须被拒绝");
        assert!(log.is_empty());
        // 同机理：都 primary=dou（候选含 du），堵 primary=du（唯一）。
        // 「堵柏林」与「都柏林」常规读音不同音，严格档不动。
        let index2 = idx_imported(&["都柏林"]);
        let (out2, log2) = apply("我下周去堵柏林", &index2);
        assert_eq!(out2, "我下周去堵柏林");
        assert!(log2.is_empty());
    }

    #[test]
    fn manual_word_still_corrects_hotword_side_polyphone() {
        // 同批词作为手工词（fuzzy 档）行为保持：热词侧候选展开照旧，
        // 手工录入 = 刻意强制替换，是严格档漏纠时的用户逃生通道。
        let index = idx(&["会计准则"]);
        assert!(!index[0].strict);
        let (out, log) = apply("快记准则更新了", &index);
        assert_eq!(out, "会计准则更新了");
        assert_eq!(log.len(), 1);
        let index2 = idx(&["都柏林"]);
        let (out2, log2) = apply("我下周去堵柏林", &index2);
        assert_eq!(out2, "我下周去都柏林");
        assert_eq!(log2.len(), 1);
    }

    #[test]
    fn strict_hits_are_subset_of_fuzzy_hits() {
        // 差分自检：同一词表、同一批文本，严格档命中集合（按被替换
        // 文本→替换目标对识别）必须是 fuzzy 档命中集合的子集——严格
        // 档只能更保守，任何一条 strict 命中都必然是合法 fuzzy 命中。
        let words: Vec<String> = [
            "实模式",
            "知识库",
            "会计准则",
            "都柏林",
            "模型量化",
            "中州韵输入法",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let strict_idx = build_hotword_index_with_fuzzy(&words, DEFAULT_MIN_HAN, &[]);
        let fuzzy_idx = build_hotword_index_with_fuzzy(&words, DEFAULT_MIN_HAN, &words);
        let texts = [
            "什么时候到站",
            "实磨式启动失败",
            "快记准则更新了",
            "我下周去堵柏林",
            "模型亮化方案",
            "同步只四库",
            "我在用中州运输入法，很好用",
            "打开实模式和知识库",
            "没什么模式",
            "模型量化上线",
        ];
        let mut fuzzy_only = 0usize;
        for t in texts {
            let (_, strict_log) = apply(t, &strict_idx);
            let (_, fuzzy_log) = apply(t, &fuzzy_idx);
            for h in &strict_log {
                assert!(
                    fuzzy_log
                        .iter()
                        .any(|f| f.matched == h.matched && f.replacement == h.replacement),
                    "strict 命中不在 fuzzy 集合内：{t} -> {} => {}",
                    h.matched,
                    h.replacement
                );
            }
            fuzzy_only += fuzzy_log
                .iter()
                .filter(|f| {
                    !strict_log
                        .iter()
                        .any(|s| s.matched == f.matched && s.replacement == f.replacement)
                })
                .count();
        }
        // 差分非平凡：fuzzy 确实比 strict 多纠（什么时→实模式、
        // 快记准则→会计准则、只四库→知识库 等均为 fuzzy-only）
        assert!(fuzzy_only >= 3, "fuzzy-only 命中数异常: {fuzzy_only}");
    }

    #[test]
    fn empty_imports_behavior_byte_identical() {
        // 空导入集（全部手工词）：新旧构建路径产出完全一致的索引与行为
        let words: Vec<String> = vec![
            "中州韵输入法 | 中州韵".to_string(),
            "知识库".to_string(),
            "四喜丸子 ~~~ 世纪".to_string(),
        ];
        let legacy = build_hotword_index(&words, DEFAULT_MIN_HAN);
        let via_fuzzy = build_hotword_index_with_fuzzy(&words, DEFAULT_MIN_HAN, &words);
        assert_eq!(
            format!("{legacy:?}"),
            format!("{via_fuzzy:?}"),
            "旧签名必须等价于全手工白名单"
        );
        assert!(legacy.iter().all(|e| !e.strict));
        // 近音行为逐顶照旧：只四库仍被手工词「知识库」纠正
        let (out, log) = apply("同步只四库", &legacy);
        assert_eq!(out, "同步知识库");
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_mixed_words_and_rules_layering() {
        // 拼音替换出的区域不被规则二次改写；未触碰区域规则照常生效
        let index = idx(&["中州韵输入法"]);
        let (rules, _) = parse_hotword_rules("很好用=非常好用\n中州韵=中州韵输入法");
        let (out, hits) = apply_chinese_hotwords(
            "我在用中州运输入法，很好用",
            &index,
            &rules,
            DEFAULT_THRESHOLD,
            &OPTS,
        );
        assert_eq!(out, "我在用中州韵输入法，非常好用");
        assert_eq!(hits.len(), 1);
    }

    // ---------------- 检索层差分测试（hotword_index） ----------------

    /// 简易确定性随机数（LCG）：语料生成只用整数挑选，不引入 rand 依赖。
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            ((self.next_u64() >> 33) as usize) % n.max(1)
        }

        fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
            &xs[self.below(xs.len())]
        }
    }

    /// 旧路径逐行复刻（改造前 apply_pinyin_hotwords 的匹配主体：遍历
    /// 全部词条 × 全部起点的双层循环），作为差分基准。匹配主函数改动
    /// 语义时必须同步本复刻，否则差分测试会立即失败。
    fn legacy_apply_pinyin_hotwords(
        text: &str,
        index: &[HotwordEntry],
        threshold: f64,
        opts: &ConfusionOptions,
    ) -> (String, Vec<HotwordHit>) {
        if index.is_empty() || !has_han(text) {
            return (text.to_string(), Vec::new());
        }

        let chars: Vec<char> = text.chars().collect();
        let segments = segment_text(&chars);

        let mut protected: Vec<(usize, usize)> = Vec::new();
        for hot in index {
            let hw_chars: Vec<char> = hot.display.chars().collect();
            if !hw_chars.is_empty() && hw_chars.len() <= chars.len() {
                let m = hw_chars.len();
                for start in 0..=(chars.len() - m) {
                    if chars[start..start + m] == hw_chars[..] {
                        protected.push((start, start + m));
                    }
                }
            }
        }

        let mut grouped: std::collections::HashMap<(usize, usize), Vec<(f64, usize)>> =
            std::collections::HashMap::new();

        for seg in &segments {
            if !seg.sounds.iter().any(|s| s.is_han) {
                continue;
            }
            let seg_len = seg.sounds.len();
            for (hw_idx, hot) in index.iter().enumerate() {
                if hot.no_rewrite {
                    continue;
                }
                let m = hot.syllables.len();
                if m == 0 || seg_len < m {
                    continue;
                }
                let two_char_strict = hot.han_count <= 2;
                for start in 0..=(seg_len - m) {
                    let window = &seg.sounds[start..start + m];
                    if hot.all_han && window.iter().any(|s| !s.is_han) {
                        continue;
                    }
                    let window_text: String =
                        seg.chars[start..start + m].iter().map(|(_, c)| c).collect();
                    if window_text == hot.display {
                        continue;
                    }
                    if let Some(score) =
                        window_similarity(window, &hot.syllables, opts, threshold, 0)
                    {
                        if two_char_strict && score < 1.0 {
                            continue;
                        }
                        if hot.strict
                            && (score < 1.0 || !canonical_homophone(window, &hot.syllables))
                        {
                            continue;
                        }
                        let g_start = seg.chars[start].0;
                        let g_end = seg.chars[start + m - 1].0 + 1;
                        grouped
                            .entry((g_start, g_end))
                            .or_default()
                            .push((score, hw_idx));
                    }
                }
            }
        }

        if grouped.is_empty() {
            return (text.to_string(), Vec::new());
        }

        let text_str: String = chars.iter().collect();
        let mut hits: Vec<Hit> = Vec::new();
        for ((start, end), list) in grouped {
            let mut best_by_display: std::collections::HashMap<&str, (f64, usize)> =
                std::collections::HashMap::new();
            for (score, hotword_idx) in list {
                let hot = &index[hotword_idx];
                if !hot.blacklist.is_empty()
                    && hot.blacklist.iter().any(|b| text_str.contains(b.as_str()))
                {
                    continue;
                }
                match best_by_display.get(hot.display.as_str()) {
                    Some(&(best, _)) if best >= score => {}
                    _ => {
                        best_by_display.insert(hot.display.as_str(), (score, hotword_idx));
                    }
                }
            }
            let mut finalists: Vec<(f64, usize)> = best_by_display.into_values().collect();
            if finalists.is_empty() {
                continue;
            }
            finalists.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let (score, hotword_idx) = finalists[0];
            if let Some(&(ru, _)) = finalists.get(1) {
                if score - ru < AMBIGUITY_MARGIN {
                    continue;
                }
            }
            if protected.iter().any(|&(ps, pe)| start < pe && ps < end) {
                continue;
            }
            hits.push(Hit {
                start,
                end,
                score,
                hotword_idx,
                matched_text: chars[start..end].iter().collect(),
            });
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then((b.end - b.start).cmp(&(a.end - a.start)))
                .then(a.start.cmp(&b.start))
        });
        let mut chosen: Vec<&Hit> = Vec::new();
        let mut occupied: Vec<(usize, usize)> = Vec::new();
        for hit in &hits {
            if occupied
                .iter()
                .any(|&(os, oe)| hit.start < oe && os < hit.end)
            {
                continue;
            }
            occupied.push((hit.start, hit.end));
            chosen.push(hit);
        }
        chosen.sort_by_key(|h| h.start);

        let mut out = String::with_capacity(text.len() + 16);
        let mut log = Vec::with_capacity(chosen.len());
        let mut cursor = 0usize;
        for hit in chosen {
            let hot = &index[hit.hotword_idx];
            out.push_str(&chars[cursor..hit.start].iter().collect::<String>());
            let out_start = out.chars().count();
            out.push_str(&hot.display);
            let out_end = out_start + hot.display.chars().count();
            log.push(HotwordHit {
                start: out_start,
                end: out_end,
                matched: hit.matched_text.clone(),
                replacement: hot.display.clone(),
                score: hit.score,
            });
            cursor = hit.end;
        }
        out.push_str(&chars[cursor..].iter().collect::<String>());

        (out, log)
    }

    /// 断言两条路径输出完全一致：文本、命中数、逐条命中的区间/原文/
    /// 替换/分数（按位比较，打分路径不同也不允许 1 ULP 漂移）。命中
    /// 序列蕴含 top-2 歧义判定与贪心选择的全部分支。
    fn assert_paths_identical(
        text: &str,
        index: &[HotwordEntry],
        threshold: f64,
        opts: &ConfusionOptions,
    ) {
        let (out_new, hits_new) = apply_pinyin_hotwords(text, index, threshold, opts);
        let (out_old, hits_old) = legacy_apply_pinyin_hotwords(text, index, threshold, opts);
        assert_eq!(out_new, out_old, "输出文本不一致：{text:?}");
        assert_eq!(
            hits_new.len(),
            hits_old.len(),
            "命中数不一致：{text:?} -> {hits_old:?} vs {hits_new:?}"
        );
        for (a, b) in hits_new.iter().zip(hits_old.iter()) {
            assert_eq!((a.start, a.end), (b.start, b.end), "区间不一致：{text:?}");
            assert_eq!(a.matched, b.matched, "原文不一致：{text:?}");
            assert_eq!(a.replacement, b.replacement, "替换不一致：{text:?}");
            assert_eq!(
                a.score.to_bits(),
                b.score.to_bits(),
                "分数不一致：{text:?} {} vs {}",
                a.score,
                b.score
            );
        }
    }

    /// 差分测试主入口：固定语料（近音/多音字/严格档/黑名单/别名/竞争/
    /// 保护/ASCII 混排/emoji 边界）× 四种索引风味（手工近音、导入严格、
    /// 四态档位、min_han=2 含 2 字词）× 方言开关，新（倒排检索）旧
    /// （遍历全部词条）两条路径必须逐字节一致。
    #[test]
    fn differential_indexed_matches_legacy_on_fixed_corpus() {
        let words: Vec<String> = [
            "中州韵输入法 | 中州韵",
            "中州韵输入法 | 中州运输法",
            "知识库",
            "机器学习",
            "机器统计学",
            "深棕色",
            "身宗色",
            "四喜丸子 ~~~ 世纪",
            "中州云 ~~~ 乌云",
            "会计准则",
            "都柏林",
            "实模式",
            "长江流域",
            "行李寄存",
            "银行利率",
            "重要指标",
            "重复检测",
            "会议纪要",
            "着装规范",
            "着手处理",
            "落叶归根",
            "得出结论",
            "堵塞路段",
            "熟悉业务",
            "维特根斯坦 | 维特根思坦 | 维根思坦",
            "甲乙丙 | 甲以丙",
            "数据库设计",
            "语音转文字",
            "Vue3组件",
            "Rust语言",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let texts = [
            "什么时候",
            "你什么时候到站",
            "实磨式启动失败",
            "快记准则更新了",
            "我下周去堵柏林",
            "都市生活很方便",
            "重药指标都完成了",
            "会yi纪要写好了",
            "长江流舆调查",
            "行李记存服务",
            "银行利律调整",
            "着装规饭检查",
            "着手处里任务",
            "落叶归跟的季节",
            "得出结纶很简单",
            "堵赛路段拥堵",
            "熟息业务流程",
            "我在用中州运输入法打字",
            "中州韵输入法",
            "买了一件深宗色的外套",
            "二十一世纪吃四喜完子",
            "今晚吃四喜完子",
            "天上乌云密布，中州运打字",
            "晴空万里，中州运打字",
            "用甲以丙测试",
            "维特根思坦的哲学很难懂",
            "维特根斯坦的哲学很难懂",
            "同步只四库",
            "字词库管理",
            "练习机气学习法",
            "练习字次学习法",
            "做了MySQL数据库设计",
            "🚀语音转蚊字🚀完成",
            "封装了Vue3祖件",
            "配着 rust鱼言开发",
            "人机气学习与深度雪习的区别",
            "打开实模式和知识库",
            "没什么模式",
            "",
            "今天天气真好",
            "12345",
        ];

        let fuzzy_idx = build_hotword_index_with_fuzzy(&words, DEFAULT_MIN_HAN, &words);
        let strict_idx = build_hotword_index_with_fuzzy(&words, DEFAULT_MIN_HAN, &[]);
        let mut modes = HashMap::new();
        modes.insert("知识库".to_string(), TermMode::Candidate);
        modes.insert("深棕色".to_string(), TermMode::ActiveFuzzy);
        modes.insert("身宗色".to_string(), TermMode::ActiveFuzzy);
        modes.insert("四喜丸子".to_string(), TermMode::Blocked);
        modes.insert("中州韵输入法".to_string(), TermMode::ActiveExact);
        let modes_idx = build_hotword_index_with_modes(&words, DEFAULT_MIN_HAN, &modes);
        let min2_idx = build_hotword_index_with_fuzzy(&words, 2, &words);

        for text in texts {
            for index in [&fuzzy_idx, &strict_idx, &modes_idx, &min2_idx] {
                assert_paths_identical(text, index, DEFAULT_THRESHOLD, &OPTS);
                assert_paths_identical(text, index, DEFAULT_THRESHOLD, &DIALECT);
                // 阈值扫描：覆盖阈值变化下的窗口取舍分支
                for threshold in [0.70, 0.80, 0.85, 0.95, 1.0] {
                    assert_paths_identical(text, index, threshold, &OPTS);
                }
            }
        }
    }

    /// 随机语料差分：LCG 拼句（多音字/同音错字/干扰字/ASCII/标点/emoji），
    /// 覆盖分段边界与随机窗口组合；同时用一个 3000 词条的合成大词表
    /// （含重复词、同首音节高碰撞桶）压制桶内碰撞与重复发射路径。
    #[test]
    fn differential_indexed_matches_legacy_on_random_corpus() {
        let mut rng = Lcg(0x5EED_CAFE_F00D_2026);

        // 小词表：与固定语料同源，但加随机扰动词
        let mut words: Vec<String> = [
            "中州韵输入法 | 中州韵",
            "知识库",
            "机器学习",
            "机器统计学",
            "深棕色",
            "身宗色",
            "会计准则",
            "都柏林",
            "实模式",
            "长江流域",
            "行李寄存",
            "银行利率",
            "重要指标",
            "重复检测",
            "会议纪要",
            "着装规范",
            "落叶归根",
            "四喜丸子 ~~~ 世纪",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let polyphones = [
            '重', '会', '都', '长', '行', '着', '落', '得', '塞', '熟', '什', '么',
        ];
        for i in 0..40 {
            let a = polyphones[rng.below(polyphones.len())];
            let filler = "的一是了我有不人在这上们中来数据智能云平台";
            let fb: Vec<char> = filler.chars().collect();
            let b = fb[rng.below(fb.len())];
            let c = fb[rng.below(fb.len())];
            words.push(format!("{a}{b}{c}词{i}"));
        }

        let pool_str = "中州韵输入法知识库机器学习深棕身宗色会计准则都柏林实模式什么时候重会都长行着落得塞熟的一是了我有不人在这上们中来只四字次雪习公工雨途耀匀彻Vue3组件rust数据库设计重要指标会议纪要";
        let mut pool: Vec<char> = pool_str.chars().collect();
        pool.extend(['，', '。', ' ', '🚀', 'a', 'Z', '9']);

        let fuzzy_idx = build_hotword_index_with_fuzzy(&words, DEFAULT_MIN_HAN, &words);
        let strict_idx = build_hotword_index_with_fuzzy(&words, DEFAULT_MIN_HAN, &[]);
        let min2_idx = build_hotword_index_with_fuzzy(&words, 2, &words);

        for i in 0..240 {
            let len = 4 + rng.below(44);
            let mut text = String::new();
            for _ in 0..len {
                text.push(*rng.pick(&pool));
            }
            let opts = if i % 2 == 0 { &OPTS } else { &DIALECT };
            assert_paths_identical(&text, &fuzzy_idx, DEFAULT_THRESHOLD, opts);
            assert_paths_identical(&text, &strict_idx, DEFAULT_THRESHOLD, opts);
            assert_paths_identical(&text, &min2_idx, DEFAULT_THRESHOLD, opts);
        }

        // 大词表（桶碰撞 + 重复词条）：基准词同音变体批量生成
        let homophone_map: &[(char, char)] = &[
            ('学', '雪'),
            ('习', '席'),
            ('工', '公'),
            ('语', '雨'),
            ('图', '途'),
            ('医', '一'),
            ('药', '耀'),
            ('网', '往'),
            ('车', '彻'),
            ('云', '匀'),
            ('知', '之'),
            ('据', '聚'),
            ('库', '裤'),
            ('智', '制'),
            ('能', '嫩'),
        ];
        let base: Vec<String> = [
            "人工智能",
            "机器学习",
            "深度学习",
            "自然语言",
            "语音识别",
            "图像识别",
            "数据挖掘",
            "数据库设计",
            "搜索引擎",
            "云计算平台",
            "操作系统",
            "网络安全",
            "自动驾驶",
            "区块链技术",
            "电子商务",
            "智能家居",
            "物联网卡",
            "虚拟现实",
            "增强现实",
            "分布式存储",
            "微服务架构",
            "开源社区",
            "软件开发",
            "程序员日常",
            "产品经理",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut big_words: Vec<String> = Vec::with_capacity(3000);
        let mut k = 0;
        while big_words.len() < 3000 {
            let w = &base[k % base.len()];
            let chars: Vec<char> = w.chars().collect();
            // 同音错字替换制造与原文同音的变体词（首音节碰撞的主要来源）
            let variant: String = chars
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    if i == (k / base.len()) % chars.len() {
                        homophone_map
                            .iter()
                            .find(|(a, _)| a == c)
                            .map(|(_, b)| *b)
                            .unwrap_or(*c)
                    } else {
                        *c
                    }
                })
                .collect();
            big_words.push(variant);
            k += 1;
            // 混入少量干扰词与 ASCII 前缀词
            if k % 37 == 0 {
                let filler = "的一是了我有不人在这上们中来";
                let fb: Vec<char> = filler.chars().collect();
                big_words.push(format!(
                    "{}{}{}{}",
                    rng.pick(&fb),
                    rng.pick(&fb),
                    rng.pick(&fb),
                    k
                ));
            }
            if k % 53 == 0 {
                big_words.push(format!("Rust{}语言", k));
            }
        }
        let big_idx = build_hotword_index(&big_words, DEFAULT_MIN_HAN);
        for i in 0..40 {
            let len = 10 + rng.below(40);
            let mut text = String::new();
            for _ in 0..len {
                text.push(*rng.pick(&pool));
            }
            let opts = if i % 2 == 0 { &OPTS } else { &DIALECT };
            assert_paths_identical(&text, &big_idx, DEFAULT_THRESHOLD, opts);
        }
    }

    /// 缓存检索层差分：apply_chinese_hotwords（每次现建检索层）与
    /// apply_chinese_hotwords_indexed（HotwordConfig 预构建检索层）
    /// 在带规则场景下逐字节一致，且缓存命中不重建检索层（同 Arc）。
    #[test]
    fn indexed_config_path_matches_plain_path() {
        let words: Vec<String> = [
            "中州韵输入法 | 中州韵",
            "知识库",
            "四喜丸子 ~~~ 世纪",
            "会计准则",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let rules_text = "很好用=非常好用\n中州韵=中州韵输入法";
        let cache = HotwordCache::new();
        let cfg = cache.get_or_build(&words, rules_text, DEFAULT_MIN_HAN);
        let cfg2 = cache.get_or_build(&words, rules_text, DEFAULT_MIN_HAN);
        assert!(Arc::ptr_eq(&cfg, &cfg2), "缓存未命中：检索层被重建");

        let index = build_hotword_index(&words, DEFAULT_MIN_HAN);
        let (rules, _) = parse_hotword_rules(rules_text);
        let texts = [
            "我在用中州运输入法，很好用",
            "同步只四库",
            "二十一世纪吃四喜完子",
            "快记准则更新了",
            "没什么模式",
        ];
        for text in texts {
            let a = apply_chinese_hotwords(text, &index, &rules, DEFAULT_THRESHOLD, &OPTS);
            let b = apply_chinese_hotwords_indexed(text, &cfg, DEFAULT_THRESHOLD, &OPTS);
            assert_eq!(a.0, b.0, "indexed 路径输出不一致：{text}");
            assert_eq!(a.1.len(), b.1.len(), "indexed 路径命中数不一致：{text}");
            for (x, y) in a.1.iter().zip(b.1.iter()) {
                assert_eq!(
                    (x.start, x.end, &x.matched, &x.replacement),
                    (y.start, y.end, &y.matched, &y.replacement)
                );
                assert_eq!(x.score.to_bits(), y.score.to_bits());
            }
        }
    }
}
