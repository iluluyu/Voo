//! 常用专名/AI 术语的**字面规范化**引擎。
//!
//! 只做字面（literal）替换，绝不参与拼音近音匹配：
//! - 数据来源：内置种子（`resources/lexicons/proper_nouns.json`，仅覆盖
//!   最高频的少量专名）+ 用户从开源词库源（A 组专名源，见
//!   `managers/lexicon_sources.rs`）**导入**的词条派生（小写形式 → 原始
//!   大小写规范写法）。
//! - 匹配语义：大小写不敏感 + ASCII 词边界（`\b` 语义：前后字符不得是
//!   ASCII 字母/数字/下划线；汉字不属于 ASCII 词字符，天然构成边界，
//!   匹配也绝不跨越汉字——变体本身不含汉字）。
//! - 幂等：命中文本与规范写法完全相同（大小写也相同）时不替换，已
//!   正确的写法永不二次改写。
//! - 优先级：**低于用户自定义热词规则**。用户规则（`模式=替换` 字面量）
//!   已覆盖的变体键在编译期直接剔除，用户为准。
//! - 隔离：不进入 `custom_words` / `imported_lexicons` / 拼音热词索引，
//!   对普通说话（中文日常语句、无 ASCII 专名的文本）零影响——引擎只
//!   在文本里出现变体时才动手。
//!
//! 本模块保持纯函数（不依赖 tauri/settings），设置感知的包装与单槽
//! 缓存见 [`crate::managers::transcription`] 的调用点。

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde::Deserialize;

/// 内置种子词库（编译期嵌入；`仅少量高频专名`，主数据走开源源导入）。
pub const PROPER_NOUN_SEED_JSON: &str = include_str!("../../resources/lexicons/proper_nouns.json");

/// 可派生字面规则的导入来源（A 组专名/术语源；见 `lexicon_source_catalog`）。
/// 值即 `imported_lexicons` 里的来源键。B/C/D 组（中文领域词、通用词频、
/// 分类术语）不派生：纯汉字词没有大小写规范化可言。
pub const PROPER_NOUN_SOURCE_IDS: &[&str] = &["zhwiki", "rime-ice-en", "rime-frost-en"];

/// 派生规则总数上限（防御：导入快照被人为塞大时保住编译/匹配开销）。
pub const MAX_DERIVED_RULES: usize = 20_000;

/// 种子词条。
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProperNounEntry {
    pub canonical: String,
    pub variants: Vec<String>,
}

/// 种子词库文件结构。
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProperNounLexicon {
    pub version: String,
    pub updated_at: String,
    pub entries: Vec<ProperNounEntry>,
}

/// 解析种子/用户覆盖的 JSON 词库。结构非法（缺字段、变体为空串）直接报错。
pub fn parse_proper_noun_lexicon(json: &str) -> Result<ProperNounLexicon, String> {
    let lexicon: ProperNounLexicon =
        serde_json::from_str(json).map_err(|e| format!("invalid proper-noun lexicon JSON: {e}"))?;
    if lexicon.entries.is_empty() {
        return Err("proper-noun lexicon has no entries".to_string());
    }
    for entry in &lexicon.entries {
        if entry.canonical.trim().is_empty() {
            return Err("proper-noun lexicon entry with empty canonical".to_string());
        }
        if entry.variants.iter().any(|v| v.trim().is_empty()) {
            return Err(format!(
                "proper-noun entry {:?} has an empty variant",
                entry.canonical
            ));
        }
    }
    Ok(lexicon)
}

/// 内置种子词库（编译期嵌入内容的解析结果；失败属于开发期错误）。
pub fn builtin_seed_lexicon() -> Result<ProperNounLexicon, String> {
    parse_proper_noun_lexicon(PROPER_NOUN_SEED_JSON)
}

/// 一条字面规则：`variant_key`（全小写比较键）→ `canonical`（规范写法）。
#[derive(Debug, Clone, PartialEq)]
pub struct ProperNounRule {
    pub variant_key: String,
    pub canonical: String,
}

/// 编译好的规则集合：规则本体 + 首字符（小写）分桶（桶内按变体字符数
/// 降序，同位置最长匹配优先，如 `GPT 4o` 优先于 `GPT 4`）。
#[derive(Debug, Default)]
pub struct ProperNounRules {
    rules: Vec<ProperNounRule>,
    buckets: HashMap<char, Vec<usize>>,
}

impl ProperNounRules {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// 规则列表（诊断/命令返回用；顺序为插入序）。
    pub fn rules(&self) -> &[ProperNounRule] {
        &self.rules
    }

    /// 由 (变体, 规范写法) 对构建规则集合。
    ///
    /// - 变体先 trim；与规范写法**逐字符完全相同**的变体丢弃（幂等底线：
    ///   规范写法自己不能成为自己的变体，否则命中即替换也恒等，但会
    ///   干扰「已正确写法不被动」的语义判断）。
    /// - 同一小写键重复时**先到先得**（种子在前、导入派生在后、来源按
    ///   `PROPER_NOUN_SOURCE_IDS` 声明序）。
    pub fn from_pairs<I, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, S)>,
        S: Into<String>,
    {
        let mut rules: Vec<ProperNounRule> = Vec::new();
        let mut seen: HashMap<String, ()> = HashMap::new();
        for (variant, canonical) in pairs {
            let variant: String = variant.into().trim().to_string();
            let canonical: String = canonical.into().trim().to_string();
            if variant.is_empty() || canonical.is_empty() || variant == canonical {
                continue;
            }
            let key = variant.to_lowercase();
            if seen.contains_key(&key) {
                continue;
            }
            seen.insert(key.clone(), ());
            rules.push(ProperNounRule {
                variant_key: key,
                canonical,
            });
        }
        Self::with_rules(rules)
    }

    fn with_rules(mut rules: Vec<ProperNounRule>) -> Self {
        let mut buckets: HashMap<char, Vec<usize>> = HashMap::new();
        for (index, rule) in rules.iter().enumerate() {
            if let Some(first) = rule.variant_key.chars().next() {
                buckets.entry(first).or_default().push(index);
            }
        }
        // 桶内按变体字符数降序（稳定），保证同位置最长优先。
        for bucket in buckets.values_mut() {
            bucket.sort_by(|&a, &b| {
                let la = rules[a].variant_key.chars().count();
                let lb = rules[b].variant_key.chars().count();
                lb.cmp(&la).then_with(|| a.cmp(&b))
            });
        }
        if rules.capacity() > rules.len() * 2 {
            rules.shrink_to_fit();
        }
        ProperNounRules { rules, buckets }
    }

    /// 对文本应用全部规则：大小写不敏感、ASCII 词边界、命中与规范写法
    /// 完全相同则保持原文（幂等）。变体不含汉字，匹配自然不跨越汉字。
    pub fn apply(&self, text: &str) -> String {
        if self.rules.is_empty() || text.is_empty() {
            return text.to_string();
        }
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::with_capacity(text.len());
        let mut index = 0usize;
        while index < chars.len() {
            let first = chars[index];
            let Some(bucket) = self
                .buckets
                .get(&first.to_lowercase().next().unwrap_or(first))
            else {
                out.push(first);
                index += 1;
                continue;
            };
            let mut replaced = false;
            for &rule_index in bucket {
                let rule = &self.rules[rule_index];
                let variant: Vec<char> = rule.variant_key.chars().collect();
                let end = index + variant.len();
                if end > chars.len() {
                    continue;
                }
                // 大小写不敏感逐字符比较（变体键已小写）。
                if !chars[index..end]
                    .iter()
                    .zip(variant.iter())
                    .all(|(text_char, variant_char)| {
                        text_char.to_lowercase().eq(variant_char.to_lowercase())
                    })
                {
                    continue;
                }
                // ASCII 词边界：前一字符与后一字符都不得是 ASCII 字母/
                // 数字/下划线（`\b` 语义）。汉字不属于 ASCII 词字符 →
                // 「deepseek的」两侧都是边界，可替换；「deepseekers」
                // 右侧仍是 ASCII 词字符 → 不替换。
                let prev_is_word = index > 0 && is_ascii_word_char(chars[index - 1]);
                let next_is_word = end < chars.len() && is_ascii_word_char(chars[end]);
                if prev_is_word || next_is_word {
                    continue;
                }
                let matched: String = chars[index..end].iter().collect();
                if matched == rule.canonical {
                    // 已是规范写法：原样保留（幂等），继续尝试更短规则
                    // 无意义（更短的是不同变体，但同位置已消费）。
                    out.push_str(&matched);
                    index = end;
                    replaced = true;
                    break;
                }
                out.push_str(&rule.canonical);
                index = end;
                replaced = true;
                break;
            }
            if !replaced {
                out.push(first);
                index += 1;
            }
        }
        out
    }
}

/// ASCII 词字符（`\b` 语义）：字母、数字、下划线。汉字/标点/空格都不是。
fn is_ascii_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// 从一个导入词条派生字面规则键：纯拉丁（无汉字）、含 ASCII 字母、
/// 2–32 字符、内部无制表符/控制字符，且小写形式与原词不同（否则没有
/// 规范化价值）。返回 `Some(小写键)`。
pub fn derived_rule_key(word: &str) -> Option<String> {
    let word = word.trim();
    let char_count = word.chars().count();
    if !(2..=32).contains(&char_count) {
        return None;
    }
    let has_han = word
        .chars()
        .any(crate::audio_toolkit::chinese_hotwords::is_han);
    if has_han {
        return None;
    }
    if !word.chars().any(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    if word.chars().any(|c| c.is_control() || c == '\t') {
        return None;
    }
    let lower = word.to_lowercase();
    (lower != word).then_some(lower)
}

/// 从导入的专名源词条批量派生 (键, 规范写法) 对（保序、有上限）。
pub fn derive_pairs_from_words<'a, I>(words: I) -> Vec<(String, String)>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut pairs = Vec::new();
    for word in words {
        if pairs.len() >= MAX_DERIVED_RULES {
            break;
        }
        if let Some(key) = derived_rule_key(word) {
            pairs.push((key, word.trim().to_string()));
        }
    }
    pairs
}

/// 从用户热词规则文本（每行 `模式=替换` 或 `模式=>替换`）提取**字面量**
/// 规则的模式键（小写）。`=>` 正则规则键含正则语法，不参与键碰撞剔除
/// （正则行为由用户规则全量负责，专名规则在其后执行）。
pub fn user_literal_rule_keys(hotword_rules: &str) -> Vec<String> {
    let mut keys = Vec::new();
    for raw_line in hotword_rules.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(pos) = line.find("=>") else {
            if let Some(pos) = line.find('=') {
                let pattern = line[..pos].trim();
                if !pattern.is_empty() {
                    keys.push(pattern.to_lowercase());
                }
            }
            continue;
        };
        // `=>` 正则：字面量键不提取（见 doc）。
        let _ = pos;
    }
    keys
}

/// 字面规则编译输入（设置感知的调用点组装；本模块保持纯函数）。
pub struct ProperNounConfig<'a> {
    /// 总开关（`proper_noun_lexicon_enabled`）。
    pub enabled: bool,
    /// 用户热词规则文本（优先级高于内置/派生规则）。
    pub user_rules: &'a str,
    /// 参与派生的导入来源：(来源键, 词条快照)。
    pub sources: Vec<(&'static str, &'a [String])>,
}

impl ProperNounConfig<'_> {
    /// 缓存键：影响规则集的全部输入的精确拼接（与既有热词缓存同风格：
    /// 单槽、精确比较，锁内只比较/克隆）。
    pub fn cache_key(&self) -> String {
        let mut key = String::new();
        key.push_str(if self.enabled { "1" } else { "0" });
        key.push('\u{0}');
        key.push_str(self.user_rules);
        key.push('\u{0}');
        for (id, words) in &self.sources {
            key.push_str(id);
            key.push('\u{1f}');
            for word in *words {
                key.push_str(word);
                key.push('\u{1e}');
            }
            key.push('\u{0}');
        }
        key
    }
}

/// 按配置编译规则集合（纯函数；调用点自行缓存）。
pub fn compile_proper_noun_rules(config: &ProperNounConfig<'_>) -> ProperNounRules {
    if !config.enabled {
        return ProperNounRules::empty();
    }
    // 用户字面量规则键优先：同键的内置/派生规则一律剔除。
    let user_keys = user_literal_rule_keys(config.user_rules);

    let mut pairs: Vec<(String, String)> = Vec::new();
    // 1) 内置种子（最高频专名的兜底；解析失败按无种子处理，不阻塞转写）。
    if let Ok(seed) = builtin_seed_lexicon() {
        for entry in &seed.entries {
            for variant in &entry.variants {
                pairs.push((variant.clone(), entry.canonical.clone()));
            }
        }
    }
    // 2) 导入源派生（声明序；同键先到先得，种子已覆盖的键胜出）。
    for (id, words) in &config.sources {
        if !PROPER_NOUN_SOURCE_IDS.contains(id) {
            continue;
        }
        for word in *words {
            if let Some(key) = derived_rule_key(word) {
                pairs.push((key, word.trim().to_string()));
            }
        }
    }

    let rules = ProperNounRules::from_pairs(
        pairs
            .into_iter()
            .filter(|(variant, _)| !user_keys.contains(&variant.to_lowercase()))
            .collect::<Vec<_>>(),
    );
    rules
}

// ---------------------------------------------------------------------------
// 有界单槽缓存（同 chinese_hotwords 的共享缓存风格）
// ---------------------------------------------------------------------------

type ProperNounCacheSlot = Mutex<Option<(String, Arc<ProperNounRules>)>>;

fn shared_proper_noun_cache() -> &'static ProperNounCacheSlot {
    static CACHE: OnceLock<ProperNounCacheSlot> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// 取（或编译后缓存）当前配置的字面规则集合。锁内只做键比较与 `Arc`
/// 克隆，编译在锁外；并发未命中时双方都编译、后写者胜（与热词缓存同
/// 语义，结果确定）。
pub fn effective_proper_noun_rules(config: &ProperNounConfig<'_>) -> Arc<ProperNounRules> {
    let cache_key = config.cache_key();
    let cache = shared_proper_noun_cache();
    if let Ok(slot) = cache.lock() {
        if let Some((key, rules)) = slot.as_ref() {
            if *key == cache_key {
                return Arc::clone(rules);
            }
        }
    }
    let compiled = Arc::new(compile_proper_noun_rules(config));
    if let Ok(mut slot) = cache.lock() {
        *slot = Some((cache_key, Arc::clone(&compiled)));
    }
    compiled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed() -> ProperNounLexicon {
        builtin_seed_lexicon().expect("builtin seed lexicon must parse")
    }

    fn config(user_rules: &str) -> ProperNounConfig<'_> {
        ProperNounConfig {
            enabled: true,
            user_rules,
            sources: Vec::new(),
        }
    }

    #[test]
    fn seed_lexicon_is_small_and_wellformed() {
        let lexicon = seed();
        // 种子仅用于兜底与单测：必须保持 ≤20 条。
        assert!(!lexicon.entries.is_empty());
        assert!(lexicon.entries.len() <= 20, "seed must stay <= 20 entries");
        assert!(!lexicon.version.is_empty());
        for entry in &lexicon.entries {
            assert!(!entry.variants.is_empty());
            // 变体不得是普通词：全部必须含 ASCII 字母，且与规范写法不同。
            assert!(entry
                .variants
                .iter()
                .all(|v| v.chars().any(|c| c.is_ascii_alphabetic()) && v != &entry.canonical));
            // 种子只收专名：不得含汉字。
            assert!(!entry
                .canonical
                .chars()
                .any(crate::audio_toolkit::chinese_hotwords::is_han));
        }
        // 关键诉求样例必须在种子内。
        let canonicals: Vec<&str> = lexicon
            .entries
            .iter()
            .map(|e| e.canonical.as_str())
            .collect();
        for expected in ["DeepSeek", "Claude", "ChatGPT"] {
            assert!(canonicals.contains(&expected), "seed missing {expected}");
        }
    }

    #[test]
    fn variants_replaced_with_canonical() {
        let rules = compile_proper_noun_rules(&config(""));
        assert_eq!(rules.apply("我用 deepseek 写代码"), "我用 DeepSeek 写代码");
        assert_eq!(rules.apply("打开 vscode"), "打开 VS Code");
        assert_eq!(rules.apply("问问 chat gpt"), "问问 ChatGPT");
        // 大小写不敏感：全大写也修正。
        assert_eq!(rules.apply("DEEPSEEK 很强"), "DeepSeek 很强");
        assert_eq!(rules.apply("用 GITHUB 提交"), "用 GitHub 提交");
    }

    #[test]
    fn already_correct_spelling_is_untouched() {
        let rules = compile_proper_noun_rules(&config(""));
        assert_eq!(rules.apply("DeepSeek 很强"), "DeepSeek 很强");
        assert_eq!(rules.apply("VS Code 和 GitHub"), "VS Code 和 GitHub");
        // 幂等：二次应用不再变化。
        let once = rules.apply("我用 deepseek");
        assert_eq!(rules.apply(&once), once);
    }

    #[test]
    fn word_boundary_protects_longer_words() {
        let rules = compile_proper_noun_rules(&config(""));
        // 右侧是 ASCII 词字符：不替换。
        assert_eq!(rules.apply("deepseekers 不错"), "deepseekers 不错");
        // 左侧是 ASCII 词字符：不替换。
        assert_eq!(rules.apply("mydeepseek 不错"), "mydeepseek 不错");
        assert_eq!(rules.apply("x_linux_y"), "x_linux_y");
        // 汉字与标点都是边界：可替换（也不跨越汉字）。
        assert_eq!(rules.apply("用deepseek的api"), "用DeepSeek的api");
        assert_eq!(rules.apply("(vscode)"), "(VS Code)");
        assert_eq!(rules.apply("linux,macos"), "Linux,macOS");
    }

    #[test]
    fn user_rules_take_precedence() {
        // 用户显式写了同名规则：该键的内置规则被剔除，交给用户规则链。
        let rules = compile_proper_noun_rules(&config("deepseek=深度求索"));
        assert_eq!(rules.apply("deepseek 很强"), "deepseek 很强");
        assert!(rules.rules().iter().all(|r| r.variant_key != "deepseek"));
        // 未冲突的键照常。
        assert_eq!(rules.apply("vscode"), "VS Code");
        // 正则（=>）规则不参与字面键剔除，但也不受影响。
        let rules_regex = compile_proper_noun_rules(&config("deep?seek=>X"));
        assert_eq!(rules_regex.apply("deepseek"), "DeepSeek");
    }

    #[test]
    fn disabled_switch_disables_everything() {
        let rules = compile_proper_noun_rules(&ProperNounConfig {
            enabled: false,
            user_rules: "",
            sources: Vec::new(),
        });
        assert!(rules.is_empty());
        assert_eq!(rules.apply("deepseek vscode"), "deepseek vscode");
    }

    #[test]
    fn imported_sources_derive_case_rules() {
        let words = vec![
            "PyTorch".to_string(),
            "openai".to_string(),   // 全小写：无可规范化，不派生
            "深度学习".to_string(), // 汉字：不派生
            "ChatGPT".to_string(),
            "A".to_string(), // 太短：不派生
        ];
        let cfg = ProperNounConfig {
            enabled: true,
            user_rules: "",
            sources: vec![("rime-ice-en", &words[..])],
        };
        let rules = compile_proper_noun_rules(&cfg);
        assert_eq!(rules.apply("pytorch 好用"), "PyTorch 好用");
        assert_eq!(rules.apply("CHATGPT 好用"), "ChatGPT 好用");
        // 全小写词条无规则：保持原样。
        assert_eq!(rules.apply("openai 官网"), "openai 官网");
    }

    #[test]
    fn ordinary_words_never_match() {
        let rules = compile_proper_noun_rules(&config(""));
        for text in [
            "the quick brown fox",
            "we are here today",
            "我们今天去公司开会",
            "今天天气不错",
            "hello world",
        ] {
            assert_eq!(rules.apply(text), text, "ordinary text must stay untouched");
        }
        // 种子里不存在普通词变体（抽样断言）。
        let keys: Vec<&str> = rules
            .rules()
            .iter()
            .map(|r| r.variant_key.as_str())
            .collect();
        for ordinary in ["the", "we", "today", "company", "hello", "world"] {
            assert!(
                !keys.contains(&ordinary),
                "{ordinary} must not be a rule key"
            );
        }
    }

    #[test]
    fn longest_variant_wins_at_same_position() {
        // 手工构造：GPT 4 与 GPT 4o 同位置，取更长。
        let rules = ProperNounRules::from_pairs([
            ("GPT 4", "GPT-4"),
            ("GPT 4o", "GPT-4o"),
            ("gpt 4", "GPT-4"),
        ]);
        assert_eq!(rules.apply("GPT 4o 发布"), "GPT-4o 发布");
        assert_eq!(rules.apply("GPT 4 发布"), "GPT-4 发布");
    }

    #[test]
    fn duplicate_keys_first_wins() {
        let rules = ProperNounRules::from_pairs([
            ("deepseek", "DeepSeek"), // 先到
            ("DeepSeek", "别的"),     // 同小写键：丢弃
        ]);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules.apply("deepseek"), "DeepSeek");
    }

    #[test]
    fn user_literal_keys_parser() {
        let keys = user_literal_rule_keys("# 注释\nteht=>the\nfoo=bar\n  spaced = x \n无等号行");
        assert_eq!(keys, vec!["foo".to_string(), "spaced".to_string()]);
    }

    #[test]
    fn cache_roundtrip_reuses_compiled_rules() {
        let cfg = config("");
        let first = effective_proper_noun_rules(&cfg);
        let second = effective_proper_noun_rules(&cfg);
        // 单槽缓存命中：同一配置两次取回的是同一份 Arc（编译只发生一次）。
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.apply("deepseek"), "DeepSeek");
    }
}
