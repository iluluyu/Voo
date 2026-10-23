//! 统一词库解析器（lexicon resolver）——词库**读路径**的单一真相源。
//!
//! 一次查询回答一个问题：「这个词现在到底怎么算？」——规范写法
//! （canonical）、生效档位（effective_mode）、胜出规则、贡献来源与
//! 被压制来源，全部由同一套优先级推出，**不依赖导入顺序或词频**。
//!
//! # 权威阶梯（高 → 低）
//!
//! 1. **单条 override**：`lexicon_term_modes`，key 本身就是规范写法
//!    （primary；别名继承所在行的档位）；
//! 2. **用户自有词**：`custom_words`（手工词 + 用户确认的候选，确认后
//!    即并入该列表，解析器不再区分二者）；
//! 3. **公开词库来源**：`imported_lexicons` 的各来源。
//!
//! # 规范写法规则
//!
//! - 权威最高的非空层定写法：上层有定论时，下层不同写法进
//!   `suppressed_sources`；
//! - 同层写法一致 → 合并来源（`contributing_sources` 全收，跨层一致
//!   也并入）；
//! - 同层写法冲突 → **保持歧义**：`canonical = None`、`effective_mode`
//!   降为 Candidate，explanation 列出冲突来源与写法——绝不按导入顺序/
//!   词频决胜；
//! - 歧义词要参与纠正必须先选定规范写法：加入用户词，或以 override
//!   key 直接声明该词为规范写法。
//!
//! # 档位（权限）与写法解耦，单独计算
//!
//! - 单条 override（按 canonical key，别名继承）> 来源默认；
//! - 用户词默认 ActiveFuzzy；公开来源默认 ActiveExact，可被
//!   `lexicon_source_modes` 整源改档（`blocked` 即「关掉该公开源」，
//!   对应四态里的 Off）；
//! - 关掉某公开源只压制该来源，**不得**让用户自有词失效；
//! - 仅公开来源贡献、档位不一致时取最保守档（任一 candidate →
//!   Candidate；active_exact / active_fuzzy 并存 → ActiveExact）。
//!
//! 不新增任何存储：全部从 [`AppSettings`] 现有字段装配，纯函数查询。

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use serde::Serialize;
use specta::Type;
use tauri::AppHandle;

use crate::audio_toolkit::chinese_hotwords::{parse_hotword_line, TermMode};
use crate::settings::{self, AppSettings};

/// 用户自有词的来源 id：`custom_words`（手工词与用户确认的候选同层）。
pub const USER_SOURCE_ID: &str = "custom_words";

/// 档位的存储短名（与 `TermMode::from_str_name` 的 snake_case 名一致），
/// 仅用于 explanation 文本。
fn mode_name(mode: TermMode) -> &'static str {
    match mode {
        TermMode::Candidate => "candidate",
        TermMode::ActiveExact => "active_exact",
        TermMode::ActiveFuzzy => "active_fuzzy",
        TermMode::Blocked => "blocked",
    }
}

/// 一条词条查询的完整裁决。
#[derive(Debug, Clone, Serialize, Type)]
pub struct LexiconResolution {
    /// 查询词（trim 后原样返回）。
    pub term: String,
    /// 规范写法（替换目标 primary）；`None` = 无定论（未收录，或写法
    /// 冲突保持歧义）。
    pub canonical: Option<String>,
    /// 最终生效档位（歧义降 Candidate、全关/未收录为 Blocked）。
    pub effective_mode: TermMode,
    /// 胜出规则。
    pub winning_rule: LexiconWinningRule,
    /// 贡献来源 id（`custom_words` 或 `imported_lexicons` 的来源键；
    /// custom_words 恒在最前，其余按来源名排序）。
    pub contributing_sources: Vec<String>,
    /// 被压制来源 id（更高权威胜出 / 写法不一致 / 来源已关）。约定：
    /// contributing ∪ suppressed = 对该词有主张的全部来源。
    pub suppressed_sources: Vec<String>,
    /// 人读解释（含来源 id），说明裁决路径。
    pub explanation: String,
}

/// 裁决的胜出规则（权威阶梯的落点）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LexiconWinningRule {
    /// 单条 override（`lexicon_term_modes`，key 即规范写法）。
    TermOverride,
    /// 用户自有词（`custom_words`：手工词 + 用户确认的候选）。
    UserWord,
    /// 公开词库来源（`imported_lexicons`；写法一致时多来源合并）。
    PublicSource,
    /// 写法冲突保持歧义（无胜者，降 Candidate）。
    Ambiguous,
    /// 未收录（无任何来源主张）。
    Unknown,
}

/// 一个来源对某个词的主张：该词（primary 或别名）应替换为 `canonical`。
#[derive(Debug, Clone)]
struct Claim {
    /// 来源 id（`USER_SOURCE_ID` 或 `imported_lexicons` 的键）。
    source: String,
    /// 是否用户自有词（不受来源级模式约束）。
    user_owned: bool,
    /// 该来源主张的规范写法（所在行的 primary）。
    canonical: String,
    /// 来源级档位（用户词为 `None`：无来源级约束）。
    source_mode: Option<TermMode>,
}

impl Claim {
    /// 来源是否被整源关闭（来源级 Blocked，即四态里的 Off）。
    fn is_off(&self) -> bool {
        self.source_mode == Some(TermMode::Blocked)
    }
}

/// 从 [`AppSettings`] 装配的只读解析器；一次装配可批量查询。
#[derive(Debug, Default)]
pub struct LexiconResolver {
    /// `lexicon_term_modes` 快照（key = 规范写法 primary）。
    overrides: HashMap<String, TermMode>,
    /// term（primary 或别名）→ 各来源的主张列表。
    claims: HashMap<String, Vec<Claim>>,
}

impl LexiconResolver {
    /// 从设置装配：手工词在前、导入来源按名称稳定排序，保证同内容
    /// 产出确定性的主张顺序（冲突 explanation 的列举顺序随之确定）。
    pub fn from_settings(settings: &AppSettings) -> Self {
        let overrides = settings.lexicon_term_modes.clone();
        let mut claims: HashMap<String, Vec<Claim>> = HashMap::new();
        let mut seen: HashSet<(String, String, String)> = HashSet::new();

        for line in &settings.custom_words {
            let parsed = parse_hotword_line(line);
            if parsed.primary.is_empty() {
                continue;
            }
            let mut row_terms = vec![parsed.primary.clone()];
            row_terms.extend(parsed.aliases.iter().cloned());
            for term in row_terms {
                push_claim(
                    &mut claims,
                    &mut seen,
                    &term,
                    Claim {
                        source: USER_SOURCE_ID.to_string(),
                        user_owned: true,
                        canonical: parsed.primary.clone(),
                        source_mode: None,
                    },
                );
            }
        }

        // 来源名排序保证确定性（与 effective_custom_words 一致）。
        let mut sources: Vec<&String> = settings.imported_lexicons.keys().collect();
        sources.sort_unstable();
        for source in sources {
            let source_mode = settings.lexicon_source_modes.get(source).copied();
            for line in &settings.imported_lexicons[source] {
                let parsed = parse_hotword_line(line);
                if parsed.primary.is_empty() {
                    continue;
                }
                let mut row_terms = vec![parsed.primary.clone()];
                row_terms.extend(parsed.aliases.iter().cloned());
                for term in row_terms {
                    push_claim(
                        &mut claims,
                        &mut seen,
                        &term,
                        Claim {
                            source: source.clone(),
                            user_owned: false,
                            canonical: parsed.primary.clone(),
                            source_mode,
                        },
                    );
                }
            }
        }

        Self { overrides, claims }
    }

    /// 解析单个词的完整裁决（纯查询，不写任何状态）。
    pub fn resolve_term(&self, raw: &str) -> LexiconResolution {
        let term = raw.trim().to_string();
        if term.is_empty() {
            return LexiconResolution {
                explanation: "空词：不建条目。".to_string(),
                term,
                canonical: None,
                effective_mode: TermMode::Blocked,
                winning_rule: LexiconWinningRule::Unknown,
                contributing_sources: Vec::new(),
                suppressed_sources: Vec::new(),
            };
        }

        let claims: &[Claim] = self.claims.get(&term).map(Vec::as_slice).unwrap_or(&[]);
        let user_claims: Vec<&Claim> = claims.iter().filter(|c| c.user_owned).collect();
        let public_claims: Vec<&Claim> = claims.iter().filter(|c| !c.user_owned).collect();
        let term_override = self.overrides.get(term.as_str()).copied();

        // ---- 权威第 1 层：单条 override（key 本身即规范写法）----
        if let Some(mode) = term_override {
            return self.resolve_by_term_override(&term, mode, claims);
        }

        // ---- 未收录 ----
        if claims.is_empty() {
            return LexiconResolution {
                explanation: format!(
                    "未收录：custom_words 与 imported_lexicons 均无「{term}」，无生效条目。"
                ),
                term,
                canonical: None,
                effective_mode: TermMode::Blocked,
                winning_rule: LexiconWinningRule::Unknown,
                contributing_sources: Vec::new(),
                suppressed_sources: Vec::new(),
            };
        }

        // ---- 权威第 2/3 层：用户自有词 > 公开来源 ----
        let (tier_claims, tier_is_user) = if !user_claims.is_empty() {
            (&user_claims, true)
        } else {
            (&public_claims, false)
        };

        // 同层写法集合（保持装配顺序，去重）。
        let mut spellings: Vec<&str> = Vec::new();
        for claim in tier_claims {
            if !spellings.contains(&claim.canonical.as_str()) {
                spellings.push(claim.canonical.as_str());
            }
        }

        if spellings.len() > 1 {
            // 同层写法冲突：保持歧义，不决胜。
            let conflicts = tier_claims
                .iter()
                .map(|c| format!("{}→「{}」", c.source, c.canonical))
                .collect::<Vec<_>>()
                .join("、");
            let suppressed_sources =
                order_sources(claims.iter().map(|c| c.source.clone()).collect());
            let tier_label = if tier_is_user {
                "用户自有词内部"
            } else {
                "公开来源"
            };
            let mut explanation = format!(
                "{tier_label}写法冲突，保持歧义：{conflicts}；不按导入顺序/词频决胜，\
                 已降为 candidate，先选定规范写法（加入 custom_words 或设单条档位）再参与纠正"
            );
            if tier_is_user && !public_claims.is_empty() {
                let ids = public_claims.iter().map(|c| c.source.clone()).collect();
                explanation.push_str(&format!(
                    "；公开来源另有写法，同样搁置：{}",
                    join_ids(&order_sources(ids))
                ));
            }
            return LexiconResolution {
                explanation,
                term,
                canonical: None,
                effective_mode: TermMode::Candidate,
                winning_rule: LexiconWinningRule::Ambiguous,
                contributing_sources: Vec::new(),
                suppressed_sources,
            };
        }

        // 单一写法：定 canonical，合并一致来源（含更低权威层的一致写法）。
        let canonical = spellings[0].to_string();
        let mut contributing: Vec<String> = Vec::new();
        let mut has_user_contrib = false;
        let mut suppressed_claims: Vec<&Claim> = Vec::new();
        for claim in claims {
            if claim.canonical == canonical && !claim.is_off() {
                if !contributing.contains(&claim.source) {
                    contributing.push(claim.source.clone());
                }
                if claim.user_owned {
                    has_user_contrib = true;
                }
            } else {
                suppressed_claims.push(claim);
            }
        }
        let contributing = order_sources(contributing);
        let suppressed_sources =
            order_sources(suppressed_claims.iter().map(|c| c.source.clone()).collect());

        // ---- 档位：override（按 canonical，别名继承）> 用户词默认 > 来源默认 ----
        let canonical_override = self.overrides.get(canonical.as_str()).copied();
        let any_candidate_source = claims
            .iter()
            .any(|c| c.canonical == canonical && c.source_mode == Some(TermMode::Candidate));
        let any_explicit_source_mode = claims
            .iter()
            .any(|c| c.canonical == canonical && c.source_mode.is_some());

        let (effective_mode, winning_rule, mode_note) = if let Some(m) = canonical_override {
            (
                m,
                LexiconWinningRule::TermOverride,
                format!(
                    "；档位 lexicon_term_modes[{canonical}]={} 覆盖来源默认",
                    mode_name(m)
                ),
            )
        } else if has_user_contrib {
            // 用户词在场即用户层默认；公开来源档位不影响用户自有词。
            (
                TermMode::ActiveFuzzy,
                LexiconWinningRule::UserWord,
                "；用户词默认 active_fuzzy（公开来源档位不影响用户自有词）".to_string(),
            )
        } else if contributing.is_empty() {
            // 写法已知，但贡献来源全部整源关闭。
            (
                TermMode::Blocked,
                LexiconWinningRule::PublicSource,
                "；贡献来源全部整源关闭，按 blocked 处理".to_string(),
            )
        } else if any_candidate_source {
            (
                TermMode::Candidate,
                LexiconWinningRule::PublicSource,
                "；含 candidate 档来源，合并取最保守 candidate".to_string(),
            )
        } else if any_explicit_source_mode {
            (
                TermMode::ActiveExact,
                LexiconWinningRule::PublicSource,
                "；来源档位不一致时取最保守生效档 active_exact".to_string(),
            )
        } else {
            (
                TermMode::ActiveExact,
                LexiconWinningRule::PublicSource,
                "；来源默认 active_exact".to_string(),
            )
        };

        let mut explanation = if tier_is_user {
            format!("用户自有词（custom_words）胜出，规范写法「{canonical}」")
        } else {
            format!("公开来源胜出，规范写法「{canonical}」")
        };
        if !contributing.is_empty() {
            explanation.push_str(&format!(
                "；写法一致，合并来源：{}",
                join_ids(&contributing)
            ));
        }
        for claim in &suppressed_claims {
            if claim.is_off() {
                explanation.push_str(&format!(
                    "；{} 来源已关（blocked），其「{}」不生效",
                    claim.source, claim.canonical
                ));
            } else {
                explanation.push_str(&format!(
                    "；{} 写法「{}」被更高权威压制",
                    claim.source, claim.canonical
                ));
            }
        }
        explanation.push_str(&mode_note);

        LexiconResolution {
            term,
            canonical: Some(canonical),
            effective_mode,
            winning_rule,
            contributing_sources: contributing,
            suppressed_sources,
            explanation,
        }
    }

    /// 权威第 1 层：override key 即规范写法，档位即 override 档位。
    fn resolve_by_term_override(
        &self,
        term: &str,
        mode: TermMode,
        claims: &[Claim],
    ) -> LexiconResolution {
        let canonical = term.to_string();
        let mut contributing: Vec<String> = Vec::new();
        let mut suppressed: Vec<&Claim> = Vec::new();
        for claim in claims {
            if claim.canonical == canonical && !claim.is_off() {
                if !contributing.contains(&claim.source) {
                    contributing.push(claim.source.clone());
                }
            } else {
                suppressed.push(claim);
            }
        }
        let contributing = order_sources(contributing);
        let suppressed_sources =
            order_sources(suppressed.iter().map(|c| c.source.clone()).collect());

        let mut explanation = format!(
            "单条档位 override 胜出：lexicon_term_modes[{term}]={}，「{term}」即规范写法",
            mode_name(mode)
        );
        if mode == TermMode::Blocked {
            if !suppressed_sources.is_empty() {
                explanation.push_str(&format!(
                    "；整词屏蔽，{} 的相关词条全部不生效",
                    join_ids(&suppressed_sources)
                ));
            }
        } else {
            if !contributing.is_empty() {
                explanation.push_str(&format!("；写法一致来源：{}", join_ids(&contributing)));
            }
            for claim in &suppressed {
                if claim.is_off() {
                    explanation.push_str(&format!(
                        "；{} 来源已关（blocked），其「{}」不生效",
                        claim.source, claim.canonical
                    ));
                } else {
                    explanation.push_str(&format!(
                        "；{} 写法「{}」被 override 压制",
                        claim.source, claim.canonical
                    ));
                }
            }
            if claims.is_empty() {
                explanation
                    .push_str("；当前生效词表未收录该词（override 保留在持久层，词回来即生效）");
            }
        }

        LexiconResolution {
            term: term.to_string(),
            canonical: Some(canonical),
            effective_mode: mode,
            winning_rule: LexiconWinningRule::TermOverride,
            contributing_sources: contributing,
            suppressed_sources,
            explanation,
        }
    }
}

/// 登记一条主张：term / 来源 / 写法 三元组去重（同一来源里两行都主张
/// 同一个词的不同写法时两条都保留——那是真实的行级冲突）。
fn push_claim(
    claims: &mut HashMap<String, Vec<Claim>>,
    seen: &mut HashSet<(String, String, String)>,
    term: &str,
    claim: Claim,
) {
    let term = term.trim();
    if term.is_empty() {
        return;
    }
    if !seen.insert((
        term.to_string(),
        claim.source.clone(),
        claim.canonical.clone(),
    )) {
        return;
    }
    claims.entry(term.to_string()).or_default().push(claim);
}

/// 来源 id 列表排序：`custom_words` 恒在最前，其余按名称排序后去重。
fn order_sources(mut ids: Vec<String>) -> Vec<String> {
    ids.sort_by(|a, b| match (a == USER_SOURCE_ID, b == USER_SOURCE_ID) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => a.cmp(b),
    });
    ids.dedup();
    ids
}

/// 中文顿号连接（用于 explanation 与来源列举）。
fn join_ids(ids: &[String]) -> String {
    ids.join("、")
}

// ---------------------------------------------------------------------------
// 只读命令（前端批量查询）
// ---------------------------------------------------------------------------

/// 只读：统一词库解析器批量查询词条裁决（规范写法/档位/来源归属）。
///
/// 纯查询，不写任何状态；`terms` 逐条独立解析，顺序原样返回。
#[tauri::command]
#[specta::specta]
pub fn lookup_lexicon_terms(
    app: AppHandle,
    terms: Vec<String>,
) -> Result<Vec<LexiconResolution>, String> {
    let settings = settings::get_settings(&app);
    let resolver = LexiconResolver::from_settings(&settings);
    Ok(terms.iter().map(|t| resolver.resolve_term(t)).collect())
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn settings_with(
        custom: &[&str],
        imported: &[(&str, &[&str])],
        term_modes: &[(&str, TermMode)],
        source_modes: &[(&str, TermMode)],
    ) -> AppSettings {
        let mut settings = AppSettings::default();
        settings.custom_words = custom.iter().map(|s| s.to_string()).collect();
        for (source, words) in imported {
            settings.imported_lexicons.insert(
                source.to_string(),
                words.iter().map(|s| s.to_string()).collect(),
            );
        }
        for (term, mode) in term_modes {
            settings.lexicon_term_modes.insert(term.to_string(), *mode);
        }
        for (source, mode) in source_modes {
            settings
                .lexicon_source_modes
                .insert(source.to_string(), *mode);
        }
        settings
    }

    fn resolve(settings: &AppSettings, term: &str) -> LexiconResolution {
        LexiconResolver::from_settings(settings).resolve_term(term)
    }

    /// 1) 用户自有词胜公开来源：用户写法为 canonical，公开来源被压制。
    #[test]
    fn user_word_beats_public_source() {
        let settings = settings_with(
            &["中州韵输入法|中州运输入法"],
            &[("THUOCL_IT", &["中州云输入法|中州运输入法"])],
            &[],
            &[],
        );
        let r = resolve(&settings, "中州运输入法");
        assert_eq!(r.canonical.as_deref(), Some("中州韵输入法"));
        assert_eq!(r.winning_rule, LexiconWinningRule::UserWord);
        assert_eq!(r.effective_mode, TermMode::ActiveFuzzy);
        assert_eq!(r.contributing_sources, vec!["custom_words"]);
        assert_eq!(r.suppressed_sources, vec!["THUOCL_IT"]);
    }

    /// 2) 公开来源写法冲突 → 保持歧义，降为 Candidate，不按顺序决胜。
    #[test]
    fn public_source_conflict_downgrades_to_candidate() {
        let settings = settings_with(
            &[],
            &[
                ("THUOCL_IT", &["甲写法|碰撞词"]),
                ("THUOCL_Law", &["乙写法|碰撞词"]),
            ],
            &[],
            &[],
        );
        let r = resolve(&settings, "碰撞词");
        assert_eq!(r.canonical, None);
        assert_eq!(r.effective_mode, TermMode::Candidate);
        assert_eq!(r.winning_rule, LexiconWinningRule::Ambiguous);
        assert!(r.contributing_sources.is_empty());
        assert_eq!(r.suppressed_sources, vec!["THUOCL_IT", "THUOCL_Law"]);
        assert!(r.explanation.contains("甲写法") && r.explanation.contains("乙写法"));
        assert!(r.explanation.contains("先选定规范写法"));
    }

    /// 3) 公开来源写法相同 → 合并来源，无压制。
    #[test]
    fn same_spelling_merges_sources() {
        let settings = settings_with(
            &[],
            &[("THUOCL_Law", &["规范词"]), ("THUOCL_IT", &["规范词"])],
            &[],
            &[],
        );
        let r = resolve(&settings, "规范词");
        assert_eq!(r.canonical.as_deref(), Some("规范词"));
        assert_eq!(r.winning_rule, LexiconWinningRule::PublicSource);
        assert_eq!(
            r.contributing_sources,
            vec!["THUOCL_IT".to_string(), "THUOCL_Law".to_string()]
        );
        assert!(r.suppressed_sources.is_empty());
        assert_eq!(r.effective_mode, TermMode::ActiveExact);
    }

    /// 4) 关掉某公开源不得让用户自有词失效；只剩关掉来源的词 → Blocked。
    #[test]
    fn disabled_source_does_not_disable_user_words() {
        let settings = settings_with(
            &["我的词"],
            &[("THUOCL_IT", &["我的词", "关掉的词"])],
            &[],
            &[("THUOCL_IT", TermMode::Blocked)],
        );
        let r = resolve(&settings, "我的词");
        assert_eq!(r.canonical.as_deref(), Some("我的词"));
        assert_eq!(r.winning_rule, LexiconWinningRule::UserWord);
        assert_eq!(r.effective_mode, TermMode::ActiveFuzzy);
        assert_eq!(r.contributing_sources, vec!["custom_words"]);
        assert_eq!(r.suppressed_sources, vec!["THUOCL_IT"]);

        let off = resolve(&settings, "关掉的词");
        assert_eq!(off.canonical.as_deref(), Some("关掉的词"));
        assert_eq!(off.effective_mode, TermMode::Blocked);
        assert!(off.contributing_sources.is_empty());
        assert_eq!(off.suppressed_sources, vec!["THUOCL_IT"]);
    }

    /// 5) 单条 override 覆盖来源级模式（来源 Candidate 被 override 改档）。
    #[test]
    fn term_override_overrides_source_mode() {
        let settings = settings_with(
            &[],
            &[("THUOCL_IT", &["领域术语"])],
            &[("领域术语", TermMode::ActiveFuzzy)],
            &[("THUOCL_IT", TermMode::Candidate)],
        );
        let r = resolve(&settings, "领域术语");
        assert_eq!(r.canonical.as_deref(), Some("领域术语"));
        assert_eq!(r.winning_rule, LexiconWinningRule::TermOverride);
        assert_eq!(r.effective_mode, TermMode::ActiveFuzzy);
        assert_eq!(r.contributing_sources, vec!["THUOCL_IT"]);
    }

    /// 6) 歧义词不纠正：对某个冲突写法设 override 不解除别名歧义；
    ///    override key 直接落在该词上才算「选定规范写法」。
    #[test]
    fn ambiguous_alias_not_rescued_by_canonical_override() {
        let base = [
            ("THUOCL_IT", &["甲写法|碰撞词"][..]),
            ("THUOCL_Law", &["乙写法|碰撞词"][..]),
        ];
        let rescued_on_spelling =
            settings_with(&[], &base, &[("甲写法", TermMode::ActiveExact)], &[]);
        let r = resolve(&rescued_on_spelling, "碰撞词");
        assert_eq!(r.canonical, None);
        assert_eq!(r.effective_mode, TermMode::Candidate);
        assert_eq!(r.winning_rule, LexiconWinningRule::Ambiguous);

        // override key = 该词本身 → 规范写法已选定，恢复纠正资格。
        let picked = settings_with(&[], &base, &[("碰撞词", TermMode::ActiveExact)], &[]);
        let r = resolve(&picked, "碰撞词");
        assert_eq!(r.canonical.as_deref(), Some("碰撞词"));
        assert_eq!(r.effective_mode, TermMode::ActiveExact);
        assert_eq!(r.winning_rule, LexiconWinningRule::TermOverride);
    }

    /// 7) Blocked 全屏蔽：单条 override Blocked 把用户词与公开源一起压掉。
    #[test]
    fn blocked_override_suppresses_all_sources() {
        let settings = settings_with(
            &["正常词|坏词"],
            &[("THUOCL_IT", &["公开词|坏词"])],
            &[("坏词", TermMode::Blocked)],
            &[],
        );
        let r = resolve(&settings, "坏词");
        assert_eq!(r.effective_mode, TermMode::Blocked);
        assert_eq!(r.winning_rule, LexiconWinningRule::TermOverride);
        assert!(r.contributing_sources.is_empty());
        assert_eq!(
            r.suppressed_sources,
            vec!["custom_words".to_string(), "THUOCL_IT".to_string()]
        );
    }

    /// 8) explanation 含来源 id：各典型场景都能看到来源标识。
    #[test]
    fn explanation_contains_source_ids() {
        let merged = settings_with(
            &[],
            &[("THUOCL_IT", &["合并词"]), ("THUOCL_Law", &["合并词"])],
            &[],
            &[],
        );
        let r = resolve(&merged, "合并词");
        assert!(r.explanation.contains("THUOCL_IT"));
        assert!(r.explanation.contains("THUOCL_Law"));

        let beaten = settings_with(
            &["用户写法|混拼词"],
            &[("THUOCL_Law", &["公开写法|混拼词"])],
            &[],
            &[],
        );
        let r = resolve(&beaten, "混拼词");
        assert!(r.explanation.contains("custom_words"));
        assert!(r.explanation.contains("THUOCL_Law"));

        let overrode = settings_with(
            &[],
            &[("THUOCL_IT", &["领域词"])],
            &[("领域词", TermMode::ActiveFuzzy)],
            &[],
        );
        let r = resolve(&overrode, "领域词");
        assert!(r.explanation.contains("THUOCL_IT"));
        assert!(r
            .explanation
            .contains("lexicon_term_modes[领域词]=active_fuzzy"));
    }

    /// 9) 别名继承所在行 primary 的 override 档位。
    #[test]
    fn alias_inherits_primary_override_mode() {
        let settings = settings_with(
            &["甲乙丙丁|首拼混写"],
            &[],
            &[("甲乙丙丁", TermMode::ActiveExact)],
            &[],
        );
        let r = resolve(&settings, "首拼混写");
        assert_eq!(r.canonical.as_deref(), Some("甲乙丙丁"));
        assert_eq!(r.effective_mode, TermMode::ActiveExact);
        assert_eq!(r.winning_rule, LexiconWinningRule::TermOverride);
        assert_eq!(r.contributing_sources, vec!["custom_words"]);
    }

    /// 10) 未收录 → Unknown + Blocked，无来源归属。
    #[test]
    fn unknown_term_is_unknown_blocked() {
        let settings = settings_with(&["无关词"], &[("THUOCL_IT", &["无关词"])], &[], &[]);
        let r = resolve(&settings, "根本不存在的词");
        assert_eq!(r.canonical, None);
        assert_eq!(r.winning_rule, LexiconWinningRule::Unknown);
        assert_eq!(r.effective_mode, TermMode::Blocked);
        assert!(r.contributing_sources.is_empty());
        assert!(r.suppressed_sources.is_empty());
    }

    /// 11) 仅公开来源贡献、档位不一致时取最保守档（candidate 参与 →
    ///     合并降 Candidate）。
    #[test]
    fn merged_public_sources_take_conservative_mode() {
        let settings = settings_with(
            &[],
            &[("THUOCL_IT", &["共同词"]), ("THUOCL_Law", &["共同词"])],
            &[],
            &[("THUOCL_IT", TermMode::Candidate)],
        );
        let r = resolve(&settings, "共同词");
        assert_eq!(r.canonical.as_deref(), Some("共同词"));
        assert_eq!(r.effective_mode, TermMode::Candidate);
        assert_eq!(
            r.contributing_sources,
            vec!["THUOCL_IT".to_string(), "THUOCL_Law".to_string()]
        );
    }

    /// 12) 用户自有词内部写法冲突同样保持歧义（用户层不内部决胜）。
    #[test]
    fn user_words_conflict_stays_ambiguous() {
        let settings = settings_with(&["甲词|共用混拼", "乙词|共用混拼"], &[], &[], &[]);
        let r = resolve(&settings, "共用混拼");
        assert_eq!(r.canonical, None);
        assert_eq!(r.effective_mode, TermMode::Candidate);
        assert_eq!(r.winning_rule, LexiconWinningRule::Ambiguous);
        assert_eq!(r.suppressed_sources, vec!["custom_words"]);
    }

    /// 13) 查询词 trim；别名与 primary 双入口都能命中同一裁决。
    #[test]
    fn primary_and_alias_entries_resolve_consistently() {
        let settings = settings_with(&["规范主词|别名词"], &[], &[], &[]);
        let by_primary = resolve(&settings, "  规范主词  ");
        let by_alias = resolve(&settings, "别名词");
        assert_eq!(by_primary.canonical.as_deref(), Some("规范主词"));
        assert_eq!(by_alias.canonical.as_deref(), Some("规范主词"));
        assert_eq!(by_primary.effective_mode, by_alias.effective_mode);
    }
}
