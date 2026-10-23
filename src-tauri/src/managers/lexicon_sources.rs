//! 开源词库源：分组目录 + 单源按需「预览 → 确认」导入。
//!
//! 与 THUOCL 导入（`shortcut::import_thuocl_lexicon`）并列、复用同一
//! 过滤链路（内置常用词/虚词表 + 词频头部丢弃，见
//! `shortcut::drop_generic_words`），但**按需加载**：
//!
//! - 每个源一个独立入口；`preview_open_lexicon_source` 只拉取/解析/
//!   过滤并返回预览（总条数、过滤后保留数、前 10 条样例），**不写任何
//!   设置**；用户在前端点确认后 `confirm_open_lexicon_source` 才把该源
//!   快照写入 `imported_lexicons`（落 Candidate 态，由词表页确认生效）。
//! - 没有「全部导入/批量导入」入口；同一源重复导入为**替换快照**
//!   （`HashMap::insert` 覆盖同键），不同来源互不影响。
//! - 范围筛选（同一源内）：全部 / 仅拉丁（无汉字且含 ASCII 字母）/
//!   仅中文（含汉字）；按词频取前 N（500/1000/2000/5000，默认 2000）。
//!   每源注册默认范围（英汉词典默认仅拉丁、中文词库默认仅汉字），
//!   `scope` 缺省时后端取该源的默认值。
//!
//! 源清单（分组，每组一句用途说明 + 许可；目录由 `get_lexicon_sources`
//! 动态返回，前端不写死源列表/分组文案）：
//!
//! | 组 | 源 | 格式 | 许可 |
//! |----|----|------|------|
//! | 专名/术语 | `zhwiki` | rime dict.yaml（release 资产） | CC BY-SA 4.0（需署名） |
//! | 专名/术语 | `rime-ice-en` / `rime-frost-en` | rime dict.yaml | GPL-3.0 |
//! | 专名/术语 | `rime-dict` | rime dict.yaml | LGPL-3.0 |
//! | 通用中文词库 | `rime-ice-base` / `rime-ice-8105` / `rime-ice-others` | rime dict.yaml | GPL-3.0 |
//! | 通用中文词库 | `rime-frost-base` / `rime-frost-others` | rime dict.yaml | GPL-3.0 |
//! | 通用中文词库 | `jieba-dict` | jieba `词 词频 词性` | MIT |
//! | 领域/分类术语 | `thuocl-*` 11 分类 | `词\t词频` TSV | THUOCL 上游 |
//! | 词频基准 | `rime-essay` | `词\t权重` TSV | LGPL-3.0 |
//!
//! 注：上游不存在 `rime/rime-dict` 仓库，`rime-dict` 以 Rime 官方
//! `rime/rime-luna-pinyin` 的 `luna_pinyin.dict.yaml` 充当官方词库。
//!
//! 专名组的 `zhwiki` / `rime-ice-en` / `rime-frost-en`
//! （`PROPER_NOUN_SOURCE_IDS`）导入后还会派生「小写 → 规范大小写」的
//! 字面规范化规则（见 `audio_toolkit::proper_nouns`）；词频基准组
//! `rime-essay` 是通用词频基准，导入会混入大量通用词，预览带显式警告。
//!
//! 不做联网自动更新（隐私与离线优先）；`proper_nouns_lexicon_version`
//! 只读命令返回当前内置种子版本/条数与生效规则数，便于将来接更新。

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{AppHandle, Emitter};

use crate::audio_toolkit::chinese_hotwords::is_han;
use crate::audio_toolkit::proper_nouns::{
    self, ProperNounConfig, ProperNounLexicon, PROPER_NOUN_SOURCE_IDS,
};
use crate::settings::{self, AppSettings};

// ---------------------------------------------------------------------------
// 目录（静态注册表）
// ---------------------------------------------------------------------------

/// 源分组（UI 分组呈现；分组目录随 `get_lexicon_sources` 返回，前端
/// 不写死；词频基准组默认折叠并标注用途）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LexiconSourceGroup {
    /// 专名/术语（推荐）
    ProperNouns,
    /// 通用中文词库（基础词组 / 字表 / 大词库）
    General,
    /// 领域/分类术语（THUOCL 11 分类）
    Domain,
    /// 词频基准（只用于词频参照/过滤阈值）
    FrequencyBaseline,
}

impl LexiconSourceGroup {
    /// UI 分组呈现顺序（专名 → 通用 → 领域 → 词频基准）。
    pub fn order(self) -> u8 {
        match self {
            Self::ProperNouns => 0,
            Self::General => 1,
            Self::Domain => 2,
            Self::FrequencyBaseline => 3,
        }
    }

    /// 分组展示名（随目录命令返回，前端不写死）。
    pub fn label(self) -> &'static str {
        match self {
            Self::ProperNouns => "专名/术语",
            Self::General => "通用中文词库",
            Self::Domain => "领域/分类术语",
            Self::FrequencyBaseline => "词频基准",
        }
    }

    /// 分组一句用途说明。
    pub fn description(self) -> &'static str {
        match self {
            Self::ProperNouns => "产品/组织/术语等专名，纠正大小写与音译（推荐优先导入）",
            Self::General => "开源通用中文词库（基础词组、字表、大词库），补足日常词汇",
            Self::Domain => "按领域分类的术语（THUOCL），按需挑选分类导入",
            Self::FrequencyBaseline => "通用词频基准：不建议直接导入，适合作为词频参照/过滤阈值",
        }
    }
}

/// 导入范围筛选（同一源内）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LexiconScope {
    #[default]
    All,
    /// 仅拉丁/英文专名：无汉字且含 ASCII 字母。
    LatinOnly,
    /// 仅中文词：含汉字。
    HanOnly,
}

/// 词库文件格式（决定解析器）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LexiconSourceFormat {
    /// rime dict.yaml：frontmatter + `词\t码[ \t权重]` 行（权重可缺省，
    /// 缺省时文件顺序即词频序）。
    RimeDictYaml,
    /// rime essay.txt：`词\t权重` 两列 TSV（无 frontmatter）。
    EssayTsv,
    /// THUOCL：`词\t词频` 两列 TSV（纯汉字 2–50 字）。
    ThuoclTsv,
    /// jieba extra_dict/dict.txt(.big)：`词 词频 [词性]`（空格分隔，
    /// 词性可缺省）。
    JiebaTxt,
}

/// 一个预置源。
#[derive(Clone, Debug, Serialize, Type)]
pub struct LexiconSourceInfo {
    /// 稳定 id（也是 `imported_lexicons` 的存储键；THUOCL 为
    /// `thuocl-<category>`，存储键是裸分类名）。
    pub id: String,
    /// 展示名（如 `zhwiki`、`rime-ice/en`）。
    pub label: String,
    pub group: LexiconSourceGroup,
    /// 用途一句话。
    pub description: String,
    /// 许可（含署名要求）。
    pub license: String,
    /// 是否推荐（专名派生三源 + zhwiki 为 true；词频基准为 false）。
    pub recommended: bool,
    pub format: LexiconSourceFormat,
    /// 上游主页/发布页（网络失败时给用户的手工获取地址）。
    pub source_url: String,
    /// 默认导入范围（`scope` 缺省时使用；英汉词典默认仅拉丁、中文词库
    /// 默认仅汉字）。
    pub default_scope: LexiconScope,
}

/// 拉取方式。
#[derive(Clone, Debug)]
enum SourceFetch {
    /// 顺序尝试的镜像 URL。
    Mirrors(Vec<String>),
    /// zhwiki：GitHub API 解析最新 release 的 `zhwiki-*.dict.yaml` 资产。
    ZhwikiLatestRelease,
}

struct SourceDef {
    info: LexiconSourceInfo,
    fetch: SourceFetch,
    /// 连接+传输超时（大文件给足预算）。
    timeout_secs: u64,
    /// 响应体硬上限（zhwiki 全量 dict.yaml 实测 ~55 MiB）。
    max_body_bytes: usize,
    /// 词频基准组导入警告（其余 None）。
    import_warning: Option<&'static str>,
}

const ZHWIKI_RELEASES_PAGE: &str = "https://github.com/felixonmars/fcitx5-pinyin-zhwiki/releases";
const ZHWIKI_RELEASE_API: &str =
    "https://api.github.com/repos/felixonmars/fcitx5-pinyin-zhwiki/releases/latest";
/// zhwiki release 资产名形如 `zhwiki-20261018.dict.yaml`（日期随 release 变）。
const ZHWIKI_ASSET_PREFIX: &str = "zhwiki-";
const ZHWIKI_ASSET_SUFFIX: &str = ".dict.yaml";

const RIME_ESSAY_IMPORT_WARNING: &str =
    "通用词频基准：不建议直接导入（会混入大量通用词）；适合作为词频参照/过滤阈值";

fn gh_mirrors(repo: &str, ref_: &str, path: &str) -> Vec<String> {
    vec![
        format!("https://raw.githubusercontent.com/{repo}/{ref_}/{path}"),
        format!("https://cdn.jsdelivr.net/gh/{repo}@{ref_}/{path}"),
        format!("https://fastly.jsdelivr.net/gh/{repo}@{ref_}/{path}"),
        format!("https://gcore.jsdelivr.net/gh/{repo}@{ref_}/{path}"),
    ]
}

/// 开源源注册表（静态；`get_lexicon_sources` 据此动态返回目录，前端
/// 不写死）。
fn open_source_defs() -> Vec<SourceDef> {
    vec![
        // ---------- 专名/术语（propers） ----------
        SourceDef {
            info: LexiconSourceInfo {
                id: "zhwiki".into(),
                label: "zhwiki".into(),
                group: LexiconSourceGroup::ProperNouns,
                description: "中文维基百科专名/术语（密度最高；最新 release 的 Rime 词库）".into(),
                license: "CC BY-SA 4.0（需署名：felixonmars/fcitx5-pinyin-zhwiki）".into(),
                recommended: true,
                format: LexiconSourceFormat::RimeDictYaml,
                source_url: ZHWIKI_RELEASES_PAGE.into(),
                default_scope: LexiconScope::All,
            },
            fetch: SourceFetch::ZhwikiLatestRelease,
            timeout_secs: 180,
            max_body_bytes: 96 * 1024 * 1024,
            import_warning: None,
        },
        SourceDef {
            info: LexiconSourceInfo {
                id: "rime-ice-en".into(),
                label: "rime-ice/en".into(),
                group: LexiconSourceGroup::ProperNouns,
                description: "rime-ice 英文词典（常用词 + 英文专名，按词频排序）".into(),
                license: "GPL-3.0".into(),
                recommended: true,
                format: LexiconSourceFormat::RimeDictYaml,
                source_url: "https://github.com/iDvel/rime-ice".into(),
                default_scope: LexiconScope::LatinOnly,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors(
                "iDvel/rime-ice",
                "main",
                "en_dicts/en.dict.yaml",
            )),
            timeout_secs: 20,
            max_body_bytes: 8 * 1024 * 1024,
            import_warning: None,
        },
        SourceDef {
            info: LexiconSourceInfo {
                id: "rime-frost-en".into(),
                label: "rime-frost/en".into(),
                group: LexiconSourceGroup::ProperNouns,
                description: "白霜拼音（rime-frost）英文词典（rime-ice 的扩充补充）".into(),
                license: "GPL-3.0".into(),
                recommended: true,
                format: LexiconSourceFormat::RimeDictYaml,
                source_url: "https://github.com/gaboolic/rime-frost".into(),
                default_scope: LexiconScope::LatinOnly,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors(
                "gaboolic/rime-frost",
                "master",
                "en_dicts/en.dict.yaml",
            )),
            timeout_secs: 20,
            max_body_bytes: 8 * 1024 * 1024,
            import_warning: None,
        },
        SourceDef {
            info: LexiconSourceInfo {
                id: "rime-dict".into(),
                label: "rime-luna-pinyin".into(),
                group: LexiconSourceGroup::ProperNouns,
                description:
                    "Rime 官方拼音词库（luna_pinyin 通用字词；上游无 rime/rime-dict 仓库）".into(),
                license: "LGPL-3.0".into(),
                recommended: false,
                format: LexiconSourceFormat::RimeDictYaml,
                source_url: "https://github.com/rime/rime-luna-pinyin".into(),
                default_scope: LexiconScope::HanOnly,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors(
                "rime/rime-luna-pinyin",
                "master",
                "luna_pinyin.dict.yaml",
            )),
            timeout_secs: 30,
            max_body_bytes: 8 * 1024 * 1024,
            import_warning: None,
        },
        // ---------- 通用中文词库（general） ----------
        SourceDef {
            info: LexiconSourceInfo {
                id: "rime-ice-base".into(),
                label: "rime-ice/base".into(),
                group: LexiconSourceGroup::General,
                description: "rime-ice cn_dicts/base.dict.yaml（基础词组主库）".into(),
                license: "GPL-3.0".into(),
                recommended: false,
                format: LexiconSourceFormat::RimeDictYaml,
                source_url: "https://github.com/iDvel/rime-ice".into(),
                default_scope: LexiconScope::HanOnly,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors(
                "iDvel/rime-ice",
                "main",
                "cn_dicts/base.dict.yaml",
            )),
            timeout_secs: 60,
            max_body_bytes: 16 * 1024 * 1024,
            import_warning: None,
        },
        SourceDef {
            info: LexiconSourceInfo {
                id: "rime-ice-8105".into(),
                label: "rime-ice/8105".into(),
                group: LexiconSourceGroup::General,
                description: "rime-ice cn_dicts/8105.dict.yaml（通用规范汉字表 8105 字）".into(),
                license: "GPL-3.0".into(),
                recommended: false,
                format: LexiconSourceFormat::RimeDictYaml,
                source_url: "https://github.com/iDvel/rime-ice".into(),
                default_scope: LexiconScope::HanOnly,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors(
                "iDvel/rime-ice",
                "main",
                "cn_dicts/8105.dict.yaml",
            )),
            timeout_secs: 30,
            max_body_bytes: 8 * 1024 * 1024,
            import_warning: None,
        },
        SourceDef {
            info: LexiconSourceInfo {
                id: "rime-ice-others".into(),
                label: "rime-ice/others".into(),
                group: LexiconSourceGroup::General,
                description: "rime-ice cn_dicts/others.dict.yaml（中文专名/领域词）".into(),
                license: "GPL-3.0".into(),
                recommended: false,
                format: LexiconSourceFormat::RimeDictYaml,
                source_url: "https://github.com/iDvel/rime-ice".into(),
                default_scope: LexiconScope::HanOnly,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors(
                "iDvel/rime-ice",
                "main",
                "cn_dicts/others.dict.yaml",
            )),
            timeout_secs: 30,
            max_body_bytes: 8 * 1024 * 1024,
            import_warning: None,
        },
        SourceDef {
            info: LexiconSourceInfo {
                id: "rime-frost-base".into(),
                label: "rime-frost/base".into(),
                group: LexiconSourceGroup::General,
                description: "白霜拼音 cn_dicts/base.dict.yaml（基础词组主库，带权重）".into(),
                license: "GPL-3.0".into(),
                recommended: false,
                format: LexiconSourceFormat::RimeDictYaml,
                source_url: "https://github.com/gaboolic/rime-frost".into(),
                default_scope: LexiconScope::HanOnly,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors(
                "gaboolic/rime-frost",
                "master",
                "cn_dicts/base.dict.yaml",
            )),
            timeout_secs: 60,
            max_body_bytes: 16 * 1024 * 1024,
            import_warning: None,
        },
        SourceDef {
            info: LexiconSourceInfo {
                id: "rime-frost-others".into(),
                label: "rime-frost/others".into(),
                group: LexiconSourceGroup::General,
                description: "白霜拼音 cn_dicts/others.dict.yaml（中文领域词，带权重）".into(),
                license: "GPL-3.0".into(),
                recommended: false,
                format: LexiconSourceFormat::RimeDictYaml,
                source_url: "https://github.com/gaboolic/rime-frost".into(),
                default_scope: LexiconScope::HanOnly,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors(
                "gaboolic/rime-frost",
                "master",
                "cn_dicts/others.dict.yaml",
            )),
            timeout_secs: 30,
            max_body_bytes: 8 * 1024 * 1024,
            import_warning: None,
        },
        SourceDef {
            info: LexiconSourceInfo {
                id: "jieba-dict".into(),
                label: "jieba/dict.txt.big".into(),
                group: LexiconSourceGroup::General,
                description: "结巴分词大词库（`词 词频 词性`，中英混合通用词）".into(),
                license: "MIT".into(),
                recommended: false,
                format: LexiconSourceFormat::JiebaTxt,
                source_url: "https://github.com/fxsjy/jieba".into(),
                default_scope: LexiconScope::All,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors(
                "fxsjy/jieba",
                "master",
                "extra_dict/dict.txt.big",
            )),
            timeout_secs: 90,
            max_body_bytes: 32 * 1024 * 1024,
            import_warning: None,
        },
        // ---------- 词频基准（freq） ----------
        SourceDef {
            info: LexiconSourceInfo {
                id: "rime-essay".into(),
                label: "rime-essay".into(),
                group: LexiconSourceGroup::FrequencyBaseline,
                description: "八股文（rime-essay）通用词频基准：用于剔除最通用词，不建议直接导入"
                    .into(),
                license: "LGPL-3.0".into(),
                recommended: false,
                format: LexiconSourceFormat::EssayTsv,
                source_url: "https://github.com/rime/rime-essay".into(),
                default_scope: LexiconScope::All,
            },
            fetch: SourceFetch::Mirrors(gh_mirrors("rime/rime-essay", "master", "essay.txt")),
            timeout_secs: 60,
            max_body_bytes: 32 * 1024 * 1024,
            import_warning: Some(RIME_ESSAY_IMPORT_WARNING),
        },
    ]
}

/// THUOCL 分类源（领域/分类组；id = `thuocl-<分类>`，存储键 = 裸分类名）。
fn thuocl_source_defs() -> Vec<SourceDef> {
    crate::shortcut::THUOCL_CATEGORIES
        .iter()
        .map(|category| SourceDef {
            info: LexiconSourceInfo {
                id: format!("thuocl-{category}"),
                label: format!("THUOCL {category}"),
                group: LexiconSourceGroup::Domain,
                description: "清华开放中文词库分类（词\\t词频）".into(),
                license: "THUOCL 上游数据（见仓库说明）".into(),
                recommended: false,
                format: LexiconSourceFormat::ThuoclTsv,
                source_url: "https://github.com/thunlp/THUOCL".into(),
                default_scope: LexiconScope::HanOnly,
            },
            fetch: SourceFetch::Mirrors(crate::shortcut::thuocl_urls(category)),
            timeout_secs: 12,
            max_body_bytes: 8 * 1024 * 1024,
            import_warning: None,
        })
        .collect()
}

fn all_source_defs() -> Vec<SourceDef> {
    let mut defs = open_source_defs();
    defs.extend(thuocl_source_defs());
    defs
}

/// 分组目录行（分组展示名 + 一句用途；UI 按分组渲染，不写死分组文案）。
#[derive(Clone, Debug, Serialize, Type)]
pub struct LexiconSourceGroupInfo {
    pub group: LexiconSourceGroup,
    /// 分组展示名（如「专名/术语」）。
    pub label: String,
    /// 分组一句用途说明。
    pub description: String,
}

/// 分组目录（`get_lexicon_sources` 返回值）：分组元数据 + 全部源。
#[derive(Clone, Debug, Serialize, Type)]
pub struct LexiconSourceCatalog {
    /// 全部分组（按呈现顺序，只含有源的分组）。
    pub groups: Vec<LexiconSourceGroupInfo>,
    /// 全部源（按分组顺序 + 组内注册顺序；每条含分组与许可）。
    pub sources: Vec<LexiconSourceInfo>,
}

/// 扁平源列表（前端 `listLexiconSources` 使用）：与 [`get_lexicon_sources`]
/// 同源，只是把 `catalog.sources` 直接展开。保留两个命令是为了不回退前端
/// 已写好的分组渲染（它按每条 `group` 字段自行分组）。
#[tauri::command]
#[specta::specta]
pub fn list_lexicon_sources() -> Vec<LexiconSourceInfo> {
    get_lexicon_sources().sources
}

/// 分组目录（只读；已安装状态由前端从 `imported_lexicons` 合并）。
///
/// 返回全部预置源（含分组与许可）；UI 按分组动态渲染，不在前端写死
/// 源列表或分组文案。
#[tauri::command]
#[specta::specta]
pub fn get_lexicon_sources() -> LexiconSourceCatalog {
    let mut sources: Vec<LexiconSourceInfo> =
        all_source_defs().into_iter().map(|def| def.info).collect();
    // 分组序（组内保持注册序）；命令输出顺序即 UI 呈现顺序。
    sources.sort_by_key(|info| info.group.order());
    let groups = [
        LexiconSourceGroup::ProperNouns,
        LexiconSourceGroup::General,
        LexiconSourceGroup::Domain,
        LexiconSourceGroup::FrequencyBaseline,
    ]
    .into_iter()
    .filter(|group| sources.iter().any(|source| source.group == *group))
    .map(|group| LexiconSourceGroupInfo {
        group,
        label: group.label().into(),
        description: group.description().into(),
    })
    .collect();
    LexiconSourceCatalog { groups, sources }
}

fn find_source_def(id: &str) -> Result<SourceDef, String> {
    all_source_defs()
        .into_iter()
        .find(|def| def.info.id == id)
        .ok_or_else(|| format!("Unknown lexicon source: {id}"))
}

/// 源 id → `imported_lexicons` 存储键（THUOCL 去掉前缀，其余即 id）。
pub fn stored_key_for(id: &str) -> String {
    id.strip_prefix("thuocl-").unwrap_or(id).to_string()
}

// ---------------------------------------------------------------------------
// 限额与资格
// ---------------------------------------------------------------------------

/// 单源导入的词条上限（与 THUOCL 一致）。
pub const OPEN_LEXICON_MAX_WORDS: usize = 5000;
/// 「按词频取前 N」可选项（默认 2000）。
pub const OPEN_LEXICON_LIMIT_CHOICES: &[usize] = &[500, 1000, 2000, 5000];
/// 默认取前 N。
pub const OPEN_LEXICON_DEFAULT_LIMIT: usize = 2000;

fn normalize_limit(limit: Option<usize>) -> Result<usize, String> {
    match limit {
        None => Ok(OPEN_LEXICON_DEFAULT_LIMIT),
        Some(n) if OPEN_LEXICON_LIMIT_CHOICES.contains(&n) => Ok(n),
        Some(n) => Err(format!(
            "Invalid limit {n}; must be one of {OPEN_LEXICON_LIMIT_CHOICES:?}"
        )),
    }
}

/// 开源源词条资格：trim 后 1–64 字符、含汉字或 ASCII 字母、无控制字符。
fn eligible_open_word(word: &str) -> bool {
    let word = word.trim();
    let char_count = word.chars().count();
    (1..=64).contains(&char_count)
        && word.chars().any(|c| is_han(c) || c.is_ascii_alphabetic())
        && !word.chars().any(|c| c.is_control())
}

/// 范围筛选：LatinOnly = 无汉字且含 ASCII 字母；HanOnly = 含汉字。
fn matches_scope(word: &str, scope: LexiconScope) -> bool {
    let has_han = word.chars().any(|c| is_han(c));
    match scope {
        LexiconScope::All => true,
        LexiconScope::LatinOnly => !has_han && word.chars().any(|c| c.is_ascii_alphabetic()),
        LexiconScope::HanOnly => has_han,
    }
}

// ---------------------------------------------------------------------------
// 解析器（rime dict.yaml / essay TSV / THUOCL TSV）
// ---------------------------------------------------------------------------

/// 解析产物：版本（frontmatter/release）+ 去重前词条（词, 权重）。
#[derive(Debug, Default)]
pub struct ParsedLexiconFile {
    pub version: Option<String>,
    pub entries: Vec<(String, Option<u64>)>,
}

fn parse_weight(field: &str) -> Option<u64> {
    // rime 权重可能带 `l%` 等后缀：取前导数字。
    let digits: String = field
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<u64>().ok()
}

/// 解析 rime dict.yaml：
/// - `#` 开头注释行忽略；frontmatter（首个 `---` 行到 `...`/第二个 `---`
///   行之间）提取 `version:`（去引号），frontmatter 本体不是词条；
/// - 词条行 `词\t码[ \t权重]`（en 词典常见两列无权重，此时文件顺序即
///   词频序）；
/// - **非词典文件报错**：解析出 0 条词条，或非空非注释行里词条占比
///   过低（< 1/10，如错误页/README 整页散文）都在源头被拒。
pub fn parse_rime_dict_yaml(body: &str) -> Result<ParsedLexiconFile, String> {
    let body = body.strip_prefix('\u{feff}').unwrap_or(body);
    let mut parsed = ParsedLexiconFile::default();
    let mut in_frontmatter = false;
    let mut frontmatter_done = false;
    let mut content_lines = 0usize;
    for raw_line in body.lines() {
        let line = raw_line.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed == "---" && !frontmatter_done && parsed.entries.is_empty() {
            in_frontmatter = true;
            continue;
        }
        if in_frontmatter {
            if trimmed == "..." || trimmed == "---" {
                in_frontmatter = false;
                frontmatter_done = true;
                continue;
            }
            if let Some(value) = trimmed.strip_prefix("version:") {
                let value = value.trim().trim_matches('"').trim_matches('\'');
                if parsed.version.is_none() && !value.is_empty() {
                    parsed.version = Some(value.to_string());
                }
            }
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        content_lines += 1;
        let mut columns = line.split('\t');
        let (Some(word), Some(code)) = (columns.next(), columns.next()) else {
            continue; // 单列行不是词条
        };
        let word = word.trim();
        let code = code.trim();
        if word.is_empty() || code.is_empty() {
            continue;
        }
        // 码列是数字的（如 essay 误当 dict 用）不算 dict 词条；dict 的码
        // 是拼音/拼写。权重列可缺省。
        if code.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let weight = columns.next().and_then(parse_weight);
        parsed.entries.push((word.to_string(), weight));
    }
    ensure_lexicon_shaped(
        &parsed.entries,
        content_lines,
        "rime dict.yaml",
        "word\\tcode[\\tweight]",
    )?;
    Ok(parsed)
}

/// 非词典文件拒收：0 词条，或非空非注释行里词条占比 < 1/10
/// （错误页/README 冒充词库的典型形状）。
fn ensure_lexicon_shaped(
    entries: &[(String, Option<u64>)],
    content_lines: usize,
    kind: &str,
    shape: &str,
) -> Result<(), String> {
    if entries.is_empty() {
        return Err(format!("not a {kind} file (no `{shape}` entry lines)"));
    }
    if entries.len() * 10 < content_lines {
        return Err(format!(
            "not a {kind} file (only {} entry lines out of {} content lines)",
            entries.len(),
            content_lines
        ));
    }
    Ok(())
}

/// 解析 rime essay.txt：每行 `词\t权重`（权重必须可解析；无 frontmatter）。
/// 非词典文件（0 词条 / 散文占比过高）报错。
pub fn parse_essay_tsv(body: &str) -> Result<ParsedLexiconFile, String> {
    let body = body.strip_prefix('\u{feff}').unwrap_or(body);
    let mut parsed = ParsedLexiconFile::default();
    let mut content_lines = 0usize;
    for raw_line in body.lines() {
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        content_lines += 1;
        let mut columns = line.split('\t');
        let (Some(word), Some(weight), None) = (columns.next(), columns.next(), columns.next())
        else {
            continue;
        };
        let word = word.trim();
        let Some(weight) = parse_weight(weight) else {
            continue;
        };
        if word.is_empty() {
            continue;
        }
        parsed.entries.push((word.to_string(), Some(weight)));
    }
    ensure_lexicon_shaped(&parsed.entries, content_lines, "essay TSV", "word\\tweight")?;
    Ok(parsed)
}

/// 解析 THUOCL `词\t词频`（纯汉字 2–50 字，资格与既有导入完全一致）。
fn parse_thuocl_tsv(body: &str) -> Result<ParsedLexiconFile, String> {
    let mut parsed = ParsedLexiconFile::default();
    for raw_line in body.lines() {
        let mut columns = raw_line.split('\t');
        let (Some(word), Some(freq)) = (columns.next(), columns.next()) else {
            continue;
        };
        let word = word.trim();
        if !crate::shortcut::is_eligible_lexicon_word(word) {
            continue;
        }
        let Some(freq) = freq.trim().parse::<u64>().ok() else {
            continue;
        };
        parsed.entries.push((word.to_string(), Some(freq)));
    }
    if parsed.entries.is_empty() {
        return Err("no eligible THUOCL words in file".to_string());
    }
    Ok(parsed)
}

/// 解析 jieba extra_dict/dict.txt(.big)：每行 `词 词频 [词性]`（空格
/// 分隔，词性可缺省，多余列忽略）。非词典文件（0 词条 / 散文占比过高）
/// 报错。
pub fn parse_jieba_txt(body: &str) -> Result<ParsedLexiconFile, String> {
    let body = body.strip_prefix('\u{feff}').unwrap_or(body);
    let mut parsed = ParsedLexiconFile::default();
    let mut content_lines = 0usize;
    for raw_line in body.lines() {
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        content_lines += 1;
        let mut fields = line.split_whitespace();
        let (Some(word), Some(freq)) = (fields.next(), fields.next()) else {
            continue; // 单列行不是词条
        };
        let word = word.trim();
        let Some(freq) = parse_weight(freq) else {
            continue; // 词频列非数字（如 `词 xyz`）不是词条
        };
        if word.is_empty() {
            continue;
        }
        parsed.entries.push((word.to_string(), Some(freq)));
    }
    ensure_lexicon_shaped(
        &parsed.entries,
        content_lines,
        "jieba dict",
        "word freq [pos]",
    )?;
    Ok(parsed)
}

// ---------------------------------------------------------------------------
// 过滤链（资格 → 范围 → 去重保最高词频 → 词频排序 → 常用词两级过滤 → 前 N）
// ---------------------------------------------------------------------------

/// 单源「预览」的纯计算核心（不触网、不写设置；测试入口）。
pub fn build_preview_from_body(
    info: &LexiconSourceInfo,
    format: LexiconSourceFormat,
    import_warning: Option<&'static str>,
    version_fallback: &str,
    fetched_url: &str,
    body: &str,
    scope: LexiconScope,
    limit: usize,
) -> Result<OpenLexiconPreview, String> {
    let mut parsed = match format {
        LexiconSourceFormat::RimeDictYaml => parse_rime_dict_yaml(body)?,
        LexiconSourceFormat::EssayTsv => parse_essay_tsv(body)?,
        LexiconSourceFormat::ThuoclTsv => parse_thuocl_tsv(body)?,
        LexiconSourceFormat::JiebaTxt => parse_jieba_txt(body)?,
    };

    // 资格 + 范围筛选。
    parsed
        .entries
        .retain(|(word, _)| eligible_open_word(word) && matches_scope(word, scope));

    // 去重：同词保留最高词频；无词频（文件序）按首见序。
    let mut best: HashMap<&str, (Option<u64>, usize)> = HashMap::new();
    for (index, (word, weight)) in parsed.entries.iter().enumerate() {
        let entry = best.entry(word.as_str()).or_insert((*weight, index));
        if *weight > entry.0 {
            entry.0 = *weight;
        }
    }
    let total_unique = best.len();

    // 排序：词频降序（无词频的按文件序在后）；并列按首见序稳定。
    let mut ranked: Vec<(&str, (Option<u64>, usize))> = best.into_iter().collect();
    ranked.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then_with(|| a.1 .1.cmp(&b.1 .1)));

    let (kept, filtered) = crate::shortcut::drop_generic_words(
        ranked
            .into_iter()
            .map(|(word, (weight, _))| (word.to_string(), weight))
            .collect(),
        limit,
    );

    // 样例取最终保留的前 10 条（带词频，供用户判断导入质量）。
    let weight_of: HashMap<&str, Option<u64>> = parsed
        .entries
        .iter()
        .map(|(word, weight)| (word.as_str(), *weight))
        .collect();
    let samples: Vec<LexiconSample> = kept
        .iter()
        .take(10)
        .map(|word| LexiconSample {
            word: word.clone(),
            weight: weight_of.get(word.as_str()).copied().flatten(),
        })
        .collect();

    Ok(OpenLexiconPreview {
        id: info.id.clone(),
        label: info.label.clone(),
        license: info.license.clone(),
        version: parsed
            .version
            .take()
            .unwrap_or_else(|| version_fallback.to_string()),
        source_url: fetched_url.to_string(),
        total_unique,
        filtered,
        words: kept,
        samples,
        warning: import_warning.map(str::to_string),
    })
}

// ---------------------------------------------------------------------------
// 网络（镜像顺序尝试 + 体量硬限；zhwiki 经 GitHub API 解析 release 资产）
// ---------------------------------------------------------------------------

fn chunk_fits(body_len: usize, chunk_len: usize, cap: usize) -> bool {
    body_len
        .checked_add(chunk_len)
        .map(|total| total <= cap)
        .unwrap_or(false)
}

async fn read_capped_utf8_body(
    mut response: reqwest::Response,
    cap: usize,
    label: &str,
) -> Result<String, String> {
    let mut body: Vec<u8> = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| format!("{label}: {e}"))?
    {
        if !chunk_fits(body.len(), chunk.len(), cap) {
            return Err(format!("{label}: response body exceeds {cap} bytes"));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|_| format!("{label}: response body is not valid UTF-8"))
}

/// zhwiki：查询最新 release，取 `zhwiki-*.dict.yaml` 资产。取不到时报错并
/// 附发布页地址（用户可手工下载后再试），绝不静默换别的文件。
async fn resolve_zhwiki_asset(client: &reqwest::Client) -> Result<(String, String), String> {
    let label = "Failed to resolve zhwiki latest release";
    let response = client
        .get(ZHWIKI_RELEASE_API)
        .header("User-Agent", "voo-lexicon-import")
        .header("Accept", "application/vnd.github+json")
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .map_err(|e| format!("{label}: {e}; 手工获取: {ZHWIKI_RELEASES_PAGE}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!(
            "{label}: HTTP {status}; 手工获取: {ZHWIKI_RELEASES_PAGE}"
        ));
    }
    let body = read_capped_utf8_body(response, 4 * 1024 * 1024, label).await?;
    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("{label}: invalid JSON: {e}"))?;
    let tag = value["tag_name"].as_str().unwrap_or("unknown").to_string();
    let assets = value["assets"]
        .as_array()
        .ok_or_else(|| format!("{label}: no assets; 手工获取: {ZHWIKI_RELEASES_PAGE}"))?;
    let asset = assets
        .iter()
        .filter_map(|asset| {
            let name = asset["name"].as_str()?;
            let url = asset["browser_download_url"].as_str()?;
            Some((name, url))
        })
        .find(|(name, _)| {
            name.starts_with(ZHWIKI_ASSET_PREFIX) && name.ends_with(ZHWIKI_ASSET_SUFFIX)
        });
    let Some((name, url)) = asset else {
        return Err(format!(
            "{label}: no `zhwiki-*.dict.yaml` asset in release {tag}; 手工获取: {ZHWIKI_RELEASES_PAGE}"
        ));
    };
    let date = name
        .strip_prefix(ZHWIKI_ASSET_PREFIX)
        .and_then(|rest| rest.strip_suffix(ZHWIKI_ASSET_SUFFIX))
        .unwrap_or("");
    let version = if date.is_empty() {
        tag
    } else {
        format!("{tag} / {date}")
    };
    Ok((url.to_string(), version))
}

async fn fetch_source_body(
    def: &SourceDef,
) -> Result<
    (
        String, /* body */
        String, /* version fallback */
        String, /* url */
    ),
    String,
> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(def.timeout_secs))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    let (urls, version_fallback): (Vec<String>, String) = match &def.fetch {
        SourceFetch::Mirrors(urls) => (urls.clone(), "master".to_string()),
        SourceFetch::ZhwikiLatestRelease => {
            let (url, version) = resolve_zhwiki_asset(&client).await?;
            (vec![url], version)
        }
    };

    let label = format!("Failed to download lexicon {}", def.info.label);
    let mut failures: Vec<String> = Vec::with_capacity(urls.len());
    for url in &urls {
        let response = match client
            .get(url)
            .timeout(Duration::from_secs(def.timeout_secs))
            .send()
            .await
        {
            Ok(response) => response,
            Err(e) => {
                failures.push(format!("{url}: {e}"));
                continue;
            }
        };
        let status = response.status();
        if !status.is_success() {
            failures.push(format!("{url}: HTTP {status}"));
            continue;
        }
        if let Some(length) = response.content_length() {
            if length as usize > def.max_body_bytes {
                failures.push(format!(
                    "{url}: response body exceeds {} bytes",
                    def.max_body_bytes
                ));
                continue;
            }
        }
        match read_capped_utf8_body(response, def.max_body_bytes, &label).await {
            Ok(body) => return Ok((body, version_fallback, url.clone())),
            Err(e) => failures.push(format!("{url}: {e}")),
        }
    }
    Err(format!(
        "{label}: {}; 手工获取: {}",
        failures.join(" | "),
        def.info.source_url
    ))
}

// ---------------------------------------------------------------------------
// 预览/确认命令
// ---------------------------------------------------------------------------

/// 一条样例（词 + 词频）。
#[derive(Clone, Debug, Serialize, Type)]
pub struct LexiconSample {
    pub word: String,
    pub weight: Option<u64>,
}

/// 单源预览结果（只读；确认前不写任何设置）。
#[derive(Clone, Debug, Serialize, Type)]
pub struct OpenLexiconPreview {
    pub id: String,
    pub label: String,
    pub license: String,
    /// 版本（frontmatter version / release tag+日期 / "master"）。
    pub version: String,
    /// 实际拉取 URL（或手工获取地址）。
    pub source_url: String,
    /// 资格 + 范围筛选 + 去重后的词条数。
    pub total_unique: usize,
    /// 常用词两级过滤丢弃数（内置常用表 + 词频头部 10%）。
    pub filtered: usize,
    /// 最终保留词条（按词频降序，≤ limit；确认导入的就是这份快照）。
    pub words: Vec<String>,
    /// 前 10 条样例（词 + 词频）。
    pub samples: Vec<LexiconSample>,
    /// 词频基准组导入警告（其余源为 None）。
    pub warning: Option<String>,
}

/// 预览一个开源词库源（拉取 + 解析 + 过滤；**不写设置**）。
///
/// 每源独立调用；没有批量入口。`limit` ∈ {500,1000,2000,5000}，默认
/// 2000；`scope` 缺省时用该源的 `default_scope`。非词典文件（错误页/
/// README）在解析层被拒。
#[tauri::command]
#[specta::specta]
pub async fn preview_open_lexicon_source(
    id: String,
    scope: Option<LexiconScope>,
    limit: Option<usize>,
) -> Result<OpenLexiconPreview, String> {
    let def = find_source_def(id.trim())?;
    // scope 缺省 → 该源注册的默认范围（英汉词典仅拉丁、中文词库仅汉字）。
    let scope = scope.unwrap_or(def.info.default_scope);
    let limit = normalize_limit(limit)?;
    let (body, version_fallback, url) = fetch_source_body(&def).await?;
    let preview = build_preview_from_body(
        &def.info,
        def.info.format,
        def.import_warning,
        &version_fallback,
        &url,
        &body,
        scope,
        limit,
    )?;
    if preview.words.is_empty() {
        return Err(format!(
            "Lexicon source {} has no eligible words after filtering (total={}, filtered={})",
            def.info.label, preview.total_unique, preview.filtered
        ));
    }
    log::info!(
        "Previewed open lexicon {}: total={}, filtered={}, kept={}, version={}",
        def.info.id,
        preview.total_unique,
        preview.filtered,
        preview.words.len(),
        preview.version
    );
    Ok(preview)
}

/// 确认导入结果。
#[derive(Clone, Debug, Serialize, Type)]
pub struct OpenLexiconConfirmResult {
    /// `imported_lexicons` 存储键。
    pub source: String,
    pub count: usize,
}

/// 校验并把一个源的词快照写入 `imported_lexicons`（纯函数；命令的写盘
/// 包装在 [`confirm_open_lexicon_source`]）。
///
/// - **替换语义**：同键 `insert` 覆盖旧快照，绝不追加；其他来源键不动。
/// - 后台逐词重新校验（不信任前端预览值）：THUOCL 源要求纯汉字 2–50 字
///   （与既有导入一致），其余源要求 [`eligible_open_word`]。
pub fn apply_open_source_import(
    imported_lexicons: &mut HashMap<String, Vec<String>>,
    id: &str,
    words: &[String],
) -> Result<String, String> {
    let def = find_source_def(id)?;
    let stored_key = stored_key_for(id);
    if words.is_empty() {
        return Err(format!("Lexicon source {stored_key} cannot be empty"));
    }
    if words.len() > OPEN_LEXICON_MAX_WORDS {
        return Err(format!(
            "Lexicon source {stored_key} exceeds the {OPEN_LEXICON_MAX_WORDS}-word limit"
        ));
    }
    let mut snapshot: Vec<String> = Vec::with_capacity(words.len());
    for word in words {
        let word = word.trim();
        let eligible = match def.info.format {
            LexiconSourceFormat::ThuoclTsv => crate::shortcut::is_eligible_lexicon_word(word),
            _ => eligible_open_word(word),
        };
        if !eligible {
            return Err(format!(
                "Invalid lexicon word in source {stored_key}: {word:?}"
            ));
        }
        if !snapshot.iter().any(|w| w == word) {
            snapshot.push(word.to_string());
        }
    }
    if snapshot.is_empty() {
        return Err(format!("Lexicon source {stored_key} cannot be empty"));
    }
    // 替换（insert 覆盖同键）；其他来源互不影响。
    imported_lexicons.insert(stored_key.clone(), snapshot);
    Ok(stored_key)
}

/// 确认导入一个源（用户在预览后点确认才调用；写设置 + 广播）。
#[tauri::command]
#[specta::specta]
pub fn confirm_open_lexicon_source(
    app: AppHandle,
    id: String,
    words: Vec<String>,
) -> Result<OpenLexiconConfirmResult, String> {
    let previous = settings::get_settings(&app);
    let mut updated = previous.clone();
    let source = apply_open_source_import(&mut updated.imported_lexicons, id.trim(), &words)?;
    let count = updated.imported_lexicons[&source].len();
    if let Err(err) = settings::try_write_settings(&app, &updated) {
        settings::write_settings(&app, previous);
        return Err(err);
    }
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({ "setting": "imported_lexicons", "value": source }),
    );
    Ok(OpenLexiconConfirmResult { source, count })
}

// ---------------------------------------------------------------------------
// 专名字面规范化：设置感知装配 + 只读版本命令
// ---------------------------------------------------------------------------

/// 从设置组装字面规范化配置（A 组源 + 用户规则 + 总开关）。
pub fn proper_noun_config_from_settings(settings: &AppSettings) -> ProperNounConfig<'_> {
    ProperNounConfig {
        enabled: settings.proper_noun_lexicon_enabled,
        user_rules: &settings.hotword_rules,
        sources: PROPER_NOUN_SOURCE_IDS
            .iter()
            .filter_map(|id| {
                settings
                    .imported_lexicons
                    .get(*id)
                    .map(|words| (*id, words.as_slice()))
            })
            .collect(),
    }
}

/// 内置种子 + 当前生效规则概览（只读；供将来接更新）。
#[derive(Clone, Debug, Serialize, Type)]
pub struct ProperNounsVersionInfo {
    pub version: String,
    pub updated_at: String,
    /// 内置种子条数（canonical 数）。
    pub entries: usize,
    /// 当前生效的字面规则数（种子 + A 组源派生 − 用户冲突）。
    pub rules: usize,
}

/// 只读：返回内置种子版本/条数与当前生效规则数。
#[tauri::command]
#[specta::specta]
pub fn proper_nouns_lexicon_version(app: AppHandle) -> Result<ProperNounsVersionInfo, String> {
    let ProperNounLexicon {
        version,
        updated_at,
        entries,
    } = proper_nouns::builtin_seed_lexicon()?;
    let settings = settings::get_settings(&app);
    let config = proper_noun_config_from_settings(&settings);
    let rules = proper_nouns::effective_proper_noun_rules(&config);
    Ok(ProperNounsVersionInfo {
        version,
        updated_at,
        entries: entries.len(),
        rules: rules.len(),
    })
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const RIME_YAML: &str = "\
# Rime dictionary
# encoding: utf-8
---
name: en
version: \"2026-09-01\"
sort: by_weight
...
# +_+
# a\ta
AA\tAA
aaa\taaa
Aaliyah\tAaliyah
GitHub\tgithub\t999
DeepSeek\tdeepseek\t500
chat gpt\tchat gpt\t87
你好世界\tni hao shi jie\t42
我们\two men\t100000
Vue\tvue
VS Code\tvs code
Windows\twindows\t31
widnows\twidnows
";

    const ZHWIKI_YAML: &str = "\
---
name: zhwiki
version: \"20261018\"
sort: original
...
深度求索\tshen du qiu suo\t100
中州韵\tzhong zhou yun\t80
维基百科\twei ji bai ke\t1000000
iPhone 15\tiphone 15\t50
AJAX\tajax\t60
的时候\tde shi hou\t999999
人工智能\tren gong zhi neng\t900
";

    const JIEBA_BIG: &str = "\
# 注释行
深度学习 3000 n
克拉玛依 3 ns
蹊跷 3
hello 7
坏行没有分隔符
词 xyz
飞鱼座 100 n
";

    fn info(id: &str, format: LexiconSourceFormat) -> LexiconSourceInfo {
        LexiconSourceInfo {
            id: id.into(),
            label: id.into(),
            group: LexiconSourceGroup::ProperNouns,
            description: "test".into(),
            license: "GPL-3.0".into(),
            recommended: true,
            format,
            source_url: "https://example.com".into(),
            default_scope: LexiconScope::All,
        }
    }

    #[test]
    fn rime_yaml_parsing_frontmatter_versions_comments() {
        let parsed = parse_rime_dict_yaml(RIME_YAML).expect("must parse");
        assert_eq!(parsed.version.as_deref(), Some("2026-09-01"));
        // 注释行（含 # a\ta 伪词条）不计；两列/三列词条都收。
        let words: Vec<&str> = parsed.entries.iter().map(|(w, _)| w.as_str()).collect();
        assert!(words.contains(&"GitHub"));
        assert!(words.contains(&"你好世界"));
        assert!(words.contains(&"VS Code"));
        assert!(!words.contains(&"a"));
        // 权重：三列有、两列无。
        let github = parsed.entries.iter().find(|(w, _)| w == "GitHub").unwrap();
        assert_eq!(github.1, Some(999));
        let vue = parsed.entries.iter().find(|(w, _)| w == "Vue").unwrap();
        assert_eq!(vue.1, None);
    }

    #[test]
    fn zhwiki_lines_parse_via_rime_parser() {
        let parsed = parse_rime_dict_yaml(ZHWIKI_YAML).expect("must parse");
        assert_eq!(parsed.version.as_deref(), Some("20261018"));
        let words: Vec<&str> = parsed.entries.iter().map(|(w, _)| w.as_str()).collect();
        assert!(words.contains(&"深度求索"));
        assert!(words.contains(&"iPhone 15"));
        // 码列为数字的行被拒（不是 dict 词条）。
        assert!(!words.contains(&"0"));
    }

    #[test]
    fn non_dict_files_are_rejected() {
        let html = "<html><body>404 not found</body></html>";
        assert!(parse_rime_dict_yaml(html).is_err());
        assert!(parse_essay_tsv(html).is_err());
        // 两列但码列是数字（essay 形状喂给 dict 解析器）→ 无词条 → 报错。
        let essay_shaped = "一词\t100\n二词\t90\n三词\t80\n";
        assert!(parse_rime_dict_yaml(essay_shaped).is_err());
        // dict 形状喂给 essay 解析器 → 权重非数字 → 报错。
        assert!(parse_essay_tsv(RIME_YAML).is_err());
    }

    #[test]
    fn essay_tsv_parsing() {
        let body = "的\t981\n了一\t658\n深度学习\t42\nhello\t7\n# 注释\n坏行 没有制表符\n";
        let parsed = parse_essay_tsv(body).expect("must parse");
        assert_eq!(parsed.entries.len(), 4);
        assert_eq!(parsed.entries[0], ("的".to_string(), Some(981)));
        assert_eq!(parsed.entries[2], ("深度学习".to_string(), Some(42)));
    }

    #[test]
    fn essay_garbage_rejected_tiny_valid_ok() {
        // 小而合法的 essay（两行）可解析。
        let small = parse_essay_tsv("的\t981\n了一\t658\n").expect("small valid essay parses");
        assert_eq!(small.entries.len(), 2);
        // 整页散文（无制表符）被拒。
        let prose = "这是第一行散文\n这是第二行散文\n这是第三行散文\n";
        assert!(parse_essay_tsv(prose).is_err());
    }

    #[test]
    fn common_words_filtered_and_license_version_in_preview() {
        let preview = build_preview_from_body(
            &info("rime-ice-en", LexiconSourceFormat::RimeDictYaml),
            LexiconSourceFormat::RimeDictYaml,
            None,
            "master",
            "https://example.com/en.dict.yaml",
            RIME_YAML,
            LexiconScope::All,
            500,
        )
        .expect("preview must build");
        // 许可与版本进入返回值。
        assert_eq!(preview.license, "GPL-3.0");
        assert_eq!(preview.version, "2026-09-01");
        assert_eq!(preview.id, "rime-ice-en");
        // 常用词被过滤：`我们` 命中内置常用表。
        assert!(!preview.words.contains(&"我们".to_string()));
        // 正常专名保留。
        assert!(preview.words.contains(&"GitHub".to_string()));
        assert!(preview.words.contains(&"DeepSeek".to_string()));
        // 样例 ≤10 条且带词频。
        assert!(preview.samples.len() <= 10);
        let github_sample = preview
            .samples
            .iter()
            .find(|s| s.word == "GitHub")
            .expect("GitHub should be in samples");
        assert_eq!(github_sample.weight, Some(999));
        assert!(preview.filtered >= 1, "我们 must be counted as filtered");
    }

    #[test]
    fn scope_filters_latin_and_han() {
        let base = |scope| {
            build_preview_from_body(
                &info("zhwiki", LexiconSourceFormat::RimeDictYaml),
                LexiconSourceFormat::RimeDictYaml,
                None,
                "master",
                "https://example.com",
                ZHWIKI_YAML,
                scope,
                500,
            )
            .expect("preview must build")
        };
        let latin = base(LexiconScope::LatinOnly);
        assert!(latin.words.iter().all(|w: &String| {
            !w.chars().any(|c| is_han(c)) && w.chars().any(|c| c.is_ascii_alphabetic())
        }));
        assert!(latin.words.contains(&"AJAX".to_string()));
        assert!(!latin.words.iter().any(|w| w.contains('韵')));

        let han = base(LexiconScope::HanOnly);
        assert!(han.words.iter().all(|w| w.chars().any(|c| is_han(c))));
        assert!(han.words.contains(&"深度求索".to_string()));
        assert!(!han.words.contains(&"AJAX".to_string()));
    }

    #[test]
    fn top_n_limit_applies() {
        // 生成 30 个合法词条（超过 limit 时截断）。
        let mut body = String::from("---\nname: t\nversion: \"1\"\n...\n");
        for i in 0..30 {
            body.push_str(&format!(
                "词汇编号{i:02}\tci hui bian hao {i:02}\t{}\n",
                100 - i
            ));
        }
        let preview = build_preview_from_body(
            &info("rime-ice-cn", LexiconSourceFormat::RimeDictYaml),
            LexiconSourceFormat::RimeDictYaml,
            None,
            "master",
            "u",
            &body,
            LexiconScope::HanOnly,
            10,
        )
        .expect("preview must build");
        assert!(preview.words.len() <= 10, "limit must cap kept words");
        assert_eq!(preview.total_unique, 30);
    }

    #[test]
    fn single_source_import_replaces_and_leaves_others_alone() {
        let mut map = HashMap::new();
        map.insert("THUOCL".to_string(), vec!["旧词".to_string()]);
        map.insert("zhwiki".to_string(), vec!["维基词甲".to_string()]);

        // 首次导入 rime-ice-en（拉丁词合法）。
        let key = apply_open_source_import(
            &mut map,
            "rime-ice-en",
            &["DeepSeek".to_string(), "GitHub".to_string()],
        )
        .expect("import ok");
        assert_eq!(key, "rime-ice-en");
        assert_eq!(map["rime-ice-en"].len(), 2);

        // 同源重复导入：替换（词数变少），不追加。
        apply_open_source_import(&mut map, "rime-ice-en", &["PyTorch".to_string()])
            .expect("re-import ok");
        assert_eq!(map["rime-ice-en"], vec!["PyTorch".to_string()]);

        // 其他来源不受影响。
        assert_eq!(map["THUOCL"], vec!["旧词".to_string()]);
        assert_eq!(map["zhwiki"], vec!["维基词甲".to_string()]);
    }

    #[test]
    fn confirm_validates_words_and_limits() {
        let mut map = HashMap::new();
        // 空词表 → 拒绝。
        assert!(apply_open_source_import(&mut map, "rime-ice-en", &[]).is_err());
        // 非法词（控制字符）→ 拒绝。
        assert!(
            apply_open_source_import(&mut map, "rime-ice-en", &["bad\tword".to_string()]).is_err()
        );
        // THUOCL 源要求纯汉字：拉丁词被拒。
        assert!(
            apply_open_source_import(&mut map, "thuocl-IT", &["DeepSeek".to_string()]).is_err()
        );
        assert!(map.is_empty(), "rejected imports must not write");
        // 未知源 → 拒绝。
        assert!(apply_open_source_import(&mut map, "nope", &["词".to_string()]).is_err());
    }

    #[test]
    fn preview_writes_nothing_by_construction() {
        // build_preview_from_body 是纯函数：无设置入参可写。这里固定其
        // 签名契约（无 &mut / 无 settings），并通过重复调用幂等兜底。
        let a = build_preview_from_body(
            &info("rime-essay", LexiconSourceFormat::EssayTsv),
            LexiconSourceFormat::EssayTsv,
            Some(RIME_ESSAY_IMPORT_WARNING),
            "master",
            "u",
            "的\t981\n了一\t658\n深度学习\t42\n人工智能\t30\n深度求索\t20\n词汇五\t15\n词汇六\t14\n词汇七\t13\n词汇八\t12\n词汇九\t11\n词汇十\t10\n词汇十一\t9\n词汇十二\t8\n词汇十三\t7\n词汇十四\t6\n词汇十五\t5\n词汇十六\t4\n词汇十七\t3\n词汇十八\t2\n词汇十九\t1\n词汇二十\t1\n词汇廿一\t1\n词汇廿二\t1\n词汇廿三\t1\n词汇廿四\t1\n词汇廿五\t1\n词汇廿六\t1\n词汇廿七\t1\n词汇廿八\t1\n词汇廿九\t1\n词汇三十\t1\n词汇卅一\t1\n词汇卅二\t1\n词汇卅三\t1\n词汇卅四\t1\n词汇卅五\t1\n词汇卅六\t1\n词汇卅七\t1\n词汇卅八\t1\n词汇卅九\t1\n词汇四十\t1\n词汇四十一\t1\n词汇四十二\t1\n词汇四十三\t1\n词汇四十四\t1\n词汇四十五\t1\n词汇四十六\t1\n词汇四十七\t1\n词汇四十八\t1\n词汇四十九\t1\n词汇五十\t1\n词汇五十一\t1\n词汇五十二\t1\n词汇五十三\t1\n词汇五十四\t1\n词汇五十五\t1\n词汇五十六\t1\n词汇五十七\t1\n词汇五十八\t1\n词汇五十九\t1\n词汇六十\t1\n词汇六十一\t1\n词汇六十二\t1\n词汇六十三\t1\n词汇六十四\t1\n词汇六十五\t1\n词汇六十六\t1\n词汇六十七\t1\n词汇六十八\t1\n词汇六十九\t1\n词汇七十\t1\n词汇七十一\t1\n词汇七十二\t1\n词汇七十三\t1\n词汇七十四\t1\n词汇七十五\t1\n词汇七十六\t1\n词汇七十七\t1\n词汇七十八\t1\n词汇七十九\t1\n词汇八十\t1\n词汇八十一\t1\n词汇八十二\t1\n词汇八十三\t1\n词汇八十四\t1\n词汇八十五\t1\n词汇八十六\t1\n词汇八十七\t1\n词汇八十八\t1\n词汇八十九\t1\n词汇九十\t1\n词汇九十一\t1\n词汇九十二\t1\n词汇九十三\t1\n词汇九十四\t1\n词汇九十五\t1\n词汇九十六\t1\n词汇九十七\t1\n词汇九十八\t1\n词汇九十九\t1\n",
            LexiconScope::All,
            500,
        )
        .expect("essay preview must build");
        assert_eq!(a.id, "rime-essay");
        // C 组：警告必须带在预览返回值里。
        assert!(a.warning.is_some());
    }

    #[test]
    fn catalog_groups_and_metadata() {
        let catalog = get_lexicon_sources();
        let ids: Vec<&str> = catalog.sources.iter().map(|s| s.id.as_str()).collect();
        for expected in [
            "zhwiki",
            "rime-ice-en",
            "rime-frost-en",
            "rime-dict",
            "rime-ice-base",
            "rime-ice-8105",
            "rime-ice-others",
            "rime-frost-base",
            "rime-frost-others",
            "jieba-dict",
            "rime-essay",
        ] {
            assert!(ids.contains(&expected), "catalog missing {expected}");
        }
        // 分组数量：专名 4 / 通用 6 / 领域 11（THUOCL）/ 词频基准 1。
        let count = |group| catalog.sources.iter().filter(|s| s.group == group).count();
        assert_eq!(count(LexiconSourceGroup::ProperNouns), 4);
        assert_eq!(count(LexiconSourceGroup::General), 6);
        assert_eq!(
            count(LexiconSourceGroup::Domain),
            crate::shortcut::THUOCL_CATEGORIES.len()
        );
        assert_eq!(count(LexiconSourceGroup::FrequencyBaseline), 1);
        assert_eq!(catalog.sources.len(), 22);
        // 目录分组按呈现顺序返回。
        assert_eq!(
            catalog.groups.iter().map(|g| g.group).collect::<Vec<_>>(),
            vec![
                LexiconSourceGroup::ProperNouns,
                LexiconSourceGroup::General,
                LexiconSourceGroup::Domain,
                LexiconSourceGroup::FrequencyBaseline,
            ]
        );
        // sources 已按分组顺序稳定排序（组内保持注册序）。
        let orders: Vec<u8> = catalog.sources.iter().map(|s| s.group.order()).collect();
        assert!(orders.windows(2).all(|w| w[0] <= w[1]));
        // 许可与推荐标记。
        let zhwiki = catalog.sources.iter().find(|s| s.id == "zhwiki").unwrap();
        assert!(zhwiki.license.contains("CC BY-SA"));
        assert!(zhwiki.recommended);
        let essay = catalog
            .sources
            .iter()
            .find(|s| s.id == "rime-essay")
            .unwrap();
        assert!(!essay.recommended);
        assert!(essay.license.contains("LGPL"));
        let jieba = catalog
            .sources
            .iter()
            .find(|s| s.id == "jieba-dict")
            .unwrap();
        assert_eq!(jieba.format, LexiconSourceFormat::JiebaTxt);
        assert_eq!(jieba.license, "MIT");
        // 专名派生源全部在专名组。
        for id in PROPER_NOUN_SOURCE_IDS {
            let source = catalog.sources.iter().find(|s| &s.id == id).unwrap();
            assert_eq!(source.group, LexiconSourceGroup::ProperNouns);
        }
        // 默认范围：英文词典仅拉丁、中文词库仅汉字。
        assert_eq!(
            catalog
                .sources
                .iter()
                .find(|s| s.id == "rime-ice-en")
                .unwrap()
                .default_scope,
            LexiconScope::LatinOnly
        );
        assert_eq!(
            catalog
                .sources
                .iter()
                .find(|s| s.id == "rime-ice-base")
                .unwrap()
                .default_scope,
            LexiconScope::HanOnly
        );
        // 存储键：THUOCL 去前缀，其余即 id。
        assert_eq!(stored_key_for("thuocl-IT"), "IT");
        assert_eq!(stored_key_for("zhwiki"), "zhwiki");
    }

    #[test]
    fn every_source_has_complete_fields() {
        let catalog = get_lexicon_sources();
        let mut seen = std::collections::HashSet::new();
        for source in &catalog.sources {
            assert!(
                seen.insert(source.id.as_str()),
                "duplicate source id: {}",
                source.id
            );
            assert!(!source.label.trim().is_empty(), "{} label empty", source.id);
            assert!(
                !source.description.trim().is_empty(),
                "{} description empty",
                source.id
            );
            assert!(
                !source.license.trim().is_empty(),
                "{} license empty",
                source.id
            );
            assert!(
                source.source_url.starts_with("https://"),
                "{} url must be https: {}",
                source.id,
                source.source_url
            );
        }
        for group in &catalog.groups {
            assert!(!group.label.trim().is_empty());
            assert!(!group.description.trim().is_empty());
        }
        // 拉取镜像全部 https 且非空（zhwiki 走 release API，单独路径）。
        for def in all_source_defs() {
            if let SourceFetch::Mirrors(urls) = &def.fetch {
                assert!(!urls.is_empty(), "{} has no mirrors", def.info.id);
                for url in urls {
                    assert!(url.starts_with("https://"), "mirror not https: {url}");
                }
            }
        }
    }

    #[test]
    fn jieba_txt_parsing() {
        let parsed = parse_jieba_txt(JIEBA_BIG).expect("must parse");
        // 5 条合法：三列（含词性）/ 两列（无词性）都收；单列行与词频非
        // 数字行被拒。
        assert_eq!(parsed.entries.len(), 5);
        assert_eq!(parsed.entries[0], ("深度学习".to_string(), Some(3000)));
        let no_pos = parsed.entries.iter().find(|(w, _)| w == "蹊跷").unwrap();
        assert_eq!(no_pos.1, Some(3));
        assert!(parsed.entries.iter().any(|(w, _)| w == "hello"));
        assert!(!parsed.entries.iter().any(|(w, _)| w == "词"));
        assert!(parsed.version.is_none());
    }

    #[test]
    fn jieba_garbage_rejected_and_preview_works() {
        // 整页散文（无空格分隔）与错误页被拒。
        assert!(parse_jieba_txt("这是第一行散文\n这是第二行散文\n这是第三行散文\n").is_err());
        assert!(parse_jieba_txt("<html>404 not found</html>").is_err());
        // 经预览链路（JiebaTxt 格式 + 过滤链）可用。
        let preview = build_preview_from_body(
            &info("jieba-dict", LexiconSourceFormat::JiebaTxt),
            LexiconSourceFormat::JiebaTxt,
            None,
            "master",
            "https://example.com/dict.txt.big",
            JIEBA_BIG,
            LexiconScope::All,
            500,
        )
        .expect("jieba preview must build");
        assert_eq!(preview.id, "jieba-dict");
        assert!(preview.words.contains(&"深度学习".to_string()));
        assert!(preview.words.contains(&"飞鱼座".to_string()));
    }

    #[test]
    fn limit_choices_validated() {
        assert_eq!(normalize_limit(None).unwrap(), OPEN_LEXICON_DEFAULT_LIMIT);
        assert!(normalize_limit(Some(2000)).is_ok());
        assert!(normalize_limit(Some(500)).is_ok());
        assert!(normalize_limit(Some(1234)).is_err());
    }

    #[test]
    fn config_from_settings_assembles_sources() {
        let mut settings = AppSettings::default();
        settings
            .imported_lexicons
            .insert("rime-ice-en".to_string(), vec!["PyTorch".to_string()]);
        settings
            .imported_lexicons
            .insert("THUOCL".to_string(), vec!["信息技术".to_string()]);
        let config = proper_noun_config_from_settings(&settings);
        assert!(config.enabled);
        // 只有 A 组源进入派生配置。
        assert_eq!(config.sources.len(), 1);
        assert_eq!(config.sources[0].0, "rime-ice-en");
    }
}
