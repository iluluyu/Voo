//! Tray menu internationalization
//!
//! Everything is auto-generated at compile time by build.rs from the
//! frontend locale files (src/i18n/locales/*/translation.json).
//!
//! The English translation.json is the single source of truth:
//! - TrayStrings struct fields are derived from the English "tray" keys
//! - All languages are auto-discovered from the locales directory
//!
//! To add a new tray menu item:
//! 1. Add the key to en/translation.json under "tray"
//! 2. Add translations to other locale files
//! 3. Update tray.rs to use the new field (e.g., strings.new_field)

use crate::chinese_script::chinese_script_for_locale;
use crate::settings::ChineseScript;
use once_cell::sync::Lazy;
use std::collections::HashMap;

// Include the auto-generated TrayStrings struct and TRANSLATIONS static
include!(concat!(env!("OUT_DIR"), "/tray_translations.rs"));

/// Get localized tray menu strings based on the system locale.
///
/// Lookup order: exact locale → Chinese script/region fallback → language code → English.
pub fn get_tray_translations(locale: Option<String>) -> TrayStrings {
    let normalized = locale
        .as_deref()
        .unwrap_or("en")
        .to_lowercase()
        .replace('_', "-");
    let language = normalized.split('-').next().unwrap_or("en");

    let exact_match = TRANSLATIONS
        .iter()
        .find_map(|(code, strings)| code.eq_ignore_ascii_case(&normalized).then_some(strings));
    let fallback = match chinese_script_for_locale(&normalized) {
        Some(ChineseScript::Traditional) => "zh-TW",
        Some(_) => "zh",
        None => language,
    };

    exact_match
        .or_else(|| TRANSLATIONS.get(fallback))
        .or_else(|| TRANSLATIONS.get("en"))
        .cloned()
        .expect("English translations must exist")
}

/// Strings for the tray "add hotwords from clipboard" flow (menu label,
/// dialogs). These live here instead of the generated `TrayStrings` set
/// because they are tray-only and the frontend locale files (the source of
/// the generated set) are out of scope for that change. Locale fallback
/// mirrors [`get_tray_translations`]: Traditional Chinese -> zh-TW, other
/// Chinese -> zh, everything else -> English.
pub(crate) struct HotwordsStrings {
    pub menu_label: &'static str,
    pub preview_title: &'static str,
    pub add: &'static str,
    pub cancel: &'static str,
    pub none_new: &'static str,
    pub clipboard_unreadable: &'static str,
    /// `added` (success message), `count_line` (preview header) and
    /// `more_line` (overflow note) take the count.
    pub added: fn(usize) -> String,
    pub count_line: fn(usize) -> String,
    pub more_line: fn(usize) -> String,
}

fn added_zh(n: usize) -> String {
    format!("已添加 {n} 个热词")
}
fn count_line_zh(n: usize) -> String {
    format!("共 {n} 个新热词：")
}
fn more_line_zh(m: usize) -> String {
    format!("（其余 {m} 个未显示）")
}

fn added_zh_tw(n: usize) -> String {
    format!("已加入 {n} 個熱詞")
}
fn count_line_zh_tw(n: usize) -> String {
    format!("共 {n} 個新熱詞：")
}
fn more_line_zh_tw(m: usize) -> String {
    format!("（其餘 {m} 個未顯示）")
}

fn added_en(n: usize) -> String {
    if n == 1 {
        "Added 1 hotword".to_string()
    } else {
        format!("Added {n} hotwords")
    }
}
fn count_line_en(n: usize) -> String {
    if n == 1 {
        "1 new hotword to add:".to_string()
    } else {
        format!("{n} new hotwords to add:")
    }
}
fn more_line_en(m: usize) -> String {
    format!("({m} more not shown)")
}

pub(crate) fn hotwords_strings(locale: Option<&str>) -> HotwordsStrings {
    let normalized = locale.unwrap_or("en").to_lowercase().replace('_', "-");
    match chinese_script_for_locale(&normalized) {
        Some(ChineseScript::Traditional) => HotwordsStrings {
            menu_label: "從剪貼簿加入熱詞…",
            preview_title: "加入熱詞",
            add: "加入",
            cancel: "取消",
            none_new: "剪貼簿中沒有新的熱詞",
            clipboard_unreadable: "無法讀取剪貼簿",
            added: added_zh_tw,
            count_line: count_line_zh_tw,
            more_line: more_line_zh_tw,
        },
        Some(_) => HotwordsStrings {
            menu_label: "从剪贴板添加热词…",
            preview_title: "添加热词",
            add: "添加",
            cancel: "取消",
            none_new: "剪贴板中没有新的热词",
            clipboard_unreadable: "无法读取剪贴板",
            added: added_zh,
            count_line: count_line_zh,
            more_line: more_line_zh,
        },
        None => HotwordsStrings {
            menu_label: "Add Hotwords from Clipboard…",
            preview_title: "Add Hotwords",
            add: "Add",
            cancel: "Cancel",
            none_new: "No new hotwords in the clipboard",
            clipboard_unreadable: "Could not read the clipboard",
            added: added_en,
            count_line: count_line_en,
            more_line: more_line_en,
        },
    }
}

/// Success message for the tray add-hotwords flow ("已添加 N 个热词").
pub(crate) fn hotwords_added_message(locale: Option<&str>, count: usize) -> String {
    (hotwords_strings(locale).added)(count)
}

#[cfg(test)]
mod tests {
    use super::{get_tray_translations, hotwords_added_message, hotwords_strings, TRANSLATIONS};

    #[test]
    fn resolves_locale_fallbacks() {
        for (locale, expected) in [
            ("zh-Hant-TW", "zh-TW"),
            ("zh-Hant-HK", "zh-TW"),
            ("zh-HK", "zh-TW"),
            ("zh-MO", "zh-TW"),
            ("ZH-TW", "zh-TW"),
            ("zh_Hant_TW", "zh-TW"),
            ("zh-Hans-CN", "zh"),
            ("yue-Hant-HK", "zh-TW"),
            ("yue-Hans-CN", "zh"),
            ("fr-FR", "fr"),
            ("xx-YY", "en"),
        ] {
            assert_eq!(
                format!("{:?}", get_tray_translations(Some(locale.into()))),
                format!("{:?}", TRANSLATIONS[expected]),
                "{locale} should resolve to {expected}"
            );
        }
    }

    #[test]
    fn hotwords_strings_follow_chinese_script_fallback() {
        assert_eq!(
            hotwords_strings(Some("zh-Hant-TW")).menu_label,
            "從剪貼簿加入熱詞…"
        );
        assert_eq!(
            hotwords_strings(Some("zh-CN")).menu_label,
            "从剪贴板添加热词…"
        );
        assert_eq!(
            hotwords_strings(Some("en-US")).menu_label,
            "Add Hotwords from Clipboard…"
        );
        // Non-Chinese locales without tray-only translations fall back to
        // English, like untranslated generated keys do.
        assert_eq!(
            hotwords_strings(Some("fr-FR")).menu_label,
            "Add Hotwords from Clipboard…"
        );
        assert_eq!(
            hotwords_strings(None).menu_label,
            "Add Hotwords from Clipboard…"
        );
    }

    #[test]
    fn hotwords_added_message_carries_count() {
        assert_eq!(hotwords_added_message(Some("zh"), 3), "已添加 3 个热词");
        assert_eq!(hotwords_added_message(Some("en"), 1), "Added 1 hotword");
        assert_eq!(hotwords_added_message(Some("en"), 2), "Added 2 hotwords");
    }
}
