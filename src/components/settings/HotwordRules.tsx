import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../hooks/useSettings";
import { SettingContainer } from "../ui/SettingContainer";
import { Textarea } from "../ui/Textarea";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { Input } from "../ui/Input";

interface HotwordRulesProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * Voo 中文热词系统设置：
 * - 规则文本（每行 `模式=替换` 字面量 / `模式=>替换` 正则）
 * - 拼音匹配阈值（越大越保守）
 * - 方言混淆开关（n/l、f/h）
 *
 * 拼音热词本身复用 Custom Words 列表（含汉字的词自动进入拼音索引）。
 */
export const HotwordRules: React.FC<HotwordRulesProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const savedRules = getSetting("hotword_rules") ?? "";
    const threshold = getSetting("hotword_pinyin_threshold") ?? 0.85;
    const dialect = getSetting("hotword_dialect_confusion") ?? false;

    // textarea 本地编辑，失焦或 Ctrl/Cmd+Enter 时提交，避免每次击键写库
    const [draft, setDraft] = useState(savedRules);
    const [dirty, setDirty] = useState(false);
    useEffect(() => {
      if (!dirty) setDraft(savedRules);
    }, [savedRules, dirty]);

    const commit = () => {
      if (dirty) {
        updateSetting("hotword_rules", draft);
        setDirty(false);
      }
    };

    return (
      <>
        <SettingContainer
          title={t("settings.advanced.hotwordRules.title")}
          description={t("settings.advanced.hotwordRules.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        >
          <div className="w-full max-w-md">
            <Textarea
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value);
                setDirty(true);
              }}
              onBlur={commit}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  e.preventDefault();
                  commit();
                }
              }}
              rows={5}
              className="font-mono text-xs"
              placeholder={t("settings.advanced.hotwordRules.placeholder")}
              spellCheck={false}
              disabled={isUpdating("hotword_rules")}
            />
            <div className="mt-1 flex items-center justify-between">
              <span className="text-xs text-mid-gray">
                {t("settings.advanced.hotwordRules.hint")}
              </span>
              {dirty && (
                <button
                  type="button"
                  onClick={commit}
                  className="text-xs underline cursor-pointer"
                >
                  {t("settings.advanced.hotwordRules.save")}
                </button>
              )}
            </div>
          </div>
        </SettingContainer>

        <SettingContainer
          title={t("settings.advanced.hotwordThreshold.title")}
          description={t("settings.advanced.hotwordThreshold.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        >
          <Input
            type="number"
            className="max-w-24"
            value={threshold}
            min={0.5}
            max={1.0}
            step={0.05}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (!Number.isNaN(v) && v >= 0.5 && v <= 1.0) {
                updateSetting("hotword_pinyin_threshold", v);
              }
            }}
            variant="compact"
            disabled={isUpdating("hotword_pinyin_threshold")}
          />
        </SettingContainer>

        <ToggleSwitch
          checked={dialect}
          onChange={(v) => updateSetting("hotword_dialect_confusion", v)}
          isUpdating={isUpdating("hotword_dialect_confusion")}
          label={t("settings.advanced.hotwordDialect.label")}
          description={t("settings.advanced.hotwordDialect.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        />
      </>
    );
  },
);
