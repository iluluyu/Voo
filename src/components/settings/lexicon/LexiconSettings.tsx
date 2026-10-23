import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { commands } from "@/bindings";
import { useSettings } from "../../../hooks/useSettings";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { ToggleSwitch } from "../../ui/ToggleSwitch";
import { Button } from "../../ui/Button";
import { CustomWords } from "../CustomWords";
import { HotwordRules } from "../HotwordRules";
import { LexiconImport } from "../LexiconImport";
import { LexiconSources } from "./LexiconSources";
import { RecentTermCandidates } from "../RecentTermCandidates";
import { LearnedCandidates } from "../LearnedCandidates";
import { LlmLexiconCandidates } from "../LlmLexiconCandidates";

/** `preview_lexicon_curation` 是否存在于当前 bindings（与
 * LlmLexiconCandidates 同款守卫：命令未生成时隐藏手动预览入口）。 */
const CURATION_COMMAND_SUPPORTED =
  typeof commands.previewLexiconCuration === "function";

/**
 * 「词库」设置页：集中所有词库相关内容（原来散落在「高级」里的六块）。
 *
 * 1) 自定义词汇 + 中文热词规则
 * 2) 已安装来源 + 开源词库源导入
 * 3) 候选三块（最近常出现 / 已学习 / LLM 候选）
 * 4) AI 整理：开关 `post_process_lexicon_curation_enabled`、触发策略与
 *    节流说明、手动「立即整理」（只读预览当前候选数；真正的定时整理由
 *    后端在启动后与每次转写落库后检查触发，见 lexicon_curation.rs）。
 */
export const LexiconSettings: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const [running, setRunning] = useState(false);
  const [candidateCount, setCandidateCount] = useState<number | null>(null);

  const curationEnabled =
    getSetting("post_process_lexicon_curation_enabled") ?? true;

  const runNow = async () => {
    if (!CURATION_COMMAND_SUPPORTED || running) return;
    setRunning(true);
    try {
      const result = await commands.previewLexiconCuration();
      if (result.status === "error") {
        throw new Error(result.error);
      }
      setCandidateCount(result.data.length);
    } catch (e) {
      toast.error(
        t("settings.lexicon.curation.error", {
          message: e instanceof Error ? e.message : String(e),
        }),
      );
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.lexicon.groups.words")}>
        <CustomWords descriptionMode="tooltip" grouped />
        <HotwordRules descriptionMode="tooltip" grouped />
      </SettingsGroup>

      <SettingsGroup title={t("settings.lexicon.groups.sources")}>
        <LexiconSources descriptionMode="tooltip" grouped />
        <LexiconImport descriptionMode="tooltip" grouped />
      </SettingsGroup>

      <SettingsGroup title={t("settings.lexicon.groups.candidates")}>
        <RecentTermCandidates descriptionMode="tooltip" grouped />
        <LearnedCandidates descriptionMode="tooltip" grouped />
        <LlmLexiconCandidates descriptionMode="tooltip" grouped />
      </SettingsGroup>

      <SettingsGroup title={t("settings.lexicon.groups.curation")}>
        <ToggleSwitch
          checked={curationEnabled}
          onChange={(enabled) =>
            updateSetting("post_process_lexicon_curation_enabled", enabled)
          }
          isUpdating={isUpdating("post_process_lexicon_curation_enabled")}
          label={t("settings.lexicon.curation.label")}
          description={t("settings.lexicon.curation.description")}
          descriptionMode="tooltip"
          grouped
        />
        <div className="space-y-1 px-3 py-2 text-sm text-text/60">
          <p>{t("settings.lexicon.curation.strategy")}</p>
          <p>{t("settings.lexicon.curation.throttle")}</p>
        </div>
        {CURATION_COMMAND_SUPPORTED && (
          <div className="flex items-center gap-3 px-3 py-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void runNow()}
              disabled={running}
            >
              {running
                ? t("settings.lexicon.curation.running")
                : t("settings.lexicon.curation.runNow")}
            </Button>
            {candidateCount !== null && !running && (
              <span className="text-sm text-text/60">
                {t("settings.lexicon.curation.count", {
                  count: candidateCount,
                })}
              </span>
            )}
          </div>
        )}
      </SettingsGroup>
    </div>
  );
};
