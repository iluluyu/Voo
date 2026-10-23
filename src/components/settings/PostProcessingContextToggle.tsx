import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

/**
 * LLM 后处理的聚焦上下文开关（post_process_context_enabled，默认开）。
 * 仅在后处理启用时生效；录音开始前一次性采集焦点 App 标识与（平台可
 * 行时）游标前 ≤128 字符，作为不可信数据字段发给所配置的 LLM 服务商。
 */
export const PostProcessingContextToggle: React.FC = React.memo(() => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const enabled = getSetting("post_process_context_enabled") ?? true;

  return (
    <ToggleSwitch
      checked={enabled}
      onChange={(enabled) =>
        updateSetting("post_process_context_enabled", enabled)
      }
      isUpdating={isUpdating("post_process_context_enabled")}
      label={t("settings.postProcessing.context.toggle.label")}
      description={t("settings.postProcessing.context.toggle.description")}
      descriptionMode="tooltip"
      grouped={true}
    />
  );
});
