import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface IncrementalCommitProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

/**
 * 增量上屏开关。
 *
 * 开启后，录音过程中已确定的文字会先出现：流式模型用引擎的 committed
 * 前缀，批处理模型用停顿分段。关闭时保持"停止后一次性粘贴"。
 */
export const IncrementalCommit: React.FC<IncrementalCommitProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const enabled = getSetting("incremental_commit") ?? false;

  return (
    <ToggleSwitch
      checked={enabled}
      onChange={(enabled) => updateSetting("incremental_commit", enabled)}
      isUpdating={isUpdating("incremental_commit")}
      label={t("settings.advanced.incrementalCommit.title")}
      description={t("settings.advanced.incrementalCommit.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
    />
  );
};
