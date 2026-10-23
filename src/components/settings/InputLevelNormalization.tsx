import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface InputLevelNormalizationProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

/**
 * 输入电平调理开关。
 *
 * 增益作用在**送进 VAD 的同一份样本**上，因此轻声说话不会被 VAD 当噪声丢掉；
 * 内部有底噪 min-stat 与 -3 dBFS 限幅，静音期不提升增益。
 */
export const InputLevelNormalization: React.FC<
  InputLevelNormalizationProps
> = ({ descriptionMode = "tooltip", grouped = false }) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const enabled = getSetting("input_level_normalization") ?? true;

  return (
    <ToggleSwitch
      checked={enabled}
      onChange={(enabled) =>
        updateSetting("input_level_normalization", enabled)
      }
      isUpdating={isUpdating("input_level_normalization")}
      label={t("settings.advanced.inputLevelNormalization.title")}
      description={t("settings.advanced.inputLevelNormalization.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
    />
  );
};
