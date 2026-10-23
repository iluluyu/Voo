import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface DecodeVocabularyBiasProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
}

/**
 * 解码期词表偏置开关。
 *
 * 开启时热词会作为 vocabulary 交给引擎（模型在解码阶段就倾向识别这些词）。
 * 某些模型对偏置敏感（反馈：Qwen3-ASR 会输出无关热词或重复），关闭后只保留
 * 拼音后处理纠错。
 */
export const DecodeVocabularyBias: React.FC<DecodeVocabularyBiasProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const enabled = getSetting("decode_vocabulary_bias") ?? true;

  return (
    <ToggleSwitch
      checked={enabled}
      onChange={(enabled) => updateSetting("decode_vocabulary_bias", enabled)}
      isUpdating={isUpdating("decode_vocabulary_bias")}
      label={t("settings.advanced.decodeVocabularyBias.title")}
      description={t("settings.advanced.decodeVocabularyBias.description")}
      descriptionMode={descriptionMode}
      grouped={grouped}
    />
  );
};
