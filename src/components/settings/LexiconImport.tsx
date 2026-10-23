import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { commands } from "@/bindings";
import { useSettings } from "../../hooks/useSettings";
import { Button } from "../ui/Button";
import { SettingContainer } from "../ui/SettingContainer";

/**
 * THUOCL data/ 目录实际存在的分类文件名
 * （与 Rust 端 shortcut::THUOCL_CATEGORIES 对应，实测 11 个分类，
 * README 里的 medicial/econ 等旧名已 404）。
 */
const THUOCL_CATEGORIES = [
  "IT",
  "animal",
  "caijing",
  "car",
  "chengyu",
  "diming",
  "food",
  "law",
  "lishimingren",
  "medical",
  "poem",
] as const;

type ThuoclCategory = (typeof THUOCL_CATEGORIES)[number];

/** 预览里最多展示的样例词数 */
const SAMPLE_WORDS = 40;

interface LexiconImportProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * THUOCL 清华开放中文词库一键导入：
 * 勾选分类 → 下载并过滤（纯汉字、按词频取前 5000）→ 预览将追加的词数
 * → 确认后追加到 custom_words（进入拼音热词索引）。
 */
export const LexiconImport: React.FC<LexiconImportProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const [selected, setSelected] = useState<Set<ThuoclCategory>>(new Set());
    const [fetching, setFetching] = useState(false);
    const [preview, setPreview] = useState<string[] | null>(null);
    const [error, setError] = useState<string | null>(null);

    const customWords: string[] = getSetting("custom_words") || [];

    const toggleCategory = (category: ThuoclCategory) => {
      setPreview(null);
      setError(null);
      setSelected((prev) => {
        const next = new Set(prev);
        if (next.has(category)) {
          next.delete(category);
        } else {
          next.add(category);
        }
        return next;
      });
    };

    const handleFetch = async () => {
      if (selected.size === 0 || fetching) return;
      setFetching(true);
      setError(null);
      setPreview(null);
      try {
        const results = await Promise.all(
          [...selected].map((category) =>
            commands.importThuoclLexicon(category),
          ),
        );
        // 跨分类去重，并与现有 custom_words 去重
        const seen = new Set(customWords);
        const merged: string[] = [];
        for (const result of results) {
          if (result.status === "error") {
            throw new Error(result.error);
          }
          for (const word of result.data) {
            if (!seen.has(word)) {
              seen.add(word);
              merged.push(word);
            }
          }
        }
        setPreview(merged);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setFetching(false);
      }
    };

    const handleConfirm = () => {
      if (!preview || preview.length === 0) return;
      updateSetting("custom_words", [...customWords, ...preview]);
      toast.success(
        t("settings.advanced.lexiconImport.success", {
          count: preview.length,
        }),
      );
      setPreview(null);
      setSelected(new Set());
    };

    const handleCancel = () => {
      setPreview(null);
      setError(null);
    };

    return (
      <SettingContainer
        title={t("settings.advanced.lexiconImport.title")}
        description={t("settings.advanced.lexiconImport.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      >
        <div className="w-full max-w-md space-y-2">
          <div className="flex flex-wrap gap-1">
            {THUOCL_CATEGORIES.map((category) => (
              <Button
                key={category}
                onClick={() => toggleCategory(category)}
                disabled={fetching || isUpdating("custom_words")}
                variant={selected.has(category) ? "primary" : "secondary"}
                size="sm"
                aria-pressed={selected.has(category)}
              >
                {t(`settings.advanced.lexiconImport.categories.${category}`)}
              </Button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <Button
              onClick={handleFetch}
              disabled={selected.size === 0 || fetching}
              variant="primary"
              size="md"
            >
              {fetching
                ? t("settings.advanced.lexiconImport.fetching")
                : t("settings.advanced.lexiconImport.fetch")}
            </Button>
            {selected.size === 0 && !fetching && (
              <span className="text-xs text-mid-gray">
                {t("settings.advanced.lexiconImport.selectHint")}
              </span>
            )}
          </div>
          {error && (
            <p className="text-xs text-red-400">
              {t("settings.advanced.lexiconImport.error", { message: error })}
            </p>
          )}
          {preview && (
            <div className="space-y-2 rounded-lg border border-mid-gray/20 p-2">
              <p className="text-xs text-mid-gray">
                {preview.length > 0
                  ? t("settings.advanced.lexiconImport.preview", {
                      count: preview.length,
                    })
                  : t("settings.advanced.lexiconImport.emptyResult")}
              </p>
              {preview.length > 0 && (
                <p className="text-xs text-mid-gray/80">
                  {preview.slice(0, SAMPLE_WORDS).join("、")}
                  {preview.length > SAMPLE_WORDS && " …"}
                </p>
              )}
              {preview.length > 0 && (
                <div className="flex items-center gap-2">
                  <Button
                    onClick={handleConfirm}
                    disabled={isUpdating("custom_words")}
                    variant="primary"
                    size="sm"
                  >
                    {t("settings.advanced.lexiconImport.append")}
                  </Button>
                  <Button
                    onClick={handleCancel}
                    disabled={isUpdating("custom_words")}
                    variant="secondary"
                    size="sm"
                  >
                    {t("settings.advanced.lexiconImport.cancel")}
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      </SettingContainer>
    );
  },
);
