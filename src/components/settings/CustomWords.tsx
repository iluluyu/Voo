import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { useSettings } from "../../hooks/useSettings";
import { Input } from "../ui/Input";
import { Button } from "../ui/Button";
import { SettingContainer } from "../ui/SettingContainer";

interface CustomWordsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/** 一次最多渲染多少条（大词表时避免上万 DOM 节点卡死界面）。 */
const RENDER_LIMIT = 200;

const normalizeCustomWord = (word: string) =>
  word
    .replace(/[<>"']/g, "")
    .replace(/\s+/g, " ")
    .trim();

interface BulkPreview {
  added: string[];
  duplicates: number;
  invalid: number;
}

/**
 * 自定义词汇：可按行批量添加、搜索、分页渲染与导出。
 *
 * 之前的版本只有"一条条加"和"全量渲染 chips"，词条上千后既难用也卡；
 * 这里保留单条添加入口，同时提供多行粘贴（一行一条，支持
 * `目标|别名~~~黑名单` 语法），并只渲染前 200 条。
 */
export const CustomWords: React.FC<CustomWordsProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const [newWord, setNewWord] = useState("");
    const [search, setSearch] = useState("");
    const [bulkText, setBulkText] = useState<string | null>(null);
    const [visible, setVisible] = useState(RENDER_LIMIT);

    const customWords = getSetting("custom_words") || [];
    const normalizedWord = normalizeCustomWord(newWord);
    const updating = isUpdating("custom_words");

    const filtered = useMemo(() => {
      const query = search.trim().toLowerCase();
      if (!query) return customWords;
      return customWords.filter((word) => word.toLowerCase().includes(query));
    }, [customWords, search]);

    const preview: BulkPreview | null = useMemo(() => {
      if (bulkText === null) return null;
      const seen = new Set(customWords);
      const added: string[] = [];
      let duplicates = 0;
      let invalid = 0;
      for (const rawLine of bulkText.split(/\r?\n/)) {
        const line = normalizeCustomWord(rawLine);
        if (!line) continue;
        if (line.length > 80) {
          invalid += 1;
          continue;
        }
        if (seen.has(line)) {
          duplicates += 1;
          continue;
        }
        seen.add(line);
        added.push(line);
      }
      return { added, duplicates, invalid };
    }, [bulkText, customWords]);

    const handleAddWord = () => {
      if (normalizedWord && normalizedWord.length <= 50) {
        if (customWords.includes(normalizedWord)) {
          toast.error(
            t("settings.advanced.customWords.duplicate", {
              word: normalizedWord,
            }),
          );
          return;
        }
        updateSetting("custom_words", [...customWords, normalizedWord]);
        setNewWord("");
      }
    };

    const handleBulkConfirm = () => {
      if (!preview || preview.added.length === 0) return;
      updateSetting("custom_words", [...customWords, ...preview.added]);
      toast.success(
        t("settings.advanced.customWords.bulkDone", {
          count: preview.added.length,
        }),
      );
      setBulkText(null);
    };

    const handleRemoveWord = (wordToRemove: string) => {
      updateSetting(
        "custom_words",
        customWords.filter((word) => word !== wordToRemove),
      );
    };

    const handleExport = () => {
      const blob = new Blob([customWords.join("\n")], {
        type: "text/plain;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "custom-words.txt";
      anchor.click();
      URL.revokeObjectURL(url);
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleAddWord();
      }
    };

    const shown = filtered.slice(0, visible);
    const remaining = filtered.length - shown.length;

    return (
      <>
        <SettingContainer
          title={t("settings.advanced.customWords.title")}
          description={t("settings.advanced.customWords.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        >
          <div className="flex flex-wrap items-center gap-2">
            <Input
              type="text"
              className="max-w-40"
              value={newWord}
              onChange={(e) => setNewWord(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={t("settings.advanced.customWords.placeholder")}
              variant="compact"
              disabled={updating}
            />
            <Button
              onClick={handleAddWord}
              disabled={
                !normalizedWord || normalizedWord.length > 50 || updating
              }
              variant="primary"
              size="md"
            >
              {t("settings.advanced.customWords.add")}
            </Button>
            <Button
              onClick={() => setBulkText(bulkText === null ? "" : null)}
              disabled={updating}
              variant="secondary"
              size="md"
            >
              {t("settings.advanced.customWords.bulkAdd")}
            </Button>
            <Button
              onClick={handleExport}
              disabled={customWords.length === 0}
              variant="secondary"
              size="md"
            >
              {t("settings.advanced.customWords.export")}
            </Button>
            <span className="text-xs text-mid-gray">
              {t("settings.advanced.customWords.count", {
                count: customWords.length,
              })}
            </span>
          </div>

          {bulkText !== null && (
            <div className="mt-2 w-full space-y-2">
              <textarea
                className="h-32 w-full resize-y rounded border border-mid-gray/20 bg-transparent p-2 text-xs"
                value={bulkText}
                placeholder={t("settings.advanced.customWords.bulkPlaceholder")}
                onChange={(e) => setBulkText(e.target.value)}
                disabled={updating}
              />
              {preview && (
                <p className="text-xs text-mid-gray">
                  {t("settings.advanced.customWords.bulkPreview", {
                    added: preview.added.length,
                    duplicates: preview.duplicates,
                    invalid: preview.invalid,
                  })}
                </p>
              )}
              <div className="flex items-center gap-2">
                <Button
                  onClick={handleBulkConfirm}
                  disabled={!preview || preview.added.length === 0 || updating}
                  variant="primary"
                  size="sm"
                >
                  {t("settings.advanced.customWords.bulkConfirm")}
                </Button>
                <Button
                  onClick={() => setBulkText(null)}
                  disabled={updating}
                  variant="secondary"
                  size="sm"
                >
                  {t("settings.advanced.customWords.bulkCancel")}
                </Button>
              </div>
            </div>
          )}

          {customWords.length > 0 && (
            <div className="mt-2 w-full">
              <Input
                type="text"
                className="w-full"
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setVisible(RENDER_LIMIT);
                }}
                placeholder={t(
                  "settings.advanced.customWords.searchPlaceholder",
                )}
                variant="compact"
              />
            </div>
          )}
        </SettingContainer>

        {filtered.length > 0 && (
          <div
            className={`px-4 p-2 ${grouped ? "" : "rounded-lg border border-mid-gray/20"} max-h-64 flex flex-wrap gap-1 overflow-auto`}
          >
            {shown.map((word) => (
              <Button
                key={word}
                onClick={() => handleRemoveWord(word)}
                disabled={updating}
                variant="secondary"
                size="sm"
                className="inline-flex items-center gap-1 cursor-pointer"
                aria-label={t("settings.advanced.customWords.remove", { word })}
              >
                <span className="max-w-52 truncate">{word}</span>
                <svg
                  className="w-3 h-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </Button>
            ))}
            {remaining > 0 && (
              <Button
                onClick={() => setVisible((v) => v + RENDER_LIMIT)}
                variant="secondary"
                size="sm"
              >
                {t("settings.advanced.customWords.showMore", {
                  count: remaining,
                })}
              </Button>
            )}
          </div>
        )}
      </>
    );
  },
);
