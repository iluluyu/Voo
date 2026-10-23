import React, { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { toast } from "sonner";
import { commands } from "@/bindings";
import { useSettings } from "../../hooks/useSettings";
import { Button } from "../ui/Button";
import { SettingContainer } from "../ui/SettingContainer";
import { saveSourcesSequentially } from "./lexiconSources";

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

interface PreviewState {
  /** 每个所选分类各自的词表（保存时按来源逐个写入快照） */
  lists: Partial<Record<ThuoclCategory, string[]>>;
  /** 跨分类去重后的预览总数与样例 */
  merged: string[];
}

/**
 * THUOCL 清华开放中文词库来源管理：
 * 勾选分类 → 下载预览（纯汉字 2–50 字、按词频取前 5000）→ 确认后保存为
 * `imported_lexicons` 来源快照（可按来源更新/移除）。
 *
 * 导入词不进入 `custom_words`（手工词），两者由后端
 * `effective_custom_words()` 合并生效。所有命令结果都显式检查
 * `status`：失败只显示错误，绝不 toast 成功；请求 id 递增使旧的
 * 异步结果失效，避免覆盖用户新的选择。
 */
export const LexiconImport: React.FC<LexiconImportProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting } = useSettings();

    const [selected, setSelected] = useState<Set<ThuoclCategory>>(new Set());
    const [fetching, setFetching] = useState(false);
    const [preview, setPreview] = useState<PreviewState | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);
    const [removing, setRemoving] = useState<ThuoclCategory | null>(null);
    /** 待确认的移除目标：第一次点击不删除，避免误触丢掉整个词库快照 */
    const [pendingRemove, setPendingRemove] = useState<ThuoclCategory | null>(
      null,
    );

    /** 每次发起新交互时递增；异步回调若发现 id 已变则丢弃结果 */
    const requestIdRef = useRef(0);

    const importedLexicons = getSetting("imported_lexicons") || {};
    const installedSources = THUOCL_CATEGORIES.filter(
      (category) => importedLexicons[category],
    );

    const toggleCategory = (category: ThuoclCategory) => {
      requestIdRef.current += 1;
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
      if (selected.size === 0 || fetching || saving || removing !== null)
        return;
      const requestId = ++requestIdRef.current;
      setFetching(true);
      setError(null);
      setPreview(null);
      try {
        const results = await Promise.all(
          [...selected].map(async (category) => {
            const result = await commands.importThuoclLexicon(category);
            if (result.status === "error") {
              throw new Error(result.error);
            }
            return [category, result.data] as const;
          }),
        );
        if (requestId !== requestIdRef.current) return;
        // 跨分类去重（仅用于预览展示；保存时各来源保留各自完整词表）
        const seen = new Set<string>();
        const merged: string[] = [];
        const lists: PreviewState["lists"] = {};
        for (const [category, words] of results) {
          lists[category] = words;
          for (const word of words) {
            if (!seen.has(word)) {
              seen.add(word);
              merged.push(word);
            }
          }
        }
        setPreview({ lists, merged });
      } catch (e) {
        if (requestId !== requestIdRef.current) return;
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (requestId === requestIdRef.current) {
          setFetching(false);
        }
      }
    };

    /**
     * 离线兜底：网络完全不可用时，用户手动下载 THUOCL_<分类>.txt 后在这里导入。
     * 只允许在已选中一个分类时使用（一个文件对应一个分类），校验在后端。
     */
    const handleLocalFile = async () => {
      if (selected.size !== 1 || fetching || saving || removing !== null)
        return;
      const [category] = [...selected];
      const requestId = ++requestIdRef.current;
      setFetching(true);
      setError(null);
      setPreview(null);
      try {
        const picked = await openDialog({
          multiple: false,
          directory: false,
          filters: [{ name: "THUOCL", extensions: ["txt"] }],
        });
        if (typeof picked !== "string") {
          if (requestId === requestIdRef.current) setFetching(false);
          return;
        }
        const result = await commands.importThuoclLexiconFile(category, picked);
        if (requestId !== requestIdRef.current) return;
        if (result.status === "error") {
          throw new Error(result.error);
        }
        const words = result.data;
        setPreview({ lists: { [category]: words }, merged: words });
      } catch (e) {
        if (requestId !== requestIdRef.current) return;
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (requestId === requestIdRef.current) {
          setFetching(false);
        }
      }
    };

    const copyError = async () => {
      if (!error) return;
      try {
        await navigator.clipboard.writeText(error);
        toast.success(t("settings.advanced.lexiconImport.copied"));
      } catch {
        // 复制失败时保持原文可见，用户可手动选中
      }
    };

    const handleConfirm = async () => {
      if (!preview || saving || removing !== null || fetching) return;
      const requestId = ++requestIdRef.current;
      setSaving(true);
      setError(null);
      try {
        const entries = Object.entries(preview.lists) as [
          ThuoclCategory,
          string[],
        ][];
        // 后端每个 set_imported_lexicon 都是整份设置的读-改-写，
        // 必须串行 await；并发会互相覆盖丢来源。部分失败不中断，
        // 由 saved/failed 结果分别可见地报告。
        const { saved, failed } = await saveSourcesSequentially(
          entries,
          (source, words) => commands.setImportedLexicon(source, words),
        );
        if (requestId !== requestIdRef.current) return;
        if (failed.length > 0) {
          const detail = failed
            .map(
              (f) =>
                `${t(`settings.advanced.lexiconImport.categories.${f.source}`)}: ${f.error}`,
            )
            .join("; ");
          setError(detail);
          toast.error(
            t("settings.advanced.lexiconImport.saveError", {
              message: detail,
            }),
          );
          // 部分成功也要可见：已保存来源的词数单独报成功
          if (saved.length > 0) {
            const savedCount = saved.reduce(
              (sum, source) => sum + (preview.lists[source]?.length ?? 0),
              0,
            );
            toast.success(
              t("settings.advanced.lexiconImport.success", {
                count: savedCount,
              }),
            );
          }
          return;
        }
        // 后端已广播 settings-changed，store 会自动重拉来源列表
        toast.success(
          t("settings.advanced.lexiconImport.success", {
            count: preview.merged.length,
          }),
        );
        setPreview(null);
        setSelected(new Set());
      } catch (e) {
        if (requestId !== requestIdRef.current) return;
        const message = e instanceof Error ? e.message : String(e);
        setError(message);
        toast.error(
          t("settings.advanced.lexiconImport.saveError", { message }),
        );
      } finally {
        if (requestId === requestIdRef.current) {
          setSaving(false);
        }
      }
    };

    const handleRemoveSource = async (category: ThuoclCategory) => {
      if (removing || saving) return;
      const requestId = ++requestIdRef.current;
      // 移除立即失效进行中的预览（它可能还包含该来源）
      setPreview(null);
      setError(null);
      setSelected((prev) => {
        const next = new Set(prev);
        next.delete(category);
        return next;
      });
      setRemoving(category);
      try {
        const result = await commands.removeImportedLexicon(category);
        if (result.status === "error") {
          throw new Error(result.error);
        }
      } catch (e) {
        if (requestId !== requestIdRef.current) return;
        const message = e instanceof Error ? e.message : String(e);
        setError(t("settings.advanced.lexiconImport.removeError", { message }));
        toast.error(
          t("settings.advanced.lexiconImport.removeError", { message }),
        );
      } finally {
        if (requestId === requestIdRef.current) {
          setRemoving(null);
        }
      }
    };

    const handleCancel = () => {
      requestIdRef.current += 1;
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
          {installedSources.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs text-mid-gray">
                {t("settings.advanced.lexiconImport.installed")}
              </p>
              <div className="flex flex-wrap gap-1">
                {installedSources.map((category) => (
                  <div
                    key={category}
                    className="inline-flex items-center gap-1"
                  >
                    <Button
                      onClick={() => setPendingRemove(category)}
                      disabled={removing !== null || saving || fetching}
                      variant="secondary"
                      size="sm"
                      className="inline-flex items-center gap-1"
                      aria-label={t(
                        "settings.advanced.lexiconImport.removeSource",
                        {
                          source: t(
                            `settings.advanced.lexiconImport.categories.${category}`,
                          ),
                        },
                      )}
                    >
                      <span>
                        {t(
                          `settings.advanced.lexiconImport.categories.${category}`,
                        )}
                        {removing === category
                          ? "…"
                          : ` (${t(
                              "settings.advanced.lexiconImport.wordCount",
                              {
                                count: importedLexicons[category]?.length ?? 0,
                              },
                            )})`}
                      </span>
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
                    {pendingRemove === category && (
                      <span className="inline-flex items-center gap-1 rounded border border-red-400/40 px-1 py-0.5">
                        <span className="text-xs text-red-400">
                          {t("settings.advanced.lexiconImport.removeConfirm", {
                            source: t(
                              `settings.advanced.lexiconImport.categories.${category}`,
                            ),
                          })}
                        </span>
                        <Button
                          onClick={() => {
                            setPendingRemove(null);
                            void handleRemoveSource(category);
                          }}
                          disabled={removing !== null}
                          variant="primary"
                          size="sm"
                        >
                          {t(
                            "settings.advanced.lexiconImport.removeConfirmYes",
                          )}
                        </Button>
                        <Button
                          onClick={() => setPendingRemove(null)}
                          disabled={removing !== null}
                          variant="secondary"
                          size="sm"
                        >
                          {t("settings.advanced.lexiconImport.cancel")}
                        </Button>
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="flex flex-wrap gap-1">
            {THUOCL_CATEGORIES.map((category) => (
              <Button
                key={category}
                onClick={() => toggleCategory(category)}
                disabled={fetching || saving || removing !== null}
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
              disabled={
                selected.size === 0 || fetching || saving || removing !== null
              }
              variant="primary"
              size="md"
            >
              {fetching
                ? t("settings.advanced.lexiconImport.fetching")
                : t("settings.advanced.lexiconImport.fetch")}
            </Button>
            <Button
              onClick={handleLocalFile}
              disabled={
                selected.size !== 1 || fetching || saving || removing !== null
              }
              variant="secondary"
              size="md"
            >
              {t("settings.advanced.lexiconImport.fromFile")}
            </Button>
            {selected.size === 0 && !fetching && (
              <span className="text-xs text-mid-gray">
                {t("settings.advanced.lexiconImport.selectHint")}
              </span>
            )}
          </div>
          {error && (
            <div className="space-y-1">
              <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded border border-mid-gray/20 bg-black/10 p-2 text-xs text-red-400 select-text">
                {t("settings.advanced.lexiconImport.error", {
                  message: error,
                })}
              </pre>
              <Button
                onClick={copyError}
                variant="secondary"
                size="sm"
                aria-label={t("settings.advanced.lexiconImport.copyError")}
              >
                {t("settings.advanced.lexiconImport.copyError")}
              </Button>
            </div>
          )}
          {preview && (
            <div className="space-y-2 rounded-lg border border-mid-gray/20 p-2">
              <p className="text-xs text-mid-gray">
                {preview.merged.length > 0
                  ? t("settings.advanced.lexiconImport.preview", {
                      count: preview.merged.length,
                    })
                  : t("settings.advanced.lexiconImport.emptyResult")}
              </p>
              {preview.merged.length > 0 && (
                <p className="text-xs text-mid-gray/80">
                  {preview.merged.slice(0, SAMPLE_WORDS).join("、")}
                  {preview.merged.length > SAMPLE_WORDS && " …"}
                </p>
              )}
              {preview.merged.length > 0 && (
                <div className="flex items-center gap-2">
                  <Button
                    onClick={handleConfirm}
                    disabled={saving || removing !== null}
                    variant="primary"
                    size="sm"
                  >
                    {t("settings.advanced.lexiconImport.save")}
                  </Button>
                  <Button
                    onClick={handleCancel}
                    disabled={saving || removing !== null}
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
