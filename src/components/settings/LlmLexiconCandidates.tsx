import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { commands, type LexiconCandidate } from "@/bindings";
import { Button } from "../ui/Button";

interface LlmLexiconCandidatesProps {
  /** 兼容旧 SettingContainer 用法保留；紧凑折叠样式下描述始终走悬浮提示 */
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * `preview_lexicon_curation` 是否存在于当前 bindings。该命令由 tauri-specta
 * 生成，前端重生成前不存在时这里为 undefined —— 整块不渲染（任务要求：
 * 命令不存在则 LLM 候选块整体隐藏）。
 */
const CURATION_COMMAND_SUPPORTED =
  typeof commands.previewLexiconCuration === "function";

/** 折叠头箭头图标（chevron） */
const ChevronIcon: React.FC<{ open: boolean }> = ({ open }) => (
  <svg
    className={`h-3.5 w-3.5 shrink-0 text-mid-gray transition-transform ${open ? "rotate-90" : ""}`}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    aria-hidden="true"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M9 5l7 7-7 7"
    />
  </svg>
);

/**
 * 「LLM 候选」：词库整理候选的只读预览（Candidate 态，未激活、不影响
 * 纠错）。紧凑折叠样式：默认只显示一行标题 + 计数徽章，点击展开单行
 * 条目列表（限高滚动）。数据来自 `preview_lexicon_curation`，每条展示
 * 词条、建议变体与出现次数，evidence 作为悬浮提示；本块不提供任何写
 * 操作（命令契约即只读预览）。
 *
 * 列表加载成功且为空（无错误、非加载中）时整块不渲染；加载态与错误态
 * 保留，错误文本可复制（错误态即使折叠也展示）。
 */
export const LlmLexiconCandidates: React.FC<LlmLexiconCandidatesProps> =
  React.memo(({ grouped = false }) => {
    const { t } = useTranslation();
    const [candidates, setCandidates] = useState<LexiconCandidate[] | null>(
      null,
    );
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    /** 默认折叠，只占一行标题 + 计数徽章 */
    const [open, setOpen] = useState(false);

    const refresh = useCallback(async () => {
      if (!CURATION_COMMAND_SUPPORTED) return;
      setLoading(true);
      setError(null);
      try {
        const result = await commands.previewLexiconCuration();
        if (result.status === "error") {
          throw new Error(result.error);
        }
        setCandidates(result.data);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    }, []);

    useEffect(() => {
      if (!CURATION_COMMAND_SUPPORTED) return;
      void refresh();
    }, [refresh]);

    // 命令不存在：整块不渲染（非空态也隐藏，见组件 doc comment）
    if (!CURATION_COMMAND_SUPPORTED) {
      return null;
    }

    const copyError = async () => {
      if (!error) return;
      try {
        await navigator.clipboard.writeText(error);
        toast.success(t("settings.advanced.llmCandidates.copied"));
      } catch {
        // 复制失败时保持原文可见，用户可手动选中
      }
    };

    const title = t("settings.advanced.llmCandidates.title");
    const expandLabel = t("settings.advanced.candidateList.expand");
    const collapseLabel = t("settings.advanced.candidateList.collapse");

    // 空态：加载成功、无错误、非加载中且无候选 → 整块不渲染
    if (candidates !== null && candidates.length === 0 && !loading && !error) {
      return null;
    }

    return (
      <div
        className={`w-full max-w-md px-4 p-2 ${grouped ? "" : "rounded-lg border border-mid-gray/20"}`}
      >
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-label={`${title} · ${open ? collapseLabel : expandLabel}`}
          title={t("settings.advanced.llmCandidates.description")}
          className="flex w-full cursor-pointer items-center gap-2 text-left"
        >
          <ChevronIcon open={open} />
          <span className="truncate text-sm font-medium">{title}</span>
          {candidates !== null && (
            <span className="shrink-0 rounded-full bg-mid-gray/10 px-1.5 py-px text-xs text-mid-gray tabular-nums">
              {candidates.length}
            </span>
          )}
          {loading && (
            <span className="shrink-0 text-xs text-mid-gray">
              {t("settings.advanced.llmCandidates.loading")}
            </span>
          )}
        </button>
        {(open || error) && (
          <div className="mt-2 space-y-1">
            {error && (
              <div className="space-y-1">
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded border border-mid-gray/20 bg-black/10 p-2 text-xs text-red-400 select-text">
                  {t("settings.advanced.llmCandidates.error", { message: error })}
                </pre>
                <Button
                  onClick={copyError}
                  variant="secondary"
                  size="sm"
                  aria-label={t("settings.advanced.llmCandidates.copyError")}
                >
                  {t("settings.advanced.llmCandidates.copyError")}
                </Button>
              </div>
            )}
            {open && (
              <>
                <div className="flex items-center gap-2 text-xs text-mid-gray">
                  <span className="truncate">
                    {t("settings.advanced.candidateList.summary", {
                      count: (candidates ?? []).length,
                    })}
                  </span>
                  <span className="flex-1" />
                  <Button
                    onClick={() => void refresh()}
                    disabled={loading}
                    variant="ghost"
                    size="sm"
                  >
                    {loading
                      ? t("settings.advanced.llmCandidates.loading")
                      : t("settings.advanced.llmCandidates.refresh")}
                  </Button>
                </div>
                <ul className="max-h-56 overflow-auto">
                  {(candidates ?? []).map((c) => (
                    <li
                      key={c.term}
                      className="flex items-center gap-2 py-0.5 text-xs whitespace-nowrap"
                    >
                      <span
                        className="max-w-[9rem] truncate font-medium"
                        title={c.evidence}
                      >
                        {c.term}
                      </span>
                      {c.aliases.length > 0 && (
                        <span
                          className="max-w-[12rem] truncate text-mid-gray/80"
                          title={c.evidence}
                        >
                          {t("settings.advanced.llmCandidates.variants", {
                            aliases: c.aliases.join(", "),
                          })}
                        </span>
                      )}
                      <span className="shrink-0 text-mid-gray">
                        {t("settings.advanced.llmCandidates.count", {
                          count: c.count,
                        })}
                      </span>
                      <span className="flex-1" />
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
    );
  });
