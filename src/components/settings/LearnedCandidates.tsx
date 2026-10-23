import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { commands, type LearnedCandidate } from "@/bindings";
import { Button } from "../ui/Button";

interface LearnedCandidatesProps {
  /** 兼容旧 SettingContainer 用法保留；紧凑折叠样式下描述始终走悬浮提示 */
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

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
 * 「已学习」：从粘贴修正中本地学到的词对（原词 → 新词，Candidate 态，
 * 未激活、不影响任何纠错）。紧凑折叠样式：默认只显示一行标题 + 计数
 * 徽章，点击展开单行条目列表（限高滚动）。数据来自只读命令
 * `preview_learned_candidates`（只写 learned_corrections.json，不写
 * custom_words）。
 *
 * - 「采纳」把词对写入自定义热词。任务指定的 `set_lexicon_term_mode`
 *   在当前 bindings 中不存在，按预案本应以 `update_custom_words` 追加
 *   代替；但 updateCustomWords 是整表覆盖，前端读改写会与后端设置写
 *   路径并发竞争（可能丢词），故复用同一允许清单中的
 *   `suggest_correction_to_hotword(corrected, wrong)`：后端原子校验、
 *   去重、追加写入 custom_words 并广播 settings-changed —— 语义等同
 *   「追加代替」且无并发丢词风险。
 * - 「撤销」= `undo_learned_candidate(wrong)`：删除该候选并把词对加入
 *   blocked 集合持久化（同一对之后不再被学习）。
 * - 列表加载成功且为空（无错误、非加载中）时整块不渲染，避免界面噪声；
 *   加载态与错误态保留，错误文本可复制（错误态即使折叠也展示）。
 */
export const LearnedCandidates: React.FC<LearnedCandidatesProps> = React.memo(
  ({ grouped = false }) => {
    const { t } = useTranslation();
    const [candidates, setCandidates] = useState<LearnedCandidate[] | null>(
      null,
    );
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    /** 正在操作的词对（按原词标识，学习中词对唯一）；期间禁用所有行按钮 */
    const [busy, setBusy] = useState<string | null>(null);
    /** 默认折叠，只占一行标题 + 计数徽章 */
    const [open, setOpen] = useState(false);

    const refresh = useCallback(async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await commands.previewLearnedCandidates();
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
      void refresh();
    }, [refresh]);

    const copyError = async () => {
      if (!error) return;
      try {
        await navigator.clipboard.writeText(error);
        toast.success(t("settings.advanced.learnedCandidates.copied"));
      } catch {
        // 复制失败时保持原文可见，用户可手动选中
      }
    };

    const handleAdopt = async (candidate: LearnedCandidate) => {
      if (busy !== null) return;
      setBusy(candidate.wrong);
      try {
        const result = await commands.suggestCorrectionToHotword(
          candidate.corrected,
          candidate.wrong,
        );
        if (result.status === "error") {
          throw new Error(result.error);
        }
        if (result.data) {
          toast.success(
            t("settings.advanced.learnedCandidates.adopted", {
              wrong: candidate.wrong,
              corrected: candidate.corrected,
            }),
          );
        } else {
          toast.info(
            t("settings.advanced.learnedCandidates.exists", {
              wrong: candidate.wrong,
              corrected: candidate.corrected,
            }),
          );
        }
        // 采纳只写 custom_words，学习库本身不变，候选仍可撤销，无需重拉
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        toast.error(t("settings.advanced.learnedCandidates.error", { message }));
      } finally {
        setBusy(null);
      }
    };

    const handleUndo = async (wrong: string) => {
      if (busy !== null) return;
      setBusy(wrong);
      try {
        const result = await commands.undoLearnedCandidate(wrong);
        if (result.status === "error") {
          throw new Error(result.error);
        }
        toast.success(
          t("settings.advanced.learnedCandidates.undone", {
            count: result.data,
          }),
        );
        // 撤销改变了学习库，重拉保持列表一致
        await refresh();
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        toast.error(t("settings.advanced.learnedCandidates.error", { message }));
      } finally {
        setBusy(null);
      }
    };

    const title = t("settings.advanced.learnedCandidates.title");
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
          title={t("settings.advanced.learnedCandidates.description")}
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
              {t("settings.advanced.learnedCandidates.loading")}
            </span>
          )}
        </button>
        {(open || error) && (
          <div className="mt-2 space-y-1">
            {error && (
              <div className="space-y-1">
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded border border-mid-gray/20 bg-black/10 p-2 text-xs text-red-400 select-text">
                  {t("settings.advanced.learnedCandidates.error", {
                    message: error,
                  })}
                </pre>
                <Button
                  onClick={copyError}
                  variant="secondary"
                  size="sm"
                  aria-label={t("settings.advanced.learnedCandidates.copyError")}
                >
                  {t("settings.advanced.learnedCandidates.copyError")}
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
                    disabled={loading || busy !== null}
                    variant="ghost"
                    size="sm"
                  >
                    {loading
                      ? t("settings.advanced.learnedCandidates.loading")
                      : t("settings.advanced.learnedCandidates.refresh")}
                  </Button>
                </div>
                <ul className="max-h-56 overflow-auto">
                  {(candidates ?? []).map((c) => (
                    <li
                      key={c.wrong}
                      className="flex items-center gap-2 py-0.5 text-xs whitespace-nowrap"
                    >
                      <span
                        className="max-w-[8rem] truncate font-medium"
                        title={`${c.wrong} → ${c.corrected}`}
                      >
                        {c.wrong}
                      </span>
                      <span aria-hidden="true" className="shrink-0 text-mid-gray">
                        →
                      </span>
                      <span
                        className="max-w-[8rem] truncate font-medium"
                        title={`${c.wrong} → ${c.corrected}`}
                      >
                        {c.corrected}
                      </span>
                      <span className="shrink-0 text-mid-gray">
                        {t("settings.advanced.learnedCandidates.count", {
                          count: c.count,
                        })}
                      </span>
                      <span className="flex-1" />
                      <Button
                        onClick={() => void handleAdopt(c)}
                        disabled={busy !== null}
                        variant="primary-soft"
                        size="sm"
                      >
                        {t("settings.advanced.learnedCandidates.adopt")}
                      </Button>
                      <Button
                        onClick={() => void handleUndo(c.wrong)}
                        disabled={busy !== null}
                        variant="ghost"
                        size="sm"
                      >
                        {t("settings.advanced.learnedCandidates.undo")}
                      </Button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
    );
  },
);
