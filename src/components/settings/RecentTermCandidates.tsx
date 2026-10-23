import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { commands, type Candidate } from "@/bindings";
import { Button } from "../ui/Button";

/** 回看窗口（天）：与后端默认一致 */
const LOOKBACK_DAYS = 30;

interface RecentTermCandidatesProps {
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
 * 「最近常出现」：从最近 30 天历史本地提取的高频候选词（四态模型中的
 * Candidate——只展示，不自动激活）。紧凑折叠样式：默认只显示一行
 * 标题 + 计数徽章，点击展开单行条目列表（限高滚动）。
 *
 * - 数据来自只读命令 `get_recent_term_candidates`（本地统计，无网络）；
 * - 「提升为热词」复用现有「记住这条修正」命令
 *   `suggest_correction_to_hotword(term, null)`：后台校验、去重、落盘
 *   失败回滚，成功后广播 settings-changed；
 * - 「忽略」/「全部忽略」只写组件内 state（本次会话隐藏），不写盘；
 * - 错误态即使折叠也展示（可复制），避免静默失败。
 */
export const RecentTermCandidates: React.FC<RecentTermCandidatesProps> =
  React.memo(({ grouped = false }) => {
    const { t } = useTranslation();
    const [candidates, setCandidates] = useState<Candidate[] | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    /** 会话内忽略的词：仅内存，不落盘 */
    const [hidden, setHidden] = useState<Set<string>>(new Set());
    const [promoting, setPromoting] = useState<string | null>(null);
    /** 默认折叠，只占一行标题 + 计数徽章 */
    const [open, setOpen] = useState(false);

    const refresh = useCallback(async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await commands.getRecentTermCandidates(LOOKBACK_DAYS);
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

    const handlePromote = async (term: string) => {
      if (promoting !== null) return;
      setPromoting(term);
      try {
        const result = await commands.suggestCorrectionToHotword(term, null);
        if (result.status === "error") {
          throw new Error(result.error);
        }
        if (result.data) {
          toast.success(
            t("settings.advanced.recentCandidates.promoted", { term }),
          );
        } else {
          toast.info(t("settings.advanced.recentCandidates.exists", { term }));
        }
        // 后端会排除已提升词（含其子串），重拉保持列表一致
        await refresh();
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        toast.error(
          t("settings.advanced.recentCandidates.error", { message }),
        );
      } finally {
        setPromoting(null);
      }
    };

    const handleIgnore = (term: string) => {
      setHidden((prev) => {
        const next = new Set(prev);
        next.add(term);
        return next;
      });
    };

    /** 批量忽略：同样只写会话内 state，不落盘、无后端调用 */
    const handleIgnoreAll = () => {
      setHidden((prev) => {
        const next = new Set(prev);
        for (const c of candidates ?? []) {
          next.add(c.term);
        }
        return next;
      });
    };

    const copyError = async () => {
      if (!error) return;
      try {
        await navigator.clipboard.writeText(error);
        toast.success(t("settings.advanced.recentCandidates.copied"));
      } catch {
        // 复制失败时保持原文可见，用户可手动选中
      }
    };

    const visible = (candidates ?? []).filter((c) => !hidden.has(c.term));
    const title = t("settings.advanced.recentCandidates.title");
    const expandLabel = t("settings.advanced.candidateList.expand");
    const collapseLabel = t("settings.advanced.candidateList.collapse");

    // 空态：加载成功、无错误、非加载中且可见候选为空（含全部被忽略）
    // → 整块不渲染，避免界面噪声
    if (candidates !== null && visible.length === 0 && !loading && !error) {
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
          title={t("settings.advanced.recentCandidates.description")}
          className="flex w-full cursor-pointer items-center gap-2 text-left"
        >
          <ChevronIcon open={open} />
          <span className="truncate text-sm font-medium">{title}</span>
          {candidates !== null && (
            <span className="shrink-0 rounded-full bg-mid-gray/10 px-1.5 py-px text-xs text-mid-gray tabular-nums">
              {visible.length}
            </span>
          )}
          {loading && (
            <span className="shrink-0 text-xs text-mid-gray">
              {t("settings.advanced.recentCandidates.loading")}
            </span>
          )}
        </button>
        {(open || error) && (
          <div className="mt-2 space-y-1">
            {error && (
              <div className="space-y-1">
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded border border-mid-gray/20 bg-black/10 p-2 text-xs text-red-400 select-text">
                  {t("settings.advanced.recentCandidates.error", {
                    message: error,
                  })}
                </pre>
                <Button
                  onClick={copyError}
                  variant="secondary"
                  size="sm"
                  aria-label={t("settings.advanced.recentCandidates.copyError")}
                >
                  {t("settings.advanced.recentCandidates.copyError")}
                </Button>
              </div>
            )}
            {open && (
              <>
                <div className="flex items-center gap-2 text-xs text-mid-gray">
                  <span className="truncate">
                    {t("settings.advanced.candidateList.summary", {
                      count: visible.length,
                    })}
                  </span>
                  <span className="flex-1" />
                  <Button
                    onClick={() => void refresh()}
                    disabled={loading || promoting !== null}
                    variant="ghost"
                    size="sm"
                  >
                    {loading
                      ? t("settings.advanced.recentCandidates.loading")
                      : t("settings.advanced.recentCandidates.refresh")}
                  </Button>
                  <Button
                    onClick={handleIgnoreAll}
                    disabled={visible.length === 0}
                    variant="ghost"
                    size="sm"
                  >
                    {t("settings.advanced.recentCandidates.ignoreAll")}
                  </Button>
                </div>
                <ul className="max-h-56 overflow-auto">
                  {visible.map((c) => (
                    <li
                      key={c.term}
                      className="flex items-center gap-2 py-0.5 text-xs whitespace-nowrap"
                    >
                      <span
                        className="max-w-[9rem] truncate font-medium"
                        title={c.sample}
                      >
                        {c.term}
                      </span>
                      {/* Candidate 无独立 score 字段：docs（出现过该词的转写
                          条数）即其频次打分依据，按 score 名义展示，悬浮提示
                          说明含义；后端将来加真 score 只需改这一处 */}
                      <span
                        className="shrink-0 text-mid-gray"
                        title={t("settings.advanced.recentCandidates.scoreHint")}
                      >
                        {t("settings.advanced.recentCandidates.count", {
                          count: c.count,
                        })}{" "}
                        ·{" "}
                        {t("settings.advanced.recentCandidates.score", {
                          docs: c.docs,
                        })}
                      </span>
                      <span className="flex-1" />
                      <Button
                        onClick={() => void handlePromote(c.term)}
                        disabled={promoting !== null}
                        variant="primary-soft"
                        size="sm"
                      >
                        {t("settings.advanced.recentCandidates.promote")}
                      </Button>
                      <Button
                        onClick={() => handleIgnore(c.term)}
                        disabled={promoting !== null}
                        variant="ghost"
                        size="sm"
                      >
                        {t("settings.advanced.recentCandidates.ignore")}
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
  });
