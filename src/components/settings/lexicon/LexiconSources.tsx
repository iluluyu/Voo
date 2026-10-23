import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  commands,
  type LexiconScope,
  type LexiconSourceInfo,
  type OpenLexiconPreview,
  type TermMode,
} from "@/bindings";
import { useSettings } from "../../../hooks/useSettings";
import Badge from "../../ui/Badge";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { SettingContainer } from "../../ui/SettingContainer";

/**
 * 在 `LexiconSettings` 里管理的开源来源分组（THUOCL 组仍走下方
 * `LexiconImport` 既有入口，不在这里重复渲染）。
 */
const MANAGED_GROUPS = [
  "proper_nouns",
  "domain_han",
  "frequency_baseline",
] as const;
type ManagedGroup = (typeof MANAGED_GROUPS)[number];

/** C 组（词频基准）默认折叠；rime-essay 不建议直接导入。 */
const COLLAPSED_BY_DEFAULT: readonly ManagedGroup[] = ["frequency_baseline"];

const LIMIT_CHOICES = [500, 1000, 2000, 5000] as const;
type LimitChoice = (typeof LIMIT_CHOICES)[number];
const DEFAULT_LIMIT: LimitChoice = 2000;

const SCOPES = ["all", "latin_only", "han_only"] as const;

/** 来源级三档模式（`blocked` 不是来源级档位，见后端命令注释）。 */
const SOURCE_MODES = ["candidate", "active_exact", "active_fuzzy"] as const;
type SourceMode = (typeof SOURCE_MODES)[number];

/** 后端未显式记录时导入源的缺省档位（与解析器一致，见 lexicon_resolver）。 */
const DEFAULT_SOURCE_MODE: SourceMode = "active_exact";

const MODE_LABEL_KEYS: Record<SourceMode, string> = {
  candidate: "modeCandidate",
  active_exact: "modeExact",
  active_fuzzy: "modeFuzzy",
};

/** 目录 id → `imported_lexicons` 存储键（THUOCL 目录 id 带 `thuocl-` 前缀，
 * 存储键是裸分类名）。 */
const storageKeyOf = (s: LexiconSourceInfo) =>
  s.group === "thuocl" ? s.id.replace(/^thuocl-/, "") : s.id;

interface InstalledRow {
  /** `imported_lexicons` 的存储键 */
  sourceKey: string;
  count: number;
  /** 目录里的对应源；未知来源（如旧版残留键）为 null */
  info: LexiconSourceInfo | null;
}

interface DialogState {
  source: LexiconSourceInfo;
  /** 已安装来源再次进入预览时按钮文案换成“替换快照” */
  mode: "import" | "update";
  scope: LexiconScope;
  limit: LimitChoice;
  preview: OpenLexiconPreview | null;
  loading: boolean;
  error: string | null;
  confirming: boolean;
}

interface LexiconSourcesProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/**
 * 开源词库来源管理：分组目录（专名/术语 · 中文领域 · 词频基准）+
 * 已安装来源一栏 + 单源「预览 → 确认」导入弹窗。
 *
 * - 来源目录来自后端 `list_lexicon_sources`（只读），已安装状态由前端按
 *   `imported_lexicons` 合并（来源 id 即存储键）。
 * - 每源独立预览/确认：`preview_open_lexicon_source` 不写设置；用户在
 *   弹窗里选范围/上限后重新预览，点确认才调 `confirm_open_lexicon_source`
 *   写入快照（替换式更新）。没有“全部导入”入口。
 * - 移除走两步确认（第一次点击只亮出确认条），复用
 *   `remove_imported_lexicon`；所有命令结果显式检查 `status`，失败只报
 *   错绝不 toast 成功。`requestId` 递增让旧的异步结果失效。
 * - THUOCL 11 分类（`thuocl` 组）保留既有 `LexiconImport` 入口，这里只放
 *   一行指引。
 */
export const LexiconSources: React.FC<LexiconSourcesProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting } = useSettings();

    const [sources, setSources] = useState<LexiconSourceInfo[] | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [collapsedC, setCollapsedC] = useState(true);
    const [dialog, setDialog] = useState<DialogState | null>(null);
    const [pendingRemove, setPendingRemove] = useState<string | null>(null);
    const [removing, setRemoving] = useState<string | null>(null);
    /** 正在切换模式的来源键（单飞：整设置读改写，禁止并发） */
    const [modeBusy, setModeBusy] = useState<string | null>(null);
    /** 模式切换的乐观覆盖：store 尚未同步时先显示目标档位；
     * 失败即清空（回滚），成功则等 settings 广播追平后由 effect 清空。 */
    const [modeOverride, setModeOverride] = useState<{
      sourceKey: string;
      mode: SourceMode;
    } | null>(null);
    /** 模式切换失败的行内错误文案（key 定位到出错的那一行）。 */
    const [modeError, setModeError] = useState<{
      sourceKey: string;
      message: string;
    } | null>(null);
    /** 弹窗内预览/确认的请求序号：任何新交互使旧的异步结果失效 */
    const requestIdRef = useRef(0);

    const importedLexicons = getSetting("imported_lexicons") || {};
    const lexiconSourceModes = getSetting("lexicon_source_modes") || {};

    useEffect(() => {
      let cancelled = false;
      commands
        .listLexiconSources()
        .then((data) => {
          if (!cancelled) setSources(data);
        })
        .catch((e) => {
          if (!cancelled) {
            setLoadError(e instanceof Error ? e.message : String(e));
          }
        });
      return () => {
        cancelled = true;
      };
    }, []);

    const installedCount = (source: LexiconSourceInfo) =>
      importedLexicons[storageKeyOf(source)]?.length ?? 0;

    /** 存储键已存在即视为已安装（空快照也算，避免重复导入）。 */
    const isInstalled = (source: LexiconSourceInfo) =>
      importedLexicons[storageKeyOf(source)] !== undefined;

    /** 已安装来源全量清单：以 `imported_lexicons` 为准（一源一行，
     * 渲染**每一个**键），目录只用来补充名称/描述/更新入口；
     * 目录加载失败也照常渲染。 */
    const installedRows: InstalledRow[] = useMemo(() => {
      const byKey = new Map<string, LexiconSourceInfo>();
      for (const s of sources ?? []) byKey.set(storageKeyOf(s), s);
      return Object.entries(importedLexicons)
        .map(([sourceKey, words]) => ({
          sourceKey,
          count: words?.length ?? 0,
          info: byKey.get(sourceKey) ?? null,
        }))
        .sort((a, b) =>
          (a.info?.label ?? a.sourceKey).localeCompare(
            b.info?.label ?? b.sourceKey,
          ),
        );
    }, [sources, importedLexicons]);

    /** 当前生效档位：乐观覆盖优先，未显式记录时为导入源缺省
     * （active_exact）。 */
    const currentMode = (sourceKey: string): TermMode =>
      modeOverride?.sourceKey === sourceKey
        ? modeOverride.mode
        : (lexiconSourceModes[sourceKey] ?? DEFAULT_SOURCE_MODE);

    /** 乐观值被 store 追平（settings 广播已重拉）后清掉覆盖，
     * 避免残留的覆盖值遮蔽后续真实状态。 */
    useEffect(() => {
      if (
        modeOverride &&
        lexiconSourceModes[modeOverride.sourceKey] === modeOverride.mode
      ) {
        setModeOverride(null);
      }
    }, [modeOverride, lexiconSourceModes]);

    const copyText = async (text: string, copiedKey: string) => {
      try {
        await navigator.clipboard.writeText(text);
        toast.success(t(copiedKey));
      } catch {
        // 复制失败时保持原文可见，用户可手动选中
      }
    };

    /** 拉取单源预览（不写设置）。scope/limit 变化即重新预览。 */
    const runPreview = async (
      source: LexiconSourceInfo,
      scope: LexiconScope,
      limit: LimitChoice,
    ) => {
      const requestId = ++requestIdRef.current;
      setDialog((prev) =>
        prev ? { ...prev, loading: true, error: null, preview: null } : prev,
      );
      try {
        const result = await commands.previewOpenLexiconSource(
          source.id,
          scope,
          limit,
        );
        if (requestId !== requestIdRef.current) return;
        if (result.status === "error") {
          throw new Error(result.error);
        }
        setDialog((prev) =>
          prev ? { ...prev, loading: false, preview: result.data } : prev,
        );
      } catch (e) {
        if (requestId !== requestIdRef.current) return;
        const message = e instanceof Error ? e.message : String(e);
        setDialog((prev) =>
          prev ? { ...prev, loading: false, error: message } : prev,
        );
      }
    };

    const openDialog = (source: LexiconSourceInfo) => {
      setPendingRemove(null);
      setLoadError(null);
      setDialog({
        source,
        mode: isInstalled(source) ? "update" : "import",
        scope: "all",
        limit: DEFAULT_LIMIT,
        preview: null,
        loading: true,
        error: null,
        confirming: false,
      });
      void runPreview(source, "all", DEFAULT_LIMIT);
    };

    const closeDialog = () => {
      if (dialog?.confirming) return;
      requestIdRef.current += 1;
      setDialog(null);
    };

    const changeScope = (scope: LexiconScope) => {
      if (!dialog || dialog.loading || dialog.confirming) return;
      setDialog({ ...dialog, scope });
      void runPreview(dialog.source, scope, dialog.limit);
    };

    const changeLimit = (limit: LimitChoice) => {
      if (!dialog || dialog.loading || dialog.confirming) return;
      setDialog({ ...dialog, limit });
      void runPreview(dialog.source, dialog.scope, limit);
    };

    const handleConfirm = async () => {
      if (!dialog || !dialog.preview || dialog.loading || dialog.confirming)
        return;
      const requestId = ++requestIdRef.current;
      setDialog({ ...dialog, confirming: true, error: null });
      try {
        const result = await commands.confirmOpenLexiconSource(
          dialog.source.id,
          dialog.preview.words,
        );
        if (requestId !== requestIdRef.current) return;
        if (result.status === "error") {
          throw new Error(result.error);
        }
        // 后端已广播 settings-changed，store 会自动重拉 installed 状态
        toast.success(
          t("settings.lexicon.sources.dialog.success", {
            count: result.data.count,
          }),
        );
        setDialog(null);
      } catch (e) {
        if (requestId !== requestIdRef.current) return;
        const message = e instanceof Error ? e.message : String(e);
        setDialog((prev) =>
          prev ? { ...prev, confirming: false, error: message } : prev,
        );
        toast.error(
          t("settings.lexicon.sources.dialog.confirmError", { message }),
        );
      }
    };

    const handleRemove = async (sourceId: string) => {
      if (removing || dialog) return;
      const requestId = ++requestIdRef.current;
      setLoadError(null);
      setRemoving(sourceId);
      try {
        const result = await commands.removeImportedLexicon(sourceId);
        if (result.status === "error") {
          throw new Error(result.error);
        }
      } catch (e) {
        if (requestId !== requestIdRef.current) return;
        const message = e instanceof Error ? e.message : String(e);
        toast.error(t("settings.lexicon.sources.removeError", { message }));
        setLoadError(t("settings.lexicon.sources.removeError", { message }));
      } finally {
        if (requestId === requestIdRef.current) {
          setRemoving(null);
          setPendingRemove(null);
        }
      }
    };

    /** 切换来源级三档模式。后端整设置读改写，所以同一时刻只允许一个
     * 在途请求；先乐观显示目标档位，成功由 settings-changed 广播驱动
     * store 重拉追平，失败回滚乐观覆盖并显示行内错误。 */
    const handleSetMode = async (sourceKey: string, mode: SourceMode) => {
      if (modeBusy) return;
      setModeBusy(sourceKey);
      setModeOverride({ sourceKey, mode });
      setModeError(null);
      try {
        const result = await commands.setLexiconSourceMode(sourceKey, mode);
        if (result.status === "error") {
          throw new Error(result.error);
        }
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        // 回滚本地乐观状态（档位显示回到 store 里的真实值）
        setModeOverride((prev) =>
          prev?.sourceKey === sourceKey ? null : prev,
        );
        setModeError({ sourceKey, message });
        toast.error(t(key("modeError"), { message }));
      } finally {
        setModeBusy(null);
      }
    };

    const key = (k: string) => `settings.lexicon.sources.${k}`;

    const managed = (sources ?? []).filter((s) =>
      (MANAGED_GROUPS as readonly string[]).includes(s.group),
    );
    const unknownGroupSources = (sources ?? []).filter(
      (s) =>
        !(MANAGED_GROUPS as readonly string[]).includes(s.group) &&
        s.group !== "thuocl",
    );

    const renderRow = (source: LexiconSourceInfo) => {
      const count = installedCount(source);
      const installed = isInstalled(source);
      const rowBusy = removing !== null || dialog !== null;
      return (
        <div
          key={`g-${source.id}`}
          className="flex flex-col gap-1 rounded-md border border-mid-gray/20 p-2"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium">{source.label}</span>
            <Badge variant="secondary">{source.license}</Badge>
            {source.recommended ? (
              <Badge variant="success">{t(key("recommended"))}</Badge>
            ) : (
              <span className="text-xs text-mid-gray">
                {t(key("notRecommended"))}
              </span>
            )}
            <span className="ml-auto text-xs text-mid-gray">
              {installed
                ? t(key("statusInstalled"), { count })
                : t(key("statusNotInstalled"))}
            </span>
          </div>
          <p className="text-xs text-mid-gray">{source.description}</p>
          <div className="flex flex-wrap items-center gap-2">
            {/* 已安装的源在这里只显示状态：导入按钮禁用（避免重复导入），
                更新入口在上方“已安装”区块。 */}
            <Button
              onClick={() => openDialog(source)}
              disabled={rowBusy || installed}
              variant={installed ? "secondary" : "primary"}
              size="sm"
            >
              {installed ? t(key("installedDisabled")) : t(key("import"))}
            </Button>
          </div>
        </div>
      );
    };

    /** 已安装来源一行：名称、词数、三档模式切换、更新（有目录条目时）、
     * 移除（两步确认）。以 `imported_lexicons` 为准全量渲染。 */
    const renderInstalledRow = (row: InstalledRow) => {
      const { sourceKey, count, info } = row;
      const label = info?.label ?? sourceKey;
      const mode = currentMode(sourceKey);
      const busy = removing !== null || dialog !== null || modeBusy !== null;
      return (
        <div
          key={`installed-${sourceKey}`}
          className="flex flex-col gap-1 rounded-md border border-mid-gray/20 p-2"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium">{label}</span>
            {info ? (
              <Badge variant="secondary">{info.license}</Badge>
            ) : (
              <Badge variant="secondary">{t(key("unknownSource"))}</Badge>
            )}
            <span className="ml-auto text-xs text-mid-gray">
              {t(key("statusInstalled"), { count })}
            </span>
          </div>
          {info && <p className="text-xs text-mid-gray">{info.description}</p>}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-mid-gray">
              {t(key("modeLabel"))}
            </span>
            {SOURCE_MODES.map((m) => (
              <Button
                key={m}
                onClick={() => void handleSetMode(sourceKey, m)}
                disabled={busy || mode === m}
                variant={mode === m ? "primary" : "secondary"}
                size="sm"
                aria-pressed={mode === m}
              >
                {t(key(MODE_LABEL_KEYS[m]))}
              </Button>
            ))}
            {mode === "blocked" && (
              <Badge variant="secondary">{t(key("modeBlocked"))}</Badge>
            )}
            {modeError?.sourceKey === sourceKey && (
              <p className="w-full text-xs text-red-400">
                {t(key("modeError"), { message: modeError.message })}
              </p>
            )}
            {info && (
              <Button
                onClick={() => openDialog(info)}
                disabled={busy}
                variant="secondary"
                size="sm"
              >
                {t(key("update"))}
              </Button>
            )}
            {pendingRemove === sourceKey ? (
              <span className="inline-flex items-center gap-1 rounded border border-red-400/40 px-1 py-0.5">
                <span className="text-xs text-red-400">
                  {t(key("removeConfirm"), { source: label })}
                </span>
                <Button
                  onClick={() => {
                    setPendingRemove(null);
                    void handleRemove(sourceKey);
                  }}
                  disabled={removing !== null}
                  variant="primary"
                  size="sm"
                >
                  {t(key("removeConfirmYes"))}
                </Button>
                <Button
                  onClick={() => setPendingRemove(null)}
                  disabled={removing !== null}
                  variant="secondary"
                  size="sm"
                >
                  {t(key("cancel"))}
                </Button>
              </span>
            ) : (
              <Button
                onClick={() => setPendingRemove(sourceKey)}
                disabled={busy}
                variant="secondary"
                size="sm"
                aria-label={t(key("remove"))}
              >
                {removing === sourceKey
                  ? t(key("removing"))
                  : t(key("remove"))}
              </Button>
            )}
          </div>
        </div>
      );
    };

    const renderGroup = (group: ManagedGroup) => {
      const groupSources = managed.filter((s) => s.group === group);
      if (groupSources.length === 0) return null;
      const collapsed = group === "frequency_baseline" ? collapsedC : false;
      return (
        <div key={group} className="space-y-1">
          {group === "frequency_baseline" ? (
            <button
              type="button"
              className="flex w-full items-center gap-1 text-left text-sm font-medium"
              aria-expanded={!collapsedC}
              onClick={() => setCollapsedC((v) => !v)}
            >
              {collapsedC ? (
                <ChevronRight className="w-4 h-4 shrink-0" />
              ) : (
                <ChevronDown className="w-4 h-4 shrink-0" />
              )}
              {t(key("groupFrequencyBaseline"))}
              <span className="text-xs font-normal text-mid-gray">
                {t(key("notRecommended"))}
              </span>
            </button>
          ) : (
            <p className="text-sm font-medium">
              {t(
                key(
                  group === "proper_nouns"
                    ? "groupProperNouns"
                    : "groupDomainHan",
                ),
              )}
            </p>
          )}
          {!collapsed && (
            <div className="space-y-1">
              {groupSources.map((s) => renderRow(s))}
            </div>
          )}
        </div>
      );
    };

    const d = dialog;
    const dBusy = d ? d.loading || d.confirming : false;

    return (
      <SettingContainer
        title={t(key("title"))}
        description={t(key("description"))}
        descriptionMode={descriptionMode}
        grouped={grouped}
      >
        <div className="w-full max-w-md space-y-3">
          {loadError && (
            <div className="space-y-1">
              <p className="text-xs text-red-400">{t(key("loadFailed"))}</p>
              <pre className="max-h-24 overflow-auto whitespace-pre-wrap break-all rounded border border-mid-gray/20 bg-black/10 p-2 text-xs text-red-400 select-text">
                {loadError}
              </pre>
              <Button
                onClick={() => void copyText(loadError, key("copied"))}
                variant="secondary"
                size="sm"
                aria-label={t(key("copyError"))}
              >
                {t(key("copyError"))}
              </Button>
            </div>
          )}

          {/* 已安装来源：以 settings.imported_lexicons 为准全量渲染，
              目录加载失败也照常展示（可切模式/移除）。 */}
          <div className="space-y-1">
            <p className="text-sm font-medium">
              {t(key("installedTitle"), { count: installedRows.length })}
            </p>
            {installedRows.length > 0 ? (
              installedRows.map(renderInstalledRow)
            ) : (
              <p className="text-xs text-mid-gray">
                {t(key("installedEmpty"))}
              </p>
            )}
          </div>

          {sources === null && !loadError && (
            <p className="text-xs text-mid-gray">{t(key("loading"))}</p>
          )}

          {sources !== null && managed.length === 0 && (
            <p className="text-xs text-mid-gray">{t(key("empty"))}</p>
          )}

          {sources !== null && (
            <>
              {MANAGED_GROUPS.map((g) => renderGroup(g))}
              {unknownGroupSources.length > 0 && (
                <div className="space-y-1">
                  <p className="text-sm font-medium">{t(key("groupOther"))}</p>
                  {unknownGroupSources.map((s) => renderRow(s))}
                </div>
              )}
              <p className="text-xs text-mid-gray">{t(key("thuoclHint"))}</p>
            </>
          )}
        </div>

        {d && (
          <Dialog
            open
            title={t(
              key(
                d.mode === "update"
                  ? "dialog.titleUpdate"
                  : "dialog.titleImport",
              ),
              { source: d.source.label },
            )}
            description={
              <span className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary">{d.source.license}</Badge>
                {d.preview && (
                  <span className="text-xs text-mid-gray">
                    {t(key("dialog.version"), { version: d.preview.version })}
                  </span>
                )}
              </span>
            }
            onOpenChange={(open) => {
              if (!open) closeDialog();
            }}
            closeLabel={t(key("cancel"))}
            dismissible={!d.confirming}
            closeOnBackdrop={!d.confirming}
            showCloseButton={!d.confirming}
            footer={
              <div className="flex items-center gap-2">
                <Button
                  onClick={() => void handleConfirm()}
                  disabled={!d.preview || d.loading || d.confirming}
                  variant="primary"
                  size="sm"
                >
                  {d.confirming
                    ? t(key("dialog.confirming"))
                    : d.preview
                      ? t(
                          key(
                            d.mode === "update"
                              ? "dialog.confirmUpdate"
                              : "dialog.confirmImport",
                          ),
                          { count: d.preview.words.length },
                        )
                      : t(key("dialog.confirmImport"), { count: 0 })}
                </Button>
                <Button
                  onClick={closeDialog}
                  disabled={d.confirming}
                  variant="secondary"
                  size="sm"
                >
                  {t(key("cancel"))}
                </Button>
              </div>
            }
          >
            <div className="space-y-3">
              {d.loading && (
                <p className="text-xs text-mid-gray">
                  {t(key("dialog.loadingPreview"))}
                </p>
              )}

              {d.error && (
                <div className="space-y-1">
                  <pre className="max-h-24 overflow-auto whitespace-pre-wrap break-all rounded border border-mid-gray/20 bg-black/10 p-2 text-xs text-red-400 select-text">
                    {t(key("dialog.previewError"), { message: d.error })}
                  </pre>
                  <Button
                    onClick={() => void copyText(d.error ?? "", key("copied"))}
                    variant="secondary"
                    size="sm"
                    aria-label={t(key("copyError"))}
                  >
                    {t(key("copyError"))}
                  </Button>
                </div>
              )}

              {d.preview && (
                <>
                  {d.preview.warning && (
                    <p className="rounded border border-red-400/40 p-2 text-xs text-red-400">
                      {d.preview.warning}
                    </p>
                  )}
                  <div className="space-y-1 text-xs text-mid-gray">
                    <p>
                      {t(key("dialog.entries"), {
                        count: d.preview.total_unique,
                      })}
                    </p>
                    <p>
                      {t(key("dialog.filtered"), {
                        count: d.preview.filtered,
                      })}
                    </p>
                    <p className="text-text">
                      {t(key("dialog.kept"), {
                        count: d.preview.words.length,
                      })}
                    </p>
                  </div>

                  <div className="space-y-1">
                    <p className="text-xs text-mid-gray">
                      {t(key("dialog.samples"), {
                        count: d.preview.samples.length,
                      })}
                    </p>
                    <div className="max-h-32 overflow-auto rounded border border-mid-gray/20 p-2">
                      <ul className="space-y-0.5">
                        {d.preview.samples.map((sample) => (
                          <li
                            key={sample.word}
                            className="flex items-center justify-between gap-2 text-xs"
                          >
                            <span>{sample.word}</span>
                            <span className="text-mid-gray">
                              {sample.weight ?? t(key("dialog.noWeight"))}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <p className="text-xs text-mid-gray">
                      {t(key("dialog.scope"))}
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {SCOPES.map((scope) => (
                        <Button
                          key={scope}
                          onClick={() => changeScope(scope)}
                          disabled={dBusy}
                          variant={d.scope === scope ? "primary" : "secondary"}
                          size="sm"
                          aria-pressed={d.scope === scope}
                        >
                          {t(key(`dialog.scope_${scope}`))}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-1">
                    <p className="text-xs text-mid-gray">
                      {t(key("dialog.limit"))}
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {LIMIT_CHOICES.map((limit) => (
                        <Button
                          key={limit}
                          onClick={() => changeLimit(limit)}
                          disabled={dBusy}
                          variant={d.limit === limit ? "primary" : "secondary"}
                          size="sm"
                          aria-pressed={d.limit === limit}
                        >
                          {limit}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <p className="text-xs text-mid-gray">
                    {t(key("dialog.upstream"))}:{" "}
                    <a
                      className="underline"
                      href={d.preview.source_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {d.preview.source_url}
                    </a>
                  </p>
                </>
              )}
            </div>
          </Dialog>
        )}
      </SettingContainer>
    );
  },
);
