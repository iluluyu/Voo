/**
 * 词库来源保存的纯执行器（与 React 解耦，便于回归测试）。
 *
 * 逐个来源顺序 await：后端每个 `set_imported_lexicon` 都是整份设置的
 * 读-改-写，并发 `Promise.all` 会互相覆盖丢来源，必须串行。某个来源
 * 失败不中断后续来源，返回 saved/failed 两个列表让 UI 把部分成功
 * 可见地报告出来（成功数 toast + 失败详情内联）。
 */

/** 与 bindings `Result<null, string>` 结构兼容的最小结果形状 */
export type CommandResult =
  | { status: "ok" }
  | { status: "error"; error: string };

export interface SaveOutcome<S extends string> {
  saved: S[];
  failed: { source: S; error: string }[];
}

export async function saveSourcesSequentially<S extends string>(
  entries: readonly (readonly [S, readonly string[]])[],
  invoke: (source: S, words: string[]) => Promise<CommandResult>,
): Promise<SaveOutcome<S>> {
  const saved: S[] = [];
  const failed: { source: S; error: string }[] = [];
  for (const [source, words] of entries) {
    try {
      const result = await invoke(source, [...words]);
      if (result.status === "error") {
        failed.push({ source, error: result.error });
      } else {
        saved.push(source);
      }
    } catch (error) {
      // IPC/transport failures can reject instead of returning a Result.
      // Keep earlier successes visible and still attempt the remaining sources.
      failed.push({
        source,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
  return { saved, failed };
}
