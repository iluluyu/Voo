import assert from "node:assert/strict";
import { saveSourcesSequentially } from "./lexiconSources";

void (async () => {
  // 顺序执行：逐个 await（串行，不并发），失败不中断后续来源
  const calls: string[] = [];
  const outcome = await saveSourcesSequentially(
    [
      ["IT", ["信息技术"]],
      ["food", ["美食佳肴"]],
      ["medical", ["医学影像"]],
    ] as const,
    async (source, words) => {
      calls.push(`${source}:${words.length}`);
      return source === "food"
        ? { status: "error", error: "boom" }
        : { status: "ok" };
    },
  );
  assert.deepEqual(calls, ["IT:1", "food:1", "medical:1"]);
  assert.deepEqual(outcome.saved, ["IT", "medical"]);
  assert.deepEqual(outcome.failed, [{ source: "food", error: "boom" }]);

  // 全部成功
  const allOk = await saveSourcesSequentially(
    [["law", ["法律条文"]]] as const,
    async () => ({ status: "ok" }),
  );
  assert.deepEqual(allOk, { saved: ["law"], failed: [] });

  // Real asynchronous read/modify/write: parallel calls would lose snapshots.
  let persisted: string[] = [];
  let active = 0;
  let maxActive = 0;
  await saveSourcesSequentially(
    [
      ["IT", []],
      ["law", []],
      ["food", []],
    ] as const,
    async (source) => {
      active++;
      maxActive = Math.max(maxActive, active);
      const snapshot = [...persisted];
      await Promise.resolve();
      persisted = [...snapshot, source];
      active--;
      return { status: "ok" };
    },
  );
  assert.equal(
    maxActive,
    1,
    "only one backend settings write may be in flight",
  );
  assert.deepEqual(persisted, ["IT", "law", "food"]);

  // A rejected IPC invocation is a per-source failure, not a lost partial result.
  const rejected = await saveSourcesSequentially(
    [
      ["IT", []],
      ["law", []],
      ["food", []],
    ] as const,
    async (source) => {
      if (source === "law") throw new Error("IPC unavailable");
      return { status: "ok" };
    },
  );
  assert.deepEqual(rejected.saved, ["IT", "food"]);
  assert.deepEqual(rejected.failed, [
    { source: "law", error: "IPC unavailable" },
  ]);

  console.log("lexiconSources tests passed");
})();
