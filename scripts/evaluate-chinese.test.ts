import { describe, expect, test } from "bun:test";
import { fileURLToPath } from "node:url";
import {
  editDistance,
  evaluate,
  parseSamples,
  percentile,
} from "./evaluate-chinese";

const sample = {
  id: "term-1",
  reference: "模型量化",
  baseline: "模型亮化",
  candidate: "模型量化",
  terms: ["模型量化", "模型量化"],
  forbidden_terms: ["模型良化"],
  baseline_latency_ms: 100,
  candidate_latency_ms: 120,
};

describe("Chinese evaluation", () => {
  test("uses code points, including supplementary Han and emoji", () => {
    expect(editDistance("𠀀😀", "𠀀")).toBe(1);
    expect(editDistance("", "中英混合")).toBe(4);
    expect(editDistance("模型亮化", "模型量化")).toBe(1);
    expect(editDistance("abc", "acb")).toBe(2);
  });

  test("reports paired CER and deduplicated annotated recall", () => {
    const result = evaluate([sample]);
    expect(result.baseline.cer).toBe(0.25);
    expect(result.candidate.cer).toBe(0);
    expect(result.annotated_terms).toBe(1);
    expect(result.baseline.term_recall).toBe(0);
    expect(result.candidate.term_recall).toBe(1);
    expect(result.improved_samples).toBe(1);
    expect(result.candidate.latency_p95_ms).toBe(120);
  });

  test("detects regressions and newly introduced negative terms", () => {
    const result = evaluate([{ ...sample, candidate: "模型良化" }]);
    expect(result.introduced_false_terms).toBe(1);
    expect(result.introduced_false_term_rate).toBe(1);
    expect(result.candidate.forbidden_term_rate).toBe(1);
    const regressed = evaluate([
      { ...sample, baseline: sample.reference, candidate: "错" },
    ]);
    expect(regressed.regressed_samples).toBe(1);
  });

  test("does not count existing baseline hallucinations as introduced", () => {
    const result = evaluate([
      { ...sample, baseline: "模型良化", candidate: "模型良化" },
    ]);
    expect(result.introduced_false_terms).toBe(0);
  });

  test("normalization is explicit and does not change term annotation scoring", () => {
    const input = [
      { id: "nfc", reference: "é。", baseline: "e\u0301 。", candidate: "é" },
    ];
    expect(evaluate(input).candidate.cer).toBe(0.5);
    expect(evaluate(input, true).candidate.cer).toBe(0);
    expect(evaluate(input, true).baseline.cer).toBe(0);
  });

  test("missing annotations/latency do not turn into perfect scores", () => {
    const result = evaluate([
      { id: "empty", reference: "", baseline: "", candidate: "错" },
    ]);
    expect(result.candidate.edits).toBe(1);
    expect(result.candidate.cer).toBeNull();
    expect(result.candidate.term_recall).toBeNull();
    expect(result.candidate.latency_p50_ms).toBeNull();
    expect(result.introduced_false_term_rate).toBeNull();
  });

  test("uses nearest rank percentiles without mutating observations", () => {
    const data = [100, 5, 20, 40];
    expect(percentile(data, 50)).toBe(20);
    expect(percentile(data, 95)).toBe(100);
    expect(data).toEqual([100, 5, 20, 40]);
  });

  test("parses JSONL and ignores blank lines", () => {
    expect(parseSamples(`\n${JSON.stringify(sample)}\n`)).toEqual([sample]);
  });

  test("rejects empty, malformed and duplicate records", () => {
    expect(() => parseSamples("\n")).toThrow("No evaluation samples");
    expect(() => parseSamples("bad")).toThrow("Line 1: invalid JSON");
    expect(() => parseSamples("null")).toThrow("expected an object");
    expect(() => parseSamples(JSON.stringify({ ...sample, id: " " }))).toThrow(
      "unique",
    );
    expect(() =>
      parseSamples(`${JSON.stringify(sample)}\n${JSON.stringify(sample)}`),
    ).toThrow("Line 2");
    expect(() =>
      parseSamples(JSON.stringify({ ...sample, candidate: 1 })),
    ).toThrow("candidate");
  });

  test("validates errors without echoing sensitive transcript content", () => {
    const secret = "秘密转写内容";
    try {
      parseSamples(
        JSON.stringify({
          ...sample,
          reference: secret,
          terms: [secret + "缺失"],
        }),
      );
      throw new Error("expected rejection");
    } catch (error) {
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).not.toContain(secret);
      expect((error as Error).message).toContain("terms must occur");
    }
    expect(() => parseSamples(`{"reference":"${secret}",`)).toThrow(
      "invalid JSON",
    );
  });

  test("counts missing latency independently for each configuration", () => {
    const result = evaluate([
      {
        id: "timing",
        reference: "中",
        baseline: "中",
        candidate: "中",
        baseline_latency_ms: 0,
      },
    ]);
    expect(result.baseline.latency_samples).toBe(1);
    expect(result.baseline.latency_p50_ms).toBe(0);
    expect(result.candidate.latency_samples).toBe(0);
    expect(result.candidate.latency_p50_ms).toBeNull();
  });

  test("CLI scores the synthetic fixture and accepts explicit normalization", async () => {
    const run = async (args: string[]) => {
      const child = Bun.spawn(
        [process.execPath, "run", "scripts/evaluate-chinese.ts", ...args],
        {
          cwd: fileURLToPath(new URL("../", import.meta.url)),
          stdout: "pipe",
          stderr: "pipe",
        },
      );
      const [code, stdout, stderr] = await Promise.all([
        child.exited,
        new Response(child.stdout).text(),
        new Response(child.stderr).text(),
      ]);
      return { code, stdout, stderr };
    };
    for (const options of [[], ["--ignore-punctuation"]]) {
      const result = await run([
        "tests/fixtures/chinese-eval.synthetic.jsonl",
        ...options,
      ]);
      expect(result.code).toBe(0);
      expect(result.stderr).toBe("");
      const score = JSON.parse(result.stdout);
      expect(score.samples).toBe(2);
      expect(score.baseline.cer).toBe(0.125);
      expect(score.candidate.term_recall).toBe(1);
      expect(score.introduced_false_terms).toBe(1);
      expect(score.candidate.latency_p95_ms).toBe(130);
    }
    expect((await run(["--bogus-flag"])).code).toBe(2);
    expect((await run([])).code).toBe(2);
    const missing = await run(["tests/fixtures/nonexistent-evaluation.jsonl"]);
    expect(missing.code).toBe(1);
    expect(missing.stdout).toBe("");
  });

  test("rejects impossible annotations and invalid timing", () => {
    for (const extra of [
      { terms: ["不在参考中"] },
      { forbidden_terms: [sample.reference] },
      { terms: [""] },
      { terms: "invalid" },
      { candidate_latency_ms: -1 },
      { baseline_latency_ms: "fast" },
    ]) {
      expect(() =>
        parseSamples(JSON.stringify({ ...sample, ...extra })),
      ).toThrow();
    }
  });
});
