#!/usr/bin/env bun
/** Score paired transcriptions of the SAME audio. No audio/text is uploaded. */
export interface EvaluationSample {
  id: string;
  reference: string;
  baseline: string;
  candidate: string;
  terms?: string[];
  forbidden_terms?: string[];
  baseline_latency_ms?: number;
  candidate_latency_ms?: number;
}

/** Unicode code-point Levenshtein distance, O(min(m,n)) memory. */
export function editDistance(a: string, b: string): number {
  let left = Array.from(a);
  let right = Array.from(b);
  if (left.length < right.length) [left, right] = [right, left];
  const row = Array.from({ length: right.length + 1 }, (_, i) => i);
  for (let i = 1; i <= left.length; i++) {
    let diagonal = row[0];
    row[0] = i;
    for (let j = 1; j <= right.length; j++) {
      const above = row[j];
      row[j] = Math.min(
        row[j] + 1,
        row[j - 1] + 1,
        diagonal + Number(left[i - 1] !== right[j - 1]),
      );
      diagonal = above;
    }
  }
  return row[right.length];
}

/** Nearest-rank percentiles, not interpolation; missing observations stay null. */
export function percentile(values: number[], percent: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.max(0, Math.ceil((percent / 100) * sorted.length) - 1)];
}

export function parseSamples(input: string): EvaluationSample[] {
  const seen = new Set<string>();
  const samples: EvaluationSample[] = [];
  for (const [index, line] of input.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    const fail = (message: string): never => {
      throw new Error(`Line ${index + 1}: ${message}`);
    };
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      fail("invalid JSON");
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      fail("expected an object");
    }
    const record = value as Record<string, unknown>;
    for (const key of ["id", "reference", "baseline", "candidate"]) {
      if (typeof record[key] !== "string") fail(`${key} must be a string`);
    }
    if (!(record.id as string).trim() || seen.has(record.id as string)) {
      fail("id must be non-empty and unique");
    }
    for (const key of ["terms", "forbidden_terms"]) {
      if (
        record[key] !== undefined &&
        (!Array.isArray(record[key]) ||
          !(record[key] as unknown[]).every(
            (term) => typeof term === "string" && term.trim().length > 0,
          ))
      ) {
        fail(`${key} must be an array of non-empty strings`);
      }
    }
    for (const key of ["baseline_latency_ms", "candidate_latency_ms"]) {
      if (
        record[key] !== undefined &&
        (typeof record[key] !== "number" ||
          !Number.isFinite(record[key]) ||
          (record[key] as number) < 0)
      ) {
        fail(`${key} must be a finite non-negative number`);
      }
    }
    const sample = record as unknown as EvaluationSample;
    for (const term of sample.terms ?? []) {
      if (!sample.reference.includes(term))
        fail("terms must occur in reference");
    }
    for (const term of sample.forbidden_terms ?? []) {
      if (sample.reference.includes(term)) {
        fail("forbidden_terms must not occur in reference");
      }
    }
    seen.add(sample.id);
    samples.push(sample);
  }
  if (samples.length === 0) throw new Error("No evaluation samples");
  return samples;
}

export function evaluate(
  samples: EvaluationSample[],
  ignorePunctuation = false,
) {
  const normalize = (text: string) => {
    const nfc = text.normalize("NFC");
    return ignorePunctuation ? nfc.replace(/[\p{P}\s]/gu, "") : nfc;
  };
  let referenceCharacters = 0;
  let termCount = 0;
  let negativeCount = 0;
  let introducedFalseTerms = 0;
  let improved = 0;
  let regressed = 0;
  const counts = {
    baseline: {
      edits: 0,
      termHits: 0,
      forbiddenHits: 0,
      latency: [] as number[],
    },
    candidate: {
      edits: 0,
      termHits: 0,
      forbiddenHits: 0,
      latency: [] as number[],
    },
  };
  for (const sample of samples) {
    const reference = normalize(sample.reference);
    const baselineEdits = editDistance(reference, normalize(sample.baseline));
    const candidateEdits = editDistance(reference, normalize(sample.candidate));
    referenceCharacters += Array.from(reference).length;
    improved += Number(candidateEdits < baselineEdits);
    regressed += Number(candidateEdits > baselineEdits);
    counts.baseline.edits += baselineEdits;
    counts.candidate.edits += candidateEdits;
    // Score each distinct annotated term once per sample, using exact strings.
    const terms = [...new Set(sample.terms ?? [])];
    const forbidden = [...new Set(sample.forbidden_terms ?? [])];
    termCount += terms.length;
    negativeCount += forbidden.length;
    for (const name of ["baseline", "candidate"] as const) {
      counts[name].termHits += terms.filter((t) =>
        sample[name].includes(t),
      ).length;
      counts[name].forbiddenHits += forbidden.filter((t) =>
        sample[name].includes(t),
      ).length;
      const latency = sample[`${name}_latency_ms`];
      if (latency !== undefined) counts[name].latency.push(latency);
    }
    introducedFalseTerms += forbidden.filter(
      (t) => !sample.baseline.includes(t) && sample.candidate.includes(t),
    ).length;
  }
  const summarize = (name: "baseline" | "candidate") => ({
    edits: counts[name].edits,
    cer: referenceCharacters ? counts[name].edits / referenceCharacters : null,
    term_recall: termCount ? counts[name].termHits / termCount : null,
    forbidden_term_rate: negativeCount
      ? counts[name].forbiddenHits / negativeCount
      : null,
    latency_samples: counts[name].latency.length,
    latency_p50_ms: percentile(counts[name].latency, 50),
    latency_p95_ms: percentile(counts[name].latency, 95),
  });
  return {
    samples: samples.length,
    normalization: ignorePunctuation
      ? "NFC, omit punctuation/whitespace"
      : "NFC",
    reference_characters: referenceCharacters,
    annotated_terms: termCount,
    annotated_negatives: negativeCount,
    baseline: summarize("baseline"),
    candidate: summarize("candidate"),
    improved_samples: improved,
    regressed_samples: regressed,
    introduced_false_terms: introducedFalseTerms,
    introduced_false_term_rate: negativeCount
      ? introducedFalseTerms / negativeCount
      : null,
  };
}

if (import.meta.main) {
  const args = Bun.argv.slice(2);
  const paths = args.filter((arg) => !arg.startsWith("--"));
  if (
    paths.length !== 1 ||
    args.some((arg) => arg.startsWith("--") && arg !== "--ignore-punctuation")
  ) {
    console.error(
      "Usage: bun scripts/evaluate-chinese.ts <paired-results.jsonl> [--ignore-punctuation]",
    );
    process.exit(2);
  }
  try {
    const samples = parseSamples(await Bun.file(paths[0]).text());
    console.log(
      JSON.stringify(
        evaluate(samples, args.includes("--ignore-punctuation")),
        null,
        2,
      ),
    );
  } catch (error) {
    // Do not echo transcripts or the dataset's contents in diagnostics.
    console.error(error instanceof Error ? error.message : "Evaluation failed");
    process.exit(1);
  }
}
