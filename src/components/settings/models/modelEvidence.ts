import type { ModelInfo } from "@/bindings";

/**
 * Frontend evidence helpers for the compact models table.
 *
 * `getAvailableModels()` only exposes normalized 0–1 scores; the publisher's
 * per-language CER and the benchmark RTF never reach the UI. The numbers
 * below are mirrored from the bundled catalog
 * (`src-tauri/src/catalog/catalog.json`, `results` map — the same data the
 * backend scores are derived from) for the zh-capable catalog models. The
 * catalog is pinned by `bun run check:catalog`; update this table together
 * with it.
 *
 * - `zhCer`: publisher-measured character error rate on Chinese (percent),
 *   at the default quant.
 * - `rtfAccel`: benchmark-GPU (Vulkan) realtime multiple — higher is faster.
 *   `secondsPerAudioMinute()` converts it to "1 min of audio → X s".
 */
export interface CatalogEvidence {
  family: string;
  zhCer: number;
  rtfAccel: number;
}

export const CATALOG_EVIDENCE: Record<string, CatalogEvidence> = {
  "cohere-transcribe-03-2026": {
    family: "Cohere",
    zhCer: 11.18,
    rtfAccel: 8.095,
  },
  "whisper-medium": { family: "Whisper", zhCer: 13.13, rtfAccel: 4.085 },
  "Voxtral-Mini-4B-Realtime-2602": {
    family: "Voxtral",
    zhCer: 10.41,
    rtfAccel: 0.76,
  },
  "Qwen3-ASR-0.6B": { family: "Qwen3-ASR", zhCer: 7.57, rtfAccel: 9.705 },
  "Fun-ASR-MLT-Nano-2512": { family: "Fun-ASR", zhCer: 8.64, rtfAccel: 8.835 },
  "Fun-ASR-Nano-2512": { family: "Fun-ASR", zhCer: 8.59, rtfAccel: 10.73 },
  "Qwen3-ASR-1.7B": { family: "Qwen3-ASR", zhCer: 7.14, rtfAccel: 4.085 },
  SenseVoiceSmall: { family: "SenseVoice", zhCer: 10.12, rtfAccel: 27.69 },
  "whisper-small": { family: "Whisper", zhCer: 23.06, rtfAccel: 11.695 },
  "whisper-large-v3-turbo": { family: "Whisper", zhCer: 8.5, rtfAccel: 3.095 },
  "Breeze-ASR-25": { family: "Breeze-ASR", zhCer: 8.1, rtfAccel: 2.02 },
  "whisper-large-v3": { family: "Whisper", zhCer: 7.98, rtfAccel: 1.995 },
  "moonshine-base-zh": { family: "Moonshine", zhCer: 17.79, rtfAccel: 28.02 },
  "moonshine-tiny-zh": { family: "Moonshine", zhCer: 13.88, rtfAccel: 42.93 },
  "MOSS-Transcribe-Diarize": { family: "MOSS", zhCer: 9.23, rtfAccel: 3.065 },
  "whisper-base": { family: "Whisper", zhCer: 36.21, rtfAccel: 29.015 },
  "whisper-large": { family: "Whisper", zhCer: 19.32, rtfAccel: 2.005 },
  "whisper-large-v2": { family: "Whisper", zhCer: 15.39, rtfAccel: 2.01 },
  "whisper-tiny": { family: "Whisper", zhCer: 41.91, rtfAccel: 43.09 },
  "nemotron-3.5-asr-streaming-0.6b": {
    family: "Nemotron",
    zhCer: 18.87,
    rtfAccel: 17.135,
  },
};

/** Known architecture families, matched as a prefix of the catalog id. */
const FAMILY_PREFIXES: Array<[string, string]> = [
  ["whisper", "Whisper"],
  ["parakeet", "Parakeet"],
  ["canary", "Canary"],
  ["nemotron", "Nemotron"],
  ["cohere", "Cohere"],
  ["voxtral", "Voxtral"],
  ["qwen3-asr", "Qwen3-ASR"],
  ["fun-asr", "Fun-ASR"],
  ["sensevoice", "SenseVoice"],
  ["moonshine", "Moonshine"],
  ["breeze-asr", "Breeze-ASR"],
  ["moss", "MOSS"],
  ["gigaam", "GigaAM"],
];

/** The catalog id ("whisper-medium") behind a catalog ModelInfo, else null. */
export const catalogIdOf = (model: ModelInfo): string | null => {
  if (model.is_custom) return null;
  const repo = model.id.includes("/") ? model.id.split("/")[1] : "";
  if (!repo) return null;
  return repo.replace(/-gguf$/i, "");
};

/** Quantization label parsed from a GGUF filename, e.g. "Q4_K_M". */
export const quantOf = (model: ModelInfo): string | null => {
  const stem = model.filename.replace(/\.gguf$/i, "");
  const match = stem.match(/(IQ?\d+_[A-Z0-9]+|Q\d+_\d+|BF16|F16|F32)$/i);
  return match ? match[1].toUpperCase() : null;
};

export type QuantTier = "full" | "high" | "balanced" | "smaller" | "smallest";

/** Friendly tier for a quant label, or null when unrecognized. */
export const quantTierOf = (quant: string | null): QuantTier | null => {
  if (!quant) return null;
  const q = quant.toUpperCase();
  if (q === "F32" || q === "BF16" || q === "F16") return "full";
  if (q.startsWith("Q8") || q.startsWith("Q6")) return "high";
  if (q.startsWith("Q5") || q.startsWith("Q4")) return "balanced";
  if (q.startsWith("IQ")) return "smaller";
  if (q.startsWith("Q3") || q.startsWith("Q2") || q.startsWith("Q1"))
    return "smallest";
  return null;
};

/** Architecture family label for display, or null when unknown. */
export const familyOf = (model: ModelInfo): string | null => {
  const catalogId = catalogIdOf(model);
  if (catalogId) {
    const known = CATALOG_EVIDENCE[catalogId];
    if (known) return known.family;
    const lower = catalogId.toLowerCase();
    for (const [prefix, label] of FAMILY_PREFIXES) {
      if (lower.startsWith(prefix)) return label;
    }
    return null;
  }
  // Custom models: best-effort family from the filename, quant stripped.
  const stem = model.filename.replace(/\.gguf$/i, "").replace(/\.bin$/i, "");
  const quant = stem.match(/[-_](IQ?\d+_[A-Z0-9]+|Q\d+_\d+|BF16|F16|F32)$/i);
  if (!stem) return null;
  return quant
    ? stem.slice(0, quant.index ?? stem.length).replace(/[-_]$/, "") || null
    : stem;
};

export type LanguageTagKey =
  | "mandarin"
  | "mixed"
  | "cantonese"
  | "multilingual"
  | "unknown";

/**
 * Compact language tags for the 语言 column: 普通话 / 中英混说 / 方言（粤语）/
 * 多语言. Custom models without probed languages get "unknown".
 */
export const languageTagKeys = (
  languages: string[],
  isCustom: boolean,
): LanguageTagKey[] => {
  if (isCustom && languages.length === 0) return ["unknown"];
  const base = languages.map((l) => l.toLowerCase());
  const hasZh = base.some((l) => l === "zh" || l.startsWith("zh-"));
  const hasYue = base.some((l) => l === "yue" || l.startsWith("yue-"));
  const hasEn = base.some((l) => l === "en" || l.startsWith("en-"));
  const tags: LanguageTagKey[] = [];
  if (hasZh && hasEn && languages.length <= 4) {
    tags.push("mixed");
  } else if (hasZh) {
    tags.push("mandarin");
  }
  if (hasYue) tags.push("cantonese");
  if (languages.length > 8) tags.push("multilingual");
  if (tags.length === 0) tags.push("unknown");
  return tags;
};

/**
 * Rough RAM estimate for a GGUF: weights are mmapped fully resident during
 * inference plus a fixed context/decoder overhead. Clearly labeled 估算 in
 * the UI; not a measurement.
 */
export const estimateMemoryMb = (sizeMb: number): number | null => {
  if (!sizeMb || sizeMb <= 0) return null;
  const estimate = sizeMb * 1.08 + 160;
  return Math.ceil(estimate / 50) * 50;
};

/** Seconds of processing for 1 minute of audio at a given realtime multiple. */
export const secondsPerAudioMinute = (rtf: number): number => {
  if (rtf <= 0) return 0;
  return 60 / rtf;
};

/** "1.5 GB" / "754 MB" formatting for download size and memory estimate. */
export const formatMb = (mb: number): string => {
  if (mb >= 1000) {
    const gb = mb / 1000;
    return `${gb >= 10 ? Math.round(gb) : Math.round(gb * 10) / 10} GB`;
  }
  return `${Math.round(mb)} MB`;
};

/** One decimal for CER percentages. */
export const formatCer = (cer: number): string => cer.toFixed(1);
