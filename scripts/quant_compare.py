#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fun-ASR-Nano-2512 量化对比实测：F16 vs Q8_0 vs Q5_K_M vs Q4_K_M（用户真实录音）。

四步流水线：
  ① 从 src-tauri/src/catalog/catalog.json 取 Fun-ASR-Nano-2512 四个量化档的
     文件名/大小/sha256，下载到 ~/.cache/voo-quant-bench/models/ 并校验
     （优先复用本机 HuggingFace 缓存里同 sha 的 blob；hf.co 失败退 hf-mirror.com）。
  ② 从用户真实录音目录选 24 段（按时长分层：短<2s / 中2-5s / 长>5s 各 8 段，
     固定随机种子可复现），ffmpeg 统一重采样 16k mono pcm_s16le。
  ③ 逐模型跑 cargo example `quant_bench`（CPU-only 后端、language=zh、Itn=On，
     与 app 的 funasr 运行配置一致），每模型 2 个 pass 取稳定数字。
  ④ 以 F16 输出为参考算每档 CER（与 scripts/evaluate-chinese.ts 的
     editDistance 同算法：码点级 Levenshtein、两行滚动数组；分含标点/去标点两种
     归一化），统计单句 P50/总耗时，写 docs/quant-benchmark.md。

用法：
  python3 scripts/quant_compare.py            # 全流程
  python3 scripts/quant_compare.py --skip-download   # 复用已下载模型
环境要求：ffmpeg/ffprobe、cargo（构建 --example quant_bench --release）、网络。
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO / "src-tauri" / "src" / "catalog" / "catalog.json"
RECORDINGS = Path(
    "/mnt/c/Users/Administrator/AppData/Roaming/com.voo.desktop/recordings"
)
MODEL_ID = "Fun-ASR-Nano-2512"
QUANTS = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M"]
HF_REPO_TEMPLATE = "https://huggingface.co/{repo}/resolve/{rev}/{name}"
HF_MIRROR_TEMPLATE = "https://hf-mirror.com/{repo}/resolve/{rev}/{name}"
# 本机可能已有同 sha blob 的 HF 缓存目录（跨盘只能 copy，不能 link）。
HF_CACHE_CANDIDATES = [
    Path.home() / ".cache/huggingface/hub/models--handy-computer--Fun-ASR-Nano-2512-gguf/blobs",
    Path("/mnt/c/Users/Administrator/.cache/huggingface/hub/models--handy-computer--Fun-ASR-Nano-2512-gguf/blobs"),
]

CACHE = Path.home() / ".cache/voo-quant-bench"
MODEL_DIR = CACHE / "models"
WORK = CACHE / "work"          # wav16/ lists/ results/
DOCS_OUT = REPO / "docs/quant-benchmark.md"

PER_BUCKET = 8                 # 短/中/长 各 8 段 → 共 24
SEED = 2512
PASSES = 2


def log(msg):
        print(f"[quant-compare] {msg}", flush=True)


def die(msg):
        print(f"[quant-compare] FATAL: {msg}", file=sys.stderr, flush=True)
        sys.exit(1)


# --------------------------------------------------------------- ① 模型


def load_catalog_entry():
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        for model in catalog["models"]:
                if model["id"] == MODEL_ID:
                        return model
        die(f"catalog.json 中找不到 {MODEL_ID}")


def sha256_of(path, chunk=1 << 22):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
                while True:
                        block = fh.read(chunk)
                        if not block:
                                break
                        h.update(block)
        return h.hexdigest()


def ensure_model(entry, quant, skip_download):
        info = entry["files"][quant]
        name, size, sha = info["filename"], info["size"], info["sha256"]
        dest = MODEL_DIR / name
        if dest.exists() and dest.stat().st_size == size and sha256_of(dest) == sha:
                log(f"{quant}: 已存在且校验通过 ({name})")
                return dest
        if skip_download:
                die(f"{quant}: --skip-download 但 {dest} 缺失或校验不过")
        # 复用本机 HF 缓存同 sha blob（HF blob 文件名即 sha256）。
        for cache_dir in HF_CACHE_CANDIDATES:
                blob = cache_dir / sha
                if blob.exists() and blob.stat().st_size == size:
                        log(f"{quant}: 复用 HF 缓存 blob {blob}")
                        shutil.copyfile(blob, dest)
                        if sha256_of(dest) == sha:
                                return dest
                        dest.unlink(missing_ok=True)
                        log(f"{quant}: 缓存 blob 校验失败，改走下载")
                        break
        urls = [
                HF_REPO_TEMPLATE.format(repo=entry["repository"], rev=entry["revision"], name=name),
                HF_MIRROR_TEMPLATE.format(repo=entry["repository"], rev=entry["revision"], name=name),
        ]
        part = dest.with_suffix(dest.suffix + ".part")
        # 大文件容易被中间盒重置 HTTP/2 流：强制 http1.1 + 断点续传重试循环，
        # 每次失败后从已有 .part 续传，直到 size+sha 都对才算完。
        for url in urls:
                for attempt in range(1, 13):
                        have = part.stat().st_size if part.exists() else 0
                        if have > size:
                                part.unlink(missing_ok=True)
                                have = 0
                        log(f"{quant}: 下载 {url} (attempt {attempt}, 已有 {have / 2**20:.0f}MB)")
                        rc = subprocess.call(
                                ["curl", "--http1.1", "-L", "--fail", "-C", "-",
                                 "--speed-limit", "16384", "--speed-time", "30",
                                 "--connect-timeout", "20", "--silent", "--show-error",
                                 "-o", str(part), url],
                        )
                        if part.exists() and part.stat().st_size == size and sha256_of(part) == sha:
                                part.rename(dest)
                                log(f"{quant}: 下载完成，sha256 校验通过")
                                return dest
                        got = part.stat().st_size if part.exists() else 0
                        log(f"{quant}: attempt 失败 (curl rc={rc}, size={got}/{size})，续传重试")
                log(f"{quant}: 源 {url} 多次重试后仍不完整")
        die(f"{quant}: 所有下载源失败")


# --------------------------------------------------------------- ② 选材


def probe_duration(path):
        try:
                out = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "csv=p=0", str(path)],
                        capture_output=True, text=True, check=True,
                ).stdout.strip()
                return float(out)
        except Exception:
                return None


def bucket_of(duration):
        if duration < 2.0:
                return "short"
        if duration <= 5.0:
                return "mid"
        return "long"


def select_files():
        wavs = sorted(RECORDINGS.glob("*.wav"))
        if len(wavs) < 20:
                die(f"录音目录只有 {len(wavs)} 个 wav（需要 >=20）: {RECORDINGS}")
        log(f"扫描 {len(wavs)} 个录音的时长…")
        by_bucket = {"short": [], "mid": [], "long": []}
        for wav in wavs:
                dur = probe_duration(wav)
                if dur is not None and dur >= 0.3:      # 忽略过短的碎片
                        by_bucket[bucket_of(dur)].append((wav, dur))
        rng = random.Random(SEED)
        picked = []
        # 桶内随机取 PER_BUCKET 个；不足的桶取全部，缺额补给其他桶（总数 20–30）。
        quota = dict.fromkeys(by_bucket, PER_BUCKET)
        for name in by_bucket:
                pool = sorted(by_bucket[name])
                k = min(quota[name], len(pool))
                picked.extend(rng.sample(pool, k))
                quota[name] -= k
        for name, deficit in quota.items():
                if deficit > 0:
                        rest = [x for x in by_bucket[name] if x not in picked]
                        picked.extend(rng.sample(rest, min(deficit, len(rest))))
        if len(picked) < 20:
                die(f"分层后只有 {len(picked)} 段可用（短{len(by_bucket['short'])}/中{len(by_bucket['mid'])}/长{len(by_bucket['long'])}）")
        # 固定顺序：按桶（短→中→长）再按文件名。
        order = {"short": 0, "mid": 1, "long": 2}
        picked.sort(key=lambda wd: (order[bucket_of(wd[1])], wd[0].name))
        log(f"选中 {len(picked)} 段（短{sum(bucket_of(d)=='short' for _, d in picked)}"
            f"/中{sum(bucket_of(d)=='mid' for _, d in picked)}"
            f"/长{sum(bucket_of(d)=='long' for _, d in picked)}）")
        return picked


def convert_files(picked):
        wav16 = WORK / "wav16"
        wav16.mkdir(parents=True, exist_ok=True)
        manifest = []
        for wav, dur in picked:
                dst = wav16 / wav.name
                if not dst.exists():
                        subprocess.run(
                                ["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
                                check=True,
                        )
                manifest.append({"src": wav.name, "wav16": str(dst), "dur": round(dur, 2),
                                 "bucket": bucket_of(dur)})
        (WORK / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        return manifest


# --------------------------------------------------------------- ③ 跑引擎


def build_example():
        log("cargo build --release --example quant_bench（可能与其他 worker 争锁，稍等）")
        subprocess.run(
                ["cargo", "build", "--release", "--example", "quant_bench"],
                cwd=REPO / "src-tauri", check=True,
        )
        binary = REPO / "src-tauri/target/release/examples/quant_bench"
        if not binary.exists():
                die(f"example 未生成: {binary}")
        return binary


def warm_page_cache(model_path):
        with open(model_path, "rb") as fh:
                while fh.read(1 << 24):
                        pass


def run_pass(binary, model_path, list_path):
        """返回 (ok, rows|error)。rows: 每 {file,text,wall_ms,mel_ms,encode_ms,decode_ms}。"""
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(REPO / "src-tauri/transcribe-libs")
        t0 = time.monotonic()
        proc = subprocess.run(
                [str(binary), str(model_path), str(list_path)],
                capture_output=True, text=True, env=env,
        )
        total_s = time.monotonic() - t0
        rows, model_error = [], None
        for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                        continue
                try:
                        rec = json.loads(line)
                except json.JSONDecodeError:
                        continue
                if "model_error" in rec:
                        model_error = rec["model_error"]
                elif "file" in rec:
                        rows.append(rec)
        if model_error:
                return False, f"模型加载失败: {model_error}"
        if proc.returncode != 0:
                return False, f"exit={proc.returncode}: {proc.stderr[-400:]}"
        bad = [r for r in rows if not r.get("ok")]
        if bad:
                return False, f"{len(bad)}/{len(rows)} 个文件转写失败，如 {bad[0].get('error')}"
        return True, {"rows": rows, "wall_s": total_s}


def run_all_models(entry, models, manifest, binary):
        list_path = WORK / "list.txt"
        list_path.write_text("".join(m["wav16"] + "\n" for m in manifest))
        results = {}
        for quant in QUANTS:
                model_path = models[quant]
                warm_page_cache(model_path)
                passes = []
                err = None
                for pi in range(PASSES):
                        ok, payload = run_pass(binary, model_path, list_path)
                        if not ok:
                                err = payload
                                break
                        passes.append(payload)
                        log(f"{quant} pass{pi + 1}: {payload['wall_s']:.1f}s "
                            f"wall, {len(payload['rows'])} 段")
                results[quant] = {"ok": err is None, "error": err, "passes": passes}
        (WORK / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
        return results


# --------------------------------------------------------------- ④ 指标
# 与 scripts/evaluate-chinese.ts 的 editDistance 同算法：码点级 Levenshtein，
# 两行滚动数组。CER = edits / len(reference)。


def edit_distance(a, b):
        left, right = list(a), list(b)
        if len(left) < len(right):
                left, right = right, left
        row = list(range(len(right) + 1))
        for i in range(1, len(left) + 1):
                diagonal = row[0]
                row[0] = i
                for j in range(1, len(right) + 1):
                        above = row[j]
                        cost = 0 if left[i - 1] == right[j - 1] else 1
                        row[j] = min(row[j] + 1, row[j - 1] + 1, diagonal + cost)
                        diagonal = above
        return row[len(right)]


def normalize(text, strip_punct):
        text = unicodedata.normalize("NFC", text)
        out = []
        for ch in text:
                if ch.isspace():
                        continue
                if strip_punct and unicodedata.category(ch).startswith("P"):
                        continue
                out.append(ch)
        return "".join(out)


def percentile(values, pct):
        """最近秩百分位，与 evaluate-chinese.ts 的 percentile 一致。"""
        if not values:
                return None
        s = sorted(values)
        return s[max(0, min(len(s) - 1, math.ceil(pct / 100 * len(s)) - 1))]


def analyze(results, manifest):
        """returns per-quant metrics dict（F16 自身作为参考不进 CER 表）。"""
        by_file = {}          # quant -> file -> text
        for quant, res in results.items():
                if not res["ok"]:
                        continue
                merged = {}
                for p in res["passes"]:
                        for r in p["rows"]:
                                merged[os.path.basename(r["file"])] = r
                by_file[quant] = merged
        f16 = by_file.get("F16", {})
        metrics = {}
        for quant in QUANTS:
                m = {"size_mb": None, "runnable": results[quant]["ok"],
                     "error": results[quant]["error"]}
                if not results[quant]["ok"]:
                        metrics[quant] = m
                        continue
                rows = by_file[quant]
                wall, engine, per_file_cer, per_file_cer_np = [], [], [], []
                edits_tot = chars_tot = edits_np_tot = chars_np_tot = 0
                skipped_empty_ref = 0
                diffs = []
                for item in manifest:
                        name = item["src"]
                        if name not in rows or name not in f16:
                                continue
                        r = rows[name]
                        wall.append(r["wall_ms"])
                        engine.append(r.get("encode_ms", 0.0) + r.get("decode_ms", 0.0) + r.get("mel_ms", 0.0))
                        if quant == "F16":
                                continue
                        ref = f16[name]["text"]
                        hyp = r["text"]
                        ref_n, hyp_n = normalize(ref, False), normalize(hyp, False)
                        if not ref_n:
                                skipped_empty_ref += 1
                                continue
                        ed = edit_distance(ref_n, hyp_n)
                        per_file_cer.append(ed / len(ref_n))
                        edits_tot += ed
                        chars_tot += len(ref_n)
                        ref_np, hyp_np = normalize(ref, True), normalize(hyp, True)
                        ed_np = edit_distance(ref_np, hyp_np)
                        per_file_cer_np.append(ed_np / max(1, len(ref_np)))
                        edits_np_tot += ed_np
                        chars_np_tot += len(ref_np)
                        diffs.append({"file": name, "edits": ed, "ref_len": len(ref_n),
                                      "bucket": item["bucket"], "dur": item["dur"],
                                      "f16": ref, "hyp": hyp})
                m.update({
                        "p50_wall_ms": percentile(wall, 50),
                        "mean_wall_ms": statistics.mean(wall) if wall else None,
                        "p50_engine_ms": percentile(engine, 50),
                        "total_wall_s": sum(p["wall_s"] for p in results[quant]["passes"]),
                        "rows": len(wall),
                        "corpus_cer": edits_tot / chars_tot if chars_tot else None,
                        "corpus_cer_no_punct": edits_np_tot / chars_np_tot if chars_np_tot else None,
                        "mean_file_cer": statistics.mean(per_file_cer) if per_file_cer else None,
                        "max_file_cer": max(per_file_cer) if per_file_cer else None,
                        "empty_ref_skipped": skipped_empty_ref,
                        "diffs": diffs,
                })
                metrics[quant] = m
        return metrics


# --------------------------------------------------------------- 报告


REDACT_DIGITS = str.maketrans(
        "0123456789０１２３４５６７８９", "X" * 10 + "X" * 10,
)


def redact(text, limit=60):
        text = text.translate(REDACT_DIGITS)
        return text if len(text) <= limit else text[:limit] + "…"


def fmt_ms(v):
        return "—" if v is None else f"{v:,.0f}"


def fmt_pct(v):
        return "—" if v is None else f"{v * 100:.2f}%"


def write_report(metrics, entry, manifest, backend_note):
        f16 = metrics.get("F16", {})
        f16_p50 = f16.get("p50_wall_ms")
        f16_total = f16.get("total_wall_s")
        lines = []
        lines.append("# Fun-ASR-Nano-2512 量化档对比（用户真实录音实测）")
        lines.append("")
        lines.append(f"> 生成于 {time.strftime('%Y-%m-%d %H:%M')}，由 `scripts/quant_compare.py` 产出；"
                     f"引擎走 `src-tauri/examples/quant_bench.rs`（transcribe-cpp，CPU 后端，"
                     f"language=zh、Itn=On、Task=Transcribe，与 app 的 funasr 运行配置一致）。")
        lines.append("")
        lines.append("## 怎么读这张表（诚实声明）")
        lines.append("")
        lines.append("- 表中 CER 是**各量化档相对本机 F16 输出的偏离**，不是绝对识别准确率"
                     "（没有人工标注；`history.db` 只是粗参考，未用作 ground truth）。")
        lines.append("- 样本来自**单一用户、单一麦克风**的真实录音，只代表这台机器、这位用户的"
                     "口音/话题分布，不能外推到其他用户。")
        lines.append("- 耗时为同机同条件（同 CPU、同线程数、同一批 wav、每模型先预热页缓存再跑"
                     f" {PASSES} 个 pass）；数字受当时后台负载影响，见文末环境注记。")
        lines.append("")
        lines.append("## 主表")
        lines.append("")
        lines.append("| 量化 | 体积 | 相对 F16 体积 | 可运行 | CER vs F16（含标点） |"
                     " CER vs F16（去标点） | 单句 P50 耗时 | 两 pass 总耗时 | 相对增速(P50) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        f16_size = entry["files"]["F16"]["size"]
        for quant in QUANTS:
                m = metrics[quant]
                size = entry["files"][quant]["size"]
                if not m["runnable"]:
                        lines.append(f"| {quant} | {size / 2**20:,.0f} MB | "
                                     f"{size / f16_size * 100:.0f}% | ❌ 不可运行 | — | — | — | — | — |")
                        continue
                speedup = (f16_p50 / m["p50_wall_ms"]) if f16_p50 and m.get("p50_wall_ms") else None
                lines.append(
                        f"| {quant} | {size / 2**20:,.0f} MB | {size / f16_size * 100:.0f}% | ✅ | "
                        f"{fmt_pct(m['corpus_cer'])} | {fmt_pct(m['corpus_cer_no_punct'])} | "
                        f"{fmt_ms(m['p50_wall_ms'])} ms | {m['total_wall_s']:.0f} s | "
                        f"{'—' if quant == 'F16' else ('%.2fx' % speedup if speedup else '—')} |",
                )
        lines.append("")
        lines.append(f"样本：{len(manifest)} 段（短<2s {sum(1 for x in manifest if x['bucket']=='short')} / "
                     f"中2-5s {sum(1 for x in manifest if x['bucket']=='mid')} / "
                     f"长>5s {sum(1 for x in manifest if x['bucket']=='long')}），"
                     f"共 {sum(x['dur'] for x in manifest):.0f}s 音频；"
                     f"每段 16k mono 重采样后逐段独立转写，每模型 {PASSES} pass。")
        lines.append("")
        lines.append(f"模型 revision `{entry['revision'][:12]}…`（catalog sha256 校验通过）。"
                     f"{backend_note}")
        if any(m.get("empty_ref_skipped") for m in metrics.values()):
                skipped = max(m.get("empty_ref_skipped", 0) for m in metrics.values())
                lines.append(f"\n注：{skipped} 段 F16 输出为空（过短/静音），已从 CER 统计中剔除。")
        lines.append("")
        lines.append("## 差异样例（按编辑距离取最大，最多 5 条，数字已脱敏）")
        for quant in ["Q8_0", "Q5_K_M", "Q4_K_M"]:
                m = metrics[quant]
                if not m["runnable"] or not m.get("diffs"):
                        continue
                top = sorted(m["diffs"], key=lambda d: (-d["edits"], -d["ref_len"]))[:5]
                lines.append("")
                lines.append(f"### {quant} vs F16")
                for d in top:
                        lines.append(f"- `{d['file']}`（{d['bucket']}，{d['dur']:.1f}s，"
                                     f"编辑距离 {d['edits']}/{d['ref_len']}）")
                        lines.append(f"  - F16: {redact(d['f16'])}")
                        lines.append(f"  - {quant}: {redact(d['hyp'])}")
        lines.append("")
        lines.append("## 结论")
        lines.append(CONCLUSION_PLACEHOLDER)
        lines.append("")
        lines.append("## 复现")
        lines.append("")
        lines.append("```bash")
        lines.append("python3 scripts/quant_compare.py            # 全流程（下载+选材+跑+写本文件）")
        lines.append("# 缓存与中间产物: ~/.cache/voo-quant-bench/{models,work}")
        lines.append("```")
        lines.append("")
        DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
        DOCS_OUT.write_text("\n".join(lines), encoding="utf-8")
        log(f"报告已写入 {DOCS_OUT}")


CONCLUSION_PLACEHOLDER = None  # 已在跑完后人工填写，见 write_report 后续 edit


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip-download", action="store_true")
        args = parser.parse_args()

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        WORK.mkdir(parents=True, exist_ok=True)

        entry = load_catalog_entry()
        log(f"catalog: {MODEL_ID} rev={entry['revision'][:12]}…")

        models = {}
        for quant in QUANTS:
                models[quant] = ensure_model(entry, quant, args.skip_download)

        manifest_path = WORK / "manifest.json"
        if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                log(f"复用已选样本 {len(manifest)} 段（{manifest_path}）")
        else:
                manifest = convert_files(select_files())

        binary = build_example()
        results = run_all_models(entry, models, manifest, binary)
        metrics = analyze(results, manifest)

        load_note = []
        for quant in QUANTS:
                res = results[quant]
                if res["ok"] and res["passes"]:
                        first = res["passes"][0]["rows"]
                        if first and "load_ms" in first[0]:
                                load_note.append(f"{quant} 加载 {first[0]['load_ms']:.0f}ms")
        backend_note = ("引擎内部计时（mel+encode+decode）的单句 P50 另见 results.json；"
                        + "；".join(load_note) + "。") if load_note else ""
        host = subprocess.run(["uname", "-r"], capture_output=True, text=True).stdout.strip()
        backend_note += f" 主机: WSL2 kernel {host}, {os.cpu_count()} 逻辑核。"

        write_report(metrics, entry, manifest, backend_note)

        # 摘要打到终端
        print("\n==== 摘要 ====")
        for quant in QUANTS:
                m = metrics[quant]
                if not m["runnable"]:
                        print(f"{quant}: 不可运行 — {m['error']}")
                        continue
                print(f"{quant}: CER={fmt_pct(m['corpus_cer'])} "
                      f"(去标点 {fmt_pct(m['corpus_cer_no_punct'])}) "
                      f"P50={fmt_ms(m['p50_wall_ms'])}ms 总耗时={m['total_wall_s']:.0f}s")
        print(f"\n报告: {DOCS_OUT}")


if __name__ == "__main__":
        main()
