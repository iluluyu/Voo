//! 离线回放调参：48 kHz 单声道 wav → 自适应降噪（nnnoiseless 真实后端
//! + FSM 混合）→ 处理后 wav + CSV 遥调（stdout）。
//!
//! 用法：
//! ```text
//! cargo run --release --example denoise_offline -- path/to/input.wav [output.wav] > ticks.csv
//! ```
//!
//! - 输入要求 48 kHz 单声道 wav（f32/i16/i32 整数都会归一化到 [-1,1] f32；
//!   其他采样率请先重采样）。
//! - 输出默认写到 `<输入卡去扩展名>.denoised.wav`（f32 48k mono）：内容为
//!   `g·denoised + (1-g)·HP`，g 由现 FSM 驱动，总长与输入逐样本相等
//!   （finish 尾帧零丢失、零 padding 上屏）。
//! - CSV 列见 `telemetry_header()`：`(t, Ls, floor, snr, p, r_lf, sigma_rlf,
//!   state, g)`，每 20ms 一行。遥调在 flush 之前 drain（flush 帧的行不进
//!   CSV，避免把流结束后的零填充 tick 当真实时间轴）。
//!
//! 【offline-only】本工具不读 `VOO_EXPERIMENTAL_DENOISE`：它本身就是显式
//! 的离线实验入口；录音链路（16k）尚未接线，生产路径零引用。

use hound::{SampleFormat, WavReader, WavWriter};
use std::io::Write;
use voo_app_lib::audio_toolkit::denoise::{
    nn::NnnoiselessBackend, AdaptiveDenoiseStage, SAMPLE_RATE,
};

/// 每次喂给降噪级的样本数（100ms；分块只影响遥调 drain 时机，
/// mix_chunk_invariance 测试保证分块不影响输出）。
const BLOCK: usize = 4800;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!(
            "用法: {} <input.wav> [output.wav]   # 48kHz 单声道 wav，输出 f32 wav + stdout CSV",
            args[0]
        );
        std::process::exit(2);
    }
    let path = args[1].clone();
    let out_path = match args.get(2) {
        Some(p) => p.clone(),
        // 默认：<输入>.denoised.wav（in.wav → in.denoised.wav）。
        None => std::path::Path::new(&path)
            .with_extension("denoised.wav")
            .to_string_lossy()
            .into_owned(),
    };
    // 拒绝输出 == 输入（含规范化后同文件）：避免覆盖原始录音。
    let same_file = |a: &str, b: &str| -> bool {
        let canon = |p: &str| {
            std::fs::canonicalize(p).ok().or_else(|| {
                // 目标可能尚不存在：规范化父目录 + 文件名。
                let pp = std::path::Path::new(p);
                let parent = pp.parent().map(|d| d.to_path_buf()).unwrap_or_default();
                let name = pp.file_name()?.to_owned();
                Some(std::fs::canonicalize(parent).ok()?.join(name))
            })
        };
        match (canon(a), canon(b)) {
            (Some(x), Some(y)) => x == y,
            // 无法规范化（如父目录缺失）时退回字面比较。
            _ => a == b,
        }
    };
    if same_file(&path, &out_path) {
        eprintln!("拒绝：输出路径与输入相同（{path}），覆盖原始录音不受支持");
        std::process::exit(1);
    }
    let mut reader = match WavReader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("无法打开 {path}: {e}");
            std::process::exit(1);
        }
    };
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE {
        eprintln!(
            "需要 {} Hz wav，实际 {} Hz（请先重采样）",
            SAMPLE_RATE, spec.sample_rate
        );
        std::process::exit(1);
    }
    if spec.channels != 1 {
        eprintln!(
            "需要单声道 wav，实际 {} 声道（请先混缩为 mono）",
            spec.channels
        );
        std::process::exit(1);
    }

    fn die<T>(e: hound::Error) -> T {
        eprintln!("wav 读取失败: {e}");
        std::process::exit(1)
    }

    // 读入全部样本并归一化为 f32（离线工具，内存换简单）。
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.unwrap_or_else(die))
            .collect(),
        SampleFormat::Int => {
            let full_scale = 1i64 << (spec.bits_per_sample - 1).min(31);
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or_else(die) as f32 / full_scale as f32)
                .collect()
        }
    };

    println!("{}", AdaptiveDenoiseStage::telemetry_header());
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut stage = AdaptiveDenoiseStage::with_backend_and_mix_preprocessed(Box::new(
        NnnoiselessBackend::new(),
    ));
    let mut processed: Vec<f32> = Vec::with_capacity(samples.len());
    let mut total_ticks = 0usize;
    let mut pos = 0usize;
    while pos < samples.len() {
        let end = (pos + BLOCK).min(samples.len());
        processed.extend(stage.process_mix(&samples[pos..end]));
        for row in stage.drain_telemetry() {
            let _ = writeln!(out, "{row}");
            total_ticks += 1;
        }
        pos = end;
    }
    // flush 前把真实时间轴的遥调全部 drain（见模块文档）。
    for row in stage.drain_telemetry() {
        let _ = writeln!(out, "{row}");
        total_ticks += 1;
    }
    let tail = stage.finish_mix();
    processed.extend(tail);
    drop(out);

    // 输出处理后的 wav（f32 48k mono；样本已由链路保证有限）。
    // create_new：已存在（含指向同一 inode 的 hardlink）一律拒绝，
    // 不会截断覆盖既有文件。
    let wav_spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let write_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        use std::io::BufWriter;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&out_path)
            .map_err(|e| format!("输出已存在或不可创建 {out_path}: {e}"))?;
        let mut writer = WavWriter::new(BufWriter::new(file), wav_spec)?;
        for &s in &processed {
            writer.write_sample(s)?;
        }
        writer.finalize()?;
        Ok(())
    })();
    if let Err(e) = write_result {
        eprintln!("无法写 {out_path}: {e}");
        std::process::exit(1);
    }

    eprintln!(
        "处理完成: {} 样本 ({:.2}s) → {out_path}, {} ticks, 终态 {}, floor {:.1} dBFS, gain {:.3}, agc={:.1} dB, plosive={} 次 (duck {:.2}), degraded={} bad_input={}",
        samples.len(),
        samples.len() as f64 / SAMPLE_RATE as f64,
        total_ticks,
        stage.state().as_str(),
        stage.noise_floor_db(),
        stage.gain(),
        stage.agc_gain_db(),
        stage.plosive_detections(),
        stage.plosive_duck(),
        stage.backend_degraded(),
        stage.bad_input_samples(),
    );
    if processed.len() != samples.len() {
        eprintln!(
            "FATAL: 输出长度 {} != 输入长度 {}（违反长度守恒契约）",
            processed.len(),
            samples.len()
        );
        std::process::exit(1);
    }
}
