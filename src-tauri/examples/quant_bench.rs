//! Fun-ASR-Nano-2512 量化对比基准：加载指定 gguf，把一批 16k mono wav
//! 逐段转写，每段输出一行 JSON（text + 引擎内部计时 + 墙钟）。
//!
//! 仅供 `scripts/quant_compare.py` 使用，是最小入口，不是业务代码：
//! 不经 engine_supervisor（那是给 app 管理常驻 worker 的），直接进程内
//! 调 transcribe-cpp，模型只加载一次、逐文件 run，行为与 app 对 funasr
//! 的运行配置一致（Task::Transcribe、language=zh、Itn::On）。
//!
//! 用法：
//!   quant_bench [--gpu] <model.gguf> <filelist>
//!
//! - `--gpu` 缺省时强制 CPU（BackendMask::CPU），保证四档同机同条件；
//!   传 `--gpu` 则 init_backends_default()（本机有什么用什么）。
//! - `<filelist>` 每行一个 wav 路径（16kHz mono，任意 PCM 位深）。
//! - stdout 输出 JSONL，每行一个文件；native 日志全部关掉避免污染 stdout。
//! - 模型加载失败打印 `{"model_error": "..."}` 并 exit 3（脚本据此记
//!   「不可运行」而不是静默跳过）。

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use transcribe_cpp::{BackendMask, Itn, Model, RunOptions, Task};

/// JSON 字符串转义（只处理会被破坏的最小集合：引号/反斜杠/控制字符）。
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// 读 16kHz mono wav 为 [-1, 1] f32。hound 处理 Float/Int 两种 spec，
/// ffmpeg 重采样产物是 pcm_s16le。
fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>, String> {
    let reader = hound::WavReader::open(path).map_err(|e| format!("open wav: {e}"))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 {
        return Err(format!("sample_rate {} != 16000", spec.sample_rate));
    }
    if spec.channels != 1 {
        return Err(format!("channels {} != 1", spec.channels));
    }
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.map_err(|e| format!("read sample: {e}")))
            .collect(),
        hound::SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| {
                s.map_err(|e| format!("read sample: {e}"))
                    .map(|v| v as f32 / 32_768.0)
            })
            .collect(),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let gpu = matches!(args.first().map(String::as_str), Some("--gpu"));
    let rest = if gpu { &args[1..] } else { &args[..] };
    if rest.len() != 2 {
        eprintln!("usage: quant_bench [--gpu] <model.gguf> <filelist>");
        return ExitCode::from(2);
    }
    let (model_path, list_path) = (&rest[0], &rest[1]);

    // native 日志走 stderr/关闭，stdout 只留 JSONL。
    transcribe_cpp::disable_logging();
    let init = if gpu {
        transcribe_cpp::init_backends_default()
    } else {
        transcribe_cpp::init_backends_with(None::<&Path>, BackendMask::CPU)
    };
    if let Err(e) = init {
        println!(
            "{{\"model_error\": \"init backends: {}\"}}",
            json_escape(&e.to_string())
        );
        return ExitCode::from(3);
    }

    let files: Vec<String> = match fs::File::open(list_path) {
        Ok(f) => io::BufReader::new(f)
            .lines()
            .map_while(Result::ok)
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        Err(e) => {
            eprintln!("quant_bench: cannot open filelist {list_path}: {e}");
            return ExitCode::from(2);
        }
    };
    eprintln!("quant_bench: {} files, model={}", files.len(), model_path);

    let t_load = Instant::now();
    let model = match Model::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            println!(
                "{{\"model_error\": \"load: {}\"}}",
                json_escape(&e.to_string())
            );
            return ExitCode::from(3);
        }
    };
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    let mut session = match model.session() {
        Ok(s) => s,
        Err(e) => {
            println!(
                "{{\"model_error\": \"session: {}\"}}",
                json_escape(&e.to_string())
            );
            return ExitCode::from(3);
        }
    };
    eprintln!("quant_bench: model loaded in {load_ms:.0} ms");

    let run = RunOptions {
        task: Task::Transcribe,
        language: Some("zh".into()),
        itn: Itn::On,
        ..RunOptions::default()
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    for file in &files {
        let pcm = match read_wav_mono_16k(Path::new(file)) {
            Ok(p) => p,
            Err(e) => {
                let _ = writeln!(
                    out,
                    "{{\"file\": \"{}\", \"ok\": false, \"error\": \"{}\"}}",
                    json_escape(file),
                    json_escape(&e)
                );
                let _ = out.flush();
                continue;
            }
        };
        let t0 = Instant::now();
        let result = session.run(&pcm, &run);
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let line = match result {
            Ok(t) => format!(
                "{{\"file\": \"{}\", \"ok\": true, \"text\": \"{}\", \"wall_ms\": {:.1}, \
                 \"load_ms\": {:.1}, \"mel_ms\": {:.1}, \"encode_ms\": {:.1}, \"decode_ms\": {:.1}}}",
                json_escape(file),
                json_escape(&t.text),
                wall_ms,
                load_ms,
                t.timings.mel_ms,
                t.timings.encode_ms,
                t.timings.decode_ms,
            ),
            Err(e) => format!(
                "{{\"file\": \"{}\", \"ok\": false, \"error\": \"run: {}\"}}",
                json_escape(file),
                json_escape(&e.to_string())
            ),
        };
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
    ExitCode::SUCCESS
}
