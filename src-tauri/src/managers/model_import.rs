//! Manual model import — bring-your-own GGUF files, direct URL downloads, and
//! the header-only compat precheck that gates both.
//!
//! [`import_model_file`] copies a local GGUF into the models directory (the
//! same `app_data_dir/models` location [`super::model::ModelManager`] scans for
//! custom models, so an imported file is discovered on the next rescan under
//! the id `filename-stem`). [`import_model_url`] streams an `https://` download
//! into a `*.partial` sibling and only atomically renames it into place after
//! the optional sha256 check and a GGUF magic check pass — a failed download
//! never leaves a half-written model behind. [`check_model_compat`] reads the
//! GGUF header (via [`super::model_capabilities::read_header_metadata_keys`],
//! never the tensor data) and reports hard blockers (unknown architecture,
//! missing/empty vocabulary) separately from soft warnings (quantizations the
//! transcription backend does not ship).

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use specta::Type;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tauri::AppHandle;

use super::gguf_meta::GgufValue;
use super::model_capabilities::read_header_metadata_keys;

/// First four bytes of every GGUF container.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Hard cap on a single URL import.
pub(crate) const MAX_DOWNLOAD_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Connect and per-read (stall) timeout for URL downloads, in the spirit of
/// the model manager's stall watchdog but simplified to reqwest timeouts.
const NET_TIMEOUT: Duration = Duration::from_secs(60);

/// Fallback filename when a download URL carries none.
const DEFAULT_DOWNLOAD_NAME: &str = "model.gguf";

/// Architectures the compat precheck treats as loadable; anything else is a
/// hard block (distinct from [`super::model_capabilities::KNOWN_ARCHES`], which
/// is the broader transcribe-cpp load list used by search listings).
pub(crate) const COMPAT_ARCHES: &[&str] =
    &["whisper", "qwen3_asr", "funasr", "sensevoice", "parakeet"];

const KEY_ARCH: &str = "general.architecture";
const KEY_FILE_TYPE: &str = "general.file_type";
const KEY_TOKENIZER_MODEL: &str = "tokenizer.ggml.model";
/// Keys of the first header pass; `{arch}.vocab_size` is added once the
/// architecture is known (second pass).
const FIRST_PASS_KEYS: &[&str] = &[KEY_ARCH, KEY_FILE_TYPE, KEY_TOKENIZER_MODEL];

/// `general.file_type` values (llama.cpp's quantization enum) the classic
/// whisper.cpp-era backend ships. K/I-quants parse fine but warn.
const BACKEND_SUPPORTED_FILE_TYPES: &[u64] = &[
    0, /* F32 */
    1, /* F16 */
    2, /* Q4_0 */
    3, /* Q4_1 */
    7, /* Q8_0 */
    8, /* Q5_0 */
    9, /* Q5_1 */
];

/// A model that landed in the models directory.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ImportedModel {
    /// Filename stem — the id custom-model discovery registers it under.
    pub id: String,
    /// Final filename inside the models directory (deduped, never overwriting).
    pub filename: String,
    pub size_bytes: u64,
}

/// Header-only compatibility verdict for a GGUF file.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Type)]
pub struct ModelCompat {
    pub architecture: String,
    /// Human-readable quantization from `general.file_type` ("F16", "Q2_K", …).
    pub quant: String,
    pub vocab_size: u64,
    /// `tokenizer.ggml.model` present in the header.
    pub has_tokenizer: bool,
    /// True when nothing hard-blocks loading; quant warnings do not flip this.
    pub backend_supported: bool,
    /// Hard reasons the backend cannot load this model.
    pub blocked: Vec<String>,
    /// Soft concerns — the model may still load or degrade gracefully.
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested below)
// ---------------------------------------------------------------------------

/// `general.file_type` → quantization name; `None`/unmapped values stay honest
/// as "unknown" / `ft<N>" rather than guessing.
pub(crate) fn file_type_quant(file_type: Option<u64>) -> String {
    match file_type {
        None => "unknown".to_string(),
        Some(0) => "F32".to_string(),
        Some(1) => "F16".to_string(),
        Some(2) => "Q4_0".to_string(),
        Some(3) => "Q4_1".to_string(),
        Some(7) => "Q8_0".to_string(),
        Some(8) => "Q5_0".to_string(),
        Some(9) => "Q5_1".to_string(),
        Some(10) => "Q2_K".to_string(),
        Some(11) => "Q3_K_S".to_string(),
        Some(12) => "Q3_K_M".to_string(),
        Some(13) => "Q3_K_L".to_string(),
        Some(14) => "Q4_K_S".to_string(),
        Some(15) => "Q4_K_M".to_string(),
        Some(16) => "Q5_K_S".to_string(),
        Some(17) => "Q5_K_M".to_string(),
        Some(18) => "Q6_K".to_string(),
        Some(19) => "IQ2_XXS".to_string(),
        Some(20) => "IQ2_XS".to_string(),
        Some(21) => "Q2_K_S".to_string(),
        Some(ft) => format!("ft{ft}"),
    }
}

/// Whether the transcription backend ships the quantization named by this
/// `general.file_type` value.
pub(crate) fn backend_supports_file_type(file_type: u64) -> bool {
    BACKEND_SUPPORTED_FILE_TYPES.contains(&file_type)
}

/// The compat verdict, given the header facts. Hard blocks: architecture not
/// in [`COMPAT_ARCHES`] (or missing) and — only when the architecture is known,
/// because the key name `{arch}.vocab_size` is otherwise unknowable — a missing
/// or zero vocabulary. Quantization and tokenizer gaps stay warnings.
pub(crate) fn evaluate_compat(
    architecture: &str,
    file_type: Option<u64>,
    vocab_size: Option<u64>,
    has_tokenizer: bool,
) -> ModelCompat {
    let mut blocked = Vec::new();
    let mut warnings = Vec::new();

    let arch_known = COMPAT_ARCHES.contains(&architecture);
    if architecture.is_empty() {
        blocked.push(format!(
            "missing {KEY_ARCH} (supported: {})",
            COMPAT_ARCHES.join(", ")
        ));
    } else if !arch_known {
        blocked.push(format!(
            "unknown architecture '{architecture}' (supported: {})",
            COMPAT_ARCHES.join(", ")
        ));
    } else {
        let arch_label = if architecture.is_empty() {
            "model"
        } else {
            architecture
        };
        match vocab_size {
            None => blocked.push(format!("missing {arch_label}.vocab_size")),
            Some(0) => blocked.push(format!("{arch_label}.vocab_size is 0 (no vocabulary)")),
            Some(_) => {}
        }
    }

    let quant = file_type_quant(file_type);
    match file_type {
        None => warnings.push(format!("missing {KEY_FILE_TYPE}; quantization unknown")),
        Some(ft) if !backend_supports_file_type(ft) => warnings.push(format!(
            "quantization {quant} ({KEY_FILE_TYPE}={ft}) is not supported by the transcription backend; the model may fail to load"
        )),
        Some(_) => {}
    }
    if !has_tokenizer {
        warnings.push(format!(
            "no tokenizer metadata ({KEY_TOKENIZER_MODEL}) in the header"
        ));
    }

    ModelCompat {
        architecture: architecture.to_string(),
        quant,
        vocab_size: vocab_size.unwrap_or(0),
        has_tokenizer,
        backend_supported: blocked.is_empty(),
        blocked,
        warnings,
    }
}

/// `dir/filename`, or `dir/{stem}-2{ext}`, `-3`, … when taken — imports never
/// overwrite an existing model.
pub(crate) fn unique_destination(dir: &Path, filename: &str) -> PathBuf {
    let direct = dir.join(filename);
    if !direct.exists() {
        return direct;
    }
    let as_path = Path::new(filename);
    let stem = as_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let ext = as_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{e}"))
        .unwrap_or_default();
    for n in 2u64.. {
        let candidate = dir.join(format!("{stem}-{n}{ext}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("an unclaimed numeric suffix always exists")
}

/// Parse and enforce an `https://` URL with a host.
pub(crate) fn validate_https_url(raw: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(raw.trim()).map_err(|e| format!("invalid URL: {e}"))?;
    if url.scheme() != "https" {
        return Err(format!(
            "only https:// URLs are accepted, got '{}://'",
            url.scheme()
        ));
    }
    if url.host_str().is_none() {
        return Err("URL has no host".to_string());
    }
    Ok(url)
}

/// Decode `%XX` escapes in a URL path segment (kept local to avoid pulling in
/// a percent-encoding crate for one call site).
fn percent_decode(s: &str) -> String {
    fn hex_val(b: u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push(hi * 16 + lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Last path segment of a download URL, sanitized to a portable filename.
/// URLs without a usable segment fall back to [`DEFAULT_DOWNLOAD_NAME`].
pub(crate) fn filename_from_url(url: &reqwest::Url) -> Result<String, String> {
    let raw = percent_decode(
        url.path_segments()
            .and_then(|mut segments| segments.next_back())
            .unwrap_or("")
            .trim(),
    );
    if raw.is_empty() || raw == "." || raw == ".." || raw.contains('\\') || raw.contains('/') {
        return Ok(DEFAULT_DOWNLOAD_NAME.to_string());
    }
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        return Ok(DEFAULT_DOWNLOAD_NAME.to_string());
    }
    Ok(sanitized)
}

/// Fail unless the file starts with the GGUF magic.
pub(crate) fn has_gguf_magic(path: &Path) -> Result<(), String> {
    let mut magic = [0u8; 4];
    let mut file = File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    match file.read_exact(&mut magic) {
        Ok(()) if &magic == GGUF_MAGIC => Ok(()),
        Ok(()) => Err(format!("{} is not a GGUF file (bad magic)", path.display())),
        Err(_) => Err(format!(
            "{} is not a GGUF file (too short to read the magic)",
            path.display()
        )),
    }
}

/// Lowercase hex sha256 of a file, streamed so an 8 GiB import stays bounded.
pub(crate) fn sha256_hex(path: &Path) -> Result<String, std::io::Error> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 << 10];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Verify + promote a finished `.partial` download: optional sha256 check,
/// GGUF magic check, then atomic rename. Any failure removes the partial so a
/// broken download never masquerades as a model.
pub(crate) fn promote_partial(
    partial: &Path,
    final_path: &Path,
    expected_sha256: Option<&str>,
) -> Result<(), String> {
    let fail = |msg: String| -> String {
        let _ = fs::remove_file(partial);
        msg
    };
    if let Some(expected) = expected_sha256.map(str::trim).filter(|s| !s.is_empty()) {
        let computed =
            sha256_hex(partial).map_err(|e| fail(format!("cannot hash download: {e}")))?;
        if !computed.eq_ignore_ascii_case(expected) {
            return Err(fail(format!(
                "sha256 mismatch: expected {expected}, got {computed}"
            )));
        }
    }
    has_gguf_magic(partial).map_err(fail)?;
    fs::rename(partial, final_path).map_err(|e| fail(format!("failed to finalize download: {e}")))
}

// ---------------------------------------------------------------------------
// Command plumbing
// ---------------------------------------------------------------------------

/// The models directory custom-model discovery scans (`app_data_dir/models`).
fn models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = crate::portable::app_data_dir(app)
        .map_err(|e| format!("failed to resolve app data dir: {e}"))?
        .join("models");
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create models directory: {e}"))?;
    Ok(dir)
}

fn imported_model_from_path(path: &Path, size_bytes: u64) -> ImportedModel {
    ImportedModel {
        id: path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string(),
        filename: path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string(),
        size_bytes,
    }
}

/// Validate and copy a local GGUF into the models directory.
///
/// Requires a `.gguf` extension and a valid GGUF magic; an existing file of
/// the same name is never overwritten (numeric `-2`/`-3`… suffix instead).
#[tauri::command]
#[specta::specta]
pub fn import_model_file(app: AppHandle, path: String) -> Result<ImportedModel, String> {
    let dir = models_dir(&app)?;
    import_model_file_blocking(&dir, &path)
}

/// The [`import_model_file`] body, split out so the copy can run on a blocking
/// thread and be tested without a Tauri app handle.
fn import_model_file_blocking(models_dir: &Path, path: &str) -> Result<ImportedModel, String> {
    let src = PathBuf::from(path);
    let filename = src
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| "import path has no file name".to_string())?;
    let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("");
    if !ext.eq_ignore_ascii_case("gguf") {
        return Err(format!(
            "'{filename}' is not a .gguf model; only GGUF files can be imported"
        ));
    }
    if !src.is_file() {
        return Err(format!("model file not found: {path}"));
    }
    has_gguf_magic(&src)?;

    let dest = unique_destination(models_dir, &filename);
    fs::copy(&src, &dest)
        .map_err(|e| format!("failed to copy model into the models directory: {e}"))?;
    let size_bytes = fs::metadata(&dest)
        .map(|m| m.len())
        .map_err(|e| format!("failed to stat imported model: {e}"))?;
    Ok(imported_model_from_path(&dest, size_bytes))
}

/// Download an `https://` GGUF into the models directory.
///
/// Streams into `{name}.partial` with 60 s connect/stall timeouts and an
/// 8 GiB cap; on completion the optional `expected_sha256` (hex) and the GGUF
/// magic are verified before an atomic rename. Any failure removes the partial.
#[tauri::command]
#[specta::specta]
pub async fn import_model_url(
    app: AppHandle,
    url: String,
    expected_sha256: Option<String>,
) -> Result<ImportedModel, String> {
    let url = validate_https_url(&url)?;
    let filename = filename_from_url(&url)?;
    let dir = models_dir(&app)?;
    let dest = unique_destination(&dir, &filename);
    let final_name = dest
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(DEFAULT_DOWNLOAD_NAME)
        .to_string();
    let partial = dir.join(format!("{final_name}.partial"));

    let fail = |msg: String| -> String {
        let _ = fs::remove_file(&partial);
        msg
    };

    let client = reqwest::Client::builder()
        .connect_timeout(NET_TIMEOUT)
        .read_timeout(NET_TIMEOUT)
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| fail(format!("download failed: {e}")))?;
    if !response.status().is_success() {
        return Err(fail(format!("download failed: HTTP {}", response.status())));
    }

    let mut file =
        File::create(&partial).map_err(|e| fail(format!("cannot create download file: {e}")))?;
    let mut stream = response.bytes_stream();
    let mut total: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| fail(format!("download failed: {e}")))?;
        total += chunk.len() as u64;
        if total > MAX_DOWNLOAD_BYTES {
            return Err(fail(format!(
                "download exceeds the {} GiB limit",
                MAX_DOWNLOAD_BYTES / (1024 * 1024 * 1024)
            )));
        }
        file.write_all(&chunk)
            .map_err(|e| fail(format!("failed writing download: {e}")))?;
    }
    file.flush()
        .and_then(|()| file.sync_all())
        .map_err(|e| fail(format!("failed writing download: {e}")))?;

    promote_partial(&partial, &dest, expected_sha256.as_deref())?;
    let size_bytes = fs::metadata(&dest)
        .map(|m| m.len())
        .map_err(|e| format!("failed to stat downloaded model: {e}"))?;
    Ok(imported_model_from_path(&dest, size_bytes))
}

/// Header-only compatibility report for a GGUF file. Never fails: an
/// unreadable or non-GGUF file is reported through `blocked`.
#[tauri::command]
#[specta::specta]
pub fn check_model_compat(path: String) -> ModelCompat {
    let path = PathBuf::from(&path);
    let meta = match read_header_metadata_keys(&path, FIRST_PASS_KEYS) {
        Ok(meta) => meta,
        Err(e) => {
            return ModelCompat {
                architecture: String::new(),
                quant: file_type_quant(None),
                vocab_size: 0,
                has_tokenizer: false,
                backend_supported: false,
                blocked: vec![format!(
                    "cannot read GGUF header of {}: {e}",
                    path.display()
                )],
                warnings: Vec::new(),
            };
        }
    };
    let architecture = meta.get_str(KEY_ARCH).unwrap_or("").to_string();
    let file_type = meta.kv.get(KEY_FILE_TYPE).and_then(GgufValue::as_u64);
    let has_tokenizer = meta.get_str(KEY_TOKENIZER_MODEL).is_some();
    // The vocabulary key is architecture-specific, so it can only be requested
    // once the first pass named the architecture.
    let vocab_size = if COMPAT_ARCHES.contains(&architecture.as_str()) {
        let vocab_key = format!("{architecture}.vocab_size");
        read_header_metadata_keys(&path, &[vocab_key.as_str()])
            .ok()
            .and_then(|meta| meta.kv.get(vocab_key.as_str()).and_then(GgufValue::as_u64))
    } else {
        None
    };
    evaluate_compat(&architecture, file_type, vocab_size, has_tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn unique_destination_appends_numeric_suffixes_without_overwriting() {
        let dir = tempfile::tempdir().unwrap();
        // Fresh name maps to itself.
        assert_eq!(
            unique_destination(dir.path(), "model.gguf"),
            dir.path().join("model.gguf")
        );
        // Existing model.gguf + model-2.gguf → model-3.gguf, extension intact.
        write_file(&dir.path().join("model.gguf"), b"GGUF");
        write_file(&dir.path().join("model-2.gguf"), b"GGUF");
        assert_eq!(
            unique_destination(dir.path(), "model.gguf"),
            dir.path().join("model-3.gguf")
        );
        // And the pre-existing files are untouched.
        assert!(dir.path().join("model.gguf").is_file());
        assert!(dir.path().join("model-2.gguf").is_file());
    }

    #[test]
    fn import_copies_valid_gguf_and_dedupes_names() {
        let src_dir = tempfile::tempdir().unwrap();
        let models = tempfile::tempdir().unwrap();
        let src = src_dir.path().join("my-model.gguf");
        write_file(&src, b"GGUFv3-header-bytes");

        let first = import_model_file_blocking(models.path(), src.to_str().unwrap()).unwrap();
        assert_eq!(first.id, "my-model");
        assert_eq!(first.filename, "my-model.gguf");
        assert_eq!(first.size_bytes, b"GGUFv3-header-bytes".len() as u64);

        // Importing the same file again must not clobber — it lands as -2.
        let second = import_model_file_blocking(models.path(), src.to_str().unwrap()).unwrap();
        assert_eq!(second.id, "my-model-2");
        assert_eq!(second.filename, "my-model-2.gguf");
        assert!(models.path().join("my-model.gguf").is_file());
        assert!(models.path().join("my-model-2.gguf").is_file());
    }

    #[test]
    fn import_rejects_non_gguf_extension_and_bad_magic() {
        let src_dir = tempfile::tempdir().unwrap();
        let models = tempfile::tempdir().unwrap();

        let bin = src_dir.path().join("model.bin");
        write_file(&bin, b"GGUF");
        let err = import_model_file_blocking(models.path(), bin.to_str().unwrap()).unwrap_err();
        assert!(err.contains(".gguf"), "extension rejection: {err}");

        let fake = src_dir.path().join("fake.gguf");
        write_file(&fake, b"JUNKJUNK");
        let err = import_model_file_blocking(models.path(), fake.to_str().unwrap()).unwrap_err();
        assert!(err.contains("not a GGUF"), "magic rejection: {err}");
        // Neither rejection leaves anything behind.
        assert_eq!(fs::read_dir(models.path()).unwrap().count(), 0);
    }

    #[test]
    fn promote_partial_deletes_file_on_sha256_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let partial = dir.path().join("m.gguf.partial");
        write_file(&partial, b"GGUF-hello");
        let wrong = sha256_hex(&partial).unwrap().replace('0', "1");
        // Not just the wrong hash — also make sure it differs.
        let wrong = if wrong == sha256_hex(&partial).unwrap() {
            format!("{}f", wrong.trim_end_matches('f'))
        } else {
            wrong
        };
        let err = promote_partial(&partial, &dir.path().join("m.gguf"), Some(&wrong)).unwrap_err();
        assert!(err.contains("sha256 mismatch"), "{err}");
        assert!(!partial.exists(), "partial must be removed on mismatch");
        assert!(!dir.path().join("m.gguf").exists());
    }

    #[test]
    fn promote_partial_renames_on_matching_sha_and_magic() {
        let dir = tempfile::tempdir().unwrap();
        let partial = dir.path().join("m.gguf.partial");
        write_file(&partial, b"GGUF-payload");
        let expected = sha256_hex(&partial).unwrap();
        let dest = dir.path().join("m.gguf");
        promote_partial(&partial, &dest, Some(&expected)).unwrap();
        assert!(dest.is_file());
        assert!(!partial.exists());
    }

    #[test]
    fn promote_partial_rejects_non_gguf_magic_even_with_valid_sha() {
        let dir = tempfile::tempdir().unwrap();
        let partial = dir.path().join("m.gguf.partial");
        write_file(&partial, b"PK-zip-not-gguf");
        let expected = sha256_hex(&partial).unwrap();
        let err =
            promote_partial(&partial, &dir.path().join("m.gguf"), Some(&expected)).unwrap_err();
        assert!(err.contains("not a GGUF"), "{err}");
        assert!(!partial.exists(), "partial must be removed on bad magic");
    }

    #[test]
    fn validate_https_url_rejects_non_https_schemes() {
        assert!(validate_https_url("http://example.com/m.gguf").is_err());
        assert!(validate_https_url("ftp://example.com/m.gguf").is_err());
        assert!(validate_https_url("file:///tmp/m.gguf").is_err());
        assert!(validate_https_url("not a url at all").is_err());
        let ok = validate_https_url("https://example.com/m.gguf").unwrap();
        assert_eq!(ok.as_str(), "https://example.com/m.gguf");
        // Case-insensitive scheme check.
        assert!(validate_https_url("HTTPS://example.com/m.gguf").is_ok());
    }

    #[test]
    fn filename_from_url_sanitizes_and_defaults() {
        let url = reqwest::Url::parse("https://h.example/a/b/model.gguf?x=1").unwrap();
        assert_eq!(filename_from_url(&url).unwrap(), "model.gguf");
        let odd = reqwest::Url::parse("https://h.example/a/My Model%20v2.GGUF").unwrap();
        assert_eq!(filename_from_url(&odd).unwrap(), "My_Model_v2.GGUF");
        let bare = reqwest::Url::parse("https://h.example/").unwrap();
        assert_eq!(filename_from_url(&bare).unwrap(), "model.gguf");
        let dotdot = reqwest::Url::parse("https://h.example/a/..").unwrap();
        assert_eq!(filename_from_url(&dotdot).unwrap(), "model.gguf");
    }

    #[test]
    fn unknown_architecture_is_blocked() {
        let compat = evaluate_compat("llama", Some(1), None, false);
        assert!(!compat.backend_supported);
        assert!(!compat.blocked.is_empty());
        assert!(compat.blocked[0].contains("unknown architecture 'llama'"));
        // Architecture unknown → the vocab key is unknowable, so no noisy
        // second block about a missing vocabulary.
        assert_eq!(compat.blocked.len(), 1);
        // Missing architecture blocks too.
        let compat = evaluate_compat("", None, None, true);
        assert!(compat
            .blocked
            .iter()
            .any(|b| b.contains("missing general.architecture")));
    }

    #[test]
    fn missing_or_zero_vocab_is_blocked_for_known_arch() {
        let compat = evaluate_compat("whisper", Some(1), None, true);
        assert!(!compat.backend_supported);
        assert!(compat
            .blocked
            .iter()
            .any(|b| b.contains("whisper.vocab_size")));

        let compat = evaluate_compat("sensevoice", Some(1), Some(0), true);
        assert!(!compat.backend_supported);
        assert!(compat
            .blocked
            .iter()
            .any(|b| b.contains("is 0 (no vocabulary)")));

        // A real vocab clears the blocks entirely.
        let compat = evaluate_compat("whisper", Some(1), Some(51865), true);
        assert_eq!(compat.blocked, Vec::<String>::new());
        assert!(compat.backend_supported);
        assert_eq!(compat.vocab_size, 51865);
        assert_eq!(compat.quant, "F16");
    }

    #[test]
    fn unsupported_quant_only_warns() {
        // Q2_K (file_type 10) parses but the backend does not ship it.
        let compat = evaluate_compat("whisper", Some(10), Some(51865), true);
        assert_eq!(compat.blocked, Vec::<String>::new());
        assert!(
            compat.backend_supported,
            "quant gaps are warnings, not blocks"
        );
        assert!(compat.warnings.iter().any(|w| w.contains("Q2_K")));
        assert_eq!(compat.quant, "Q2_K");

        // A shipped quant with a tokenizer has no warnings at all.
        let compat = evaluate_compat("parakeet", Some(1), Some(1024), true);
        assert!(compat.warnings.is_empty());

        // Missing tokenizer metadata is a warning too.
        let compat = evaluate_compat("parakeet", Some(1), Some(1024), false);
        assert!(compat.warnings.iter().any(|w| w.contains("tokenizer")));
    }

    #[test]
    fn file_type_quant_names_and_fallbacks() {
        assert_eq!(file_type_quant(None), "unknown");
        assert_eq!(file_type_quant(Some(0)), "F32");
        assert_eq!(file_type_quant(Some(7)), "Q8_0");
        assert_eq!(file_type_quant(Some(18)), "Q6_K");
        assert_eq!(file_type_quant(Some(999)), "ft999");
        assert!(backend_supports_file_type(2));
        assert!(!backend_supports_file_type(10));
    }
}
