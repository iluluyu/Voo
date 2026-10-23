use crate::audio_toolkit::chinese_hotwords;
use crate::audio_toolkit::chunking;
use crate::audio_toolkit::{
    apply_custom_words, detect_output_language, normalize_transcription_output,
    remove_filler_words, OutputLanguageEvidence,
};
use crate::chinese_script::{convert_chinese_script, ChineseVariety};
use crate::engine_supervisor::{
    CancelMark, DeviceInfo, DeviceSelector, EngineError, EngineSupervisor, LoadSpec, LoadedInfo,
    Unloading,
};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::model::ModelManager;
use crate::settings::{
    get_settings, AppSettings, ChineseScript, ModelUnloadTimeout, TranscribeAcceleratorSetting,
};
use anyhow::Result;
use chunked::ChunkedDecode;
use live::LiveSession;
use log::{debug, error, info, warn};
use policy::{batch_decode, live_decode, BatchDecode, ModelTraits};
use serde::{Deserialize, Serialize};
use specta::Type;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use tauri::{AppHandle, Emitter, Manager};
use tauri_specta::Event;
use transcribe_cpp::{Backend, Itn, RunOptions, Task, Transcript};

mod chunked;
pub(crate) mod commit;
mod live;
pub mod policy;

const STREAM_PERF_LOG_INTERVAL: Duration = Duration::from_secs(5);

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStateEvent {
    pub event_type: String,
    pub model_id: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

/// Live transcription snapshot emitted to the overlay during a streaming run.
/// `committed` is the append-only, flicker-free prefix; `tentative` is the
/// volatile suffix the model may still rewrite.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
pub struct StreamTextEvent {
    pub committed: String,
    pub tentative: String,
}

/// Phase of the streaming overlay card, emitted to drive its UI state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum StreamPhase {
    /// Receiving audio / live text (or waiting for the stream to begin). Rust
    /// does not emit this today; the frontend starts in this phase and Rust only
    /// emits transitions away from it.
    Listening,
    /// Finalizing or post-processing — show a spinner.
    Working,
}

/// Semantic kind of "working" phase, used to localize the spinner label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum StreamWorkKind {
    Transcribing,
    Polishing,
}

/// Emitted to switch the streaming overlay to a working spinner.
#[derive(Clone, Debug, Serialize, Deserialize, Type, tauri_specta::Event)]
pub struct StreamPhaseEvent {
    pub phase: StreamPhase,
    /// Present only when `phase` is `Working`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<StreamWorkKind>,
}

/// Commands sent to the streaming worker thread. Audio frames and the finalize
/// request travel the same channel so FIFO ordering guarantees every fed frame
/// is processed before finalize runs.
enum StreamCmd {
    Feed(Vec<f32>),
    /// Flush the live decode and reply with what it decoded, or `None` if
    /// there is no usable live result (caller should fall back to batch
    /// transcription). `Err(Cancelled)` if the finalize itself was cancelled.
    /// `samples` is the whole recording's length, for a chunked decode that
    /// checks it saw it all.
    Finalize {
        reply: mpsc::Sender<StreamFinalizeReply>,
        samples: usize,
    },
    Cancel,
}

type StreamFinalizeReply = Result<Option<RunOutcome>, EngineError>;

/// Routes real-time audio frames to the active streaming worker. Shared between
/// the [`TranscriptionManager`] (opens/closes the route) and the audio recorder's
/// per-frame callback (feeds frames). The recorder holds an `Arc<StreamRouter>`
/// directly, so a frame with no stream pending costs a single relaxed atomic
/// load — no Tauri state lookup, no mutex lock.
pub struct StreamRouter {
    /// The active streaming worker's route, present from `start_stream`
    /// until `finalize_live`/`cancel_stream`.
    route: Mutex<Option<Route>>,
    /// True while a stream is pending or active (channel is open). The audio
    /// callback checks this first to avoid the mutex lock when no stream runs.
    open: Arc<AtomicBool>,
}

/// One streaming worker's command channel.
struct Route {
    tx: mpsc::Sender<StreamCmd>,
    /// Set by `cancel_stream`, so the worker drops audio still queued ahead
    /// of the `Cancel` instead of decoding it.
    cancelled: Arc<AtomicBool>,
}

impl StreamRouter {
    fn new() -> Self {
        Self {
            route: Mutex::new(None),
            open: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Open a fresh command channel for a new streaming session, returning the
    /// receiver the worker should drain and its cancel flag. Caller must
    /// ensure no prior channel is still open.
    fn open(&self) -> (mpsc::Receiver<StreamCmd>, Arc<AtomicBool>) {
        let (tx, rx) = mpsc::channel::<StreamCmd>();
        let cancelled = Arc::new(AtomicBool::new(false));
        *self.route.lock().unwrap() = Some(Route {
            tx,
            cancelled: Arc::clone(&cancelled),
        });
        self.open.store(true, Ordering::Relaxed);
        (rx, cancelled)
    }

    /// Take the route out (closing the channel to new feeds), so the caller
    /// can send the final `Finalize`/`Cancel` command.
    fn take(&self) -> Option<Route> {
        self.open.store(false, Ordering::Relaxed);
        self.route.lock().unwrap().take()
    }

    /// Drop the channel and mark closed without sending a final command (used
    /// when the worker exits without a finalize/cancel handshake).
    fn clear(&self) {
        self.open.store(false, Ordering::Relaxed);
        *self.route.lock().unwrap() = None;
    }

    /// Forward a 16 kHz frame to the active streaming worker. Cheap no-op (a
    /// single relaxed atomic load) when no stream is pending.
    pub fn feed(&self, frame: &[f32]) {
        if !self.open.load(Ordering::Relaxed) {
            return;
        }
        if let Some(route) = self.route.lock().unwrap().as_ref() {
            let _ = route.tx.send(StreamCmd::Feed(frame.to_vec()));
        }
    }

    /// Whether a stream is pending or active.
    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Relaxed)
    }
}

/// RAII guard that clears the `is_loading` flag and notifies waiters on drop.
/// Ensures the loading flag is always reset, even on early returns or panics.
pub struct LoadingGuard {
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl Drop for LoadingGuard {
    fn drop(&mut self) {
        // Recover from a poisoned mutex instead of panicking —
        // a panic inside Drop calls abort().
        let mut is_loading = match self.is_loading.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Recovered poisoned is_loading mutex during LoadingGuard drop — a panic occurred earlier this session");
                e.into_inner()
            }
        };
        *is_loading = false;
        self.loading_condvar.notify_all();
    }
}

/// RAII guard that clears the streaming worker flags on any worker exit -
/// normal return, early return, or a panic that unwinds the detached worker
/// thread. Tokens prevent an older worker from clearing a newer worker's
/// state if a start/finalize race ever slips through.
struct StreamWorkerGuard {
    worker_id: u64,
    active_stream_worker: Arc<AtomicU64>,
    stream_active: Arc<AtomicBool>,
}

impl Drop for StreamWorkerGuard {
    fn drop(&mut self) {
        if self.active_stream_worker.load(Ordering::Acquire) == self.worker_id {
            self.stream_active.store(false, Ordering::Release);
        }
        let _ = self.active_stream_worker.compare_exchange(
            self.worker_id,
            0,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

#[derive(Clone)]
pub struct TranscriptionManager {
    /// The transcribe-cpp engine: owns the worker process, its model, and
    /// recovery from crashes and hangs.
    engine: EngineSupervisor,
    model_manager: Arc<ModelManager>,
    app_handle: AppHandle,
    current_model_id: Arc<Mutex<Option<String>>>,
    last_activity: Arc<AtomicU64>,
    shutdown_signal: Arc<AtomicBool>,
    watcher_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
    reload_model_on_next_use: Arc<AtomicBool>,
    /// Routes real-time audio frames to the active streaming worker; see
    /// [`StreamRouter`]. Shared with the audio recorder so per-frame feeds skip
    /// Tauri state and the manager lock.
    router: Arc<StreamRouter>,
    /// True only while a transcribe-cpp stream is actually in flight (set by
    /// the worker once the stream begins). Used for overlay/UI decisions.
    stream_active: Arc<AtomicBool>,
    /// Streaming uses three independent flags: router open = frames should
    /// route, worker active = no second worker may start, stream active = UI
    /// should show a live session.
    ///
    /// Monotonic id source for stream workers; zero means "no worker".
    next_stream_worker_id: Arc<AtomicU64>,
    /// Nonzero while a stream worker exists, even if it has not leased the engine
    /// yet. This prevents a second worker from starting after finalize/cancel
    /// closes the router but before the first worker has fully exited.
    active_stream_worker: Arc<AtomicU64>,
}

impl TranscriptionManager {
    pub fn new(app_handle: &AppHandle, model_manager: Arc<ModelManager>) -> Result<Self> {
        let manager = Self {
            engine: EngineSupervisor::new(!transcribe_gpu_disabled_for_host()),
            model_manager,
            app_handle: app_handle.clone(),
            current_model_id: Arc::new(Mutex::new(None)),
            last_activity: Arc::new(AtomicU64::new(Self::now_ms())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            watcher_handle: Arc::new(Mutex::new(None)),
            is_loading: Arc::new(Mutex::new(false)),
            loading_condvar: Arc::new(Condvar::new()),
            reload_model_on_next_use: Arc::new(AtomicBool::new(false)),
            router: Arc::new(StreamRouter::new()),
            stream_active: Arc::new(AtomicBool::new(false)),
            next_stream_worker_id: Arc::new(AtomicU64::new(1)),
            active_stream_worker: Arc::new(AtomicU64::new(0)),
        };

        // Start the idle watcher
        {
            let app_handle_cloned = app_handle.clone();
            let manager_cloned = manager.clone();
            let shutdown_signal = manager.shutdown_signal.clone();
            let handle = thread::spawn(move || {
                debug!("Idle watcher thread started");
                while !shutdown_signal.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(10)); // Check every 10 seconds

                    // Check shutdown signal again after sleep
                    if shutdown_signal.load(Ordering::Relaxed) {
                        break;
                    }

                    let settings = get_settings(&app_handle_cloned);
                    let timeout = settings.model_unload_timeout;

                    // Skip Immediately — that variant is handled by
                    // maybe_unload_immediately() after each transcription.
                    // Treating it as 0s here would unload the model mid-recording.
                    if timeout == ModelUnloadTimeout::Immediately {
                        continue;
                    }

                    // While recording, keep the idle timer fresh so the
                    // model is never unloaded mid-session.
                    let is_recording = app_handle_cloned
                        .try_state::<Arc<AudioRecordingManager>>()
                        .is_some_and(|a| a.is_recording());
                    if is_recording {
                        manager_cloned.touch_activity();
                        continue;
                    }

                    if let Some(limit_seconds) = timeout.to_seconds() {
                        let last = manager_cloned.last_activity.load(Ordering::Relaxed);
                        let now_ms = TranscriptionManager::now_ms();
                        let idle_ms = now_ms.saturating_sub(last);
                        let limit_ms = limit_seconds * 1000;

                        if idle_ms > limit_ms {
                            // idle -> unload
                            if manager_cloned.is_model_loaded() {
                                let unload_start = std::time::Instant::now();
                                info!(
                                    "Model idle for {}s (limit: {}s), unloading",
                                    idle_ms / 1000,
                                    limit_seconds
                                );
                                match manager_cloned.unload_model() {
                                    Ok(()) => {
                                        let unload_duration = unload_start.elapsed();
                                        info!(
                                            "Model unloaded due to inactivity (took {}ms)",
                                            unload_duration.as_millis()
                                        );
                                    }
                                    Err(e) => {
                                        error!("Failed to unload idle model: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                debug!("Idle watcher thread shutting down gracefully");
            });
            *manager.watcher_handle.lock().unwrap() = Some(handle);
        }

        Ok(manager)
    }

    pub fn is_model_loaded(&self) -> bool {
        // A transcribe-cpp model stays loaded while it is busy, so a batch
        // run or stream in progress never reads as "unloaded".
        self.engine.loaded().is_some()
    }

    /// Accelerator changes should not disturb the current transcription. Mark
    /// the cached engine stale; the next model-use path reloads it with the
    /// latest settings.
    pub fn reload_model_on_next_use(&self) {
        self.reload_model_on_next_use.store(true, Ordering::Release);
    }

    /// An explicit transcribe.cpp accelerator or GPU device change is the
    /// user's way of trying the GPU again: forget every GPU failure so far,
    /// so the next load may use (and list) the GPU.
    pub fn retry_transcribe_gpu(&self, reason: &str) {
        self.engine.retry_gpu(reason);
    }

    /// The engine gives up on a model by itself when its worker can't be
    /// restarted (e.g. the file is gone, or it crashes on CPU too). Keep the
    /// loaded model id and the UI in step, so the next use loads the model
    /// again and reports why it can't.
    fn forget_model_if_engine_dropped(&self, model_id: &str, error: &str) {
        if self.engine.loaded().is_some() {
            return;
        }
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            if current_model.as_deref() != Some(model_id) {
                return;
            }
            *current_model = None;
        }
        warn!(
            "Transcription model '{}' is no longer loaded: {}",
            model_id, error
        );
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "unloaded".to_string(),
                model_id: None,
                model_name: None,
                error: Some(error.to_string()),
            },
        );
    }

    /// Atomically check whether a model load is in progress and, if not, mark
    /// one as starting. Returns a [`LoadingGuard`] whose [`Drop`] impl will
    /// clear the flag and wake waiters. Returns `None` if a load is already in
    /// progress.
    pub fn try_start_loading(&self) -> Option<LoadingGuard> {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return None;
        }
        *is_loading = true;
        Some(LoadingGuard {
            is_loading: self.is_loading.clone(),
            loading_condvar: self.loading_condvar.clone(),
        })
    }

    /// Unload the model. Returns once the transcribe-cpp worker has exited,
    /// which frees all of its memory (e.g. before its file is deleted).
    pub fn unload_model(&self) -> Result<()> {
        let unload_start = std::time::Instant::now();
        debug!("Starting to unload model");
        self.begin_unload().wait();
        debug!(
            "Model unloaded manually (took {}ms)",
            unload_start.elapsed().as_millis()
        );
        Ok(())
    }

    /// Unload the model without waiting for the transcribe-cpp worker to
    /// exit, so it never blocks the caller (e.g. the event loop) behind a
    /// transcription in progress. The model reads as unloaded at once.
    pub fn request_unload(&self) {
        drop(self.begin_unload());
    }

    /// Every state change of an unload, done now on the caller's thread: a
    /// dictation started right after sees no model and loads it again, and
    /// its load is queued behind this unload, so the unload can never affect
    /// it. Only the worker's exit is left to wait for.
    fn begin_unload(&self) -> Unloading {
        let unloading = self.engine.unload();
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // Emit unloaded event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "unloaded".to_string(),
                model_id: None,
                model_name: None,
                error: None,
            },
        );
        unloading
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Reset the idle timer to now.
    fn touch_activity(&self) {
        self.last_activity.store(Self::now_ms(), Ordering::Relaxed);
    }

    /// Unloads the model immediately if the setting is enabled and the model
    /// is loaded. Never waits for the worker to exit, so it is safe on any
    /// thread, including the event loop.
    pub fn maybe_unload_immediately(&self, context: &str) {
        let settings = get_settings(&self.app_handle);
        if settings.model_unload_timeout == ModelUnloadTimeout::Immediately
            && self.is_model_loaded()
        {
            info!("Immediately unloading model after {}", context);
            self.request_unload();
        }
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        self.load_model_with_device(model_id, None)
    }

    /// Like [`load_model`](Self::load_model), but lets a caller hard-select the
    /// compute device for this one load by its transcribe-cpp device
    /// registry index (the index shown by `--list-devices`). `None` keeps the
    /// persisted accelerator setting (which may be Auto). Only affects
    /// transcribe-cpp (whisper-family) models; the selection is not persisted.
    pub fn load_model_with_device(
        &self,
        model_id: &str,
        device_index: Option<usize>,
    ) -> Result<()> {
        let load_start = std::time::Instant::now();
        debug!("Starting to load model: {}", model_id);

        // Emit loading started event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_started".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: None,
                error: None,
            },
        );

        let model_info = match self.model_manager.get_model_info(model_id) {
            Some(model_info) => model_info,
            None => {
                let error_msg = format!("Model not found: {}", model_id);
                let _ = self.app_handle.emit(
                    "model-state-changed",
                    ModelStateEvent {
                        event_type: "loading_failed".to_string(),
                        model_id: Some(model_id.to_string()),
                        model_name: None,
                        error: Some(error_msg.clone()),
                    },
                );
                return Err(anyhow::anyhow!(error_msg));
            }
        };

        // Every failure after loading starts must emit a terminal event so the
        // frontend can never remain in its loading state.
        let emit_loading_failed = |error_msg: &str| {
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
        };

        if !model_info.is_downloaded {
            let error_msg = "Model not downloaded";
            emit_loading_failed(error_msg);
            return Err(anyhow::anyhow!(error_msg));
        }

        let model_path = self
            .model_manager
            .get_model_path(model_id)
            .inspect_err(|error| emit_loading_failed(&error.to_string()))?;

        // Clear the id before loading: if the new load fails, status should
        // read "no loaded model", not the model the worker is replacing.
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = None;
        }

        // The backend is chosen at load time. With an explicit
        // `device_index` (the --device-index flag) hard-select that
        // registered device; otherwise re-read the persisted
        // accelerator preference (so an accelerator change marked for
        // reload takes effect here). The worker resolves the selector
        // to a device handle, and rejects an index that isn't a
        // loadable device (a host without GPU access lists none).
        let (backend, device) = match device_index {
            Some(index) => (Backend::Auto, DeviceSelector::Index(index)),
            None => {
                let settings = get_settings(&self.app_handle);
                let accelerator = settings.transcribe_accelerator;
                match resolve_gpu_device(accelerator, settings.transcribe_gpu_device.as_deref()) {
                    // Backend::Auto accepts an exact GPU device.
                    Some(key) => (Backend::Auto, DeviceSelector::Key(key)),
                    // Without a valid exact device, backend selection
                    // handles the retired generic GPU state and host
                    // CPU guard.
                    None => (select_transcribe_backend(accelerator), DeviceSelector::Auto),
                }
            }
        };
        let requested_device = format!("{:?}", device);
        let info = self
            .engine
            .load(LoadSpec {
                path: model_path.clone(),
                backend,
                device,
            })
            .map_err(|e| {
                let error_msg = format!("Failed to load model {}: {}", model_id, e);
                emit_loading_failed(&error_msg);
                anyhow::anyhow!(error_msg)
            })?;
        // Reconcile the registry's advertised capabilities with the
        // loaded model's real ones (GGUF metadata) so badges/gating
        // reflect runtime truth, not the pre-download probe. The
        // load-completed event below triggers the frontend refresh.
        let caps = &info.capabilities;
        self.model_manager.set_runtime_capabilities(
            model_id,
            caps.supports_streaming,
            caps.supports_translate,
            caps.supports_language_detect,
            caps.languages.clone(),
        );
        // The bound backend may differ from the request (e.g. CPU
        // fallback under Auto or after a GPU crash); log what loaded.
        info!(
            "Loaded transcribe-cpp model '{}' (requested {:?}, requested device {}, \
             bound backend '{}', bound device '{}', supports_streaming={}, \
             supports_translate={}, supports_language_detect={})",
            model_id,
            backend,
            requested_device,
            info.backend,
            info.device,
            caps.supports_streaming,
            caps.supports_translate,
            caps.supports_language_detect
        );

        // Update the current model ID
        {
            let mut current_model = self.current_model_id.lock().unwrap();
            *current_model = Some(model_id.to_string());
        }

        // Reset idle timer so the watcher doesn't immediately unload a just-loaded model
        self.touch_activity();

        // Emit loading completed event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_completed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );

        let load_duration = load_start.elapsed();
        debug!(
            "Successfully loaded transcription model: {} (took {}ms)",
            model_id,
            load_duration.as_millis()
        );
        Ok(())
    }

    /// Kicks off the model loading in a background thread if it's not already loaded
    pub fn initiate_model_load(&self) {
        let mut is_loading = self.is_loading.lock().unwrap();
        if *is_loading {
            return;
        }

        let reload_pending = self.reload_model_on_next_use.load(Ordering::Acquire);
        if !reload_pending && self.is_model_loaded() {
            return;
        }

        *is_loading = true;
        let self_clone = self.clone();
        thread::spawn(move || {
            if reload_pending {
                self_clone
                    .reload_model_on_next_use
                    .store(false, Ordering::Release);
            }
            let settings = get_settings(&self_clone.app_handle);
            if let Err(e) = self_clone.load_model(&settings.selected_model) {
                error!("Failed to load model: {}", e);
            }
            let mut is_loading = self_clone.is_loading.lock().unwrap();
            *is_loading = false;
            self_clone.loading_condvar.notify_all();
        });
    }

    pub fn get_current_model(&self) -> Option<String> {
        let current_model = self.current_model_id.lock().unwrap();
        current_model.clone()
    }

    /// The compute backend the currently-loaded engine is bound to, for
    /// diagnostics (e.g. confirming `--device-index` actually bound a GPU rather
    /// than falling back to CPU/auto); `None` when no model is loaded.
    pub fn current_backend(&self) -> Option<String> {
        self.engine.loaded().map(|info| info.backend)
    }

    /// Whether a live streaming run is currently in flight.
    pub fn is_streaming(&self) -> bool {
        self.stream_active.load(Ordering::Acquire)
    }

    /// Shared handle to the stream router, used by the audio recorder to feed
    /// real-time frames without going through Tauri state on every frame.
    pub fn stream_router(&self) -> Arc<StreamRouter> {
        Arc::clone(&self.router)
    }

    /// Begin a recording's live decode on the held engine's session. Audio
    /// frames pushed via [`StreamRouter::feed`] (captured directly by the
    /// audio recorder) are streamed, with preview text emitted to the overlay
    /// as [`StreamTextEvent`], or cut into long-form chunks decoded as the
    /// recording goes; [`policy::live_decode`] picks which.
    ///
    /// Non-blocking: spawns a worker that waits for any in-progress model load,
    /// then begins the live decode the loaded model calls for. With none, the
    /// worker idles until finalize/cancel and reports `None` so the caller
    /// falls back to batch transcription. Frames sent before the decode begins
    /// queue on the channel and are not lost.
    pub fn start_stream(&self) {
        if self.router.is_open() || self.active_stream_worker.load(Ordering::Acquire) != 0 {
            warn!("start_stream called while a stream worker is already active");
            return;
        }
        let worker_id = self.next_stream_worker_id.fetch_add(1, Ordering::Relaxed);
        if self
            .active_stream_worker
            .compare_exchange(0, worker_id, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            warn!("start_stream lost a race with another stream worker");
            return;
        }
        let (rx, cancelled) = self.router.open();
        self.stream_active.store(false, Ordering::Release);

        let manager = self.clone();
        thread::spawn(move || manager.run_stream_worker(rx, cancelled, worker_id));
    }

    fn run_stream_worker(
        &self,
        rx: mpsc::Receiver<StreamCmd>,
        cancelled: Arc<AtomicBool>,
        worker_id: u64,
    ) {
        let _worker = StreamWorkerGuard {
            worker_id,
            active_stream_worker: Arc::clone(&self.active_stream_worker),
            stream_active: Arc::clone(&self.stream_active),
        };

        // Wait for any in-progress model load to finish (start_stream races the
        // background load kicked off when recording starts).
        {
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }
        }

        let model_id = self.get_current_model().unwrap_or_default();

        // The loaded model (not the ModelManager copy) is the source of truth
        // for run-path capabilities.
        let Some(info) = self.engine.loaded() else {
            info!(
                "Live preview: model '{}' is not a loaded transcribe-cpp model; \
                 streaming is unavailable, using batch transcription",
                model_id
            );
            self.router.clear();
            drain_until_finalize(rx);
            return;
        };
        let caps = &info.capabilities;
        info!(
            "Live preview: model '{}' arch='{}' variant='{}' supports_streaming={} \
             supports_translate={} languages={:?}",
            model_id,
            info.arch,
            info.variant,
            caps.supports_streaming,
            caps.supports_translate,
            caps.languages,
        );
        let mode = live_decode(
            ModelTraits::of(&info),
            get_settings(&self.app_handle).long_form_chunking,
        );
        let Some(mut session) = LiveSession::begin(self, mode, &info, &model_id) else {
            // Nothing to decode live: the recording is transcribed in batch.
            self.router.clear();
            drain_until_finalize(rx);
            return;
        };
        self.touch_activity();

        while let Ok(cmd) = rx.recv() {
            match cmd {
                StreamCmd::Feed(pcm) => {
                    // Audio queued ahead of a cancel is dropped, not decoded.
                    if cancelled.load(Ordering::Acquire) {
                        continue;
                    }
                    self.touch_activity();
                    session.feed(pcm);
                }
                StreamCmd::Finalize { reply, samples } => {
                    let _ = reply.send(session.finish(samples));
                    return;
                }
                // Dropping the session resets the worker's stream.
                StreamCmd::Cancel => return,
            }
        }
        // The channel closed without finalize/cancel; dropping the session
        // resets the worker's stream. `_worker` drops after, clearing this
        // worker's flags.
    }

    /// Flush the live decode (a stream, or chunks decoded while recording)
    /// and return what it produced. `recorded_samples` is the length of the
    /// whole recording: a chunked decode that saw less reports no result.
    ///
    /// `Ok(None)` means no usable live result exists and the caller should
    /// fall back to batch transcription. `Err` means the transcription was
    /// cancelled. There is no timeout here: waiting behind a model load is a
    /// queue position, not a failure (#1841), and the engine bounds the
    /// finalize itself by the audio it still has to decode.
    fn finalize_live(&self, recorded_samples: usize) -> Result<Option<RunOutcome>> {
        let Some(route) = self.router.take() else {
            return Ok(None);
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        let finalize = StreamCmd::Finalize {
            reply: reply_tx,
            samples: recorded_samples,
        };
        if route.tx.send(finalize).is_err() {
            return Ok(None);
        }
        match reply_rx.recv() {
            Ok(Ok(outcome)) => Ok(outcome),
            Ok(Err(e)) => Err(e.into()),
            Err(_) => Ok(None),
        }
    }

    /// Abandon any live decode without producing text (e.g. on cancel). A
    /// chunk already decoding finishes; audio queued behind it is dropped.
    pub fn cancel_stream(&self) {
        if let Some(route) = self.router.take() {
            route.cancelled.store(true, Ordering::Release);
            let _ = route.tx.send(StreamCmd::Cancel);
        }
        self.stream_active.store(false, Ordering::Release);
    }

    /// Stop the transcription in progress (a batch run or stream finalize),
    /// and any queued behind it, for an explicit user cancel. The worker doing
    /// it is stopped; the model stays loaded and the next use starts a fresh
    /// worker for it.
    pub fn cancel_transcription(&self) {
        self.engine.cancel();
    }

    /// Emit a working-phase event to the streaming overlay (spinner + label).
    pub fn emit_stream_working(&self, kind: StreamWorkKind) {
        let _ = StreamPhaseEvent {
            phase: StreamPhase::Working,
            kind: Some(kind),
        }
        .emit(&self.app_handle);
    }

    /// Transcribe a whole clip in batch: a re-transcribed history entry, or
    /// a file from the CLI.
    pub fn transcribe(&self, audio: Vec<f32>) -> Result<String> {
        // Not a recording: any stale incremental-commit remainder from a
        // cancelled previous session must not adjust this paste.
        commit::clear_final_paste_prefix();
        self.complete(Instant::now(), audio, None)
    }

    /// Transcribe a finished recording: what its live decode produced, or a
    /// batch run of the whole recording when there is no live result.
    pub fn transcribe_recording(&self, samples: Vec<f32>) -> Result<String> {
        // This recording's own remainder is armed later, inside finalize;
        // clearing here only retires an unspent one from an earlier session
        // (e.g. a recording cancelled between finalize and paste).
        commit::clear_final_paste_prefix();
        let started = Instant::now();
        // 停顿提交（pause-commit，批处理模型的“停顿即出字”）：段在录制
        // 中已逐段转写并粘贴，这里只等尾段收尾，然后返回空串——调用方
        // 对空结果既不二次粘贴也不做 LLM 后处理。finalize 是幂等兑底，
        // 覆盖未来不经 stop_recording 就走到这里的路径。
        if let Some(audio_manager) = self.app_handle.try_state::<Arc<AudioRecordingManager>>() {
            if let Some(runner) = audio_manager.take_pause_commit_runner() {
                runner.finalize();
                runner.wait();
                let committed = runner.committed_text();
                debug!(
                    "Pause-commit recording finished: {} segment(s), {} committed char(s), tail wait {:?}",
                    runner.committed_segments(),
                    committed.chars().count(),
                    started.elapsed()
                );
                debug!(
                    "Pause-commit committed text: {}",
                    crate::utils::redact_text(&committed)
                );
                return Ok(String::new());
            }
        }
        let live = self.finalize_live(samples.len())?;
        self.complete(started, samples, live)
    }

    /// One committed pause-commit segment, decoded in batch while the
    /// recording continues (or at its tail): the same setup and
    /// post-processing as a whole-clip run, minus the idle-unload check —
    /// more segments of the same recording follow, so the model stays
    /// loaded. Called from the pause-commit worker thread only.
    pub fn transcribe_segment(&self, audio: Vec<f32>) -> Result<String> {
        if audio.is_empty() {
            return Ok(String::new());
        }
        let started = Instant::now();
        let outcome = self.decode_batch(audio)?;
        let settings = get_settings(&self.app_handle);
        let text = finish_text(outcome, &settings);
        debug!(
            "Segment decoded in {:?}: {} char(s)",
            started.elapsed(),
            text.chars().count()
        );
        Ok(text)
    }

    /// The final text of one transcription, from its live result or else a
    /// batch run of `audio`. Every transcription finishes here.
    fn complete(
        &self,
        started: Instant,
        audio: Vec<f32>,
        live: Option<RunOutcome>,
    ) -> Result<String> {
        #[cfg(debug_assertions)]
        if std::env::var("HANDY_FORCE_TRANSCRIPTION_FAILURE").is_ok() {
            return Err(anyhow::anyhow!(
                "Simulated transcription failure (HANDY_FORCE_TRANSCRIPTION_FAILURE)"
            ));
        }

        let audio_len = audio.len();
        debug!("Audio vector length: {}", audio_len);
        if audio.is_empty() {
            debug!("Empty audio vector");
            self.maybe_unload_immediately("empty audio");
            return Ok(String::new());
        }

        let outcome = match live {
            Some(outcome) => outcome,
            None => self.decode_batch(audio)?,
        };
        let settings = get_settings(&self.app_handle);
        let final_result = finish_text(outcome, &settings);

        let translation_note = if settings.translate_to_english {
            " (translated)"
        } else {
            ""
        };
        // Real-time factor. Input PCM is 16 kHz mono, so audio length in seconds
        // is samples / 16000. `speedup` is audio_secs / elapsed_secs — e.g. 4.00x
        // means transcribed 4x faster than real time
        let elapsed_secs = started.elapsed().as_secs_f64();
        let audio_secs = audio_len as f64 / 16_000.0;
        let speedup = real_time_factor(audio_secs, elapsed_secs);
        info!(
            "Transcription completed in {:.2}s for {:.2}s of audio ({:.2}x real-time){}",
            elapsed_secs, audio_secs, speedup, translation_note
        );

        if final_result.is_empty() {
            info!("Transcription result is empty");
        } else {
            info!(
                "Transcription result: {}",
                crate::utils::redact_text(&final_result)
            );
        }

        self.maybe_unload_immediately("transcription");

        Ok(final_result)
    }

    /// Decode a whole clip with the loaded model, loading it first if a load
    /// is under way.
    fn decode_batch(&self, audio: Vec<f32>) -> Result<RunOutcome> {
        // A cancel from here on (during the model-load wait or chunk planning
        // included) stops every run of this transcription.
        let mark = self.engine.cancel_mark();

        // Update last activity timestamp
        self.touch_activity();

        // Check if model is loaded, if not try to load it
        {
            // If the model is loading, wait for it to complete.
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }

            if !self.is_model_loaded() {
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        // Get current settings for configuration
        let settings = get_settings(&self.app_handle);

        // Validate selected language against the model's supported languages.
        // If the language isn't supported, fall back to "auto" to prevent errors.
        // Validate against the model that's actually loaded (which can differ
        // from settings.selected_model when a caller loaded a specific model —
        // e.g. the --transcribe-file path's --model), not the persisted
        // selection.
        let active_model = self
            .get_current_model()
            .unwrap_or_else(|| settings.selected_model.clone());
        // Resolve the persisted language *intent* into the language this model
        // will actually use. The coercion is capability-aware (a must-pick model
        // never receives "auto") and computed fresh here — it is never written
        // back to settings, so the intent survives switching models and back.
        let validated_language =
            effective_language_for_model(&settings, self.model_manager.as_ref(), &active_model);
        if validated_language != settings.selected_language {
            debug!(
                "Language intent '{}' resolved to '{}' for model '{}'",
                settings.selected_language, validated_language, active_model
            );
        }

        let Some(info) = self.engine.loaded() else {
            return Err(anyhow::anyhow!(
                "Model failed to load after auto-load attempt. Please check your model settings."
            ));
        };
        self.transcribe_cpp(
            mark,
            info,
            audio,
            &settings,
            &validated_language,
            &active_model,
        )
    }

    /// transcribe-cpp: the model runs in the engine's worker process, which
    /// recovers from crashes and hangs itself. Long audio is decoded in
    /// chunks cut at pauses, unless the model handles it in one run.
    fn transcribe_cpp(
        &self,
        mark: CancelMark,
        info: LoadedInfo,
        audio: Vec<f32>,
        settings: &AppSettings,
        validated_language: &str,
        active_model: &str,
    ) -> Result<RunOutcome> {
        let setup = run_setup(&info, settings, validated_language);
        // 不做"补静音到固定时长"：sol 判定那是边界 workaround，不是语音证据，
        // 只有在特定模型/后端做过 A/B 证明有益时才能保留。
        let whole = |pcm: Vec<f32>| {
            self.run_batch(mark, pcm, &setup.run_options, active_model)
                .map_err(engine_error)
        };
        // `mark` covers every chunk, so a cancel between two stops the rest.
        // A failed chunk fails the whole transcription.
        let transcripts = match batch_decode(ModelTraits::of(&info), settings.long_form_chunking) {
            BatchDecode::Chunked => ChunkedDecode::decode_clip(
                self,
                mark,
                setup.run_options.clone(),
                audio,
                active_model,
                settings.vad_backend,
            )
            .map_err(engine_error)?,
            BatchDecode::Whole => vec![whole(audio)?],
        };
        let (text, model_detected_language) = join_chunks(transcripts);
        Ok(RunOutcome {
            setup,
            text,
            model_detected_language,
        })
    }

    /// One batch run of a transcription that began at `mark`, so a cancel
    /// since then stops it. Keeps the idle watcher from unloading the model
    /// between the runs of a chunked transcription.
    fn run_batch(
        &self,
        mark: CancelMark,
        pcm: Vec<f32>,
        run_options: &RunOptions,
        active_model: &str,
    ) -> Result<Transcript, EngineError> {
        self.touch_activity();
        self.engine
            .transcribe_since(mark, pcm, run_options.clone())
            .inspect_err(|e| {
                if !matches!(e, EngineError::Cancelled) {
                    self.forget_model_if_engine_dropped(active_model, &e.to_string());
                }
            })
    }
}

/// What one transcription decoded (live or in batch), with how its runs were
/// set up, which post-processing needs to know.
struct RunOutcome {
    setup: RunSetup,
    text: String,
    /// The language the model itself detected, if it reported one.
    model_detected_language: Option<String>,
}

/// One transcription's final text: its output language resolved, then
/// post-processed.
fn finish_text(outcome: RunOutcome, settings: &AppSettings) -> String {
    let RunOutcome {
        setup,
        text,
        model_detected_language,
    } = outcome;
    let output_language = setup.output_language(settings, model_detected_language);
    debug!("Output language evidence: {:?}", output_language);
    // Skip fuzzy correction if the model already got the custom words.
    post_process_transcription_text(
        text,
        settings,
        setup.custom_words_prompted(&settings),
        &output_language,
        &setup.languages,
    )
}

/// How every run of one transcription (each chunk, or a whole clip) is
/// configured on the loaded model, and what post-processing needs to know.
struct RunSetup {
    run_options: RunOptions,
    languages: Vec<String>,
    applied_language_hint: Option<String>,
    output_was_translated: bool,
}

impl RunSetup {
    /// Custom words went to the model as vocabulary; skip fuzzy correction.
    ///
    /// vocabulary 是有界注入（128 词 / 1024 码点，见 vocabulary_for_model）：
    /// 只有当全部有效手工词 primary（忽略空/注释行，不经预算截断）都
    /// 实际注入时才算“已注入”；否则保留拉丁模糊纠错兜底，避免超出
    /// 预算的手工词（如第 129 词）既未注入也永不纠错。
    fn custom_words_prompted(&self, settings: &AppSettings) -> bool {
        if self.run_options.vocabulary.is_empty() {
            return false;
        }
        settings
            .custom_words
            .iter()
            .map(|line| chinese_hotwords::parse_hotword_line(line).primary)
            .filter(|p| !p.is_empty() && !p.starts_with('#'))
            .all(|p| self.run_options.vocabulary.contains(&p))
    }

    /// What the output language is known to be, given the model's own
    /// detection when it reported one.
    fn output_language(
        &self,
        settings: &AppSettings,
        model_detected_language: Option<String>,
    ) -> OutputLanguageEvidence {
        with_model_detected_language(
            resolve_output_language_evidence(
                settings,
                self.applied_language_hint.as_deref(),
                &self.languages,
                self.output_was_translated,
            ),
            model_detected_language,
        )
    }
}

/// The loaded model's live capabilities (cheap GGUF-metadata reads) are the
/// source of truth, not the ModelManager copy; translate is gated on what the
/// model advertises.
fn run_setup(info: &LoadedInfo, settings: &AppSettings, validated_language: &str) -> RunSetup {
    let model_supports_translate = info.capabilities.supports_translate;
    let languages = info.capabilities.languages.clone();
    // 能力矩阵（决定我们走哪条路）：streaming → 真正的增量输出；
    // vocabulary/initial_prompt → 热词偏置；否则只能靠停顿分段。
    info!(
        "Model capabilities: '{}' on '{}' streaming={} long_form={} vocabulary={} initial_prompt={} translate={} languages={:?}",
        settings.selected_model,
        info.backend,
        info.capabilities.supports_streaming,
        info.supports_long_form,
        info.supports_vocabulary,
        info.supports_initial_prompt,
        model_supports_translate,
        languages
    );

    let run_plan = transcribe_cpp_run_plan(
        settings.translate_to_english,
        validated_language,
        &languages,
        model_supports_translate,
    );
    // 诊断基线：出问题时用户只需回传这一行即可确认用的是哪个模型/语言。
    info!(
        "Transcribe run: model='{}' backend='{}' requested_language='{}' applied_language={:?} translate={} vocab_bias={}",
        settings.selected_model,
        info.backend,
        validated_language,
        run_plan.language,
        settings.translate_to_english,
        settings.decode_vocabulary_bias
    );
    let output_was_translated = run_plan.target_language.as_deref() == Some("en");
    let applied_language_hint = run_plan.language.clone();

    let run_options = RunOptions {
        task: run_plan.task,
        language: run_plan.language,
        target_language: run_plan.target_language,
        // 词库整合：vocabulary 用 effective_custom_words（手工词 + 导入
        // 词库合并结果），不再仅 custom_words
        // 预算按 sol 的实验包络（8 词 / 64 码点 / 单词 16 码点）而非旧的
        // 128/1024：旧数字是资源上限，从未被证明是"安全偏置量"。
        //
        // 偏置来源只取**手工词**，不取导入的通用词库。
        //
        // 证据：导入 THUOCL 后 effective_custom_words 有上万条，预算截断
        // 后注入的仍是 IT 类通用词（专家/创业/数据组织…）。对 LLM 式解码器
        // （Qwen3-ASR 等）这等于把一大串无关词强塞进上下文，用户实测会输出
        // 无关热词甚至重复。手工词才是"用户自己的词"，偏置收益高、风险低；
        // 导入词库继续只参与拼音后处理。
        vocabulary: vocabulary_for_model(
            info.supports_vocabulary && settings.decode_vocabulary_bias,
            &settings.custom_words,
        ),
        // 上游 0.9.8 曾显式 use_itn=Some(true)；transcribe-cpp API 迁移到
        // itn 枚举时丢失，Fun-ASR 默认 DEFAULT→false 会往 prompt 追加
        // “不进行文本规整”，导致中文无标点。这里恢复开启。
        itn: Itn::On,
        ..Default::default()
    };

    debug!(
        "transcribe-cpp run: task={:?}, language={:?}, vocabulary_terms={}",
        run_options.task,
        run_options.language,
        run_options.vocabulary.len()
    );

    RunSetup {
        run_options,
        languages,
        applied_language_hint,
        output_was_translated,
    }
}

/// A batch run's error: a cancel stays recognisable as one.
fn engine_error(e: EngineError) -> anyhow::Error {
    match e {
        EngineError::Cancelled => EngineError::Cancelled.into(),
        e => anyhow::anyhow!("transcribe-cpp transcription failed: {}", e),
    }
}

/// One transcription's text from its chunks, in order, and the model's
/// detected language (Whisper's audio-based LID, auto mode only; `None` when a
/// language hint was passed). Chunks that disagree leave it unknown. A single
/// transcript passes through untouched.
fn join_chunks(transcripts: Vec<Transcript>) -> (String, Option<String>) {
    let mut detected = transcripts.iter().map(|t| t.language.clone());
    let language = detected
        .next()
        .flatten()
        .filter(|first| detected.all(|language| language.as_ref() == Some(first)));
    let text = match <[Transcript; 1]>::try_from(transcripts) {
        Ok([one]) => one.text,
        Err(many) => {
            chunking::join_transcripts(&many.iter().map(|t| t.text.as_str()).collect::<Vec<_>>())
        }
    };
    (text, language)
}

fn emit_stream_text(app_handle: &AppHandle, committed: &str, tentative: &str) {
    let _ = StreamTextEvent {
        committed: committed.to_string(),
        tentative: tentative.to_string(),
    }
    .emit(app_handle);
}

fn lock_perf(perf: &Mutex<StreamPerf>) -> MutexGuard<'_, StreamPerf> {
    perf.lock().unwrap_or_else(|e| e.into_inner())
}

struct StreamPerf {
    feed_count: u64,
    emit_count: u64,
    streamed_samples: u64,
    stream_compute_elapsed: Duration,
    last_log: Instant,
    latest_revision: i32,
    latest_input_received_ms: i64,
    latest_audio_committed_ms: i64,
    latest_buffered_ms: i64,
}

impl StreamPerf {
    fn new() -> Self {
        Self {
            feed_count: 0,
            emit_count: 0,
            streamed_samples: 0,
            stream_compute_elapsed: Duration::ZERO,
            last_log: Instant::now(),
            latest_revision: 0,
            latest_input_received_ms: 0,
            latest_audio_committed_ms: 0,
            latest_buffered_ms: 0,
        }
    }

    fn record_feed(&mut self, samples: usize) {
        self.feed_count += 1;
        self.streamed_samples += samples as u64;
    }

    fn record_compute(&mut self, elapsed: Duration) {
        self.stream_compute_elapsed += elapsed;
    }

    fn record_update(&mut self, update: &transcribe_cpp::StreamUpdate) {
        self.latest_revision = update.revision;
        self.latest_input_received_ms = update.input_received_ms;
        self.latest_audio_committed_ms = update.audio_committed_ms;
        self.latest_buffered_ms = update.buffered_ms;
    }

    fn record_emit(&mut self) {
        self.emit_count += 1;
    }

    fn maybe_log(&mut self) {
        if self.last_log.elapsed() < STREAM_PERF_LOG_INTERVAL {
            return;
        }

        let audio_secs = self.audio_secs();
        let compute_secs = self.compute_secs();
        debug!(
            "Live preview perf: {:.2}s streamed audio, {:.2}s model compute ({:.2}x real-time), \
             input_received={:.2}s, committed_audio={:.2}s, buffered={}ms, revision={}, \
             {} frames fed, {} updates emitted",
            audio_secs,
            compute_secs,
            real_time_factor(audio_secs, compute_secs),
            self.latest_input_received_ms as f64 / 1000.0,
            self.latest_audio_committed_ms as f64 / 1000.0,
            self.latest_buffered_ms,
            self.latest_revision,
            self.feed_count,
            self.emit_count,
        );
        self.last_log = Instant::now();
    }

    fn log_finalized(&self, chars: usize) {
        let audio_secs = self.audio_secs();
        let compute_secs = self.compute_secs();
        info!(
            "Live preview finalized in {:.2}s model compute for {:.2}s streamed audio ({:.2}x real-time): \
             input_received={:.2}s, committed_audio={:.2}s, buffered={}ms, revision={}, \
             {} frames fed, {} updates emitted, {} chars",
            compute_secs,
            audio_secs,
            real_time_factor(audio_secs, compute_secs),
            self.latest_input_received_ms as f64 / 1000.0,
            self.latest_audio_committed_ms as f64 / 1000.0,
            self.latest_buffered_ms,
            self.latest_revision,
            self.feed_count,
            self.emit_count,
            chars
        );
    }

    fn audio_secs(&self) -> f64 {
        self.streamed_samples as f64 / 16_000.0
    }

    fn compute_secs(&self) -> f64 {
        self.stream_compute_elapsed.as_secs_f64()
    }
}

fn real_time_factor(audio_secs: f64, compute_secs: f64) -> f64 {
    if compute_secs > 0.0 {
        audio_secs / compute_secs
    } else {
        0.0
    }
}

/// Resolve the persisted language intent into the language a specific model can
/// use without writing the coerced value back to settings.
fn effective_language_for_model(
    settings: &AppSettings,
    model_manager: &ModelManager,
    model_id: &str,
) -> String {
    match model_manager.get_model_info(model_id) {
        Some(info) => crate::managers::model::effective_language(
            &settings.selected_language,
            &info.supported_languages,
            info.supports_language_detection,
        ),
        None => settings.selected_language.clone(),
    }
}

/// Resolve how confidently Handy knows the language of the text produced by a
/// transcription run. The UI language is deliberately not part of this
/// decision.
fn resolve_output_language_evidence(
    settings: &AppSettings,
    applied_language_hint: Option<&str>,
    supported_languages: &[String],
    translated_to_english: bool,
) -> OutputLanguageEvidence {
    if translated_to_english {
        return OutputLanguageEvidence::TranslatedToEnglish;
    }

    // Stored language intent is only evidence when this specific engine run
    // actually received the hint. Some multilingual engines (notably Parakeet
    // V3) always auto-detect and ignore Handy's selection; transcribe-cpp also
    // drops a requested hint when the loaded model does not advertise it.
    if let Some(language) = applied_language_hint.filter(|lang| !lang.is_empty() && *lang != "auto")
    {
        if settings.selected_language != "auto"
            && crate::managers::model::canonical_language_code(&settings.selected_language)
                == crate::managers::model::canonical_language_code(language)
        {
            return OutputLanguageEvidence::UserSelected(language.to_string());
        }

        // The engine may have required a concrete fallback even though the
        // user's persisted language was auto or unsupported.
        return OutputLanguageEvidence::ModelConstrained(language.to_string());
    }

    // A single-language model has a known output language without needing a
    // selectable language hint.
    if let [language] = supported_languages {
        return OutputLanguageEvidence::ModelConstrained(language.clone());
    }

    OutputLanguageEvidence::Unknown
}

/// Upgrade [`OutputLanguageEvidence::Unknown`] with the language the model
/// itself detected during the run (audio-based LID, e.g. Whisper in auto
/// mode). Stronger evidence resolved before the run is never overridden.
fn with_model_detected_language(
    evidence: OutputLanguageEvidence,
    detected: Option<String>,
) -> OutputLanguageEvidence {
    match (evidence, detected) {
        (OutputLanguageEvidence::Unknown, Some(language))
            if !language.is_empty() && language != "auto" =>
        {
            OutputLanguageEvidence::ModelDetected(language)
        }
        (evidence, _) => evidence,
    }
}

/// vocabulary 注入的词数预算：最多这么多个不同 primary（评审 P2：
/// 词库导入后 LLM 解码上下文的 token 预算必须有界）。
const VOCABULARY_MAX_TERMS: usize = 8;
/// vocabulary 注入的总码点预算（Unicode 码点，中文一字一码点）。
const VOCABULARY_MAX_CODEPOINTS: usize = 64;
/// 单个词的最大码点数，超长视为无效词直接跳过。
const VOCABULARY_MAX_TERM_CODEPOINTS: usize = 16;

/// Custom words for transcribe-cpp's `vocabulary` slot, in priority order.
/// Empty for models without Feature::Vocabulary, which would warn on every run
/// and rely on fuzzy correction instead.
/// 每行只取替换目标（第一个词）：别名与黑名单是拼音后处理的语法，
/// 注入引擎的 LLM 解码器上下文没有意义且浪费 token 预算。
///
/// 有界注入（评审 P2）：保持 effective 顺序（手工词优先），primary
/// 去重，跳过空/注释/超长（>64 码点）词；最多 `VOCABULARY_MAX_TERMS`
/// 个词、总计 `VOCABULARY_MAX_CODEPOINTS` 个码点，预算装不下即止。
/// 中文拼音后处理仍用全词表，不受此预算影响。
fn vocabulary_for_model(supports_vocabulary: bool, effective_words: &[String]) -> Vec<String> {
    if !supports_vocabulary {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    let mut total_codepoints = 0usize;
    let mut seen = std::collections::HashSet::new();
    for line in effective_words {
        if out.len() >= VOCABULARY_MAX_TERMS {
            break;
        }
        let primary = chinese_hotwords::parse_hotword_line(line).primary;
        // 跳过空段、井号注释行、超长词
        if primary.is_empty() || primary.starts_with('#') {
            continue;
        }
        let codepoints = primary.chars().count();
        if codepoints > VOCABULARY_MAX_TERM_CODEPOINTS {
            continue;
        }
        if !seen.insert(primary.clone()) {
            continue; // 重复 primary
        }
        if total_codepoints + codepoints > VOCABULARY_MAX_CODEPOINTS {
            break; // 码点预算装不下即止（保持 effective 顺序的前缀）
        }
        total_codepoints += codepoints;
        out.push(primary);
    }
    if out.len() < effective_words.len() {
        // 日志仅计数，不落词表内容
        debug!(
            "Vocabulary budget: {} effective word line(s) -> {} term(s), {} codepoint(s)",
            effective_words.len(),
            out.len(),
            total_codepoints
        );
    }
    out
}

struct TranscribeCppRunPlan {
    task: Task,
    language: Option<String>,
    target_language: Option<String>,
}

/// Build the transcribe-cpp language/task options shared by batch and live
/// streaming paths.
fn transcribe_cpp_run_plan(
    translate_to_english: bool,
    effective_language: &str,
    model_languages: &[String],
    model_supports_translate: bool,
) -> TranscribeCppRunPlan {
    let requested_language = match effective_language {
        "auto" => None,
        other => Some(other.to_string()),
    };
    // Only pass a language the loaded model actually advertises (per
    // capabilities().languages); otherwise auto-detect rather than failing with
    // UNSUPPORTED_LANGUAGE. Language-agnostic models report an empty list, so
    // they always stay on auto.
    let language = requested_language.filter(|lang| model_languages.iter().any(|l| l == lang));
    let (task, target_language) = cpp_translation_task(
        translate_to_english,
        model_supports_translate,
        language.as_deref(),
    );

    TranscribeCppRunPlan {
        task,
        language,
        target_language,
    }
}

/// FunASR 输出标记（Fun-ASR-Nano 等模型的特殊 token 会以字面量混进推理
/// 结果，如“今天天气/sil很好”）。上游 funasr 在推理后把它们替换成空格
/// 并折叠空白（batch 与单条推理同此）；transcribe-cpp 原样返回，所以
/// 在这里清理。后续遇到新的标记只需追加到此表。
const FUNASR_OUTPUT_MARKERS: &[&str] = &["/sil", "/nospeech", "/noise"];

/// 清理 FunASR 输出标记，仅对含标记的文本生效：
///
/// - 每个标记替换为单个空格，随后折叠连续空白、去掉首尾空白；
/// - 两侧都是汉字的空格删除（中文之间不插空格）；数字/拉丁两侧保留
///   单个空格（与 FunASR 自身输出风格一致）；
/// - 标记后紧跟 ASCII 字母/数字（如听写出的 `/silence`）视为普通文本
///   不替换——拉丁字母、拼音、中文内容一概不删改；
/// - 无标记文本逐字符原样返回（同一个 `String`，不重新分配）。
///
/// 清理后的既有 [`normalize_transcription_output`]（重复词折叠、多空格、
/// 修剪）照常执行。原始输出在 debug 日志留档（脱敏），便于对照清理
/// 前后。
fn strip_funasr_markers(text: String) -> String {
    if !FUNASR_OUTPUT_MARKERS.iter().any(|m| text.contains(m)) {
        return text;
    }
    let raw = text.clone();
    let replaced = replace_funasr_markers(&text);
    let cleaned = collapse_marker_whitespace(&replaced);
    debug!(
        "FunASR marker cleanup: raw_output='{}' cleaned_output='{}'",
        crate::utils::redact_text(&raw),
        crate::utils::redact_text(&cleaned)
    );
    cleaned
}

/// 把 `text` 里的每个 [`FUNASR_OUTPUT_MARKERS`] 标记替换为单个空格。
/// 标记后紧跟 ASCII 字母/数字时不替换：那是用户听写的 `/silence` 之类
/// 普通文本（上游 funasr 用朴素 replace 会连带删掉后续拉丁字母，我们
/// 明确不做）。
fn replace_funasr_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        if ch == '/' {
            if let Some(marker) = FUNASR_OUTPUT_MARKERS
                .iter()
                .find(|m| text[idx..].starts_with(*m))
            {
                let end = idx + marker.len();
                // 标记后紧跟 ASCII 字母/数字时不是标记，而是 `/silence`、
                // `/sil3` 这类真实拉丁内容——保留原文，绝不删改用户听写出的
                // 英文字母（sol 明确要求不要动拉丁/拼音）。标记后面是空白、
                // 中文、标点或行尾时才按标记清理。
                let followed_by_latin_word = text[end..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphanumeric());
                if !followed_by_latin_word {
                    out.push(' ');
                    // 跳过标记剩余字符（peek 防越界：end 是字符边界）
                    while let Some(&(next_idx, _)) = chars.peek() {
                        if next_idx < end {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    continue;
                }
            }
        }
        out.push(ch);
    }
    out
}

/// 折叠连续空白为单个空格、去掉首尾空白；两侧都是汉字的空格直接删除
/// （Fun-ASR 把 `/sil` 粘在汉字之间，替换出的空格不能留在中文里）。
/// 只对含标记文本调用。
fn collapse_marker_whitespace(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    // 已输出的最后一个非空白字符，用于判断空格两侧是否都是汉字
    let mut last_char: Option<char> = None;
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if !ch.is_whitespace() {
            out.push(ch);
            last_char = Some(ch);
            i += 1;
            continue;
        }
        // 跳过整段空白，看向右侧第一个非空白字符
        let mut j = i;
        while j < chars.len() && chars[j].is_whitespace() {
            j += 1;
        }
        if j == chars.len() {
            break; // 尾部空白丢弃（trim）
        }
        let both_han =
            last_char.is_some_and(chinese_hotwords::is_han) && chinese_hotwords::is_han(chars[j]);
        // 首部空白丢弃（trim）；汉字之间的空白删除；其余保单个空格
        if !both_han && last_char.is_some() {
            out.push(' ');
        }
        i = j;
    }
    out
}

fn post_process_transcription_text(
    raw: String,
    settings: &AppSettings,
    custom_words_already_prompted: bool,
    output_language: &OutputLanguageEvidence,
    supported_languages: &[String],
) -> String {
    let converts_script = settings.chinese_script != ChineseScript::AsTranscribed;
    fail_open_text_transform(raw, |raw| {
        // Fun-ASR 系模型会把 /sil 等标记混进原始输出；最先清理（上游
        // funasr 也是推理后立即替换），让语言检测、简繁转换与热词规则
        // 都看到干净文本。无标记文本逐字符不变。
        let raw = strip_funasr_markers(raw);
        // Last-resort language evidence: confidence-gated detection from the
        // transcribed text itself, constrained to the model's languages. Only
        // consulted when it can change the outcome (built-in gated fillers or
        // Chinese script conversion).
        let output_language = match output_language {
            OutputLanguageEvidence::Unknown
                if converts_script
                    || (settings.filler_word_removal_enabled
                        && settings.custom_filler_words.is_none()) =>
            {
                match detect_output_language(&raw, supported_languages) {
                    Some(language) => {
                        debug!("Text-based language detection resolved '{}'", language);
                        OutputLanguageEvidence::TextDetected(language)
                    }
                    None => OutputLanguageEvidence::Unknown,
                }
            }
            other => other.clone(),
        };

        // Convert the script before custom words so they match in the script
        // the user writes in. Only output known to be Chinese is touched, so
        // e.g. Japanese kanji are never rewritten.
        let variety = output_language
            .language()
            .and_then(ChineseVariety::from_language);
        let raw = match variety {
            Some(variety) if converts_script => {
                convert_chinese_script(&raw, variety, settings.chinese_script)
            }
            _ => raw,
        };

        // Voo 中文热词：中文路径用 effective 全词表（含导入词库）+ 四态
        // 档位表的精确单槽缓存（锁内只比较/克隆、编译在锁外）。档位
        // （key = 替换目标 primary，别名继承）在建索引展开条目前解析：
        // Blocked 不建条目；Candidate 建条目但不进自动纠正（仅建议）；
        // ActiveExact 走严格档（匹配时双侧常规读音校验，防词库词经多
        // 音字凑对误改日常短语，如「什么时候」→「实模式候」）；
        // ActiveFuzzy 走近音档。档位表参与缓存键，改档位即重编译。
        // 非中文文本不触碰该大缓存：拉丁纠错只用手工词（纯汉字导入词对
        // 拉丁路径无意义），规则走独立 rules-only 单槽缓存，避免 Latin
        // 转写把中文大缓存换出（5 万词索引构建是毫秒级）。规则不被
        // “无词表/非中文文本”卡死：拼音路径里规则只改未触碰区域，其余
        // 路径全量应用。
        // 非汉字文本彻底不构造 effective 全词表（50k 导入词的克隆/遍历
        // 对 Latin 路径无意义）；has_han 为假时直接空表
        let effective_words = chinese_hotwords::has_han(&raw)
            .then(|| settings.effective_custom_words())
            .unwrap_or_default();
        let chinese_path = !effective_words.is_empty();

        let corrected = if chinese_path {
            // 中文路径：拼音热词替换，随后规则只改未触碰区域（防止对刚
            // 替换出的热词二次改写，如拼音已输出“中州韵输入法”，规则
            // `中州韵=中州韵输入法` 不会再叠加）
            let term_modes = settings.effective_term_modes();
            let hotwords = chinese_hotwords::shared_hotword_cache().get_or_build_with_modes(
                &effective_words,
                &settings.hotword_rules,
                chinese_hotwords::DEFAULT_MIN_HAN,
                &term_modes,
            );
            let opts = chinese_hotwords::ConfusionOptions {
                dialect: settings.hotword_dialect_confusion,
            };
            let (corrected, hits) = chinese_hotwords::apply_chinese_hotwords_indexed(
                &raw,
                &hotwords,
                settings.hotword_pinyin_threshold,
                &opts,
            );
            for hit in &hits {
                // 命中内容含语音原文/热词文本：release 构建按仓库惯例脱敏
                info!(
                    "Chinese hotword: '{}' -> '{}' (similarity {:.2})",
                    crate::utils::redact_text(&hit.matched),
                    crate::utils::redact_text(&hit.replacement),
                    hit.score
                );
            }
            corrected
        } else {
            // 非中文文本：拉丁 custom_words 模糊纠错只用手工词，不用
            // effective 全词表（导入的纯汉字词对拉丁路径无意义）；
            // 显式规则独立生效，全量应用
            let base = if !settings.custom_words.is_empty() && !custom_words_already_prompted {
                apply_custom_words(
                    &raw,
                    &settings.custom_words,
                    settings.word_correction_threshold,
                )
            } else {
                raw
            };
            let rules = chinese_hotwords::shared_rules_cache().rules_for(&settings.hotword_rules);
            if rules.is_empty() {
                base
            } else {
                chinese_hotwords::apply_hotword_rules(&base, &rules)
            }
        };

        // 常用专名字面规范化（内置种子 + A 组开源专名源导入派生）：字面
        // 词边界替换、大小写不敏感、幂等；优先级低于上面的用户规则链
        // （用户同名规则键在编译期已剔除）；绝不参与拼音匹配。禁用开关
        // 或无规则时这里是一次空规则直通。
        let corrected = {
            let config =
                crate::managers::lexicon_sources::proper_noun_config_from_settings(settings);
            let rules = crate::audio_toolkit::proper_nouns::effective_proper_noun_rules(&config);
            rules.apply(&corrected)
        };

        let without_fillers = remove_filler_words(
            &corrected,
            &output_language,
            &settings.custom_filler_words,
            settings.filler_word_removal_enabled,
        );

        normalize_transcription_output(&without_fillers)
    })
}

/// Characters to wait for before detecting the preview's language. A lone
/// Chinese character reads as Mandarin, but Japanese often opens with kanji
/// before any kana; waiting a few characters keeps those from locking in.
const PREVIEW_DETECTION_MIN_CHARS: usize = 6;

/// Converts live-preview text into the configured Chinese script as it streams.
///
/// The preview is cosmetic: the final paste re-resolves the language and
/// converts on its own, so this only has to look right while speaking.
struct PreviewScript {
    script: ChineseScript,
    variety: Option<ChineseVariety>,
    /// The language wasn't known when the stream started (auto-detect), so it
    /// is detected from the streamed text until it turns out to be Chinese.
    detect: bool,
}

impl PreviewScript {
    fn new(script: ChineseScript, output_language: &OutputLanguageEvidence) -> Self {
        let enabled = script != ChineseScript::AsTranscribed;
        Self {
            script,
            variety: output_language
                .language()
                .and_then(ChineseVariety::from_language)
                .filter(|_| enabled),
            detect: enabled && *output_language == OutputLanguageEvidence::Unknown,
        }
    }

    /// Converts the model's raw committed/tentative text. Always fed the raw
    /// text, never a previous conversion, so it stays consistent with the
    /// final conversion. Once Chinese is detected it sticks for the rest of
    /// the stream so the preview doesn't flip back and forth.
    fn convert(
        &mut self,
        committed: &str,
        tentative: &str,
        supported_languages: &[String],
    ) -> (String, String) {
        if self.variety.is_none() && self.detect {
            let text = format!("{committed}{tentative}");
            if text.chars().count() >= PREVIEW_DETECTION_MIN_CHARS {
                self.variety = detect_output_language(&text, supported_languages)
                    .as_deref()
                    .and_then(ChineseVariety::from_language);
            }
        }

        match self.variety {
            Some(variety) => (
                convert_chinese_script(committed, variety, self.script),
                convert_chinese_script(tentative, variety, self.script),
            ),
            None => (committed.to_string(), tentative.to_string()),
        }
    }
}

/// Optional text cleanup must never discard a successful model result. The
/// transform is pure and owns its input, so recovering the untouched text is
/// safe even if a bug in custom-word or filler filtering unwinds.
fn fail_open_text_transform<F>(raw: String, transform: F) -> String
where
    F: FnOnce(String) -> String,
{
    let fallback = raw.clone();
    match catch_unwind(AssertUnwindSafe(|| transform(raw))) {
        Ok(processed) => processed,
        Err(payload) => {
            error!(
                "Optional transcription text post-processing panicked: {}; using the raw transcription",
                panic_payload_message(payload.as_ref())
            );
            fallback
        }
    }
}

/// Decide a transcribe-cpp run's task + translation target from settings.
///
/// "Translate to English" only fires where the model advertises translation.
/// Unlike transcribe-rs (which forces the target to English itself when its
/// `translate` flag is set), transcribe-cpp requires an explicit
/// `target_language`: a null target defaults to the *source*, so a non-English
/// source silently becomes e.g. es→es and Canary rejects the unadvertised pair.
/// An English source is skipped entirely — en→en is not a real translation, and
/// it's reachable by default since auto-detect-less models coerce intent to "en".
///
/// Returns `(task, target_language)` ready to drop into `RunOptions`.
fn cpp_translation_task(
    translate_to_english: bool,
    model_supports_translate: bool,
    source_language: Option<&str>,
) -> (Task, Option<String>) {
    let translate_to_en =
        translate_to_english && model_supports_translate && source_language != Some("en");
    if translate_to_en {
        (Task::Translate, Some("en".to_string()))
    } else {
        (Task::Transcribe, None)
    }
}

/// Drain a stream command channel, ignoring fed audio, until the caller
/// finalizes or cancels. Used when streaming can't actually run (model not
/// loaded / not streaming-capable) so the finalize handshake still completes
/// and the caller falls back to batch transcription.
fn drain_until_finalize(rx: mpsc::Receiver<StreamCmd>) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            StreamCmd::Feed(_) => {}
            StreamCmd::Finalize { reply, .. } => {
                let _ = reply.send(Ok(None));
                break;
            }
            StreamCmd::Cancel => break,
        }
    }
}

/// Log the compute devices transcribe-cpp registers.
///
/// Devices are listed by a transcription worker process (see
/// [`crate::engine_supervisor`]), so GPU driver initialization never runs in
/// the app process. Listing opens the GPU, which on macOS loads ggml's Metal
/// library and compiles it when the system shader cache does not hold it yet
/// (the first launch after an install or update), so the app calls this from
/// a background thread instead of its startup path.
pub fn report_compute_devices(tm: &TranscriptionManager) {
    if transcribe_gpu_disabled_for_host() {
        warn!(
            "Windows x64 build is running under emulation on an ARM64 host; \
             disabling transcribe.cpp GPU acceleration and using CPU"
        );
    }
    match transcribe_compute_devices(&tm.engine) {
        Ok(devices) => info!(
            "transcribe-cpp initialized with {} compute device(s): [{}]",
            devices.len(),
            devices
                .iter()
                .map(|d| format!("{} ({})", d.name, d.kind))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Err(e) => warn!("transcribe-cpp compute devices are unavailable: {}", e),
    }
}

/// Human-readable list of the transcribe-cpp compute devices, for the
/// `--list-devices` flag. The reported `index` is the value to pass to
/// `--device-index`. Errors if the devices could not be listed.
pub fn describe_compute_devices(tm: &TranscriptionManager) -> Result<Vec<String>> {
    Ok(transcribe_compute_devices(&tm.engine)?
        .into_iter()
        .map(|d| {
            let idx = d
                .index
                .map(|i| i.to_string())
                .unwrap_or_else(|| "-".to_string());
            let vram_mb = d.memory_total / (1024 * 1024);
            format!(
                "index={} kind={} name={} vram={}MB",
                idx,
                d.kind,
                d.label(),
                vram_mb
            )
        })
        .collect())
}

/// Map Handy's whisper accelerator setting to a transcribe-cpp [`Backend`].
///
/// `Auto` lets the library pick the best device (with CPU fallback), while
/// `Cpu` forces strict CPU. `Gpu` only remains as the companion setting for an
/// exact device; without a valid exact device it has the retired generic GPU
/// state's new Auto semantics. An emulated x64 process on Windows ARM64 forces
/// strict CPU for every setting.
fn select_transcribe_backend(setting: TranscribeAcceleratorSetting) -> Backend {
    select_transcribe_backend_for_host(setting, transcribe_gpu_disabled_for_host())
}

fn select_transcribe_backend_for_host(
    setting: TranscribeAcceleratorSetting,
    gpu_disabled: bool,
) -> Backend {
    match effective_transcribe_accelerator(setting, gpu_disabled) {
        TranscribeAcceleratorSetting::Cpu => Backend::Cpu,
        TranscribeAcceleratorSetting::Auto | TranscribeAcceleratorSetting::Gpu => Backend::Auto,
    }
}

/// Resolve the user's persisted GPU choice to the device key the worker
/// should load on. Registry indices and handles are process-local, so
/// settings store a key based on the backend's stable `device_id` (falling
/// back to name for backends such as Metal that do not report one). The
/// worker falls back to automatic selection if that device is gone.
fn resolve_gpu_device(
    setting: TranscribeAcceleratorSetting,
    gpu_device: Option<&str>,
) -> Option<String> {
    if transcribe_gpu_disabled_for_host() || setting != TranscribeAcceleratorSetting::Gpu {
        return None;
    }
    gpu_device.map(str::to_string)
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct GpuDeviceOption {
    pub id: String,
    pub name: String,
    pub total_vram_mb: usize,
}

fn transcribe_gpu_disabled_for_host() -> bool {
    crate::utils::is_windows_x64_emulated_on_arm64()
}

fn effective_transcribe_accelerator(
    setting: TranscribeAcceleratorSetting,
    gpu_disabled: bool,
) -> TranscribeAcceleratorSetting {
    if gpu_disabled {
        TranscribeAcceleratorSetting::Cpu
    } else {
        setting
    }
}

/// The compute devices the latest worker listed. On a host without GPU
/// access every worker is CPU-only, so it lists no GPU.
fn transcribe_compute_devices(engine: &EngineSupervisor) -> Result<Vec<DeviceInfo>> {
    engine
        .devices()
        .ok_or_else(|| anyhow::anyhow!("compute devices could not be listed (see the log for why)"))
}

fn available_transcribe_accelerators(gpu_disabled: bool) -> Vec<String> {
    if gpu_disabled {
        vec!["cpu".to_string()]
    } else {
        vec!["auto".to_string(), "cpu".to_string(), "gpu".to_string()]
    }
}

fn gpu_device_options(engine: &EngineSupervisor) -> Vec<GpuDeviceOption> {
    // GPU compute devices as the latest worker listed them, so a GPU that
    // comes back is picked up. `id` is a persistent identity key, never the
    // process-local registry index. It uses the backend's device_id where
    // available and its name otherwise (Metal). `total_vram_mb` is 0 when the
    // backend does not report capacity. Empty if listing failed.
    transcribe_compute_devices(engine)
        .unwrap_or_default()
        .into_iter()
        .filter(DeviceInfo::is_gpu)
        .map(|d| GpuDeviceOption {
            id: d.key.clone(),
            name: d.label().to_string(),
            total_vram_mb: (d.memory_total / (1024 * 1024)) as usize,
        })
        .collect()
}

#[derive(Serialize, Clone, Debug, Type)]
pub struct AvailableAccelerators {
    pub transcribe: Vec<String>,
    pub gpu_devices: Vec<GpuDeviceOption>,
}

/// Return the accelerators available to this process on its current host.
pub fn get_available_accelerators(tm: &TranscriptionManager) -> AvailableAccelerators {
    let transcribe_options = available_transcribe_accelerators(transcribe_gpu_disabled_for_host());

    AvailableAccelerators {
        transcribe: transcribe_options,
        gpu_devices: gpu_device_options(&tm.engine),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn languages(codes: &[&str]) -> Vec<String> {
        codes.iter().map(|code| (*code).to_string()).collect()
    }

    #[test]
    fn normal_hosts_preserve_every_transcribe_accelerator_setting() {
        for setting in [
            TranscribeAcceleratorSetting::Auto,
            TranscribeAcceleratorSetting::Cpu,
            TranscribeAcceleratorSetting::Gpu,
        ] {
            assert_eq!(effective_transcribe_accelerator(setting, false), setting);
        }
        assert_eq!(
            available_transcribe_accelerators(false),
            ["auto", "cpu", "gpu"]
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Auto, false),
            Backend::Auto
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Cpu, false),
            Backend::Cpu
        );
        assert_eq!(
            select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Gpu, false),
            Backend::Auto
        );
    }

    #[test]
    fn emulated_x64_on_arm64_forces_every_transcribe_setting_to_cpu() {
        for setting in [
            TranscribeAcceleratorSetting::Auto,
            TranscribeAcceleratorSetting::Cpu,
            TranscribeAcceleratorSetting::Gpu,
        ] {
            assert_eq!(
                effective_transcribe_accelerator(setting, true),
                TranscribeAcceleratorSetting::Cpu
            );
            assert_eq!(
                select_transcribe_backend_for_host(setting, true),
                Backend::Cpu
            );
        }
        assert_eq!(available_transcribe_accelerators(true), ["cpu"]);
    }

    #[test]
    fn optional_text_transform_falls_back_to_raw_text_after_panic() {
        let raw = "原始轉錄。".to_string();
        let result = fail_open_text_transform(raw.clone(), |_| {
            panic!("simulated optional cleanup failure")
        });

        assert_eq!(result, raw);
    }

    #[test]
    fn non_chinese_output_is_never_converted() {
        let settings = AppSettings {
            chinese_script: ChineseScript::Traditional,
            ..Default::default()
        };
        for evidence in [
            OutputLanguageEvidence::ModelDetected("ja".to_string()),
            OutputLanguageEvidence::UserSelected("en".to_string()),
            OutputLanguageEvidence::TranslatedToEnglish,
        ] {
            let result = post_process_transcription_text(
                "学校に行きます".to_string(),
                &settings,
                false,
                &evidence,
                &languages(&["zh", "ja", "en"]),
            );

            assert_eq!(result, "学校に行きます", "{evidence:?}");
        }
    }

    #[test]
    fn auto_detected_chinese_text_is_converted() {
        let settings = AppSettings {
            chinese_script: ChineseScript::Simplified,
            filler_word_removal_enabled: false,
            ..Default::default()
        };
        let result = post_process_transcription_text(
            "我們今天下午一起去學校圖書館看書，然後再去吃晚飯。".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::Unknown,
            &languages(&["zh", "en", "ja"]),
        );

        assert_eq!(result, "我们今天下午一起去学校图书馆看书，然后再去吃晚饭。");
    }

    #[test]
    fn portuguese_transcription_does_not_use_english_ui_filler_words() {
        let settings = AppSettings {
            app_language: "en".to_string(),
            selected_language: "pt-BR".to_string(),
            ..Default::default()
        };
        let supported = languages(&["en", "pt"]);
        let evidence = resolve_output_language_evidence(&settings, Some("pt"), &supported, false);

        let result = post_process_transcription_text(
            "eu vi um carro".to_string(),
            &settings,
            false,
            &evidence,
            &supported,
        );

        assert_eq!(
            evidence,
            OutputLanguageEvidence::UserSelected("pt".to_string())
        );
        assert_eq!(result, "eu vi um carro");
    }

    #[test]
    fn norwegian_alias_is_recorded_as_user_selected_evidence() {
        let settings = AppSettings {
            selected_language: "no".to_string(),
            ..Default::default()
        };

        let evidence =
            resolve_output_language_evidence(&settings, Some("nb"), &languages(&["nb"]), false);

        assert_eq!(
            evidence,
            OutputLanguageEvidence::UserSelected("nb".to_string())
        );
    }

    #[test]
    fn auto_language_without_detection_skips_gated_filler_removal() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };
        let evidence =
            resolve_output_language_evidence(&settings, None, &languages(&["en", "pt"]), false);

        // Too short for a reliable text detection, so the gated "um" must
        // survive; the universal "uhm" is removed regardless.
        let result = post_process_transcription_text(
            "um uhm ok".to_string(),
            &settings,
            false,
            &evidence,
            &languages(&["en", "pt"]),
        );

        assert_eq!(evidence, OutputLanguageEvidence::Unknown);
        assert_eq!(result, "um ok");
    }

    #[test]
    fn unknown_evidence_with_confident_text_detection_removes_gated_fillers() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let result = post_process_transcription_text(
            "um so the weather forecast said it would probably rain throughout the whole weekend"
                .to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::Unknown,
            &languages(&["en", "pt", "es", "de"]),
        );

        assert_eq!(
            result,
            "so the weather forecast said it would probably rain throughout the whole weekend"
        );
    }

    #[test]
    fn unknown_evidence_with_portuguese_text_preserves_um() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let result = post_process_transcription_text(
            "eu vi um carro na rua ontem de manhã quando fui ao mercado".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::Unknown,
            &languages(&["en", "pt", "es", "de"]),
        );

        assert_eq!(
            result,
            "eu vi um carro na rua ontem de manhã quando fui ao mercado"
        );
    }

    #[test]
    fn model_detected_language_upgrades_unknown_evidence_only() {
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, Some("en".to_string())),
            OutputLanguageEvidence::ModelDetected("en".to_string())
        );
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, Some("auto".to_string())),
            OutputLanguageEvidence::Unknown
        );
        assert_eq!(
            with_model_detected_language(OutputLanguageEvidence::Unknown, None),
            OutputLanguageEvidence::Unknown
        );
        assert_eq!(
            with_model_detected_language(
                OutputLanguageEvidence::UserSelected("pt".to_string()),
                Some("en".to_string())
            ),
            OutputLanguageEvidence::UserSelected("pt".to_string())
        );
    }

    #[test]
    fn auto_language_uses_single_language_model_as_evidence() {
        let settings = AppSettings {
            selected_language: "auto".to_string(),
            ..Default::default()
        };

        let evidence =
            resolve_output_language_evidence(&settings, None, &languages(&["en"]), false);

        assert_eq!(
            evidence,
            OutputLanguageEvidence::ModelConstrained("en".to_string())
        );
    }

    #[test]
    fn unsupported_explicit_language_uses_model_fallback_as_evidence() {
        let settings = AppSettings {
            selected_language: "pt".to_string(),
            ..Default::default()
        };

        let evidence = resolve_output_language_evidence(
            &settings,
            Some("en"),
            &languages(&["en", "de"]),
            false,
        );

        assert_eq!(
            evidence,
            OutputLanguageEvidence::ModelConstrained("en".to_string())
        );
    }

    #[test]
    fn ignored_user_language_is_not_output_evidence() {
        let settings = AppSettings {
            // Parakeet V3 ignores language hints and auto-detects even when a
            // selection from the previously active model remains persisted.
            selected_language: "en".to_string(),
            ..Default::default()
        };
        let supported = languages(&["en", "de", "pt"]);

        let evidence = resolve_output_language_evidence(&settings, None, &supported, false);
        assert_eq!(evidence, OutputLanguageEvidence::Unknown);

        let result = post_process_transcription_text(
            "eu vi um carro".to_string(),
            &settings,
            false,
            &evidence,
            &supported,
        );
        assert_eq!(result, "eu vi um carro");
    }

    #[test]
    fn unapplied_transcribe_cpp_language_is_not_output_evidence() {
        let settings = AppSettings {
            selected_language: "en".to_string(),
            ..Default::default()
        };
        let supported = languages(&[]);
        let plan = transcribe_cpp_run_plan(false, "en", &supported, false);

        assert_eq!(plan.language, None);
        assert_eq!(
            resolve_output_language_evidence(
                &settings,
                plan.language.as_deref(),
                &supported,
                false,
            ),
            OutputLanguageEvidence::Unknown
        );
    }

    #[test]
    fn translated_output_is_treated_as_english() {
        let settings = AppSettings {
            selected_language: "pt".to_string(),
            ..Default::default()
        };

        let evidence = resolve_output_language_evidence(
            &settings,
            Some("pt"),
            &languages(&["en", "pt"]),
            true,
        );

        assert_eq!(evidence, OutputLanguageEvidence::TranslatedToEnglish);
    }

    #[test]
    fn transcribe_cpp_run_plan_skips_english_translation() {
        let plan = transcribe_cpp_run_plan(true, "en", &languages(&["en", "es"]), true);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("en"));
        assert_eq!(plan.target_language, None);
    }

    #[test]
    fn transcribe_cpp_run_plan_translates_supported_non_english() {
        let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), true);

        assert!(matches!(plan.task, Task::Translate));
        assert_eq!(plan.language.as_deref(), Some("es"));
        assert_eq!(plan.target_language.as_deref(), Some("en"));
    }

    #[test]
    fn transcribe_cpp_run_plan_requires_model_translation_support() {
        let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), false);

        assert!(matches!(plan.task, Task::Transcribe));
        assert_eq!(plan.language.as_deref(), Some("es"));
        assert_eq!(plan.target_language, None);
    }

    #[test]
    fn fuzzy_correction_is_skipped_exactly_when_vocabulary_was_sent() {
        let settings = AppSettings {
            selected_language: "en".to_string(),
            custom_words: vec!["Handy".to_string()],
            ..Default::default()
        };
        let evidence = OutputLanguageEvidence::UserSelected("en".to_string());
        let supported = languages(&["en"]);

        // The model was already prompted with the term: leave its output alone.
        let prompted = post_process_transcription_text(
            "i use handy every day".to_string(),
            &settings,
            true,
            &evidence,
            &supported,
        );
        assert_eq!(prompted, "i use handy every day");

        // No vocabulary sent: the fuzzy post-correction fixes the casing.
        let corrected = post_process_transcription_text(
            "i use handy every day".to_string(),
            &settings,
            false,
            &evidence,
            &supported,
        );
        assert_eq!(corrected, "i use Handy every day");
    }

    // ---------------- 中文热词：规则独立生效 / 缓存集成 ----------------

    #[test]
    fn proper_noun_literal_pass_normalizes_variants() {
        // 默认开关开：变体被替换为规范写法，已正确写法与词边界内不动。
        let settings = AppSettings::default();
        let result = post_process_transcription_text(
            "我用 deepseek 和 vscode".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "我用 DeepSeek 和 VS Code");
        // 词边界：更长词不被误伤。
        let result = post_process_transcription_text(
            "deepseekers 是另一个词".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "deepseekers 是另一个词");
    }

    #[test]
    fn proper_noun_literal_pass_disabled_and_user_precedence() {
        // 禁用开关：不加载任何规则。
        let mut settings = AppSettings::default();
        settings.proper_noun_lexicon_enabled = false;
        let result = post_process_transcription_text(
            "我用 deepseek".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "我用 deepseek");

        // 用户同名规则优先：专名规则链不得覆盖用户规则产物。
        let settings = AppSettings {
            hotword_rules: "deepseek=深度求索".to_string(),
            ..Default::default()
        };
        let result = post_process_transcription_text(
            "deepseek 很强".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "深度求索 很强");
    }

    #[test]
    fn hotword_rules_apply_without_custom_words_on_han_text() {
        // 规则-only 配置：不被 custom_words.is_empty 卡死
        let settings = AppSettings {
            hotword_rules: "中州韵=中州韵输入法".to_string(),
            ..Default::default()
        };
        let result = post_process_transcription_text(
            "我在用中州韵打字".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "我在用中州韵输入法打字");
    }

    #[test]
    fn hotword_rules_apply_to_non_han_text() {
        // ASCII 规则不被“非汉字文本”卡死：作用于英文转写
        let settings = AppSettings {
            hotword_rules: "dont=>don't\nteh=the".to_string(),
            ..Default::default()
        };
        let result = post_process_transcription_text(
            "i dont know teh answer".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("en".to_string()),
            &languages(&["en"]),
        );
        assert_eq!(result, "i don't know the answer");
    }

    #[test]
    fn hotword_rules_apply_even_when_vocabulary_was_prompted() {
        // 词表已注入引擎 vocabulary（拉丁模糊纠错跳过）不影响显式规则
        let settings = AppSettings {
            custom_words: vec!["Handy".to_string()],
            hotword_rules: "handy=>Handy".to_string(),
            ..Default::default()
        };
        let result = post_process_transcription_text(
            "i use handy every day".to_string(),
            &settings,
            true,
            &OutputLanguageEvidence::UserSelected("en".to_string()),
            &languages(&["en"]),
        );
        assert_eq!(result, "i use Handy every day");
    }

    #[test]
    fn imported_lexicon_is_candidate_by_default_and_corrects_only_when_activated() {
        // 四态模型：导入词默认 Candidate（只进建议、不改写）。这条测试同时钉住
        // 两个方向——默认不乱改；显式提升为 ActiveExact 后才纠正完全同音错字。
        let mut settings = AppSettings::default();
        settings
            .imported_lexicons
            .insert("THUOCL".to_string(), vec!["中州韵输入法".to_string()]);

        let untouched = post_process_transcription_text(
            "我在用中州运输入法打字".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(
            untouched, "我在用中州运输入法打字",
            "Candidate 态不得改写文本（这正是「什么时候→实模式候」的防线）"
        );

        settings.lexicon_term_modes.insert(
            "中州韵输入法".to_string(),
            chinese_hotwords::TermMode::ActiveExact,
        );
        let corrected = post_process_transcription_text(
            "我在用中州运输入法打字".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(
            corrected, "我在用中州韵输入法打字",
            "显式激活后才纠正同音错字"
        );
    }

    #[test]
    fn chinese_hotwords_and_rules_layer_through_post_process() {
        // 拼音替换 + 规则分层：拼音区域不被规则二次改写，未触碰区域规则生效
        let settings = AppSettings {
            custom_words: vec!["中州韵输入法".to_string()],
            hotword_rules: "很好用=非常好用\n中州韵=中州韵输入法".to_string(),
            ..Default::default()
        };
        let result = post_process_transcription_text(
            "我在用中州运输入法，很好用".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "我在用中州韵输入法，非常好用");
    }

    // ---------------- FunASR 输出标记清理（/sil 等） ----------------

    #[test]
    fn funasr_sil_marker_replaced_with_space_and_whitespace_collapsed() {
        // 拉丁两侧：标记 → 单空格，原有空格一并折叠
        assert_eq!(
            strip_funasr_markers("hello /sil world".to_string()),
            "hello world"
        );
        // 粘在词中间（无空格）也能清
        assert_eq!(
            // 后接拉丁字母视为真实单词（如 /silence、/silworld），不是标记
            strip_funasr_markers("hello/silworld".to_string()),
            "hello/silworld"
        );
        // 不同标记混用
        assert_eq!(strip_funasr_markers("a /sil /noise b".to_string()), "a b");
    }

    #[test]
    fn text_without_funasr_markers_is_returned_unchanged() {
        let inputs = [
            "",
            "hello world",
            "人工只能 wsl",
            "已 完成，用时 3 秒",
            "wsl/linux 路径",
            "今天天气很好",
        ];
        for text in inputs {
            assert_eq!(strip_funasr_markers(text.to_string()), text);
        }
    }

    #[test]
    fn funasr_markers_never_insert_spaces_inside_chinese() {
        // 标记粘在汉字之间：替换后不留空格
        assert_eq!(
            strip_funasr_markers("今天天气/sil很好".to_string()),
            "今天天气很好"
        );
        // 数字两侧仍保留单个空格（FunASR 输出风格）
        assert_eq!(
            strip_funasr_markers("用时 3/sil秒".to_string()),
            "用时 3 秒"
        );
        // 中文与拉丁之间保留单个空格
        assert_eq!(
            strip_funasr_markers("人工只能/sil wsl".to_string()),
            "人工只能 wsl"
        );
    }

    #[test]
    fn funasr_consecutive_markers_collapse_to_one_space() {
        assert_eq!(
            strip_funasr_markers("好/sil/sil/noise吗".to_string()),
            "好吗"
        );
        // 拉丁单词两侧的空格保留一个：标记本身被清掉，字母不动
        assert_eq!(
            strip_funasr_markers("a /sil /sil /nospeech b".to_string()),
            "a b"
        );
    }

    #[test]
    fn funasr_markers_at_start_middle_and_end() {
        assert_eq!(
            strip_funasr_markers("/sil你好/sil世界/sil".to_string()),
            "你好世界"
        );
        assert_eq!(
            strip_funasr_markers("/sil /sil hello world /noise".to_string()),
            "hello world"
        );
        assert_eq!(strip_funasr_markers("你好/sil".to_string()), "你好");
        assert_eq!(strip_funasr_markers("/sil".to_string()), "");
    }

    #[test]
    fn funasr_cleanup_never_touches_latin_letters_or_pinyin() {
        // 听写出的“/sil 开头”的普通词不是标记，不删改拉丁字母
        assert_eq!(
            strip_funasr_markers("please /silence your phone".to_string()),
            "please /silence your phone"
        );
        assert_eq!(
            strip_funasr_markers("版本/sil3 不能用".to_string()),
            "版本/sil3 不能用"
        );
        // 真标记只删标记本身，拉丁内容一字不动
        assert_eq!(
            strip_funasr_markers("人工只能 /sil wsl".to_string()),
            "人工只能 wsl"
        );
    }

    #[test]
    fn post_process_strips_funasr_markers_from_final_output() {
        let settings = AppSettings::default();
        let result = post_process_transcription_text(
            "今天天气/sil很好/sil".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "今天天气很好");
    }

    #[test]
    fn post_process_leaves_marker_free_text_unchanged() {
        let settings = AppSettings::default();
        let result = post_process_transcription_text(
            "人工只能 wsl".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "人工只能 wsl");
    }

    #[test]
    fn funasr_markers_coexist_with_hotword_rules() {
        // 标记先清理，规则在干净文本上照常生效，不会把 /sil 带进匹配
        let settings = AppSettings {
            hotword_rules: "中州韵=中州韵输入法".to_string(),
            ..Default::default()
        };
        let result = post_process_transcription_text(
            "我在用中州韵/sil打字".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "我在用中州韵输入法打字");
    }

    #[test]
    fn funasr_marker_cleanup_runs_before_chinese_pinyin_correction() {
        // 标记先清掉，拼音纠错在干净文本上照常命中
        let settings = AppSettings {
            custom_words: vec!["中州韵输入法".to_string()],
            ..Default::default()
        };
        let result = post_process_transcription_text(
            "我在用中州运输入法/sil打字".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("zh".to_string()),
            &languages(&["zh"]),
        );
        assert_eq!(result, "我在用中州韵输入法打字");
    }

    // ---------------- vocabulary 有界注入（评审 P2） ----------------

    #[test]
    fn decode_vocabulary_comes_from_manual_words_only() {
        // 导入词库不得进入解码期偏置：只注入手工词。
        let mut settings = AppSettings::default();
        settings.custom_words = vec!["模型量化".to_string()];
        settings.imported_lexicons.insert(
            "IT".to_string(),
            vec![
                "专家".to_string(),
                "创业".to_string(),
                "数据组织".to_string(),
            ],
        );
        let from_manual = vocabulary_for_model(true, &settings.custom_words);
        assert_eq!(from_manual, vec!["模型量化".to_string()]);
        let from_effective = vocabulary_for_model(true, &settings.effective_custom_words());
        assert!(
            from_effective.contains(&"专家".to_string()),
            "对照组：用 effective 词表时导入词会进入偏置（这正是要避免的行为）"
        );
    }

    #[test]
    fn vocabulary_injection_is_bounded_for_large_lexicons() {
        // 5 万词导入：vocabulary 有界 —— 最多 128 词、总计 1024 码点、
        // 单词 ≤64 码点，全部去重，且保持 effective 顺序（前缀）。
        // 前 128 词各 8 码点（5 汉字 + 3 数字）→ 128×8 = 1024，两预算同时装满
        let words: Vec<String> = (0..50_000).map(|i| format!("词库词汇编{:03}", i)).collect();
        let vocab = vocabulary_for_model(true, &words);
        assert_eq!(vocab.len(), VOCABULARY_MAX_TERMS);
        let total: usize = vocab.iter().map(|w| w.chars().count()).sum();
        assert_eq!(total, VOCABULARY_MAX_CODEPOINTS);
        assert!(vocab
            .iter()
            .all(|w| w.chars().count() <= VOCABULARY_MAX_TERM_CODEPOINTS));
        let mut uniq = std::collections::HashSet::new();
        assert!(vocab.iter().all(|w| uniq.insert(w.clone())));
        assert_eq!(vocab.first().map(String::as_str), Some("词库词汇编000"));
    }

    #[test]
    fn vocabulary_total_codepoint_budget_stops_fill() {
        // 预算按 sol 的实验包络：8 词 / 64 码点。8 个 8 码点词恰好装满，
        // 第 9 个装不下即止。
        // 8 个 8 码点词恰好装满两个预算（8 词 / 64 码点），第 9 个起被拒。
        let mut words: Vec<String> = (0..8).map(|i| format!("词词词词词词词{i}")).collect();
        assert!(words.iter().all(|w| w.chars().count() == 8));
        words.push("额外词".to_string());
        words.push("再多一个".to_string());
        let vocab = vocabulary_for_model(true, &words);
        assert_eq!(vocab.len(), 8, "词数上限 8");
        assert_eq!(vocab.iter().map(|w| w.chars().count()).sum::<usize>(), 64);
        // 不支持 vocabulary 的模型返回空
        assert!(vocabulary_for_model(false, &words).is_empty());
    }

    #[test]
    fn vocabulary_skips_comments_empty_overlong_and_duplicates() {
        let words = vec![
            "# 注释行".to_string(),
            "   ".to_string(),
            format!("{}x", "a".repeat(16)),      // 17 码点超长，跳过
            "中州韵输入法 | 中州韵".to_string(), // 只取 primary
            "中州韵输入法".to_string(),          // 重复 primary，去重
            "a".repeat(16),                      // 16 码点恰好接受
        ];
        let vocab = vocabulary_for_model(true, &words);
        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab[0], "中州韵输入法");
        assert_eq!(vocab[1], "a".repeat(16));
    }

    #[test]
    fn vocabulary_keeps_manual_words_first() {
        // effective 顺序手工词优先：vocabulary 首位是手工词
        let mut settings = AppSettings::default();
        settings.custom_words = vec!["手工词优先".to_string()];
        settings.imported_lexicons.insert(
            "THUOCL".to_string(),
            (0..200).map(|i| format!("导入词汇编号{:03}", i)).collect(),
        );
        let effective = settings.effective_custom_words();
        let vocab = vocabulary_for_model(true, &effective);
        assert_eq!(vocab.first().map(String::as_str), Some("手工词优先"));
        assert!(vocab.len() > 1);
    }

    // ---------------- Latin 路径不膨胀（评审 P2） ----------------

    #[test]
    fn latin_path_uses_manual_words_only_not_imported_lexicon() {
        // 拉丁纠错只用手工词：导入词库（纯汉字）不参与 Latin 路径
        let mut settings = AppSettings::default();
        settings.custom_words = vec!["Handy".to_string()];
        settings.imported_lexicons.insert(
            "THUOCL".to_string(),
            (0..1000).map(|i| format!("词库词汇编号{:03}", i)).collect(),
        );
        let result = post_process_transcription_text(
            "i use handy every day".to_string(),
            &settings,
            false,
            &OutputLanguageEvidence::UserSelected("en".to_string()),
            &languages(&["en"]),
        );
        assert_eq!(result, "i use Handy every day");

        // 只有导入词（无手工词）时，Latin 文本不被改写
        let mut only_import = AppSettings::default();
        only_import
            .imported_lexicons
            .insert("THUOCL".to_string(), vec!["人工智能".to_string()]);
        let result2 = post_process_transcription_text(
            "artificial intelligence".to_string(),
            &only_import,
            false,
            &OutputLanguageEvidence::UserSelected("en".to_string()),
            &languages(&["en"]),
        );
        assert_eq!(result2, "artificial intelligence");
    }

    #[test]
    fn latin_path_rules_do_not_touch_chinese_hotword_cache() {
        // Latin 文本 + 巨量导入词：规则照常生效（走独立 rules-only 缓存），
        // 行为与“只有手工词”时一致，不因导入词膨胀
        let rules_text = "teh=the";
        let mut with_imports = AppSettings::default();
        with_imports.custom_words = vec!["Handy".to_string()];
        with_imports.hotword_rules = rules_text.to_string();
        with_imports.imported_lexicons.insert(
            "THUOCL".to_string(),
            (0..20_000)
                .map(|i| format!("词库词汇编号{:05}", i))
                .collect(),
        );
        let mut manual_only = with_imports.clone();
        manual_only.imported_lexicons.clear();

        for settings in [&with_imports, &manual_only] {
            let result = post_process_transcription_text(
                "i use handy and teh docs".to_string(),
                settings,
                false,
                &OutputLanguageEvidence::UserSelected("en".to_string()),
                &languages(&["en"]),
            );
            assert_eq!(result, "i use Handy and the docs");
        }
    }

    // ---------------- vocabulary 截断与模糊纠错兜底（评审 P2 回归） ----------------

    fn setup_with_vocabulary(vocabulary: Vec<String>) -> RunSetup {
        RunSetup {
            run_options: RunOptions {
                vocabulary,
                ..Default::default()
            },
            languages: Vec::new(),
            applied_language_hint: None,
            output_was_translated: false,
        }
    }

    #[test]
    fn manual_words_beyond_vocabulary_budget_still_get_fuzzy_correction() {
        // 超出预算的手工词未注入 vocabulary：custom_words_prompted 不得
        // 误判“已注入”而关掉拉丁模糊纠错（否则预算外的词永不纠错）
        let mut words: Vec<String> = (0..VOCABULARY_MAX_TERMS)
            .map(|i| format!("handyword{i:03}"))
            .collect();
        words.push("Handy".to_string()); // 预算外（12cps×85 > 1024，先耗尽）
        let settings = AppSettings {
            selected_language: "en".to_string(),
            custom_words: words.clone(),
            ..Default::default()
        };
        let injected = vocabulary_for_model(true, &words);
        assert!(injected.len() < words.len(), "预算必须截断");
        assert!(!injected.contains(&"Handy".to_string()));
        let setup = setup_with_vocabulary(injected);
        assert!(!setup.custom_words_prompted(&settings));

        // 集成：未注入的词仍走拉丁模糊纠错
        let result = post_process_transcription_text(
            "i use handy every day".to_string(),
            &settings,
            setup.custom_words_prompted(&settings),
            &OutputLanguageEvidence::UserSelected("en".to_string()),
            &languages(&["en"]),
        );
        assert_eq!(result, "i use Handy every day");
    }

    #[test]
    fn manual_words_within_budget_keep_prompted_skip() {
        // 预算内全部注入 → 已注入，跳过模糊纠错（原行为不变）；
        // 注释/空行不算有效手工词，不参与判定
        let words = vec![
            "# 注释行".to_string(),
            "   ".to_string(),
            "Handy".to_string(),
        ];
        let settings = AppSettings {
            selected_language: "en".to_string(),
            custom_words: words.clone(),
            ..Default::default()
        };
        let setup = setup_with_vocabulary(vocabulary_for_model(true, &words));
        assert_eq!(setup.run_options.vocabulary, vec!["Handy".to_string()]);
        assert!(setup.custom_words_prompted(&settings));

        let result = post_process_transcription_text(
            "i use handy every day".to_string(),
            &settings,
            setup.custom_words_prompted(&settings),
            &OutputLanguageEvidence::UserSelected("en".to_string()),
            &languages(&["en"]),
        );
        assert_eq!(result, "i use handy every day"); // 已注入：不改写
    }
}

impl Drop for TranscriptionManager {
    fn drop(&mut self) {
        // Skip shutdown unless this is the very last clone. TranscriptionManager
        // is cloned by initiate_model_load() and the watcher thread — those
        // clones dropping must not kill the watcher. The watcher thread holds
        // its own clone, so current_model_id's strong_count is always >= 2 while
        // the watcher is alive. (current_model_id is only ever cloned along with
        // the manager; shutdown_signal is not, since the watcher also holds a
        // standalone clone of it.) When it reaches 1, only this instance
        // remains and we can safely shut down.
        if Arc::strong_count(&self.current_model_id) > 1 {
            return;
        }

        // Signal the watcher thread to shutdown
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wait for the thread to finish gracefully.
        // Use match instead of unwrap to avoid panicking if the mutex is
        // poisoned — a panic inside Drop calls abort().
        let mut guard = match self.watcher_handle.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("Recovered poisoned watcher_handle mutex during TranscriptionManager drop — a panic occurred earlier this session");
                e.into_inner()
            }
        };
        if let Some(handle) = guard.take() {
            if let Err(e) = handle.join() {
                warn!("Failed to join idle watcher thread: {:?}", e);
            } else {
                debug!("Idle watcher thread joined successfully");
            }
        }
    }
}
