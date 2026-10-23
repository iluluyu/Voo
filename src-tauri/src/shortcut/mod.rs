//! Keyboard shortcut management module
//!
//! This module provides a unified interface for keyboard shortcuts with
//! multiple backend implementations:
//!
//! - `tauri`: Uses Tauri's built-in global-shortcut plugin
//! - `handy_keys`: Uses the handy-keys library for more control
//!
//! The active implementation is determined by the `keyboard_implementation`
//! setting and can be changed at runtime.

mod handler;
pub mod handy_keys;
pub mod tauri_impl;

use log::{debug, error, info, warn};
use serde::Serialize;
use specta::Type;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter, Manager};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::settings::APPLE_INTELLIGENCE_DEFAULT_MODEL_ID;
use crate::settings::{
    self, get_settings, AutoSubmitKey, ChineseScript, ClipboardHandling, KeyboardImplementation,
    LLMPrompt, OverlayPosition, OverlayStyle, PasteMethod, PostProcessMode, ShortcutActivation,
    ShortcutBinding, SoundTheme, Theme, ThinkingLevel, TypingTool, VadBackend,
    APPLE_INTELLIGENCE_PROVIDER_ID,
};
use crate::tray;

// Note: Commands are accessed via shortcut::handy_keys:: in lib.rs

/// Initialize shortcuts using the configured implementation
pub fn init_shortcuts(app: &AppHandle) {
    let user_settings = settings::load_or_create_app_settings(app);

    // Check which implementation to use
    match user_settings.keyboard_implementation {
        KeyboardImplementation::Tauri => {
            tauri_impl::init_shortcuts(app);
        }
        KeyboardImplementation::HandyKeys => {
            if let Err(e) = handy_keys::init_shortcuts(app) {
                error!("Failed to initialize handy-keys shortcuts: {}", e);
                // Fall back to Tauri implementation and persist this fallback
                warn!("Falling back to Tauri global shortcut implementation and saving fallback to settings");

                // Update settings to persist the fallback so we don't retry HandyKeys on next launch
                let mut settings = settings::get_settings(app);
                settings.keyboard_implementation = KeyboardImplementation::Tauri;
                settings::write_settings(app, settings);

                tauri_impl::init_shortcuts(app);
            }
        }
    }
}

/// Whether the recording lifecycle currently wants the cancel shortcut.
/// Written synchronously by start/stop, so it always holds the latest request.
static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Whether the cancel shortcut is actually registered with the backend.
/// The lock also serializes reconciliation passes.
#[cfg(not(target_os = "linux"))]
static CANCEL_REGISTERED: std::sync::Mutex<bool> = std::sync::Mutex::new(false);

/// Register the cancel shortcut (called when recording starts)
pub fn register_cancel_shortcut(app: &AppHandle) {
    // Track recording lifecycle independently of the current implementation so
    // switching implementations mid-recording cannot leave stale fallback state.
    crate::secure_input::register_cancel_fallback(app);

    CANCEL_REQUESTED.store(true, Ordering::SeqCst);
    schedule_cancel_reconcile(app);
}

/// Unregister the cancel shortcut (called when recording stops)
pub fn unregister_cancel_shortcut(app: &AppHandle) {
    crate::secure_input::unregister_cancel_fallback(app);

    CANCEL_REQUESTED.store(false, Ordering::SeqCst);
    schedule_cancel_reconcile(app);
}

/// Apply the requested cancel state off the calling thread. Start and stop run
/// on the shortcut handler thread, which must not block on registration, so the
/// work is spawned. Spawned tasks may run in any order (a very short recording
/// can see its unregister task run before its register task), so each pass
/// applies the latest request instead of the action that scheduled it.
fn schedule_cancel_reconcile(app: &AppHandle) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        reconcile_cancel_shortcut(&app);
    });
}

fn reconcile_cancel_shortcut(app: &AppHandle) {
    // Cancel shortcut is disabled on Linux due to instability with dynamic shortcut registration
    #[cfg(target_os = "linux")]
    {
        let _ = app;
        return;
    }

    #[cfg(not(target_os = "linux"))]
    {
        let mut registered = CANCEL_REGISTERED.lock().unwrap_or_else(|e| e.into_inner());
        let requested = CANCEL_REQUESTED.load(Ordering::SeqCst);
        if requested == *registered {
            return;
        }

        let Some(cancel_binding) = get_settings(app).bindings.get("cancel").cloned() else {
            return;
        };

        if requested {
            match register_shortcut(app, cancel_binding) {
                Ok(()) => *registered = true,
                Err(e) => error!("Failed to register cancel shortcut: {}", e),
            }
        } else {
            match unregister_shortcut(app, cancel_binding) {
                Ok(()) => *registered = false,
                Err(e) => error!("Failed to unregister cancel shortcut: {}", e),
            }
        }
    }
}

/// Register a shortcut using the appropriate implementation
pub fn register_shortcut(app: &AppHandle, binding: ShortcutBinding) -> Result<(), String> {
    let settings = get_settings(app);
    match settings.keyboard_implementation {
        KeyboardImplementation::Tauri => tauri_impl::register_shortcut(app, binding),
        KeyboardImplementation::HandyKeys => handy_keys::register_shortcut(app, binding),
    }
}

/// Unregister a shortcut using the appropriate implementation
pub fn unregister_shortcut(app: &AppHandle, binding: ShortcutBinding) -> Result<(), String> {
    let settings = get_settings(app);
    match settings.keyboard_implementation {
        KeyboardImplementation::Tauri => tauri_impl::unregister_shortcut(app, binding),
        KeyboardImplementation::HandyKeys => handy_keys::unregister_shortcut(app, binding),
    }
}

// ============================================================================
// Binding Management Commands
// ============================================================================

#[derive(Serialize, Type)]
pub struct BindingResponse {
    success: bool,
    binding: Option<ShortcutBinding>,
    error: Option<String>,
}

#[tauri::command]
#[specta::specta]
pub fn change_binding(
    app: AppHandle,
    id: String,
    binding: String,
) -> Result<BindingResponse, String> {
    // Reject empty bindings — every shortcut should have a value
    if binding.trim().is_empty() {
        return Err("Binding cannot be empty".to_string());
    }

    let mut settings = settings::get_settings(&app);

    // Get the binding to modify, or create it from defaults if it doesn't exist
    let binding_to_modify = match settings.bindings.get(&id) {
        Some(binding) => binding.clone(),
        None => {
            // Try to get the default binding for this id
            let default_settings = settings::get_default_settings();
            match default_settings.bindings.get(&id) {
                Some(default_binding) => {
                    warn!(
                        "Binding '{}' not found in settings, creating from defaults",
                        id
                    );
                    default_binding.clone()
                }
                None => {
                    let error_msg = format!("Binding with id '{}' not found in defaults", id);
                    warn!("change_binding error: {}", error_msg);
                    return Ok(BindingResponse {
                        success: false,
                        binding: None,
                        error: Some(error_msg),
                    });
                }
            }
        }
    };

    // If this is the cancel binding, just update the settings and return
    // It's managed dynamically, so we don't register/unregister here
    if id == "cancel" {
        if let Some(mut b) = settings.bindings.get(&id).cloned() {
            b.current_binding = binding;
            settings.bindings.insert(id.clone(), b.clone());
            settings::write_settings(&app, settings);
            crate::secure_input::reconcile_fallback(&app);
            return Ok(BindingResponse {
                success: true,
                binding: Some(b.clone()),
                error: None,
            });
        }
    }

    // Unregister the existing binding
    if let Err(e) = unregister_shortcut(&app, binding_to_modify.clone()) {
        let error_msg = format!("Failed to unregister shortcut: {}", e);
        error!("change_binding error: {}", error_msg);
    }

    // Validate the new shortcut for the current keyboard implementation
    if let Err(e) = validate_shortcut_for_implementation(&binding, settings.keyboard_implementation)
    {
        warn!("change_binding validation error: {}", e);
        restore_registration(&app, &binding_to_modify);
        return Err(e);
    }

    // Create an updated binding
    let mut updated_binding = binding_to_modify.clone();
    updated_binding.current_binding = binding;

    // Register the new binding
    if let Err(e) = register_shortcut(&app, updated_binding.clone()) {
        let error_msg = format!("Failed to register shortcut: {}", e);
        error!("change_binding error: {}", error_msg);
        restore_registration(&app, &binding_to_modify);
        return Ok(BindingResponse {
            success: false,
            binding: None,
            error: Some(error_msg),
        });
    }

    // Update the binding in the settings
    settings.bindings.insert(id, updated_binding.clone());

    // Save the settings and synchronize any active Secure Input shadows.
    settings::write_settings(&app, settings);
    crate::secure_input::reconcile_fallback(&app);

    // Return the updated binding
    Ok(BindingResponse {
        success: true,
        binding: Some(updated_binding),
        error: None,
    })
}

/// Best-effort re-register of the previous binding after a failed change,
/// so a failure leaves the user's shortcut working exactly as before.
fn restore_registration(app: &AppHandle, binding: &ShortcutBinding) {
    if let Err(e) = register_shortcut(app, binding.clone()) {
        error!(
            "Failed to restore previous binding '{}' ({}): {}",
            binding.id, binding.current_binding, e
        );
    }
}

#[tauri::command]
#[specta::specta]
pub fn reset_binding(app: AppHandle, id: String) -> Result<BindingResponse, String> {
    let binding = settings::get_stored_binding(&settings::get_settings(&app), &id)?;
    change_binding(app, id, binding.default_binding)
}

/// Unregister every binding while the user is recording a new shortcut in
/// the UI, so no existing shortcut can fire — or swallow the keystrokes —
/// mid-capture. The "cancel" binding is untouched: it is managed dynamically
/// by the recording lifecycle.
pub fn suspend_all_shortcuts(app: &AppHandle) {
    for (id, binding) in settings::get_bindings(app) {
        if id == "cancel" {
            continue;
        }
        if let Err(e) = unregister_shortcut(app, binding) {
            debug!(
                "suspend_all_shortcuts: could not unregister '{}': {}",
                id, e
            );
        }
    }
}

/// Re-register every binding from settings after shortcut recording ends.
/// Registering an already-registered shortcut fails cleanly in both
/// implementations, so this is idempotent and safe on every exit path.
pub fn resume_all_shortcuts(app: &AppHandle) {
    let settings = get_settings(app);
    for (id, binding) in &settings.bindings {
        if id == "cancel" {
            continue;
        }
        // The dedicated post-processing binding registers only while the
        // feature is enabled — mirroring
        // `register_all_shortcuts_for_implementation` and both `init`
        // paths, so resuming after shortcut capture leaves it as-is.
        if id == "transcribe_with_post_process" && !settings.post_process_enabled {
            continue;
        }
        if let Err(e) = register_shortcut(app, binding.clone()) {
            debug!("resume_all_shortcuts: could not register '{}': {}", id, e);
        }
    }
}

/// Temporarily unregister all bindings while the user is recording a
/// shortcut in the UI. This avoids firing actions while keys are recorded.
#[tauri::command]
#[specta::specta]
pub fn suspend_all_bindings(app: AppHandle) -> Result<(), String> {
    suspend_all_shortcuts(&app);
    Ok(())
}

/// Re-register all bindings after the user has finished recording.
#[tauri::command]
#[specta::specta]
pub fn resume_all_bindings(app: AppHandle) -> Result<(), String> {
    resume_all_shortcuts(&app);
    Ok(())
}

// ============================================================================
// Keyboard Implementation Switching
// ============================================================================

/// Result of changing keyboard implementation
#[derive(Serialize, Type)]
pub struct ImplementationChangeResult {
    pub success: bool,
    /// List of binding IDs that were reset to defaults due to incompatibility
    pub reset_bindings: Vec<String>,
}

/// Change the keyboard implementation with runtime switching.
/// This will unregister all shortcuts from the old implementation,
/// validate shortcuts for the new implementation (resetting invalid ones to defaults),
/// and register them with the new implementation.
#[tauri::command]
#[specta::specta]
pub fn change_keyboard_implementation_setting(
    app: AppHandle,
    implementation: String,
) -> Result<ImplementationChangeResult, String> {
    let current_settings = settings::get_settings(&app);
    let current_impl = current_settings.keyboard_implementation;
    let new_impl = parse_keyboard_implementation(&implementation);

    // If same implementation, nothing to do
    if current_impl == new_impl {
        return Ok(ImplementationChangeResult {
            success: true,
            reset_bindings: vec![],
        });
    }

    info!(
        "Switching keyboard implementation from {:?} to {:?}",
        current_impl, new_impl
    );

    // Unregister all shortcuts from the current implementation
    unregister_all_shortcuts(&app, current_impl);

    // Update the setting
    let mut settings = settings::get_settings(&app);
    settings.keyboard_implementation = new_impl;
    settings::write_settings(&app, settings);

    // Carbon fallback registrations use the Tauri plugin. Remove them before
    // registering the full Tauri implementation to avoid duplicate conflicts.
    if new_impl == KeyboardImplementation::Tauri {
        crate::secure_input::reconcile_fallback(&app);
    }

    // Initialize new implementation if needed (HandyKeys needs state)
    if new_impl == KeyboardImplementation::HandyKeys && initialize_handy_keys_with_rollback(&app)? {
        // Shortcuts already registered during init.
        crate::secure_input::reconcile_fallback(&app);
        return Ok(ImplementationChangeResult {
            success: true,
            reset_bindings: vec![],
        });
    }

    // Register all shortcuts with new implementation, resetting invalid ones
    let reset_bindings = register_all_shortcuts_for_implementation(&app, new_impl);
    crate::secure_input::reconcile_fallback(&app);

    // Emit event to notify frontend of the change
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "keyboard_implementation",
            "value": implementation,
            "reset_bindings": reset_bindings
        }),
    );

    info!("Keyboard implementation switched to {:?}", new_impl);

    Ok(ImplementationChangeResult {
        success: true,
        reset_bindings,
    })
}

/// Get the current keyboard implementation
#[tauri::command]
#[specta::specta]
pub fn get_keyboard_implementation(app: AppHandle) -> String {
    let settings = settings::get_settings(&app);
    match settings.keyboard_implementation {
        KeyboardImplementation::Tauri => "tauri".to_string(),
        KeyboardImplementation::HandyKeys => "handy_keys".to_string(),
    }
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate a shortcut for a specific implementation
fn validate_shortcut_for_implementation(
    raw: &str,
    implementation: KeyboardImplementation,
) -> Result<(), String> {
    match implementation {
        KeyboardImplementation::Tauri => tauri_impl::validate_shortcut(raw),
        KeyboardImplementation::HandyKeys => handy_keys::validate_shortcut(raw),
    }
}

/// Parse a keyboard implementation string into the enum
fn parse_keyboard_implementation(s: &str) -> KeyboardImplementation {
    match s {
        "tauri" => KeyboardImplementation::Tauri,
        "handy_keys" => KeyboardImplementation::HandyKeys,
        other => {
            warn!(
                "Invalid keyboard implementation '{}', defaulting to tauri",
                other
            );
            KeyboardImplementation::Tauri
        }
    }
}

/// Unregister all shortcuts for the current implementation
fn unregister_all_shortcuts(app: &AppHandle, implementation: KeyboardImplementation) {
    let bindings = settings::get_bindings(app);

    for (id, binding) in bindings {
        // Skip cancel shortcut as it's dynamically registered
        if id == "cancel" {
            continue;
        }

        let result = match implementation {
            KeyboardImplementation::Tauri => tauri_impl::unregister_shortcut(app, binding),
            KeyboardImplementation::HandyKeys => handy_keys::unregister_shortcut(app, binding),
        };

        if let Err(e) = result {
            warn!(
                "Failed to unregister shortcut '{}' during switch: {}",
                id, e
            );
        }
    }
}

/// Register all shortcuts for a specific implementation, validating and resetting invalid ones
fn register_all_shortcuts_for_implementation(
    app: &AppHandle,
    implementation: KeyboardImplementation,
) -> Vec<String> {
    let mut reset_bindings = Vec::new();
    let default_bindings = settings::get_default_settings().bindings;
    let mut current_settings = settings::get_settings(app);

    for (id, default_binding) in &default_bindings {
        // Skip cancel shortcut as it's dynamically registered
        if id == "cancel" {
            continue;
        }

        // Skip post-processing shortcut when the feature is disabled
        if id == "transcribe_with_post_process" && !current_settings.post_process_enabled {
            continue;
        }

        let mut binding = current_settings
            .bindings
            .get(id)
            .cloned()
            .unwrap_or_else(|| default_binding.clone());

        // Validate the shortcut for the target implementation
        if let Err(e) =
            validate_shortcut_for_implementation(&binding.current_binding, implementation)
        {
            info!(
                "Shortcut '{}' ({}) is invalid for {:?}: {}. Resetting to default.",
                id, binding.current_binding, implementation, e
            );

            // Reset to default
            binding.current_binding = default_binding.current_binding.clone();
            current_settings
                .bindings
                .insert(id.clone(), binding.clone());
            reset_bindings.push(id.clone());
        }

        // Register with the appropriate implementation
        let result = match implementation {
            KeyboardImplementation::Tauri => tauri_impl::register_shortcut(app, binding),
            KeyboardImplementation::HandyKeys => handy_keys::register_shortcut(app, binding),
        };

        if let Err(e) = result {
            error!(
                "Failed to register shortcut '{}' for {:?}: {}",
                id, implementation, e
            );
        }
    }

    // Save settings if any bindings were reset
    if !reset_bindings.is_empty() {
        settings::write_settings(app, current_settings);
    }

    reset_bindings
}

/// Initialize HandyKeys if not already initialized, with rollback on failure
fn initialize_handy_keys_with_rollback(app: &AppHandle) -> Result<bool, String> {
    if app.try_state::<handy_keys::HandyKeysState>().is_some() {
        return Ok(false); // Already initialized, caller should continue
    }

    if let Err(e) = handy_keys::init_shortcuts(app) {
        error!("Failed to initialize HandyKeys: {}", e);
        // Rollback to Tauri
        let mut settings = settings::get_settings(app);
        settings.keyboard_implementation = KeyboardImplementation::Tauri;
        settings::write_settings(app, settings);
        crate::secure_input::reconcile_fallback(app);
        tauri_impl::init_shortcuts(app);
        return Err(format!(
            "Failed to initialize HandyKeys: {}. Reverted to Tauri.",
            e
        ));
    }

    // init_shortcuts already registered shortcuts
    Ok(true)
}

// ============================================================================
// General Settings Commands
// ============================================================================

#[tauri::command]
#[specta::specta]
pub fn change_shortcut_activation_setting(
    app: AppHandle,
    activation: ShortcutActivation,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.shortcut_activation = activation;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_hold_threshold_ms_setting(app: AppHandle, ms: u64) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.hold_threshold_ms = ms;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_audio_feedback_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.audio_feedback = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_audio_feedback_volume_setting(app: AppHandle, volume: f32) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.audio_feedback_volume = volume;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_sound_theme_setting(app: AppHandle, theme: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match theme.as_str() {
        "marimba" => SoundTheme::Marimba,
        "pop" => SoundTheme::Pop,
        "custom" => SoundTheme::Custom,
        other => {
            warn!("Invalid sound theme '{}', defaulting to marimba", other);
            SoundTheme::Marimba
        }
    };
    settings.sound_theme = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_theme_setting(app: AppHandle, theme: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match theme.as_str() {
        "system" => Theme::System,
        "light" => Theme::Light,
        "dark" => Theme::Dark,
        other => {
            warn!("Invalid theme '{}', defaulting to system", other);
            Theme::System
        }
    };
    settings.theme = parsed;
    settings::write_settings(&app, settings);
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    apply_window_theme(&app, parsed);
    // Notify other webviews (the recording overlay) so they re-apply the palette
    // live — they set `data-theme` on their own document and can't see this one.
    let _ = app.emit("theme-changed", parsed);
    Ok(())
}

/// Applies the appearance setting to the native window chrome (title bar), which
/// CSS `data-theme` cannot reach. `System` clears the override so the window
/// follows the OS. Call this on startup and whenever the setting changes to keep
/// the title bar in sync with the in-app palette.
///
/// On Windows this themes the title bar only. On macOS `set_theme` sets
/// `NSApp.appearance` app-wide, which is what we want here: it darkens the title
/// bar and keeps the overlay in step. Linux is left to `data-theme` alone, since
/// its window theming is backend-dependent and unreliable.
#[cfg(any(target_os = "windows", target_os = "macos"))]
pub fn apply_window_theme(app: &AppHandle, theme: Theme) {
    let window_theme = match theme {
        Theme::System => None,
        Theme::Light => Some(tauri::Theme::Light),
        Theme::Dark => Some(tauri::Theme::Dark),
    };
    if let Some(window) = app.get_webview_window("main") {
        if let Err(e) = window.set_theme(window_theme) {
            warn!("Failed to apply window theme: {}", e);
        }
    }
}

#[tauri::command]
#[specta::specta]
pub fn change_translate_to_english_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.translate_to_english = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_selected_language_setting(app: AppHandle, language: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.selected_language = language;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_overlay_position_setting(app: AppHandle, position: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match position.as_str() {
        // "none" is retired (visibility is overlay_style now); fold legacy callers
        // onto Bottom rather than warn.
        "none" | "bottom" => OverlayPosition::Bottom,
        "top" => OverlayPosition::Top,
        other => {
            warn!("Invalid overlay position '{}', defaulting to bottom", other);
            OverlayPosition::Bottom
        }
    };
    settings.overlay_position = parsed;
    settings::write_settings(&app, settings);

    // Whether the overlay shows at all is owned by overlay_style now; position
    // only ever toggles Top/Bottom, so the enabled cache is untouched here.
    // Update overlay position without recreating window
    crate::utils::update_overlay_position(&app);

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_overlay_style_setting(app: AppHandle, style: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match style.as_str() {
        "none" => OverlayStyle::None,
        "minimal" => OverlayStyle::Minimal,
        "live" => OverlayStyle::Live,
        other => {
            warn!("Invalid overlay style '{}', defaulting to minimal", other);
            OverlayStyle::Minimal
        }
    };
    settings.overlay_style = parsed;
    settings::write_settings(&app, settings);

    // Keep the cached overlay-enabled flag in sync so emit_levels stops (or
    // resumes) emitting on the next audio callback.
    crate::overlay::update_overlay_enabled_cache(parsed != OverlayStyle::None);

    // Reposition in case the window needs to re-center for the new style.
    crate::utils::update_overlay_position(&app);

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_debug_mode_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.debug_mode = enabled;
    settings::write_settings(&app, settings);

    // Keep webview log streaming in sync: the live log viewer only exists in
    // debug mode, so logs are forwarded to the frontend only while it is on.
    crate::WEBVIEW_LOG_STREAMING.store(enabled, std::sync::atomic::Ordering::Relaxed);

    // Emit event to notify frontend of debug mode change
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "debug_mode",
            "value": enabled
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_start_hidden_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.start_hidden = enabled;
    settings::write_settings(&app, settings);

    // Notify frontend
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "start_hidden",
            "value": enabled
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_autostart_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.autostart_enabled = enabled;
    settings::write_settings(&app, settings);

    // Apply the autostart setting immediately
    crate::autostart::apply_autostart(&app, enabled);

    // Notify frontend
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "autostart_enabled",
            "value": enabled
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_update_checks_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    if settings::update_checks_forced_disabled() {
        return Err(
            "Update checks are disabled by system configuration (HANDY_DISABLE_UPDATER)".into(),
        );
    }

    let mut settings = settings::get_settings(&app);
    settings.update_checks_enabled = enabled;
    settings::write_settings(&app, settings);

    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "update_checks_enabled",
            "value": enabled
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_show_whats_new_on_update_setting(
    app: AppHandle,
    enabled: bool,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.show_whats_new_on_update = enabled;
    settings::write_settings(&app, settings);

    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "show_whats_new_on_update",
            "value": enabled
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_whats_new_last_seen_version_setting(
    app: AppHandle,
    version: String,
) -> Result<(), String> {
    let version = version.trim().to_string();
    let mut settings = settings::get_settings(&app);
    settings.whats_new_last_seen_version = version.clone();
    settings::write_settings(&app, settings);

    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "whats_new_last_seen_version",
            "value": version
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn update_custom_words(app: AppHandle, words: Vec<String>) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.custom_words = words;
    settings::write_settings(&app, settings);
    Ok(())
}

// ---------------------------------------------------------------------------
// 热词四态档位（lexicon_term_modes）
// ---------------------------------------------------------------------------

/// 校验一个档位覆盖项的 key（= 替换目标 primary）：trim 后非空，且
/// 不含换行、`|`（目标/别名分隔符）或 `~~~`（黑名单分隔符）——含这些
/// 字符的串不可能是合法 primary，直接拒绝而不是静默存进去。
fn validate_lexicon_term(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("term: must be non-empty after trim".to_string());
    }
    if trimmed.contains(['\n', '\r', '|']) || trimmed.contains("~~~") {
        return Err(
            "term: must not contain newlines, '|' or '~~~' (hotword syntax separators)".to_string(),
        );
    }
    Ok(trimmed.to_string())
}

/// 设置/更新一个热词条目的四态档位（`candidate` / `active_exact` /
/// `active_fuzzy` / `blocked`，大小写不敏感）。
///
/// key = 替换目标 primary（别名继承）；后台重新校验不信任前端传值，
/// 非法 term 或非法档位名都返回 Err，绝不静默。override 与
/// `imported_lexicons` 分离持久：重新导入不丢（Blocked 为暂时不存在
/// 的词保留）；生效热词缓存以档位表为键，变更后下一次转写即重编译。
#[tauri::command]
#[specta::specta]
pub fn set_lexicon_term_mode(app: AppHandle, term: String, mode: String) -> Result<(), String> {
    let term = validate_lexicon_term(&term)?;
    let mode = settings::TermMode::from_str_name(&mode).ok_or_else(|| {
        format!(
            "unknown term mode '{mode}': expected candidate | active_exact | active_fuzzy | blocked"
        )
    })?;

    let previous = settings::get_settings(&app);
    let mut updated = previous.clone();
    updated.lexicon_term_modes.insert(term, mode);
    // 落盘失败：恢复 store 内存中的旧值，避免运行态与磁盘不一致。
    if let Err(err) = settings::try_write_settings(&app, &updated) {
        settings::write_settings(&app, previous);
        return Err(err);
    }

    let _ = app.emit(
        "settings-changed",
        serde_json::json!({ "setting": "lexicon_term_modes" }),
    );
    Ok(())
}

/// 设置一个已安装词库来源的三档生效模式（`candidate` / `active_exact` /
/// `active_fuzzy`，大小写不敏感）。
///
/// 来源级档位作用于该源全部词条（单条 `lexicon_term_modes` override 仍
/// 优先）；`blocked` 不是来源级档位——想整源停用应直接移除来源。未安装
/// 的来源返回 Err，绝不静默建档。落盘走 `try_write_settings`（失败回滚
/// store 内存值），成功后广播 `settings-changed` 让前端重拉。
#[tauri::command]
#[specta::specta]
pub fn set_lexicon_source_mode(app: AppHandle, source: String, mode: String) -> Result<(), String> {
    let source = source.trim().to_string();
    if source.is_empty() {
        return Err("source: must be non-empty after trim".to_string());
    }
    let mode = settings::TermMode::from_str_name(&mode).ok_or_else(|| {
        format!("unknown source mode '{mode}': expected candidate | active_exact | active_fuzzy")
    })?;
    if matches!(mode, settings::TermMode::Blocked) {
        return Err(
            "source-level mode does not support 'blocked': remove the source instead".to_string(),
        );
    }

    let previous = settings::get_settings(&app);
    if !previous.imported_lexicons.contains_key(&source) {
        return Err(format!("source '{source}' is not installed"));
    }
    let mut updated = previous.clone();
    updated.lexicon_source_modes.insert(source, mode);
    // 落盘失败：恢复 store 内存中的旧值，避免运行态与磁盘不一致。
    if let Err(err) = settings::try_write_settings(&app, &updated) {
        settings::write_settings(&app, previous);
        return Err(err);
    }

    let _ = app.emit(
        "settings-changed",
        serde_json::json!({ "setting": "lexicon_source_modes" }),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 「记住这条修正」：历史条目 → 自定义热词
// ---------------------------------------------------------------------------

/// 校验「记住这条修正」弹窗里的一个词段：trim 后必须 2–50 字，且不得
/// 包含换行、`|`（热词行内目标/别名的分隔符）或 `~~~`（黑名单分隔符）
/// ——含这些字符的行会被 `parse_hotword_line` 拆错，所以直接拒绝。
fn validate_correction_segment(field: &str, value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    let char_count = trimmed.chars().count();
    if trimmed.is_empty() || char_count < 2 {
        return Err(format!("{field}: must be 2-50 characters after trim"));
    }
    if char_count > 50 {
        return Err(format!("{field}: must be 2-50 characters after trim"));
    }
    if trimmed.contains(['\n', '\r', '|']) || trimmed.contains("~~~") {
        return Err(format!(
            "{field}: must not contain newlines, '|' or '~~~' (hotword syntax separators)"
        ));
    }
    Ok(trimmed.to_string())
}

/// 把一次纠错拼成一行热词条目：`正确 | 错误`；错误写法缺省或全空白时
/// 仅写 `正确`。两段都经过 [`validate_correction_segment`] 校验，
/// 不信任前端传值。
fn compose_correction_entry(correct: &str, wrong: Option<&str>) -> Result<String, String> {
    let correct = validate_correction_segment("correct", correct)?;
    let wrong = wrong
        .map(str::trim)
        .filter(|w| !w.is_empty())
        .map(|w| validate_correction_segment("wrong", w))
        .transpose()?;
    Ok(match wrong {
        Some(wrong) => format!("{correct} | {wrong}"),
        None => correct,
    })
}

/// 若 `entry` 不在 `existing` 中，返回追加了它的新列表；已存在则返回
/// `None`（调用方据此向前端反馈“已存在”，而不是报错）。
fn merged_custom_words_with_entry(existing: &[String], entry: &str) -> Option<Vec<String>> {
    if existing.iter().any(|w| w == entry) {
        return None;
    }
    let mut merged = existing.to_vec();
    merged.push(entry.to_string());
    Some(merged)
}

/// 把历史里的一次纠错沉淀成自定义热词（`正确 | 错误`，错误写法可省略）。
///
/// 词段在后台重新校验（trim 后 2–50 字、不含换行/`|`/`~~~`），不信任
/// 前端。与现有 `custom_words` 完全相同的条目不重复写入，返回
/// `Ok(false)`；追加成功返回 `Ok(true)`。保存走 `try_write_settings`
/// （序列化/落盘失败返回 Err 并把旧值写回 store 内存，绝不静默成功），
/// 成功后广播 `settings-changed` 让前端设置存储重新拉取。引擎词表与
/// 中文热词索引每次转写都会重读 `custom_words`，无需额外刷新。
#[tauri::command]
#[specta::specta]
pub fn suggest_correction_to_hotword(
    app: AppHandle,
    correct: String,
    wrong: Option<String>,
) -> Result<bool, String> {
    let entry = compose_correction_entry(&correct, wrong.as_deref())?;

    let previous = settings::get_settings(&app);
    let Some(merged) = merged_custom_words_with_entry(&previous.custom_words, &entry) else {
        return Ok(false);
    };
    let mut updated = previous.clone();
    updated.custom_words = merged;
    if let Err(err) = settings::try_write_settings(&app, &updated) {
        // 落盘失败：恢复 store 内存中的旧值，避免运行态与磁盘不一致。
        settings::write_settings(&app, previous);
        return Err(err);
    }

    let _ = app.emit(
        "settings-changed",
        serde_json::json!({ "setting": "custom_words" }),
    );
    Ok(true)
}

// ---------------------------------------------------------------------------
// THUOCL 清华开放中文词库导入
// ---------------------------------------------------------------------------

/// THUOCL 数据文件基址。分类文件名以仓库 data/ 目录实际为准
/// （实测共 11 个分类；README 中的 medicial/econ/auto/edu/lulu 已不存在）。
/// THUOCL 词库镜像（按顺序尝试）。
///
/// 第一项是上游仓库；GitHub raw 在部分网络下不可达或很慢，因此补上
/// jsDelivr CDN。二者都是同一 `thunlp/THUOCL@master` 内容，分类文件路径
/// 必须完全一致，否则解析会因词表格式不符而失败。
const THUOCL_MIRRORS: &[&str] = &[
    "https://raw.githubusercontent.com/thunlp/THUOCL/master/data",
    "https://cdn.jsdelivr.net/gh/thunlp/THUOCL@master/data",
    "https://fastly.jsdelivr.net/gh/thunlp/THUOCL@master/data",
    "https://gcore.jsdelivr.net/gh/thunlp/THUOCL@master/data",
];

/// 解析 THUOCL 分类的候选 URL（顺序即尝试顺序）。
pub(crate) fn thuocl_urls(category: &str) -> Vec<String> {
    THUOCL_MIRRORS
        .iter()
        .map(|base| format!("{base}/THUOCL_{category}.txt"))
        .collect()
}

/// 允许导入的 THUOCL 分类，即 data/THUOCL_{分类}.txt 的实际文件名。
pub const THUOCL_CATEGORIES: &[&str] = &[
    "IT",
    "animal",
    "caijing",
    "car",
    "chengyu",
    "diming",
    "food",
    "law",
    "lishimingren",
    "medical",
    "poem",
];

/// 单个来源导入的最大词条数（THUOCL 与通用来源按各自排序规则取前 N）。
const LEXICON_MAX_WORDS: usize = 5000;

/// 单个镜像的连接+传输超时（秒）。镜像越多，单次超时必须越短，否则
/// “全都不可达”时要等 4×30s 才报错。
const THUOCL_TIMEOUT_SECS: u64 = 12;

/// 词库文件/下载响应体硬上限（字节）。防止错误页/恶意响应或超大文件
/// 把内存耗尽。THUOCL 与通用来源共用同一上限。
const LEXICON_MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

/// THUOCL 词库词的资格条件：**先 trim**、纯汉字、2–50 字。
///
/// - 必须 trim：上游文件的每行是 `词 \t 词频`，词字段**带尾随空格**
///   （实测 `THUOCL_IT.txt` 16,000 行里全部如此）。不 trim 时"纯汉字"
///   判断会把每一行都拒掉，最终报 "no eligible words"——这正是线上
///   四个镜像全部 HTTP 200 却"下载失败"的真实原因。
/// - 下限取 2 而不是 3：THUOCL 里 2 字词占 580/16000（含 DF 最高的
///   「专家」）。它们对**解码词表偏置**有用；而拼音近音纠错仍由
///   `chinese_hotwords::DEFAULT_MIN_HAN = 3` 把关，2 字词只参与引擎
///   vocabulary，不会放宽模糊替换。
pub(crate) fn is_eligible_lexicon_word(word: &str) -> bool {
    use crate::audio_toolkit::chinese_hotwords::is_han;
    let word = word.trim();
    let char_count = word.chars().count();
    (2..=50).contains(&char_count) && word.chars().all(is_han)
}

/// 词库导入预览结果：过滤后保留的词 + 因通用词规则被丢弃的词条数。
#[derive(Clone, Debug, Serialize, Type)]
pub struct LexiconImportResult {
    /// 过滤后保留的词（按各来源自身排序规则截取单来源上限）。
    pub words: Vec<String>,
    /// 被通用词过滤丢弃的词条数 = 命中内置常用词/虚词表的数量
    /// + 按 DF 排名丢弃的头部数量。
    pub filtered: usize,
}

/// 对**已按 DF 降序排好**（无 DF 的词必须整体排在有 DF 的词之后）的
/// 候选应用两级通用词过滤，返回 (保留词, 丢弃数)：
///
/// 1. 命中内置常用词/虚词表（复用 `lexicon_suggest::is_common_term`，
///    汉字走双向子串：「的时候」拦「时候」，「我们」拦「我们俩」）→ 直接丢弃；
/// 2. DF 头部丢弃：对过滤后的幸存者按 DF 排名丢弃最通用的前 10%——
///    数量取 `min(max_words, 幸存数) / 10`（max_words=5000 且词表足够大时
///    即 DF 最高的 500 个），并且只作用于带 DF 的头部块：无 DF 的纯文件
///    顺序词没有通用度信号，不参与头部丢弃。
///
/// 之后截取前 `max_words` 个。两级丢弃都计入返回的丢弃数，可复现：
/// 同一输入 + 同一 `max_words` 恒得同一结果（排序先于过滤且带稳定 tie-break）。
pub(crate) fn drop_generic_words(
    ranked: Vec<(String, Option<u64>)>,
    max_words: usize,
) -> (Vec<String>, usize) {
    use crate::managers::lexicon_suggest::is_common_term;

    // 第一级：内置常用词/虚词表（与 lexicon_suggest 建议过滤共用同一张表）。
    let mut kept: Vec<(String, Option<u64>)> = Vec::with_capacity(ranked.len());
    let mut filtered = 0usize;
    for (word, df) in ranked {
        // 常用词表只用于**纯汉字**词：专名/英文术语（GitHub、DeepSeek）本身
        // 就是我们要保留的内容，不能被“常用词”规则误杀。
        let has_ascii_alpha = word.chars().any(|c| c.is_ascii_alphabetic());
        if !has_ascii_alpha && is_common_term(&word) {
            filtered += 1;
        } else {
            kept.push((word, df));
        }
    }

    // 第二级：DF 头部丢弃。kept 的前段恰是 DF 降序块（Some 在前），
    // take_while 数出该块的长度作为上限，避免无 DF 文件被文件顺序误伤。
    // 只在**大数据集**上做 DF 头部截断：小词典（例如几百条以内的专名表）
    // 的头部恰恰是最好的专名，砍掉只会把 GitHub/DeepSeek 这类目标词误杀。
    let df_drop = max_words.min(kept.len()) / 10;
    // 只对纯汉字词做头部截断；含拉丁字母的专名/术语一律保留。
    let mut remaining = df_drop;
    let mut out: Vec<String> = Vec::with_capacity(kept.len().min(max_words));
    for (word, _) in kept {
        let is_latin = word.chars().any(|c| c.is_ascii_alphabetic());
        if !is_latin && remaining > 0 {
            remaining -= 1;
            filtered += 1;
            continue;
        }
        if out.len() >= max_words {
            break;
        }
        out.push(word);
    }

    (out, filtered)
}

/// 单个新分块是否仍让累计长度保持在硬上限内（`checked_add` 防溢出）。
/// 下载循环每收到一个分块都先检查，超限立即报错返回，不会把完整
/// 响应先收进内存再判断。
fn chunk_fits_under_cap(body_len: usize, chunk_len: usize, cap: usize) -> bool {
    body_len
        .checked_add(chunk_len)
        .map(|total| total <= cap)
        .unwrap_or(false)
}

/// 逐块把响应体读进单个缓冲：任一分块使累计长度超过 `cap` 立即
/// 报错（流式硬限，不先收完整响应），最终结果还必须是合法 UTF-8
/// —— 错误 HTML 页无法静默冒充空导入。
async fn read_capped_utf8_body(
    mut response: reqwest::Response,
    cap: usize,
    label: &str,
) -> Result<String, String> {
    let mut body: Vec<u8> = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| format!("{label}: {e}"))?
    {
        if !chunk_fits_under_cap(body.len(), chunk.len(), cap) {
            return Err(format!("{label}: response body exceeds {cap} bytes"));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|_| format!("{label}: response body is not valid UTF-8"))
}

/// 下载并解析一个 THUOCL 分类词库（仅预览用，不写入设置）。
///
/// 数据格式为每行 `词\t词频DF`。过滤条件：纯汉字、2–50 字（先 trim）；
/// 同词多行时保留最高 DF；命中内置常用词/虚词表或 DF 排名前 10% 的
/// 通用词被丢弃（见 [`drop_generic_words`]）；最终按 DF 降序取前
/// 5000 个词返回。超时 12 秒/镜像；响应体硬限 8 MiB；非 UTF-8 或
/// 过滤后没有合法词时报错，不会把错误 HTML 静默当成空导入。
#[tauri::command]
#[specta::specta]
pub async fn import_thuocl_lexicon(category: String) -> Result<LexiconImportResult, String> {
    let category = category.trim().to_string();
    if !THUOCL_CATEGORIES.contains(&category.as_str()) {
        return Err(format!("Unknown THUOCL category: {category}"));
    }
    let label = format!("Failed to download THUOCL {category} lexicon");
    let urls = thuocl_urls(&category);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(THUOCL_TIMEOUT_SECS))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    // 逐个镜像尝试：任何一个成功即返回。全部失败时把每个镜像的具体原因
    // 一并报出，避免用户只看到一句无从下手的 "download failed"。
    let mut failures: Vec<String> = Vec::with_capacity(urls.len());
    for url in &urls {
        // 每个镜像单独计时，避免一个挂死镜像吃掉全部预算。
        let response = match client
            .get(url)
            .timeout(std::time::Duration::from_secs(THUOCL_TIMEOUT_SECS))
            .send()
            .await
        {
            Ok(response) => response,
            Err(e) => {
                failures.push(format!("{url}: {e}"));
                continue;
            }
        };
        let status = response.status();
        if !status.is_success() {
            failures.push(format!("{url}: HTTP {status}"));
            continue;
        }
        if let Some(length) = response.content_length() {
            if length as usize > LEXICON_MAX_BODY_BYTES {
                failures.push(format!(
                    "{url}: response body exceeds {LEXICON_MAX_BODY_BYTES} bytes"
                ));
                continue;
            }
        }
        let body = match read_capped_utf8_body(response, LEXICON_MAX_BODY_BYTES, &label).await {
            Ok(body) => body,
            Err(e) => {
                failures.push(format!("{url}: {e}"));
                continue;
            }
        };
        let (words, filtered) = parse_thuocl_lexicon(&body, LEXICON_MAX_WORDS);
        if words.is_empty() {
            // 带上规模信息，避免"全是 no eligible words"时无从判断是格式
            // 变了还是筛选条件过严。
            failures.push(format!(
                "{url}: no eligible words (bytes={}, lines={}, utf8=ok)",
                body.len(),
                body.lines().count()
            ));
            continue;
        }
        log::info!(
            "Imported THUOCL {category}: {} word(s), {} filtered as generic, from {url}",
            words.len(),
            filtered
        );
        return Ok(LexiconImportResult { words, filtered });
    }

    Err(format!("{label}: {}", failures.join(" | ")))
}

/// 从本地文件导入一个 THUOCL 分类词库（离线兜底）。
///
/// 网络完全不可达时，用户可以用浏览器/下载工具拿到 `THUOCL_<分类>.txt`，
/// 再用这个命令导入。校验与网络路径完全一致（同一 `parse_thuocl_lexicon`
/// + 同一体积上限 + 同一通用词过滤），因此两条路径的词表质量相同。
#[tauri::command]
#[specta::specta]
pub async fn import_thuocl_lexicon_file(
    category: String,
    path: String,
) -> Result<LexiconImportResult, String> {
    let category = category.trim().to_string();
    if !THUOCL_CATEGORIES.contains(&category.as_str()) {
        return Err(format!("Unknown THUOCL category: {category}"));
    }
    let label = format!("Failed to read THUOCL {category} file");
    let metadata = tokio::fs::metadata(&path)
        .await
        .map_err(|e| format!("{label}: {e}"))?;
    if metadata.len() as usize > LEXICON_MAX_BODY_BYTES {
        return Err(format!(
            "{label}: file exceeds {LEXICON_MAX_BODY_BYTES} bytes"
        ));
    }
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|e| format!("{label}: {e}"))?;
    let body = String::from_utf8(bytes).map_err(|_| format!("{label}: not valid UTF-8"))?;
    let (words, filtered) = parse_thuocl_lexicon(&body, LEXICON_MAX_WORDS);
    if words.is_empty() {
        return Err(format!("THUOCL {category} file contains no eligible words"));
    }
    Ok(LexiconImportResult { words, filtered })
}

/// 通用词库来源名的资格条件：**先 trim**，之后 1–32 个字符（按字符数，
/// 不是字节数），不含控制字符，不含路径分隔符（`/` 与 `\`）。
///
/// THUOCL 分类名（如 `IT`、`medical`）天然满足这些条件，仍走
/// [`THUOCL_CATEGORIES`] 白名单的预置快捷路径；自定义来源可以是
/// 任意满足本函数的名字（中文、空格、标点都允许）。来源名只作为
/// 设置里的键使用，不会拼进任何文件路径；禁止路径分隔符是为了在
/// 源头挡住未来把来源名当路径用的误用。
pub fn normalize_custom_source_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    let char_count = name.chars().count();
    if char_count == 0 {
        return Err("Lexicon source name cannot be empty".to_string());
    }
    if char_count > 32 {
        return Err(format!(
            "Lexicon source name must be at most 32 characters (got {char_count})"
        ));
    }
    if name.chars().any(char::is_control) {
        return Err("Lexicon source name must not contain control characters".to_string());
    }
    if name.contains('/') || name.contains('\\') {
        return Err("Lexicon source name must not contain path separators".to_string());
    }
    Ok(name.to_string())
}

/// 解析通用词库文本：每行 `词` 或 `词\t词频`（DF 可缺省）。
///
/// 资格条件与 THUOCL 完全一致（先 trim、纯汉字、2–50 字）；去重时
/// 同词多行保留最高 DF（无 DF 视为低于任何有 DF 值）；排序：带 DF 的
/// 词按 DF 降序在前，无 DF 的词保持文件顺序在后——因此全无 DF 的文件
/// 就是纯文件顺序。之后应用与 THUOCL 相同的通用词过滤（见
/// [`drop_generic_words`]：无 DF 的词不参与 DF 头部丢弃），返回
/// (保留词, 丢弃数)，保留词截取前 `max_words` 个。
fn parse_generic_lexicon(body: &str, max_words: usize) -> (Vec<String>, usize) {
    use std::collections::HashMap;

    // 去掉 UTF-8 BOM；行尾的 \r 由 lines()/trim 处理。
    let body = body.strip_prefix('\u{feff}').unwrap_or(body);
    // 词 → (最高 DF（None 最低）, 首次出现序号)
    let mut best: HashMap<&str, (Option<u64>, usize)> = HashMap::new();
    for (index, line) in body.lines().enumerate() {
        let mut columns = line.split('\t');
        let Some(word) = columns.next() else {
            continue;
        };
        let word = word.trim();
        if !is_eligible_lexicon_word(word) {
            continue;
        }
        // DF 列可缺省：缺失或非数字都当作没有 DF（词仍保留，按文件顺序）。
        let df = columns.next().and_then(|df| df.trim().parse::<u64>().ok());
        let entry = best.entry(word).or_insert((df, index));
        if df > entry.0 {
            entry.0 = df;
        }
    }

    // Option<u64> 排序 None < Some(_)：b.cmp(a) 降序即「有 DF 的在前
    // （DF 高的在前），无 DF 的在后」；并列时按首次出现序号稳定排列
    //（HashMap 收集后顺序不定，必须显式 tie-break）。
    let mut ranked: Vec<(&str, (Option<u64>, usize))> = best.into_iter().collect();
    ranked.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then_with(|| a.1 .1.cmp(&b.1 .1)));

    drop_generic_words(
        ranked
            .into_iter()
            .map(|(word, (df, _))| (word.to_string(), df))
            .collect(),
        max_words,
    )
}

/// 从本地文件导入一个**通用来源**词库（仅预览用，不写入设置）。
///
/// 来源名走 [`normalize_custom_source_name`]（任意 1–32 字符，不再
/// 限于 THUOCL 分类）。文件格式：UTF-8（允许 BOM/CRLF），每行 `词`
/// 或 `词\t词频`（DF 可缺省）；资格过滤与 THUOCL 完全一致；带词频的
/// 词按 DF 降序在前、无词频的词保持文件顺序在后；导入时同样应用通用词
/// 过滤（内置常用词/虚词表 + 带 DF 词的头部 10%，见 [`drop_generic_words`]）；
/// 单来源上限 5000 词、文件体积上限 8 MiB。过滤后没有合法词时报错，
/// 不会静默当空导入。
#[tauri::command]
#[specta::specta]
pub async fn import_lexicon_file(
    name: String,
    path: String,
) -> Result<LexiconImportResult, String> {
    let name = normalize_custom_source_name(&name)?;
    let label = format!("Failed to read lexicon file for source {name}");
    let metadata = tokio::fs::metadata(&path)
        .await
        .map_err(|e| format!("{label}: {e}"))?;
    if metadata.len() as usize > LEXICON_MAX_BODY_BYTES {
        return Err(format!(
            "{label}: file exceeds {LEXICON_MAX_BODY_BYTES} bytes"
        ));
    }
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|e| format!("{label}: {e}"))?;
    let body = String::from_utf8(bytes).map_err(|_| format!("{label}: not valid UTF-8"))?;
    let (words, filtered) = parse_generic_lexicon(&body, LEXICON_MAX_WORDS);
    if words.is_empty() {
        return Err(format!(
            "Lexicon file for source {name} contains no eligible words"
        ));
    }
    log::info!(
        "Imported custom lexicon {name}: {} word(s), {} filtered as generic, from {path}",
        words.len(),
        filtered
    );
    Ok(LexiconImportResult { words, filtered })
}

/// 解析 THUOCL 词库文本：保留纯汉字且 2–50 字（先 trim）的词，按 DF 降序
/// （DF 相同时按词序稳定排列），再应用通用词过滤（内置常用词/虚词表 +
/// DF 头部 10%，见 [`drop_generic_words`]），返回 (保留词, 丢弃数)，
/// 保留词截取前 `max_words` 个。
fn parse_thuocl_lexicon(body: &str, max_words: usize) -> (Vec<String>, usize) {
    use std::collections::HashMap;

    // 去掉 UTF-8 BOM；行尾的 \r 由 trim 处理。
    let body = body.strip_prefix('\u{feff}').unwrap_or(body);
    // 词 → 最高 DF（同词多行时保留最大值）
    let mut best_df: HashMap<&str, u64> = HashMap::new();
    for line in body.lines() {
        let mut columns = line.split('\t');
        let (Some(word), Some(df)) = (columns.next(), columns.next()) else {
            continue;
        };
        // 上游词字段带尾随空格：必须先 trim 再判断资格，并以 trim 后的
        // 词作为词表内容（否则导出的词也带空格，热词永远匹配不上）。
        let word = word.trim();
        if !is_eligible_lexicon_word(word) {
            continue;
        }
        let Ok(df) = df.trim().parse::<u64>() else {
            continue;
        };
        let entry = best_df.entry(word).or_insert(0);
        if df > *entry {
            *entry = df;
        }
    }

    let mut ranked: Vec<(&str, u64)> = best_df.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    drop_generic_words(
        ranked
            .into_iter()
            .map(|(word, df)| (word.to_string(), Some(df)))
            .collect(),
        max_words,
    )
}

/// 校验并规范化一个待保存的来源快照：来源名必须是 THUOCL 分类，或满足
/// [`normalize_custom_source_name`] 的通用来源名规则（任意 1–32 字符）；
/// 词表非空、不超过 [`LEXICON_MAX_WORDS`]；逐词 trim、按序去重；每个词
/// 必须满足与预览相同的资格条件（纯汉字 2–50 字（先 trim））。返回可直接写入
/// 设置的 (来源名, 快照)。
fn normalize_lexicon_snapshot(
    source: &str,
    words: &[String],
) -> Result<(String, Vec<String>), String> {
    // THUOCL 分类走白名单；其余名字按通用来源名规则校验（同函数内
    // trim，非法名字直接报错）。THUOCL 分类名也天然满足通用规则，
    // 但白名单优先以保持预置快捷项的语义不变。
    let source = if THUOCL_CATEGORIES.contains(&source.trim()) {
        source.trim().to_string()
    } else {
        normalize_custom_source_name(source)?
    };
    if words.is_empty() {
        return Err(format!("Lexicon source {source} cannot be empty"));
    }
    if words.len() > LEXICON_MAX_WORDS {
        return Err(format!(
            "Lexicon source {source} exceeds the {LEXICON_MAX_WORDS}-word limit"
        ));
    }
    let mut snapshot: Vec<String> = Vec::with_capacity(words.len());
    for word in words {
        let word = word.trim();
        if !is_eligible_lexicon_word(word) {
            return Err(format!("Invalid lexicon word in source {source}: {word:?}"));
        }
        if !snapshot.iter().any(|w| w == word) {
            snapshot.push(word.to_string());
        }
    }
    Ok((source.to_string(), snapshot))
}

/// 保存/更新一个导入词库来源的词快照。
///
/// 来源名可以是 THUOCL 分类（预置快捷项），也可以是满足
/// [`normalize_custom_source_name`] 的自定义名；词表在后台重新验证
/// （纯汉字 2–50 字（先 trim）、非空、单来源 5000 上限），不信任前端
/// 预览结果。只写 `imported_lexicons`，手工词（`custom_words`）不受影响。
/// 保存走 `try_write_settings`（序列化/刷盘失败会返回 Err 并把旧值写回
/// store 内存，不会静默成功），成功后广播 `settings-changed` 让前端
/// 设置存储重新拉取。
#[tauri::command]
#[specta::specta]
pub fn set_imported_lexicon(
    app: AppHandle,
    source: String,
    words: Vec<String>,
) -> Result<(), String> {
    let (source, snapshot) = normalize_lexicon_snapshot(&source, &words)?;

    let previous = settings::get_settings(&app);
    let mut updated = previous.clone();
    updated.imported_lexicons.insert(source.clone(), snapshot);
    if let Err(err) = settings::try_write_settings(&app, &updated) {
        // 落盘失败：恢复 store 内存中的旧值，避免运行态与磁盘不一致。
        settings::write_settings(&app, previous);
        return Err(err);
    }

    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "imported_lexicons",
            "value": source
        }),
    );
    Ok(())
}

/// 移除一个导入词库来源（只删该来源的快照；手工词不受影响）。
/// 来源名与保存同规则（THUOCL 分类或合法自定义名）。未安装的来源
/// 报错；保存失败同样报错并恢复旧值，避免静默成功。
#[tauri::command]
#[specta::specta]
pub fn remove_imported_lexicon(app: AppHandle, source: String) -> Result<(), String> {
    let source = if THUOCL_CATEGORIES.contains(&source.trim()) {
        source.trim().to_string()
    } else {
        normalize_custom_source_name(&source)?
    };

    let previous = settings::get_settings(&app);
    if !previous.imported_lexicons.contains_key(&source) {
        return Err(format!("Lexicon source not installed: {source}"));
    }
    let mut updated = previous.clone();
    updated.imported_lexicons.remove(&source);
    if let Err(err) = settings::try_write_settings(&app, &updated) {
        settings::write_settings(&app, previous);
        return Err(err);
    }

    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "imported_lexicons",
            "value": source
        }),
    );
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_word_correction_threshold_setting(
    app: AppHandle,
    threshold: f64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.word_correction_threshold = threshold;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_hotword_rules_setting(app: AppHandle, rules: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.hotword_rules = rules;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_hotword_pinyin_threshold_setting(
    app: AppHandle,
    threshold: f64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.hotword_pinyin_threshold = threshold;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_hotword_dialect_confusion_setting(
    app: AppHandle,
    enabled: bool,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.hotword_dialect_confusion = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

/// 常用专名字面规范化总开关（内置种子 + A 组开源专名源派生的字面规则；
/// 语义见 `audio_toolkit::proper_nouns`）。
#[tauri::command]
#[specta::specta]
pub fn change_proper_noun_lexicon_enabled_setting(
    app: AppHandle,
    enabled: bool,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.proper_noun_lexicon_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_extra_recording_buffer_setting(app: AppHandle, ms: u64) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.extra_recording_buffer_ms = ms;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_paste_delay_ms_setting(app: AppHandle, ms: u64) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.paste_delay_ms = ms;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_paste_delay_after_ms_setting(app: AppHandle, ms: u64) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.paste_delay_after_ms = ms;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_reliable_paste_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.reliable_paste = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_paste_method_setting(app: AppHandle, method: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match method.as_str() {
        "ctrl_v" => PasteMethod::CtrlV,
        "direct" => PasteMethod::Direct,
        "none" => PasteMethod::None,
        "shift_insert" => PasteMethod::ShiftInsert,
        "ctrl_shift_v" => PasteMethod::CtrlShiftV,
        "external_script" => PasteMethod::ExternalScript,
        other => {
            warn!("Invalid paste method '{}', defaulting to ctrl_v", other);
            PasteMethod::CtrlV
        }
    };
    settings.paste_method = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn get_available_typing_tools() -> Vec<String> {
    #[cfg(target_os = "linux")]
    {
        crate::clipboard::get_available_typing_tools()
    }
    #[cfg(not(target_os = "linux"))]
    {
        vec!["auto".to_string()]
    }
}

#[tauri::command]
#[specta::specta]
pub fn change_typing_tool_setting(app: AppHandle, tool: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match tool.as_str() {
        "auto" => TypingTool::Auto,
        "wtype" => TypingTool::Wtype,
        "kwtype" => TypingTool::Kwtype,
        "dotool" => TypingTool::Dotool,
        "ydotool" => TypingTool::Ydotool,
        "xdotool" => TypingTool::Xdotool,
        other => {
            warn!("Invalid typing tool '{}', defaulting to auto", other);
            TypingTool::Auto
        }
    };
    settings.typing_tool = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_external_script_path_setting(
    app: AppHandle,
    path: Option<String>,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.external_script_path = path;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_clipboard_handling_setting(app: AppHandle, handling: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match handling.as_str() {
        "dont_modify" => ClipboardHandling::DontModify,
        "copy_to_clipboard" => ClipboardHandling::CopyToClipboard,
        other => {
            warn!(
                "Invalid clipboard handling '{}', defaulting to dont_modify",
                other
            );
            ClipboardHandling::DontModify
        }
    };
    settings.clipboard_handling = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

/// 后处理思考深度（none/low/medium/high）。未知值静默回落到 none：
/// 宁可不思考，也不要因为设置项拼错而悄悄变慢。
#[tauri::command]
#[specta::specta]
pub fn change_post_process_thinking_setting(app: AppHandle, level: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match level.as_str() {
        "none" => ThinkingLevel::None,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        other => {
            warn!(
                "Invalid post-process thinking level '{}', using none",
                other
            );
            ThinkingLevel::None
        }
    };
    settings.post_process_thinking = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_auto_submit_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.auto_submit = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_auto_submit_key_setting(app: AppHandle, key: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match key.as_str() {
        "enter" => AutoSubmitKey::Enter,
        "ctrl_enter" => AutoSubmitKey::CtrlEnter,
        "cmd_enter" => AutoSubmitKey::CmdEnter,
        other => {
            warn!("Invalid auto submit key '{}', defaulting to enter", other);
            AutoSubmitKey::Enter
        }
    };
    settings.auto_submit_key = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_context_enabled_setting(
    app: AppHandle,
    enabled: bool,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_process_context_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_process_enabled = enabled;
    settings::write_settings(&app, settings.clone());

    // Keep the dedicated `transcribe_with_post_process` binding's live
    // registration in sync with the switch (same gate the init paths use).
    // Best-effort: a failure here must not fail the settings write.
    if let Some(binding) = settings
        .bindings
        .get("transcribe_with_post_process")
        .cloned()
    {
        let result = if enabled {
            register_shortcut(&app, binding)
        } else {
            unregister_shortcut(&app, binding)
        };
        if let Err(e) = result {
            warn!(
                "Could not {} dedicated post-processing shortcut: {}",
                if enabled { "register" } else { "unregister" },
                e
            );
        }
    }

    crate::secure_input::reconcile_fallback(&app);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_mode_setting(
    app: AppHandle,
    mode: PostProcessMode,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_process_mode = mode;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_experimental_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.experimental_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_base_url_setting(
    app: AppHandle,
    provider_id: String,
    base_url: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let label = settings
        .post_process_provider(&provider_id)
        .map(|provider| provider.label.clone())
        .ok_or_else(|| format!("Provider '{}' not found", provider_id))?;

    let provider = settings
        .post_process_provider_mut(&provider_id)
        .expect("Provider looked up above must exist");

    if provider.id != "custom" {
        return Err(format!(
            "Provider '{}' does not allow editing the base URL",
            label
        ));
    }

    provider.base_url = base_url;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_api_format_setting(
    app: AppHandle,
    provider_id: String,
    api_format: settings::PostProcessApiFormat,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let provider = settings
        .post_process_provider_mut(&provider_id)
        .ok_or_else(|| format!("Provider '{}' not found", provider_id))?;

    if provider.id != "custom" {
        return Err(format!(
            "Provider '{}' does not allow changing the API format",
            provider.label
        ));
    }

    provider.api_format = api_format;
    settings::write_settings(&app, settings);
    Ok(())
}

/// Generic helper to validate provider exists
fn validate_provider_exists(
    settings: &settings::AppSettings,
    provider_id: &str,
) -> Result<(), String> {
    if !settings
        .post_process_providers
        .iter()
        .any(|provider| provider.id == provider_id)
    {
        return Err(format!("Provider '{}' not found", provider_id));
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_api_key_setting(
    app: AppHandle,
    provider_id: String,
    api_key: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    validate_provider_exists(&settings, &provider_id)?;
    settings.post_process_api_keys.insert(provider_id, api_key);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_model_setting(
    app: AppHandle,
    provider_id: String,
    model: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    validate_provider_exists(&settings, &provider_id)?;
    settings.post_process_models.insert(provider_id, model);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn set_post_process_provider(app: AppHandle, provider_id: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    validate_provider_exists(&settings, &provider_id)?;
    settings.post_process_provider_id = provider_id;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn add_post_process_prompt(
    app: AppHandle,
    name: String,
    prompt: String,
) -> Result<LLMPrompt, String> {
    let mut settings = settings::get_settings(&app);

    // Generate unique ID using timestamp and random component
    let id = format!("prompt_{}", chrono::Utc::now().timestamp_millis());

    let new_prompt = LLMPrompt {
        id: id.clone(),
        name,
        prompt,
    };

    settings.post_process_prompts.push(new_prompt.clone());
    settings::write_settings(&app, settings);

    Ok(new_prompt)
}

#[tauri::command]
#[specta::specta]
pub fn update_post_process_prompt(
    app: AppHandle,
    id: String,
    name: String,
    prompt: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);

    if let Some(existing_prompt) = settings
        .post_process_prompts
        .iter_mut()
        .find(|p| p.id == id)
    {
        existing_prompt.name = name;
        existing_prompt.prompt = prompt;
        settings::write_settings(&app, settings);
        Ok(())
    } else {
        Err(format!("Prompt with id '{}' not found", id))
    }
}

#[tauri::command]
#[specta::specta]
pub fn delete_post_process_prompt(app: AppHandle, id: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);

    // Don't allow deleting the last prompt
    if settings.post_process_prompts.len() <= 1 {
        return Err("Cannot delete the last prompt".to_string());
    }

    // Find and remove the prompt
    let original_len = settings.post_process_prompts.len();
    settings.post_process_prompts.retain(|p| p.id != id);

    if settings.post_process_prompts.len() == original_len {
        return Err(format!("Prompt with id '{}' not found", id));
    }

    // If the deleted prompt was selected, select the first one or None
    if settings.post_process_selected_prompt_id.as_ref() == Some(&id) {
        settings.post_process_selected_prompt_id =
            settings.post_process_prompts.first().map(|p| p.id.clone());
    }

    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn fetch_post_process_models(
    app: AppHandle,
    provider_id: String,
) -> Result<Vec<String>, String> {
    let settings = settings::get_settings(&app);

    // Find the provider
    let provider = settings
        .post_process_providers
        .iter()
        .find(|p| p.id == provider_id)
        .ok_or_else(|| format!("Provider '{}' not found", provider_id))?;

    if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return Ok(vec![APPLE_INTELLIGENCE_DEFAULT_MODEL_ID.to_string()]);
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            return Err("Apple Intelligence is only available on Apple silicon Macs running macOS 15 or later.".to_string());
        }
    }

    // Get API key
    let api_key = settings
        .post_process_api_keys
        .get(&provider_id)
        .cloned()
        .unwrap_or_default();

    // Skip fetching if no API key for providers that typically need one
    if api_key.trim().is_empty() && provider.id != "custom" {
        return Err(format!(
            "API key is required for {}. Please add an API key to list available models.",
            provider.label
        ));
    }

    crate::llm_client::fetch_models(provider, api_key).await
}

#[tauri::command]
#[specta::specta]
pub fn set_post_process_selected_prompt(app: AppHandle, id: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);

    // Verify the prompt exists
    if !settings.post_process_prompts.iter().any(|p| p.id == id) {
        return Err(format!("Prompt with id '{}' not found", id));
    }

    settings.post_process_selected_prompt_id = Some(id);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_mute_while_recording_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.mute_while_recording = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_append_trailing_space_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.append_trailing_space = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_lazy_stream_close_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.lazy_stream_close = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_vad_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.vad_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_decode_vocabulary_bias_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.decode_vocabulary_bias = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_incremental_commit_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.incremental_commit = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_input_level_normalization_setting(
    app: AppHandle,
    enabled: bool,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.input_level_normalization = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_long_form_chunking_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.long_form_chunking = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn change_vad_backend_setting(app: AppHandle, backend: VadBackend) -> Result<(), String> {
    if settings::get_settings(&app).vad_backend == backend {
        return Ok(());
    }

    // Construct/swap the detector and, when necessary, reopen cpal away from
    // the webview thread. Persist only after the runtime change succeeds so a
    // rejected in-progress switch or failed microphone reopen rolls back cleanly.
    let manager = app
        .state::<std::sync::Arc<crate::managers::audio::AudioRecordingManager>>()
        .inner()
        .clone();
    tokio::task::spawn_blocking(move || manager.update_vad_backend(backend))
        .await
        .map_err(|e| format!("audio task join failed: {e}"))?
        .map_err(|e| format!("Failed to update VAD backend: {e}"))?;

    let mut current_settings = settings::get_settings(&app);
    current_settings.vad_backend = backend;
    settings::write_settings(&app, current_settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_filler_word_removal_enabled_setting(
    app: AppHandle,
    enabled: bool,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.filler_word_removal_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_chinese_script_setting(app: AppHandle, script: ChineseScript) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.chinese_script = script;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_app_language_setting(app: AppHandle, language: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.app_language = language.clone();
    settings::write_settings(&app, settings);

    // Refresh the tray menu with the new language
    tray::update_tray_menu(&app);

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_show_tray_icon_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.show_tray_icon = enabled;
    settings::write_settings(&app, settings);

    // Apply change immediately
    tray::set_tray_visibility(&app, enabled);

    Ok(())
}

/// Save accelerator settings and make the next model use reload with them.
/// The currently running transcription, if any, keeps its existing engine.
fn save_accelerator_and_reload_next_use(app: &AppHandle, s: settings::AppSettings) {
    settings::write_settings(app, s);

    let tm = app.state::<std::sync::Arc<crate::managers::transcription::TranscriptionManager>>();
    tm.reload_model_on_next_use();
}

#[tauri::command]
#[specta::specta]
pub fn change_transcribe_accelerator_setting(
    app: AppHandle,
    accelerator: settings::TranscribeAcceleratorSetting,
) -> Result<(), String> {
    let mut s = settings::get_settings(&app);
    s.transcribe_accelerator = accelerator;
    save_accelerator_and_reload_next_use(&app, s);
    app.state::<std::sync::Arc<crate::managers::transcription::TranscriptionManager>>()
        .retry_transcribe_gpu("the transcribe.cpp accelerator setting changed");
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_transcribe_gpu_device(app: AppHandle, device: Option<String>) -> Result<(), String> {
    let mut s = settings::get_settings(&app);
    s.transcribe_gpu_device = device;
    save_accelerator_and_reload_next_use(&app, s);
    app.state::<std::sync::Arc<crate::managers::transcription::TranscriptionManager>>()
        .retry_transcribe_gpu("the transcribe.cpp GPU device setting changed");
    Ok(())
}

/// Return which accelerators and GPU devices are available for this build.
///
/// First-call cost is dominated by enumerating GPU devices through the
/// transcribe.cpp Metal/Vulkan backend, which loads dynamic libraries and
/// probes hardware. Run it on the blocking pool so the webview thread
/// stays responsive — see also the startup pre-warm in `lib.rs`.
#[tauri::command]
#[specta::specta]
pub async fn get_available_accelerators(
    app: AppHandle,
) -> crate::managers::transcription::AvailableAccelerators {
    tauri::async_runtime::spawn_blocking(move || {
        let tm =
            app.state::<std::sync::Arc<crate::managers::transcription::TranscriptionManager>>();
        crate::managers::transcription::get_available_accelerators(&tm)
    })
    .await
    .expect("get_available_accelerators panicked")
}

#[cfg(test)]
mod tests {
    use handy_keys::Hotkey;
    use tauri_plugin_global_shortcut::Shortcut;

    #[test]
    fn compound_shortcut_keys_parse_on_both_backends() {
        for key in [
            "scrolllock",
            "capslock",
            "numlock",
            "pageup",
            "pagedown",
            "printscreen",
        ] {
            assert!(key.parse::<Shortcut>().is_ok(), "Tauri rejected {key}");
            assert!(key.parse::<Hotkey>().is_ok(), "HandyKeys rejected {key}");
        }
    }

    #[test]
    fn thuocl_parser_filters_ranks_and_caps() {
        let long_word = "超".repeat(51);
        let body = format!(
            "机器学习\t100\n\
             人工智能\t200\n\
             机器学习\t50\n\
             abc\t999\n\
             汉\t500\n\
             {long_word}\t400\n\
             无频列\n\
             医学影像\tnotanumber\n"
        );
        let (words, _filtered) = super::parse_thuocl_lexicon(&body, 5000);
        // 非纯汉字、过短、过长、缺列、DF 非数字的行全部过滤；重复词保留最高 DF
        assert_eq!(words, vec!["人工智能".to_string(), "机器学习".to_string()]);
    }

    #[test]
    fn thuocl_parser_respects_cap() {
        // 后缀必须保持纯汉字：阿拉伯数字会让词被资格过滤掉。
        let stems = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
        let body: String = (0..10)
            .map(|i| format!("词库测试{}\t{}\n", stems[i], 1000 - i))
            .collect();
        let (words, _filtered) = super::parse_thuocl_lexicon(&body, 3);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0], "词库测试甲");
        assert_eq!(words[2], "词库测试丙");
    }

    #[test]
    fn lexicon_snapshot_trims_dedupes_and_normalizes_source() {
        let (source, snapshot) = super::normalize_lexicon_snapshot(
            "  medical  ",
            &[
                "医学影像".to_string(),
                " 医学影像 ".to_string(),
                "临床试验".to_string(),
            ],
        )
        .expect("valid snapshot");
        assert_eq!(source, "medical");
        assert_eq!(
            snapshot,
            vec!["医学影像".to_string(), "临床试验".to_string()]
        );
    }

    #[test]
    /// 回归：上游真实格式是 `词 \t 词频`，词字段带**尾随空格**。
    /// 曾经的实现不 trim 就判"纯汉字"，导致 16,000 行全部被拒、
    /// 四个镜像都报 "no eligible words"。
    #[test]
    fn thuocl_parser_accepts_the_real_upstream_format() {
        let body = "专家 \t 395499\n创业 \t 372767\n数据组织 \t 357924\n";
        let (words, _filtered) = super::parse_thuocl_lexicon(body, 100);
        assert_eq!(
            words,
            vec![
                "专家".to_string(),
                "创业".to_string(),
                "数据组织".to_string()
            ],
            "词必须 trim 掉尾随空格，且 2 字词应被接受"
        );
    }

    #[test]
    fn thuocl_parser_handles_bom_crlf_and_rejects_out_of_policy_words() {
        let body = "\u{feff}知识图谱 \t 100\r\n中 \t 50\r\n纯 English \t 40\r\n知 \t 30\r\n";
        let (words, _filtered) = super::parse_thuocl_lexicon(body, 100);
        assert_eq!(
            words,
            vec!["知识图谱".to_string()],
            "只保留 2–50 字的纯汉字词"
        );
    }

    #[test]
    fn thuocl_parser_keeps_highest_df_and_caps_results() {
        let body = "机器学习 \t 10\n机器学习 \t 99\n深度学习 \t 50\n";
        let (words, _filtered) = super::parse_thuocl_lexicon(body, 1);
        assert_eq!(
            words,
            vec!["机器学习".to_string()],
            "同词取最高 DF 并受上限约束"
        );
    }

    #[test]
    fn thuocl_parser_filters_builtin_common_words_and_counts() {
        // 手机/时候 命中内置常用词/虚词表（lexicon_suggest::COMMON_HAN_WORDS），
        // DF 再高也直接丢弃；专名按 DF 降序保留。不合格词（非纯汉字/单字）
        // 不计入 filtered——它们本就不在候选里。
        let body = "手机\t9999\n时候\t9998\n甲乙丙丁戊\t100\n己庚辛壬癸\t90\n子丑寅卯辰\t80\nabc\t1\n知\t5\n";
        let (words, filtered) = super::parse_thuocl_lexicon(&body, 5);
        assert_eq!(
            words,
            vec![
                "甲乙丙丁戊".to_string(),
                "己庚辛壬癸".to_string(),
                "子丑寅卯辰".to_string()
            ]
        );
        assert_eq!(filtered, 2, "只有两个内置表命中计入 filtered");
    }

    #[test]
    fn thuocl_parser_drops_df_head_ten_percent_and_is_reproducible() {
        // 20 个带 DF 的专名：头部丢弃 = min(max_words=20, 20) / 10 = 2，
        // 即 DF 最高的两个（甲甲 2000、乙乙 1990）被当作最通用词丢弃。
        let stems = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
        let body: String = (0..20)
            .map(|i| {
                format!(
                    "词库测试{}{}\t{}\n",
                    stems[i % 10],
                    stems[i / 10],
                    2000 - i * 10
                )
            })
            .collect();
        let (words, filtered) = super::parse_thuocl_lexicon(&body, 20);
        assert_eq!(words.len(), 18);
        assert_eq!(words[0], "词库测试丙甲", "DF 最高的前两个被丢弃");
        assert!(!words.contains(&"词库测试甲甲".to_string()));
        assert!(!words.contains(&"词库测试乙甲".to_string()));
        assert_eq!(filtered, 2, "DF 头部丢弃计入 filtered");
        // 可复现：同一输入 + 同一 max_words 恒得同一结果
        let (words2, filtered2) = super::parse_thuocl_lexicon(&body, 20);
        assert_eq!((words, filtered), (words2, filtered2));
    }

    #[test]
    fn thuocl_parser_counts_builtin_and_df_head_drops_together() {
        // 2 个内置表命中 + DF 头部 min(20, 20) / 10 = 2：filtered 合计 4。
        let stems = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
        let mut body = String::from("手机\t99999\n电话\t99998\n");
        for i in 0..20 {
            body.push_str(&format!(
                "词库测试{}{}\t{}\n",
                stems[i % 10],
                stems[i / 10],
                2000 - i * 10
            ));
        }
        let (words, filtered) = super::parse_thuocl_lexicon(&body, 20);
        assert_eq!(filtered, 4, "内置表命中 2 + DF 头部 2");
        assert_eq!(words.len(), 18);
        // DF 头部丢弃发生在内置表过滤之后：最高 DF 的手机/电话已计入
        // 第一级，头部再丢甲甲、乙甲两个。
        assert!(!words.contains(&"词库测试甲甲".to_string()));
        assert_eq!(words[0], "词库测试丙甲");
    }

    #[test]
    fn generic_parser_filters_common_words_but_skips_df_head_without_df() {
        // 无 DF 列的文件：只有内置表命中计入 filtered，不触发 DF 头部丢弃
        //（文件顺序不是通用度信号）。
        let body = "手机\n神经元机器\n临床对照试验\n";
        let (words, filtered) = super::parse_generic_lexicon(body, 5000);
        assert_eq!(
            words,
            vec!["神经元机器".to_string(), "临床对照试验".to_string()]
        );
        assert_eq!(filtered, 1);
    }

    #[test]
    fn generic_parser_drops_df_head_with_df_words() {
        // 带 DF 的通用文件与 THUOCL 同规则：min(12, 12) / 10 = 1 个头部丢弃。
        let stems = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
        let body: String = (0..12)
            .map(|i| format!("词测{}{}\t{}\n", stems[i % 10], stems[i / 10], 500 - i))
            .collect();
        let (words, filtered) = super::parse_generic_lexicon(&body, 12);
        assert_eq!(words.len(), 11);
        assert_eq!(words[0], "词测乙甲", "DF 最高的词测甲甲被头部丢弃");
        assert_eq!(filtered, 1);
    }

    #[test]
    fn thuocl_urls_cover_primary_and_cdn_mirror_in_order() {
        let urls = super::thuocl_urls("IT");
        assert_eq!(urls.len(), 4, "every category must try all mirrors");
        assert!(urls[0].starts_with("https://raw.githubusercontent.com/thunlp/THUOCL/"));
        assert!(urls[1].starts_with("https://cdn.jsdelivr.net/gh/thunlp/THUOCL@"));
        assert!(urls[2].starts_with("https://fastly.jsdelivr.net/gh/thunlp/THUOCL@"));
        assert!(urls[3].starts_with("https://gcore.jsdelivr.net/gh/thunlp/THUOCL@"));
        for url in &urls {
            assert!(
                url.ends_with("/THUOCL_IT.txt"),
                "mirror must expose the same category path: {url}"
            );
        }
    }

    fn lexicon_snapshot_rejects_bad_saves() {
        // 未知来源
        assert!(super::normalize_lexicon_snapshot("econ", &["词库测试甲".to_string()]).is_err());
        // 空词表 / 只剩空白词
        assert!(super::normalize_lexicon_snapshot("IT", &[]).is_err());
        assert!(super::normalize_lexicon_snapshot("IT", &["  ".to_string()]).is_err());
        // 非纯汉字、过短、过长
        assert!(super::normalize_lexicon_snapshot("IT", &["abc词汇".to_string()]).is_err());
        assert!(super::normalize_lexicon_snapshot("IT", &["汉字".to_string()]).is_err());
        assert!(super::normalize_lexicon_snapshot("IT", &["超".repeat(51)]).is_err());
        // 超过单来源上限
        let too_many: Vec<String> = (0..=5000).map(|i| format!("词库测试{i}")).collect();
        assert!(super::normalize_lexicon_snapshot("IT", &too_many).is_err());
    }

    #[test]
    fn chunk_cap_checks_cumulative_size_and_overflow() {
        // 恰好在限额内 / 一旦越界立即拒绝（含零长分块）
        assert!(super::chunk_fits_under_cap(4, 4, 8));
        assert!(!super::chunk_fits_under_cap(4, 5, 8));
        assert!(!super::chunk_fits_under_cap(9, 0, 8));
        // checked_add 防溢出：累加溢出 usize 时按超限处理
        assert!(!super::chunk_fits_under_cap(usize::MAX, 1, usize::MAX));
    }

    #[test]
    fn correction_entry_composes_and_appends() {
        // 正常追加：两段都 trim，拼成 `正确 | 错误`，尾随原列表之后
        let entry = super::compose_correction_entry("机器学习", Some(" 机七学习 "))
            .expect("valid correction should compose");
        assert_eq!(entry, "机器学习 | 机七学习");
        let existing = vec!["已有热词".to_string()];
        let merged =
            super::merged_custom_words_with_entry(&existing, &entry).expect("new entry appends");
        assert_eq!(
            merged,
            vec!["已有热词".to_string(), "机器学习 | 机七学习".to_string()]
        );
    }

    #[test]
    fn correction_entry_dedupes_against_existing_words() {
        // 与现有 custom_words 完全相同的条目不重复写入（Ok(false) 路径）
        let entry = "机器学习 | 机七学习";
        let existing = vec!["已有热词".to_string(), entry.to_string()];
        assert!(super::merged_custom_words_with_entry(&existing, entry).is_none());
        // 仅目标相同但别名不同的条目是另一行，仍允许追加
        let other = "机器学习 | 机器学习";
        assert!(super::merged_custom_words_with_entry(&existing, other).is_some());
    }

    #[test]
    fn correction_entry_rejects_newlines_and_pipes() {
        // 换行（含 \r）与竖线是热词语法分隔符，出现在任一段都拒绝
        assert!(super::compose_correction_entry("正确\n写法", None).is_err());
        assert!(super::compose_correction_entry("正确\r写法", None).is_err());
        assert!(super::compose_correction_entry("机器学习", Some("错|误")).is_err());
        assert!(super::compose_correction_entry("机|器", None).is_err());
        // `~~~` 是黑名单分隔符，同样会拆错整行
        assert!(super::compose_correction_entry("错~~~误", None).is_err());
        assert!(super::compose_correction_entry("机器学习", Some("错误~~~黑名单")).is_err());
    }

    #[test]
    fn correction_entry_rejects_empty_or_overlong_correct() {
        // correct 为空 / 全空白 / 单字都拒绝
        assert!(super::compose_correction_entry("", None).is_err());
        assert!(super::compose_correction_entry("   ", None).is_err());
        assert!(super::compose_correction_entry("字", None).is_err());
        // 超长（>50 字）拒绝，边界值 2 与 50 字合法
        let fifty_one = "词".repeat(51);
        assert!(super::compose_correction_entry(&fifty_one, None).is_err());
        assert!(super::compose_correction_entry("两字", None).is_ok());
        let fifty = "词".repeat(50);
        assert_eq!(
            super::compose_correction_entry(&fifty, None).unwrap(),
            fifty
        );
        // wrong 同样受 2–50 字限制
        assert!(super::compose_correction_entry("机器学习", Some("误")).is_err());
        assert!(super::compose_correction_entry("机器学习", Some(&"误".repeat(51))).is_err());
    }

    #[test]
    fn correction_entry_without_wrong_writes_correct_only() {
        // wrong 为 None / 空串 / 全空白时只写 correct，不产生空别名段
        assert_eq!(
            super::compose_correction_entry("机器学习", None).unwrap(),
            "机器学习"
        );
        assert_eq!(
            super::compose_correction_entry(" 机器学习 ", Some("")).unwrap(),
            "机器学习"
        );
        let entry = super::compose_correction_entry("机器学习", Some("   "))
            .expect("blank wrong is absent");
        assert_eq!(entry, "机器学习");
        assert!(!entry.contains('|'));
    }
}
