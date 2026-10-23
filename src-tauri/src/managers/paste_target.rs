//! 增量上屏的粘贴目标绑定（target guard）。
//!
//! 安全要求（增量会话）：
//! - 会话开始时记录前台窗口身份（平台最佳努力）；每段粘贴前重新校验；
//!   身份变化则**不得粘贴**：保留该段文本、把会话标记为 blocked、只记
//!   debug 日志（含原因），绝不贴到新焦点、绝不把焦点抢回来。
//! - 无法获取窗口身份的平台/环境（如 Wayland）：**静默**按 best-effort
//!   继续粘贴——不告警、不做 warned-once 逻辑、不打印任何 warn。
//!
//! 平台实现（都是最佳努力，[`capture`] 返回 [`None`] 即"此环境不可用"）：
//! - Windows：`GetForegroundWindow` + `GetWindowThreadProcessId`（进程级
//!   身份：同一应用换窗口不误伤，跨应用切换必拦）。
//! - macOS：`NSWorkspace.frontmostApplication` 的 bundle identifier
//!   （退回 pid）。只在 `macos` cfg 下编译。
//! - Linux/X11：`xdotool getactivewindow getwindowpid` 取前台窗口 PID。
//!   不用 wmctrl：它只能枚举窗口、无法指认活动窗口。Wayland 没有统一
//!   查询口——返回 [`None`]，不假装成功。
//!
//! 判定核心 [`decide`] 是纯函数（平台无关、可单测）；[`TargetGuard`] 是
//! 包了平台探测的会话状态机，探测函数可注入，测试不依赖真实桌面。

use log::debug;

/// 会话起点身份的探测函数。生产实现是 [`capture`]；测试注入假实现。
pub type CaptureFn = Box<dyn Fn() -> Option<PasteTarget> + Send>;

/// 一次粘贴会话的目标身份。平台无关：具体身份在 [`capture`] 时折算成
/// 一个不透明 `key`（只比较相等）与一个可读 `label`（只进日志，不含
/// 窗口标题，避免泄露用户内容）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PasteTarget {
    key: String,
    label: String,
}

impl PasteTarget {
    /// 直接以字符串构造（测试与非平台路径用）。
    pub fn new(key: &str, label: &str) -> Self {
        Self {
            key: key.to_string(),
            label: label.to_string(),
        }
    }

    /// 身份比较用的不透明键。
    pub fn key(&self) -> &str {
        &self.key
    }

    /// 日志用的可读描述（进程名/pid，无窗口标题）。
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// 平台最佳努力地识别当前前台窗口。
pub fn capture() -> Option<PasteTarget> {
    platform_capture()
}

/// App 标识的清洗上限（Unicode 标量数）。
const MAX_APP_IDENTITY_CHARS: usize = 64;

/// 前台应用的短标识（App 名 / bundle id 这类，**不含窗口标题**）。
/// 与 [`capture`] 同一套前台窗口捕获，供 LLM 后处理的上下文注入复用；
/// 任何失败返回 `None`（调用方静默跳过，不告警）。返回值已清洗：无控
/// 制字符、无尖括号（防数据字段标签逃逸）、空白折叠、
/// ≤[`MAX_APP_IDENTITY_CHARS`] 码点。
pub fn capture_app_identity() -> Option<String> {
    sanitize_app_identity(&platform_app_identity()?)
}

/// 清洗 App 标识：控制字符与尖括号剔除、连续空白折叠为单个空格、截
/// 到 [`MAX_APP_IDENTITY_CHARS`] 码点；清洗后为空则无效（`None`）。
/// 纯函数，平台无关可单测。
pub(crate) fn sanitize_app_identity(raw: &str) -> Option<String> {
    let mut out = String::new();
    let mut last_was_space = false;
    for ch in raw.chars() {
        if out.chars().count() >= MAX_APP_IDENTITY_CHARS {
            break;
        }
        if ch.is_control() {
            // 换行/制表符属于空白：折叠成一个空格（不能让 "第一行\n第二行"
            // 粘成 "第一行第二行"）；其他控制字符直接丢弃。
            if ch.is_whitespace() && !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
            continue;
        }
        if ch == '<' || ch == '>' {
            continue;
        }
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    let trimmed = out.trim().to_string();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// 从可执行文件完整路径取文件名，并去掉多余的 `.exe` 后缀（大小写不
/// 敏感）；空文件名返回 `None`。纯函数，可单测。
fn exe_file_name(path: &str) -> Option<String> {
    let name = path.rsplit(['\\', '/']).next()?;
    let name = if name.len() >= 4 && name[name.len() - 4..].eq_ignore_ascii_case(".exe") {
        &name[..name.len() - 4]
    } else {
        name
    };
    (!name.is_empty()).then_some(name.to_string())
}

#[cfg(target_os = "windows")]
fn platform_app_identity() -> Option<String> {
    use windows::core::PWSTR;
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

    // 与 platform_capture 同一条链：前台窗口 → PID → 进程名（只取文件
    // 名，不碰窗口标题）。SAFETY: 只读查询；句柄仅本次取名使用，随即
    // 关闭，不做任何存储复用。
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.0.is_null() {
            return None;
        }
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if pid == 0 {
            return None;
        }
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut buffer = [0u16; 1024];
        let mut len = buffer.len() as u32;
        let queried = QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            PWSTR(buffer.as_mut_ptr()),
            &mut len,
        )
        .is_ok();
        let _ = CloseHandle(process);
        if !queried || len == 0 || len as usize > buffer.len() {
            return None;
        }
        let path = String::from_utf16_lossy(&buffer[..len as usize]);
        exe_file_name(&path)
    }
}

#[cfg(target_os = "macos")]
fn platform_app_identity() -> Option<String> {
    use objc2_app_kit::NSWorkspace;

    // 只读 KVO 属性：App 本地名优先（对纠正术语最有用），退回 bundle id。
    let frontmost = NSWorkspace::sharedWorkspace().frontmostApplication()?;
    if let Some(name) = frontmost.localizedName() {
        return Some(name.to_string());
    }
    frontmost
        .bundleIdentifier()
        .map(|bundle| bundle.to_string())
}

#[cfg(target_os = "linux")]
fn platform_app_identity() -> Option<String> {
    use std::process::Command;

    // 与 platform_capture 同一条 xdotool 链取前台窗口 PID，再读 /proc 的
    // 进程名（comm）。不取窗口标题。Wayland/无 xdotool → None，不假装。
    let output = Command::new("xdotool")
        .args(["getactivewindow", "getwindowpid"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let pid = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if pid.is_empty() || !pid.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
    let name = comm.lines().next().unwrap_or("").trim();
    (!name.is_empty()).then_some(name.to_string())
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn platform_app_identity() -> Option<String> {
    None
}

#[cfg(target_os = "windows")]
fn platform_capture() -> Option<PasteTarget> {
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

    // SAFETY: 只读查询 Win32 API；句柄仅本次比较使用，不做任何存储复用。
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.0.is_null() {
            return None;
        }
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if pid == 0 {
            return None;
        }
        Some(PasteTarget {
            key: format!("windows-pid:{pid}"),
            label: format!("pid {pid} (hwnd {:p})", hwnd.0),
        })
    }
}

#[cfg(target_os = "macos")]
fn platform_capture() -> Option<PasteTarget> {
    use objc2_app_kit::NSWorkspace;

    // 只读 KVO 属性，不持有 UI 对象；bundle id / pid 即进程级身份。
    let frontmost = NSWorkspace::sharedWorkspace().frontmostApplication()?;
    let pid = frontmost.processIdentifier();
    let key = match frontmost.bundleIdentifier() {
        Some(bundle) => format!("macos-bundle:{}", bundle.to_string()),
        None => format!("macos-pid:{pid}"),
    };
    let label = match frontmost.localizedName() {
        Some(name) => format!("{name} (pid {pid})"),
        None => format!("pid {pid}"),
    };
    Some(PasteTarget { key, label })
}

#[cfg(target_os = "linux")]
fn platform_capture() -> Option<PasteTarget> {
    use std::process::Command;

    // X11：xdotool 链式取"活动窗口 → 其 PID"。任何失败（未安装、X 会话
    // 缺失、Wayland 下的工具报错）都返回 None——绝不假装拿到身份。
    let output = Command::new("xdotool")
        .args(["getactivewindow", "getwindowpid"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let pid = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if pid.is_empty() {
        return None;
    }
    Some(PasteTarget {
        key: format!("x11-pid:{pid}"),
        label: format!("pid {pid}"),
    })
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
fn platform_capture() -> Option<PasteTarget> {
    None
}

/// 纯决策核心：给定会话起点身份与当前身份，得出是否允许粘贴。
/// 平台无关，全部逻辑可单测。
#[derive(Debug, PartialEq, Eq)]
pub enum TargetDecision {
    /// 身份一致：允许粘贴。
    Allow,
    /// 起点就没有身份（平台不支持/查询失败）：允许粘贴，调用方**静默**
    /// 继续（best-effort，不告警、不记日志）。
    Unavailable,
    /// 不再是起点身份（变化或无法复核）：拒绝粘贴，且本会话不得恢复。
    Block(&'static str),
}

/// 会话身份 → 当前身份的判定。见 [`TargetDecision`]。
pub fn decide(session: Option<&PasteTarget>, current: Option<&PasteTarget>) -> TargetDecision {
    match session {
        None => TargetDecision::Unavailable,
        Some(target) => match current {
            // 起点拿到了身份、现在却查不到：身份未知按"已变化"处理
            //（fail closed），宁可少贴一段也不贴错窗口。
            None => TargetDecision::Block("could not re-identify the foreground window"),
            Some(now) if now.key == target.key => TargetDecision::Allow,
            Some(_) => {
                TargetDecision::Block("the foreground window changed since the recording started")
            }
        },
    }
}

/// 一次增量粘贴会话的目标守卫。会话开始时构造（此时记录起点身份），
/// 每段粘贴前调用 [`TargetGuard::allow_paste`]；一旦 blocked，本会话
/// 永远返回 false（不得恢复），粘贴方保留文本走最终一次性粘贴/积压
/// 报告，绝不把文字贴进新焦点。
pub struct TargetGuard {
    target: Option<PasteTarget>,
    capture: CaptureFn,
    blocked: bool,
    block_reason: Option<String>,
}

impl TargetGuard {
    /// 生产构造：起点身份用平台 [`capture`] 记录。
    pub fn new() -> Self {
        Self::with_capture(Box::new(capture))
    }

    /// 注入探测函数的构造（测试用；探测在构造时执行一次以记录起点）。
    pub fn with_capture(capture: CaptureFn) -> Self {
        let target = capture();
        Self {
            target,
            capture,
            blocked: false,
            block_reason: None,
        }
    }

    /// 每段粘贴前调用：true = 允许粘贴这一段。blocked 后恒为 false。
    pub fn allow_paste(&mut self) -> bool {
        if self.blocked {
            return false;
        }
        let current = (self.capture)();
        match decide(self.target.as_ref(), current.as_ref()) {
            TargetDecision::Allow => true,
            // 静默 best-effort：环境取不到身份时直接放行，不告警不记日志。
            TargetDecision::Unavailable => true,
            TargetDecision::Block(reason) => {
                let detail = match (self.target.as_ref(), current.as_ref()) {
                    (Some(from), Some(to)) => {
                        format!("{reason}: {} -> {}", from.label(), to.label())
                    }
                    (Some(from), None) => format!("{reason} (was {})", from.label()),
                    _ => reason.to_string(),
                };
                self.block(&detail);
                false
            }
        }
    }

    /// 永久阻断本会话（sticky：之后 [`Self::allow_paste`] 恒 false）。
    /// 原因只记一次 **debug** 日志（策略要求：不 warn）。
    pub fn block(&mut self, reason: &str) {
        if !self.blocked {
            self.blocked = true;
            self.block_reason = Some(reason.to_string());
            debug!(
                "Incremental paste blocked for this recording: {reason}; \
                 no further segments will be pasted (their text is kept)"
            );
        }
    }

    /// 会话是否已被阻断。
    pub fn blocked(&self) -> bool {
        self.blocked
    }

    /// 首次阻断的原因（诊断复用；未阻断为 `None`）。
    pub fn block_reason(&self) -> Option<&str> {
        self.block_reason.as_deref()
    }
}

impl Default for TargetGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn target(key: &str) -> PasteTarget {
        PasteTarget::new(key, &format!("label of {key}"))
    }

    #[test]
    fn same_identity_allows() {
        let a = target("pid:1");
        assert_eq!(decide(Some(&a), Some(&a)), TargetDecision::Allow);
        // 同 key 不同 label（例如窗口句柄变了但进程没变）仍放行。
        let a2 = PasteTarget::new("pid:1", "another label");
        assert_eq!(decide(Some(&a), Some(&a2)), TargetDecision::Allow);
    }

    #[test]
    fn changed_identity_blocks() {
        assert_eq!(
            decide(Some(&target("pid:1")), Some(&target("pid:2"))),
            TargetDecision::Block("the foreground window changed since the recording started")
        );
    }

    #[test]
    fn session_without_identity_is_best_effort() {
        assert_eq!(
            decide(None, Some(&target("pid:1"))),
            TargetDecision::Unavailable
        );
        assert_eq!(decide(None, None), TargetDecision::Unavailable);
    }

    #[test]
    fn untrackable_present_is_fail_closed() {
        // 起点有身份、现在查不到：按"已变化"拒绝。
        assert_eq!(
            decide(Some(&target("pid:1")), None),
            TargetDecision::Block("could not re-identify the foreground window")
        );
    }

    /// 计数并按脚本回放身份的探测函数。第 1 次回放是构造
    /// [`TargetGuard::with_capture`] 时的起点探测，之后的每次回放对应
    /// 一次 `allow_paste` 校验；超出脚本后停留在最后一项。
    fn scripted(steps: Vec<Option<&str>>) -> (CaptureFn, Arc<Mutex<usize>>) {
        let steps: Vec<Option<String>> = steps
            .into_iter()
            .map(|s| s.map(|k| k.to_string()))
            .collect();
        let calls = Arc::new(Mutex::new(0usize));
        let calls_for_fn = Arc::clone(&calls);
        let capture = Box::new(move || {
            let mut called = calls_for_fn.lock().unwrap();
            let step = (*called).min(steps.len().saturating_sub(1));
            *called += 1;
            steps.get(step).cloned().flatten().map(|k| target(&k))
        });
        (capture, calls)
    }

    #[test]
    fn guard_allows_while_identity_holds() {
        let (capture, calls) = scripted(vec![Some("pid:1"), Some("pid:1"), Some("pid:1")]);
        let mut guard = TargetGuard::with_capture(capture);
        assert!(guard.allow_paste());
        assert!(guard.allow_paste());
        assert!(!guard.blocked());
        assert_eq!(*calls.lock().unwrap(), 3);
    }

    #[test]
    fn guard_blocks_on_change_and_never_recovers() {
        // 起点探测 pid:1；校验 1 仍 pid:1（放行），校验 2 换成 pid:2。
        let (capture, _) = scripted(vec![Some("pid:1"), Some("pid:1"), Some("pid:2")]);
        let mut guard = TargetGuard::with_capture(capture);
        assert!(guard.allow_paste());
        assert!(!guard.allow_paste(), "身份变化必须拒绝");
        assert!(guard.blocked());
        assert!(guard.block_reason().is_some());
        // blocked 是 sticky：即使身份"变回来"也不得恢复。
        assert!(!guard.allow_paste());
        assert!(!guard.allow_paste());
    }

    #[test]
    fn manual_block_is_sticky() {
        let (capture, calls) = scripted(vec![Some("pid:1")]);
        let mut guard = TargetGuard::with_capture(capture);
        guard.block("test reason");
        assert!(!guard.allow_paste(), "blocked 后不得恢复");
        assert_eq!(guard.block_reason(), Some("test reason"));
        // 探测只在构造时被调用过一次；allow_paste 在 blocked 时短路，
        // 不再调用平台探测。
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn unavailable_environment_allows_silently() {
        // capture() 恒为 None：静默 best-effort——起点与每次校验都取不到
        // 身份，持续放行、不阻断、不产生任何日志（Unavailable 分支无日志
        // 语句；探测调用 = 构造 1 次 + 每次校验 1 次）。
        let (capture, calls) = scripted(vec![None]);
        let mut guard = TargetGuard::with_capture(capture);
        assert!(guard.allow_paste());
        assert!(guard.allow_paste());
        assert!(guard.allow_paste());
        assert!(!guard.blocked(), "不可用 ≠ 阻断");
        assert_eq!(*calls.lock().unwrap(), 4);
    }

    #[test]
    fn identity_lost_after_start_blocks() {
        // 起点探测 pid:1；校验 1 仍 pid:1（放行），校验 2 查不到身份。
        let (capture, _) = scripted(vec![Some("pid:1"), Some("pid:1"), None]);
        let mut guard = TargetGuard::with_capture(capture);
        assert!(guard.allow_paste());
        assert!(!guard.allow_paste(), "起点有身份而当前查不到：fail closed");
        assert!(guard.blocked());
    }

    /// App 标识清洗：控制字符/尖括号剔除、空白折叠、截断到上限、空则无效。
    #[test]
    fn app_identity_sanitized() {
        assert_eq!(sanitize_app_identity("WeChat").as_deref(), Some("WeChat"));
        assert_eq!(
            sanitize_app_identity("MS  \t Word").as_deref(),
            Some("MS Word")
        );
        assert_eq!(sanitize_app_identity("<bad>").as_deref(), Some("bad"));
        assert_eq!(
            sanitize_app_identity("第一行\n第二行").as_deref(),
            Some("第一行 第二行")
        );
        assert!(sanitize_app_identity("  \n\t ").is_none());
        let long: String = std::iter::repeat('a').take(80).collect();
        let cut = sanitize_app_identity(&long).unwrap();
        assert_eq!(cut.chars().count(), 64);
    }

    /// 可执行路径取文件名并去掉 .exe（大小写不敏感）。
    #[test]
    fn exe_file_name_extracts_leaf() {
        assert_eq!(
            exe_file_name(r"C:\Program Files\WeChat\WeChat.exe").as_deref(),
            Some("WeChat")
        );
        assert_eq!(exe_file_name("/usr/bin/code").as_deref(), Some("code"));
        assert_eq!(exe_file_name("chrome.EXE").as_deref(), Some("chrome"));
        assert!(exe_file_name("").is_none());
    }
}
