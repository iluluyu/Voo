param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [string]$BuildRoot = 'C:\Users\Administrator\voo\Voo',
    [string]$ArtifactRoot = 'C:\Users\Administrator\voo',
    [switch]$CleanNative
)

$ErrorActionPreference = 'Stop'

function Stage([string]$Message) {
    Write-Host "VOO_BUILD_STAGE $Message"
}

function Check-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

try {
    if (-not (Test-Path (Join-Path $SourceRoot 'src-tauri\Cargo.toml'))) {
        throw "SourceRoot is not a Voo checkout: $SourceRoot"
    }
    if (-not (Test-Path (Join-Path $BuildRoot 'src-tauri\Cargo.toml'))) {
        throw "BuildRoot is not the existing Voo Windows checkout: $BuildRoot"
    }
    if ($SourceRoot.TrimEnd('\') -eq $BuildRoot.TrimEnd('\')) {
        throw 'SourceRoot and BuildRoot must differ; run this from WSL against the Windows copy.'
    }
    if ($BuildRoot.StartsWith('\\')) {
        throw 'BuildRoot must be on a local Windows drive (not a UNC path).'
    }

    Stage 'Mirror current source, including uncommitted work, into the Windows checkout'
    # /MIR makes the snapshot match the WSL source. Excluded directories keep
    # the existing native caches (target, node_modules) and Linux-only/generated
    # artifacts untouched so the rebuild stays incremental.
    & robocopy.exe $SourceRoot $BuildRoot /MIR /XJ /R:1 /W:1 /NFL /NDL /NJH /NJS `
        /XD .git node_modules target transcribe-libs dist .pi .claude .direnv `
        /XF '*.log' '.env' '.env.*' 'tauri.local.test.json'
    if ($LASTEXITCODE -ge 8) {
        throw "Source sync failed: robocopy exit $LASTEXITCODE"
    }

    Stage 'Initialize native Visual Studio x64 build environment'
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vsRoot = (& $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1)
    if (-not $vsRoot) { throw 'Visual Studio C++ x64 build tools are required.' }
    & (Join-Path $vsRoot 'Common7\Tools\Launch-VsDevShell.ps1') `
        -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

    foreach ($tool in @('cargo', 'bun', 'cmake', 'cl', 'glslc', 'ninja')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "Missing native Windows build tool: $tool"
        }
    }
    if (-not $env:VULKAN_SDK) { throw 'VULKAN_SDK is not configured.' }

    # transcribe-cpp-sys lets the `cmake` crate pick the generator; with VS 18
    # installed it derived "Visual Studio 18 2026", which this CMake cannot
    # create. CMAKE_GENERATOR overrides that choice (cmake-0.1.57 build()), and
    # Ninja ships inside the VS Build Tools and is what Windows ARM CI uses.
    $env:CMAKE_GENERATOR = 'Ninja'

    # Native sources are UTF-8, but MSVC decodes them with the system codepage
    # (GBK here). A multi-byte comment sequence can decode to a trailing 0x5C
    # backslash, which continues the line and breaks the following declarations
    # (reproduced: canary/model.cpp "sp_space_id undeclared"). /utf-8 fixes it;
    # CL is honoured by every cl invocation in the CMake build.
    if ($env:CL -notmatch '/utf-8') {
        $env:CL = ((@('/utf-8') + @($env:CL)) | Where-Object { $_ }) -join ' '
    }

    if ($CleanNative) {
        Stage 'Purge stale native build trees (they were compiled without /utf-8)'
        $tcs = Join-Path $env:LOCALAPPDATA 'tcs'
        if (Test-Path -LiteralPath $tcs) { Remove-Item -LiteralPath $tcs -Recurse -Force }
        Push-Location -LiteralPath (Join-Path $BuildRoot 'src-tauri')
        try {
            & cargo clean -p transcribe-cpp-sys
            Check-Exit 'cargo clean -p transcribe-cpp-sys'
        } finally {
            Pop-Location
        }
    }

    # Bundle runtime DLLs from the same VS installation, as Windows CI does.
    $versionFile = Join-Path $vsRoot 'VC\Auxiliary\Build\Microsoft.VCRedistVersion.default.txt'
    $redistVersion = (Get-Content -LiteralPath $versionFile -Raw).Trim()
    $archDir = Join-Path $vsRoot "VC\Redist\MSVC\$redistVersion\x64"
    $crt = Get-ChildItem -LiteralPath $archDir -Directory -Filter 'Microsoft.VC*.CRT' |
        Sort-Object Name | Select-Object -Last 1
    $omp = Get-ChildItem -LiteralPath $archDir -Directory -Filter 'Microsoft.VC*.OpenMP' |
        Sort-Object Name | Select-Object -Last 1
    if (-not $crt -or -not $omp) { throw 'VS CRT/OpenMP redistributable directories are missing.' }
    foreach ($required in @((Join-Path $crt.FullName 'msvcp140.dll'), (Join-Path $omp.FullName 'vcomp140.dll'))) {
        if (-not (Test-Path -LiteralPath $required)) { throw "Missing runtime DLL: $required" }
    }
    $env:HANDY_VC_REDIST_DIRS = "$($crt.FullName);$($omp.FullName)"
    $env:CARGO_TARGET_DIR = Join-Path $BuildRoot 'src-tauri\target'
    $env:CARGO_TERM_COLOR = 'never'

    Set-Location -LiteralPath $BuildRoot
    Stage 'Install native Windows frontend dependencies'
    & bun install --frozen-lockfile
    Check-Exit 'bun install'

    # A local test build must not invoke the upstream Azure signing account (the
    # previous run failed here with "program not found"). This override exists
    # only in this disposable Windows checkout; release config is untouched.
    '{"bundle":{"windows":{"signCommand":null}}}' |
        Set-Content -LiteralPath (Join-Path $BuildRoot 'src-tauri\tauri.local.test.json') -Encoding ASCII

    Stage 'Compile Windows x64 release and package unsigned NSIS installer'
    & bun run tauri build --target x86_64-pc-windows-msvc --bundles nsis `
        --config 'src-tauri\tauri.local.test.json' --ci
    Check-Exit 'Tauri Windows build'

    $bundleDir = Join-Path $env:CARGO_TARGET_DIR 'x86_64-pc-windows-msvc\release\bundle\nsis'
    $installers = @(Get-ChildItem -LiteralPath $bundleDir -Filter '*.exe' -File)
    if ($installers.Count -eq 0) { throw "No installer generated under $bundleDir" }

    New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null
    foreach ($installer in $installers) {
        $target = Join-Path $ArtifactRoot $installer.Name
        Copy-Item -LiteralPath $installer.FullName -Destination $target -Force
        $hash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        Write-Host "VOO_BUILD_ARTIFACT $target bytes=$($installer.Length) sha256=$hash"
    }
    Write-Host 'VOO_BUILD_SUCCESS Windows x64 unsigned test installer is ready; not installed or launched.'
    exit 0
} catch {
    Write-Host "VOO_BUILD_FAILED $($_.Exception.Message)"
    exit 1
}
