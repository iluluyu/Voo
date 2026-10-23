param(
    [string]$Checkout = 'C:\Users\Administrator\voo\Voo',
    [string]$LogPath = 'C:\Users\Administrator\voo\test-report.log',
    [ValidateSet('debug', 'release')][string]$Profile = 'release'
)

$ErrorActionPreference = 'Stop'
$script:results = [System.Collections.Generic.List[object]]::new()

function Run-Step {
    param([string]$Name, [scriptblock]$Action)
    Write-Host ""
    Write-Host "VOO_TEST_STEP $Name"
    $start = Get-Date
    $code = 1
    try {
        & $Action
        $code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    } catch {
        Write-Host "VOO_TEST_ERROR $Name $($_.Exception.Message)"
        $code = 1
    }
    $seconds = [int]((Get-Date) - $start).TotalSeconds
    $state = if ($code -eq 0) { 'PASS' } else { 'FAIL' }
    $script:results.Add([pscustomobject]@{ Step = $Name; Result = $state; Seconds = $seconds; Exit = $code })
    Write-Host "VOO_TEST_RESULT $Name $state exit=$code seconds=$seconds"
    return ($code -eq 0)
}

function New-Mono48kWav {
    param([string]$Path, [int]$Samples = 24000)
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Force }
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew)
    $bw = New-Object System.IO.BinaryWriter($fs)
    try {
        $dataBytes = $Samples * 2
        $bw.Write([Text.Encoding]::ASCII.GetBytes('RIFF'))
        $bw.Write([int](36 + $dataBytes))
        $bw.Write([Text.Encoding]::ASCII.GetBytes('WAVE'))
        $bw.Write([Text.Encoding]::ASCII.GetBytes('fmt '))
        $bw.Write([int]16)
        $bw.Write([int16]1)      # PCM
        $bw.Write([int16]1)      # mono
        $bw.Write([int]48000)
        $bw.Write([int](48000 * 2))
        $bw.Write([int16]2)
        $bw.Write([int16]16)
        $bw.Write([Text.Encoding]::ASCII.GetBytes('data'))
        $bw.Write([int]$dataBytes)
        $rnd = New-Object System.Random(7)
        for ($i = 0; $i -lt $Samples; $i++) {
            $t = $i / 48000.0
            $v = 0.15 * [Math]::Sin(2 * [Math]::PI * 220 * $t) + 0.05 * ($rnd.NextDouble() * 2 - 1)
            $bw.Write([int16]([Math]::Round($v * 32767)))
        }
    } finally {
        $bw.Dispose(); $fs.Dispose()
    }
}

# Minimal RIFF walker: returns the fmt fields and the data chunk length.
function Read-WavInfo {
    param([string]$Path)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ([Text.Encoding]::ASCII.GetString($bytes, 0, 4) -ne 'RIFF' -or
        [Text.Encoding]::ASCII.GetString($bytes, 8, 4) -ne 'WAVE') {
        throw "not a RIFF/WAVE file: $Path"
    }
    $pos = 12
    $info = @{}
    while ($pos + 8 -le $bytes.Length) {
        $id = [Text.Encoding]::ASCII.GetString($bytes, $pos, 4)
        $size = [BitConverter]::ToInt32($bytes, $pos + 4)
        $body = $pos + 8
        switch ($id) {
            'fmt ' {
                $info.Channels = [BitConverter]::ToInt16($bytes, $body + 2)
                $info.SampleRate = [BitConverter]::ToInt32($bytes, $body + 4)
                $info.Bits = [BitConverter]::ToInt16($bytes, $body + 14)
                $info.FormatTag = [BitConverter]::ToInt16($bytes, $body)
                # hound emits WAVE_FORMAT_EXTENSIBLE (0xFFFE) for float data;
                # the real format is the first field of the subformat GUID.
                if ($size -ge 40 -and $info.FormatTag -eq 0xFFFE) {
                    $info.SubFormat = [BitConverter]::ToInt16($bytes, $body + 24)
                }
            }
            'data' { $info.DataBytes = $size }
        }
        $pos = $body + $size + ($size % 2)
        if ($info.ContainsKey('DataBytes')) { break }
    }
    return [pscustomobject]$info
}

try {
    if (-not (Test-Path -LiteralPath (Join-Path $Checkout 'src-tauri\Cargo.toml'))) {
        throw "Checkout is not a Voo checkout: $Checkout"
    }
    Start-Transcript -Path $LogPath -Force | Out-Null

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vsRoot = (& $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1)
    if (-not $vsRoot) { throw 'Visual Studio C++ x64 build tools are required.' }
    & (Join-Path $vsRoot 'Common7\Tools\Launch-VsDevShell.ps1') `
        -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

    # The native build needs the same generator and source-charset fixes as
    # scripts/build-windows-local.ps1, or cargo's own native rebuild fails.
    $env:CMAKE_GENERATOR = 'Ninja'
    if ($env:CL -notmatch '/utf-8') {
        $env:CL = ((@('/utf-8') + @($env:CL)) | Where-Object { $_ }) -join ' '
    }

    Set-Location -LiteralPath $Checkout
    Write-Host "Voo Windows test suite: $Checkout (profile: $Profile)"
    # The native transcribe-cpp build is the expensive part; cargo's dev profile
    # would trigger a second full CMake build tree, so default to release and
    # reuse the artifacts the installer build already produced.
    $useRelease = $Profile -eq 'release'

    Run-Step 'bun install' { & bun install --frozen-lockfile } | Out-Null
    Run-Step 'frontend build' { & bun run build } | Out-Null
    Run-Step 'frontend lint' { & bun run lint } | Out-Null
    Run-Step 'keyboard unit test' { & bun run test:keyboard } | Out-Null
    Run-Step 'lexicon source test' { & bun run test:lexicon } | Out-Null
    Run-Step 'chinese evaluator test' { & bun run test:chinese-eval } | Out-Null
    Run-Step 'translations check' { & bun run check:translations } | Out-Null
    Run-Step 'model language check' { & bun run check:model-languages } | Out-Null
    Run-Step 'catalog check' { & bun run check:catalog } | Out-Null

    Push-Location -LiteralPath (Join-Path $Checkout 'src-tauri')
    try {
        Run-Step 'rustfmt check' { & cargo fmt -- --check } | Out-Null
        Run-Step 'examples compile' {
            if ($useRelease) { & cargo check --examples --release } else { & cargo check --examples }
        } | Out-Null
        Run-Step 'rust unit tests' {
            if ($useRelease) { & cargo test --lib --release } else { & cargo test --lib }
        } | Out-Null

        $smokeDir = Join-Path $env:TEMP ('voo-denoise-' + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $smokeDir | Out-Null
        try {
            $wavIn = Join-Path $smokeDir 'input.wav'
            $wavOut = Join-Path $smokeDir 'output.wav'
            New-Mono48kWav -Path $wavIn -Samples 24000
            $wavInHash = (Get-FileHash -LiteralPath $wavIn -Algorithm SHA256).Hash

            $ok = Run-Step 'denoise offline smoke' {
                if ($useRelease) {
                    & cargo run --quiet --release --example denoise_offline -- $wavIn $wavOut
                } else {
                    & cargo run --quiet --example denoise_offline -- $wavIn $wavOut
                }
            }
            if ($ok) {
                Run-Step 'denoise output contract' {
                    $src = Read-WavInfo -Path $wavIn
                    $dst = Read-WavInfo -Path $wavOut
                    if ($src.SampleRate -ne 48000 -or $src.Channels -ne 1) { throw 'fixture is not 48k mono' }
                    if ($dst.SampleRate -ne 48000 -or $dst.Channels -ne 1) { throw 'output is not 48k mono' }
                    if ($dst.Bits -ne 32) { throw 'output must be 32-bit PCM' }
                    $tag = if ($dst.FormatTag -eq 0xFFFE) { $dst.SubFormat } else { $dst.FormatTag }
                    if ($tag -ne 3) { throw "output must be IEEE float PCM (tag=$tag)" }
                    if (($dst.DataBytes / 4) -ne ($src.DataBytes / 2)) {
                        throw "sample count mismatch: in=$($src.DataBytes / 2) out=$($dst.DataBytes / 4)"
                    }
                } | Out-Null

                Run-Step 'denoise refuses to overwrite output' {
                    if ($useRelease) {
                        & cargo run --quiet --release --example denoise_offline -- $wavIn $wavOut
                    } else {
                        & cargo run --quiet --example denoise_offline -- $wavIn $wavOut
                    }
                    if ($LASTEXITCODE -eq 0) { throw 'existing output was overwritten' }
                    $global:LASTEXITCODE = 0
                } | Out-Null

                Run-Step 'denoise refuses in-place conversion' {
                    if ($useRelease) {
                        & cargo run --quiet --release --example denoise_offline -- $wavIn $wavIn
                    } else {
                        & cargo run --quiet --example denoise_offline -- $wavIn $wavIn
                    }
                    if ($LASTEXITCODE -eq 0) { throw 'input was accepted as output' }
                    if ((Get-FileHash -LiteralPath $wavIn -Algorithm SHA256).Hash -ne $wavInHash) {
                        throw 'input file was modified'
                    }
                    $global:LASTEXITCODE = 0
                } | Out-Null
            }
        } finally {
            Remove-Item -LiteralPath $smokeDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    } finally {
        Pop-Location
    }

    Write-Host ""
    $script:results | Format-Table -AutoSize | Out-String | Write-Host
    $failed = @($script:results | Where-Object { $_.Result -ne 'PASS' })
    Write-Host "VOO_TEST_SUMMARY pass=$(@($script:results | Where-Object Result -eq 'PASS').Count) fail=$($failed.Count) log=$LogPath"
    if ($failed.Count -gt 0) {
        $names = ($failed | ForEach-Object { $_.Step }) -join ', '
        Write-Host "VOO_TEST_FAILED_STEPS $names"
        exit 1
    }
    Write-Host 'VOO_TEST_ALL_PASS'
    exit 0
} catch {
    Write-Host "VOO_TEST_FATAL $($_.Exception.Message)"
    exit 2
} finally {
    try { Stop-Transcript | Out-Null } catch {}
}
