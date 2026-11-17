param(
    [string]$Python = 'python',
    [string]$RiftPython,
    [switch]$Rift,
    [switch]$Separator
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
function Invoke-Checked {
    param([string]$Command, [string[]]$Arguments)
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command exited with code $LASTEXITCODE" }
}
Push-Location $project
try {
    if (!(Test-Path -LiteralPath '.venv-api/Scripts/python.exe')) {
        Invoke-Checked $Python @('-m', 'venv', '.venv-api')
    }
    Invoke-Checked '.venv-api/Scripts/python.exe' @('-m', 'pip', 'install', '-c', 'configs/api.requirements.txt', '-e', '.')
    Push-Location frontend
    try {
        Invoke-Checked 'pnpm' @('install', '--frozen-lockfile')
        Invoke-Checked 'pnpm' @('run', 'build')
    } finally { Pop-Location }
    if ($Rift) {
        if (!$RiftPython) { throw '请使用 -RiftPython 指定 Python 3.11 或 3.12 的路径。' }
        $riftRoot = if ($env:TIMBREFORGE_RIFT) { $env:TIMBREFORGE_RIFT } else { Join-Path $project 'vendor/rift-svc' }
        if (!(Test-Path -LiteralPath (Join-Path $riftRoot 'uv.lock'))) {
            throw '请先运行 scripts/assets.ps1 准备固定 RIFT 源码。'
        }
        $env:UV_PROJECT_ENVIRONMENT = Join-Path $project '.venv-rift'
        $uv = Join-Path $project 'vendor/python-tools/bin/uv.exe'
        if (!(Test-Path -LiteralPath $uv)) { $uv = 'uv' }
        Invoke-Checked $uv @('sync', '--locked', '--no-install-project', '--project', $riftRoot, '--python', $RiftPython)
    }
    if ($Separator) {
        $separatorPython = if ($RiftPython) { $RiftPython } else { $Python }
        $separatorExe = '.venv-separator/Scripts/python.exe'
        if (!(Test-Path -LiteralPath $separatorExe)) {
            Invoke-Checked $separatorPython @('-m', 'venv', '.venv-separator')
        }
        Invoke-Checked $separatorExe @('-m', 'pip', 'install', 'torch==2.9.0+cu128', 'torchaudio==2.9.0+cu128', 'torchvision==0.24.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128')
        Invoke-Checked $separatorExe @('-m', 'pip', 'install', '-r', 'configs/separator.requirements.txt')
    }
} finally { Pop-Location }
