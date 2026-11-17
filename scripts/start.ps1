param(
    [ValidateRange(1024,65535)][int]$Port = 7866,
    [switch]$OpenBrowser
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = if ($env:TIMBREFORGE_PYTHON) { $env:TIMBREFORGE_PYTHON } else { Join-Path $project '.venv-api/Scripts/python.exe' }
if (!(Get-Command -Name $python -CommandType Application -ErrorAction SilentlyContinue)) { throw '请先运行 scripts/setup.ps1 安装 API 环境。' }
if (! (Test-Path -LiteralPath (Join-Path $project 'frontend/dist/index.html'))) { throw '请先运行 scripts/setup.ps1 构建 WebUI。' }
$env:PYTHONPATH = Join-Path $project 'backend'
$serveArguments = @('--port', $Port)
if ($OpenBrowser) { $serveArguments += '--open-browser' }
& $python -B -m timbreforge.serve @serveArguments
exit $LASTEXITCODE
