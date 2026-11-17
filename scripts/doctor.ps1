$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $project 'backend'
$python = if ($env:TIMBREFORGE_PYTHON) { $env:TIMBREFORGE_PYTHON } else { Join-Path $project '.venv-api/Scripts/python.exe' }
if (!(Test-Path -LiteralPath $python) -and !(Get-Command $python -ErrorAction SilentlyContinue)) {
    throw '请先运行 scripts/setup.ps1 安装 API 环境。'
}
& $python -B -c 'import json; from timbreforge.environment import doctor; from timbreforge.settings import Settings; print(json.dumps(doctor(Settings.from_environment()), indent=2, ensure_ascii=True))'
exit $LASTEXITCODE
