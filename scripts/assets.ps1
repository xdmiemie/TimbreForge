param(
    [string]$HuggingFace = 'https://huggingface.co',
    [switch]$SourceOnly
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$lock = Get-Content -Raw -LiteralPath (Join-Path $project 'configs/upstreams.lock.json') | ConvertFrom-Json
$vendor = Join-Path $project 'vendor'
$rift = if ($env:TIMBREFORGE_RIFT) { $env:TIMBREFORGE_RIFT } else { Join-Path $vendor 'rift-svc' }
$data = if ($env:TIMBREFORGE_DATA) { $env:TIMBREFORGE_DATA } else { Join-Path $project 'data' }
$assets = Join-Path $data 'assets'

function Receive-Verified {
    param([string]$Url, [string]$Destination, [string]$Sha256)
    if ((Test-Path -LiteralPath $Destination) -and (Get-FileHash -LiteralPath $Destination).Hash -eq $Sha256) {
        return
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    $part = $Destination + '.part'
    if (!(Test-Path -LiteralPath $part) -or (Get-FileHash -LiteralPath $part).Hash -ne $Sha256) {
        & curl.exe -4 -fL --connect-timeout 15 --retry 2 -C - $Url -o $part
        if ($LASTEXITCODE -ne 0) { throw "下载失败。再次运行可续传 $part" }
    }
    if ((Get-FileHash -LiteralPath $part).Hash -ne $Sha256) {
        throw "SHA256 校验失败。请检查并移走 $part 后再试。"
    }
    Move-Item -LiteralPath $part -Destination $Destination -Force
}
if (!(Test-Path -LiteralPath $rift)) {
    $archive = Join-Path $vendor 'rift-source.zip'
    Receive-Verified $lock.rift.archive_url $archive $lock.rift.archive_sha256
    $parent = Split-Path -Parent $rift
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $parent
    $extracted = Join-Path $parent ('RIFT-SVC-' + $lock.rift.commit)
    Set-Content -LiteralPath (Join-Path $extracted '.timbreforge-revision') -Value $lock.rift.commit -Encoding ascii
    Move-Item -LiteralPath $extracted -Destination $rift
}
$marker = Join-Path $rift '.timbreforge-revision'
$revision = if (Test-Path -LiteralPath $marker) { (Get-Content -LiteralPath $marker -Raw).Trim() } else { git -C $rift rev-parse HEAD }
if ($revision -ne $lock.rift.commit) { throw 'RIFT 源码版本与锁定清单不同。' }
if ($SourceOnly) { return }
$baseUrl = $lock.assets.base.url.Replace('https://huggingface.co', $HuggingFace.TrimEnd('/'))
Receive-Verified $baseUrl (Join-Path $assets $lock.assets.base.path) $lock.assets.base.sha256
foreach ($file in $lock.rift_modules.files.PSObject.Properties) {
    $url = $HuggingFace.TrimEnd('/') + '/' + $lock.rift_modules.repository + '/resolve/' + $lock.rift_modules.revision + '/' + $file.Name
    Receive-Verified $url (Join-Path (Join-Path $rift 'pretrained') $file.Name) $file.Value.sha256
}
$model = $lock.assets.separation
Receive-Verified ($model.source + $model.path) (Join-Path $assets $model.path) $model.sha256
Receive-Verified ($model.source + $model.config) (Join-Path $assets $model.config) $model.config_sha256
