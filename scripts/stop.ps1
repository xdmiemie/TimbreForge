param([ValidateRange(1024,65535)][int]$Port = 7866)
$ErrorActionPreference = 'Stop'
$result = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/system/stop"
if ($result.pid) {
    $serverProcess = Get-Process -Id $result.pid -ErrorAction SilentlyContinue
    if ($serverProcess -and !$serverProcess.WaitForExit(12000)) {
        throw '服务仍在收尾，请稍后再启动。'
    }
}
$result
