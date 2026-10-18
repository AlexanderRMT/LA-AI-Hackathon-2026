# Pulls the latest camera frame from the QNX Pi and opens it.
# Usage:  .\pull_frame.ps1                 (one frame)
#         .\pull_frame.ps1 -Loop           (refresh every 2s; reopen the file to see updates)
param(
    [string]$PiHost = "qnxuser@qnxpi.local",
    [string]$Remote = "/tmp/frame.bmp",
    [switch]$Loop
)

$local = Join-Path $PSScriptRoot "frame.bmp"
$opened = $false
do {
    scp -q "${PiHost}:${Remote}" $local
    if ($LASTEXITCODE -ne 0) { Write-Host "scp failed - is frame_grab running on the Pi?"; exit 1 }
    if (-not $opened) { Invoke-Item $local; $opened = $true }
    if ($Loop) { Start-Sleep -Seconds 2 }
} while ($Loop)
