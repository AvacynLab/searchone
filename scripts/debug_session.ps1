# Debug session helper: launches backend, optional worker, and tails a job in separate PowerShell windows.
# Usage: .\scripts\debug_session.ps1 -JobId 1 -StartWorker -Mock
param(
    [int]$JobId = 1,
    [switch]$StartWorker = $false,
    [switch]$Mock = $false
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root\..\

Write-Host "Starting backend (uvicorn) in a new window..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn backend.app.api.main:app --reload --port 8000"

if($StartWorker){
    Write-Host "Starting worker in a new window..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "$env:REDIS_URL='redis://localhost:2002'; python ./backend/app/cli/debug_cli.py worker"
}

Start-Sleep -Seconds 2

Write-Host "Tailing job $JobId"
if($Mock){
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python ./backend/app/cli/debug_cli.py tail-job $JobId --interval 1"
}else{
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python ./backend/app/cli/debug_cli.py tail-job $JobId --interval 1"
}

Pop-Location
