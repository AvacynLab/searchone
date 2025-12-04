# Start both backend and frontend in separate PowerShell windows for local testing
# Usage: run this script from repo root (PowerShell)

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path "$repo\.." # script is in scripts folder

# Ensure infra containers (Redis + SearxNG) are up before launching UI/API locally
$composeFile = Join-Path $root 'docker-compose.yml'
if (-not (Test-Path $composeFile)) {
    throw "docker-compose.yml introuvable : $composeFile"
}

# Defaults for local dev ports
if (-not $env:REDIS_URL) { $env:REDIS_URL = 'redis://localhost:2002' }
if (-not $env:SEARXNG_URL) { $env:SEARXNG_URL = 'http://localhost:2003' }

function Wait-ContainerHealthy {
    param(
        [string]$Name,
        [int]$TimeoutSec = 60
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        $state = docker inspect -f "{{.State.Health.Status}}" $Name 2>$null
        if (-not $state) {
            $state = docker inspect -f "{{.State.Status}}" $Name 2>$null
        }
        if ($state -eq "healthy" -or $state -eq "running") { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

Write-Host "Démarrage / vérification des conteneurs Redis + SearxNG..."
docker-compose -f $composeFile up -d redis searxng | Out-Null
$redisOk = Wait-ContainerHealthy -Name 'searchone-redis' -TimeoutSec 40
$searxOk = Wait-ContainerHealthy -Name 'searchone-searxng' -TimeoutSec 60
if (-not ($redisOk -and $searxOk)) {
    Write-Warning "Impossible de vérifier l'état de Redis/SearxNG (healthcheck). Contrôlez 'docker ps' avant de continuer."
}

# Backend command: activate venv and run uvicorn
$backendDir = Join-Path $root 'backend'
$backendCmd = "cd '$backendDir'; if(Test-Path '.venv'){ .\.venv\Scripts\Activate.ps1 } else { python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt }; uvicorn app.api.main:app --reload --host 127.0.0.1 --port 2001"

# Frontend command: install and run vite dev
$frontendDir = Join-Path $root 'frontend'
$frontendCmd = "cd '$frontendDir'; if(-not (Test-Path 'node_modules')){ npm install }; npm run dev -- --host --port 2000 --strictPort"

Write-Host "Starting backend in new window..."
Start-Process -FilePath 'powershell' -ArgumentList ('-NoExit','-Command', $backendCmd) -WorkingDirectory $backendDir
Start-Sleep -Seconds 1
Write-Host "Starting frontend in new window..."
# Use WorkingDirectory to ensure npm runs in the frontend folder
Start-Process -FilePath 'powershell' -ArgumentList ('-NoExit','-Command', $frontendCmd) -WorkingDirectory $frontendDir

Write-Host "Started backend and frontend. Frontend: http://127.0.0.1:2000, Backend: http://127.0.0.1:2001, Redis: localhost:2002, SearxNG: http://localhost:2003"
