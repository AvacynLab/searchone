# Start worker (PowerShell)
cd ..\backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# ensure REDIS_URL env var set
if (-not $env:REDIS_URL) { $env:REDIS_URL = 'redis://localhost:2002' }
python worker.py
