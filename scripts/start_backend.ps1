# Start backend (PowerShell)
cd ..\backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 2001
