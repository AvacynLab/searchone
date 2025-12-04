# SearchOne - Agent de recherche approfondie (MVP)

Prototype minimal d'agent multi-roles pour la recherche documentaire, avec API FastAPI et UI React.

## Structure
- backend/app/api : FastAPI (routes ingestion/recherche/jobs).
- backend/app/services|data|core|workflows|cli : logique metier (ingestion, LLM, vecteur/DB, orchestrateur, debug CLI).
- frontend/src : UI React/Vite (page principale pages/App.jsx, client API api/client.js, composants dans components/, utilitaires dans utils/).
- scripts/ : helpers PowerShell pour lancer backend+frontend ou sessions de debug.

## Backend
Pre-requis : Python 3.10+, backend/.env (copier .env.example).

```powershell
cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 2001
```

Endpoints cles :
- POST /ingest (form-data file, title?) -> PDF -> texte/chunks/embeddings FAISS.
- GET /search?q=...&top_k=5 -> recherche vectorielle.
- GET /report?q=...&top_k=10 -> rapport markdown rapide.
- Jobs long-terme : POST /jobs/start, GET /jobs/{id}, timeline/overview/diagnostic, rename/delete/retry, prompts system/variant.

Debug CLI (local) :
- python -m app.cli.debug_cli run-job --query "..." --name demo --iterations 3 --mock
- python -m app.cli.debug_cli tail-job 1 --interval 1
- python -m app.cli.debug_cli show-timeline 1
- python -m app.cli.debug_cli list-jobs
- python -m app.cli.debug_cli worker (REDIS_URL requis)

Tests : pytest backend/tests.

Deploiement (resume) :
- Docker Compose local : services backend API, worker, redis, searxng (voir docker-compose.yml).
- Prod (suggestion) : deployer API et worker separes (K8s deployments + HPA), Redis manage, volume partage pour backend/data (FAISS/DB/snapshots). Config via ConfigMap/Secrets (OPENROUTER_API_KEY, modeles, budgets).

## Frontend
Pre-requis : Node.js 18+.

```powershell
cd frontend
npm install
npm run dev -- --host --port 2000
```

La page principale consomme l'API backend (/api) pour lancer/inspecter les jobs, afficher timelines, evidences, dashboards et gerer prompts. Sentry cote client optionnel via SEARCHONE_SENTRY_DSN.

## Roadmap courte (agents/features)
- Ingestion web/crawler et formats additionnels.
- Routage OpenRouter avec fallbacks, metriques couts/tokens.
- Agents specialises (explorateur, curateur, analyste, hypotheses, experimentateur, coordinateur, redacteur) + mecanisme de debat/vote.
- Scheduler long-terme avec checkpoints et arret auto (stagnation/budget).
- Observabilite : traces, metriques (tokens/run, iterations, couverture sources), gardes-fous domaines sensibles.

# searchone
