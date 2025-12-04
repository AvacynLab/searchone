# AGENTS Progress Log

## TODO (delta)

### 1.1 Outils de plots scientifiques
- [x] Créer le module `backend/app/services/plot_tools.py` avec `PlotArtifact` et HTTP-ready `generate_plot`.
-  - [x] `generate_plot(data, spec)` accepte les séries, gère la mise en forme (line/scatter/bar/hist), créé un PNG dans `data/plots/<job_id>/` et peut produire des vecteurs.
-  - [x] Ajouter l'option vectorielle (SVG/PDF) et les intégrer dans les rapports (à venir).
- [x] Déclarer et exposer le tool `plot_tool` dans `workflows/agents.py`.
- [x] Adapter `services/reporting.py` pour référencer les figures et les inclure dans le résumé structuré.

### 1.2 Outils de graphes de connaissances
- [x] Créer `backend/app/services/graph_tools.py` et exposer les outils CLI/LLM associés.
- [x] Intégrer les stats graphes dans `ResearchScore` et `reporting.build_structured_summary`.

### 2. Scénarios déclaratifs
- [x] Définir `configs/research_scenarios.yaml` (+ loader) et intégrer avec l'orchestrateur.
- [x] Implémenter les scénarios `quick_literature_review`, `deep_theory_exploration`, `multi_domain_synthesis`.

### 3. Méta-contrôle & re-planification avancée
- [x] Enrichir `ConvergenceController` et ajouter les actions dynamiques du coordinateur.
- [x] Adapter `run_agents_job` pour gérer stagnation via les nouvelles actions.

### 4. Écriture scientifique “article complet” (IMRaD / LaTeX-ready)
- [x] Créer `backend/app/workflows/writing_pipeline.py`, générer outlines / sections, composer les drafts et critiques.
- [x] Étendre `services/reporting.py` avec `build_article_report`, `export_markdown` et `export_latex`, et exporter aussi `.tex`.
- [x] Affiner `Redacteur`/`Critic` avec prompts doctorants/IMRaD + support de variantes “style journal” (`prompts/style_*.json`).

### 5. Observabilité & interface de monitoring
- [x] Ajouter `GET /jobs/{job_id}/metrics`, `/timeline` typé via RunContext et `/decisions` lisant `decisions.log`.
- [x] Créer les vues “Job detail” et “Runs overview” dans le front avec navigation dédiée, métriques/plot, décisions et filtres par statut.

### 6. Scheduler & long-running jobs
- [x] Renforcer `ResearchScheduler` avec formats cron/iso/interval, persistance SQLite (`research_schedule`), retry/backoff, cache JSON et purge des snapshots.
- [x] Compléter `resume_from_snapshot` (versions/migrations, logs, fallback glob) et exposer `POST /jobs/{job_id}/resume`.
- [x] Ajouter `POST /jobs/{job_id}/stop` + enrichir la création de schedule (`spec`, `cron`, `iso`) tout en utilisant les stop flags existants.

### 7. Intégration SearxNG “production”
- [x] Distinguer erreurs réseau / engine / vides dans `run_web_search`, ajouter circuit breaker par moteur (`ENGINE_FAILURE_STATE` + seuil/cooldown) et arracher les logs en conséquence.
- [x] Ajouter cache applicatif `web_cache` (table SQLModel, TTL configurable) + outil `web_cache_lookup` pour informer les agents.
- [x] Permettre `schedule_job` de recevoir specs (cron/iso/interval) tout en utilisant le circuit breaker et le cache pour éviter les requêtes redondantes.

### 8. Tests complémentaires
- [x] Ajouter tests unitaires pour plot_tools, graph_tools, scenarios, writing_pipeline et le scheduler/cache (web_cache + ResearchScheduler helpers).
- [x] Tests E2E (quick_literature_review + résistance SearxNG) – couverture via orchestrator stub + web_search_mock.

### 9. Finitions & polish
- [x] Documenter les scénarios, l'usage API/Orchestrator et les limites dans README_RESEARCH.md.
- [x] Exposer les budgets tokens/web et le niveau de logs via SEARCHONE_JOB_TOKEN_BUDGET, SEARCHONE_WEB_QUERY_BUDGET et SEARCHONE_LOG_LEVEL.

## Historique des actions
- `2025-12-04T06:53:32Z` : Ajout des tests E2E `quick_literature_review` (orchestrator stub) et résistance SearxNG (mock httpx + cache cleanup).
- `2025-12-04T07:10:00Z` : Ajout des tests unitaires pour plot/graph/scenarios/writing pipeline/scheduler+cache, calibration de la limite web dans `run_agents_job`, et documentation des scénarios + budgets (README_RESEARCH + config env).
- `2025-12-04T06:45:00Z` : Robustesse SearxNG (circuit breaker, logging, cache web + outil `web_cache_lookup`) et schedule_specs via `/jobs/schedule`.
- `2025-12-04T06:25:00Z` : Extension de `ResearchScheduler` (cron/iso/interval, SQLite), endpoints `/jobs/{job_id}/stop`/`/resume` et spec de schedule plus robuste.
- `2025-12-04T06:14:44Z` : Extension des endpoints métriques/timeline/décisions et création des vues “Job detail” / “Runs overview” dans le front.
- `2025-12-04T06:33:34Z` : Ajout du module `plot_tools.py` pour générer des PNG à partir de séries et mise à jour de `requirements.txt` avec `matplotlib`.
- `2025-12-04T06:33:38Z` : Création de ce fichier de suivi pour documenter la progression et les tâches en cours.
- `2025-12-04T06:37:20Z` : Exposition du tool `plot_tool` dans `workflows/agents.py`, génération de l'artefact (PNG/vectoriels), et enrichissement de `services/reporting.py` avec la section figure.
- `2025-12-04T06:42:37Z` : Ajout du module `graph_tools`, ouverture du `knowledge_graph_tool`, et propagation des stats dans `ResearchScore`/`reporting`.
- `2025-12-04T06:47:34Z` : Création du DSL de scénarios (`configs/research_scenarios.yaml`), loader `workflows/scenarios.py`, et extension de l'orchestrateur (`run_with_scenario`, phase runners, outils spécialisés).
- `2025-12-04T06:54:21Z` : Ajout de signaux enrichis dans `ConvergenceController`, nouveaux `coordinator_actions` (focus/seek/downgrade) et logique de run/job pour gérer stagnation (conclusion ou replan).
- `2025-12-04T07:03:26Z` : Pipeline d’écriture IMRaD (`writing_pipeline.py`), exports Markdown/LaTeX, et prompts Redacteur/Critic + variant journal Nature.
