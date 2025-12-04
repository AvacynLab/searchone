# README_RESEARCH

Ce document reprend les éléments spécifiques à l'orchestration scientifique avancée (scénarios déclaratifs, budgets LLM/web, observabilité métier) pour le backend `searchone-main`.

## Scénarios disponibles
Tous les scénarios sont déclarés dans `configs/research_scenarios.yaml` avec leur objectif, les rôles impliqués, les outils autorisés et les critères de sortie par phase.
- **quick_literature_review** : objectifs d'exploration rapide et de synthèse, phases Explorateur/Curator/Analyst puis Analyst/Experimenter avec plots & graphes.
- **deep_theory_exploration** : cycles Hypothèse + Experimenter, suivi d'une revue Critic axée sur cohérence et contradictions.
- **multi_domain_synthesis** : exploration parallèle de sous-domaines (groupes `domain_exploration`), puis synthèse finale avec Analyst/Redacteur/Critic.

En cas de besoin, observez la structure YAML pour ajouter d'autres scénarios ou modifier les phases (agents, outils, durées).

## Démarrage programmatique d'une recherche
- **API principale** : `POST /jobs/start` (body : `name`, `query`, `max_duration_seconds`, `max_iterations`, `max_token_budget`). On peut relancer via `POST /jobs/{job_id}/retry`.
- **Scénarios** : Pour forcer un scénario particulier, instanciez `app.workflows.orchestrator.Orchestrator` et appelez `run_with_scenario(job_id, query, scenario_name)`.
- **Observabilité** : `GET /jobs/{job_id}/metrics`, `/timeline`, `/decisions` donnent respectivement les scores, la timeline typée et les décisions enregistrées.
- **Scheduler** : `POST /jobs/schedules/run_due` déclenche les runs planifiés ; les fonctionnalités `stop`/`resume` sont exposées via les endpoints correspondants.

## Budgets & config clés
| Variable environnements | Description | Valeur par défaut |
| --- | --- | --- |
| `SEARCHONE_JOB_TOKEN_BUDGET` | Limite LLM (tokens) par job ; 0 = illimité | `0` |
| `SEARCHONE_WEB_QUERY_BUDGET` | Nombre maximal d'appels à `web_search_tool` par job | `0` (illimité) |
| `SEARCHONE_LOG_LEVEL` | Niveau global des logs backend (INFO/DEBUG/...). | `INFO` |

Ces variables sont reprises automatiquement par la configuration (`app.core.config`) et appliquées au démarrage (`logging_config.configure_logging`). En production, utilisez un fichier `.env` ou un orchestrateur pour mettre à jour ces valeurs sans toucher le code.

## Limitations actuelles
1. **Recherche web** : dépend de SearxNG ; en cas d'erreur (429, CAPTCHA, timeout) le circuit breaker suspend temporairement un moteur et on bascule sur un autre ou sur le cache SQL (`web_cache`).
2. **Logs & métriques** : les graphiques `ResearchScore` et les timelines sont rafraîchies à chaque itération, mais la visualisation front (vues Job detail / Runs overview) se construit côté frontend et peut être ralentie en cas de très longs jobs.
3. **LLM & performances** : les prompts du `Redacteur`/`Critic` sont orientés style scientifique, donc la rédaction peut être lente pour les jobs courts. Les budgets token/web permettent de contrôler les coûts grep.
4. **Tests automatisés** : une suite ciblée (plot_tools, graph_tools, scénarios, writing pipeline, scheduler/cache) garantit les briques supplémentaires ; les tests E2E restent à compléter pour couvrir les cas `quick_literature_review` et la résistance SearxNG.

Ces éléments évoluent au fil des itérations ; ce fichier sert de point de référence pour les développeurs qui reprennent un job ou cherchent à ajuster un scénario particulier.
