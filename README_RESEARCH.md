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

## Search Oracle & Fact-Checking

La nouvelle couche `search_oracle` encapsule toute la logique de recherche multi-source : elle planifie les sous-requêtes, sollicite d'abord la mémoire vectorielle interne (`search_vector`/`search_semantic`), puis bascule automatiquement vers SearxNG si la couverture est insuffisante. Chaque session garde la trace des budgets (`internal`, `web`, `api`), des gaps identifiés et des identifiants d'évidences persistés dans `knowledge_store`. Ce cerveau de recherche est exposé via `search_oracle_tool` aux rôles `Explorer`, `Researcher` et `SourceHunterEconTech`, ce qui garantit un pilotage cohérent des sources externes et des doublons.

Pour valider les conclusions, le workflow embarque également :

* `fact_check_tool` qui décompose une affirmation, recherche des preuves internes puis externes, classe chaque sous-assertion (`supported`, `uncertain`, `contradicted`) et persiste le verdict dans `knowledge_store`.
* `resolve_conflicts_tool` pour réévaluer un claim existant, comparer les sources et basculer explicitement le statut (`supported`, `refuted`, `controversial`, `unknown`).
* L'intégration de `ResearchScore` et `reporting` avec le facteur `fact_check_pass_rate` ainsi que la nouvelle section “Controverses et désaccords”.

Les agents peuvent interroger le graphe enrichi via :

* `knowledge_graph_query_tool` (sous-graphe filtré par thème) et `knowledge_graph_hubs_tool` (nœuds les plus connectés).
* Des prompts étendus qui incitent Analyst, Hypothesis, Critic et Coordinator à scruter les gaps et les hubs du graphe.

### Architecture simplifiée

```
        [ Agents ]
             │
             ▼
       [ Tools Layer ]
  (search_oracle, fact_check_tool, resolve_conflicts_tool,
   knowledge_graph_query_tool, knowledge_graph_hubs_tool, ...)
             │
             ▼
       [ search_oracle ]
             │
             ▼
    [ knowledge_store + action_log ]
             │
             ▼
       [ graph_tools ]
             │
             ▼
        [ reporting ]
```

Les agents consultent ces outils, `search_oracle` enrichit la mémoire et `knowledge_store`, le graphe reflète les entités/relations, et les rapports/fact checks recyclent ces structures pour délivrer des synthèses robustes.

## Outils avancés

Les outils plot_tool et knowledge_graph_tool permettent de générer des figures et des graphes de connaissances directement depuis les agents :

* **plot_tool** produit des PNG/SVG à partir de séries {x, y, label, yerr} et expose plot_type, description, ariables, ector_formats. Chaque preuve est annotée avec meta.source_type= plot dans la timeline pour être reprise ensuite par les rapports/figures.
* **knowledge_graph_tool** reconstruit le graphe des claims/promotions/pollutions, calcule des métriques (
odes, edges, components, hubs) via graph_stats, exporte PNG/DOT via export_graphviz et renseigne state[knowledge_graph_stats] / state[knowledge_graph_exports]. Les rapports mentionnent la topologie sous Topologie du graphe de connaissances.

## Exemple de scénario illustratif

`yaml
scenarios:
  figures_and_graphs:
    objective: Valider une hypothèse avec statistiques figures et graphes de connaissances.
    phases:
      - name: analysis
        description: Générer des statistiques et figures.
        agents: [Analyst, Experimenter]
        tools: [stats_summary, plot_tool, web_search_tool]
        max_iterations: 3
      - name: synthesis
        description: Construire le graphe de connaissances.
        agents: [Analyst, Critic]
        tools: [knowledge_graph_tool, search_semantic]
        max_iterations: 2
`

La phase nalysis produit les figures, et synthesis capture la structure du savoir via le graphe exporté.
