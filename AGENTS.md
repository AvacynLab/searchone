Parfait, on passe en “diff” par rapport à ce qui existe déjà, pas en redit de tout ce qui est en place.

Je te fais une **TODO structurée uniquement sur ce qu’il reste à implémenter / renforcer**, en partant de l’état actuel du backend `searchone-main` tel qu’on l’a audité.

---

## 1. Outils avancés : visualisation & graphes de connaissances

### 1.1. Outils de plots scientifiques

* [ ] Créer un module `backend/app/services/plot_tools.py`

  * [ ] Fonction `generate_plot(data, spec) -> PlotArtifact` :

    * `data` : séries, labels, erreurs éventuelles
    * `spec` : type de graphique (`"line"`, `"scatter"`, `"bar"`, `"hist"`), options log/linéaire, etc.
    * Sauvegarde en PNG dans `DATA_DIR/plots/<job_id>/...`
    * Retour : chemin fichier + métadonnées (titre, description, variables).
  * [ ] Option : générer une version vectorielle (SVG/PDF) pour les rapports.

* [ ] Déclarer un tool LLM `plot_tool` dans `workflows/agents.py`

  * [ ] Schéma function-style (arguments JSON : `data`, `spec`, `title`, `x_label`, `y_label`…)
  * [ ] Implémenter le handler dans `_execute_tool(...)`
  * [ ] Autoriser le tool pour les rôles `Analyst`, `Experimenter`, `Redacteur`.

* [ ] Intégration dans les rapports

  * [ ] Adapter `services/reporting.py` pour :

    * référencer les figures dans le rapport (lien ou tag spécial)
    * inclure la liste des figures dans le résumé structuré.

---

### 1.2. Outils de graphes de connaissances

* [ ] Créer `backend/app/services/graph_tools.py`

  * [ ] Construire un graphe à partir de `knowledge_store` :

    * nœuds = concepts / documents / claims
    * arêtes = citations, “supporte”, “contredit”.
  * [ ] Fonctions :

    * `build_knowledge_graph(job_id=None)` → renvoie structure (nodes, edges)
    * `graph_stats()` → degré moyen, composantes, hubs
    * `export_graphviz()` → fichier `.dot` + PNG.

* [ ] Tool LLM `knowledge_graph_tool`

  * [ ] Déclarer le schema dans `TOOLS`
  * [ ] Implémenter le handler :

    * autoriser principalement `Analyst`, `Coordinator`, `Critic`
    * option de retour : résumé textuel des stats + chemin vers graph PNG.

* [ ] Utilisation dans le débat / reporting

  * [ ] Ajouter dans `ResearchScore` un usage optionnel des stats du graphe
  * [ ] Dans `reporting.build_structured_summary`, inclure :

    * paragraphe “Topologie de la connaissance utilisée” (hubs, lacunes).

---

## 2. Scénarios de recherche déclaratifs (pipelines “métier”)

### 2.1. DSL de scénario

* [ ] Définir un format de scénario (YAML/JSON) dans `configs/research_scenarios.yaml` par exemple :

  * nom du scénario
  * objectif (courte description)
  * phases :

    * agents impliqués
    * outils autorisés
    * durée max / itérations max
    * critères de sortie par phase.

* [ ] Créer `backend/app/workflows/scenarios.py`

  * [ ] Loader de scénarios : `load_scenarios()` → dict nom → spec
  * [ ] Validation de schéma (pydantic ou équivalent)
  * [ ] Fonctions d’aide :

    * `get_scenario(name) -> ScenarioSpec`
    * `list_scenarios()` pour l’API.

### 2.2. Intégration dans l’orchestrateur

* [ ] Adapter `workflows/orchestrator.py`

  * [ ] Ajouter `run_with_scenario(query, scenario_name, ...)` :

    * construit un `WorkflowEngine` en fonction de la spec
    * mappe chaque phase → PipelineNode (ou groupe de nodes).
  * [ ] Permettre à l’API de choisir un scénario (`job_type` ou `scenario`).

* [ ] Scénarios par défaut à implémenter

  * [ ] `quick_literature_review` : peu d’itérations, focus sur Explorateur + Curator + Analyst
  * [ ] `deep_theory_exploration` : plus de cycles Hypothesis + Experimenter + Critic
  * [ ] `multi_domain_synthesis` : plusieurs agents/roules en parallèle sur des sous-domaines, phase finale de synthèse.

---

## 3. Méta-contrôle & re-planification avancée

### 3.1. Enrichir `ConvergenceController`

* [ ] Ajouter prise en compte des signaux suivants dans `backend/app/workflows/runtime.py` :

  * tendance du `ResearchScore` (variation, pas juste valeur brute)
  * nombre d’arguments répétés dans le débat (détecter la redite)
  * richesse des nouvelles sources (combien de nouveaux documents par itération).

* [ ] Introduire plusieurs “modes” :

  * mode exploration (loose stop, plus d’hypothèses)
  * mode exploitation (focus sur consolidation, peu de nouvelles sources)
  * mode clôture (préparation de la rédaction).

### 3.2. Actions de re-planification dynamiques

* [ ] Compléter `workflows/coordinator_actions.py` :

  * [ ] Ajouter des actions explicites :

    * `focus_on_conflicts()` : relance une phase où on exploite les contradictions
    * `seek_additional_sources(topic)` : pipeline court d’exploration ciblée
    * `downgrade_low_quality_sources()` : marquer des sources faibles dans `knowledge_store`.
  * [ ] Intégrer ces actions dans les réponses du Coordinateur (tool-style ou meta-planning).

* [ ] Dans `run_agents_job` :

  * [ ] Si `ConvergenceController` signale stagnation, laisser le Coordinateur choisir entre :

    * conclure
    * relancer un sous-pipeline selon les actions ci-dessus.

---

## 4. Écriture scientifique “article complet” (IMRaD / LaTeX-ready)

### 4.1. Pipeline de rédaction orchestré

* [ ] Créer un module `backend/app/workflows/writing_pipeline.py`

  * [ ] Fonction `build_scientific_article(job_state, format="markdown"/"latex")` :

    * récupère : hypothèses, evidence, débats, scores, graphes, plots
    * appelle successivement :

      * `OutlineGenerator` pour le plan IMRaD
      * `SectionWriter` (via LLM Rédacteur) pour chaque section
      * `GlobalCritic` pour relecture globale
      * `ReferenceManager` pour citations / biblio.
  * [ ] Structurer les sections :

    * Introduction : contexte, problème, contribution
    * Méthodes : description du pipeline agent, sources, outils
    * Résultats : hypothèses confirmées / infirmées, graphiques
    * Discussion : limites, perspectives.

### 4.2. Export LaTeX / Markdown riche

* [ ] Étendre `services/reporting.py` :

  * [ ] `export_markdown(report_struct)` : déjà en partie présent → à compléter pour sections IMRaD + figures + tableau de résultats.
  * [ ] `export_latex(report_struct)` :

    * générer un `.tex` simple (article) avec structure IMRaD
    * inclure `\cite{...}` à partir du `ReferenceManager`
    * option pour intégrer les figures générées (plots, graphes).

### 4.3. Prompts spécifiques pour Rédacteur & Critic

* [ ] Affiner `services/prompts.py` :

  * [ ] Ajouter des prompts dédiés pour :

    * `Redacteur` – style article scientifique, niveau doctorant+, pas de fluff
    * `Critic` – checklist de cohérence, de rigueur, de non-contradiction.
  * [ ] Permettre de charger des variantes “style journal X” dans `prompts/*.json`.

---

## 5. Observabilité & interface de monitoring

### 5.1. API d’observabilité

* [ ] Ajouter des endpoints dans `backend/app/api/main.py` :

  * [ ] `GET /jobs/{job_id}/metrics` :

    * retourne `ResearchScore`, tokens consommés, temps par phase, nb de sources, etc.
  * [ ] `GET /jobs/{job_id}/timeline` :

    * retourne timeline structurée (`RunContext.timeline`) avec typage (débat, décision, evidence ajoutée).
  * [ ] `GET /jobs/{job_id}/decisions` :

    * lecture filtrée de `decisions.log` pour ce job.

### 5.2. Vue front de supervision (si frontend déjà présent)

* [ ] Créer une page “Job detail” :

  * [ ] Graphique de progression du `ResearchScore` (coherence/couverture/robustesse/nouveauté)
  * [ ] Liste des phases avec statut (en cours, done, replan…)
  * [ ] Affichage des derniers messages du conseil d’agents.

* [ ] Créer une page “Runs overview” :

  * [ ] Liste des jobs récents (id, query, état, durée, tokens)
  * [ ] Filtres par scénario, date, durée, etc.

---

## 6. Scheduler & long-running jobs

### 6.1. Durcir le scheduler

* [ ] Renforcer `ResearchScheduler` (`workflows/scheduler.py`) :

  * [ ] Support de formats de schedule plus riches (cron-like ou iso8601 recurring)
  * [ ] Validation des schedules à l’écriture
  * [ ] Gestion des jobs ratés (retry, backoff).

* [ ] Ajouter une persistance plus robuste :

  * [ ] Option : stocker les schedules en base SQLite (table `research_schedule`)
  * [ ] Maintenir `schedules.json` comme cache.

### 6.2. Reprise après crash

* [ ] Vérifier / compléter `resume_from_snapshot(job_id)` :

  * [ ] gérer les cas où :

    * certains fichiers du run manquent (logs, snapshots incomplets)
    * la version du code a changé (migration minimale).
  * [ ] Ajouter une tâche de maintenance qui parcourt les snapshots obsolètes.

* [ ] Endpoints API :

  * [ ] `POST /jobs/{job_id}/resume`
  * [ ] `POST /jobs/{job_id}/stop` (utilise `stop_flags`).

---

## 7. Intégration SearxNG “production” (robustesse, fallback, cache)

### 7.1. Gestion fine des erreurs engines

* [ ] Dans `workflows/agents.py` → `run_web_search` :

  * [ ] Distinguer clairement :

    * erreurs réseau (timeout, DNS)
    * erreurs engine (CAPTCHA, 429 Too Many Requests)
    * réponses vides.
  * [ ] Implémenter un petit “circuit breaker” par engine :

    * compteur d’échecs consécutifs
    * suspension locale côté client pendant X secondes.

### 7.2. Cache applicatif

* [ ] Ajouter une couche de cache pour `run_web_search` :

  * [ ] Clé = `(query_normalisée, lang, safe_search, engine_set)`
  * [ ] Stockage dans Redis ou DB (`web_cache` table)
  * [ ] TTL configurable pour éviter les vieux résultats.

* [ ] Exposer un tool `web_cache_lookup` (optionnel) pour permettre aux agents de savoir s’ils reviennent sur des requêtes déjà faites.

---

## 8. Tests complémentaires

### 8.1. Tests unitaires manquants

* [ ] Ajouter tests pour :

  * [ ] `plot_tools` (quand implémenté) : création de figures, fichiers présents.
  * [ ] `graph_tools` : génération de graphe cohérent à partir d’un petit `knowledge_store` factice.
  * [ ] `scenarios.py` : chargement, validation, fallback sur scénario par défaut.
  * [ ] `writing_pipeline.build_scientific_article` : structure minimale correcte (sections présentes).

### 8.2. Tests E2E de scénarios

* [ ] Créer un test “mini-recherche complète” :

  * [ ] Ingest de quelques documents de test
  * [ ] Lancement d’un job avec scénario `quick_literature_review`
  * [ ] Attente de fin de job
  * [ ] Assertions :

    * présence d’au moins une hypothèse
    * score de couverture > 0
    * rapport généré non vide (Markdown).

* [ ] Test “résistance SearxNG” :

  * [ ] Simuler un SearxNG qui renvoie 429 / CAPTCHA sur certains appels (via mock)
  * [ ] Vérifier :

    * pas de crash du job
    * fallback sur d’autres engines / cache.

---

## 9. Finitions & polish

### 9.1. Config & ergonomie

* [ ] Documenter dans un `README_RESEARCH.md` :

  * [ ] les scénarios de recherche disponibles
  * [ ] comment lancer un job programmatique (API) avec scénario + options
  * [ ] les limites actuelles (web search, temps, coût).

* [ ] Exposer certains paramètres clés via env / config :

  * [ ] budgets de tokens par job
  * [ ] budgets de requêtes web par job
  * [ ] niveaux de logs (mode debug recherche).

---

Cette liste est pensée comme un “delta” : tout ce qui n’est pas listé est considéré comme déjà en place ou satisfaisant pour une V1.
Tu peux littéralement prendre ces blocs et les filer à un agent en ciblant module par module : outils, scénarios, méta-contrôle, rédaction, observabilité, scheduler, SearxNG, puis tests.
