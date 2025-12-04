Parfait, on fige l’écart avec le code actuel et on le transforme en checklist “agent-ready”.
Je te donne une TODO **précise**, par module/fichier, uniquement sur ce qu’il reste à implémenter.

---

## 1. Exposer les plots comme vrai tool LLM (`plot_tool`)

### 1.1. Déclaration du tool

**Fichier :** `backend/app/workflows/agents.py`

* [x] Ajouter une entrée `plot_tool` dans la liste `TOOLS` :

  * [x] `name: "plot_tool"`
  * [x] `description`: génération de figures à partir de données numériques (courbes, barres, histogrammes…)
  * [x] `parameters` (JSON schema) :

    * `job_id` (int) – optionnel, sinon dérivé de l’état courant
    * `series` (array d’objets `{x: [...], y: [...], label: str, yerr?: [...]}`)
    * `plot_type` (enum: `"line" | "scatter" | "bar" | "hist"`)
    * `title` (string)
    * `x_label` / `y_label` (string, optionnel)
    * champs optionnels : `log_scale`, `bins`, etc., mappés sur `plot_tools.generate_plot`.

### 1.2. Handler du tool

**Fichier :** `backend/app/workflows/agents.py`

* [x] Dans la fonction qui exécute les tools (l’équivalent de `_execute_tool(name, args, agent, state, ...)`) :

  * [x] Ajouter un branchement `if name == "plot_tool":`

    * [x] Valider les arguments (`series`, `plot_type`, etc.).
    * [x] Appeler `generate_plot` de `backend/app/services/plot_tools.py` :

      * [x] `from app.services.plot_tools import generate_plot`
      * [x] `artifact = generate_plot(data=..., spec=...)`
    * [x] Construire un objet `evidence` avec :

      * `content`: courte description de ce que représente la figure
      * `meta`: inclure au minimum :

        * `"figure": {"path": artifact.path, "svg_path": artifact.svg_path, "title": ..., "plot_type": ...}`
        * `"source_type": "plot"`
    * [x] Ajouter cette evidence dans `state["evidence"]` ou équivalent.

### 1.3. Rattacher aux rôles

**Fichier :** `backend/app/workflows/agents.py`

* [x] Mettre à jour `ROLE_ALLOWED_TOOLS` :

  * [x] Autoriser `"plot_tool"` pour :

    * `Analyst`
    * `Experimenter`
    * `Redacteur` (pour demander des figures pour l’article).

### 1.4. Intégration à la pipeline d’écriture

**Fichiers :**

* `backend/app/workflows/writing_pipeline.py`

* `backend/app/services/reporting.py`

* [x] Vérifier/adapter `_collect_figures(state)` :

  * [x] S'assurer qu'il récupère bien les `evidence.meta["figure"]` produites par `plot_tool`.
  * [x] Si besoin, enrichir la structure `figure` : `caption`, `variables`, `source`.

* [x] Dans `build_scientific_article(...)` :

  * [x] S'assurer que la clé `figures` du résultat contient bien toutes les figures générées par `plot_tool`.
  * [ ] Option : ajouter une section “Liste des figures” dans le Markdown/LaTeX.

### 1.5. Tests

**Fichier :** `backend/tests/test_plot_tools.py` (et/ou nouveau fichier)

* [x] Ajouter un test d’intégration tool :

  * [x] Simuler un appel tool-calling `"plot_tool"` via la fonction d’exécution de tool.
  * [x] Vérifier que :

    * [x] `generate_plot` est bien appelé (mock)
    * [x] `state["evidence"]` contient une entrée avec `meta["figure"]` valide.
* [x] Ajouter un test `test_article_includes_figures` :

  * [x] Construire un `state` minimal contenant une evidence avec `meta["figure"]`.
  * [x] Appeler `build_scientific_article(...)`.
* [x] Vérifier que `result["figures"]` n’est pas vide et que la figure y est bien.

## Progress log
- [2025-12-04] Registered `plot_tool` in `TOOLS`, enforced role permissions, extended `_execute_tool` to generate figures, and surfaced plot metadata through `_collect_figures`.
- [2025-12-04] Added integration tests covering `plot_tool` execution and `build_scientific_article`, then ran `pytest backend/tests/test_plot_tools.py`.
- [2025-12-04] Added `knowledge_graph_tool`, state tracking, and charges to `ROLE_ALLOWED_TOOLS` plus reporting/writing adjustments for stats/exports.
- [2025-12-04] Added `test_graph_tools.py` (tool + reporting), verified `knowledge_graph` summary, and ran `pytest backend/tests/test_graph_tools.py`.
- [2025-12-04] Implemented phase `tool_allowlist`, blocked unauthorized tools with a descriptive tool result, and covered the behavior via `backend/tests/test_tool_allowlist.py`.
- [2025-12-04] Ensured `configs/research_scenarios.yaml` only lists existing tools, added runtime validation/test coverage, and confirmed `backend/tests/test_scenarios.py`.
- [2025-12-04] Updated top-level tests to use `app.*` imports and verified `tests/test_functional_modules.py`, `tests/test_e2e_placeholders.py`, `tests/test_convergence.py` via pytest.
- [2025-12-04] Documented `plot_tool`/`knowledge_graph_tool` in `README_RESEARCH.md` and added an illustrative scenario example that showcases both tools.

---

## 2. Intégrer `graph_tools` comme `knowledge_graph_tool` + stats dans le rapport

### 2.1. Tool LLM `knowledge_graph_tool`

**Fichier :** `backend/app/workflows/agents.py`

* [x] Ajouter une entrée `knowledge_graph_tool` dans `TOOLS` :

  * [x] `name: "knowledge_graph_tool"`
  * [x] `description`: construit un graphe de connaissances basé sur les claims, sources et pollution actuelle, calcule des stats, exporte un visuel.
  * [x] `parameters` :

    * `job_id` (int, optionnel)
    * `scope` (`"current_job"` | `"global"`) – optionnel.

* [x] Dans le handler de tools :

  * [x] Ajouter un branchement `if name == "knowledge_graph_tool":`

    * [x] Appeler `build_knowledge_graph(job_id=...)` de `app.services.graph_tools`.
    * [x] Appeler `graph_stats(graph)` pour les métriques.
    * [x] Appeler `export_graphviz(graph, job_id, fmt="png")`.
    * [x] Mettre à jour `state` :

      * [x] `state["knowledge_graph_stats"] = stats`
      * [x] `state.setdefault("knowledge_graph_exports", []).append({"format": "png", "path": ..., "created_at": ...})`
    * [x] Retourner un “tool result” lisible :

      * résumé textuel des stats (nombre de nœuds, densité, hubs, etc.)
      * chemin du fichier PNG, si utile à l’agent.

* [x] Mise à jour de `ROLE_ALLOWED_TOOLS` :

  * [x] Autoriser `knowledge_graph_tool` pour :

    * `Analyst`
    * `Coordinator`
    * `Critic`

### 2.2. Intégration reporting & writing

**Fichier :** `backend/app/services/reporting.py`

* [x] Dans `build_structured_summary(job_state)` :

  * [x] Récupérer :

    * `kg_stats = job_state.get("knowledge_graph_stats", {})`
    * `kg_exports = job_state.get("knowledge_graph_exports", [])`
  * [x] Ajouter une section dans le résumé structuré :

    * “Topologie du graphe de connaissances”
    * Inclure quelques chiffres clés : `nodes`, `edges`, `components`, `top_hubs`, etc.
  * [x] Ajouter un champ optionnel dans la structure retournée :

    * `summary["knowledge_graph"] = {"stats": kg_stats, "exports": kg_exports}`

**Fichier :** `backend/app/workflows/writing_pipeline.py`

* [x] Dans `build_scientific_article(...)` :

  * [x] Si `state["knowledge_graph_stats"]` existe :

    * Ajouter un paragraphe dans la section Discussion ou Méthodes :

      * “Structure du graphe de connaissances utilisé” (nombre de nœuds, nature des liens, hubs).
  * [x] Si `knowledge_graph_exports` contient des graphes :

    * Les ajouter à la liste des figures ou en section annexe.

### 2.3. Tests

**Fichier :** `backend/tests/test_graph_tools.py` (et/ou nouveau fichier)

* [x] Ajouter un test `test_knowledge_graph_tool_integration` :

  * [x] Mocker `build_knowledge_graph`, `graph_stats`, `export_graphviz`.
  * [x] Simuler un appel tool `knowledge_graph_tool`.
  * [x] Vérifier que `state["knowledge_graph_stats"]` et `state["knowledge_graph_exports"]` sont correctement remplis.

* [x] Ajouter un test reporting :

  * [x] Construire un `job_state` avec `knowledge_graph_stats` + `knowledge_graph_exports`.
  * [x] Appeler `build_structured_summary`.
  * [x] Vérifier que le bloc `knowledge_graph` est bien présent et non vide.

---

## 3. Faire respecter `phase.tools` (tool_allowlist) dans `run_agents_job`

### 3.1. Étendre `run_agents_job` pour accepter un allowlist

**Fichier :** `backend/app/workflows/agents.py`

* [x] Modifier la signature de `run_agents_job` :

  ```python
  async def run_agents_job(
      job_id: int,
      query: str,
      max_iterations: int,
      roles: List[str],
      llm_client,
      embedder,
      vs,
      max_duration_seconds: int,
      max_token_budget: int,
      bus,
      run_ctx,
      controller,
      tool_allowlist: Optional[List[str]] = None,
  ):
  ```
* [x] Propager `tool_allowlist` là où tu construis les `AgentSpec` / les instance d’agents :

  * [x] stocker `agent_spec.allowed_tools = ROLE_ALLOWED_TOOLS[role] ∩ tool_allowlist` (si non-None),
  * [x] ou passer `tool_allowlist` directement au dispatch de tool.

### 3.2. Appliquer l’allowlist dans l’exécution de tools

**Fichier :** `backend/app/workflows/agents.py`

* [x] Dans `_execute_tool(...)` (ou équivalent) :

  * [x] Récupérer la liste des tools autorisés pour cet agent :

    * soit depuis `agent_spec.allowed_tools` / `agent.state.allowed_tools`,
    * soit depuis `state["tool_allowlist"]`.
  * [x] Avant d’exécuter un tool :

    * si `tool_allowlist` est non-None **et** `name` n’est pas dedans → refuser :

      * soit en renvoyant un message d’erreur au LLM,
      * soit en loggant et en retournant un “tool result” indiquant que l’outil n’est pas autorisé dans cette phase.

### 3.3. Connexion côté orchestrateur

**Fichier :** `backend/app/workflows/orchestrator.py`

* [x] Là où tu construis `kwargs` pour `run_agents_job` dans chaque phase :

  * [x] S’assurer que tu passes bien :

    ```python
    kwargs["tool_allowlist"] = phase.tools or None
    ```
  * [x] Adapter l’appel :

    ```python
    result = await run_agents_job(**kwargs)
    ```

### 3.4. Tests

**Fichier :** `backend/tests/test_e2e_scenarios.py`

* [x] Ajouter un test où :

  * [x] un scénario de test a une phase avec `tools: ["web_search_tool"]` uniquement ;
  * [x] un agent essaie d’appeler `plot_tool` (via un prompt / tool-calling simulé).
* [x] Assertions :

  * [x] `plot_tool` n’est pas exécuté (mock non appelé),
  * [x] un log ou un “tool result” indique que l’outil est interdit.

---

## 4. Alignement des noms d’outils dans `research_scenarios.yaml`

### 4.1. Audit et correction

**Fichier :** `configs/research_scenarios.yaml`

* [x] Parcourir toutes les phases de tous les scénarios :

  * [x] S’assurer que chaque `tool` listé dans `tools:` correspond **exactement** à un nom dans `TOOLS` (après ajout de `plot_tool` et `knowledge_graph_tool`).
  * [x] Corriger :

    * `plot_tool` → garder ce nom si c’est celui que tu implémentes,
    * `knowledge_graph_tool` → idem, correspond au tool que tu ajoutes,
    * supprimer ou renommer les outils inexistants.

### 4.2. Validation de cohérence au chargement

**Fichier :** `backend/app/workflows/scenarios.py`

* [x] Dans `load_scenarios()` ou juste après, ajouter un check :

  * [x] Récupérer la liste des tool names connus (depuis `TOOLS` via un import léger, ou définir une liste `KNOWN_TOOLS`).
  * [x] Pour chaque `phase.tools` :

    * si un nom ne figure pas dans `KNOWN_TOOLS` → log WARNING ou lever une exception explicite (en dev/test).

### 4.3. Tests

**Fichier :** `backend/tests/test_scenarios.py`

* [x] Ajouter un test qui :

  * [x] Charge les scénarios,
  * [x] Vérifie que tous les `phase.tools` sont bien dans la liste des tools déclarés dans `agents.TOOLS`.

---

## 5. Corriger les imports dans les tests top-level

### 5.1. Revue des imports

**Dossier :** `backend/tests/` (ou `tests/` à la racine selon ton setup)

* [x] Chercher tous les imports du type :

  * `from runtime import ...`
  * `from workflow import ...`
  * `from scheduler import ...`
  * `from debate import ...`
  * `from references import ...`
  * `from writing_pipeline import ...`
* [ ] Remplacer par des imports qualifiés, par ex. :

  * `from app.workflows.runtime import ConvergenceController`
  * `from app.workflows.workflow import WorkflowEngine`
  * `from app.workflows.scheduler import ResearchScheduler`
  * `from app.workflows.debate import run_debate`
  * `from app.services.references import ReferenceManager`
                    * `from app.workflows.writing_pipeline import build_scientific_article`

### 5.2. Adapter `conftest.py` si nécessaire

**Fichier :** `backend/tests/conftest.py` (ou équivalent)

* [x] Vérifier que :

  * `sys.path` contient bien `backend` (pour `import app...`),
  * tu n’ajoutes pas de chemin ambigu qui masque les imports `app.*`.

### 5.3. Vérification globale

* [x] Lancer `pytest` à la racine du repo.
* [x] Corriger les éventuelles erreurs d’import restantes.

---

## 6. Finitions et doc (facultatif mais recommandé)

### 6.1. Documenter les nouveaux tools

**Fichier :** `README_RESEARCH.md`

* [x] Ajouter une section “Outils avancés” :

  * [x] `plot_tool` :

    * rôle : générer des figures de données.
    * paramètres principaux.
  * [x] `knowledge_graph_tool` :

    * rôle : produire graphe de connaissances + stats.
    * comment il apparaît dans les rapports (stats, figures, etc.).

### 6.2. Exemple de scénario utilisant les nouveaux outils

**Fichier :** `README_RESEARCH.md` ou nouvel exemple

* [x] Ajouter un exemple de scénario YAML (ou extrait de `configs/research_scenarios.yaml`) :

  * Une phase analytique avec `tools: ["web_search_tool", "plot_tool", "knowledge_graph_tool"]`.
  * Explication de ce que fait chaque phase.

---

Une fois cette liste cochée, tu auras réellement :

* des **plots** et des **graphes de connaissance** exploitables par les agents et visibles dans les rapports,
* des **scénarios** qui imposent effectivement un sous-ensemble d’outils,
* des **tests** cohérents qui tournent sur la bonne arborescence de modules.

Là on sera vraiment sur une V1 “fermée” de ton architecture de recherche autonome, prête pour les raffinements de comportement plutôt que de plomberie.
