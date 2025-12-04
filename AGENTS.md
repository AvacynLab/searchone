Parfait, on passe en mode “chef de chantier cosmique”.
Je vais m’adresser directement à l’agent qui va exécuter les tâches.

---

# TODO global – Étendre SearchOne pour un pipeline de recherche / acquisition / ingestion **SOTA++**

> Scope : **uniquement** la recherche, l’acquisition, le traitement/structuration, l’ingestion dans la mémoire/graph, et l’utilisation de ces données par les agents.
> Contraintes : 100 % open-source, aucune API payante, capable de gérer texte / HTML / PDF / tableaux / JSON / images.

Je pars de l’état actuel du projet (SearchOne) tel qu’on l’a audité :

* multi-agent + orchestrateur déjà en place,
* `web_search_tool`, SearxNG, `search_vector` / `search_semantic`,
* `knowledge_store` + `graph_tools` + `plot_tool`,
* `writing_pipeline`, `reporting`, `ConvergenceController`, scénarios YAML, etc.

Les tâches ci-dessous sont **ce qu’il faut rajouter / renforcer** par dessus.

---

## 1. Créer un “cerveau de recherche” dédié : `search_oracle`

Objectif : encapsuler toute la logique de recherche (tiers de sources, budget, structuration des résultats) dans un module unique, utilisé par les agents au lieu d’appeler directement SearxNG ou `search_vector`.

### 1.1. Nouveau module `backend/app/services/search_oracle.py`

* [x] Créer le fichier `backend/app/services/search_oracle.py` avec la structure suivante :

  * [x] `class SearchSessionState`:

    * attributs :

      * `job_id: int`
      * `root_query: str`
      * `subqueries: List[Dict[str, Any]]` (texte, statut, coverage, sources)
      * `used_budget: Dict[str, int]` (par tier : `internal`, `api`, `web`)
      * `evidence_ids: List[str]` (référence aux entries de `knowledge_store` / DB)
      * `gaps: List[str]` (angles encore peu couverts)
  * [x] Fonctions de haut niveau :

    * `plan_subqueries(root_query: str) -> List[str)`
      (décomposition LLM ou heuristique en sous-questions)
    * `search_internal(session: SearchSessionState, subquery: str) -> List[Evidence]`
      (appelle `search_vector`, `search_semantic` sur DB interne)
    * `search_web_via_searx(session: SearchSessionState, subquery: str) -> List[Evidence]`
      (wrapppe `web_search_tool` + respect budgets / circuit breaker existant)
    * `enrich_knowledge_store(evidences) -> List[str]`
      (stocke dans `knowledge_store`, retourne les IDs)
    * `update_coverage(session, subquery, evidences) -> None`
      (calcul de coverage/gaps par sous-question)

### 1.2. Tool LLM `search_oracle_tool`

* [x] Dans `backend/app/workflows/agents.py` :

  * [x] Ajouter un tool dans `TOOLS` :

    * `name: "search_oracle_tool"`
    * `description`: “Planifie et exécute une recherche multi-sources (mémoire interne + web via SearxNG), retourne un panorama structuré d’évidences et de gaps.”
    * `parameters` :

      * `query: string` (obligatoire)
      * `max_depth: integer` (optionnel)
      * `focus: string` (optionnel, ex. "recent", "theorique", "applications")
  * [x] Dans `_execute_tool`, branchement :

    * [x] Appeler `SearchSessionState(job_id=...)`
    * [x] `subqueries = plan_subqueries(query)`
    * [x] Pour chaque sous-query :

      * [x] `search_internal(...)`
      * [x] si coverage insuffisant → `search_web_via_searx(...)` (respect budget)
      * [x] `enrich_knowledge_store(...)`
      * [x] `update_coverage(...)`
    * [x] Retourner au LLM :

      * résumé des sous-questions
      * pour chaque : liste d’évidences (titres, types de source, snippets)
      * `gaps` identifiés

### 1.3. Intégration scénarios & rôles

* [x] Mettre à jour `configs/research_scenarios.yaml` :

  * [x] Pour les phases de “recherche large”, préférer `search_oracle_tool` aux appels bruts `web_search_tool`.
* [x] Dans `ROLE_ALLOWED_TOOLS` :

  * [x] Autoriser `search_oracle_tool` pour :

    * `Explorer`, `Researcher`, `SourceHunterEconTech`
  * [x] Facultatif : interdire `web_search_tool` direct pour certains rôles, pour forcer le passage par l’oracle.

---

## 2. Pipeline d’ingestion multi-format avancée

Objectif : ingestion robuste de **HTML, PDF, tableaux, JSON, images** dans une forme exploitable.

### 2.1. HTML / pages web

**Fichier(s) cible(s) :** `backend/app/services/ingest.py` (existe déjà), nouveau `backend/app/services/html_parser.py`

* [x] Créer `backend/app/services/html_parser.py` :

  * [x] Fonction `clean_html(html: str) -> str` :

    * supprimer menus, footer, nav, pubs (tags classiques : `<nav>`, `<footer>`, `<aside>`, etc.)
    * désactiver scripts/styles.
  * [x] Fonction `html_to_markdown(html: str) -> str` :

    * utiliser une lib (ex. `markdownify`, si acceptable) ou fallback maison simple.
  * [x] Fonction `extract_main_content(html: str) -> Dict[str, Any]` :

    * heuristiques : densité de texte, longueur des blocs, titre, sous-titres.
    * retourne : `{"title": ..., "sections": [...], "links": [...], "raw_html": ...}`.

* [x] Adapter `ingest.py` :

  * [x] Lorsque l’on ingère une URL :

    * utiliser `html_parser.extract_main_content`
    * découper en `chunks` structurés (par section `<h1>/<h2>/paragraphes`)
    * passer ces chunks à la logique d’indexation / `vector_store`.

### 2.2. PDF et documents

**Fichier(s) cible(s) :** nouveau `backend/app/services/pdf_parser.py`, `ingest.py`

* [x] Créer `backend/app/services/pdf_parser.py` :

  * [x] Utiliser une lib open-source (`fitz`/PyMuPDF ou `pdfminer`) pour :

    * extraire le texte paginé
    * récupérer les métadonnées (titre, auteurs, date si dispo)
  * [x] Détecter les **tableaux** dans le texte (pattern lignes/colonnes, séparateurs) :

    * extraire chaque tableau comme objet structuré (liste de lignes / colonnes)
    * associer un identifiant de tableau.
  * [x] Retourner une structure :

    ```python
    {
      "full_text": "...",
      "pages": [...],
      "tables": [...],
      "metadata": {...},
    }
    ```

* [x] Dans `ingest.py` :

  * [x] Si le fichier est PDF :

    * appeler `pdf_parser.parse(...)`
    * ingérer le texte comme chunks (sections par page / titre)
    * stocker les métadonnées en DB (`Document` / `Chunk`)
    * marquer les tables pour ingestion spéciale (voir 2.3).

### 2.3. Tableaux & données structurées

**Fichier(s) cible(s) :** nouveau `backend/app/services/table_parser.py`, `ingest.py`

* [x] Créer `backend/app/services/table_parser.py` :

  * [x] Fonctions :

    * `infer_schema(table) -> Dict[str, Any]` :

      * détecter les types de colonnes (numérique, catégorielle, date…)
      * essayer de deviner les unités / signification (via patterns + LLM si dispo).
    * `table_to_records(table, schema) -> List[Dict[str, Any]]` :

      * normaliser en JSON lignes.
  * [x] Option : exposer un outil `table_summary` pour l’agent `Analyst`.

* [x] Adapter `ingest.py` :

  * [x] Lorsqu’un tableau est détecté (CSV, Excel, table PDF) :

    * passer par `table_parser.infer_schema`
    * stocker les `records` dans une table dédiée en DB (ex. `TabularData`).

---

## 3. Structuration sémantique : entités, relations, graph enrichi

Objectif : passer de “texte + embeddings” à un **graphe de connaissance riche**, alimenté par de la NER + relation extraction.

### 3.1. Module de structuration NLP

**Fichier :** nouveau `backend/app/services/nlp_structuring.py`

* [x] Créer un module capable de fonctionner **sans API externe** :

  * [x] `extract_entities(text: str) -> List[Dict]` :

    * utiliser bibliothèque NLP open-source (ex. spaCy FR/EN) ou modèle local.
    * renvoyer : `{"text": "supraconductivité", "label": "CONCEPT", "span": [i, j]}`…
  * [x] `extract_relations(text: str, entities) -> List[Dict]` :

    * heuristiques + modèle léger si possible (relation “part_of”, “causes”, “uses”, etc.)
  * [x] `normalize_entities(entities) -> List[Dict]` :

    * tentative de mapping vers des IDs connus (via Wikidata dump local ou heuristiques : lowercasing, lemmatisation).

### 3.2. Intégration au `knowledge_store`

**Fichier :** `backend/app/data/knowledge_store.py`

* [x] Étendre la structure des `claims` :

  * [x] ajouter champs :

    * `entities: List[EntityRef]` (avec type + ID éventuel)
    * `relations: List[RelationRef]`
    * `source_doc_id`, `chunk_id`.
* [x] Ajouter une fonction :

  * [x] `store_structured_knowledge(claims, entities, relations, source_meta)` :

    * insère dans les fichiers JSONL existants (ou dans une table DB, si déjà présente),
    * ajoute des entries dans une table `KnowledgeGraphNodes` / `KnowledgeGraphEdges` si tu passes par la DB.

### 3.3. Enrichir `graph_tools`

**Fichier :** `backend/app/services/graph_tools.py`

* [x] Compléter la génération de graphe :

  * [x] inclure les **entités** comme nœuds typés (concept, personne, lieu, variable, méthode…)
  * [x] créer des arêtes “mentionné dans” (entité → document/claim)
  * [x] créer des arêtes pour les relations extraites.
* [x] Ajouter des fonctions de requête :

  * [x] `find_hubs(graph, top_k=10)` (entités les plus connectées)
  * [x] `subgraph_for_topic(topic: str)` (filtre les nœuds pertinents pour un thème).

---

## 4. Validation multi-source & fact-checking

Objectif : qu’aucune affirmation importante ne soit présentée sans **validation explicable**.

### 4.1. Tool `fact_check_tool`

**Fichier :** `backend/app/workflows/agents.py`

* [x] Ajouter un tool `fact_check_tool` :

  * [x] `parameters` :

    * `claim: string`
    * `context_ids: List[str]` (optional – IDs d’evidence du knowledge_store)
  * [x] Dans `_execute_tool` :

    * [x] Pipeline :

      1. **Décomposition** : demander au LLM de découper `claim` en sous-assertions simples.
      2. **Recherche d’évidences internes** :

         * via `search_semantic` / `search_vector` sur les documents existants.
      3. **Recherche web conditionnelle** (si coverage interne faible) :

         * via `search_oracle_tool` / `web_search_tool` (en respectant budgets).
      4. **Croisement** : pour chaque sous-assertion, classifier en :

         * `supported`, `contradicted`, `uncertain`.
      5. **Synthèse** : produire un verdict global + explication.

    * [x] Stocker le résultat dans `knowledge_store` :

      * nouveau type d’entry “fact_check” avec :

        * `claim`, `subclaims`, `verdict`, `supporting_sources`, `contradicting_sources`.

* [x] Autoriser ce tool pour les rôles :

  * `FactChecker`, `Critic`, `Analyst`.

### 4.2. Intégration au scoring & reporting

**Fichiers :** `backend/app/services/research_score.py`, `backend/app/services/reporting.py`

* [x] Adapter `ResearchScore.update(...)` :

  * [x] inclure un facteur `fact_check_pass_rate` (proportion de claims importants validés).
* [x] Dans `reporting.build_structured_summary(job_state)` :

  * [x] ajouter un volet “validation” :

    * liste des claims fact-checkés avec verdict
    * marquer ceux “uncertain” / “controversial”.

---

## 5. Gestion des contradictions & controverses

Objectif : modéliser explicitement les zones où les sources ne sont pas d’accord.

### 5.1. Extension du modèle de claim

**Fichier :** `backend/app/data/knowledge_store.py`

* [x] Ajouter un champ `status` aux claims :

  * valeurs possibles : `"supported"`, `"refuted"`, `"controversial"`, `"unknown"`.
* [x] Ajouter un champ `support_evidence_ids` et `refute_evidence_ids`.

### 5.2. Tool `resolve_conflicts_tool`

**Fichier :** `backend/app/workflows/agents.py`

* [x] Définir un tool :

  * `name: "resolve_conflicts_tool"`
  * `parameters` :

    * `claim_id: string`
  * [x] Handler :

    * [x] Récupérer le claim + ses evidences via `knowledge_store`.
    * [x] Demander au LLM de :

      * comparer les sources,
      * estimer un verdict (support/refute/controversial),
      * proposer une formulation prudente.
    * [x] Mettre à jour `claim.status` et les listes de `support_evidence_ids` / `refute_evidence_ids`.

* [x] Autoriser pour `Critic`, `Coordinator`.

### 5.3. Reporting

**Fichier :** `backend/app/services/reporting.py`

* [x] Ajouter une section “Controverses et désaccords” :

  * lister les claims `status == "controversial"`
  * pour chacun : résumé des positions + sources.

---

## 6. Anti-redondance / anti-boucle à l’échelle de la recherche

Objectif : empêcher le système de refaire les mêmes requêtes ou de tourner en rond.

### 6.1. Registre des actions de recherche

**Fichiers :** nouveau `backend/app/data/action_log.py`, modifications dans `agents.py` / `search_oracle.py`

* [x] Créer `action_log.py` :

  * [x] Modèle `SearchAction` (en DB ou JSONL) :

    * `job_id`, `agent_name`, `timestamp`
    * `action_type` (ex. "web_search", "internal_search", "fact_check")
    * `query` / `subquery`
    * `normalized_query` (lowercased, stopwords removed)
    * `result_hash` (hash des URLs / doc_ids)
  * [x] Fonctions :

    * `record_action(...)`
    * `find_similar_actions(job_id, normalized_query, action_type) -> List[SearchAction]`

* [x] Intégration :

  * [x] Dans `search_oracle`, `web_search_tool`, `search_vector`, `fact_check_tool` :

    * appeler `record_action(...)` pour chaque requête.
  * [x] Avant d’exécuter une nouvelle requête :

    * appeler `find_similar_actions(...)`
    * si > N actions très proches déjà faites → soit :

      * réutiliser les résultats existants,
      * soit informer l’agent : “requête déjà explorée, reformule ou change d’angle”.

### 6.2. Connexion avec `ConvergenceController`

**Fichier :** `backend/app/workflows/runtime.py`

* [x] Étendre `ConvergenceController.record_iteration(...)` :

  * [x] Fournir un compteur de “requêtes répétées” ou “actions redondantes” pour l’itération.
* [x] Dans `check()` :

  * [x] Ajouter une condition de stagnation si :

    * fortes répétitions de requêtes
    * peu de nouvelles sources / evidences.

---

## 7. Utilisation avancée du graphe de connaissances par les agents

Objectif : que les agents puissent interroger le graphe comme un vrai outil de raisonnement.

### 7.1. Tools de requête graphe

**Fichier :** `backend/app/workflows/agents.py`

* [x] Ajouter des tools :

  * `name: "knowledge_graph_query_tool"`

    * `parameters` : `topic: string`, `max_nodes: int`
    * handler :

      * appelle `graph_tools.subgraph_for_topic(topic)`
      * retourne les nœuds / relations pertinents.
  * `name: "knowledge_graph_hubs_tool"`

    * retourne les hubs (`find_hubs(graph, top_k)`).

* [x] Autoriser pour :

  * `Analyst`, `Coordinator`, `Hypothesis`, `Critic`.

### 7.2. Intégration aux prompts

**Fichier :** `backend/app/services/prompts.py` (ou `prompts/*.json`)

* [x] Pour les rôles concernés, enrichir les instructions système :

  * inciter à utiliser le graphe pour :

    * identifier les zones peu connectées (gaps),
    * trouver des relations inattendues,
    * détecter des patterns (clusters d’évidences).

---

## 8. Tests et scénarios E2E supplémentaires

Objectif : garantir que les nouvelles briques fonctionnent ensemble.

### 8.1. Tests unitaires

* [x] `tests/test_search_oracle.py` :

  * tester `plan_subqueries`, `search_internal` (mock), `search_web_via_searx` (mock), `update_coverage`.
* [x] `tests/test_nlp_structuring.py` :

  * tester `extract_entities`, `extract_relations` sur des textes simples.
* [x] `tests/test_fact_check_tool.py` :

  * mocker `search_semantic` / `search_oracle_tool`
  * vérifier les verdicts `supported` / `refuted` / `uncertain`.
* [x] `tests/test_action_log.py` :

  * tester `record_action` + `find_similar_actions`.

### 8.2. Tests E2E

* [x] Nouveau test : `tests/test_e2e_research_oracle.py` :

  * [x] Lancer un job avec scénario dédié qui impose l’usage de `search_oracle_tool`.
  * [x] Vérifier que :

    * des subqueries sont créées,
    * `knowledge_store` est enrichi,
    * le rapport final mentionne coverage et gaps.

* [x] Nouveau test : `tests/test_e2e_conflicts_and_facts.py` :

  * [x] Créer un petit corpus avec des infos contradictoires.
  * [x] Lancer un job où :

    * `fact_check_tool` est utilisé,
    * `resolve_conflicts_tool` est invoqué.
  * [x] Vérifier que :

    * certains claims sont `status="controversial"`,
    * le rapport liste ces controverses.

---

## 9. Documentation interne

* [x] Mettre à jour `README_RESEARCH.md` :

  * [x] ajouter une section “Search Oracle & Fact-Checking”
  * [x] décrire :

    * le nouveau module `search_oracle`
    * les nouveaux tools (`search_oracle_tool`, `fact_check_tool`, `resolve_conflicts_tool`, `knowledge_graph_*_tool`)
    * les garanties de validation multi-source.
* [x] Ajouter un schéma d’architecture (même en ASCII / diagramme) montrant :

  * agents → tools → search_oracle → knowledge_store → graph_tools → reporting.

---

Avec cette liste, l’agent a une feuille de route claire pour transformer SearchOne en **pipeline de recherche autonome complet**, capable de :

* collecter des données hétérogènes sans API payante,
* les structurer finement (graph de connaissances riche),
* vérifier et croiser les faits,
* gérer les controverses,
* éviter les boucles et redondances,
* et exposer cette puissance via un “cerveau de recherche” unique (`search_oracle_tool`) aux agents de haut niveau.

---

## Progress Tracking
- [x] Search oracle service skeleton added (`backend/app/services/search_oracle.py`).
- [x] `search_oracle_tool` defined and wired into `backend/app/workflows/agents.py`.
- [x] HTML parser service built and URL ingestion now leverages structured sections (`backend/app/services/html_parser.py`, `backend/app/services/ingest.py`).
- [x] PDF parser implemented with table detection and page-aware chunking/metadata for ingest (`backend/app/services/pdf_parser.py`, `backend/app/services/ingest.py`).
- [x] Table parser + storage pipeline added, normalizing tabular data into `TabularData` entries (`backend/app/services/table_parser.py`, `backend/app/services/ingest.py`, `backend/app/data/db.py`).
- [x] NLP structuring heuristics added for entity extraction, relations, and normalization (`backend/app/services/nlp_structuring.py`).
- [x] Knowledge store now records structured claims/entities/relations with new persist helper (`backend/app/data/knowledge_store.py`).
- [x] Graph tooling reinforced with entity-aware nodes, relation edges, `find_hubs`, and `subgraph_for_topic` helpers (`backend/app/services/graph_tools.py`).
- [x] Action log tracking deployed and `fact_check_tool` added for verification workflows (`backend/app/data/action_log.py`, `backend/app/workflows/agents.py`).
- [x] Knowledge graph query/hubs tools plus updated prompts encourage graph-based reasoning (`backend/app/workflows/agents.py`, `backend/app/services/prompts.py`).
- [x] Conflict resolution tool + reporting inclusion for fact-checks/controversies (`backend/app/workflows/agents.py`, `backend/app/services/reporting.py`).
- [x] Tests for action logging and NLP structuring added to keep regressions visible (`tests/test_action_log.py`, `tests/test_nlp_structuring.py`).
- [x] Continue with fact-checking analysis, reporting, and action logging.
- [x] Added unit tests for `search_oracle` and `fact_check_tool` to validate planning, search, and verdict synthesis (`tests/test_search_oracle.py`, `tests/test_fact_check_tool.py`).
- [x] README_RESEARCH now documents the Search Oracle / fact-check pipeline and includes the requested architecture sketch.
- [x] ConvergenceController consumes action-log redundancy counts to flag repeated access patterns (`backend/app/workflows/runtime.py`, `backend/app/workflows/agents.py`, `backend/app/data/action_log.py`).
- [x] Research scenarios now prefer `search_oracle_tool` and the new E2E tests validate coverage/gaps plus controversy reporting (`configs/research_scenarios.yaml`, `tests/test_e2e_research_oracle.py`, `tests/test_e2e_conflicts_and_facts.py`).

## Action History
- 2025-12-04 10:03:30 — Scaffolding of the search oracle service and its tool integration (session state, planning, coverage tracking) completed.
- 2025-12-04 10:06:50 — Added HTML parser helpers and wired ingest_web_page to use structured sections/links before chunking.
- 2025-12-04 10:09:19 — Created `pdf_parser` with table heuristics and updated `ingest_pdf_file` to consume structured pages/tables.
- 2025-12-04 10:12:47 — Added table parser normalization + `TabularData` persistence so detected tables become structured records.
- 2025-12-04 10:45:21 — Instrumented action logging + `fact_check_tool` for redundancy checks and multi-source validation.
- 2025-12-04 10:48:10 — Added knowledge graph query/hubs tools and updated prompts so agents interrogate the graph proactively.
- 2025-12-04 10:52:05 — Added conflict resolution tool and enriched reporting to highlight fact-checks & controversies.
- 2025-12-04 11:55:14 — Extended research score/reporting with fact-check coverage metrics and controversy summaries.
- 2025-12-04 10:20:10 - Structured knowledge support introduced (entity/relation persistence plus entity-aware graph hubs/subgraphs).
- 2025-12-04 10:16:42 - Added `nlp_structuring` helpers for entity extraction, relation spotting, and normalization without external APIs.
- 2025-12-04 12:14:18 - Wired the action log into Convergence, added search_oracle/fact_check tests, and documented the Search Oracle + fact-check pipeline in `README_RESEARCH.md`.
- 2025-12-04 13:05:00 - Codified search_oracle-first scenarios and added E2E regression tests that check coverage, gaps, and controversial fact-checks.
