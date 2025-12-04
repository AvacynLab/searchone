import asyncio
import logging
import os
import platform
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
import time
import json
import requests
import httpx
from collections import Counter
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
from datetime import datetime as _dt
from app.data.db import (
    Job,
    get_session,
    Chunk,
    save_job_state,
    get_web_cache_entry,
    store_web_cache_entry,
    cleanup_web_cache_entries,
)
from app.services.llm import LLMClient
from app.data.vector_store import FaissStore
from app.core.config import (
    EMBEDDING_MODEL,
    DATA_DIR,
    SEARXNG_URL,
    WEB_CACHE_ENABLED,
    WEB_CACHE_TTL_SECONDS,
    WEB_CACHE_CLEANUP_INTERVAL,
    WEB_SEARCH_ENGINE_NAME,
    WEB_SEARCH_ENGINE_SET,
    WEB_SEARCH_FAILURE_THRESHOLD,
    WEB_SEARCH_BREAKER_COOLDOWN,
    JOB_TOKEN_BUDGET,
    WEB_QUERY_BUDGET,
)
from app.services.prompts import propose_prompt, vote_prompt
from app.core.stop_flags import is_stop_requested, clear_stop
from app.core.model_router import resolve_model
from app.workflows.runtime import AgentSpec, RunContext, MessageBus, ConvergenceController, AgentInterface
from app.data.memory import ConversationMemory
from app.data.knowledge_store import search_semantic, get_related_nodes, store_claim, promote_knowledge
from app.services.research_score import ResearchScore
from app.workflows.debate import DebateRound, tally_votes, run_debate
from app.services.writing import OutlineGenerator, SectionWriter, StyleCritic, FinalComposer, GlobalCritic
from app.services.reporting import build_diagnostic
from app.core.tracing import start_span
from app.services.references import ReferenceManager
env_simple = os.getenv("SEARCHONE_SIMPLE_EMBEDDER")
env_force_st = os.getenv("SEARCHONE_FORCE_ST")
env_enable_rerank = os.getenv("SEARCHONE_ENABLE_RERANK")
CROSS_ENCODER_MODEL = os.getenv("SEARCHONE_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COUNCIL_MAX_MSGS = int(os.getenv("SEARCHONE_COUNCIL_MAX_MSGS", "1"))
COUNCIL_MAX_VOTES_PER_MSG = int(os.getenv("SEARCHONE_COUNCIL_MAX_VOTES_PER_MSG", "1"))
MIN_RELIABILITY = float(os.getenv("SEARCHONE_MIN_SOURCE_RELIABILITY", "0.5"))
PREFERRED_TYPES = ("pdf", "url")
AUTO_SEED_CORPUS = os.getenv("SEARCHONE_AUTO_SEED_CORPUS", "")
AUTO_SEED_ENDPOINT = os.getenv("SEARCHONE_SEED_ENDPOINT", "http://127.0.0.1:2001/ingest/seed_corpus")
AGENT_TOKEN_BUDGET = int(os.getenv("SEARCHONE_AGENT_TOKEN_BUDGET", "0"))  # 0 means no extra limit
WEB_REQUEST_BUDGET = WEB_QUERY_BUDGET
MAX_EMPTY_ITERS = int(os.getenv("SEARCHONE_MAX_EMPTY_ITERS", "2"))
HEALTHCHECK_TIMEOUT = float(os.getenv("SEARCHONE_HEALTHCHECK_TIMEOUT", "5"))
RETRIEVAL_ONLY_THRESHOLD = int(os.getenv("SEARCHONE_RETRIEVAL_ONLY_THRESHOLD", "2"))
RETRIEVAL_ONLY_TOPK = int(os.getenv("SEARCHONE_RETRIEVAL_ONLY_TOPK", "8"))
ESTIMATED_COST_PER_TOKEN = float(os.getenv("SEARCHONE_ESTIMATED_COST_PER_TOKEN", "0"))
STAGNATION_WINDOW = int(os.getenv("SEARCHONE_STAGNATION_WINDOW", "3"))
STAGNATION_MIN_DELTA = float(os.getenv("SEARCHONE_STAGNATION_MIN_DELTA", "0.01"))
EXPERIMENT_TIMEOUT = float(os.getenv("SEARCHONE_EXPERIMENT_TIMEOUT", "6"))
SOURCE_PRIORITY_RECENCY_DAYS = int(os.getenv("SEARCHONE_SOURCE_PRIORITY_RECENCY_DAYS", "365"))
SOURCE_PRIORITY_TYPE_BONUS = {"pdf": 0.1, "url": 0.05}
AUTO_STOP_AGREE = float(os.getenv("SEARCHONE_AUTO_STOP_AGREE", "0.65"))
AUTO_STOP_DISAGREE = float(os.getenv("SEARCHONE_AUTO_STOP_DISAGREE", "0.65"))
REPLAN_MIN_COVERAGE = float(os.getenv("SEARCHONE_REPLAN_MIN_COVERAGE", "0.15"))
REPLAN_MIN_EVIDENCE = int(os.getenv("SEARCHONE_REPLAN_MIN_EVIDENCE", "1"))
DEFAULT_SEARX_API = f"{SEARXNG_URL.rstrip('/')}/search" if SEARXNG_URL else ""
WEB_SEARCH_ENDPOINT = os.getenv("SEARCHONE_WEB_SEARCH_ENDPOINT") or DEFAULT_SEARX_API
WEB_SEARCH_TIMEOUT = float(os.getenv("SEARCHONE_WEB_SEARCH_TIMEOUT", "8"))
WEB_FETCH_TIMEOUT = float(os.getenv("SEARCHONE_WEB_FETCH_TIMEOUT", "10"))
WEB_FETCH_MAX_CHARS = int(os.getenv("SEARCHONE_WEB_FETCH_MAX_CHARS", "6000"))
default_simple = platform.system().lower() == "windows"
USE_SIMPLE_EMBEDDER = (
    env_simple.lower() in ("1", "true", "yes", "on") if env_simple is not None else default_simple
)
if env_force_st and env_force_st.lower() in ("1", "true", "yes", "on"):
    USE_SIMPLE_EMBEDDER = False

if not USE_SIMPLE_EMBEDDER:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception:
            CrossEncoder = None  # type: ignore
        _HAS_ST = True
    except Exception:
        SentenceTransformer = None
        CrossEncoder = None  # type: ignore
        _HAS_ST = False
else:
    SentenceTransformer = None
    CrossEncoder = None  # type: ignore
    _HAS_ST = False

# lightweight fallback embedder used when sentence-transformers is not installed or disabled
if not _HAS_ST:
    class SimpleEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, *args, **kwargs):
            import numpy as _np
            out = []
            for t in texts:
                h = abs(hash(t)) % 997
                out.append([float((h % 10) / 10.0), float((h % 7) / 7.0), float((h % 5) / 5.0)])
            return _np.array(out)

        def get_sentence_embedding_dimension(self):
            return 3
from app.core.logging_config import configure_logging
from app.services.analytics import stats_summary
from app.services.analytics import ttest_independent
from app.core.stats_tools import correlation_matrix
from app.services.sympy_tools import simplify_expression, solve_equation
from app.core.observability import compute_run_metrics, log_decision
from app.core.guardrails import is_high_risk, filter_tools, audit_entry
from app.workflows.coordinator_actions import evaluate_replan
from app.workflows.workflow import execute_actions_from_bus
from sqlmodel import select

configure_logging()
logger = logging.getLogger(__name__)
DEFAULT_ROLES = ["Researcher", "Critic", "Experimenter", "FactChecker", "SourceHunterEconTech"]
ENABLE_RERANK = env_enable_rerank.lower() in ("1", "true", "yes", "on") if env_enable_rerank else False
_cross_encoder = None
ROLE_ALLOWED_TOOLS = {
    "Explorer": {"web_search_tool", "fetch_and_parse_url", "search_hybrid", "search_vector", "search_semantic", "web_cache_lookup"},
    "Curator": {"search_hybrid", "search_vector", "search_semantic", "web_cache_lookup"},
    "Analyst": {"search_semantic", "search_vector", "stats_summary", "correlation_matrix", "ttest_independent", "simplify_expression", "solve_equation", "web_cache_lookup"},
    "Experimenter": {"run_experiment", "stats_summary", "ttest_independent", "simplify_expression", "web_cache_lookup"},
    "Coordinator": {"web_cache_lookup"},
    "Critic": {"search_semantic", "search_vector", "web_cache_lookup"},
    "Redacteur": {"web_cache_lookup"},
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_hybrid",
            "description": "Recherche locale dans le vector store (docs ingeres) pour trouver des passages pertinents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Requete a chercher"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 12, "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vector",
            "description": "Recherche simple dans le vector store local (docs ingeres).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 12, "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sources_summary",
            "description": "Recupere un resume des sources ingerees (top domaines, docs recents, fiabilite).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_async",
            "description": "Enqueue des URLs/PDF pour ingestion asynchrone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "title": {"type": "string"},
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": "Recherche web externe (API configurable via env SEARCHONE_WEB_SEARCH_ENDPOINT) pour trouver des URLs pertinentes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Requete a chercher"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_cache_lookup",
            "description": "Interroger le cache applicatif pour savoir si une requête web a déjà été exécutée.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Requête recherchée"},
                    "lang": {"type": "string", "description": "Lang de la requête", "default": "fr"},
                    "safe_search": {"type": "boolean", "description": "Activer le filtrage safe search", "default": True},
                    "engine": {"type": "string", "description": "Moteur utilisé pour le cache", "default": WEB_SEARCH_ENGINE_NAME},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_and_parse_url",
            "description": "Recupere une page web et renvoie du texte nettoye (HTML -> texte brut).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL a recuperer"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_experiment",
            "description": "Execute un script Python dans un sandbox controle avec timeout, retourne stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "Code Python a executer"},
                    "timeout": {"type": "number", "minimum": 1, "maximum": 30, "default": 5},
                },
                "required": ["script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stats_summary",
            "description": "Calcule des statistiques simples (mean/median/std/min/max) sur une liste de nombres.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correlation_matrix",
            "description": "Calcule une matrice de correlation (Pearson) pour un tableau 2D de valeurs numeriques.",
            "parameters": {
                "type": "object",
                "properties": {
                    "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                },
                "required": ["matrix"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ttest_independent",
            "description": "Effectue un t-test de Welch (echantillons independants, variances inegales) sur deux listes de valeurs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": {"type": "number"}},
                    "b": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simplify_expression",
            "description": "Simplifie une expression mathematique symbolique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string"},
                },
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": "Resout une equation symbolique simple lhs=rhs pour une inconnue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lhs": {"type": "string"},
                    "rhs": {"type": "string", "default": "0"},
                    "symbol": {"type": "string", "default": "x"},
                },
                "required": ["lhs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic",
            "description": "Recherche semantique dans la base persistante (vector store) et retourne les metadonnees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 12, "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_claim",
            "description": "Stocke une affirmation et ses evidences (long terme).",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "evidence_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_nodes",
            "description": "Recupere des noeuds relies a une entite dans le graphe de connaissances.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                },
                "required": ["entity"],
            },
        },
    },
]

ENGINE_FAILURE_STATE: Dict[str, Dict[str, Any]] = {}
LAST_CACHE_CLEANUP = 0.0

def _normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def _build_cache_key(query: str, lang: str, safe_search: bool, engine_set: str) -> str:
    normalized = _normalize_query(query)
    return f"{normalized}|{lang}|{int(bool(safe_search))}|{engine_set}"


def _lookup_web_cache(query: str, lang: str, safe_search: bool, engine_set: str):
    if not WEB_CACHE_ENABLED:
        return None
    key = _build_cache_key(query, lang, safe_search, engine_set)
    entry = get_web_cache_entry(key)
    if not entry or not entry.result_payload:
        return None
    try:
        return json.loads(entry.result_payload)
    except Exception:
        return None


def _persist_web_cache(query: str, lang: str, safe_search: bool, engine_set: str, results: List[Dict[str, Any]]) -> None:
    if not WEB_CACHE_ENABLED:
        return
    key = _build_cache_key(query, lang, safe_search, engine_set)
    payload = json.dumps(results, ensure_ascii=False)
    expires = datetime.now(timezone.utc) + timedelta(seconds=WEB_CACHE_TTL_SECONDS)
    store_web_cache_entry(key, query, lang, safe_search, engine_set, payload, expires)


def _clean_cache_if_due() -> None:
    global LAST_CACHE_CLEANUP
    if not WEB_CACHE_ENABLED or WEB_CACHE_CLEANUP_INTERVAL <= 0:
        return
    now_ts = time.time()
    if now_ts - LAST_CACHE_CLEANUP < WEB_CACHE_CLEANUP_INTERVAL:
        return
    cleanup_web_cache_entries(datetime.now(timezone.utc))
    LAST_CACHE_CLEANUP = now_ts


def _is_engine_paused(engine: str) -> bool:
    state = ENGINE_FAILURE_STATE.get(engine) or {}
    paused_until = state.get("paused_until")
    return bool(paused_until and paused_until > time.time())


def _record_engine_result(engine: str, success: bool) -> None:
    state = ENGINE_FAILURE_STATE.setdefault(engine, {"failures": 0, "paused_until": None})
    if success:
        state["failures"] = 0
        state["paused_until"] = None
        return
    state["failures"] += 1
    if state["failures"] >= WEB_SEARCH_FAILURE_THRESHOLD:
        state["paused_until"] = time.time() + WEB_SEARCH_BREAKER_COOLDOWN
        logger.warning("web_search.engine_paused", extra={"engine": engine, "until": state["paused_until"]})

TOOL_WHITELIST = {t["function"]["name"] for t in TOOLS}
TOOL_CALL_TIMEOUT = float(os.getenv("SEARCHONE_TOOL_TIMEOUT", "8"))
TOOL_HTTP_TIMEOUT = float(os.getenv("SEARCHONE_TOOL_HTTP_TIMEOUT", "10"))
TOOL_METRICS = Counter()
COUNCIL_METRICS = Counter()
LLM_USAGE = Counter()


def _format_evidence_for_prompt(evidence: List[Dict[str, Any]], max_items: int = 3, max_len: int = 320) -> str:
    """Compact evidence list into a few snippets for vote prompts."""
    if not evidence:
        return ""
    lines = []
    for ev in evidence[:max_items]:
        meta = ev.get("meta") or {}
        title = meta.get("title") or meta.get("source") or ""
        domain = meta.get("domain") or ""
        src_type = meta.get("source_type") or meta.get("type") or ""
        header_parts = [p for p in [title, domain, src_type] if p]
        header = " | ".join(header_parts) if header_parts else ""
        snippet = (ev.get("text") or "")[:max_len].replace("\n", " ")
        if header:
            lines.append(f"- {header}: {snippet}")
        else:
            lines.append(f"- {snippet}")
    return "\n".join(lines)


async def index_texts(vs: FaissStore, embedder, texts: List[str], meta_base: Dict[str, Any]):
    """Index arbitrary texts with shared metadata + timestamp for traceability."""
    payload = []
    metas = []
    now = datetime.now(timezone.utc).isoformat()
    for txt in texts:
        if not txt:
            continue
        if not isinstance(txt, str):
            try:
                txt = json.dumps(txt, ensure_ascii=False)
            except Exception:
                txt = str(txt)
        payload.append(txt)
        try:
            meta = dict(meta_base or {})
        except Exception:
            meta = {}
        meta["timestamp"] = now
        # store a truncated version for retrieval without DB (messages/summaries)
        meta["text"] = txt[:2000]
        metas.append(meta)
    if not payload:
        return
    try:
        emb = await asyncio.to_thread(embedder.encode, payload, convert_to_numpy=True, show_progress_bar=False)
        vs.add(emb, metas)
    except Exception as e:
        logger.warning("Indexing failed for texts (meta=%s): %s", meta_base, e, exc_info=True)


class Agent:
    def __init__(self, name: str, role: str, llm: LLMClient, vs: FaissStore, embedder: SentenceTransformer):
        self.name = name
        self.role = role
        self.llm = llm
        self.vs = vs
        self.embedder = embedder
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.state: Dict[str, Any] = {"hypotheses": [], "notes": "", "coverage": {}}
        self.bus: MessageBus = None  # type: ignore
        self.memory = ConversationMemory()
        self.outline_gen = OutlineGenerator()
        self.section_writer = SectionWriter()
        self.refs = ReferenceManager()
        self.style_critic = StyleCritic()
        self.final_composer = FinalComposer()
        self.global_critic = GlobalCritic()

    async def retrieve_evidence(self, query: str, top_k: int = 12):
        """Embed query, search vector store, and hydrate with chunk text. Prefer real docs (pdf/url), skip chat echoes."""
        try:
            emb = await asyncio.to_thread(self.embedder.encode, [query], show_progress_bar=False, convert_to_numpy=True)
            # overfetch then filter/dedupe
            hits = self.vs.search(emb, top_k=top_k * 2)
            logger.info(
                "retrieval.raw_hits",
                extra={
                    "agent": self.name,
                    "query": query[:200],
                    "requested_top_k": top_k,
                    "raw_hits": len(hits),
                },
            )
            filtered = []
            domains_seen = set()
            kept_types = set()
            for h in hits:
                meta = h.get('metadata') or {}
                # Skip echoed chat messages
                if meta.get("type") == "message":
                    continue
                src_type = meta.get("source_type") or meta.get("type")
                reliability = meta.get("reliability") or meta.get("score") or 0
                domain = meta.get("domain")
                # enforce preferred source types and reliability threshold
                if src_type and src_type not in PREFERRED_TYPES:
                    continue
                if MIN_RELIABILITY and reliability and reliability < MIN_RELIABILITY:
                    continue
                # diversity by domain
                if domain:
                    if domain in domains_seen:
                        continue
                    domains_seen.add(domain)
                filtered.append(h)
                if src_type:
                    kept_types.add(src_type)
                if len(filtered) >= top_k:
                    break
            hits = filtered
            results = []
            with get_session() as s:
                for h in hits:
                    meta = h.get('metadata')
                    if not meta:
                        continue
                    # if text is already present in meta (messages/summaries), use it; else hydrate from DB
                    if meta.get("text"):
                        results.append({"score": h.get('score'), "text": meta.get("text"), "meta": meta})
                        continue
                    stmt = select(Chunk).where(Chunk.document_id == meta.get("document_id"), Chunk.chunk_index == meta.get("chunk_index"))
                    ch = s.exec(stmt).first()
                    results.append({"score": h.get('score'), "text": ch.text if ch else None, "meta": meta})
            results = await maybe_rerank(query, results)
            results, coverage = deduplicate_and_score(results)
            # composite score using vector/rerank + reliability if available
            for r in results:
                meta = r.get("meta") or {}
                reliability = meta.get("reliability") or 0.0
                base = r.get("rerank_score", r.get("score", 0.0)) or 0.0
                r["composite_score"] = float(base) * 0.8 + float(reliability) * 0.2
            results = sorted(results, key=lambda x: x.get("composite_score", x.get("score", 0.0)), reverse=True)
            self.state["coverage"] = coverage
            logger.info(
                "retrieval.filtered_hits",
                extra={
                    "agent": self.name,
                    "query": query[:200],
                    "kept": len(results),
                    "coverage": coverage,
                    "domains": list(domains_seen),
                    "types": list(kept_types),
                },
            )
            return results
        except Exception as e:
            logger.exception("Error in retrieve_evidence: %s", e)
            return []
        # If no evidence because index is empty, still return a minimal note
        if not results:
            self.state["coverage"] = {"unique_chunks": 0, "unique_sources": 0, "coverage_score": 0.0}
            return [{"score": None, "text": "[no local evidence indexed]", "meta": {}}]

    async def propose(self, prompt_context: str, extra_context: str = ""):
        ctx = prompt_context
        if extra_context:
            ctx = f"{prompt_context}\n\nContext:\n{extra_context}"
        mem_summary = self.memory.summarize()
        if mem_summary:
            ctx = f"{ctx}\n\nDerniers echanges internes:\n{mem_summary}"
        prompt = propose_prompt(self.name, self.role, ctx)
        resp = await call_llm_with_retry(self.llm, prompt, max_tokens=256, label=f"{self.name}-propose", tools=TOOLS, tool_choice="auto")
        base_txt = resp.get("text") if isinstance(resp, dict) else resp
        txt = base_txt
        tool_calls = resp.get("tool_calls", []) if isinstance(resp, dict) else []
        evidence_from_tools = []
        for tc in tool_calls or []:
            ev = await self._execute_tool_call(tc)
            if ev:
                evidence_from_tools.extend(ev)
        # if tools returned evidence, reinject into a follow-up turn so the hypothesis uses them explicitly
        if evidence_from_tools:
            tool_ctx = _format_evidence_for_prompt(evidence_from_tools, max_items=5, max_len=240)
            followup_prompt = (
                f"{prompt}\n\nResultats d'outils disponibles (utilise-les pour ajuster/resserrer l'hypothese en 1 phrase):\n{tool_ctx}"
            )
            try:
                followup = await call_llm_with_retry(self.llm, followup_prompt, max_tokens=192, label=f"{self.name}-propose-followup")
                txt = followup.get("text") if isinstance(followup, dict) else followup or base_txt
            except Exception as e:
                logger.warning("Follow-up LLM after tools failed (%s): %s", self.name, e, exc_info=True)
        self.state['hypotheses'].append(txt)
        self.memory.add("assistant", txt or "")
        return txt, evidence_from_tools

    def _score_sources(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute a rough quality score per source for curation/explorer roles."""
        sources: Dict[str, Dict[str, Any]] = {}
        for ev in evidence or []:
            meta = ev.get("meta") or {}
            source = meta.get("source") or meta.get("domain") or "unknown"
            if source not in sources:
                sources[source] = {
                    "domain": meta.get("domain") or "",
                    "type": meta.get("source_type") or meta.get("type") or "",
                    "reliability": float(meta.get("reliability") or meta.get("score") or 0.0),
                    "latest": meta.get("published_at"),
                    "snippet": (ev.get("text") or "")[:200],
                    "score": 0.0,
                }
            base = ev.get("composite_score") or ev.get("rerank_score") or ev.get("score") or 0.0
            reliability = float(meta.get("reliability") or meta.get("score") or 0.0)
            sources[source]["score"] = max(sources[source]["score"], float(base) * 0.6 + reliability * 0.4)
        ranked = sorted(sources.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        self.state["source_ratings"] = ranked
        return ranked

    async def _build_curation_report(self, evidence: List[Dict[str, Any]]):
        """Produce a raw curation report with tagged sources and mini-summaries."""
        ratings = self._score_sources(evidence)
        lines = []
        for r in ratings[:8]:
            tags = []
            if r.get("type"):
                tags.append(r["type"])
            if r.get("domain"):
                tags.append(r["domain"])
            tag_str = ", ".join(tags)
            lines.append(f"- {tag_str or 'source'} | rel={round(r.get('reliability', 0), 3)} | {r.get('snippet')}")
        report = "\n".join(lines)
        self.state["curation_report"] = report
        if self.bus:
            self.bus.publish("curation", {"agent": self.name, "report": report, "ratings": ratings})
        return report

    async def _derive_claims(self, hypothesis: str, evidence: List[Dict[str, Any]]):
        """Ask the LLM to synthesize explicit claims grounded in the gathered evidence."""
        ev_ctx = _format_evidence_for_prompt(evidence, max_items=6, max_len=280)
        if not ev_ctx:
            return []
        prompt = (
            "Synthese rapide pour l'agent Analyste.\n"
            "Hypothese courante:\n"
            f"{hypothesis}\n\n"
            "Evidence (selection):\n"
            f"{ev_ctx}\n\n"
            "Produis 3 claims factuels, numerotes, chacun avec 1 justification courte. Format: '- Claim -- justification'."
        )
        claims_txt = await call_llm_with_retry(self.llm, prompt, max_tokens=220, label=f"{self.name}-claims")
        text = claims_txt.get("text") if isinstance(claims_txt, dict) else claims_txt
        claims = []
        if isinstance(text, str):
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line[0].isdigit() and "." in line:
                    line = line.split(".", 1)[1].strip()
                if line.startswith("-"):
                    line = line.lstrip("-").strip()
                claims.append(line)
        self.state["claims"] = claims
        if self.bus:
            self.bus.publish("claims", {"agent": self.name, "claims": claims, "hypothesis": hypothesis})
        return claims

    async def _build_experiment_plan(self, hypothesis: str, evidence: List[Dict[str, Any]]):
        """Translate a hypothesis into a minimal experiment protocol and metrics."""
        ev_ctx = _format_evidence_for_prompt(evidence, max_items=4, max_len=200)
        prompt = (
            "Plan d'experimentation rapide.\n"
            f"Hypothese: {hypothesis}\n"
            f"Elements disponibles:\n{ev_ctx}\n\n"
            "Propose un protocole court: type de test (simulation/analyse), donnees/inputs, metriques clefs, critere de succes."
        )
        resp = await call_llm_with_retry(self.llm, prompt, max_tokens=200, label=f"{self.name}-exp-plan")
        plan = resp.get("text") if isinstance(resp, dict) else resp
        self.state["experiment_plan"] = plan
        return plan

    async def _build_experiment_journal(self, hypothesis: str, plan: str, evidence: List[Dict[str, Any]]):
        """Assemble a short experimental journal entry from plan + experiment outputs if present."""
        experiment_outputs = [ev for ev in evidence if (ev.get("meta") or {}).get("source_type") == "experiment"]
        lines = [f"Hypothese: {hypothesis}", f"Plan: {plan or '[non defini]'}"]
        if experiment_outputs:
            for ev in experiment_outputs[:3]:
                lines.append(f"Resultat: {ev.get('text')}")
        journal = "\n".join(lines)
        self.state["experiment_journal"] = journal
        if self.bus:
            self.bus.publish("experiments", {"agent": self.name, "journal": journal, "outputs": experiment_outputs})
        return journal

    async def _draft_outline(self, hypothesis: str, evidence: List[Dict[str, Any]], shared_context: str = ""):
        """Produce a lightweight outline for the Redacteur based on current findings."""
        ev_ctx = _format_evidence_for_prompt(evidence, max_items=5, max_len=200)
        prompt = (
            "Esquisse IMRaD courte pour le redacteur.\n"
            f"Contexte: {shared_context}\n"
            f"Hypothese clef: {hypothesis}\n"
            f"Evidence: \n{ev_ctx}\n\n"
            "Donne un plan avec 4 sections max (Intro, Methode, Resultats attendus, Discussion rapide)."
        )
        resp = await call_llm_with_retry(self.llm, prompt, max_tokens=180, label=f"{self.name}-outline")
        outline = resp.get("text") if isinstance(resp, dict) else resp
        self.state["outline"] = outline
        return outline

    def _collect_references(self, evidence: List[Dict[str, Any]]):
        """Build a simple reference list from evidence metadata."""
        refs_added = []
        for ev in evidence or []:
            meta = ev.get("meta") or {}
            title = meta.get("title") or meta.get("source") or ""
            if not title:
                continue
            author = meta.get("author") or ""
            year = ""
            if meta.get("published_at"):
                year = str(meta.get("published_at"))[:4]
            ref = self.refs.add(title=title, author=author, year=year, doi=meta.get("doi", ""), url=meta.get("source", ""))
            refs_added.append(ref)
        if refs_added and self.bus:
            self.bus.publish("references", {"agent": self.name, "references": refs_added})
        if refs_added:
            self.state["references"] = self.refs.list()
            self.state["bibliography"] = self.refs.bibliography()
            for ref in refs_added:
                promote_knowledge(ref.get("title") or ref.get("url") or "", reason="reference_cited", score=ref.get("reliability", 1.0) if isinstance(ref, dict) else 1.0)

    async def _suggest_tests(self, hypothesis: str, evidence: List[Dict[str, Any]], claims: List[str]):
        """Generate quick test ideas/variables for the hypothesis."""
        ev_ctx = _format_evidence_for_prompt(evidence, max_items=4, max_len=160)
        claim_ctx = "\n".join([f"- {c}" for c in claims[:4]]) if claims else ""
        prompt = (
            f"Hypothese: {hypothesis}\n"
            f"Claims supports:\n{claim_ctx}\n"
            f"Evidence:\n{ev_ctx}\n\n"
            "Propose 2 tests/mesures concretes (variables + methode) pour verifier l'hypothese."
        )
        resp = await call_llm_with_retry(self.llm, prompt, max_tokens=180, label=f"{self.name}-tests")
        txt = resp.get("text") if isinstance(resp, dict) else resp
        self.state["hypothesis_tests"] = txt
        if self.bus:
            self.bus.publish("tests", {"agent": self.name, "hypothesis": hypothesis, "tests": txt})
        return txt

    async def act(self, shared_context: str, council_inbox: asyncio.Queue):
        # main agent loop iteration: propose -> retrieve -> enrich -> send evidence to council
        context_snippets = await self.retrieve_context_texts(shared_context, top_k=3)
        extra_ctx = "\n".join([f"- {c}" for c in context_snippets]) if context_snippets else ""
        claims_for_hypo: List[str] = []
        if self.bus:
            tasks = self.bus.peek("tasks")
            if tasks:
                latest = tasks[-1]
                tq = latest.get("query")
                if tq and tq != shared_context:
                    extra_ctx = f"- Task: {tq}\n" + extra_ctx
            claims_for_hypo: List[str] = []
            # surface recent claims to the Hypothesis agent for grounding
            if self.role == "Hypothesis":
                claims_msgs = self.bus.peek("claims")
                if claims_msgs:
                    claim_lines = []
                    for m in claims_msgs[-3:]:
                        for c in m.get("claims", []) or []:
                            claim_lines.append(f"- {c}")
                            claims_for_hypo.append(c)
                    if claim_lines:
                        extra_ctx = f"{extra_ctx}\nClaims recents:\n" + "\n".join(claim_lines)
        hypo, tool_evidence = await self.propose(shared_context, extra_context=extra_ctx)
        # run multiple evidence calls in parallel (hypothesis + original query) to enrich the note
        tasks = [
            self.retrieve_evidence(hypo, top_k=3),
            self.retrieve_evidence(shared_context, top_k=3),
        ]
        ev_lists = await asyncio.gather(*tasks, return_exceptions=False)
        evidence = []
        evidence.extend(tool_evidence)
        for ev in ev_lists:
            evidence.extend(ev or [])
        # fallback retrieval-only phase if still no evidence
        if not evidence:
            more = await self.retrieve_evidence(shared_context, top_k=6)
            if more:
                logger.info("retrieval.fallback_success", extra={"agent": self.name, "count": len(more)})
                evidence.extend(more)
            else:
                logger.info("retrieval.fallback_empty", extra={"agent": self.name})
        # retrieval-only boost via tool-calls when evidence is below the configured threshold
        if RETRIEVAL_ONLY_THRESHOLD and len(evidence) < RETRIEVAL_ONLY_THRESHOLD:
            search_queries = [shared_context]
            if hypo:
                search_queries.append(hypo)
            tool_tasks = []
            for q in search_queries:
                tc = {"function": {"name": "search_hybrid", "arguments": json.dumps({"query": q, "top_k": RETRIEVAL_ONLY_TOPK})}}
                tool_tasks.append(self._execute_tool_call(tc))
            try:
                tool_results = await asyncio.gather(*tool_tasks)
                added = 0
                for evs in tool_results:
                    if evs:
                        evidence.extend(evs)
                        added += len(evs)
                logger.info(
                    "retrieval.only_phase",
                    extra={"agent": self.name, "threshold": RETRIEVAL_ONLY_THRESHOLD, "added": added},
                )
            except Exception as e:
                logger.warning("retrieval-only tool phase failed (%s): %s", self.name, e, exc_info=True)
        # role-specific enrichments
        if self.role == "Explorer":
            self._score_sources(evidence)
        if self.role == "Curator":
            await self._build_curation_report(evidence)
        if self.role == "Analyst":
            await self._derive_claims(hypo, evidence)
        if self.role == "Experimenter":
            plan = await self._build_experiment_plan(hypo, evidence)
            await self._build_experiment_journal(hypo, plan, evidence)
        if self.role == "Hypothesis":
            claims = claims_for_hypo or self.state.get("claims") or []
            await self._suggest_tests(hypo, evidence, claims)
        if self.role == "Redacteur":
            outline = await self._draft_outline(hypo, evidence, shared_context)
            sections = self.outline_gen.generate(shared_context or hypo, self.state.get("claims") or [], [e.get("text") or "" for e in evidence[:3]])
            plan = self.outline_gen.select_plan(sections)
            drafted_sections = []
            inline_cites = [self.refs.cite_inline(i) for i in range(len(self.refs.list()))]
            for idx, sec in enumerate(plan.get("sections", [])[:4]):
                cites = [inline_cites[idx]] if idx < len(inline_cites) else inline_cites[:1]
                drafted_sections.append(
                    self.section_writer.draft_section(
                        sec.get("title", ""),
                        sec.get("bullets", []),
                        claims=self.state.get("claims") or [],
                        evidence_snippets=[(e.get("text") or "")[:120] for e in evidence[:2]],
                        citations=cites,
                    )
                )
            self.state["outline_plan"] = plan
            self.state["draft_sections"] = drafted_sections
            critiques = [self.style_critic.critique(s) for s in drafted_sections]
            self.state["style_feedback"] = critiques
            if self.bus:
                self.bus.publish("drafts", {"agent": self.name, "outline": outline, "sections": drafted_sections, "bibliography": self.refs.bibliography(), "style_feedback": critiques})
        if self.role == "Critic":
            drafts = self.bus.peek("drafts") if self.bus else []
            if drafts:
                latest = drafts[-1]
                sections = latest.get("sections") or []
                feedback = [self.style_critic.critique(s) for s in sections]
                review = {
                    "agent": self.name,
                    "summary": "Validation finale (heuristique) sur les sections redigees.",
                    "issues": sum((f.get("issues", []) for f in feedback), []),
                    "scores": [f.get("score") for f in feedback],
                }
                self.state["final_review"] = review
                if self.bus:
                    self.bus.publish("final_review", review)
                # compose final article + resume
                composed = self.final_composer.compose(shared_context or "Resultats de recherche", sections, self.refs.bibliography())
                self.state["final_article"] = composed.get("article")
                self.state["final_summary"] = composed.get("summary")
                # global critique on final article
                global_review = self.global_critic.review(
                    composed.get("article") or "",
                    self.state.get("claims") or [],
                    self.state.get("decision") or {},
                )
                self.state["global_review"] = global_review
                if self.bus:
                    self.bus.publish("global_review", global_review)
                if self.bus:
                    self.bus.publish("final_documents", composed)
        # collect references regardless of role
        self._collect_references(evidence)

        note = {"agent": self.name, "role": self.role, "hypothesis": hypo, "evidence": evidence}
        # persist in local state
        coverage_score = self.state.get("coverage", {}).get("coverage_score")
        cv = f", coverage={coverage_score:.2f}" if coverage_score is not None else ""
        self.state['notes'] = f"Last run found {len(evidence)} evidences{cv}"
        self.memory.add("system", f"Evidence count={len(evidence)}, coverage={coverage_score}")
        if self.bus:
            self.bus.publish("notes", note)
        # send to council
        await council_inbox.put(note)
        if self.bus:
            self.bus.publish("to_council", note)

    async def plan(self, observation: str) -> str:
        """Minimal planning helper for interface compatibility."""
        return f"Focus on query: {observation}"

    async def reflect(self) -> str:
        """Return last hypothesis for introspection/debug."""
        if self.state.get("hypotheses"):
            return self.state["hypotheses"][-1]
        return ""

    async def retrieve_context_texts(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve contextual snippets from the vector store (any type) for prompt enrichment."""
        try:
            emb = await asyncio.to_thread(self.embedder.encode, [query], show_progress_bar=False, convert_to_numpy=True)
            hits = self.vs.search(emb, top_k=top_k)
            snippets = []
            with get_session() as s:
                for h in hits:
                    meta = h.get('metadata') or {}
                    txt = meta.get("text")
                    if not txt and meta.get("document_id") is not None:
                        stmt = select(Chunk).where(Chunk.document_id == meta.get('document_id'), Chunk.chunk_index == meta.get('chunk_index'))
                        ch = s.exec(stmt).first()
                        txt = ch.text if ch else None
                    if txt:
                        snippets.append(txt[:800])
            return snippets
        except Exception as e:
            logger.warning("Context retrieval failed: %s", e, exc_info=True)
            return []


class Council:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.inbox: asyncio.Queue = asyncio.Queue()

    def drain_messages(self) -> List[Dict[str, Any]]:
        msgs = []
        while not self.inbox.empty():
            try:
                msgs.append(self.inbox.get_nowait())
            except Exception:
                break
        return msgs

    async def meet_and_summarize(self, agents: List[Agent], msgs: List[Dict[str, Any]] = None, job_id: int = None) -> Dict[str, Any]:
        # collect all messages currently available
        msgs = msgs if msgs is not None else self.drain_messages()
        if not msgs:
            logger.info("council.no_messages", extra={"job": job_id})
            return {"summary": "no_messages", "messages": [], "votes": []}
        if COUNCIL_MAX_MSGS and len(msgs) > COUNCIL_MAX_MSGS:
            msgs = msgs[:COUNCIL_MAX_MSGS]
            logger.debug("Council trimming messages to %d", len(msgs))

        # First: solicit votes from each agent for each hypothesis
        votes = []
        vote_start = time.time()
        for m in msgs:
            hyp = m.get('hypothesis')
            per_message_votes = {"hypothesis": hyp, "votes": []}
            coros = []
            for idx, agent in enumerate(agents):
                if COUNCIL_MAX_VOTES_PER_MSG and idx >= COUNCIL_MAX_VOTES_PER_MSG:
                    break
                ev_ctx = _format_evidence_for_prompt(m.get("evidence") or [])
                coros.append(call_llm_with_retry(agent.llm, vote_prompt(agent.name, agent.role, hyp, ev_ctx), max_tokens=80, label=f"{agent.name}-vote"))
            vote_texts = await asyncio.gather(*coros) if coros else []
            for agent_obj, vtxt in zip(agents, vote_texts):
                if isinstance(vtxt, dict):
                    vtxt = vtxt.get("text") or ""
                if vtxt is None:
                    vtxt = ""
                # parse simple 'VOTE -- justification' pattern
                if '--' in vtxt:
                    v, just = vtxt.split('--', 1)
                    v = v.strip().lower()
                    just = just.strip()
                else:
                    parts = vtxt.strip().split('\n', 1)
                    v = parts[0].strip().lower() if parts else 'neutral'
                    just = parts[1].strip() if len(parts) > 1 else ''
                if v not in ('agree', 'neutral', 'disagree'):
                    # try to map common variants
                    if v.startswith('a'):
                        v = 'agree'
                    elif v.startswith('d'):
                        v = 'disagree'
                    else:
                        v = 'neutral'
                per_message_votes['votes'].append({"agent": agent_obj.name, "vote": v, "justification": just})
            votes.append(per_message_votes)
        logger.info(
            "council.votes",
            extra={
                "job": job_id,
                "messages": len(msgs),
                "votes": sum(len(v.get('votes', [])) for v in votes),
                "duration": round(time.time() - vote_start, 3),
            },
        )
        COUNCIL_METRICS["votes"] += sum(len(v.get('votes', [])) for v in votes)
        COUNCIL_METRICS["messages_voted"] += len(msgs)

        # build prompt for summary including votes
        prompt_parts = [f"Agent {m['agent']} ({m['role']}): hypothesis: {m['hypothesis']}. Evidence count: {len(m['evidence'])}" for m in msgs]
        vote_parts = []
        for v in votes:
            vote_parts.append(f"Hypothesis: {v['hypothesis']}")
            for vv in v['votes']:
                vote_parts.append(f"- {vv['agent']}: {vv['vote']} ({vv['justification']})")

        prompt = "\n".join(prompt_parts + ["\nVotes:"] + vote_parts) + "\nSynthese des accords, contradictions et prochaines etapes courtes."
        try:
            with start_span("council.summary.llm"):
                summary = await self.llm.generate(prompt, max_tokens=512)
            logger.info(
                "council.summary",
                extra={
                    "job": job_id,
                    "messages": len(msgs),
                    "votes": sum(len(v.get('votes', [])) for v in votes),
                    "prompt_tokens_approx": len(prompt) // 4,
                },
            )
            COUNCIL_METRICS["summaries"] += 1
        except Exception as e:
            logger.exception("LLM error in council: %s", e)
            summary = f"[LLM error: {e}]"
        return {"summary": summary, "messages": msgs, "votes": votes}


async def run_agents_job(job_id: int, query: str, max_iterations: int = 5, roles: List[str] = None, llm_client: LLMClient = None, embedder: SentenceTransformer = None, vs: FaissStore = None, max_duration_seconds: int = 300, max_token_budget: int = 0, bus: MessageBus = None, run_ctx: RunContext = None, controller: ConvergenceController = None):
    roles = roles or DEFAULT_ROLES
    logger.info("Starting agents job %s with roles %s", job_id, roles)
    # Always create a fresh LLM client per run to avoid stale semaphores / event loops
    base_llm = llm_client or LLMClient()
    LLM_USAGE.clear()
    bus = bus or MessageBus()
    run_ctx = run_ctx or RunContext(job_id=job_id, query=query, roles=roles)
    rscore = ResearchScore()
    bus.publish("tasks", {"query": query, "job_id": job_id})
    def _llm_for_model(model_id: str) -> LLMClient:
        if not isinstance(base_llm, LLMClient):
            return base_llm  # fallback mock
        base_model = getattr(base_llm, "model", None)
        if not model_id or model_id == base_model:
            return base_llm
        return LLMClient(
            api_key=getattr(base_llm, "api_key", None),
            model=model_id,
            use_sdk=getattr(base_llm, "use_sdk", False),
            fallback_local=getattr(base_llm, "fallback_local", True),
        )
    start_ts = time.time()
    def _read_timeout(env_name: str, default: float) -> float:
        try:
            val = float(os.getenv(env_name, str(default)))
            return val
        except Exception:
            return default

    # If you really want no timeout, set env to 0. Default keeps jobs from hanging indefinitely.
    ACT_TIMEOUT = _read_timeout("SEARCHONE_ACT_TIMEOUT", 120)
    COUNCIL_TIMEOUT = _read_timeout("SEARCHONE_COUNCIL_TIMEOUT", 120)

    # allow injection of lightweight mocks for embedder and vector store (useful for tests/debug)
    if embedder is None:
        if _HAS_ST and SentenceTransformer is not None:
            embedder = SentenceTransformer(EMBEDDING_MODEL)
        else:
            embedder = SimpleEmbedder()
    if vs is None:
        vs = FaissStore(dim=embedder.get_sentence_embedding_dimension())

    agents = []
    agent_specs: List[AgentSpec] = []
    high_risk = is_high_risk(query)
    for i in range(len(roles)):
        role = roles[i % len(roles)]
        model_id = resolve_model(role)
        agent_llm = _llm_for_model(model_id)
        agent = Agent(name=f"A{i+1}", role=role, llm=agent_llm, vs=vs, embedder=embedder)
        agent.state["model"] = model_id
        agent.bus = bus
        base_tools = ROLE_ALLOWED_TOOLS.get(role, TOOL_WHITELIST)
        agent.allowed_tools = set(filter_tools(base_tools, allow_web=not high_risk))
        spec = AgentSpec(
            name=agent.name,
            role_description=role,
            capabilities=["propose", "retrieve_evidence", "vote"],
            model_profile=model_id,
            memory_scopes=["session", "job"],
            allowed_tools=list(agent.allowed_tools),
        )
        agent_specs.append(spec)
        agents.append(agent)
    council = Council(llm=base_llm)

    state = {
        "job_id": job_id,
        "query": query,
        "agents": {},
        "council": {},
        "timeline": [],
        "token_spent": 0,
        "web_requests": 0,
        "usage": {},
        "agent_specs": [spec.to_dict() for spec in agent_specs],
    }
    for agent in agents:
        agent.job_state = state
        agent.job_id = job_id

    if high_risk:
        audit = audit_entry("high_risk_query", {"query": query})
        state.setdefault("audit", []).append(audit)
        try:
            log_decision(job_id, 0, audit)
        except Exception:
            pass
    controller = controller or ConvergenceController(window=STAGNATION_WINDOW, min_delta=STAGNATION_MIN_DELTA)
    preflight = {}
    try:
        async with httpx.AsyncClient(timeout=HEALTHCHECK_TIMEOUT) as client:
            try:
                resp_src = await client.get("http://127.0.0.1:2001/sources/summary")
                resp_src.raise_for_status()
                preflight["sources_summary"] = resp_src.json()
            except Exception as e:
                preflight["sources_summary"] = {"error": str(e)}
            try:
                resp_doc = await client.get("http://127.0.0.1:2001/doctor/index/status")
                resp_doc.raise_for_status()
                preflight["doctor_status"] = resp_doc.json()
            except Exception as e:
                preflight["doctor_status"] = {"error": str(e)}
    except Exception as e:
        preflight["error"] = str(e)
    state["preflight"] = preflight
    def _doctor_index_empty(payload: Dict[str, Any]) -> bool:
        try:
            if not isinstance(payload, dict):
                return False
            idx = payload.get("index") or {}
            return (idx.get("ntotal") or idx.get("emb_count") or 0) == 0
        except Exception:
            return False
    preflight_index_empty = _doctor_index_empty(preflight.get("doctor_status", {}))
    # if index empty, continue but log warning (no fail-fast)
    try:
        stats = vs.stats()
        state["preflight"]["faiss_stats"] = stats
        if preflight_index_empty and (stats.get("ntotal") or stats.get("emb_count") or 0) == 0:
            logger.warning("Job %s proceeding with empty index (doctor/index empty)", job_id)
        if (stats.get("ntotal") or stats.get("emb_count") or 0) == 0:
            # try auto seed if configured
            if AUTO_SEED_CORPUS:
                try:
                    resp = requests.post(AUTO_SEED_ENDPOINT, params={"name": AUTO_SEED_CORPUS}, timeout=60)
                    logger.info("Auto seed triggered (corpus=%s): status=%s", AUTO_SEED_CORPUS, resp.status_code)
                    time.sleep(2.0)
                    stats = vs.stats()
                except Exception as e:
                    logger.warning("Auto seed failed: %s", e)
            if (stats.get("ntotal") or stats.get("emb_count") or 0) == 0:
                logger.warning("Job %s continuing with empty index (no prior data)", job_id)
    except Exception:
        pass
    # index the query itself for later retrieval/trace
    await index_texts(vs, embedder, [query], {"type": "query", "job_id": job_id})
    # persist an initial checkpoint so history is never empty even if a failure happens early
    save_job_state(job_id, json_str(state), status='running')
    await asyncio.sleep(0.1)
    failure_count = 0
    max_failures = 3
    empty_iters = 0

    for it in range(max_iterations):
        if is_stop_requested(job_id):
            save_job_state(job_id, json_str(state), status='failed_stopped')
            clear_stop(job_id)
            logger.info("Job %s received stop flag (redis), aborting loop", job_id)
            return state
        elapsed = time.time() - start_ts
        if max_duration_seconds and elapsed > max_duration_seconds:
            # stop gracefully on timeout to avoid runaway jobs
            save_job_state(job_id, json_str(state), status='stalled_timeout')
            logger.warning("Job %s timed out after %.1fs (limit=%ss)", job_id, elapsed, max_duration_seconds)
            clear_stop(job_id)
            return state
        if AGENT_TOKEN_BUDGET and state.get("token_spent", 0) >= AGENT_TOKEN_BUDGET:
            save_job_state(job_id, json_str(state), status='failed_cost')
            logger.warning("Job %s stopped: agent token budget exceeded (%s)", job_id, AGENT_TOKEN_BUDGET)
            clear_stop(job_id)
            return state
        if WEB_REQUEST_BUDGET and state.get("web_requests", 0) >= WEB_REQUEST_BUDGET:
            save_job_state(job_id, json_str(state), status='failed_web_budget')
            logger.warning("Job %s stopped: web request budget exceeded (%s)", job_id, WEB_REQUEST_BUDGET)
            clear_stop(job_id)
            return state
        # check for external pause/stop commands
        try:
            with get_session() as s:
                job_row = s.get(Job, job_id)
                if job_row and job_row.status == 'paused':
                    save_job_state(job_id, json_str(state), status='paused')
                    logger.info("Job %s paused externally, stopping loop", job_id)
                    clear_stop(job_id)
                    return state
                if job_row and job_row.status == 'failed':
                    logger.info("Job %s marked failed externally, stopping loop", job_id)
                    clear_stop(job_id)
                    return state
        except Exception:
            pass
        if max_token_budget and state.get("token_spent", 0) >= max_token_budget:
            save_job_state(job_id, json_str(state), status='failed_cost')
            logger.warning("Job %s stopped due to token budget exceeded (%s)", job_id, max_token_budget)
            clear_stop(job_id)
            return state
        logger.info("Job %s iteration %d", job_id, it + 1)
        iter_start = time.time()
        # let agents act concurrently
        tasks = [agent.act(query, council.inbox) for agent in agents]
        try:
            if ACT_TIMEOUT and ACT_TIMEOUT > 0:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=ACT_TIMEOUT)
            else:
                await asyncio.gather(*tasks)
            # persist after agents actions so partial work is visible
            logger.info("Job %s iteration %d agents done in %.2fs", job_id, it + 1, time.time() - iter_start)
            save_job_state(job_id, json_str(state), status='running')
        except Exception as e:
            failure_count += 1
            state['last_error'] = f"agents_act: {e}"
            logger.exception("Error during agents act in iteration %s: %s", it + 1, e)
            save_job_state(job_id, json_str(state), status='running')
            if failure_count >= max_failures:
                save_job_state(job_id, json_str(state), status='failed')
                clear_stop(job_id)
                return state
            continue

        # collect messages once and reuse to avoid losing them on timeout
        drained_msgs = council.drain_messages()
        total_evidence = sum(len(m.get('evidence') or []) for m in drained_msgs)
        if total_evidence == 0:
            empty_iters += 1
        else:
            empty_iters = 0
        for m in drained_msgs:
            run_ctx.record_message(sender=m.get("agent"), payload=m)
        run_ctx.track_iteration(
            coverage=sum((m.get("meta", {}) or {}).get("reliability", 0.0) for m in (drained_msgs or [])) / (len(drained_msgs) or 1),
            evidence_count=total_evidence,
            hypotheses=[m.get("hypothesis", "") for m in drained_msgs],
        )
        controller.record_iteration(
            coverage=sum((m.get("meta", {}) or {}).get("reliability", 0.0) for m in (drained_msgs or [])) / (len(drained_msgs) or 1),
            evidence_count=total_evidence,
            hypotheses=[m.get("hypothesis", "") for m in drained_msgs],
        )
        rscore.update(
            evidence_count=total_evidence,
            unique_sources=len({(m.get("meta", {}) or {}).get("source") for m in drained_msgs if m.get("meta")}),
            hypotheses=[m.get("hypothesis", "") for m in drained_msgs],
        )
        if MAX_EMPTY_ITERS and empty_iters >= MAX_EMPTY_ITERS:
            state['last_error'] = "no_evidence"
            save_job_state(job_id, json_str(state), status='failed_no_evidence')
            logger.warning("Job %s stopped: no evidence after %d consecutive iterations", job_id, empty_iters)
            clear_stop(job_id)
            return state

        # council meets
        try:
            if COUNCIL_TIMEOUT and COUNCIL_TIMEOUT > 0:
                summary = await asyncio.wait_for(council.meet_and_summarize(agents, msgs=drained_msgs, job_id=job_id), timeout=COUNCIL_TIMEOUT)
            else:
                summary = await council.meet_and_summarize(agents, msgs=drained_msgs, job_id=job_id)
            logger.info("Job %s iteration %d council done in %.2fs", job_id, it + 1, time.time() - iter_start)
            # organize a debate snapshot for the coordinator
            try:
                arguments = [f"pro: {m.get('hypothesis')}" for m in drained_msgs]
                critics = []
                for v in summary.get("votes", []):
                    for vv in v.get("votes", []):
                        if vv.get("vote") == "disagree":
                            critics.append(vv.get("justification") or "")
                debate_round = run_debate(question=query, arguments=arguments, critics=critics)
                state.setdefault("debates", []).append(debate_round.to_dict())
                if bus:
                    bus.publish("debate", debate_round.to_dict())
            except Exception:
                pass
        except asyncio.TimeoutError as e:
            # fallback: if timeout, try a quick local summary (no votes) to keep UI populated
            summary_text = "\n".join([f"- {m.get('agent')}: {m.get('hypothesis')}" for m in drained_msgs]) or "[council timeout]"
            summary = {"summary": summary_text, "messages": drained_msgs, "votes": []}
            state['last_error'] = "council_timeout"
            logger.warning("Council timeout for job %s iter %d after %.1fs; using fallback summary", job_id, it + 1, COUNCIL_TIMEOUT)
            save_job_state(job_id, json_str(state), status='running')
            COUNCIL_METRICS["timeouts"] += 1
        except Exception as e:
            failure_count += 1
            state['last_error'] = f"council: {e}"
            logger.exception("Error during council meeting iteration %s: %s", it + 1, e)
            save_job_state(job_id, json_str(state), status='running')
            if failure_count >= max_failures:
                save_job_state(job_id, json_str(state), status='failed')
                clear_stop(job_id)
                return state
            summary = {"summary": f"[error council: {e}]", "messages": [], "votes": []}
        state['council'][f'iter_{it+1}'] = summary
        # debate round representation
        debate_round = DebateRound(
            question=query,
            arguments_for=[m.get("hypothesis") for m in summary.get("messages", []) if m.get("hypothesis")],
            arguments_against=[v.get("justification") for vb in summary.get("votes", []) for v in vb.get("votes", []) if (v.get("vote") or "").startswith("disagree")],
            critics=[v.get("agent") for vb in summary.get("votes", []) for v in vb.get("votes", []) if (v.get("vote") or "").startswith("disagree")],
            summary=summary.get("summary") or "",
        )
        state.setdefault("debate_rounds", {})[f"iter_{it+1}"] = debate_round.to_dict()
        state.setdefault("votes", {})[f"iter_{it+1}"] = tally_votes(summary.get("votes", []))

        # build timeline entry for this iteration
        timeline_entry = {
            'iteration': it + 1,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'messages': summary.get('messages', []),
            'votes': summary.get('votes', []),
            'summary': summary.get('summary')
        }
        state['timeline'].append(timeline_entry)
        run_ctx.add_event("council_summary", timeline_entry)
        if bus:
            bus.publish("summary", timeline_entry)
        # structured decision logging (for dashboard/audit)
        try:
            log_decision(job_id, it + 1, timeline_entry)
        except Exception:
            pass
        # index messages and summary for future search (temporal trace)
        msg_texts = [m.get('hypothesis') for m in summary.get('messages', []) if m.get('hypothesis')]
        await index_texts(vs, embedder, msg_texts, {"type": "message", "job_id": job_id, "iteration": it + 1})
        if summary.get('summary'):
            await index_texts(vs, embedder, [summary.get('summary')], {"type": "summary", "job_id": job_id, "iteration": it + 1})

        # token accounting: prefer usage returned by LLM, fallback to naive length estimate
        token_spent = int(LLM_USAGE.get("total_tokens", 0))
        if not token_spent:
            token_spent = state.get("token_spent", 0)
            for agent in agents:
                for h in agent.state.get('hypotheses', []):
                    token_spent += len(h) // 4 + 1
            if summary.get('summary'):
                token_spent += len(summary.get('summary')) // 4 + 1
        state['token_spent'] = token_spent
        state['usage'] = {
            "prompt_tokens": int(LLM_USAGE.get("prompt_tokens", 0)),
            "completion_tokens": int(LLM_USAGE.get("completion_tokens", 0)),
            "total_tokens": int(LLM_USAGE.get("total_tokens", 0)),
            "calls": int(LLM_USAGE.get("calls", 0)),
            "cost_estimate_usd": float(round(LLM_USAGE.get("cost_usd", 0.0), 6)),
        }
        if max_token_budget and token_spent >= max_token_budget:
            save_job_state(job_id, json_str(state), status='failed_cost')
            logger.warning("Job %s stopped due to token budget exceeded (%s)", job_id, max_token_budget)
            clear_stop(job_id)
            return state

        # broadcast summary to agents inboxes
        for agent in agents:
            await agent.inbox.put(summary)
            state['agents'][agent.name] = agent.state

        # attach tool/council metrics for observability
        state["tool_metrics"] = dict(TOOL_METRICS)
        state["council_metrics"] = dict(COUNCIL_METRICS)
        state["run_context"] = run_ctx.to_dict()
        state["run_metrics"] = compute_run_metrics(state)
        state["message_bus"] = bus.snapshot()
        state["research_score"] = rscore.to_dict()
        # publish metrics for coordinator/monitoring
        if bus:
            bus.publish("metrics", {"run_metrics": state["run_metrics"], "research_score": state["research_score"]})
        # promote adopted hypotheses to knowledge store
        for iter_key, vote_info in (state.get("votes") or {}).items():
            if vote_info.get("decision") == "adopt":
                for msg in timeline_entry.get("messages", []):
                    hypo = msg.get("hypothesis")
                    if hypo:
                        promote_knowledge(hypo, reason="vote_adopt")

        stagnation_reason = controller.check()
        if stagnation_reason:
            state['last_error'] = stagnation_reason
            save_job_state(job_id, json_str(state), status='failed_stagnation')
            logger.warning("Job %s stopped due to %s", job_id, stagnation_reason)
            clear_stop(job_id)
            return state

        # auto-stop criteria driven by votes
        vote_key = f"iter_{it+1}"
        vote_info = (state.get("votes") or {}).get(vote_key, {})
        scores = vote_info.get("scores") or {}
        decision = vote_info.get("decision") or ""
        if decision == "adopt" and scores.get("agree", 0.0) >= AUTO_STOP_AGREE:
            state["stop_reason"] = "auto_stop_agree"
            save_job_state(job_id, json_str(state), status='completed_decided')
            clear_stop(job_id)
            logger.info("Job %s stopped (consensus agree >= %.2f)", job_id, AUTO_STOP_AGREE)
            return state
        if decision == "reject" and scores.get("disagree", 0.0) >= AUTO_STOP_DISAGREE:
            state["stop_reason"] = "auto_stop_reject"
            save_job_state(job_id, json_str(state), status='completed_rejected')
            clear_stop(job_id)
            logger.info("Job %s stopped (consensus reject >= %.2f)", job_id, AUTO_STOP_DISAGREE)
            return state
        # lightweight replan hook: if rejected but under threshold, inject a new collection task
        if decision == "reject" and scores.get("disagree", 0.0) > 0:
            state.setdefault("replans", []).append({"iteration": it + 1, "reason": "votes_reject", "action": "collect_more"})
            bus.publish("tasks", {"query": f"{query} collecte ciblee complementaire", "job_id": job_id, "iteration": it + 1})
            logger.info("Job %s triggered replan (collect_more) after reject vote", job_id)
            try:
                mark_polluted(query, reason="council_reject")
            except Exception:
                pass
        # replan based on poor coverage/evidence
        coverage_score = state.get("run_metrics", {}).get("coverage_score", 0.0)
        ev_count = state.get("run_metrics", {}).get("evidence_count", 0)
        if coverage_score is not None and coverage_score < REPLAN_MIN_COVERAGE and ev_count < REPLAN_MIN_EVIDENCE:
            state.setdefault("replans", []).append({"iteration": it + 1, "reason": "low_coverage", "action": "collect_more"})
            bus.publish("tasks", {"query": f"{query} collecte supplementaire (replan)", "job_id": job_id, "iteration": it + 1})
            logger.info("Job %s triggered replan (low coverage %.3f, evidence %s)", job_id, coverage_score, ev_count)
        # coordinator actions hook
        evaluate_replan(state, bus, query)
        executed_actions = await execute_actions_from_bus(bus, state, query)
        if executed_actions and run_ctx:
            for act in executed_actions:
                run_ctx.add_event("pipeline_injected", {"action": act.get("action"), "context": act.get("context")})

        # persist checkpoint
        save_job_state(job_id, json_str(state), status='running')
        try:
            snap_path = DATA_DIR / f"run_{job_id}.json"
            snap_path.write_text(json_str(state), encoding="utf-8")
        except Exception:
            pass
        logger.info(
            "Job %s iteration %d persisted: timeline=%d agents=%d token_spent=%s",
            job_id,
            it + 1,
            len(state.get('timeline', [])),
            len(state.get('agents', {})),
            state.get('token_spent'),
        )
        await asyncio.sleep(1)

    # finalization
    save_job_state(job_id, json_str(state), status='completed')
    clear_stop(job_id)
    state["diagnostic"] = build_diagnostic(state)
    logger.info("Job %s finished", job_id)
    return state


def json_str(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


async def _parse_tool_args(raw_args: Any) -> Dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return {}
    return {}


def _domain_from_url(url: str) -> str:
    try:
        return urlsplit(url).netloc or ""
    except Exception:
        return ""


async def run_web_search(
    query: str,
    top_k: int = 4,
    engine_name: Optional[str] = None,
    lang: str = "fr",
    safe_search: bool = True,
) -> List[Dict[str, Any]]:
    """Call an external search API (env SEARCHONE_WEB_SEARCH_ENDPOINT)."""
    if not query:
        return []
    engine = engine_name or WEB_SEARCH_ENGINE_NAME
    engine_set = WEB_SEARCH_ENGINE_SET or engine
    if _is_engine_paused(engine):
        logger.warning(
            "web_search.breaker_paused",
            extra={"engine": engine, "key": query[:200], "until": ENGINE_FAILURE_STATE.get(engine, {}).get("paused_until")},
        )
        return []
    _clean_cache_if_due()
    cached = _lookup_web_cache(query, lang, safe_search, engine_set)
    if cached is not None:
        logger.info("web_search.cache_hit", extra={"engine": engine, "query": query[:200], "count": len(cached)})
        return cached
    data = None
    headers = {"User-Agent": "Mozilla/5.0 (SearchOne/LMStudio)"}
    params = {"q": query, "format": "json", "language": lang, "safesearch": 1 if safe_search else 0}
    primary_error = None
    if WEB_SEARCH_ENDPOINT:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT, headers=headers) as client:
                resp = await client.get(WEB_SEARCH_ENDPOINT, params=params)
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("results") if isinstance(payload, dict) else payload
            _record_engine_result(engine, True)
        except httpx.HTTPStatusError as exc:
            primary_error = f"status_{exc.response.status_code}"
            logger.warning(
                "web_search.http_status",
                extra={"engine": engine, "query": query[:200], "status": exc.response.status_code},
            )
            _record_engine_result(engine, False)
        except httpx.RequestError as exc:
            primary_error = "network_error"
            logger.warning(
                "web_search.network_failure",
                extra={"engine": engine, "query": query[:200], "error": str(exc)},
            )
            _record_engine_result(engine, False)
        except Exception as exc:
            primary_error = "unknown_error"
            logger.warning(
                "web_search.failure",
                extra={"engine": engine, "query": query[:200], "error": str(exc)},
            )
            _record_engine_result(engine, False)
        if data is None:
            try:
                async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT, headers=headers) as client:
                    resp = await client.get(WEB_SEARCH_ENDPOINT, params={"q": query, "language": lang})
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "html.parser")
                    items = soup.select(".result") or soup.select("article.result")
                    parsed = []
                    for it in items:
                        a = it.find("a")
                        if not a or not a.get("href"):
                            continue
                        title = a.get_text(strip=True)
                        url = a.get("href")
                        snippet_tag = it.find("p") or it.find("div")
                        snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
                        parsed.append({"title": title, "url": url, "snippet": snippet})
                    data = parsed
                    logger.info("web_search.html_fallback", extra={"engine": engine, "query": query[:200], "count": len(parsed)})
                _record_engine_result(engine, True)
            except Exception as exc:
                logger.warning("web_search.html_fallback_failure", extra={"engine": engine, "error": str(exc)})
    if data is None:
        try:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as client:
                resp = await client.get("https://duckduckgo.com/html/", params={"q": query, "kl": f"{lang}-{lang}"})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                items = soup.select(".result") or soup.select(".result__body")
                parsed = []
                for it in items:
                    a = it.find("a", {"class": "result__a"})
                    if not a or not a.get("href"):
                        continue
                    title = a.get_text(strip=True)
                    url = a.get("href")
                    snippet_tag = it.find("a", {"class": "result__snippet"}) or it.find("div", {"class": "result__snippet"})
                    snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
                    parsed.append({"title": title, "url": url, "snippet": snippet})
                data = parsed
                logger.info("web_search.ddg_fallback", extra={"engine": engine, "query": query[:200], "count": len(parsed), "error": primary_error})
            _record_engine_result(engine, True)
        except Exception as exc:
            logger.warning(
                "web_search.ddg_failure",
                extra={"engine": engine, "query": query[:200], "error": str(exc), "prev_error": primary_error},
            )
            _record_engine_result(engine, False)
            return []
    results = data if isinstance(data, list) else data.get("results", [])
    out = []
    for item in results[:top_k]:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("name") or ""
        url = item.get("url") or item.get("link") or ""
        snippet = item.get("snippet") or item.get("content") or item.get("description") or ""
        text = "\n".join([p for p in [title, snippet, url] if p])
        meta = {"source": url or "web_search", "source_type": "url", "title": title, "domain": _domain_from_url(url)}
        out.append({"score": item.get("score"), "text": text[:WEB_FETCH_MAX_CHARS], "meta": meta})
    _persist_web_cache(query, lang, safe_search, engine_set, out)
    logger.info(
        "web_search.success",
        extra={"engine": engine, "query": query[:200], "count": len(out), "safe": safe_search, "lang": lang},
    )
    return out


async def run_experiment(script: str, timeout: float = EXPERIMENT_TIMEOUT) -> Dict[str, Any]:
    """Execute python script in a constrained subprocess; returns stdout/stderr."""
    import asyncio.subprocess as asp
    cmd = ["python", "-c", script]
    try:
        proc = await asp.create_subprocess_exec(
            *cmd,
            stdout=asp.PIPE,
            stderr=asp.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"score": None, "text": "[timeout]", "meta": {"source_type": "experiment", "timeout": timeout}}
        out = (stdout or b"").decode("utf-8", errors="ignore")
        err = (stderr or b"").decode("utf-8", errors="ignore")
        text = out.strip()
        if err:
            text = (text + "\n[stderr]\n" + err.strip()).strip()
        return {"score": None, "text": text[:WEB_FETCH_MAX_CHARS], "meta": {"source_type": "experiment", "timeout": timeout}}
    except Exception as e:
        logger.warning("run_experiment failed: %s", e)
        return {"score": None, "text": f"[error run_experiment: {e}]", "meta": {"source_type": "experiment"}}


async def run_search_semantic(agent: Agent, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Embed query and search persistent vector store (metadata-only)."""
    try:
        emb = await asyncio.to_thread(agent.embedder.encode, [query], show_progress_bar=False, convert_to_numpy=True)
        hits = search_semantic(agent.vs, emb, top_k=top_k)
        return hits
    except Exception as e:
        logger.warning("search_semantic failed: %s", e)
        return []


async def fetch_and_parse_url(url: str) -> Tuple[str, Dict[str, Any]]:
    """Fetch HTML/PDF and return cleaned text; tries direct fetch then r.jina.ai proxy."""
    if not url:
        return "", {}

    async def _fetch(target_url: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=WEB_FETCH_TIMEOUT) as client:
            resp = await client.get(target_url, headers={"User-Agent": "SearchOneBot/1.0"})
            resp.raise_for_status()
            return resp

    async def _extract_pdf(content: bytes) -> str:
        try:
            import fitz  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            logger.warning("pdf.parse.dependency_missing", extra={"error": str(e)})
            return ""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            texts = []
            for page in doc:
                texts.append(page.get_text("text"))
            return "\n".join(texts)
        except Exception as e:
            logger.warning("pdf.parse.failed", extra={"url": url, "error": str(e)})
            return ""

    resp: httpx.Response
    try:
        resp = await _fetch(url)
    except Exception as e:
        # fallback via jina ai proxy for basic readability
        proxy = f"https://r.jina.ai/{url}"
        logger.debug("fetch_url.primary_failed", extra={"url": url, "error": str(e)})
        try:
            resp = await _fetch(proxy)
        except Exception as e2:
            logger.warning("fetch_url.failed", extra={"url": url, "error": str(e2)})
            return "", {}

    content_type = resp.headers.get("Content-Type", "").lower()
    is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")

    if is_pdf:
        pdf_text = await _extract_pdf(resp.content)
        if pdf_text:
            return pdf_text[:WEB_FETCH_MAX_CHARS], {"source": url, "source_type": "pdf", "domain": _domain_from_url(url)}

    try:
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        text = text[:WEB_FETCH_MAX_CHARS]
    except Exception as e:
        logger.warning("fetch_url.parse_failed", extra={"url": url, "error": str(e)})
        return "", {}
    meta = {"source": url, "source_type": "url", "domain": _domain_from_url(url)}
    return text, meta


async def _execute_tool(name: str, args: Dict[str, Any], agent: Agent) -> List[Dict[str, Any]]:
    allowed = getattr(agent, "allowed_tools", None)
    if allowed is not None and name not in allowed:
        logger.info("tool_call.blocked_by_role", extra={"agent": getattr(agent, "name", "?"), "tool": name, "role": getattr(agent, "role", "?")})
        return []
    if not name or name not in TOOL_WHITELIST:
        logger.warning(
            "tool_call.rejected",
            extra={"agent": getattr(agent, "name", "?"), "tool": name, "reason": "not_whitelisted"},
        )
        return []
    if name == "search_hybrid":
        q = args.get("query") or ""
        k = args.get("top_k") or 3
        return await agent.retrieve_evidence(q, top_k=min(max(int(k), 1), 12))
    if name == "search_vector":
        q = args.get("query") or ""
        k = args.get("top_k") or 3
        return await agent.retrieve_evidence(q, top_k=min(max(int(k), 1), 12))
    if name == "sources_summary":
        # summarize recent docs/domains for situational awareness
        try:
            async with httpx.AsyncClient(timeout=TOOL_HTTP_TIMEOUT) as client:
                resp = await client.get("http://127.0.0.1:2001/sources/summary")
                resp.raise_for_status()
                data = resp.json()
                txt = json.dumps(data, ensure_ascii=False)
                return [{"score": None, "text": txt, "meta": {"source_type": "summary", "source": "sources_summary"}}]
        except Exception as e:
            logger.warning("sources_summary tool failed: %s", e)
            return []
    if name == "ingest_async":
        try:
            urls = args.get("urls") or []
            if not urls:
                return []
            title = args.get("title") or ""
            async with httpx.AsyncClient(timeout=TOOL_HTTP_TIMEOUT) as client:
                resp = await client.post("http://127.0.0.1:2001/ingest/async", json=urls, params={"title": title})
                resp.raise_for_status()
                return [{"score": None, "text": resp.text, "meta": {"source_type": "ingest_async", "source": "ingest"}}]
        except Exception as e:
            logger.warning("ingest_async tool failed: %s", e)
            return []
    if name == "web_search_tool":
        q = args.get("query") or ""
        k = args.get("top_k") or 4
        state_ref = getattr(agent, "job_state", None)
        if WEB_REQUEST_BUDGET and state_ref and state_ref.get("web_requests", 0) >= WEB_REQUEST_BUDGET:
            state_ref["web_request_budget_exceeded"] = True
            logger.warning(
                "web_search.budget_blocked",
                extra={
                    "agent": getattr(agent, "name", "?"),
                    "job": getattr(agent, "job_id", None),
                    "limit": WEB_REQUEST_BUDGET,
                },
            )
            return []
        results = await run_web_search(q, top_k=min(max(int(k), 1), 8))
        if state_ref is not None:
            state_ref["web_requests"] = state_ref.get("web_requests", 0) + 1
        return results
    if name == "web_cache_lookup":
        q = args.get("query") or ""
        if not q:
            return []
        lang = args.get("lang") or "fr"
        safe = args.get("safe_search")
        safe = bool(safe) if safe is not None else True
        engine = args.get("engine") or WEB_SEARCH_ENGINE_NAME
        cached = _lookup_web_cache(q, lang, safe, WEB_SEARCH_ENGINE_SET or engine)
        if cached:
            return [
                {
                    "score": None,
                    "text": f"Cache.hit: {len(cached)} résultats pour '{q}'",
                    "meta": {"source": "web_cache", "engine": engine, "lang": lang, "safe_search": safe},
                }
            ]
        return [
            {
                "score": None,
                "text": f"Aucun cache trouvé pour '{q}' ({lang}, safe={safe}).",
                "meta": {"source": "web_cache", "engine": engine, "lang": lang, "safe_search": safe},
            }
        ]
    if name == "fetch_and_parse_url":
        url = args.get("url") or ""
        if not url:
            return []
        txt, meta = await fetch_and_parse_url(url)
        if not txt:
            return []
        return [{"score": None, "text": txt, "meta": meta}]
    if name == "run_experiment":
        script = args.get("script") or ""
        timeout = float(args.get("timeout") or EXPERIMENT_TIMEOUT)
        if not script:
            return []
        res = await run_experiment(script, timeout=timeout)
        return [res] if res else []
    if name == "stats_summary":
        vals = args.get("values") or []
        try:
            stats = stats_summary([float(v) for v in vals])
        except Exception as e:
            logger.warning("stats_summary failed: %s", e)
            return []
        return [{"score": None, "text": str(stats), "meta": {"source_type": "stats_summary"}}]
    if name == "correlation_matrix":
        matrix = args.get("matrix") or []
        try:
            corr = correlation_matrix(matrix)
        except Exception as e:
            logger.warning("correlation_matrix failed: %s", e)
            return []
        return [{"score": None, "text": str(corr), "meta": {"source_type": "correlation_matrix"}}]
    if name == "ttest_independent":
        a = args.get("a") or []
        b = args.get("b") or []
        try:
            res = ttest_independent([float(x) for x in a], [float(x) for x in b])
        except Exception as e:
            logger.warning("ttest_independent failed: %s", e)
            return []
        return [{"score": None, "text": str(res), "meta": {"source_type": "ttest_independent"}}]
    if name == "simplify_expression":
        expr = args.get("expr") or ""
        if not expr:
            return []
        return [{"score": None, "text": simplify_expression(expr), "meta": {"source_type": "sympy"} }]
    if name == "solve_equation":
        lhs = args.get("lhs") or ""
        rhs = args.get("rhs") or "0"
        symbol = args.get("symbol") or "x"
        if not lhs:
            return []
        res = solve_equation(lhs, rhs, symbol)
        return [{"score": None, "text": str(res), "meta": {"source_type": "sympy"} }]
    if name == "search_semantic":
        query = args.get("query") or ""
        top_k = int(args.get("top_k") or 5)
        if not query:
            return []
        res = await run_search_semantic(agent, query, top_k=top_k)
        return res
    if name == "store_claim":
        claim = args.get("claim") or ""
        evidence_ids = args.get("evidence_ids") or []
        if not claim:
            return []
        try:
            store_claim(claim, evidence_ids)
            return [{"score": None, "text": "[claim stored]", "meta": {"source_type": "claim", "count": len(evidence_ids)}}]
        except Exception as e:
            logger.warning("store_claim failed: %s", e)
            return []
    if name == "get_related_nodes":
        entity = args.get("entity") or ""
        if not entity:
            return []
        nodes = get_related_nodes(entity)
        return [{"score": None, "text": str(nodes), "meta": {"source_type": "knowledge_graph"}}]
    return []


# Monkey-patch Agent with a helper to execute tool calls
async def _execute_tool_call(self: Agent, tool_call: Dict[str, Any]) -> List[Dict[str, Any]]:
    start = time.time()
    name = tool_call.get("function", {}).get("name") or tool_call.get("name")
    raw_args = tool_call.get("function", {}).get("arguments") or tool_call.get("arguments")
    try:
        args = await _parse_tool_args(raw_args)
        res = await asyncio.wait_for(_execute_tool(name, args, self), timeout=TOOL_CALL_TIMEOUT)
        TOOL_METRICS["success"] += 1
        TOOL_METRICS[f"success.{name}"] += 1
        logger.info(
            "tool_call.success",
            extra={
                "agent": getattr(self, "name", "?"),
                "tool": name,
                "tool_args": args,
                "duration": round(time.time() - start, 3),
                "count": len(res),
            },
        )
        return res
    except asyncio.TimeoutError:
        TOOL_METRICS["timeout"] += 1
        TOOL_METRICS[f"timeout.{name}"] += 1
        logger.warning(
            "tool_call.timeout",
            extra={
                "agent": getattr(self, "name", "?"),
                "tool": name,
                "duration": round(time.time() - start, 3),
                "timeout": TOOL_CALL_TIMEOUT,
            },
        )
        return []
    except Exception as e:
        TOOL_METRICS["fail"] += 1
        TOOL_METRICS[f"fail.{name}"] += 1
        logger.warning(
            "tool_call.failed",
            extra={
                "agent": getattr(self, "name", "?"),
                "tool": name,
                "error": str(e),
                "tool_args": raw_args,
                "duration": round(time.time() - start, 3),
            },
            exc_info=True,
        )
        return []


# bind helper
Agent._execute_tool_call = _execute_tool_call  # type: ignore[attr-defined]


def _extract_usage(payload: Any) -> Dict[str, int]:
    """Extract token usage from an LLM payload or response dict."""
    if not isinstance(payload, dict):
        return {}
    usage = payload.get("usage") or {}
    raw = payload.get("raw")
    if not usage and isinstance(raw, dict):
        usage = raw.get("usage") or raw.get("usage_summary") or {}
    if not isinstance(usage, dict):
        return {}
    out = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            out[k] = int(usage.get(k) or 0)
        except Exception:
            out[k] = 0
    if not out.get("total_tokens"):
        out["total_tokens"] = out.get("prompt_tokens", 0) + out.get("completion_tokens", 0)
    return out


def _record_usage(resp: Any, model_hint: str = None, label: str = "") -> Any:
    """Enrich response with usage and cost estimate, update global counters."""
    if not isinstance(resp, dict):
        return resp
    usage = _extract_usage(resp)
    if model_hint and not resp.get("model"):
        resp["model"] = model_hint
    if usage:
        resp["usage"] = usage
        LLM_USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
        LLM_USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
        LLM_USAGE["total_tokens"] += usage.get("total_tokens", 0)
        if ESTIMATED_COST_PER_TOKEN:
            cost = usage.get("total_tokens", 0) * ESTIMATED_COST_PER_TOKEN
            resp["cost_estimate_usd"] = cost
            LLM_USAGE["cost_usd"] += cost
    return resp


async def call_llm_with_retry(llm: LLMClient, prompt: str, max_tokens: int, label: str, attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, tools: list = None, tool_choice: str = None):
    """Retry wrapper around llm.generate with exponential backoff for transient failures."""
    last_err = None
    for i in range(attempts):
        try:
            resp = await llm.generate(prompt, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)
            LLM_USAGE["calls"] += 1
            return _record_usage(resp, getattr(llm, "model", None), label)
        except Exception as e:
            last_err = e
            logger.warning("LLM call failed (%s) attempt %d/%d: %s", label, i + 1, attempts, e, exc_info=True)
            if i < attempts - 1:
                await asyncio.sleep(delay)
                delay *= backoff
    err_msg = f"LLM error after retries: {last_err}"
    logger.exception("LLM call failed after %d attempts (%s): %s", attempts, label, last_err)
    return f"[{err_msg}]"


def _get_cross_encoder():
    """Lazily load the cross-encoder if rerank is enabled and dependencies are present."""
    global _cross_encoder
    if not ENABLE_RERANK or CrossEncoder is None:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder loaded (%s)", CROSS_ENCODER_MODEL)
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning("Failed to load cross-encoder (%s): %s", CROSS_ENCODER_MODEL, e)
        _cross_encoder = None
    return _cross_encoder


def _source_priority(meta: Dict[str, Any]) -> float:
    reliability = float(meta.get("reliability") or meta.get("score") or 0.0)
    src_type = meta.get("source_type") or meta.get("type") or ""
    bonus = SOURCE_PRIORITY_TYPE_BONUS.get(src_type, 0.0)
    recency_bonus = 0.0
    if meta.get("published_at"):
        try:
            dt = _dt.fromisoformat(str(meta.get("published_at")))
            days = (_dt.utcnow() - dt).days
            if days <= SOURCE_PRIORITY_RECENCY_DAYS:
                recency_bonus = 0.1
        except Exception:
            pass
    return reliability + bonus + recency_bonus


async def maybe_rerank(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank evidence results with a cross-encoder when enabled; otherwise return unchanged."""
    if not results or not ENABLE_RERANK:
        return results
    ce = _get_cross_encoder()
    if ce is None:
        return results
    pairs = [(query, r.get("text") or "") for r in results]
    try:
        scores = await asyncio.to_thread(ce.predict, pairs)
    except Exception as e:  # pragma: no cover - heavy path, optional
        logger.warning("Cross-encoder rerank failed, keeping vector scores: %s", e)
        return results
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    results = sorted(results, key=lambda r: r.get("rerank_score", r.get("score", 0.0)), reverse=True)
    return results


def deduplicate_and_score(results: List[Dict[str, Any]]) -> (List[Dict[str, Any]], Dict[str, Any]):
    """Remove duplicate chunks and compute a simple coverage score on sources/domains."""
    seen = set()
    deduped = []
    for r in results:
        meta = r.get("meta") or {}
        key = (meta.get("document_id"), meta.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    sources = set()
    for r in deduped:
        meta = r.get("meta") or {}
        src = meta.get("source") or meta.get("document_id")
        if src is not None:
            sources.add(str(src))

    coverage_score = (len(sources) / len(deduped)) if deduped else 0.0
    metrics = {
        "unique_chunks": len(deduped),
        "unique_sources": len(sources),
        "coverage_score": coverage_score,
    }
    # prioritize by source reliability/type/recency
    for r in deduped:
        meta = r.get("meta") or {}
        r["source_priority"] = _source_priority(meta)
    deduped = sorted(deduped, key=lambda x: x.get("source_priority", 0.0), reverse=True)
    return deduped, metrics
