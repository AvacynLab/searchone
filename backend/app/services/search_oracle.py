from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.data.knowledge_store import search_semantic

WebSearchFn = Callable[[str, int], Awaitable[List[Dict[str, Any]]]]


@dataclass
class Evidence:
    source: str
    snippet: str
    source_type: str = "internal"
    title: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "snippet": self.snippet,
            "source_type": self.source_type,
            "title": self.title,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class SearchSessionState:
    job_id: int
    root_query: str
    focus: Optional[str] = None
    subqueries: List[Dict[str, Any]] = field(default_factory=list)
    used_budget: Dict[str, int] = field(default_factory=lambda: {"internal": 0, "api": 0, "web": 0})
    evidence_ids: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    vector_store: Optional[Any] = None
    embedder: Optional[Any] = None
    web_search: Optional[WebSearchFn] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def plan_subqueries(root_query: str, max_depth: int = 2, focus: Optional[str] = None) -> List[str]:
    clean_query = " ".join(root_query.split())
    if not clean_query:
        return []
    parts = [p.strip() for p in re.split(r"[;,.!?]+", clean_query) if p.strip()]
    if focus:
        focus_clause = focus.strip()
        if focus_clause and focus_clause not in clean_query:
            parts.append(f"{clean_query} ({focus_clause})")
    if not parts:
        parts = [clean_query]
    if len(parts) > max_depth > 0:
        parts = parts[:max_depth]
    return parts


async def search_internal(session: SearchSessionState, subquery: str, top_k: int = 5) -> List[Evidence]:
    if not session.embedder or not session.vector_store:
        return []
    try:
        embeddings = await asyncio.to_thread(
            session.embedder.encode, [subquery], show_progress_bar=False, convert_to_numpy=True
        )
    except Exception:
        embeddings = await asyncio.to_thread(session.embedder.encode, [subquery])
    hits = search_semantic(session.vector_store, embeddings, top_k=top_k)
    session.used_budget["internal"] += 1
    evidences: List[Evidence] = []
    for hit in hits:
        meta = hit.get("meta") or {}
        title = meta.get("title") or meta.get("chunk_title") or ""
        snippet = meta.get("text") or meta.get("snippet") or subquery
        evidences.append(
            Evidence(
                source=meta.get("source") or f"internal:{session.job_id}",
                snippet=snippet,
                source_type="internal",
                title=title,
                score=hit.get("score"),
                metadata=meta,
            )
        )
    return evidences


async def search_web_via_searx(session: SearchSessionState, subquery: str, top_k: int = 4) -> List[Evidence]:
    if not session.web_search:
        return []
    session.used_budget["web"] += 1
    try:
        raw = await session.web_search(subquery, top_k)
    except Exception:
        return []
    evidences: List[Evidence] = []
    for entry in raw:
        snippet = entry.get("text") or entry.get("title") or subquery
        meta = entry.get("meta") or {}
        source = meta.get("source") or entry.get("source") or entry.get("url") or f"web:{subquery[:30]}"
        evidences.append(
            Evidence(
                source=source,
                snippet=snippet,
                source_type="web",
                title=meta.get("title") or entry.get("title"),
                score=entry.get("score"),
                metadata=meta,
            )
        )
    return evidences


def enrich_knowledge_store(evidences: List[Evidence]) -> List[str]:
    ids: List[str] = []
    for evidence in evidences:
        source_key = evidence.source or ""
        snippet_key = evidence.snippet or ""
        digest = hashlib.sha256(f"{source_key}:{snippet_key}".encode("utf-8")).hexdigest()
        ids.append(digest)
    return ids


def update_coverage(session: SearchSessionState, subquery: str, evidences: List[Evidence]) -> None:
    coverage_score = min(len(evidences) / 3, 1.0) if evidences else 0.0
    session.subqueries.append(
        {
            "subquery": subquery,
            "coverage": coverage_score,
            "evidence_count": len(evidences),
            "sources": [ev.source for ev in evidences],
        }
    )
    if coverage_score < 0.5:
        session.gaps.append(subquery)
