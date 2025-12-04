import hashlib
import json
from collections import defaultdict
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import DATA_DIR
from app.data.vector_store import FaissStore

CLAIMS_FILE = DATA_DIR / "claims.jsonl"
PROMOTION_FILE = DATA_DIR / "knowledge_promotion.jsonl"
POLLUTION_FILE = DATA_DIR / "knowledge_pollution.jsonl"
GRAPH_NODES_FILE = DATA_DIR / "knowledge_graph_nodes.jsonl"
GRAPH_EDGES_FILE = DATA_DIR / "knowledge_graph_edges.jsonl"
FACT_CHECK_FILE = DATA_DIR / "fact_checks.jsonl"


def search_semantic(vs: FaissStore, query_emb, top_k: int = 5) -> List[Dict[str, Any]]:
    """Semantic search wrapper around FaissStore; returns metadata only."""
    hits = vs.search(query_emb, top_k=top_k)
    results: List[Dict[str, Any]] = []
    for h in hits:
        meta = h.get("metadata") or {}
        results.append({"score": h.get("score"), "meta": meta})
    return results


def get_related_nodes(entity: str, graph: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Placeholder graph lookup; returns related nodes if provided."""
    graph = graph or {}
    return graph.get(entity, [])


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_data_dir()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _claim_identifier(claim_text: str) -> str:
    return md5(claim_text.encode("utf-8")).hexdigest()


def _default_claim_payload(
    claim: str,
    evidence_ids: List[str],
    *,
    status: str = "unknown",
    support_evidence_ids: Optional[List[str]] = None,
    refute_evidence_ids: Optional[List[str]] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
    relations: Optional[List[Dict[str, Any]]] = None,
    source_doc_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    source_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    support_evidence_ids = support_evidence_ids or []
    refute_evidence_ids = refute_evidence_ids or []
    payload = {
        "claim": claim,
        "evidence_ids": evidence_ids,
        "support_evidence_ids": support_evidence_ids,
        "refute_evidence_ids": refute_evidence_ids,
        "status": status,
        "entities": entities or [],
        "relations": relations or [],
        "source_doc_id": source_doc_id,
        "chunk_id": chunk_id,
        "source_meta": source_meta or {},
        "claim_id": _claim_identifier(claim),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    return payload


def store_claim(claim: str, evidence_ids: List[str]) -> None:
    """Persist claims with evidence ids in a JSONL file."""
    payload = _default_claim_payload(claim, evidence_ids)
    _write_jsonl(CLAIMS_FILE, payload)


def promote_knowledge(node: str, reason: str = "", score: float = 1.0) -> None:
    """Mark a node/claim as promoted to global memory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"node": node, "reason": reason, "score": score}
    with PROMOTION_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def mark_polluted(node: str, reason: str = "") -> None:
    """Mark knowledge as polluted/erroneous for later downranking."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"node": node, "reason": reason, "hash": hashlib.md5(node.encode("utf-8")).hexdigest()}
    with POLLUTION_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def consolidate_promotions(max_entries: int = 500) -> Dict[str, Any]:
    """
    Deduplicate promoted nodes and mark conflicts with pollution list.
    Returns summary stats to surface in observability/dashboard.
    """
    promotions = _load_jsonl(PROMOTION_FILE)
    pollutions = _load_jsonl(POLLUTION_FILE)
    polluted_hashes = {p.get("hash") for p in pollutions if p.get("hash")}
    # dedupe by node hash
    dedup: Dict[str, Dict[str, Any]] = {}
    for p in promotions:
        node = p.get("node")
        if not node:
            continue
        h = hashlib.md5(node.encode("utf-8")).hexdigest()
        if h in dedup:
            dedup[h]["score"] = max(dedup[h].get("score", 0), p.get("score", 0))
            dedup[h].setdefault("reasons", []).append(p.get("reason"))
        else:
            dedup[h] = {**p, "hash": h, "reasons": [p.get("reason")]}
        dedup[h]["polluted"] = h in polluted_hashes
    # keep top entries by score
    sorted_nodes = sorted(dedup.values(), key=lambda x: x.get("score", 0), reverse=True)[:max_entries]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "knowledge_promoted_consolidated.json"
    out_path.write_text(json.dumps(sorted_nodes, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"count": len(sorted_nodes), "polluted": len([n for n in sorted_nodes if n.get("polluted")]), "path": str(out_path)}


def load_claims() -> List[Dict[str, Any]]:
    return _load_jsonl(CLAIMS_FILE)


def store_structured_knowledge(
    claims: List[Dict[str, Any]],
    entities: Optional[List[Dict[str, Any]]] = None,
    relations: Optional[List[Dict[str, Any]]] = None,
    source_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist claims along with structured entity/relation annotations."""
    entities = entities or []
    relations = relations or []
    for claim_entry in claims:
        claim_text = claim_entry.get("text") or claim_entry.get("claim") or ""
        if not claim_text:
            continue
        payload = _default_claim_payload(
            claim_text,
            claim_entry.get("evidence_ids") or [],
            status=claim_entry.get("status", "unknown"),
            support_evidence_ids=claim_entry.get("support_evidence_ids"),
            refute_evidence_ids=claim_entry.get("refute_evidence_ids"),
            entities=claim_entry.get("entities") or [],
            relations=claim_entry.get("relations") or [],
            source_doc_id=claim_entry.get("source_doc_id"),
            chunk_id=claim_entry.get("chunk_id"),
            source_meta=(claim_entry.get("source_meta") or source_meta),
        )
        _write_jsonl(CLAIMS_FILE, payload)
    for entity in entities:
        node_entry = {"entity": entity.get("text"), "label": entity.get("label"), "normalized": entity.get("normalized")}
        _write_jsonl(GRAPH_NODES_FILE, node_entry)
    for relation in relations:
        edge_entry = {
            "type": relation.get("type"),
            "from": relation.get("from"),
            "to": relation.get("to"),
            "context": relation.get("context"),
        }
        _write_jsonl(GRAPH_EDGES_FILE, edge_entry)


def store_fact_check_result(result: Dict[str, Any]) -> None:
    """Persist a fact check verdict for later reporting."""
    if not isinstance(result, dict):
        return
    _write_jsonl(FACT_CHECK_FILE, result)


def get_claim_by_id(claim_id: str) -> Optional[Dict[str, Any]]:
    for entry in _load_jsonl(CLAIMS_FILE):
        if entry.get("claim_id") == claim_id:
            return entry
    return None


def _rewrite_claims(entries: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    with CLAIMS_FILE.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_claim_status(
    claim_id: str,
    status: str,
    support_evidence_ids: Optional[List[str]] = None,
    refute_evidence_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    entries = _load_jsonl(CLAIMS_FILE)
    updated = None
    for entry in entries:
        if entry.get("claim_id") != claim_id:
            continue
        entry["status"] = status
        if support_evidence_ids is not None:
            entry["support_evidence_ids"] = support_evidence_ids
        if refute_evidence_ids is not None:
            entry["refute_evidence_ids"] = refute_evidence_ids
        entry["updated_at"] = datetime.utcnow().isoformat() + "Z"
        updated = entry
        break
    if updated:
        _rewrite_claims(entries)
    return updated
