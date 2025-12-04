import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.core.config import DATA_DIR
from app.data.vector_store import FaissStore
import hashlib
from collections import defaultdict

CLAIMS_FILE = DATA_DIR / "claims.jsonl"
PROMOTION_FILE = DATA_DIR / "knowledge_promotion.jsonl"
POLLUTION_FILE = DATA_DIR / "knowledge_pollution.jsonl"


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


def store_claim(claim: str, evidence_ids: List[str]) -> None:
    """Persist claims with evidence ids in a JSONL file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"claim": claim, "evidence_ids": evidence_ids}
    line = json.dumps(payload, ensure_ascii=False)
    with CLAIMS_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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
