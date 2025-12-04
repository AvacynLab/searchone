from __future__ import annotations

import json
import logging
import hashlib
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import DATA_DIR
from app.data.knowledge_store import CLAIMS_FILE, PROMOTION_FILE, POLLUTION_FILE, _load_jsonl

logger = logging.getLogger(__name__)

GRAPH_DIR = DATA_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def _hash_id(prefix: str, value: str) -> str:
    key = f"{prefix}:{value}"
    return f"{prefix}_{hashlib.md5(value.encode('utf-8')).hexdigest()[:8]}"


def _node_label(text: str, limit: int = 80) -> str:
    clean = text.strip()
    if len(clean) > limit:
        clean = clean[: limit - 3] + "..."
    return clean or "<unnamed>"


def _ensure_nodes(nodes: Dict[str, Dict[str, Any]], node_id: str, label: str, node_type: str, extra: Optional[Dict[str, Any]] = None) -> None:
    if node_id in nodes:
        return
    payload: Dict[str, Any] = {"id": node_id, "label": label, "type": node_type}
    if extra:
        payload.update(extra)
    nodes[node_id] = payload


def _build_claim_nodes(claims: List[Dict[str, Any]], nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    for entry in claims:
        claim_text = str(entry.get("claim") or "").strip()
        if not claim_text:
            continue
        claim_id = _hash_id("claim", claim_text)
        _ensure_nodes(nodes, claim_id, _node_label(claim_text), "claim", {"raw": claim_text})

        for evidence_id in (entry.get("evidence_ids") or []):
            ev_str = str(evidence_id)
            if not ev_str:
                continue
            evidence_id = _hash_id("evidence", ev_str)
            _ensure_nodes(nodes, evidence_id, _node_label(ev_str, limit=60), "evidence", {"reference": ev_str})
            edges.append({"source": evidence_id, "target": claim_id, "relation": "supports"})


def _build_concept_nodes(promotions: List[Dict[str, Any]], nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]], claims: List[Dict[str, Any]]) -> None:
    for entry in promotions:
        concept = str(entry.get("node") or "").strip()
        if not concept:
            continue
        concept_id = _hash_id("concept", concept)
        _ensure_nodes(
            nodes,
            concept_id,
            _node_label(concept),
            "concept",
            {"reason": entry.get("reason"), "score": entry.get("score", 0)},
        )
        for claim in claims:
            claim_text = str(claim.get("claim") or "")
            if not claim_text:
                continue
            if concept.lower() in claim_text.lower():
                claim_id = _hash_id("claim", claim_text)
                edges.append({"source": concept_id, "target": claim_id, "relation": "supports"})


def _build_pollution_nodes(pollutions: List[Dict[str, Any]], nodes: Dict[str, Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    for entry in pollutions:
        node = str(entry.get("node") or "").strip()
        if not node:
            continue
        node_id = _hash_id("polluted", node)
        _ensure_nodes(nodes, node_id, _node_label(node), "polluted", {"reason": entry.get("reason")})
        # double-edge to represent contradictions
        edges.append({"source": node_id, "target": node_id, "relation": "contradicts"})


def build_knowledge_graph(job_id: Optional[int] = None) -> Dict[str, Any]:
    claims = _load_jsonl(CLAIMS_FILE)
    promotions = _load_jsonl(PROMOTION_FILE)
    pollutions = _load_jsonl(POLLUTION_FILE)

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    _build_claim_nodes(claims, nodes, edges)
    _build_concept_nodes(promotions, nodes, edges, claims)
    _build_pollution_nodes(pollutions, nodes, edges)

    timestamp = datetime.utcnow().isoformat() + "Z"

    graph = {
        "job_id": job_id,
        "nodes": list(nodes.values()),
        "edges": edges,
        "generated_at": timestamp,
        "counts": {"nodes": len(nodes), "edges": len(edges)},
    }
    return graph


def graph_stats(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    edges = graph.get("edges") or []
    adjacency: Dict[str, set] = defaultdict(set)
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src:
            adjacency[src].add(tgt)
        if tgt:
            adjacency[tgt].add(src)

    degrees = {nid: len(neighbors) for nid, neighbors in adjacency.items()}
    for nid in nodes:
        degrees.setdefault(nid, 0)

    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0.0

    visited = set()
    component_count = 0
    for nid in nodes:
        if nid in visited:
            continue
        component_count += 1
        stack = [nid]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor and neighbor not in visited:
                    stack.append(neighbor)

    hubs = sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:3]
    hub_list = [
        {"node": nodes[nid]["label"], "degree": degree}
        for nid, degree in hubs
        if nid in nodes
    ]

    return {
        "avg_degree": round(avg_degree, 2),
        "component_count": component_count,
        "hubs": hub_list,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def export_graphviz(graph: Dict[str, Any], job_id: Optional[int] = None) -> Dict[str, Optional[str]]:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    folder = GRAPH_DIR / (f"job_{job_id}" if job_id is not None else "misc")
    folder.mkdir(parents=True, exist_ok=True)
    base_name = f"knowledge_graph_{timestamp}"
    dot_path = folder / f"{base_name}.dot"
    try:
        lines = ["digraph KnowledgeGraph {"]
        for node in graph.get("nodes", []):
            label = node.get("label") or node.get("id")
            escaped_label = label.replace('"', '\\"')
            shape = "ellipse"
            if node.get("type") == "polluted":
                shape = "octagon"
            elif node.get("type") == "concept":
                shape = "box"
            lines.append(f'  "{node["id"]}" [label="{escaped_label}" shape={shape}];')
        for edge in graph.get("edges", []):
            relation = edge.get("relation") or ""
            rel_label = relation if relation else ""
            lines.append(
                f'  "{edge.get("source")}" -> "{edge.get("target")}" [label="{rel_label}"];'
            )
        lines.append("}")
        dot_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write graphviz DOT: %s", exc)
        return {"dot": None, "png": None}

    png_path = folder / f"{base_name}.png"
    dot_exec = shutil.which("dot")
    if dot_exec:
        try:
            subprocess.run([dot_exec, str(dot_path), "-Tpng", "-o", str(png_path)], check=True)
            return {"dot": str(dot_path), "png": str(png_path)}
        except Exception as exc:
            logger.warning("Graphviz failed to render PNG: %s", exc)
            return {"dot": str(dot_path), "png": None}
    return {"dot": str(dot_path), "png": None}
