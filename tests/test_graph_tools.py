import json
from pathlib import Path

from app.services import graph_tools
from app.data import knowledge_store


def _write_jsonl(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_build_knowledge_graph_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge_store, "CLAIMS_FILE", tmp_path / "claims.jsonl")
    monkeypatch.setattr(knowledge_store, "PROMOTION_FILE", tmp_path / "promotions.jsonl")
    monkeypatch.setattr(knowledge_store, "POLLUTION_FILE", tmp_path / "pollutions.jsonl")

    claims = [{"claim": "L'horizon est arrondi", "evidence_ids": ["doc-1", "doc-2"]}]
    promotions = [{"node": "horizon", "reason": "concept highlight", "score": 0.9}]
    pollutions = [{"node": "acide", "reason": "contradiction identifiee"}]
    _write_jsonl(knowledge_store.CLAIMS_FILE, claims)
    _write_jsonl(knowledge_store.PROMOTION_FILE, promotions)
    _write_jsonl(knowledge_store.POLLUTION_FILE, pollutions)

    monkeypatch.setattr(graph_tools, "GRAPH_DIR", tmp_path / "graphs")
    graph = graph_tools.build_knowledge_graph(job_id=7)
    assert graph["job_id"] == 7
    assert graph["counts"]["nodes"] >= 1

    stats = graph_tools.graph_stats(graph)
    assert stats["node_count"] == graph["counts"]["nodes"]
    assert "avg_degree" in stats

    monkeypatch.setattr(graph_tools.shutil, "which", lambda _: None)
    exported = graph_tools.export_graphviz(graph, job_id=7)
    assert exported["dot"]
    assert Path(exported["dot"]).exists()
