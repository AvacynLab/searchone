import asyncio
import os
import sys
from unittest.mock import Mock

CURRENT_DIR = os.path.dirname(__file__)
PARENT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from app.workflows import agents
from app.cli.debug_cli import MockLLM, MockEmbedder, MockFaissStore
from app.services.reporting import build_structured_summary


def test_knowledge_graph_tool_integration(monkeypatch):
    graph = {"nodes": [], "edges": [], "generated_at": "2025-01-01T00:00:00Z"}
    stats = {
        "node_count": 2,
        "edge_count": 1,
        "component_count": 1,
        "hubs": [{"node": "Claim A", "degree": 1}],
    }
    exports = {"png": "graphs/graph.png", "dot": "graphs/graph.dot"}

    monkeypatch.setattr(agents, "build_knowledge_graph", Mock(return_value=graph))
    monkeypatch.setattr(agents, "graph_stats", Mock(return_value=stats))
    monkeypatch.setattr(agents, "export_graphviz", Mock(return_value=exports))

    agent = agents.Agent(
        name="GraphTester",
        role="Analyst",
        llm=MockLLM(),
        vs=MockFaissStore(),
        embedder=MockEmbedder(),
    )
    agent.job_id = 314
    agent.job_state = {}

    results = asyncio.run(agents._execute_tool("knowledge_graph_tool", {"job_id": 314, "scope": "current_job"}, agent))

    assert results
    assert "Graphe de connaissances" in results[0]["text"]
    assert "2 nœuds" in results[0]["text"]
    agents.build_knowledge_graph.assert_called_once_with(job_id=314)
    agents.graph_stats.assert_called_once_with(graph)
    agents.export_graphviz.assert_called_once_with(graph, job_id=314)

    job_state = agent.job_state
    assert job_state["knowledge_graph_stats"] == stats
    exports_list = job_state.get("knowledge_graph_exports") or []
    assert any(export["path"] == exports["png"] for export in exports_list)
    assert any(export["path"] == exports["dot"] for export in exports_list)


def test_structured_summary_includes_knowledge_graph():
    job_state = {
        "knowledge_graph_stats": {
            "node_count": 5,
            "edge_count": 4,
            "component_count": 1,
            "hubs": [{"node": "ClaimOne", "degree": 2}],
        },
        "knowledge_graph_exports": [
            {"format": "png", "path": "graphs/graph.png", "created_at": "2025-01-01T00:00:00Z"},
        ],
    }
    summary = build_structured_summary(job_state)
    kg_summary = summary.get("knowledge_graph") or {}
    assert kg_summary.get("stats") == job_state["knowledge_graph_stats"]
    assert kg_summary.get("exports") == job_state["knowledge_graph_exports"]
    topology = summary.get("knowledge_topology") or {}
    assert topology.get("description", "").startswith("Topologie du graphe de connaissances")
