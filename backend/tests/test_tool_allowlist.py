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


def test_tool_allowlist_blocks_plot_tool(monkeypatch):
    monkeypatch.setattr(agents, "generate_plot", Mock())
    agent = agents.Agent(
        name="AllowlistTester",
        role="Analyst",
        llm=MockLLM(),
        vs=MockFaissStore(),
        embedder=MockEmbedder(),
    )
    agent.allowed_tools = {"plot_tool", "web_search_tool"}
    agent.state["tool_allowlist"] = ["web_search_tool"]
    args = {"series": [{"y": [1, 2, 3]}], "plot_type": "line"}

    result = asyncio.run(agents._execute_tool("plot_tool", args, agent))

    assert agents.generate_plot.call_count == 0
    assert result
    evidence = result[0]
    assert "n'est pas autorisé" in evidence["text"]
    assert evidence["meta"]["reason"] == "tool_not_allowed"
    assert evidence["meta"]["tool"] == "plot_tool"
