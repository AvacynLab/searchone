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
from app.services.plot_tools import PlotArtifact
from app.workflows.writing_pipeline import build_scientific_article


def test_plot_tool_integration_records_figure(tmp_path, monkeypatch):
    png_path = tmp_path / "figure.png"
    svg_path = tmp_path / "figure.svg"
    png_path.write_text("png")
    svg_path.write_text("svg")
    artifact = PlotArtifact(
        png_path=png_path,
        metadata={
            "title": "Sample plot",
            "plot_type": "line",
            "variables": ["x", "y"],
            "generated_at": "2025-01-01T00:00:00Z",
        },
        vector_paths={"svg": svg_path},
    )
    mock_generate = Mock(return_value=artifact)
    monkeypatch.setattr(agents, "generate_plot", mock_generate)

    agent = agents.Agent(
        name="ToolTester",
        role="Analyst",
        llm=MockLLM(),
        vs=MockFaissStore(),
        embedder=MockEmbedder(),
    )
    agent.job_id = 7
    agent.job_state = {}

    args = {
        "series": [{"x": [0, 1], "y": [3, 4], "label": "series A"}],
        "plot_type": "line",
        "title": "Integration figure",
        "vector_formats": ["svg"],
    }
    results = asyncio.run(agents._execute_tool("plot_tool", args, agent))

    assert len(results) == 1
    mock_generate.assert_called_once()
    called = mock_generate.call_args.kwargs
    assert called["data"] == args["series"]
    assert called["spec"]["type"] == "line"
    assert called["spec"]["title"] == "Integration figure"

    evidence = agent.job_state.get("evidence") or []
    assert evidence
    recorded = evidence[-1]
    assert recorded is results[0]
    fig_meta = recorded["meta"]["figure"]
    assert fig_meta["path"] == str(png_path)
    assert fig_meta["title"] == artifact.metadata["title"]
    assert fig_meta["plot_type"] == "line"


def test_article_includes_figures_from_timeline():
    state = {
        "timeline": [
            {
                "messages": [
                    {
                        "agent": "A1",
                        "role": "Analyst",
                        "hypothesis": "H1",
                        "evidence": [
                            {
                                "score": None,
                                "text": "Evidence for figure",
                                "meta": {
                                    "source_type": "plot",
                                    "figure": {
                                        "title": "Test figure",
                                        "path": "plots/test.png",
                                        "description": "A test graph",
                                        "plot_type": "line",
                                        "variables": ["x", "y"],
                                    },
                                },
                            }
                        ],
                    }
                ],
                "votes": [],
                "summary": "Figure entry",
            }
        ],
        "query": "Testing diagrams",
    }
    result = build_scientific_article(state)
    figures = result.get("figures") or []
    assert figures
    assert figures[0]["title"] == "Test figure"
    assert figures[0]["path"] == "plots/test.png"
