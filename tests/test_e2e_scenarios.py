import json
import pytest

from unittest.mock import Mock

from app.workflows import orchestrator
from app.workflows.scenarios import PhaseSpec, ScenarioSpec, get_scenario, load_scenarios


@pytest.mark.asyncio
async def test_quick_literature_review_scenario(monkeypatch):
    recorded = []

    async def fake_run_agents_job(**kwargs):
        recorded.append(kwargs)
        return {
            "hypotheses": ["Hypothèse synthétique"],
            "research_score": {"coverage": 0.78},
            "run_metrics": {"coverage_score": 0.72, "evidence_count": 3},
            "report": {"status": "draft", "content": "Synthèse"},
        }

    monkeypatch.setattr("app.workflows.orchestrator.run_agents_job", fake_run_agents_job)

    load_scenarios.cache_clear()
    orch = orchestrator.Orchestrator(roles=["Explorer", "Analyst"], max_iterations=1, max_duration_seconds=120)
    result = await orch.run_with_scenario(101, "impact du climat", "quick_literature_review")

    assert result["scenario"] == "quick_literature_review"
    scenario = get_scenario("quick_literature_review")
    assert scenario is not None
    assert len(result.get("phase_history", [])) == len(scenario.phases)

    first_phase = result["phase_history"][0]
    assert first_phase["phase"] == scenario.phases[0].name
    assert first_phase["meta"]["name"] == scenario.phases[0].name

    final_phase = result["phase_history"][-1]
    assert final_phase["result"]["hypotheses"], "Hypotheses should be present"
    assert final_phase["result"]["research_score"]["coverage"] > 0, "Coverage must be positive"
    assert final_phase["result"]["report"]["content"], "Report should exist"


@pytest.mark.asyncio
async def test_phase_tool_allowlist_blocks_plot_tool(monkeypatch):
    tool_results = []

    async def fake_act(self, shared_context, council_inbox):
        call = {
            "function": {
                "name": "plot_tool",
                "arguments": json.dumps({"series": [{"y": [1, 2, 3]}]}),
            }
        }
        res = await self._execute_tool_call(call)
        tool_results.append(res)
        await council_inbox.put({"agent": self.name, "role": self.role, "hypothesis": "blocked", "evidence": []})

    monkeypatch.setattr("app.workflows.agents.Agent.act", fake_act)
    scenario = ScenarioSpec(
        name="allowlist_phase",
        objective="Test allowlist",
        phases=[PhaseSpec(name="limited_tools", agents=["Analyst"], tools=["web_search_tool"])],
    )
    monkeypatch.setattr("app.workflows.scenarios.get_scenario", lambda name: scenario if name == scenario.name else None)
    monkeypatch.setattr("app.workflows.scenarios.list_scenarios", lambda: [scenario.name])
    monkeypatch.setattr("app.workflows.orchestrator.get_scenario", lambda name: scenario if name == scenario.name else None)
    monkeypatch.setattr("app.workflows.agents.generate_plot", Mock())

    orch = orchestrator.Orchestrator(roles=["Analyst"], max_iterations=1, max_duration_seconds=30)
    result = await orch.run_with_scenario(502, "test query", scenario.name)

    assert result["scenario"] == scenario.name
    assert tool_results
    assert tool_results[0][0]["meta"]["reason"] == "tool_not_allowed"
    assert tool_results[0][0]["meta"]["tool"] == "plot_tool"
