import pytest

from app.workflows import orchestrator


@pytest.mark.asyncio
async def test_e2e_research_oracle_scenario(monkeypatch):
    captured = {}

    async def fake_run_agents_job(**kwargs):
        captured.setdefault("calls", []).append(
            {
                "phase_meta": kwargs.get("phase_meta") or {},
                "tool_allowlist": kwargs.get("tool_allowlist") or [],
            }
        )
        return {
            "search_oracle": [
                {
                    "subqueries": [
                        {
                            "subquery": "Energy storage applications",
                            "coverage": 0.9,
                            "evidence_count": 2,
                            "sources": ["doc:A1", "doc:A2"],
                        }
                    ],
                    "gaps": ["economics"],
                    "used_budget": {"internal": 1, "web": 0, "api": 0},
                }
            ],
            "knowledge_store": {"entries": [{"id": "alpha", "claim": "Storage tech"}]},
            "report": {"coverage": 0.85, "gaps": ["economics"]},
        }

    monkeypatch.setattr("app.workflows.orchestrator.run_agents_job", fake_run_agents_job)

    orch = orchestrator.Orchestrator(roles=["Explorer"], max_iterations=1, max_duration_seconds=20)
    result = await orch.run_with_scenario(401, "storage materials", "quick_literature_review")

    exp_call = next(
        (c for c in captured.get("calls", []) if c.get("phase_meta", {}).get("name") == "exploration"),
        None,
    )
    assert exp_call is not None
    assert "search_oracle_tool" in exp_call.get("tool_allowlist", [])
    phase_result = result["phase_history"][0]["result"]
    assert phase_result["search_oracle"][0]["subqueries"]
    assert "economics" in phase_result["search_oracle"][0]["gaps"]
    assert phase_result["report"]["gaps"]
