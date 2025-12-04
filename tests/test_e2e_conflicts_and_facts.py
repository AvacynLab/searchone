import pytest

from app.workflows import orchestrator
from app.workflows.scenarios import PhaseSpec, ScenarioSpec


@pytest.mark.asyncio
async def test_e2e_conflicts_and_facts(monkeypatch):
    scenario = ScenarioSpec(
        name="conflicts_and_facts",
        objective="Vérifier le croisement de faits contradictoires",
        phases=[
            PhaseSpec(name="fact_phase", agents=["FactChecker"], tools=["fact_check_tool"]),
            PhaseSpec(name="resolve_phase", agents=["Critic"], tools=["resolve_conflicts_tool"]),
        ],
    )

    call_idx = {"n": 0}

    async def fake_run_agents_job(**kwargs):
        order = call_idx["n"]
        call_idx["n"] += 1
        allowlist = kwargs.get("tool_allowlist") or []
        if order == 0:
            assert "fact_check_tool" in allowlist
            return {
                "claims": [
                    {
                        "id": "claim_a",
                        "claim": "Hypothèse initiale",
                        "status": "unknown",
                        "support_evidence_ids": [],
                        "refute_evidence_ids": [],
                    }
                ],
                "fact_checks": [{"claim": "Hypothèse initiale", "verdict": "uncertain"}],
                "report": {"controversies": []},
            }
        return {
            "claims": [
                {
                    "id": "claim_a",
                    "claim": "Hypothèse initiale",
                    "status": "controversial",
                    "support_evidence_ids": ["k1"],
                    "refute_evidence_ids": ["k2"],
                }
            ],
            "fact_checks": [{"claim": "Hypothèse initiale", "verdict": "controversial"}],
            "report": {
                "controversies": [
                    {"claim": "Hypothèse initiale", "status": "controversial", "sources": ["s1", "s2"]}
                ]
            },
        }

    monkeypatch.setattr("app.workflows.orchestrator.run_agents_job", fake_run_agents_job)
    monkeypatch.setattr("app.workflows.orchestrator.get_scenario", lambda name: scenario if name == scenario.name else None)

    orch = orchestrator.Orchestrator(roles=["FactChecker", "Critic"], max_iterations=1, max_duration_seconds=20)
    result = await orch.run_with_scenario(402, "contradictions", scenario.name)

    final_phase = result["phase_history"][-1]["result"]
    assert any(claim.get("status") == "controversial" for claim in final_phase.get("claims", []))
    assert final_phase["report"].get("controversies")
