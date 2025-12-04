import pytest

from runtime import MessageBus
from coordinator_actions import evaluate_replan
from workflow import (
    WorkflowEngine,
    build_collect_subpipeline,
    build_full_research_pipeline,
    execute_actions_from_bus,
)


@pytest.mark.asyncio
async def test_scenario_simple_compression_models():
    """Mini E2E: pipeline full_research avec requête simple."""
    ctx = {"query": "comparer deux approches de compression de modeles"}
    pipeline = build_full_research_pipeline()
    engine = WorkflowEngine(pipeline.linearize())
    res = await engine.run(context=ctx)
    final_ctx = res[-1]["result"]
    assert "claims" in final_ctx
    assert "report" in final_ctx
    assert final_ctx["report"]["status"] == "draft"


@pytest.mark.asyncio
async def test_scenario_controverse_scientifique_replan():
    """Débat -> vote reject -> replan collecte complémentaire + exécution pipeline dynamique."""
    bus = MessageBus()
    state = {
        "votes": {1: {"decision": "reject", "scores": {"agree": 0.2, "disagree": 0.8}}},
        "run_metrics": {"coverage_score": 0.05, "evidence_count": 0},
    }
    evaluate_replan(state, bus, "controverse scientifique")
    actions = bus.drain("actions")
    assert actions and actions[0]["type"] == "replan_collect"

    # re-inject action and execute sub-pipeline
    bus.publish("actions", actions[0])
    executed = await execute_actions_from_bus(bus, state, "controverse scientifique")
    assert executed, "Sub-pipeline should have been executed"
    assert state.get("pipeline_events"), "Pipeline events must be recorded"
    first_exec = state["pipeline_events"][0]
    results = first_exec.get("results") or []
    assert any(
        step.get("result", {}).get("collection")
        for step in results
    ), "Collection step must be present in results"


def test_collect_subpipeline_linearization():
    """Linearisation du sous-pipeline de collecte: ordre des étapes stable."""
    pipe = build_collect_subpipeline("dummy", reason="test")
    steps = pipe.linearize()
    names = [s["name"] for s in steps]
    assert names[:2] == ["clarify_question", "collect_complementary"]
