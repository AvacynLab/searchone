import asyncio
from pathlib import Path
import json

from app.workflows.workflow import build_full_research_pipeline, WorkflowEngine, execute_actions_from_bus
from app.workflows.debate import tally_votes
from app.workflows.scheduler import ResearchScheduler, SCHEDULES_FILE
from app.services.references import ReferenceManager
from app.workflows.runtime import MessageBus


def test_workflow_full_pipeline_produces_decision():
    ctx = {"query": "Compression de modeles pour inference rapide"}
    wf = build_full_research_pipeline()
    engine = WorkflowEngine(wf.linearize())
    results = asyncio.get_event_loop().run_until_complete(engine.run(context=ctx))
    final_ctx = results[-1]["result"]
    assert final_ctx.get("decision", {}).get("status") in {"adopt", "replan", "investigate"}
    assert "draft" in final_ctx.get("report", {}).get("status", "")
    assert final_ctx.get("claims")


def test_debate_tally_weighted_and_veto():
    votes = [
        {"agent": "A1", "votes": [{"vote": "agree"}]},
        {"agent": "Critic", "votes": [{"vote": "disagree"}]},
    ]
    res = tally_votes(votes, mode="weighted", weights={"A1": 2.0, "Critic": 1.0})
    assert res["decision"] == "adopt"
    veto_res = tally_votes(votes, mode="weighted", weights={"A1": 1.0, "Critic": 3.0}, veto_role="Critic")
    assert veto_res["decision"] == "reject"


def test_scheduler_run_due(tmp_path, monkeypatch):
    monkeypatch.setattr("app.workflows.scheduler.SCHEDULES_FILE", tmp_path / "schedules.json")
    sched = ResearchScheduler(snapshot_dir=tmp_path)
    sched.add_schedule("test query", interval_seconds=-1)
    launched = []

    def launcher(q: str):
        launched.append(q)

    asyncio.get_event_loop().run_until_complete(sched.run_due(launcher))
    assert launched and launched[0] == "test query"
    assert sched.list_schedules()[0]["last_run"]


def test_reference_manager_dedup():
    mgr = ReferenceManager()
    mgr.add("Titre", author="A", year="2020", doi="10.1/abc")
    mgr.add("Titre", author="B", year="2021", doi="10.1/abc")
    assert len(mgr.list()) == 1
    assert mgr.find(doi="10.1/abc")["author"] == "A"


def test_execute_actions_from_bus_runs_pipeline():
    bus = MessageBus()
    bus.publish("actions", {"type": "replan_collect", "query": "test", "reason": "reject"})
    state = {"query": "test"}
    results = asyncio.get_event_loop().run_until_complete(execute_actions_from_bus(bus, state, "test"))
    assert results
    assert state["pipeline_events"]
