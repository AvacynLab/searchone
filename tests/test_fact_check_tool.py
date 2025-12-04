import pytest

from app.workflows.agents import _execute_tool


class DummyAgent:
    def __init__(self, evidence=None):
        self.job_id = 1
        self.name = "tester"
        self.job_state = {}
        self._evidence = evidence or []

    async def retrieve_evidence(self, query, top_k=3):
        return self._evidence if self._evidence and "supported" in query else []


@pytest.mark.asyncio
async def test_fact_check_tool_marks_supported(monkeypatch):
    results = []
    agent = DummyAgent(evidence=[{"meta": {"source": "doc1"}, "text": "info"}])

    monkeypatch.setattr("app.workflows.agents.record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.workflows.agents.store_fact_check_result", lambda outcome: results.append(outcome))

    summary = await _execute_tool("fact_check_tool", {"claim": "supported claim"}, agent)
    assert summary
    assert results
    fact = results[-1]
    assert fact["verdict"] == "supported"
    assert agent.job_state["fact_checks"][0]["verdict"] == "supported"


@pytest.mark.asyncio
async def test_fact_check_tool_marks_uncertain_without_hits(monkeypatch):
    results = []
    agent = DummyAgent()

    async def fake_web_search(query, top_k=3):
        return []

    monkeypatch.setattr("app.workflows.agents.record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.workflows.agents.store_fact_check_result", lambda outcome: results.append(outcome))
    monkeypatch.setattr("app.workflows.agents.run_web_search", fake_web_search)

    summary = await _execute_tool("fact_check_tool", {"claim": "uncertain claim"}, agent)
    assert summary
    assert results
    assert results[-1]["verdict"] == "uncertain"
