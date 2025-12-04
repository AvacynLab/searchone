import pytest

from app.services.search_oracle import (
    Evidence,
    SearchSessionState,
    plan_subqueries,
    search_internal,
    search_web_via_searx,
    update_coverage,
)


def test_plan_subqueries_respects_depth_and_focus():
    query = "Superconductivity; applications in energy"
    parts = plan_subqueries(query, max_depth=3, focus="recent")
    assert any("recent" in p for p in parts)
    assert len(parts) <= 3


class DummyEmbedder:
    def encode(self, texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.mark.asyncio
async def test_search_internal_populates_evidence(monkeypatch):
    session = SearchSessionState(job_id=1, root_query="test")
    session.vector_store = object()
    session.embedder = DummyEmbedder()

    hits = [
        {"meta": {"source": "doc1", "text": "insight", "title": "Doc 1"}, "score": 0.75}
    ]

    monkeypatch.setattr(
        "app.services.search_oracle.search_semantic",
        lambda store, embeddings, top_k=5: hits,
    )

    evidences = await search_internal(session, "foo")
    assert evidences
    assert evidences[0].source == "doc1"
    assert session.used_budget["internal"] == 1


@pytest.mark.asyncio
async def test_search_web_via_searx_builds_evidence():
    session = SearchSessionState(job_id=2, root_query="web query")

    async def fake_search(subquery, top_k=4):
        return [
            {"meta": {"source": "https://example.com"}, "text": "web snippet", "score": 0.5}
        ]

    session.web_search = fake_search
    results = await search_web_via_searx(session, "topic")
    assert results
    assert results[0].source_type == "web"
    assert session.used_budget["web"] == 1


def test_update_coverage_records_gaps():
    session = SearchSessionState(job_id=3, root_query="gap target")
    evidences = [Evidence(source="doc", snippet="text")]
    update_coverage(session, "topic", evidences[:1])
    assert session.subqueries[-1]["coverage"] > 0
    update_coverage(session, "topic2", [])
    assert "topic2" in session.gaps
