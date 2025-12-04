import httpx
from sqlmodel import delete, select
import pytest

from app.workflows.agents import (
    ENGINE_FAILURE_STATE,
    _build_cache_key,
    _lookup_web_cache,
    run_web_search,
    WEB_SEARCH_ENGINE_NAME,
    WEB_SEARCH_ENGINE_SET,
)
from app.data.db import WebCache, get_session, init_db


class FakeAsyncClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None, **kwargs):
        FakeAsyncClient.calls += 1
        if FakeAsyncClient.calls == 1:
            req = httpx.Request("GET", url)
            resp = httpx.Response(429, request=req)
            raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)
        html = '<div class="result"><a href="http://example.com">Title</a><p>Snippet</p></div>'
        return httpx.Response(200, content=html.encode("utf-8"), request=httpx.Request("GET", url))


@pytest.mark.asyncio
async def test_web_search_resilience(monkeypatch):
    FakeAsyncClient.calls = 0
    ENGINE_FAILURE_STATE.clear()
    init_db()
    with get_session() as session:
        session.exec(delete(WebCache))
        session.commit()

    monkeypatch.setattr("app.workflows.agents.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("app.workflows.agents.WEB_SEARCH_ENDPOINT", "http://dummy")

    query = "résilience searxng"

    results = await run_web_search(query, top_k=2)
    assert results, "Fallback results should be returned"
    assert FakeAsyncClient.calls == 2, "Two HTTP attempts should have occurred"

    engine_state = ENGINE_FAILURE_STATE.get(WEB_SEARCH_ENGINE_NAME)
    assert engine_state is not None, "Engine failure state should exist after a failed call"

    cached = _lookup_web_cache(query, "fr", True, WEB_SEARCH_ENGINE_SET or WEB_SEARCH_ENGINE_NAME)
    assert cached, "Results should be cached"

    cache_key = _build_cache_key(query, "fr", True, WEB_SEARCH_ENGINE_SET or WEB_SEARCH_ENGINE_NAME)
    previous_calls = FakeAsyncClient.calls
    cached_results = await run_web_search(query, top_k=1)
    assert cached_results == cached
    assert FakeAsyncClient.calls == previous_calls, "No additional HTTP calls when cache is hit"

    with get_session() as session:
        entry = session.exec(select(WebCache).where(WebCache.cache_key == cache_key)).first()
        if entry:
            session.delete(entry)
            session.commit()
