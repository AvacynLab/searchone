import json
from fastapi.testclient import TestClient
import app.api.main as main


def _long_html(text: str, repeat: int = 120) -> str:
    return "<html><body>" + (" ".join([text] * repeat)) + "</body></html>"


def test_ingest_url_and_search_hybrid(monkeypatch, tmp_path):
    """
    Smoke test: ingest a mocked URL, ensure it is indexed, and hybrid search retrieves it.
    """
    client = TestClient(main.app)
    # make domain allowed and lower length threshold for test speed
    monkeypatch.setattr(main, "_domain_allowed", lambda url: True)
    monkeypatch.setattr(main, "MIN_SOURCE_CHARS", 50)
    # reset vector store to avoid cross-test bleed
    try:
        main.vs.reset()
    except Exception:
        pass

    sample_url = "https://example.com/climate"

    def fake_download(url: str, timeout: int = 15) -> str:
        return _long_html("Climate change mitigation and adaptation strategies", repeat=40)

    monkeypatch.setattr(main, "download_url", fake_download)

    resp = client.post("/ingest/url", params={"url": sample_url, "title": "Climate doc"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["document_id"] > 0
    assert data["chunks"] > 0

    search = client.get("/search/hybrid", params={"q": "climate change", "top_k": 3, "alpha": 0.5})
    assert search.status_code == 200, search.text
    payload = search.json()
    assert payload["results"], "Hybrid search should return at least one result"
    top = payload["results"][0]
    assert "climate" in (top.get("text") or "").lower()


def test_doctor_index_status(monkeypatch):
    """
    Ensure doctor index status responds with ok and domain stats structure.
    """
    client = TestClient(main.app)
    res = client.get("/doctor/index/status")
    assert res.status_code == 200
    data = res.json()
    assert data.get("ok") is True
    assert "index" in data and "domains" in data
