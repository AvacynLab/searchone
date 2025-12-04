from fastapi.testclient import TestClient
from app.api.main import app
from app.core.prompt_state import set_system_prompt, set_prompt_variant


client = TestClient(app)


def test_set_and_get_system_prompt():
    set_system_prompt("")  # reset
    resp = client.post("/prompts/system", params={"prompt": "hello system"})
    assert resp.status_code == 200
    assert resp.json().get("system_prompt") == "hello system"

    got = client.get("/prompts/system")
    assert got.status_code == 200
    assert got.json().get("system_prompt") == "hello system"

    set_system_prompt("")  # cleanup


def test_set_and_get_variant():
    set_prompt_variant(None)  # reset
    resp = client.post("/prompts/variant", params={"variant": "concise"})
    assert resp.status_code == 200
    assert resp.json().get("variant") == "concise"

    got = client.get("/prompts/variant")
    assert got.status_code == 200
    assert got.json().get("variant") == "concise"

    set_prompt_variant(None)  # cleanup
