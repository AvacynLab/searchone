from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health_ok():
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert 'ok' in data
    assert 'db_file' in data
