import json
from pathlib import Path
from app.data import action_log


def test_record_action_and_find_similar(tmp_path):
    temp = tmp_path / "datadir"
    temp.mkdir()
    action_log.DATA_DIR = temp
    action_log.ACTION_LOG_FILE = temp / "action_log.jsonl"
    if action_log.ACTION_LOG_FILE.exists():
        action_log.ACTION_LOG_FILE.unlink()

    hits = [{"meta": {"source": "doc1"}}]
    record = action_log.record_action(1, "A1", "web_search", "Quantum entanglement", hits)
    assert record.job_id == 1
    assert "quantum entanglement" in record.normalized_query

    similar = action_log.find_similar_actions(1, record.normalized_query, "web_search")
    assert any(entry.get("result_hash") == record.result_hash for entry in similar)

    # different normalized query should not match
    no_match = action_log.find_similar_actions(1, "different query", "web_search")
    assert not no_match
