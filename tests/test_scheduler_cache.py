import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import SQLModel, create_engine

from app.data import db as data_db
from app.workflows.scheduler import ResearchScheduler, _normalize_spec, _next_run_from_spec
from app.core.config import APP_VERSION


@pytest.fixture
def in_memory_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(data_db, "engine", engine)
    return engine


def _build_scheduler(tmp_path, monkeypatch):
    schedule_file = tmp_path / "schedules.json"
    monkeypatch.setattr("backend.app.workflows.scheduler.SCHEDULES_FILE", schedule_file)
    return ResearchScheduler(snapshot_dir=tmp_path / "snapshots")


def test_normalize_and_next_run():
    interval_spec = _normalize_spec({"type": "interval", "seconds": 5}, None)
    cron_spec = _normalize_spec({"type": "cron", "expr": "*/1 * * * *"}, None)
    iso_spec = _normalize_spec({"type": "iso", "expr": "R/P1DT2H"}, None)

    assert interval_spec["type"] == "interval"
    assert cron_spec["expr"] == "*/1 * * * *"

    base = datetime.now(timezone.utc)
    assert _next_run_from_spec(interval_spec, base) > base
    assert _next_run_from_spec(cron_spec, base) > base
    assert _next_run_from_spec(iso_spec, base) > base


def test_scheduler_resume_snapshot(tmp_path, monkeypatch, in_memory_db):
    scheduler = _build_scheduler(tmp_path, monkeypatch)
    scheduler.add_schedule("test query", interval_seconds=1)
    snapshot_path = tmp_path / "snapshots" / "run_101.json"
    snapshot_path.write_text(json.dumps({"job_id": 101, "version": "0.0.0", "state": {}}), encoding="utf-8")

    resumed = scheduler.resume_from_snapshot(101)
    assert resumed is not None
    assert resumed["version"] == APP_VERSION
    assert "restored_at" in resumed


def test_web_cache_entry_lifecycle(tmp_path, in_memory_db):
    now = datetime.now(timezone.utc)
    data_db.store_web_cache_entry(
        "key1",
        "query",
        "fr",
        True,
        "eng",
        "{}",
        now + timedelta(seconds=60),
    )
    fetched = data_db.get_web_cache_entry("key1")
    assert fetched is not None

    data_db.store_web_cache_entry(
        "key2",
        "query2",
        "fr",
        True,
        "eng",
        "{}",
        now - timedelta(seconds=10),
    )
    removed = data_db.cleanup_web_cache_entries()
    assert removed >= 1
    assert data_db.get_web_cache_entry("key2") is None
