import asyncio
import json
import os
import sys
# Ensure the backend package is importable when tests are run from repo root or backend folder
CURRENT_DIR = os.path.dirname(__file__)
PARENT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from app.workflows import agents
from app.cli.debug_cli import MockLLM, MockEmbedder, MockFaissStore
from app.workflows.jobs import create_job, get_job


def test_run_agents_job_creates_timeline(tmp_path):
    # Initialize DB tables for test and create a job
    from app.data.db import init_db
    init_db()
    job_id = create_job('test-run')
    # run async orchestrator synchronously
    asyncio.run(agents.run_agents_job(job_id, 'Test query for agents', max_iterations=2, llm_client=MockLLM(), embedder=MockEmbedder(), vs=MockFaissStore()))
    job = get_job(job_id)
    assert job is not None
    assert job.state is not None
    st = json.loads(job.state)
    assert 'timeline' in st
    assert len(st['timeline']) >= 1
    # each timeline entry should have summary and votes
    for entry in st['timeline']:
        assert 'summary' in entry
        assert 'votes' in entry
        assert isinstance(entry['votes'], list)
