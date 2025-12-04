"""
Debug CLI for SearchOne backend (developer tools).

Provides commands to run and inspect agent jobs from the terminal, with optional LLM mocking
for deterministic behavior during local debugging.

Usage (PowerShell examples are provided in README):
  python -m app.cli.debug_cli run-job --query "my question" --name debug-job --iterations 3 --mock
  python -m app.cli.debug_cli tail-job 1 --interval 1
  python -m app.cli.debug_cli show-timeline 1
  python -m app.cli.debug_cli list-jobs
  python -m app.cli.debug_cli inspect-db
  python -m app.cli.debug_cli worker

This tool is intended for local dev/debug only.
"""
import argparse
import time
import json
import asyncio
import os
import subprocess
import logging
import sys
from pathlib import Path
from typing import Optional
from sqlmodel import select

# Ensure backend dir is importable when run as a script
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.data.db import get_session, Job, init_db
from app.workflows import agents
from app.workflows import worker as worker_mod
from app.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
try:
    init_db()
except Exception:
    logger.warning("init_db() failed (database may be missing tables)", exc_info=True)


class MockLLM:
    """A very small deterministic mock LLM for debugging.

    Behavior heuristics based on prompt content to return plausible, repeatable outputs.
    """
    def __init__(self, persona: str = "mock"):
        self.persona = persona

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # simple heuristics to detect the requested task
        pl = prompt.lower()
        if 'propose' in pl and 'hypoth' in pl:
            # return a short hypothesis
            return "Proposition: L'hypothèse testée est que X influence Y."
        if 'réponse attendue' in pl or 'vote' in pl or 'agree' in pl:
            # return a vote format: VOTE -- justification
            # alternate based on persona for slight variation
            return "neutral -- Mocked: not enough evidence"
        # default: echo a short summary
        return "Résumé: mock response based on prompt snippet: " + (prompt[:120].replace('\n',' '))

    def generate_sync(self, prompt: str, max_tokens: int = 512) -> str:
        # synchronous wrapper for compatibility
        return asyncio.run(self.generate(prompt, max_tokens=max_tokens))


class MockEmbedder:
    def __init__(self):
        pass

    def encode(self, texts, *args, **kwargs):
        # return simple numeric vectors of length 3
        import numpy as _np
        out = []
        for t in texts:
            h = abs(hash(t)) % 997
            out.append([float((h % 10) / 10.0), float((h % 7) / 7.0), float((h % 5) / 5.0)])
        return _np.array(out)


class MockFaissStore:
    def __init__(self, dim=3):
        self.dim = dim
        self._items = []

    def add(self, embeddings, metas):
        for e, m in zip(embeddings, metas):
            self._items.append({'emb': list(e), 'meta': m, 'score': 1.0})

    def search(self, emb, top_k=5):
        # naive return of all items with decreasing score
        out = []
        for i, it in enumerate(self._items[:top_k]):
            out.append({'score': float(1.0 - i * 0.1), 'metadata': it['meta']})
        return out


def run_job_sync(name: str, query: str, iterations: int = 5, mock: bool = False):
    try:
        from .jobs import create_job, run_agents_job_sync
    except Exception:
        from jobs import create_job, run_agents_job_sync
    job_id = create_job(name)
    logger.info("Created job %s", job_id)
    if mock:
        llm = MockLLM()
        me = MockEmbedder()
        mfs = MockFaissStore()
        # run the async orchestrator directly with injected mock LLM + embedder + faiss
        asyncio.run(agents.run_agents_job(job_id, query, max_iterations=iterations, llm_client=llm, embedder=me, vs=mfs))
    else:
        # use the existing sync wrapper which will use real LLM
        run_agents_job_sync(job_id, query, max_iterations=iterations)


def tail_job(job_id: int, interval: float = 2.0):
    try:
        from .db import get_session
    except Exception:
        from db import get_session
    try:
        while True:
            with get_session() as s:
                job = s.get(Job, job_id)
                if not job:
                    print(f"Job {job_id} not found")
                    break
                print(f"[{job.updated_at}] status={job.status}\nState snippet:\n")
                try:
                    st = json.loads(job.state) if job.state else {}
                    print(json.dumps(st if isinstance(st, dict) else {'state': str(st)}, indent=2, ensure_ascii=False)[:4000])
                except Exception:
                    print((job.state or '')[:4000])
            time.sleep(interval)
    except KeyboardInterrupt:
        print('Stopped tailing')


def show_timeline(job_id: int):
    try:
        from .db import get_session
    except Exception:
        from db import get_session
    with get_session() as s:
        job = s.get(Job, job_id)
        if not job:
            print(f"Job {job_id} not found")
            return
        try:
            state = json.loads(job.state) if job.state else {}
        except Exception:
            state = {}
        timeline = state.get('timeline') or []
        print(json.dumps(timeline, indent=2, ensure_ascii=False))


def list_jobs():
    with get_session() as s:
        jobs = list(s.exec(select(Job).order_by(Job.created_at.desc()).limit(50)).all())
        for j in jobs:
            print(f"{j.id}\t{j.name}\t{j.status}\tupdated:{j.updated_at}")


def start_worker():
    # Start RQ worker in subprocess using the worker module
    # Note: this spawns a process which will inherit env vars
    worker_path = BACKEND_ROOT / "app" / "workflows" / "worker.py"
    cmd = [os.sys.executable, "-u", str(worker_path)]
    print('Starting worker subprocess:', ' '.join(cmd))
    # use Popen so the caller can keep using the terminal
    p = subprocess.Popen(cmd)
    print('Worker started with PID', p.pid)


def inspect_db():
    try:
        from .db import get_session, Document, Chunk
    except Exception:
        from db import get_session, Document, Chunk
    with get_session() as s:
        docs = len(list(s.exec(select(Document)).all()))
        chunks = len(list(s.exec(select(Chunk)).all()))
        jobs = len(list(s.exec(select(Job)).all()))
        print(f"Documents: {docs}\nChunks: {chunks}\nJobs: {jobs}")


def main():
    parser = argparse.ArgumentParser(prog='debug_cli', description='SearchOne developer debug CLI')
    sub = parser.add_subparsers(dest='cmd')

    p_run = sub.add_parser('run-job')
    p_run.add_argument('--name', '-n', default='debug-job')
    p_run.add_argument('--query', '-q', required=True)
    p_run.add_argument('--iterations', '-i', type=int, default=5)
    p_run.add_argument('--mock', action='store_true', help='Use MockLLM for deterministic outputs')

    p_tail = sub.add_parser('tail-job')
    p_tail.add_argument('job_id', type=int)
    p_tail.add_argument('--interval', '-t', type=float, default=2.0)

    p_tl = sub.add_parser('show-timeline')
    p_tl.add_argument('job_id', type=int)

    p_list = sub.add_parser('list-jobs')

    p_worker = sub.add_parser('worker')

    p_inspect = sub.add_parser('inspect-db')
    p_replay = sub.add_parser('replay')
    p_replay.add_argument('job_id', type=int)
    p_replay.add_argument('--mock', action='store_true', help='Use MockLLM for deterministic outputs')

    args = parser.parse_args()
    if args.cmd == 'run-job':
        run_job_sync(args.name, args.query, iterations=args.iterations, mock=args.mock)
    elif args.cmd == 'tail-job':
        tail_job(args.job_id, interval=args.interval)
    elif args.cmd == 'show-timeline':
        show_timeline(args.job_id)
    elif args.cmd == 'list-jobs':
        list_jobs()
    elif args.cmd == 'worker':
        start_worker()
    elif args.cmd == 'inspect-db':
        inspect_db()
    elif args.cmd == 'replay':
        # replay a job by re-running the orchestrator with the same query
        with get_session() as s:
            job = s.get(Job, args.job_id)
            if not job:
                print(f"Job {args.job_id} not found")
            else:
                try:
                    st = json.loads(job.state) if job.state else {}
                except Exception:
                    st = {}
                query = st.get('query') if isinstance(st, dict) else None
                if not query:
                    print('Original query not found in job state; aborting.')
                else:
                    # create a new job and run
                    print(f"Replaying job {args.job_id} as new job (mock={args.mock})")
                    run_job_sync(f"replay-of-{job.id}", query, iterations=st.get('max_iterations',5) if isinstance(st, dict) else 5, mock=args.mock)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
