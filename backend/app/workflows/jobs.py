import time
import json
import threading
import os
import platform
import logging
from rq import Queue
from redis import Redis
from typing import Dict, Any
from app.data.db import get_session, Job, save_job_state, Document, Chunk
from app.services.llm import LLMClient
import app.workflows.agents
from app.data.vector_store import FaissStore
from app.core.config import EMBEDDING_MODEL, validate_config
from app.core.stop_flags import clear_stop
from sqlmodel import select
env_simple = os.getenv("SEARCHONE_SIMPLE_EMBEDDER")
env_force_st = os.getenv("SEARCHONE_FORCE_ST")
default_simple = platform.system().lower() == "windows"
USE_SIMPLE_EMBEDDER = (
    env_simple.lower() in ("1", "true", "yes", "on") if env_simple is not None else default_simple
)
if env_force_st and env_force_st.lower() in ("1", "true", "yes", "on"):
    USE_SIMPLE_EMBEDDER = False

if not USE_SIMPLE_EMBEDDER:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _HAS_ST = True
    except Exception:
        SentenceTransformer = None
        _HAS_ST = False
else:
    SentenceTransformer = None
    _HAS_ST = False

# lightweight fallback embedder used when sentence-transformers is not installed or disabled
if not _HAS_ST:
    class SimpleEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, *args, **kwargs):
            import numpy as _np
            out = []
            for t in texts:
                h = abs(hash(t)) % 997
                out.append([float((h % 10) / 10.0), float((h % 7) / 7.0), float((h % 5) / 5.0)])
            return _np.array(out)

        def get_sentence_embedding_dimension(self):
            return 3
from app.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
validate_config()

# Simple in-process job runner. For production, replace with Redis/RQ or Celery.

logger.info("Initializing embedder and vector store for jobs module")
if _HAS_ST and SentenceTransformer is not None:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
else:
    embedder = SimpleEmbedder()
EMB_DIM = 3
try:
    EMB_DIM = embedder.get_sentence_embedding_dimension()
except Exception:
    EMB_DIM = 3
vs = FaissStore(dim=EMB_DIM)
llm = LLMClient()
job_threads: Dict[int, threading.Thread] = {}


def warm_vector_store_from_db():
    """Load existing chunks from DB into the jobs module FAISS index."""
    try:
        ntotal = vs.index.ntotal if getattr(vs, "index", None) is not None else len(getattr(vs, "_embs", []))
        if ntotal > 0:
            return
    except Exception:
        pass
    try:
        with get_session() as s:
            stmt = select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)
            chunks = list(s.exec(stmt).all())
        if not chunks:
            return
        texts = [c.text for c in chunks]
        metas = [{"document_id": c.document_id, "chunk_index": c.chunk_index} for c in chunks]
        emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vs.add(emb, metas)
        logger.info("Warm-loaded %d chunks into jobs FAISS index", len(chunks))
    except Exception as e:
        logger.warning("Could not warm-load FAISS in jobs module: %s", e, exc_info=True)


# pre-load existing chunks so agents can search ingested sources
warm_vector_store_from_db()


def _persist_state(job_id: int, state: Dict[str, Any], status: str = None):
    logger.debug("Persisting state for job %s (status=%s)", job_id, status)
    save_job_state(job_id, json.dumps(state, ensure_ascii=False), status=status)


def _load_chunks_for_meta(meta):
    # meta: {document_id, chunk_index}
    with get_session() as s:
        stmt = select(Chunk).where(Chunk.document_id == meta["document_id"], Chunk.chunk_index == meta["chunk_index"])
        ch = s.exec(stmt).first()
        return ch.text if ch else None


def _llm_generate_with_retry(llm: LLMClient, prompt: str, max_tokens: int, label: str, attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    last_err = None
    for i in range(attempts):
        try:
            return llm.generate_sync(prompt, max_tokens=max_tokens)
        except Exception as e:
            last_err = e
            logger.warning("LLM sync call failed (%s) attempt %d/%d: %s", label, i + 1, attempts, e, exc_info=True)
            if i < attempts - 1:
                import time as _t
                _t.sleep(delay)
                delay *= backoff
    err_msg = f"LLM error after retries: {last_err}"
    logger.exception("LLM sync call failed after %d attempts (%s): %s", attempts, label, last_err)
    return f"[{err_msg}]"


def agent_loop(job_id: int, query: str, max_iterations: int = 5, sleep_between: int = 5, max_failures: int = 3):
    # Very simple agent loop: propose hypotheses, search for evidence, evaluate, checkpoint state
    state = {"query": query, "hypotheses": [], "evidence": [], "notes": ""}
    failure_count = 0
    logger.info("Starting agent loop for job %s (query=%s)", job_id, query)
    _persist_state(job_id, state, status="running")
    try:
        for iteration in range(max_iterations):
            prompt = (
                "Propose 3 hypotheses claires (une par ligne) pour la requete suivante:\n"
                f"{query}\n" +
                ("Hypotheses precedentes:\n" + "\n".join(state['hypotheses']) if state['hypotheses'] else "")
            )
            text = _llm_generate_with_retry(llm, prompt, max_tokens=400, label=f"job{job_id}-iter{iteration+1}")
            state['hypotheses'].append(text)
            # for each hypothesis, do a retrieval
            for h in [text]:
                # simple search using hypothesis as query
                try:
                    emb = embedder.encode([h], convert_to_numpy=True, show_progress_bar=False)
                    hits = vs.search(emb, top_k=5)
                    logger.debug("Job %s retrieval got %d hits", job_id, len(hits))
                except Exception as e:
                    logger.exception("Retrieval failed in job %s: %s", job_id, e)
                    state['last_error'] = f"retrieval: {e}"
                    hits = []
                for hit in hits:
                    meta = hit.get('metadata')
                    snippet = _load_chunks_for_meta(meta) if meta else None
                    ev = {
                        "hypothesis": h[:200],
                        "score": hit.get('score'),
                        "document_id": meta.get('document_id') if meta else None,
                        "chunk_index": meta.get('chunk_index') if meta else None,
                        "text": snippet
                    }
                    state['evidence'].append(ev)
            # quick evaluation: count hits with score>0.2
            good = [e for e in state['evidence'] if e['score'] and e['score'] > 0.2]
            state['notes'] = f"Iteration {iteration+1}: {len(good)} strong evidences found."
            _persist_state(job_id, state, status="running")
            time.sleep(sleep_between)
        _persist_state(job_id, state, status="completed")
        logger.info("Job %s completed", job_id)
    except Exception as e:
        failure_count += 1
        state['notes'] += f"\nError: {e}"
        _persist_state(job_id, state, status="failed" if failure_count >= max_failures else "running")
        logger.exception("Unhandled error in job %s: %s", job_id, e)
        if failure_count < max_failures:
            return False
        return


def start_job_background(job_id: int, query: str, max_duration_seconds: int = 300, max_token_budget: int = 0, max_iterations: int = 5):
    # If REDIS_URL present, enqueue job in RQ; otherwise run in-process thread
    clear_stop(job_id)  # ensure stale stop flags do not kill new runs immediately
    redis_url = os.getenv('REDIS_URL')
    if redis_url:
        try:
            redis_conn = Redis.from_url(redis_url)
            redis_conn.ping()
            logger.info("Enqueuing job %s to RQ (redis=%s)", job_id, redis_url)
            q = Queue('searchone', connection=redis_conn)
            # mark as queued
            save_job_state(job_id, "", status="queued")
            # enqueue the agents orchestrator wrapper so RQ worker runs it
            q.enqueue(
                run_agents_job_sync,
                job_id,
                query,
                max_iterations,
                max_duration_seconds,
                max_token_budget,
                job_id=str(job_id),
                meta={"db_job_id": job_id},
            )
            return True
        except Exception as e:
            logger.warning("Redis unreachable (%s), falling back to in-process execution: %s", redis_url, e, exc_info=True)
    else:
        logger.info("Starting in-process thread for job %s", job_id)
        save_job_state(job_id, "", status="running")
        # run the async orchestrator in a separate thread
        def _start():
            try:
                import asyncio
                asyncio.run(agents.run_agents_job(job_id, query, max_iterations=max_iterations, max_duration_seconds=max_duration_seconds, max_token_budget=max_token_budget))
            except Exception as e:
                logger.exception("Error running agents job in-thread: %s", e)
            finally:
                job_threads.pop(job_id, None)
                clear_stop(job_id)

        t = threading.Thread(target=_start, daemon=True)
        t.start()
        job_threads[job_id] = t
        return True

# convenience API functions for creating job entries


def create_job(name: str, priority: int = 0) -> int:
    with get_session() as s:
        job = Job(name=name, priority=priority)
        s.add(job)
        s.commit()
        s.refresh(job)
        return job.id


def get_job(job_id: int):
    with get_session() as s:
        return s.get(Job, job_id)


def run_agents_job_sync(job_id: int, query: str, max_iterations: int = 5, max_duration_seconds: int = 300, max_token_budget: int = 0):
    """Synchronous wrapper to run the async agents orchestrator. Use this for RQ workers."""
    try:
        import asyncio
        # mark job as running when the worker starts
        save_job_state(job_id, json.dumps({"status": "running"}), status="running")
        # run with default (real) LLM client
        asyncio.run(agents.run_agents_job(job_id, query, max_iterations=max_iterations, max_duration_seconds=max_duration_seconds, max_token_budget=max_token_budget))
    except Exception as e:
        logger.exception("run_agents_job_sync failed: %s", e)
        # persist failure
        save_job_state(job_id, json.dumps({"error": str(e)}), status='failed')
        raise


def cancel_rq_job(job_id: int) -> bool:
    """Try to cancel a queued RQ job and mark its meta as stop_requested."""
    redis_url = os.getenv('REDIS_URL')
    if not redis_url:
        return False
    try:
        from rq.job import Job as RQJob
        conn = Redis.from_url(redis_url)
        rq_job = RQJob.fetch(str(job_id), connection=conn)
    except Exception as e:
        logger.debug("No RQ job to cancel for %s: %s", job_id, e)
        return False
    try:
        rq_job.meta["stop_requested"] = True
        rq_job.save_meta()
    except Exception:
        pass
    try:
        rq_job.cancel()
        return True
    except Exception as e:
        logger.debug("Failed to cancel RQ job %s: %s", job_id, e)
        return False


def rename_job(job_id: int, new_name: str) -> bool:
    """Update a job name."""
    if not new_name:
        return False
    with get_session() as s:
        job = s.get(Job, job_id)
        if not job:
            return False
        job.name = new_name
        job.updated_at = datetime.now(timezone.utc)
        s.add(job)
        s.commit()
        return True


def soft_delete_job(job_id: int) -> bool:
    """Soft-delete a job by marking status deleted and clearing state string."""
    with get_session() as s:
        job = s.get(Job, job_id)
        if not job:
            return False
        job.status = "deleted"
        job.state = job.state or ""
        job.updated_at = datetime.now(timezone.utc)
        s.add(job)
        s.commit()
        return True
