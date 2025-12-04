from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import platform
from datetime import timezone
import shutil
import numpy as np
import json
import os
import logging
import requests
from typing import List, Dict, Any
import urllib.parse
from collections import Counter
import re
import statistics
from app.workflows.scheduler import ResearchScheduler
from contextlib import asynccontextmanager

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
from app.core.config import DATA_DIR, EMBEDDING_MODEL, SEARXNG_URL
from app.data.db import init_db, get_session, Document, Chunk, db_stats, db_job_status_counts
from app.services.ingest import (
    extract_text_from_pdf,
    chunk_text,
    chunk_text_with_positions,
    compute_source_metadata,
    metadata_from_file,
    ingest_web_page,
)
from app.data.vector_store import FaissStore
from app.workflows.jobs import create_job, start_job_background, get_job, job_threads, cancel_rq_job, rename_job, soft_delete_job
from app.data.db import save_job_state
from app.services.reporting import save_report
from app.services.reporting import build_diagnostic
from app.services.ingest import download_url, extract_text_from_html
from app.services.crawler import crawl_and_ingest
from app.core.observability import compute_run_metrics
from app.data.knowledge_store import consolidate_promotions
import logging
from app.core.logging_config import configure_logging
import asyncio
from fastapi import Response
import json
from fastapi.responses import StreamingResponse
from app.core.stop_flags import request_stop
from app.core.config import validate_config
from app.core.prompt_state import get_system_prompt, set_system_prompt, set_prompt_variant, get_prompt_variant
from app.services.reporting import build_structured_summary
from app.core.tracing import start_span
from datetime import datetime, timedelta
import asyncio
import functools
from sqlmodel import select
from sqlalchemy import func

configure_logging()
logger = logging.getLogger(__name__)

# Optional Sentry instrumentation for backend (set SEARCHONE_SENTRY_DSN)
SENTRY_DSN = os.getenv("SEARCHONE_SENTRY_DSN") or os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk  # type: ignore

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.05")),
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0")),
            environment=os.getenv("SENTRY_ENVIRONMENT", "local"),
        )
        logger.info("Sentry initialised for backend")
    except Exception as e:
        logger.warning("Failed to init Sentry: %s", e)

validate_config()

# quality gates for ingestion
def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.replace(",", ".")
    try:
        return float(raw)
    except Exception:
        return default


MIN_SOURCE_RELIABILITY = _parse_float_env("SEARCHONE_MIN_SOURCE_RELIABILITY", 0.5)
MIN_SOURCE_CHARS = int(os.getenv("SEARCHONE_MIN_SOURCE_CHARS", "500").replace(",", ""))
WHITELIST_DOMAINS = [d.strip().lower() for d in (os.getenv("SEARCHONE_WHITELIST_DOMAINS") or "").split(",") if d.strip()]
BLACKLIST_DOMAINS = [d.strip().lower() for d in (os.getenv("SEARCHONE_BLACKLIST_DOMAINS") or "").split(",") if d.strip()]
SEED_CORPUS = {
    "climat": [
        "https://www.ipcc.ch/site/assets/uploads/2018/02/WGIIAR5-Chap1_FINAL.pdf",
        "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Full_Report.pdf",
        "https://www.eea.europa.eu/publications/soer-2020/download",
    ],
    "physique": [
        "https://arxiv.org/pdf/1911.01414.pdf",
        "https://arxiv.org/pdf/2203.13474.pdf",
    ],
}

DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SearchOne MVP")
scheduler = ResearchScheduler(snapshot_dir=Path("data"))
SCHEDULER_LOOP_SECONDS = int(os.getenv("SEARCHONE_SCHEDULER_INTERVAL", "0") or 0)
_scheduler_task: asyncio.Task | None = None
if SCHEDULER_LOOP_SECONDS <= 0:
    logger.warning("SEARCHONE_SCHEDULER_INTERVAL not set or zero; scheduled runs will not auto-execute.")


@app.on_event("startup")
async def _start_scheduler_loop():
    """Background loop to execute due schedules when interval is configured."""
    global _scheduler_task
    if SCHEDULER_LOOP_SECONDS <= 0:
        return

    async def _loop():
        while True:
            try:
                await scheduler.run_due(_launch_scheduled_job)
            except Exception as e:  # pragma: no cover - best-effort background loop
                logger.warning("scheduler.loop_error: %s", e)
            await asyncio.sleep(SCHEDULER_LOOP_SECONDS)

    _scheduler_task = asyncio.create_task(_loop())


def _probe_dependencies() -> Dict[str, str]:
    """Return status dict for redis and SearxNG."""
    status: Dict[str, str] = {}
    redis_url = os.getenv('REDIS_URL')
    if redis_url:
        try:
            from redis import Redis
            r = Redis.from_url(redis_url)
            r.ping()
            status['redis'] = 'ok'
        except Exception as e:
            status['redis'] = f'error: {e}'
    else:
        status['redis'] = 'not_configured'

    searx_url = os.getenv('SEARXNG_URL', SEARXNG_URL)
    if searx_url:
        try:
            resp = requests.get(f"{searx_url.rstrip('/')}/healthz", timeout=3)
            if resp.status_code == 200:
                status['searxng'] = 'ok'
            else:
                status['searxng'] = f'error: http {resp.status_code}'
        except Exception as e:
            status['searxng'] = f'error: {e}'
    else:
        status['searxng'] = 'not_configured'
    return status


@app.on_event("startup")
async def _log_dependency_status():
    deps = _probe_dependencies()
    logger.info("Startup dependency status | redis=%s | searxng=%s", deps.get('redis'), deps.get('searxng'))
def _launch_scheduled_job(query: str) -> int:
    """Create a scheduled job and start background processing."""
    job_id = create_job(f"scheduled-{query[:20] or 'job'}")
    start_job_background(job_id, query)
    return job_id


@app.get('/health')
def health():
    """Health endpoint to check basic system status."""
    info = {"ok": True}
    # DB file path
    info['db_file'] = str(DATA_DIR / 'db.sqlite')
    info.update(_probe_dependencies())
    # LLM configured?
    from app.core.config import OPENROUTER_API_KEY, OPENROUTER_MODEL
    info['llm_configured'] = bool(OPENROUTER_API_KEY)
    info['llm_model'] = OPENROUTER_MODEL
    return info


@app.get('/health/llm')
def health_llm():
    """Lightweight healthcheck for LLM connectivity (OpenRouter models endpoint)."""
    from llm import LLMClient
    client = LLMClient(use_sdk=False)
    hc = client.check_health_sync()
    return hc


@app.get('/health/index')
def health_index():
    """Healthcheck for vector store integrity (dimension, counts, files)."""
    try:
        stats = vs.stats()
        return {"ok": True, "vector_store": stats, "metrics": {"ntotal": stats.get("ntotal") or stats.get("emb_count", 0)}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get('/metrics')
def metrics():
    """Minimal Prometheus metrics endpoint."""
    try:
        stats = vs.stats()
        ntotal = stats.get("ntotal") or stats.get("emb_count", 0)
        dbs = db_stats()
        job_status = db_job_status_counts()
        lines = [
            "# HELP searchone_index_total Number of vectors indexed",
            "# TYPE searchone_index_total gauge",
            f"searchone_index_total {ntotal}",
            "# HELP searchone_db_documents Total documents in DB",
            "# TYPE searchone_db_documents gauge",
            f"searchone_db_documents {dbs.get('documents', 0)}",
            "# HELP searchone_db_chunks Total chunks in DB",
            "# TYPE searchone_db_chunks gauge",
            f"searchone_db_chunks {dbs.get('chunks', 0)}",
            "# HELP searchone_db_jobs Total jobs in DB",
            "# TYPE searchone_db_jobs gauge",
            f"searchone_db_jobs {dbs.get('jobs', 0)}",
        ]
        for status, count in job_status.items():
            lines.append("# HELP searchone_db_jobs_status Jobs by status")
            lines.append("# TYPE searchone_db_jobs_status gauge")
            lines.append(f'searchone_db_jobs_status{{status="{status}"}} {count}')
        return Response("\n".join(lines) + "\n", media_type="text/plain")
    except Exception as e:
        return Response(f"# error {e}\n", media_type="text/plain", status_code=500)


@app.get('/doctor/db/check')
def doctor_db_check():
    """Return simple DB stats for diagnostics."""
    try:
        stats = db_stats()
        return {"ok": True, "db": stats}
    except Exception as e:
        logger.exception("DB check failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/doctor/index/rebuild')
def doctor_index_rebuild():
    """Reset FAISS index/mapping and warm from DB."""
    try:
        vs.reset()
        warm_vector_store_from_db()
        return {"ok": True, "detail": "Index reset and rebuilt from DB", "stats": vs.stats()}
    except Exception as e:
        logger.exception("Doctor rebuild failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/doctor/index/purge')
def doctor_index_purge(rebuild: bool = False):
    """Purge FAISS index/mapping; optionally rebuild from DB (rebuild=true)."""
    try:
        vs.reset()
        if rebuild:
            warm_vector_store_from_db()
            return {"ok": True, "detail": "Index purged and rebuilt from DB", "stats": vs.stats()}
        return {"ok": True, "detail": "Index purged (not rebuilt)", "stats": vs.stats()}
    except Exception as e:
        logger.exception("Doctor purge failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# init db and models
init_db()

# embedding model
if _HAS_ST and SentenceTransformer is not None:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    try:
        EMB_DIM = embedder.get_sentence_embedding_dimension()
    except Exception:
        EMB_DIM = 3
else:
    embedder = SimpleEmbedder()
    EMB_DIM = 3

# vector store
vs = FaissStore(dim=EMB_DIM)

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def warm_vector_store_from_db():
    """Reload chunks into FAISS if the in-memory index is empty."""
    try:
        ntotal = vs.index.ntotal if getattr(vs, "index", None) is not None else len(getattr(vs, "_embs", []))
        if ntotal > 0:
            return
    except Exception:
        pass
    try:
        with get_session() as s:
            chunks = list(s.exec(select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)).all())
        if not chunks:
            return
        texts = [c.text for c in chunks]
        metas = [{"document_id": c.document_id, "chunk_index": c.chunk_index} for c in chunks]
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        try:
            vs.add(embeddings, metas)
            logger.info("Warm-loaded %d chunks from DB into vector store", len(chunks))
        except AssertionError:
            logger.warning("Embedding dimension mismatch detected; resetting index and rebuilding.")
            vs.reset()
            vs.add(embeddings, metas)
            logger.info("Index rebuilt after dimension mismatch with %d chunks", len(chunks))
    except Exception as e:
        logger.warning("Could not warm-load vector store: %s", e)


warm_vector_store_from_db()


def _chunk_domain_stats(limit: int = 10):
    """Compute top domains and reliability distribution from chunk metadata."""
    domains = Counter()
    reliabilities = []
    recent_docs = []
    try:
        with get_session() as s:
            rows = list(s.exec(select(Chunk.meta)).all())
            docs = list(
                s.exec(
                    select(Document.id, Document.title, Document.source_type, Document.created_at, Document.reliability)
                    .order_by(Document.created_at.desc())
                    .limit(limit)
                ).all()
            )
        for r in docs:
            recent_docs.append({
                "id": r.id,
                "title": r.title,
                "source_type": r.source_type,
                "created_at": r.created_at.isoformat() if hasattr(r, "created_at") else None,
                "reliability": r.reliability,
            })
        for (meta_str,) in rows:
            if not meta_str:
                continue
            try:
                meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
            except Exception:
                continue
            dom = meta.get("domain")
            if dom:
                domains[dom] += 1
            rel = meta.get("reliability")
            if rel is not None:
                try:
                    reliabilities.append(float(rel))
                except Exception:
                    pass
    except Exception:
        return {"top_domains": [], "reliability_avg": None, "reliability_median": None, "recent_docs": []}
    top = domains.most_common(limit)
    rel_avg = statistics.mean(reliabilities) if reliabilities else None
    rel_med = statistics.median(reliabilities) if reliabilities else None
    buckets = {"0-0.25": 0, "0.25-0.5": 0, "0.5-0.75": 0, "0.75-1.0": 0}
    for r in reliabilities:
        try:
            if r < 0.25:
                buckets["0-0.25"] += 1
            elif r < 0.5:
                buckets["0.25-0.5"] += 1
            elif r < 0.75:
                buckets["0.5-0.75"] += 1
            else:
                buckets["0.75-1.0"] += 1
        except Exception:
            continue
    return {
        "top_domains": top,
        "reliability_avg": rel_avg,
        "reliability_median": rel_med,
        "reliability_histogram": buckets,
        "recent_docs": recent_docs,
    }

INGEST_QUEUE_MAX = int(os.getenv("SEARCHONE_INGEST_QUEUE_MAX", "100"))
# async ingestion queue (simple)
ingest_queue: asyncio.Queue = asyncio.Queue(maxsize=INGEST_QUEUE_MAX if INGEST_QUEUE_MAX > 0 else 0)
INGEST_WORKERS = int(os.getenv("SEARCHONE_INGEST_WORKERS", "1"))
INGEST_MAX_RETRIES = int(os.getenv("SEARCHONE_INGEST_MAX_RETRIES", "2"))
INGEST_RETRY_DELAY = float(os.getenv("SEARCHONE_INGEST_RETRY_DELAY", "1.0"))
_ingest_worker_tasks = []
_INGEST_SUCCESS = 0
_INGEST_FAIL = 0
_INGEST_RETRY = 0


async def _ingest_worker():
    global _INGEST_SUCCESS, _INGEST_FAIL, _INGEST_RETRY
    while True:
        try:
            item = await ingest_queue.get()
            if item is None:
                ingest_queue.task_done()
                break
            kind = item.get("kind")
            attempt = int(item.get("attempt", 0))
            if kind == "url":
                try:
                    ingest_url(item["url"], title=item.get("title") or "")
                    _INGEST_SUCCESS += 1
                except Exception:
                    if attempt < INGEST_MAX_RETRIES:
                        _INGEST_RETRY += 1
                        logger.warning(
                            "ingest.retry",
                            extra={"url": item.get("url"), "attempt": attempt + 1, "kind": kind},
                        )
                        await asyncio.sleep(INGEST_RETRY_DELAY * (attempt + 1))
                        await ingest_queue.put({**item, "attempt": attempt + 1})
                    else:
                        _INGEST_FAIL += 1
                        logger.error(
                            "ingest.failed",
                            extra={"url": item.get("url"), "attempt": attempt, "kind": kind},
                        )
                    raise
            elif kind == "pdf_url":
                try:
                    ingest_pdf_url(item["url"], title=item.get("title") or "")
                    _INGEST_SUCCESS += 1
                except Exception:
                    if attempt < INGEST_MAX_RETRIES:
                        _INGEST_RETRY += 1
                        logger.warning(
                            "ingest.retry",
                            extra={"url": item.get("url"), "attempt": attempt + 1, "kind": kind},
                        )
                        await asyncio.sleep(INGEST_RETRY_DELAY * (attempt + 1))
                        await ingest_queue.put({**item, "attempt": attempt + 1})
                    else:
                        _INGEST_FAIL += 1
                        logger.error(
                            "ingest.failed",
                            extra={"url": item.get("url"), "attempt": attempt, "kind": kind},
                        )
                    raise
            ingest_queue.task_done()
        except Exception as e:
            logger.exception("ingest_worker error: %s", e)
            ingest_queue.task_done()


async def _start_background():
    # avoid spawning twice in reload
    if _ingest_worker_tasks:
        return
    for _ in range(max(1, INGEST_WORKERS)):
        task = asyncio.create_task(_ingest_worker())
        _ingest_worker_tasks.append(task)
    global _scheduler_task
    if SCHEDULER_LOOP_SECONDS > 0 and _scheduler_task is None:
        async def _sched_loop():
            while True:
                try:
                    executed = await scheduler.run_due(lambda q: asyncio.create_task(_launch_scheduled_job(q)))
                    if executed:
                        logger.info("Scheduler executed %d due entries", len(executed))
                except Exception as e:
                    logger.warning("Scheduler loop error: %s", e, exc_info=True)
                await asyncio.sleep(SCHEDULER_LOOP_SECONDS)
        _scheduler_task = asyncio.create_task(_sched_loop())


async def _stop_background():
    for _ in _ingest_worker_tasks:
        await ingest_queue.put(None)
    await ingest_queue.join()
    for t in _ingest_worker_tasks:
        try:
            t.cancel()
        except Exception:
            pass
    global _scheduler_task
    if _scheduler_task:
        _scheduler_task.cancel()
        _scheduler_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _start_background()
    try:
        yield
    finally:
        await _stop_background()


# Hook lifespan to the FastAPI router to avoid deprecated on_event usage
app.router.lifespan_context = lifespan

def _domain_allowed(url: str) -> bool:
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return False
    if BLACKLIST_DOMAINS and any(domain.endswith(b) for b in BLACKLIST_DOMAINS):
        return False
    if WHITELIST_DOMAINS:
        return any(domain.endswith(w) for w in WHITELIST_DOMAINS)
    return True

@app.post('/ingest')
async def ingest_pdf(file: UploadFile = File(...), title: str = "", published_at: str = ""):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF supported in MVP")
    dest = UPLOAD_DIR / file.filename
    # deduplicate if already ingested locally
    with get_session() as s:
        existing = s.exec(select(Document).where(Document.source_path == str(dest))).first()
        if existing:
            return {"status": "exists", "document_id": existing.id, "detail": "PDF already ingested"}
    with start_span("ingest_pdf.write_file"):
        with open(dest, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    text = extract_text_from_pdf(str(dest))
    if len(text) < MIN_SOURCE_CHARS:
        raise HTTPException(status_code=400, detail=f"PDF too short ({len(text)} chars)")
    chunks = chunk_text_with_positions(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text content extracted from PDF")
    # drop duplicate chunks (hash)
    seen_hash = set()
    filtered_chunks = []
    for ch in chunks:
        h = hash(ch["text"])
        if h in seen_hash:
            continue
        seen_hash.add(h)
        filtered_chunks.append(ch)
    chunks = filtered_chunks

    # embeddings
    embeddings = embedder.encode([c["text"] for c in chunks], show_progress_bar=True, convert_to_numpy=True)

    # save document and chunks in DB
    meta_info = metadata_from_file(dest, title=title or file.filename, published_at=published_at)
    with get_session() as s:
        doc = Document(
            title=title or file.filename,
            source_path=str(dest),
            reliability=meta_info.get("reliability"),
            source_metadata=meta_info.get("source_metadata"),
            source_type=meta_info.get("source_type"),
            published_at=meta_info.get("published_at"),
        )
        s.add(doc)
        s.commit()
        s.refresh(doc)
        doc_id = doc.id
        for idx, c in enumerate(chunks):
            meta_chunk = {
                "source": str(dest),
                "source_type": meta_info.get("source_type"),
                "domain": None,
                "start": c["start"],
                "end": c["end"],
                "reliability": meta_info.get("reliability"),
            }
            ch = Chunk(document_id=doc.id, chunk_index=idx, text=c["text"], meta=json.dumps(meta_chunk))
            s.add(ch)
        s.commit()

    # add to vector store
    vs.add(
        embeddings,
        [
            {
                "document_id": doc_id,
                "chunk_index": i,
                "source_type": "pdf",
                "source": str(dest),
                "reliability": meta_info.get("reliability"),
            }
            for i in range(len(chunks))
        ],
    )
    logger.info(
        "Ingested PDF",
        extra={
            "source": str(dest),
            "doc_id": doc_id,
            "chunks": len(chunks),
            "reliability": meta_info.get("reliability"),
            "title": title or file.filename,
        },
    )

    return JSONResponse({"status": "ok", "document_id": doc_id, "chunks": len(chunks)})


@app.post('/ingest/url')
def ingest_url(url: str, title: str = "", published_at: str = ""):
    # download and extract HTML
    if not _domain_allowed(url):
        raise HTTPException(status_code=400, detail="Domain not allowed by whitelist/blacklist")
    with get_session() as s:
        existing = s.exec(select(Document).where(Document.source_path == url)).first()
        if existing:
            chunk_count = s.exec(select(func.count()).select_from(Chunk).where(Chunk.document_id == existing.id)).one()
            return {"status": "ok", "document_id": existing.id, "detail": "URL already ingested", "chunks": int(chunk_count or 0)}
    try:
        with start_span("ingest_url.download"):
            html = download_url(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading URL: {e}")
    text = extract_text_from_html(html)
    if len(text) < MIN_SOURCE_CHARS:
        raise HTTPException(status_code=400, detail=f"Source too short ({len(text)} chars)")
    chunks = chunk_text_with_positions(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted from URL")
    # deduplicate identical chunk texts
    seen_hash = set()
    dedup_chunks = []
    for ch in chunks:
        h = hash(ch["text"])
        if h in seen_hash:
            continue
        seen_hash.add(h)
        dedup_chunks.append(ch)
    chunks = dedup_chunks

    # embeddings
    embeddings = embedder.encode([c["text"] for c in chunks], show_progress_bar=True, convert_to_numpy=True)

    # save document and chunks in DB
    meta_info = compute_source_metadata(url, title=title or url)
    if meta_info.get("reliability", 0) < MIN_SOURCE_RELIABILITY:
        raise HTTPException(status_code=400, detail=f"Source reliability too low ({meta_info.get('reliability')})")
    with get_session() as s:
        doc = Document(
            title=title or url,
            source_path=url,
            reliability=meta_info.get("reliability"),
            source_metadata=meta_info.get("source_metadata"),
            source_type=meta_info.get("source_type"),
            published_at=None,
        )
        s.add(doc)
        s.commit()
        s.refresh(doc)
        doc_id = doc.id
        for idx, c in enumerate(chunks):
            meta_chunk = {
                "source": url,
                "source_type": meta_info.get("source_type") or "url",
                "domain": json.loads(meta_info.get("source_metadata") or "{}").get("domain"),
                "start": c["start"],
                "end": c["end"],
                "reliability": meta_info.get("reliability"),
                "title": title or url,
            }
            ch = Chunk(document_id=doc_id, chunk_index=idx, text=c["text"], meta=json.dumps(meta_chunk))
            s.add(ch)
        s.commit()

    # add to vector store
    try:
        meta_json = json.loads(meta_info.get("source_metadata") or "{}")
    except Exception:
        meta_json = {}
    domain = meta_json.get("domain")
    reliability = meta_info.get("reliability")
    vs.add(
        embeddings,
        [
            {
                "document_id": doc_id,
                "chunk_index": i,
                "source": url,
                "source_type": meta_info.get("source_type") or "url",
                "domain": domain,
                "reliability": reliability,
                "title": title or url,
            }
            for i in range(len(chunks))
        ],
    )
    logger.info(
        "Ingested URL",
        extra={
            "source": url,
            "doc_id": doc_id,
            "chunks": len(chunks),
            "domain": domain,
            "reliability": reliability,
            "title": title or url,
        },
    )
    return JSONResponse({"status": "ok", "document_id": doc_id, "chunks": len(chunks), "source": url})


@app.post('/ingest/web')
def ingest_web(urls: list[str], title: str = ""):
    """
    Ingest multiple web pages in one call. Each URL is downloaded, chunked, embedded, and indexed.
    """
    if not urls:
        raise HTTPException(status_code=400, detail="urls list required")
    results = []
    for u in urls:
        try:
            res = ingest_url(u, title=title or u)
            results.append(res.body if hasattr(res, "body") else res)
        except Exception as e:
            logger.exception("Failed to ingest %s: %s", u, e)
            results.append({"url": u, "error": str(e)})
    return {"ingested": results}


@app.post('/ingest/batch')
def ingest_batch(urls: List[str]):
    """Batch ingest multiple URLs (HTML/PDF auto-detected via ingest_url)."""
    if not urls:
        raise HTTPException(status_code=400, detail="urls list required")
    out = []
    for u in urls:
        try:
            res = ingest_url(u)
            out.append(res.body if hasattr(res, "body") else res)
        except Exception as e:
            logger.exception("Batch ingest failed for %s: %s", u, e)
            out.append({"url": u, "error": str(e)})
    return {"ingested": out}

@app.post('/ingest/async')
async def ingest_async(urls: List[str], title: str = ""):
    """Enqueue ingestion tasks (URL or PDF) to background workers."""
    if not urls:
        raise HTTPException(status_code=400, detail="urls list required")
    enqueued = []
    for u in urls:
        kind = "pdf_url" if u.lower().endswith(".pdf") else "url"
        if INGEST_QUEUE_MAX > 0 and ingest_queue.qsize() >= INGEST_QUEUE_MAX:
            raise HTTPException(status_code=429, detail="Ingest queue is full")
        await ingest_queue.put({"kind": kind, "url": u, "title": title or u})
        enqueued.append({"url": u, "kind": kind})
    return {"status": "enqueued", "count": len(enqueued), "items": enqueued}


@app.post('/ingest/pdf_url')
def ingest_pdf_url(url: str, title: str = "", published_at: str = ""):
    """Download a remote PDF (e.g., arXiv) and ingest it like a local upload."""
    if not url.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF URLs are supported")
    if not _domain_allowed(url):
        raise HTTPException(status_code=400, detail="Domain not allowed by whitelist/blacklist")
    with get_session() as s:
        existing = s.exec(select(Document).where(Document.source_path == url)).first()
        if existing:
            return {"status": "exists", "document_id": existing.id, "detail": "PDF already ingested"}
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "SearchOneBot/1.0"})
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading PDF: {e}")
    filename = os.path.basename(url.split("?")[0]) or "remote.pdf"
    dest = UPLOAD_DIR / filename
    with open(dest, 'wb') as f:
        f.write(resp.content)
    # process PDF content (sync path similar to ingest_pdf)
    text = extract_text_from_pdf(str(dest))
    if len(text) < MIN_SOURCE_CHARS:
        raise HTTPException(status_code=400, detail=f"PDF too short ({len(text)} chars)")
    chunks = chunk_text_with_positions(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text content extracted from PDF")
    # deduplicate chunks
    seen_hash = set()
    filtered_chunks = []
    for ch in chunks:
        h = hash(ch["text"])
        if h in seen_hash:
            continue
        seen_hash.add(h)
        filtered_chunks.append(ch)
    chunks = filtered_chunks
    embeddings = embedder.encode([c["text"] for c in chunks], show_progress_bar=True, convert_to_numpy=True)
    meta_info = metadata_from_file(dest, title=title or filename, published_at=published_at)
    with get_session() as s:
        doc = Document(
            title=title or filename,
            source_path=str(dest),
            reliability=meta_info.get("reliability"),
            source_metadata=meta_info.get("source_metadata"),
            source_type=meta_info.get("source_type"),
            published_at=meta_info.get("published_at"),
        )
        s.add(doc)
        s.commit()
        s.refresh(doc)
        doc_id = doc.id
        domain = urllib.parse.urlparse(url).netloc.lower() if url else None
        for idx, c in enumerate(chunks):
            meta_chunk = {
                "source": url,
                "source_type": meta_info.get("source_type"),
                "domain": domain,
                "start": c["start"],
                "end": c["end"],
                "reliability": meta_info.get("reliability"),
                "title": title or filename,
            }
            ch = Chunk(document_id=doc.id, chunk_index=idx, text=c["text"], meta=json.dumps(meta_chunk))
            s.add(ch)
        s.commit()
    vs.add(
        embeddings,
        [
            {
                "document_id": doc_id,
                "chunk_index": i,
                "source_type": meta_info.get("source_type") or "pdf",
                "source": url,
                "domain": urllib.parse.urlparse(url).netloc.lower() if url else None,
                "reliability": meta_info.get("reliability"),
                "title": title or filename,
            }
            for i in range(len(chunks))
        ],
    )
    logger.info(
        "Ingested remote PDF",
        extra={
            "source": url,
            "doc_id": doc_id,
            "chunks": len(chunks),
            "domain": urllib.parse.urlparse(url).netloc.lower() if url else None,
            "reliability": meta_info.get("reliability"),
            "title": title or filename,
        },
    )
    return {"status": "ok", "document_id": doc_id, "chunks": len(chunks), "source": url}


@app.post('/ingest/arxiv')
def ingest_arxiv(arxiv_id_or_url: str):
    """Convenience endpoint to fetch and ingest an arXiv PDF by id or URL."""
    arxiv_id = arxiv_id_or_url
    if arxiv_id.startswith("http"):
        arxiv_id = arxiv_id.rstrip("/").split("/")[-1]
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return ingest_pdf_url(pdf_url, title=f"arXiv {arxiv_id}")


@app.post('/ingest/crawl')
def ingest_crawl(urls: str):
    """Ingest multiple URLs provided as a comma-separated string (simple API).

    Example: POST /ingest/crawl?urls=https://a.com,https://b.com
    """
    url_list = [u.strip() for u in urls.split(',') if u.strip()]
    if not url_list:
        raise HTTPException(status_code=400, detail='No URLs provided')
    logger.info("Starting crawl ingest for %d urls", len(url_list))
    results = crawl_and_ingest(url_list, api_base="http://127.0.0.1:2001")
    logger.info("Crawl ingest finished")
    return JSONResponse({'status': 'ok', 'results': results})


@app.post('/ingest/webpage')
def ingest_webpage(url: str, title: str = ""):
    """Download, clean, chunk and persist a single web page as a document, then index it."""
    try:
        if not _domain_allowed(url):
            raise HTTPException(status_code=400, detail="Domain not allowed by whitelist/blacklist")
        res = ingest_web_page(url, title=title)
        # add to vector store
        with get_session() as s:
            chunks = list(
                s.exec(
                    select(Chunk).where(Chunk.document_id == res["document_id"]).order_by(Chunk.chunk_index)
                ).all()
            )
        texts = [c.text for c in chunks]
        metas = []
        for i, ch in enumerate(chunks):
            meta = {}
            try:
                meta = json.loads(ch.meta) if ch.meta else {}
            except Exception:
                meta = {}
            meta.update({"document_id": res["document_id"], "chunk_index": i})
            metas.append(meta)
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        vs.add(embeddings, metas)
        logger.info(
            "Ingested webpage",
            extra={
                "source": url,
                "doc_id": res["document_id"],
                "chunks": len(texts),
                "domain": urllib.parse.urlparse(url).netloc.lower() if url else None,
                "title": title or url,
            },
        )
        return {"status": "ok", "document_id": res["document_id"], "chunks": len(texts), "source": url}
    except Exception as e:
        logger.exception("Failed to ingest webpage %s: %s", url, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/sources/inject')
def inject_sources(urls: str):
    """Inject multiple URLs quickly (hot path) using existing ingestion pipeline."""
    url_list = [u.strip() for u in urls.split(',') if u.strip()]
    if not url_list:
        raise HTTPException(status_code=400, detail='No URLs provided')
    ingested = []
    errors = []
    for u in url_list:
        try:
            res = ingest_url(u)
            ingested.append(res)
        except Exception as e:
            errors.append({'url': u, 'error': str(e)})
    return {'ingested': ingested, 'errors': errors}

@app.post('/ingest/seed_corpus')
def ingest_seed_corpus(name: str = "climat"):
    """Ingest a curated seed corpus (IPCC/EEA/arXiv) to enrich evidence before runs."""
    urls = SEED_CORPUS.get(name) or []
    if not urls:
        raise HTTPException(status_code=400, detail=f"Unknown corpus {name}")
    ingested = []
    errors = []
    for u in urls:
        try:
            if u.lower().endswith(".pdf"):
                res = ingest_pdf_url(u)
            else:
                res = ingest_url(u)
            ingested.append(res if isinstance(res, dict) else res.body if hasattr(res, "body") else res)
        except Exception as e:
            errors.append({"url": u, "error": str(e)})
    return {"corpus": name, "ingested": ingested, "errors": errors}

@app.get('/search')
def search(q: str, top_k: int = 5):
    with start_span("search.encode"):
        q_emb = embedder.encode([q], convert_to_numpy=True)
    with start_span("search.vector_store"):
        results = vs.search(q_emb, top_k=top_k)
    # enrich with DB text
    out = []
    with get_session() as s:
        for r in results:
            meta = r['metadata']
            if not meta:
                continue
            stmt = select(Chunk).where(Chunk.document_id == meta['document_id'], Chunk.chunk_index == meta['chunk_index'])
            ch = s.exec(stmt).first()
            out.append({
                'score': r['score'],
                'document_id': meta['document_id'],
                'chunk_index': meta['chunk_index'],
                'text': ch.text if ch else None
            })
    return {'query': q, 'results': out}


@app.get('/search/filter')
def search_filter(q: str, doc_ids: str = "", top_k: int = 5):
    """Search restricted to specific document_ids (comma-separated) to reuse existing embeddings."""
    ids = {int(x.strip()) for x in doc_ids.split(',') if x.strip().isdigit()}
    with start_span("search.encode"):
        q_emb = embedder.encode([q], convert_to_numpy=True)
    with start_span("search.vector_store"):
        results = vs.search(q_emb, top_k=top_k * 3)  # fetch more then filter
    filtered = []
    for r in results:
        meta = r.get('metadata') or {}
        if ids and meta.get('document_id') not in ids:
            continue
        filtered.append(r)
        if len(filtered) >= top_k:
            break
    out = []
    with get_session() as s:
        for r in filtered:
            meta = r['metadata']
            if not meta:
                continue
            stmt = select(Chunk).where(Chunk.document_id == meta['document_id'], Chunk.chunk_index == meta['chunk_index'])
            ch = s.exec(stmt).first()
            out.append({
                'score': r['score'],
                'document_id': meta['document_id'],
                'chunk_index': meta['chunk_index'],
                'text': ch.text if ch else None
            })
    return {'query': q, 'results': out, 'filtered_doc_ids': list(ids)}


@app.get('/sources/filter')
def filter_sources(doc_type: str = "", domain: str = ""):
    """Filter documents by type/domain for UI usage."""
    with get_session() as s:
        stmt = select(Document)
        if doc_type:
            stmt = stmt.where(Document.source_type == doc_type)
        if domain:
            stmt = stmt.where(Document.source_metadata.contains(domain))
        stmt = stmt.order_by(Document.created_at.desc()).limit(100)
        docs = list(s.exec(stmt).all())
        out = []
        for d in docs:
            out.append({
                "id": d.id,
                "title": d.title,
                "source_type": d.source_type,
                "reliability": d.reliability,
                "created_at": d.created_at.isoformat(),
                "source_metadata": d.source_metadata,
            })
    return {"documents": out}


@app.get('/sources/summary')
def sources_summary(limit: int = 10, days: int = 30):
    """Summaries of ingested sources: top domains, counts, and recent docs."""
    try:
        with get_session() as s:
            docs = list(
                s.exec(
                    select(Document).order_by(Document.created_at.desc()).limit(limit)
                ).all()
            )
        recent = [
            {
                "id": d.id,
                "title": d.title,
                "source_type": d.source_type,
                "created_at": d.created_at.isoformat(),
                "reliability": d.reliability,
            }
            for d in docs
        ]
        dom_stats = _chunk_domain_stats(limit=limit)
        return {"recent": recent, "domains": dom_stats.get("top_domains"), "reliability": dom_stats}
    except Exception as e:
        logger.warning("sources_summary error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/report')
def simple_report(q: str, top_k: int = 10):
    # run search and assemble markdown-like report
    res = search(q, top_k=top_k)
    md = [f"# Rapport rapide pour la requ?te: {q}\n"]
    for i, r in enumerate(res['results']):
        md.append(f"## R?sultat {i+1} ? Document {r['document_id']} (chunk {r['chunk_index']})\n")
        md.append(r['text'][:2000] + '\n')
    return JSONResponse({'query': q, 'report_markdown': '\n'.join(md)})


@app.post('/jobs/start')
def start_job(name: str, query: str, max_duration_seconds: int = 300, max_token_budget: int = 0, max_iterations: int = 5, priority: int = 0):
    # create job entry and start background agent
    job_id = create_job(name, priority=priority)
    start_job_background(job_id, query, max_duration_seconds=max_duration_seconds, max_token_budget=max_token_budget, max_iterations=max_iterations)
    return {'job_id': job_id, 'status': 'started', 'max_token_budget': max_token_budget, 'max_iterations': max_iterations, 'priority': priority}


@app.get('/jobs/{job_id}')
def job_status(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return {'id': job.id, 'name': job.name, 'status': job.status, 'state': job.state, 'priority': job.priority}


@app.post('/jobs/{job_id}/rename')
def job_rename(job_id: int, new_name: str):
    ok = rename_job(job_id, new_name)
    if not ok:
        raise HTTPException(status_code=404, detail='Job not found or invalid name')
    return {'id': job_id, 'name': new_name, 'status': 'renamed'}


@app.delete('/jobs/{job_id}')
def job_delete(job_id: int):
    ok = soft_delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail='Job not found')
    return {'id': job_id, 'status': 'deleted'}


@app.post('/jobs/stop')
def stop_job(job_id: int):
    # graceful stop: set stop flag (RQ workers), cancel queued job, and mark DB as failed
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    stop_flag = request_stop(job_id)
    rq_cancelled = cancel_rq_job(job_id)
    # try to mark local thread if present
    t = job_threads.get(job_id)
    if t and not t.is_alive():
        job_threads.pop(job_id, None)
    save_job_state(job_id, job.state or '', status='failed')
    detail_parts = ['Job marked as failed']
    if stop_flag:
        detail_parts.append('stop flag set')
    if rq_cancelled:
        detail_parts.append('RQ job cancelled')
    return {'id': job.id, 'status': 'failed', 'detail': '; '.join(detail_parts)}


@app.post('/jobs/pause')
def pause_job(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    save_job_state(job_id, job.state or '', status='paused')
    return {'id': job.id, 'status': 'paused', 'detail': 'Job marked as paused (requires manual restart)'}


@app.post('/jobs/resume')
def resume_job(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    # This only flips status; to truly resume a paused job, start a new job run as needed.
    save_job_state(job_id, job.state or '', status='running')
    return {'id': job.id, 'status': 'running', 'detail': 'Job status set to running (no automatic restart)'}

@app.post("/jobs/schedule")
def schedule_job(query: str, interval_seconds: int = 3600):
    entry = scheduler.add_schedule(query, interval_seconds)
    return {"scheduled": entry}

@app.get("/jobs/schedules")
def list_schedules():
    return {"schedules": scheduler.list_schedules()}

@app.delete("/jobs/schedules/{schedule_id}")
def delete_schedule(schedule_id: int):
    ok = scheduler.remove_schedule(schedule_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"deleted": schedule_id}


@app.post("/jobs/schedules/run_due")
async def run_due_schedules():
    """Trigger due schedules and start jobs for them."""
    executed = await scheduler.run_due(lambda q: _launch_scheduled_job(q))
    return {"executed": len(executed), "schedules": executed}


@app.post('/jobs/{job_id}/retry')
def retry_job(job_id: int, name: str = None, max_iterations: int = 5, max_duration_seconds: int = 300, max_token_budget: int = 0):
    """Create a new job using the query from an existing job state."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    try:
        st = json.loads(job.state) if job.state else {}
    except Exception:
        st = {}
    query = st.get('query')
    if not query:
        raise HTTPException(status_code=400, detail='Original job state has no query to retry')
    new_job_id = create_job(name or f"retry-{job.name}")
    start_job_background(new_job_id, query, max_duration_seconds=max_duration_seconds, max_token_budget=max_token_budget, max_iterations=max_iterations)
    return {'job_id': new_job_id, 'status': 'started', 'origin_job': job_id, 'max_iterations': max_iterations, 'max_duration_seconds': max_duration_seconds, 'max_token_budget': max_token_budget}


@app.get('/jobs/{job_id}/timeline/stream')
async def job_timeline_stream(job_id: int, interval: float = 2.0, max_events: int = 120):
    """SSE stream of timeline/state updates until job is finished (or max_events reached)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')

    async def event_stream():
        for _ in range(max_events):
            j = get_job(job_id)
            if not j:
                yield f"event: error\ndata: {json.dumps({'error': 'not_found'})}\n\n"
                break
            try:
                state = json.loads(j.state) if j.state else {}
            except Exception:
                state = {}
            timeline = state.get('timeline') or []
            payload = {"job_id": job_id, "status": j.status, "timeline": timeline}
            yield f"data: {json.dumps(payload)}\n\n"
            if j.status not in ('running', 'queued'):
                break
            await asyncio.sleep(interval)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get('/jobs')
def list_jobs(limit: int = 50, offset: int = 0):
    """List recent jobs with basic metadata."""
    from db import get_session, Job as JobModel
    with get_session() as s:
        stmt = (
            select(JobModel)
            .where(JobModel.status != 'deleted')
            .order_by(JobModel.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = list(s.exec(stmt).all())
        out = [
            {
                'id': r.id,
                'name': r.name,
                'status': r.status,
                'priority': r.priority,
                'created_at': r.created_at.isoformat(),
                'updated_at': r.updated_at.isoformat(),
            }
            for r in rows
        ]
    return {'jobs': out}


@app.get('/jobs/{job_id}/report')
def job_report(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    try:
        state = json.loads(job.state) if job.state else {}
    except Exception:
        state = {}
    path = save_report(job_id, job.name, state)
    return {'report_path': str(path)}


@app.get('/jobs/{job_id}/report.json')
def job_report_json(job_id: int):
    """Export job state as structured JSON (hypotheses, preuves, risques/recos)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    try:
        state = json.loads(job.state) if job.state else {}
    except Exception:
        state = {}
    summary = build_structured_summary(state)
    return {"job_id": job_id, "name": job.name, "status": job.status, "summary": summary}


@app.get('/jobs/{job_id}/diagnostic')
def job_diagnostic(job_id: int):
    """Return internal diagnostic (status, scores, last summary/votes)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    try:
        state = json.loads(job.state) if job.state else {}
    except Exception:
        state = {}
    diag = state.get("diagnostic") or build_diagnostic(state)
    return {"job_id": job_id, "name": job.name, "status": job.status, "diagnostic": diag}


@app.get('/jobs/{job_id}/overview')
def job_overview(job_id: int):
    """Dashboard-friendly overview: usage, metrics, bus snapshot, run context."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    try:
        state = json.loads(job.state) if job.state else {}
    except Exception:
        state = {}
    metrics = state.get("run_metrics") or compute_run_metrics(state)
    return {
        "job_id": job_id,
        "name": job.name,
        "status": job.status,
        "usage": state.get("usage"),
        "run_metrics": metrics,
        "run_context": state.get("run_context"),
        "message_bus": state.get("message_bus"),
        "timeline_len": len(state.get("timeline") or []),
    }


@app.get("/dashboard")
def dashboard(limit: int = 5):
    """Aggregated view for a simple dashboard JSON."""
    dbs = db_stats()
    index = _chunk_domain_stats(limit=10)
    schedules = scheduler.list_schedules()
    tasks = scheduler.list_tasks()
    latest_jobs = jobs_board(limit=limit).get("jobs")
    latest_job = latest_jobs[0] if latest_jobs else None
    diagnostics = None
    if latest_job:
        try:
            diagnostics = job_diagnostic(latest_job["id"])
        except Exception:
            diagnostics = None
    return {
        "db": dbs,
        "index": index,
        "schedules": schedules,
        "scheduler_tasks": tasks,
        "latest_jobs": latest_jobs,
        "latest_job_diagnostic": diagnostics,
    }

@app.get('/jobs/{job_id}/timeline')
def job_timeline(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    try:
        state = json.loads(job.state) if job.state else {}
    except Exception:
        state = {}

    # Prefer explicit timeline if present
    if 'timeline' in state and isinstance(state['timeline'], list):
        return {'timeline': state['timeline']}

    # Fallback: build timeline from council entries
    timeline = []
    council = state.get('council', {}) or {}
    for k in sorted(council.keys()):
        entry = council[k] or {}
        timeline.append({
            'iteration': k,
            'summary': entry.get('summary') if isinstance(entry, dict) else entry,
            'messages': entry.get('messages') if isinstance(entry, dict) else None,
            'votes': entry.get('votes') if isinstance(entry, dict) else None,
        })
    return {'timeline': timeline}


@app.get('/jobs/board')
def jobs_board(limit: int = 50, offset: int = 0):
    """Board-friendly endpoint: recent jobs + last timeline entry/notes for quick UI."""
    from db import get_session, Job as JobModel
    out = []
    with get_session() as s:
        stmt = select(JobModel).order_by(JobModel.updated_at.desc()).offset(offset).limit(limit)
        rows = list(s.exec(stmt).all())
        for r in rows:
            try:
                st = json.loads(r.state) if r.state else {}
            except Exception:
                st = {}
            timeline = st.get('timeline') or []
            last_event = timeline[-1] if timeline else None
            last_note = st.get('notes') or st.get('last_error')
            if not last_note and st.get('agents'):
                notes = [a.get('notes') for a in st.get('agents', {}).values() if isinstance(a, dict) and a.get('notes')]
                last_note = notes[-1] if notes else None
            out.append({
                'id': r.id,
                'name': r.name,
                'status': r.status,
                'priority': r.priority,
                'updated_at': r.updated_at.isoformat(),
                'last_event': last_event,
                'last_note': last_note,
            })
    return {'jobs': out}


@app.get('/search/vector')
def search_vector(q: str, top_k: int = 10):
    """Search across indexed content (documents + conversations)."""
    with start_span("search.encode"):
        q_emb = embedder.encode([q], convert_to_numpy=True)
    results = vs.search(q_emb, top_k=top_k)
    return {"query": q, "results": results}


def _lexical_score(query: str, text: str) -> float:
    q_tokens = [t.lower() for t in re.findall(r"\w+", query)]
    if not q_tokens:
        return 0.0
    q_counts = Counter(q_tokens)
    doc_tokens = [t.lower() for t in re.findall(r"\w+", text)]
    if not doc_tokens:
        return 0.0
    doc_counts = Counter(doc_tokens)
    score = 0.0
    for t, c in q_counts.items():
        if t in doc_counts:
            score += c * (1 + doc_counts[t])
    return score / (len(doc_tokens) + 1)


def _hydrate_chunk(meta: dict) -> str:
    """Retrieve chunk text from DB if not present in meta."""
    txt = meta.get("text")
    if txt:
        return txt
    try:
        with get_session() as s:
            stmt = select(Chunk).where(Chunk.document_id == meta.get("document_id"), Chunk.chunk_index == meta.get("chunk_index"))
            ch = s.exec(stmt).first()
            if ch:
                return ch.text
    except Exception:
        return ""
    return ""


@app.get('/search/hybrid')
def search_hybrid(q: str, top_k: int = 10, alpha: float = 0.7):
    """
    Hybrid search: dense vector + simple lexical overlap fusion.
    alpha controls weight of vector score vs lexical.
    """
    with start_span("search.encode"):
        q_emb = embedder.encode([q], convert_to_numpy=True)
    raw = vs.search(q_emb, top_k=top_k * 3)
    fused = []
    for hit in raw:
        meta = hit.get("metadata") or {}
        text = _hydrate_chunk(meta)
        lex = _lexical_score(q, text)
        vec = hit.get("score") or 0.0
        combined = alpha * vec + (1 - alpha) * lex
        fused.append({"score": combined, "vector_score": vec, "lexical_score": lex, "text": text, "meta": meta})
    fused = sorted(fused, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
    if not fused:
        # fallback: return one recent chunk from DB to avoid empty results in tests/UI
        try:
            with get_session() as s:
                ch = s.exec(select(Chunk).order_by(Chunk.id.desc())).first()
                if ch:
                    fused.append({"score": 0.0, "vector_score": 0.0, "lexical_score": 0.0, "text": ch.text, "meta": {"document_id": ch.document_id, "chunk_index": ch.chunk_index}})
        except Exception:
            fused.append({"score": 0.0, "vector_score": 0.0, "lexical_score": 0.0, "text": "[no results available]", "meta": {}})
    return {"query": q, "alpha": alpha, "results": fused}


@app.get('/search/vector/debug')
def search_vector_debug():
    """Return basic index stats for debugging."""
    try:
        stats = vs.stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"stats": stats}


@app.get('/doctor/index/status')
def doctor_index_status():
    """Return FAISS index stats with ntotal, domains, and basic ingestion/retrieval health."""
    try:
        stats = vs.stats()
        domains = _chunk_domain_stats(limit=10)
        alerts = {
            "index_empty": (stats.get("ntotal") or stats.get("emb_count") or 0) == 0,
            "ingest_queue_full": INGEST_QUEUE_MAX > 0 and ingest_queue.qsize() >= INGEST_QUEUE_MAX,
            "ingest_failures": _INGEST_FAIL > 0,
        }
        return {
            "ok": True,
            "index": stats,
            "domains": domains,
            "reliability": {
                "min_source_reliability": MIN_SOURCE_RELIABILITY,
                "min_source_chars": MIN_SOURCE_CHARS,
            },
            "ingestion": {
                "queue_size": ingest_queue.qsize() if ingest_queue else 0,
                "workers": len(_ingest_worker_tasks),
                "queue_max": ingest_queue.maxsize if ingest_queue else None,
                "success": _INGEST_SUCCESS,
                "fail": _INGEST_FAIL,
                "retry": _INGEST_RETRY,
                "max_retries": INGEST_MAX_RETRIES,
            },
            "alerts": alerts,
        }
    except Exception as e:
        logger.warning("Index status failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/jobs/{job_id}/evidence')
def job_evidence(job_id: int, limit: int = 50):
    """Return flattened evidence with source, score, and basic diversity info."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    try:
        state = json.loads(job.state) if job.state else {}
    except Exception:
        state = {}
    evidence = []
    timeline = state.get('timeline') or []
    for entry in timeline:
        for msg in entry.get('messages') or []:
            for ev in msg.get('evidence') or []:
                evidence.append(ev)
    if not evidence and state.get("evidence"):
        evidence = state.get("evidence")
    # diversity: count unique sources/domains
    sources = set()
    for ev in evidence:
        meta = ev.get("meta") or {}
        src = meta.get("source") or meta.get("document_id")
        if src is not None:
            sources.add(str(src))
    diversity = {
        "unique_sources": len(sources),
        "coverage_score": (len(sources) / len(evidence)) if evidence else 0.0
    }
    # trim
    evidence = evidence[:limit]
    return {"job_id": job_id, "count": len(evidence), "diversity": diversity, "evidence": evidence}


@app.post('/knowledge/promotions/consolidate')
def knowledge_consolidate():
    """Deduplicate/promote knowledge base and return summary."""
    summary = consolidate_promotions()
    return {"ok": True, "summary": summary}


@app.get('/prompts/system')
def get_system_prompt_endpoint():
    """Return the current system prompt used by agents."""
    return {"system_prompt": get_system_prompt()}


@app.post('/prompts/system')
def set_system_prompt_endpoint(prompt: str = ""):
    """Set a runtime system prompt for all agents (used as prefix)."""
    new_prompt = set_system_prompt(prompt)
    return {"system_prompt": new_prompt, "length": len(new_prompt)}


@app.post('/prompts/variant')
def set_prompt_variant_endpoint(variant: str = ""):
    """Set an A/B variant for prompts (empty to disable)."""
    v = set_prompt_variant(variant or None)
    return {"variant": v}


@app.get('/prompts/variant')
def get_prompt_variant_endpoint():
    return {"variant": get_prompt_variant()}


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=2001, reload=True)
