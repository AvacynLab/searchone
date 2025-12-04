from sqlmodel import SQLModel, Field, create_engine, Session, select, delete
from typing import Optional
from datetime import datetime, timezone, timedelta
from app.core.config import DB_FILE
from sqlalchemy import Column, Text, func
import logging
from app.core.logging_config import configure_logging
import json

configure_logging()
logger = logging.getLogger(__name__)

sqlite_url = f"sqlite:///{DB_FILE}"
engine = create_engine(sqlite_url, echo=False)

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    source_path: str
    reliability: Optional[float] = Field(default=None, description="Confidence score for the source (0-1)")
    source_metadata: Optional[str] = Field(default=None, description="JSON metadata for source (domain, tags, type)")
    source_type: Optional[str] = Field(default=None, description="Type of source (pdf, url, report, etc.)")
    published_at: Optional[datetime] = Field(default=None, description="Publication date if known")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int
    chunk_index: int
    text: str
    meta: Optional[str] = None


class TabularData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: Optional[int] = Field(default=None)
    table_id: str
    table_schema: Optional[str] = Field(default=None, sa_column=Column("schema", Text))
    records: Optional[str] = Field(default=None)
    table_metadata: Optional[str] = Field(default=None, sa_column=Column("metadata", Text))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    status: str = Field(default="pending")
    state: Optional[str] = None
    priority: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchSchedule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    query: str
    spec_type: str
    spec_expr: str
    spec_extra: Optional[str] = None
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    failures: int = Field(default=0)
    max_retries: int = Field(default=3)
    backoff_seconds: int = Field(default=60)
    active: bool = Field(default=True)
    last_error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WebCache(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    cache_key: str = Field(index=True)
    query: str
    lang: str = Field(default="fr")
    safe_search: bool = Field(default=True)
    engine_set: str
    result_payload: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

def init_db():
    SQLModel.metadata.create_all(engine)
    _ensure_columns()
    logger.info("Database initialized at %s", DB_FILE)


def _ensure_columns():
    """Simple schema fixups for new columns when using existing SQLite file."""
    try:
        with engine.connect() as conn:
            # add Job.priority if missing
            cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(job)").fetchall()}
            if "priority" not in cols:
                conn.exec_driver_sql("ALTER TABLE job ADD COLUMN priority INTEGER DEFAULT 0")
            # Document.source_type and published_at
            cols_doc = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(document)").fetchall()}
            if "reliability" not in cols_doc:
                conn.exec_driver_sql("ALTER TABLE document ADD COLUMN reliability FLOAT")
            if "source_metadata" not in cols_doc:
                conn.exec_driver_sql("ALTER TABLE document ADD COLUMN source_metadata VARCHAR")
            if "source_type" not in cols_doc:
                conn.exec_driver_sql("ALTER TABLE document ADD COLUMN source_type VARCHAR")
            if "published_at" not in cols_doc:
                conn.exec_driver_sql("ALTER TABLE document ADD COLUMN published_at DATETIME")
    except Exception as e:
        logger.warning("Schema ensure_columns skipped: %s", e)

def get_session():
    return Session(engine)

def save_job_state(job_id: int, state_str: str, status: str = None):
    with get_session() as s:
        job = s.get(Job, job_id)
        if not job:
            return False
        job.state = state_str
        if status:
            job.status = status
        job.updated_at = datetime.now(timezone.utc)
        s.add(job)
        s.commit()
    return True


def db_stats():
    """Return simple counts for diagnostics."""
    with get_session() as s:
        docs = s.exec(select(func.count()).select_from(Document)).scalar_one()
        chunks = s.exec(select(func.count()).select_from(Chunk)).scalar_one()
        jobs = s.exec(select(func.count()).select_from(Job)).scalar_one()
    return {"documents": docs, "chunks": chunks, "jobs": jobs}


def db_job_status_counts():
    """Return counts of jobs per status."""
    with get_session() as s:
        rows = s.exec(select(Job.status, func.count(Job.id)).group_by(Job.status)).all()
    return {status: count for status, count in rows}


def get_web_cache_entry(cache_key: str) -> Optional[WebCache]:
    now = datetime.now(timezone.utc)
    with get_session() as s:
        entry = s.exec(select(WebCache).where(WebCache.cache_key == cache_key)).first()
        if not entry:
            return None
        exp_at = entry.expires_at
        if exp_at:
            if exp_at.tzinfo is None:
                exp_at = exp_at.replace(tzinfo=timezone.utc)
            if exp_at < now:
                s.delete(entry)
                s.commit()
                return None
        return entry


def store_web_cache_entry(
    cache_key: str,
    query: str,
    lang: str,
    safe_search: bool,
    engine_set: str,
    result_payload: str,
    expires_at: datetime,
) -> WebCache:
    now = datetime.now(timezone.utc)
    with get_session() as s:
        entry = s.exec(select(WebCache).where(WebCache.cache_key == cache_key)).first()
        if not entry:
            entry = WebCache(
                cache_key=cache_key,
                query=query,
                lang=lang,
                safe_search=safe_search,
                engine_set=engine_set,
                created_at=now,
            )
        entry.result_payload = result_payload
        entry.expires_at = expires_at
        entry.updated_at = now
        s.add(entry)
        s.commit()
        s.refresh(entry)
    return entry


def cleanup_web_cache_entries(before: Optional[datetime] = None) -> int:
    if before is None:
        before = datetime.now(timezone.utc)
    with get_session() as s:
        stmt = delete(WebCache).where(WebCache.expires_at != None).where(WebCache.expires_at < before)
        result = s.exec(stmt)
        s.commit()
    return result.rowcount if result is not None else 0
