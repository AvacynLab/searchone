"""
Resilient research job scheduler supporting rich specs, persistence, and snapshot helpers.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone

from croniter import croniter, CroniterBadCronError
import isodate

from sqlmodel import select

from app.core.config import APP_VERSION
from app.data.db import ResearchSchedule, get_session

logger = logging.getLogger(__name__)

SCHEDULES_FILE = Path(__file__).resolve().parents[2] / "data" / "schedules.json"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:  # pragma: no cover - best effort
            logger.debug("Could not parse iso timestamp %s", value)
            return None


def _normalize_spec(spec: Optional[Dict[str, Any]], interval_seconds: Optional[int]) -> Dict[str, Any]:
    sanitized = dict(spec or {})
    typ = sanitized.get("type")
    if not typ:
        typ = "interval"
        sanitized = {"type": "interval"}
    sanitized["type"] = typ
    if typ == "interval":
        sanitized["seconds"] = int(sanitized.get("seconds") or interval_seconds or 3600)
        sanitized.setdefault("expr", str(sanitized["seconds"]))
    elif typ == "cron":
        expr = sanitized.get("expr")
        if not expr:
            raise ValueError("cron spec requires 'expr' parameter")
        sanitized["expr"] = str(expr).strip()
    elif typ == "iso":
        expr = sanitized.get("expr") or sanitized.get("duration")
        if not expr:
            raise ValueError("iso spec requires 'expr' parameter")
        sanitized["expr"] = str(expr).strip()
    else:
        raise ValueError(f"Unsupported schedule type '{typ}'")
    details = {k: v for k, v in sanitized.items() if k not in ("type", "expr", "seconds")}
    sanitized["details"] = details
    return sanitized


def _next_run_from_spec(spec: Dict[str, Any], base: Optional[datetime] = None) -> datetime:
    base = base or _utcnow()
    typ = spec.get("type")
    if typ == "cron":
        expr = spec.get("expr")
        try:
            next_dt = croniter(expr, base).get_next(datetime)
        except CroniterBadCronError as exc:
            raise ValueError(f"Invalid cron syntax '{expr}'") from exc
        if next_dt.tzinfo is None:
            next_dt = next_dt.replace(tzinfo=timezone.utc)
        return next_dt
    if typ == "iso":
        expr = spec.get("expr")
        duration_expr = expr.split("/")[-1]
        try:
            duration = isodate.parse_duration(duration_expr)
        except Exception as exc:
            raise ValueError(f"Invalid iso duration '{duration_expr}'") from exc
        if isinstance(duration, isodate.Duration):
            duration = duration.totimedelta()
        if not isinstance(duration, timedelta):
            duration = timedelta(seconds=float(duration))
        return base + duration
    if typ == "interval":
        seconds = float(spec.get("seconds") or 3600)
        return base + timedelta(seconds=seconds)
    raise ValueError(f"Cannot compute next run for schedule type '{typ}'")


def _spec_extra_json(spec: Dict[str, Any]) -> Optional[str]:
    details = spec.get("details") or {}
    if not details:
        return None
    try:
        return json.dumps(details, ensure_ascii=False)
    except Exception:
        return None


def _load_file_schedules() -> List[Dict[str, Any]]:
    if not SCHEDULES_FILE.exists():
        return []
    try:
        raw = json.loads(SCHEDULES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            spec = _normalize_spec(entry.get("spec"), entry.get("interval_seconds"))
        except Exception:
            continue
        normalized: Dict[str, Any] = {
            "name": entry.get("name"),
            "query": entry.get("query"),
            "spec": spec,
            "next_run": entry.get("next_run"),
            "last_run": entry.get("last_run"),
            "backoff_seconds": entry.get("backoff_seconds", 60),
            "max_retries": entry.get("max_retries", 3),
            "failures": entry.get("failures", 0),
            "last_error": entry.get("last_error"),
        }
        entries.append({k: v for k, v in normalized.items() if v is not None})
    return entries


def _write_file_schedules(entries: List[Dict[str, Any]]) -> None:
    SCHEDULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        SCHEDULES_FILE.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to write schedules cache: %s", exc)


class ResearchScheduler:
    """Schedule research runs with richer formats and snapshot helpers."""

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.queue: List[asyncio.Task] = []
        self.schedules: List[Dict[str, Any]] = []
        self._initialize_state()
        self.cleanup_snapshots()

    def _initialize_state(self) -> None:
        entries = self._load_from_db()
        if not entries:
            file_entries = _load_file_schedules()
            for entry in file_entries:
                try:
                    spec = entry["spec"]
                    next_run = _parse_iso(entry.get("next_run"))
                    if not next_run:
                        next_run = _next_run_from_spec(spec)
                    self._create_db_schedule(entry, next_run)
                except Exception as exc:
                    logger.debug("Skipping schedule from cache: %s", exc)
            entries = self._load_from_db()
        self.schedules = entries
        _write_file_schedules(self.schedules)

    def _refresh_cache(self) -> None:
        self.schedules = self._load_from_db()
        _write_file_schedules(self.schedules)

    def _load_from_db(self) -> List[Dict[str, Any]]:
        with get_session() as s:
            rows = s.exec(select(ResearchSchedule).where(ResearchSchedule.active == True)).all()
        return [self._row_to_entry(row) for row in rows]

    def _row_to_entry(self, row: ResearchSchedule) -> Dict[str, Any]:
        extras: Dict[str, Any] = {}
        if row.spec_extra:
            try:
                extras = json.loads(row.spec_extra)
            except Exception:
                extras = {}
        spec = {"type": row.spec_type, "expr": row.spec_expr}
        spec.update(extras)
        return {
            "id": row.id,
            "name": row.name,
            "query": row.query,
            "spec": spec,
            "next_run": _to_iso(row.next_run),
            "last_run": _to_iso(row.last_run),
            "failures": row.failures,
            "max_retries": row.max_retries,
            "backoff_seconds": row.backoff_seconds,
            "last_error": row.last_error,
            "active": row.active,
        }

    def _create_db_schedule(
        self, payload: Dict[str, Any], next_run: datetime, failures: int = 0
    ) -> ResearchSchedule:
        spec = payload["spec"]
        row = ResearchSchedule(
            name=payload.get("name") or f"schedule-{payload.get('query', '')[:16]}",
            query=payload["query"],
            spec_type=spec["type"],
            spec_expr=spec.get("expr") or str(spec.get("seconds") or ""),
            spec_extra=_spec_extra_json(spec),
            next_run=next_run,
            failures=failures,
            max_retries=payload.get("max_retries", 3),
            backoff_seconds=payload.get("backoff_seconds", 60),
            active=payload.get("active", True),
            last_error=payload.get("last_error"),
        )
        with get_session() as s:
            s.add(row)
            s.commit()
            s.refresh(row)
        return row

    def add_schedule(
        self,
        query: str,
        spec: Optional[Dict[str, Any]] = None,
        interval_seconds: Optional[int] = None,
        name: Optional[str] = None,
        max_retries: int = 3,
        backoff_seconds: int = 60,
    ) -> Dict[str, Any]:
        normalized = _normalize_spec(spec, interval_seconds)
        next_run = _next_run_from_spec(normalized)
        payload = {
            "name": name,
            "query": query,
            "spec": normalized,
            "backoff_seconds": backoff_seconds,
            "max_retries": max_retries,
        }
        self._create_db_schedule(payload, next_run)
        self._refresh_cache()
        return self.schedules[-1] if self.schedules else {}

    def list_schedules(self) -> List[Dict[str, Any]]:
        return list(self.schedules)

    def remove_schedule(self, sched_id: int) -> bool:
        with get_session() as s:
            row = s.get(ResearchSchedule, sched_id)
            if not row:
                return False
            row.active = False
            row.updated_at = _utcnow()
            s.add(row)
            s.commit()
        self._refresh_cache()
        return True

    def list_tasks(self) -> List[Dict[str, Any]]:
        return [{"done": t.done(), "cancelled": t.cancelled()} for t in self.queue]

    def _find_snapshot(self, job_id: int) -> Optional[Path]:
        candidate = self.snapshot_dir / f"run_{job_id}.json"
        if candidate.exists():
            return candidate
        pattern = list(self.snapshot_dir.glob(f"run_{job_id}*.json"))
        return sorted(pattern)[-1] if pattern else None

    def resume_from_snapshot(self, job_id: int) -> Optional[Dict[str, Any]]:
        path = self._find_snapshot(job_id)
        if not path:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse snapshot for %s: %s", job_id, exc)
            return None
        version = data.get("version")
        if version != APP_VERSION:
            data.setdefault("migrations", []).append(
                {"from": version, "to": APP_VERSION, "ts": _utcnow().isoformat()}
            )
            data["version"] = APP_VERSION
        data["restored_at"] = _utcnow().isoformat()
        return data

    def due_schedules(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = now or _utcnow()
        due: List[Dict[str, Any]] = []
        for entry in self.schedules:
            if not entry.get("active", True):
                continue
            next_run = _parse_iso(entry.get("next_run"))
            if not next_run or next_run <= now:
                due.append(entry)
        return due

    def _update_schedule_row(self, entry: Dict[str, Any]) -> None:
        with get_session() as s:
            row = s.get(ResearchSchedule, entry["id"])
            if not row:
                return
            row.next_run = _parse_iso(entry.get("next_run"))
            row.last_run = _parse_iso(entry.get("last_run"))
            row.failures = entry.get("failures", row.failures)
            row.backoff_seconds = entry.get("backoff_seconds", row.backoff_seconds)
            row.max_retries = entry.get("max_retries", row.max_retries)
            row.last_error = entry.get("last_error")
            row.active = entry.get("active", row.active)
            row.spec_expr = entry["spec"].get("expr") or row.spec_expr
            row.spec_extra = _spec_extra_json(entry["spec"]) or row.spec_extra
            row.updated_at = _utcnow()
            s.add(row)
            s.commit()

    async def schedule(self, coro_factory: Callable[[], asyncio.Future], delay_seconds: float = 0) -> asyncio.Task:
        async def _runner():
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
            return await coro_factory()

        task = asyncio.create_task(_runner())
        self.queue.append(task)
        return task

    def cleanup_snapshots(self, max_age_days: int = 30) -> None:
        cutoff = _utcnow() - timedelta(days=max_age_days)
        removed = 0
        for path in self.snapshot_dir.glob("run_*.json"):
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                if modified < cutoff:
                    path.unlink()
                    removed += 1
            except Exception:
                continue
        if removed:
            logger.info("Cleaned %d stale snapshots", removed)

    def _record_execution(self, entry: Dict[str, Any], success: bool, error: Optional[str]) -> None:
        now = _utcnow()
        entry["last_run"] = _to_iso(now)
        if success:
            entry["failures"] = 0
            entry["last_error"] = None
            entry["next_run"] = _to_iso(_next_run_from_spec(entry["spec"], base=now))
        else:
            entry["failures"] = entry.get("failures", 0) + 1
            entry["last_error"] = error
            if entry["failures"] >= entry.get("max_retries", 3):
                entry["active"] = False
                entry["next_run"] = None
            else:
                delay = entry.get("backoff_seconds", 60) * entry["failures"]
                entry["next_run"] = _to_iso(now + timedelta(seconds=delay))
        self._update_schedule_row(entry)

    async def run_due(self, launcher: Callable[[str], Any]) -> List[Dict[str, Any]]:
        executed: List[Dict[str, Any]] = []
        for entry in self.due_schedules():
            record: Dict[str, Any] = {"id": entry.get("id"), "query": entry.get("query"), "success": False}
            try:
                result = launcher(entry.get("query", ""))
                if asyncio.iscoroutine(result):
                    await result
                self._record_execution(entry, success=True, error=None)
                record["success"] = True
                record["next_run"] = entry.get("next_run")
            except Exception as exc:
                logger.warning("Scheduled job failed: %s", exc)
                self._record_execution(entry, success=False, error=str(exc))
                record["error"] = str(exc)
            executed.append(record)
        if executed:
            self._refresh_cache()
        return executed
