"""
Minimal research job scheduler.
Provides delayed execution and resume-from-snapshot hooks.
"""
import asyncio
import json
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime, timedelta


SCHEDULES_FILE = Path(__file__).resolve().parents[2] / "data" / "schedules.json"


class ResearchScheduler:
    """Schedule research runs with simple delays and checkpoint resume."""

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.queue: List[asyncio.Task] = []
        self.schedules = self._load_schedules()

    def _load_schedules(self) -> List[Dict[str, Any]]:
        if not SCHEDULES_FILE.exists():
            return []
        try:
            return json.loads(SCHEDULES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_schedules(self):
        SCHEDULES_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCHEDULES_FILE.write_text(json.dumps(self.schedules, ensure_ascii=False, indent=2), encoding="utf-8")

    async def schedule(self, coro_factory: Callable[[], asyncio.Future], delay_seconds: float = 0) -> asyncio.Task:
        async def _runner():
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
            return await coro_factory()

        task = asyncio.create_task(_runner())
        self.queue.append(task)
        return task

    def add_schedule(self, query: str, interval_seconds: int) -> Dict[str, Any]:
        now = datetime.utcnow()
        entry = {
            "id": len(self.schedules) + 1,
            "query": query,
            "interval_seconds": interval_seconds,
            "next_run": (now + timedelta(seconds=interval_seconds)).isoformat(),
            "last_run": None,
        }
        self.schedules.append(entry)
        self._save_schedules()
        return entry

    def list_schedules(self) -> List[Dict[str, Any]]:
        return list(self.schedules)

    def remove_schedule(self, sched_id: int) -> bool:
        before = len(self.schedules)
        self.schedules = [s for s in self.schedules if s.get("id") != sched_id]
        if len(self.schedules) != before:
            self._save_schedules()
            return True
        return False

    def list_tasks(self) -> List[Dict[str, Any]]:
        return [{"done": t.done(), "cancelled": t.cancelled()} for t in self.queue]

    def resume_from_snapshot(self, job_id: int) -> Optional[Dict[str, Any]]:
        path = self.snapshot_dir / f"run_{job_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def due_schedules(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = now or datetime.utcnow()
        due: List[Dict[str, Any]] = []
        for s in self.schedules:
            try:
                nxt = datetime.fromisoformat(s.get("next_run"))
            except Exception:
                nxt = now
            if nxt <= now:
                due.append(s)
        return due

    def update_after_run(self, sched: Dict[str, Any], now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        sched["last_run"] = now.isoformat()
        interval = int(sched.get("interval_seconds", 0)) or 0
        sched["next_run"] = (now + timedelta(seconds=interval)).isoformat()

    async def run_due(self, launcher: Callable[[str], Any]) -> List[Dict[str, Any]]:
        """
        Run due schedules by calling launcher(query).
        Launcher is expected to trigger job creation.
        """
        executed: List[Dict[str, Any]] = []
        for sched in self.due_schedules():
            launcher(sched.get("query", ""))
            self.update_after_run(sched)
            executed.append(sched)
        if executed:
            self._save_schedules()
        return executed
