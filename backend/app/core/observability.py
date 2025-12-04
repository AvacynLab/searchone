from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import json
from app.core.config import DATA_DIR


def compute_run_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate a few observability metrics from job state."""
    timeline = state.get("timeline", [])
    agents = state.get("agents", {})
    total_iters = len(timeline)
    evidence_counts = []
    for entry in timeline:
        msgs = entry.get("messages", [])
        evidence_counts.append(sum(len(m.get("evidence") or []) for m in msgs))
    avg_evidence = sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0
    return {
        "iterations": total_iters,
        "avg_evidence_per_iter": avg_evidence,
        "agents_count": len(agents),
        "token_spent": state.get("token_spent"),
        "calls": (state.get("usage") or {}).get("calls"),
    }


def log_decision(job_id: int, iteration: int, payload: Dict[str, Any]) -> None:
    """Append structured decision data (summary/votes) to a JSONL file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"job_id": job_id, "iteration": iteration, **payload}
    path = DATA_DIR / "decisions.log"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _parse_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).astimezone(timezone.utc).timestamp()
        except Exception:
            try:
                return float(value)
            except Exception:
                return None
    return None


def build_typed_timeline(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a typed timeline using RunContext if available."""
    run_ctx = state.get("run_context") or {}
    timeline = run_ctx.get("timeline") or []
    if timeline:
        return timeline

    fallback = []
    for entry in state.get("timeline") or []:
        typ = "update"
        if entry.get("votes"):
            typ = "decision"
        elif entry.get("messages"):
            typ = "debate"
        payload = entry.copy()
        ts = _parse_timestamp(entry.get("timestamp"))
        fallback.append({"type": typ, "ts": ts, "payload": payload})
    return fallback


def _build_iteration_profile(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries = sorted(timeline, key=lambda e: e.get("ts") or _parse_timestamp((e.get("payload") or {}).get("timestamp")))
    profile = []
    prev_ts = None
    for entry in entries:
        ts = entry.get("ts")
        if ts is None:
            ts = _parse_timestamp((entry.get("payload") or {}).get("timestamp"))
        duration = None
        if prev_ts is not None and ts is not None:
            duration = max(ts - prev_ts, 0.0)
        prev_ts = ts or prev_ts
        iteration = (entry.get("payload") or {}).get("iteration")
        profile.append(
            {
                "iteration": iteration,
                "type": entry.get("type"),
                "duration_sec": duration,
                "summary": (entry.get("payload") or {}).get("summary"),
            }
        )
    return profile


def _collect_unique_sources(state: Dict[str, Any]) -> List[str]:
    sources = set()
    for entry in state.get("timeline") or []:
        for msg in entry.get("messages") or []:
            for ev in msg.get("evidence") or []:
                meta = ev.get("meta") or {}
                candidate = meta.get("source") or meta.get("domain") or meta.get("document_id")
                if candidate:
                    sources.add(str(candidate))
    return sorted(sources)


def build_job_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a richer metrics payload for observability APIs."""
    timeline = build_typed_timeline(state)
    iteration_profile = _build_iteration_profile(timeline)
    durations = [p["duration_sec"] for p in iteration_profile if p.get("duration_sec") is not None]
    avg_duration = sum(durations) / len(durations) if durations else None
    run_ctx = state.get("run_context") or {}
    coverage_history = run_ctx.get("coverage_history") or []
    evidence_history = run_ctx.get("evidence_history") or []
    return {
        "research_score": state.get("research_score") or {},
        "run_metrics": state.get("run_metrics") or {},
        "tokens_spent": state.get("token_spent"),
        "timeline_length": len(timeline),
        "unique_sources": len(_collect_unique_sources(state)),
        "iteration_profile": iteration_profile,
        "avg_iteration_duration": avg_duration,
        "coverage_history": coverage_history,
        "evidence_history": evidence_history,
        "mode": state.get("convergence_mode"),
        "stagnation_reason": state.get("stagnation_reason"),
        "phase_meta": state.get("phase_meta"),
        "phase_runs": state.get("phase_runs") or [],
        "run_context": run_ctx,
    }


def read_decisions(job_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    path = DATA_DIR / "decisions.log"
    if not path.exists():
        return []
    entries = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("job_id") == job_id:
                    entries.append(rec)
    except Exception:
        return []
    return list(entries)
