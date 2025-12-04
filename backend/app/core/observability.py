from typing import Dict, Any
import json
from pathlib import Path
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
