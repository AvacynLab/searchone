from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import DATA_DIR

ACTION_LOG_FILE = DATA_DIR / "action_log.jsonl"
STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "with",
    "without",
    "about",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
}


@dataclass
class SearchAction:
    job_id: Optional[int]
    agent_name: Optional[str]
    timestamp: str
    action_type: str
    query: str
    normalized_query: str
    result_hash: str


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_query(query: str) -> str:
    lowered = re.sub(r"[^a-z0-9\s]", " ", (query or "").lower())
    tokens = [tok for tok in lowered.split() if tok and tok not in STOPWORDS]
    return " ".join(tokens)


def _result_hash(results: List[Dict[str, Any]]) -> str:
    try:
        payload = json.dumps(results, ensure_ascii=False, sort_keys=True)
    except Exception:
        payload = str(results)
    return md5(payload.encode("utf-8")).hexdigest()


def _write_entry(entry: Dict[str, Any]) -> None:
    _ensure_data_dir()
    with ACTION_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def record_action(
    job_id: Optional[int],
    agent_name: Optional[str],
    action_type: str,
    query: str,
    results: List[Dict[str, Any]],
) -> SearchAction:
    action = SearchAction(
        job_id=job_id,
        agent_name=agent_name,
        timestamp=datetime.utcnow().isoformat() + "Z",
        action_type=action_type,
        query=query,
        normalized_query=normalize_query(query),
        result_hash=_result_hash(results),
    )
    _write_entry(asdict(action))
    return action


def _load_actions() -> List[Dict[str, Any]]:
    if not ACTION_LOG_FILE.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for line in ACTION_LOG_FILE.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    return entries


def find_similar_actions(
    job_id: Optional[int],
    normalized_query: str,
    action_type: str,
    max_age_minutes: int = 60,
) -> List[Dict[str, Any]]:
    threshold = (datetime.utcnow().timestamp() - max_age_minutes * 60)
    matches: List[Dict[str, Any]] = []
    for entry in _load_actions():
        if entry.get("action_type") != action_type:
            continue
        if normalized_query and entry.get("normalized_query") != normalized_query:
            continue
        if job_id is not None and entry.get("job_id") != job_id:
            continue
        timestamp = entry.get("timestamp")
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = 0
        if ts < threshold:
            continue
        matches.append(entry)
    return matches


def count_recent_redundant_actions(
    job_id: Optional[int],
    window_minutes: int = 15,
) -> int:
    """Return the number of repeated normalized actions for a job in the window."""
    threshold = datetime.utcnow().timestamp() - window_minutes * 60
    counter: Counter = Counter()
    for entry in _load_actions():
        if job_id is not None and entry.get("job_id") != job_id:
            continue
        timestamp = entry.get("timestamp")
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = 0
        if ts < threshold:
            continue
        normalized = entry.get("normalized_query") or ""
        action_type = entry.get("action_type") or ""
        if not normalized or not action_type:
            continue
        counter[(normalized, action_type)] += 1
    redundant = sum(count - 1 for count in counter.values() if count > 1)
    return redundant
