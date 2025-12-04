"""
Coordinator actions: decide on replan/collecte complementaire based on votes/coverage.
Publishes actions on the bus for dynamic pipeline insertion.
"""
from typing import Dict, Any, List, Optional

from app.data.knowledge_store import mark_polluted


def _publish_action(action: Dict[str, Any], bus, state: Dict[str, Any]) -> Dict[str, Any]:
    if bus and action:
        bus.publish("actions", action)
    if action:
        state.setdefault("coordinator_actions", []).append(action)
    return action


def focus_on_conflicts(state: Dict[str, Any], bus, query: str) -> Dict[str, Any]:
    votes = state.get("votes") or {}
    conflicts: List[str] = []
    for rounds in votes.values():
        for vote in rounds.get("votes") or []:
            if (vote.get("vote") or "").startswith("disagree"):
                conflicts.append(vote.get("justification") or rounds.get("hypothesis") or query)
    reason = conflicts[0] if conflicts else "contradictions detectees"
    action = {
        "type": "focus_conflicts",
        "query": f"{query} contradictions ({reason[:80]})",
        "reason": reason,
        "targets": conflicts[:3],
    }
    return _publish_action(action, bus, state)


def seek_additional_sources(state: Dict[str, Any], bus, topic: Optional[str] = None) -> Dict[str, Any]:
    base = topic or state.get("query") or "sources supplementaires"
    action = {
        "type": "seek_additional_sources",
        "query": f"{base} nouvelles sources",
        "topic": base,
        "reason": "richesse faible de documents",
    }
    return _publish_action(action, bus, state)


def downgrade_low_quality_sources(state: Dict[str, Any], threshold: float = 0.35, limit: int = 5) -> Dict[str, Any]:
    timeline = reversed(state.get("timeline") or [])
    sources_marked = []
    for entry in timeline:
        for msg in entry.get("messages") or []:
            for ev in msg.get("evidence") or []:
                meta = ev.get("meta") or {}
                score = float(meta.get("reliability") or meta.get("score") or 0.0)
                source = meta.get("source") or meta.get("domain")
                if source and score < threshold and source not in sources_marked:
                    try:
                        mark_polluted(source, reason="low_quality")
                        sources_marked.append(source)
                    except Exception:
                        continue
                if len(sources_marked) >= limit:
                    break
            if len(sources_marked) >= limit:
                break
        if len(sources_marked) >= limit:
            break
    action = {
        "type": "downgrade_low_quality_sources",
        "marked": sources_marked,
        "threshold": threshold,
    }
    state.setdefault("coordinator_actions", []).append(action)
    return action


def evaluate_replan(
    state: Dict[str, Any],
    bus,
    query: str,
    stagnation_reason: Optional[str] = None,
    research_score: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    new_sources: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Inspect state (votes, run_metrics) and publish replan actions when needed.
    """
    actions: List[Dict[str, Any]] = []
    votes = state.get("votes") or {}
    latest_key = sorted(votes.keys())[-1] if votes else None
    latest = votes.get(latest_key, {}) if latest_key else {}
    scores = latest.get("scores") or {}
    decision = latest.get("decision") or ""
    metrics = state.get("run_metrics") or {}
    coverage = metrics.get("coverage_score", 0.0)
    evidence = metrics.get("evidence_count", 0)

    if decision == "reject" or coverage < 0.1 or evidence == 0:
        action = {
            "type": "replan_collect",
            "reason": "reject" if decision == "reject" else "low_coverage",
            "query": f"{query} collecte complementaire ciblee",
        }
        if bus:
            bus.publish("actions", action)
        actions.append(action)

    if decision == "reject" and scores.get("disagree", 0.0) > 0:
        try:
            mark_polluted(query, reason="council_reject")
        except Exception:
            pass

    if coverage < 0.3 and (new_sources or 0) < 2:
        action = seek_additional_sources(state, bus, query)
        actions.append(action)

    if stagnation_reason in ("coverage_stagnation", "score_stagnation", "repeated_arguments"):
        action = focus_on_conflicts(state, bus, query)
        actions.append(action)

    if stagnation_reason in ("low_new_sources", "no_evidence_window") or (research_score and (research_score.get("coverage") or 0.0) < 0.2):
        action = seek_additional_sources(state, bus, query)
        actions.append(action)

    if stagnation_reason == "score_stagnation" and mode == "closure":
        conclude_action = {"type": "conclude", "reason": "score_stagnation_closure"}
        state.setdefault("coordinator_actions", []).append(conclude_action)
        actions.append(conclude_action)

    if coverage < 0.25:
        downgrade_low_quality_sources(state)

    return actions
