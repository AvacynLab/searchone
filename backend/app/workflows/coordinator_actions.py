"""
Coordinator actions: decide on replan/collecte complementaire based on votes/coverage.
Publishes actions on the bus for dynamic pipeline insertion.
"""
from typing import Dict, Any


def evaluate_replan(state: Dict[str, Any], bus, query: str) -> None:
    """
    Inspect state (votes, run_metrics) and publish replan actions when needed.
    """
    if not bus:
        return
    votes = state.get("votes") or {}
    latest_key = sorted(votes.keys())[-1] if votes else None
    latest = votes.get(latest_key, {}) if latest_key else {}
    scores = latest.get("scores") or {}
    decision = latest.get("decision") or ""
    metrics = state.get("run_metrics") or {}
    coverage = metrics.get("coverage_score", 0.0)
    evidence = metrics.get("evidence_count", 0)

    # Trigger collection if reject or low coverage/evidence
    if decision == "reject" or coverage < 0.1 or evidence == 0:
        bus.publish(
            "actions",
            {
                "type": "replan_collect",
                "reason": "reject" if decision == "reject" else "low_coverage",
                "query": f"{query} collecte complementaire ciblee",
            },
        )
