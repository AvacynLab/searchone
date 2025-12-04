from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class DebateRound:
    question: str
    arguments_for: List[str]
    arguments_against: List[str]
    critics: List[str] = None
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def tally_votes(
    votes: List[Dict[str, Any]],
    mode: str = "simple",
    weights: Optional[Dict[str, float]] = None,
    veto_role: Optional[str] = None,
) -> Dict[str, Any]:
    """Majority tally supporting simple, weighted, and veto modes."""
    weights = weights or {}
    counts = {"agree": 0.0, "neutral": 0.0, "disagree": 0.0}
    veto_triggered = False
    for v in votes or []:
        agent = v.get("agent") or ""
        for d in v.get("votes", []):
            val = (d.get("vote") or "").lower()
            weight = weights.get(agent, 1.0) if mode == "weighted" else 1.0
            if veto_role and agent == veto_role and val == "disagree":
                veto_triggered = True
            if val in counts:
                counts[val] += weight
    total = sum(counts.values()) or 1.0
    scores = {k: float(v) / total for k, v in counts.items()}
    decision = "pending"
    thresholds = {"adopt": 0.6, "reject": 0.6}
    if veto_triggered:
        decision = "reject"
        thresholds["veto"] = veto_role
    elif scores["agree"] >= thresholds["adopt"]:
        decision = "adopt"
    elif scores["disagree"] >= thresholds["reject"]:
        decision = "reject"
    return {"counts": counts, "scores": scores, "decision": decision, "thresholds": thresholds}


def run_debate(question: str, arguments: List[str], critics: List[str] = None) -> DebateRound:
    """Split arguments into for/against and build a concise summary."""
    critics = critics or []
    pros = [a for a in arguments if a.lower().startswith(("pro:", "+", "for:"))]
    cons = [a for a in arguments if a.lower().startswith(("con:", "-", "against:"))]
    # fallback: alternate assignment to keep balance
    if not pros and not cons:
        for i, arg in enumerate(arguments):
            (pros if i % 2 == 0 else cons).append(arg)
    summary = f"{len(pros)} arguments pour, {len(cons)} contre. Critics: {len(critics)}"
    return DebateRound(question=question, arguments_for=pros, arguments_against=cons, critics=critics, summary=summary)
