"""
Simple guardrails for high-risk domains (health, bio, etc.).
Can be extended with taxonomy/regex as needed.
"""
from typing import List, Dict, Any

RISK_KEYWORDS = [
    "bio",
    "bioweapon",
    "virus",
    "pathogen",
    "chemical weapon",
    "toxine",
    "toxins",
    "nuclear",
    "armement",
    "armes",
    "sante",
    "medical",
    "traitement",
]


def is_high_risk(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in RISK_KEYWORDS)


def filter_tools(tools: List[str], allow_web: bool = True) -> List[str]:
    """Remove web-facing tools when not allowed."""
    if allow_web:
        return tools
    blocked = {"web_search_tool", "fetch_and_parse_url"}
    return [t for t in tools if t not in blocked]


def audit_entry(reason: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"reason": reason, "payload": payload}
