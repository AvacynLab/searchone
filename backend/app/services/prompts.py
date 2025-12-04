"""
Prompt registry for agents (centralized strings for consistency and easy tuning).
Supports per-agent overrides via JSON files in backend/prompts/<AgentName>.json
with a `prompt` (or `system` / `text`) field.
"""
from app.core.prompt_state import get_system_prompt
import random
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_CUSTOM_CACHE = {}
_CUSTOM_ERRORS = set()

_AB_VARIANTS = {
    "concise": "Reponds de maniere concise, factuelle, sans elaboration inutile.",
    "detailed": "Fournis des reponses detaillees avec contexte et nuances.",
}

ROLE_SYSTEM_PROMPTS = {
    "Explorer": "Tu es l'agent Explorateur. Ta mission est de trouver des sources pertinentes, recentes, diversifiees. Reformule les requetes, privilegie la fiabilite, et rappelle les URLs cles.",
    "Curator": "Tu es l'agent Curateur. Tu filtres le bruit, de-dupliques et tagges les documents selon fiabilite/source/type. Tu dois fournir un rapport brut avec mini-resumes.",
    "Analyst": "Tu es l'agent Analyste. Tu raisonnes explicitement, identifies contradictions, consolides les faits robustes et soulignes les angles morts.",
    "Hypothesis": "Tu es l'agent Generateur d'hypotheses. Propose des hypotheses plausibles et testables, en listant variables clefs et pistes de test.",
    "Experimenter": "Tu es l'agent Experimentateur. Tu traduis une hypothese en protocole d'experimentation, choisis metriques et interpretes rapidement les resultats.",
    "Coordinator": "Tu es l'agent Coordinateur. Tu decomposes la mission en phases, coordonnes les agents, arbitres l'arret ou la re-planification.",
    "Redacteur": "Tu es l'agent Redacteur. Tu produis un brouillon scientifique structure IMRaD, sobre, avec citations claires et references en fin de texte.",
    "Critic": "Tu es l'agent Critique. Tu pointes les failles, risques et biais; tu proposes des contre-arguments et tu demandes des confirmations supplementaires.",
}


def _load_custom_prompt(agent_name: str) -> str:
    """Load custom prompt override from prompts/<AgentName>.json (cached)."""
    if not agent_name:
        return ""
    if agent_name in _CUSTOM_CACHE:
        return _CUSTOM_CACHE[agent_name]
    path = PROMPTS_DIR / f"{agent_name}.json"
    if not path.exists():
        _CUSTOM_CACHE[agent_name] = ""
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            prompt = data.get("prompt") or data.get("system") or data.get("text") or ""
        elif isinstance(data, str):
            prompt = data
        else:
            prompt = ""
        _CUSTOM_CACHE[agent_name] = prompt
        return prompt
    except Exception as e:
        if agent_name not in _CUSTOM_ERRORS:
            logger.warning("Failed to load custom prompt for %s: %s", agent_name, e)
            _CUSTOM_ERRORS.add(agent_name)
        _CUSTOM_CACHE[agent_name] = ""
        return ""


def _role_prefix(agent_role: str, agent_name: str = "") -> str:
    custom = _load_custom_prompt(agent_name)
    base = custom or ROLE_SYSTEM_PROMPTS.get(agent_role) or ""
    sys = get_system_prompt()
    if sys:
        base = f"{sys}\n\n{base}"
    return base.strip()


def propose_prompt(agent_name: str, agent_role: str, context: str) -> str:
    prefix = _role_prefix(agent_role, agent_name)
    header = f"{prefix}\n\n" if prefix else ""
    return (
        f"{header}Agent {agent_name} (role: {agent_role}) propose une hypothese courte pour:\n"
        f"{context}\n"
    )


def vote_prompt(agent_name: str, agent_role: str, hypothesis: str, evidence: str = "") -> str:
    prefix = _role_prefix(agent_role, agent_name)
    parts = []
    if prefix:
        parts.append(prefix)
    parts += [
        f"Vous etes l'agent {agent_name} (role: {agent_role}).",
        "Pour l'hypothese suivante, indiquez seulement une valeur parmi: agree / neutral / disagree, puis une justification courte (une phrase).",
    ]
    if evidence:
        parts.append(f"Elements factuels disponibles:\n{evidence}")
    parts.append(f"Hypothese: {hypothesis}")
    parts.append("Reponse attendue (format: VOTE -- justification):")
    return "\n".join(parts)
