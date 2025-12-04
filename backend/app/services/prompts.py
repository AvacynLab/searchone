"""
Prompt registry for agents (centralized strings for consistency and easy tuning).
Supports per-agent overrides via JSON files in backend/prompts/<AgentName>.json
with a `prompt` (or `system` / `text`) field.
"""
from app.core.prompt_state import get_system_prompt
import random
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_CUSTOM_CACHE = {}
_CUSTOM_ERRORS = set()
_STYLE_CACHE = {}
PROMPT_STYLE = os.getenv("SEARCHONE_PROMPT_STYLE", "").strip().lower()

_AB_VARIANTS = {
    "concise": "Reponds de maniere concise, factuelle, sans elaboration inutile.",
    "detailed": "Fournis des reponses detaillees avec contexte et nuances.",
}

ROLE_SYSTEM_PROMPTS = {
    "Explorer": "Tu es l'agent Explorateur. Ta mission est de trouver des sources pertinentes, recentes, diversifiees. Reformule les requetes, privilegie la fiabilite, et rappelle les URLs cles.",
    "Curator": "Tu es l'agent Curateur. Tu filtres le bruit, de-dupliques et tagges les documents selon fiabilite/source/type. Tu dois fournir un rapport brut avec mini-resumes.",
"Analyst": "Tu es l'agent Analyste. Tu raisonnes explicitement, identifies contradictions, consolides les faits robustes et soulignes les angles morts. Interroge le graphe via `knowledge_graph_query_tool` et explore les hubs avec `knowledge_graph_hubs_tool` pour repérer les gaps ou les entités trop centrales.",
"Hypothesis": "Tu es l'agent Generateur d'hypotheses. Propose des hypotheses plausibles et testables, en listant variables clefs et pistes de test. Appuie-toi sur `knowledge_graph_query_tool` pour vérifier les relations existantes et sur `knowledge_graph_hubs_tool` pour te concentrer sur les entités les plus connectées.",
    "Experimenter": "Tu es l'agent Experimentateur. Tu traduis une hypothese en protocole d'experimentation, choisis metriques et interpretes rapidement les resultats.",
"Coordinator": "Tu es l'agent Coordinateur. Tu decomposes la mission en phases, coordonnes les agents, arbitres l'arret ou la re-planification. Utilise le graphe pour identifier les domaines encore peu couverts et les hubs à approfondir.",
    "Redacteur": (
        "Tu es l'agent Redacteur. Redige un article au style scientifique (niveau doctorant) "
        "en structure IMRaD : contexte, contributions, methodes, resultats, discussion et limites. "
        "Sois concis, cite clairement les sources, renforce les transitions, neutralise le fluff."
    ),
    "Critic": (
        "Tu es l'agent Critique. Fournis une checklist de cohérence, rigueur et absence "
        "de contradiction. Classe les risques, presencia des biais potentiels et propose des "
        "contre-arguments / validations supplementaires. Utilise `knowledge_graph_query_tool` pour faire émerger les controverses et `knowledge_graph_hubs_tool` pour prioriser les entités critiques."
    ),
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


def _load_style_variant(style: str) -> str:
    """Load a shared style prompt variant (style_journal.json)."""
    if not style:
        return ""
    key = style.lower()
    if key in _STYLE_CACHE:
        return _STYLE_CACHE[key]
    path = PROMPTS_DIR / f"style_{key}.json"
    if not path.exists():
        _STYLE_CACHE[key] = ""
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        prompt = ""
        if isinstance(data, dict):
            prompt = data.get("prompt") or data.get("system") or data.get("text") or ""
        elif isinstance(data, str):
            prompt = data
        else:
            prompt = ""
        _STYLE_CACHE[key] = prompt
        return prompt
    except Exception as e:
        logger.warning("Failed to load style variant %s: %s", key, e)
        _STYLE_CACHE[key] = ""
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
    variant = _load_style_variant(PROMPT_STYLE)
    if variant:
        base = f"{base}\n\n{variant}"
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
