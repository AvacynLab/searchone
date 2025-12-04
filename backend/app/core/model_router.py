import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from app.core.config import (
    DATA_DIR,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    MODEL_PROFILES_CFG,
    ROLE_MODEL_OVERRIDES,
    SAMPLING_DEFAULTS,
)

logger = logging.getLogger(__name__)

MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
CACHE_FILE = DATA_DIR / "openrouter_models.json"
CACHE_TTL_SECONDS = int(os.getenv("OPENROUTER_MODELS_CACHE_TTL", "900"))
HTTP_TIMEOUT = float(os.getenv("OPENROUTER_MODELS_HTTP_TIMEOUT", "10"))

# Minimal profiles aligned with the backlog (heavy brain, fast draft, code)
DEFAULT_MODEL_PROFILES: Dict[str, Dict[str, List[str]]] = {
    "brain": {
        "prefer": [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-sonnet",
            "openai/gpt-4.1",
            "openai/gpt-4o",
        ],
        "fallback": [],
    },
    "fast": {
        "prefer": [
            "openai/gpt-4o-mini",
            "google/gemini-flash-1.5",
            "mistralai/mistral-small",
        ],
        "fallback": [],
    },
    "code": {
        "prefer": [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4.1",
            "deepseek/deepseek-coder",
        ],
        "fallback": [],
    },
}
MODEL_PROFILES = MODEL_PROFILES_CFG or DEFAULT_MODEL_PROFILES


def _headers() -> Dict[str, str]:
    site = os.getenv("OPENROUTER_SITE")
    title = os.getenv("OPENROUTER_TITLE", "SearchOne")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
        "HTTP-Referer": site or "",
        "X-Title": title,
    }
    # Clean empty headers to avoid 400
    return {k: v for k, v in headers.items() if v}


def _cache_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    try:
        age = time.time() - CACHE_FILE.stat().st_mtime
        return age < CACHE_TTL_SECONDS
    except Exception:
        return False


def _read_cache() -> List[Dict]:
    try:
        return json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_cache(models: List[Dict]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        Path(CACHE_FILE).write_text(json.dumps(models), encoding="utf-8")
    except Exception as e:
        logger.debug("Could not write model cache: %s", e)


def fetch_models(force: bool = False) -> List[Dict]:
    """Fetch OpenRouter models with caching. Returns an empty list on failure."""
    if not force and _cache_fresh():
        return _read_cache()
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY missing; returning cached models if any.")
        return _read_cache()

    try:
        resp = httpx.get(MODELS_ENDPOINT, headers=_headers(), timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        _write_cache(data)
        return data
    except Exception as e:
        logger.warning("Failed to fetch models from OpenRouter: %s", e, exc_info=True)
        return _read_cache()


def list_models(force_refresh: bool = False) -> List[str]:
    """Return model ids only (cached)."""
    return [m.get("id") for m in fetch_models(force=force_refresh) if m.get("id")]


def pick_model(profile: str = "brain", models: Optional[List[Dict]] = None) -> str:
    """Pick the best model id for a profile or fall back to OPENROUTER_MODEL."""
    models = models or fetch_models()
    profile_cfg = MODEL_PROFILES.get(profile, {})
    prefer = profile_cfg.get("prefer", [])
    fallback = profile_cfg.get("fallback", [])
    candidates = [m for m in models if m.get("id")]
    for candidate in prefer + fallback:
        if any(m.get("id") == candidate for m in candidates):
            return candidate
    # Fallback to configured default or first available
    if OPENROUTER_MODEL:
        return OPENROUTER_MODEL
    return candidates[0].get("id") if candidates else "openai/gpt-4o"


def profile_for_role(role: str) -> str:
    """Map an agent role to a profile name."""
    # override by config
    if role in ROLE_MODEL_OVERRIDES:
        return ROLE_MODEL_OVERRIDES[role]
    role_lower = (role or "").lower()
    if any(key in role_lower for key in ["critic", "analyst", "reason", "brain"]):
        return "brain"
    if any(key in role_lower for key in ["code", "dev", "experiment", "builder"]):
        return "code"
    return "fast"


def resolve_model(role: Optional[str] = None, profile: Optional[str] = None) -> str:
    """Resolve the model id for an agent role or explicit profile."""
    target_profile = profile or profile_for_role(role or "")
    try:
        return pick_model(target_profile)
    except Exception:
        logger.warning("Falling back to OPENROUTER_MODEL for role=%s profile=%s", role, target_profile)
        return OPENROUTER_MODEL or "openai/gpt-4o"
