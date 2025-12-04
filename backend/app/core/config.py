from pathlib import Path
import os
import logging
from dotenv import load_dotenv
import json

# Resolve the backend root regardless of where this module lives in the package tree
BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")

DATA_DIR = BACKEND_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = DATA_DIR / "db.sqlite"
FAISS_INDEX_FILE = DATA_DIR / "faiss.index"
MAPPING_FILE = DATA_DIR / "id_mapping.json"

# Embedding model for MVP (local)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM provider routing
PROVIDER = os.getenv("PROVIDER", "openrouter").lower()  # "openrouter" or "local"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-mini:free")
# Optional local inference via LM Studio (OpenAI-compatible server)
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL")  # e.g., http://127.0.0.1:1234/v1/chat/completions
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL")

# SearxNG metasearch endpoint (optional)
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:2003")

# If using Docker secrets, the secret may be mounted at /run/secrets/openrouter_api_key
if not OPENROUTER_API_KEY:
    secret_path = os.getenv('OPENROUTER_API_KEY_FILE') or '/run/secrets/openrouter_api_key'
    try:
        p = Path(secret_path)
        if p.exists():
            OPENROUTER_API_KEY = p.read_text().strip()
    except Exception:
        OPENROUTER_API_KEY = OPENROUTER_API_KEY

# Application metadata
APP_VERSION = os.getenv("SEARCHONE_VERSION", "0.0.1")
JOB_TOKEN_BUDGET = int(os.getenv("SEARCHONE_JOB_TOKEN_BUDGET", "0"))
WEB_QUERY_BUDGET = int(os.getenv("SEARCHONE_WEB_QUERY_BUDGET", "0"))
LOG_LEVEL = os.getenv("SEARCHONE_LOG_LEVEL", "INFO").upper()

# Search caching parameters
WEB_CACHE_ENABLED = os.getenv("SEARCHONE_WEB_CACHE_ENABLED", "1").lower() in ("1", "true", "yes", "on")
WEB_CACHE_TTL_SECONDS = int(os.getenv("SEARCHONE_WEB_CACHE_TTL_SECONDS", "600"))
WEB_CACHE_CLEANUP_INTERVAL = int(os.getenv("SEARCHONE_WEB_CACHE_CLEANUP_INTERVAL", "600"))
WEB_SEARCH_ENGINE_NAME = os.getenv("SEARCHONE_WEB_SEARCH_ENGINE", "searxng")
WEB_SEARCH_ENGINE_LIST = tuple(
    sorted(
        {
            entry.strip()
            for entry in os.getenv("SEARCHONE_WEB_SEARCH_ENGINES", WEB_SEARCH_ENGINE_NAME).split(",")
            if entry.strip()
        }
    )
)
WEB_SEARCH_ENGINE_SET = ",".join(WEB_SEARCH_ENGINE_LIST) or "default"
WEB_SEARCH_FAILURE_THRESHOLD = int(os.getenv("SEARCHONE_WEB_SEARCH_FAILURE_THRESHOLD", "3"))
WEB_SEARCH_BREAKER_COOLDOWN = int(os.getenv("SEARCHONE_WEB_SEARCH_BREAKER_COOLDOWN", "90"))

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

CONFIG_FILE = BACKEND_ROOT / "config.json"
_CONFIG: dict = {}
MODEL_PROFILES_CFG = {}
ROLE_MODEL_OVERRIDES = {}
SAMPLING_DEFAULTS = {}
TIMEOUTS_DEFAULTS = {}
LOGGING_CONFIG = {}


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to read config.json: %s", e)
        return {}


def _populate_from_config() -> None:
    global MODEL_PROFILES_CFG, ROLE_MODEL_OVERRIDES, SAMPLING_DEFAULTS, TIMEOUTS_DEFAULTS, LOGGING_CONFIG, _CONFIG
    _CONFIG = _load_config()
    MODEL_PROFILES_CFG = _CONFIG.get("model_profiles") or {}
    ROLE_MODEL_OVERRIDES = _CONFIG.get("role_models") or {}
    SAMPLING_DEFAULTS = _CONFIG.get("sampling") or {}
    TIMEOUTS_DEFAULTS = _CONFIG.get("timeouts") or {}
    LOGGING_CONFIG = _CONFIG.get("logging") or {}


_populate_from_config()

logger = logging.getLogger(__name__)


def validate_config() -> dict:
    """Return config validation results and emit warnings for missing critical items."""
    warnings = []
    if PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            warnings.append("OPENROUTER_API_KEY missing (LLM calls will fail).")
        if not os.getenv("OPENROUTER_SITE"):
            warnings.append("OPENROUTER_SITE missing (set HTTP-Referer recommended by OpenRouter).")
        if not os.getenv("OPENROUTER_TITLE"):
            warnings.append("OPENROUTER_TITLE missing (set X-Title recommended by OpenRouter).")
    if PROVIDER == "local" and not LMSTUDIO_URL:
        warnings.append("LMSTUDIO_URL missing while PROVIDER=local (LLM calls will fail).")
    for w in warnings:
        logger.warning(w)
    return {
        "warnings": warnings,
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "provider": PROVIDER,
    }
