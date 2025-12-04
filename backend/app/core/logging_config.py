import logging
import os
import warnings

DEFAULT_LOG_LEVEL = os.getenv("SEARCHONE_LOG_LEVEL", "INFO")


def _resolve_level(value) -> int:
    if isinstance(value, str):
        return getattr(logging, value.upper(), logging.INFO)
    if isinstance(value, int):
        return value
    return logging.INFO


def configure_logging(level=None):
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    resolved = _resolve_level(level or DEFAULT_LOG_LEVEL)
    logging.basicConfig(level=resolved, format=fmt)
    # reduce verbosity of some libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    # silence noisy SwigPy* deprecations (faiss/redis bindings)
    warnings.filterwarnings("ignore", message=".*SwigPy.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*swigvarlink.*", category=DeprecationWarning)

# call configure_logging() from modules that import this file
