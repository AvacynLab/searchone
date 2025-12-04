import logging
import warnings

def configure_logging(level=logging.INFO):
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)
    # reduce verbosity of some libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    # silence noisy SwigPy* deprecations (faiss/redis bindings)
    warnings.filterwarnings("ignore", message=".*SwigPy.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*swigvarlink.*", category=DeprecationWarning)

# call configure_logging() from modules that import this file
