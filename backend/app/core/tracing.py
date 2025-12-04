"""Lightweight OpenTelemetry helpers (no-op if opentelemetry not installed)."""
import os
from contextlib import contextmanager

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.trace import Tracer  # type: ignore
    _HAS_OTEL = True
except Exception:  # pragma: no cover - optional dependency
    trace = None  # type: ignore
    Tracer = None  # type: ignore
    _HAS_OTEL = False

SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "searchone-backend")
_tracer = trace.get_tracer(SERVICE_NAME) if _HAS_OTEL else None


def get_tracer():
    return _tracer


@contextmanager
def start_span(name: str):
    """Context manager that starts an OTEL span if tracing is available."""
    if not _HAS_OTEL or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:  # type: ignore[attr-defined]
        yield span
