"""Runtime prompt configuration (system prompt override + AB variant)."""
import os
from typing import Optional

_system_prompt = os.getenv("SEARCHONE_SYSTEM_PROMPT", "")
_ab_variant = None  # type: Optional[str]


def get_system_prompt() -> str:
    base = _system_prompt or ""
    variant_prompt = ""
    if _ab_variant:
        from prompts import _AB_VARIANTS  # type: ignore
        variant_prompt = _AB_VARIANTS.get(_ab_variant, "")
    return "\n".join([p for p in [base, variant_prompt] if p])


def set_system_prompt(value: Optional[str]) -> str:
    global _system_prompt
    _system_prompt = value or ""
    return _system_prompt


def set_prompt_variant(variant: Optional[str]) -> str:
    global _ab_variant
    _ab_variant = variant
    return _ab_variant or ""


def get_prompt_variant() -> Optional[str]:
    return _ab_variant
