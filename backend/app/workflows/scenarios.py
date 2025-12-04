from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging
import yaml
from pydantic import BaseModel, Field, validator

from app.core.config import BACKEND_ROOT
from app.core.logging_config import configure_logging
from app.workflows.agents import TOOL_WHITELIST

configure_logging()
logger = logging.getLogger(__name__)

SCENARIOS_PATH = (BACKEND_ROOT.parent / "configs" / "research_scenarios.yaml").resolve()
KNOWN_TOOLS = set(TOOL_WHITELIST)


class PhaseSpec(BaseModel):
    name: str
    description: Optional[str] = None
    agents: List[str]
    tools: Optional[List[str]] = Field(default_factory=list)
    max_iterations: Optional[int] = Field(None, gt=0)
    max_duration_seconds: Optional[int] = Field(None, gt=0)
    exit_criteria: Dict[str, Any] = Field(default_factory=dict)
    parallel_group: Optional[str] = None
    domain: Optional[str] = None

    @validator("agents", pre=True, each_item=False)
    def ensure_agents(cls, v: Any) -> List[str]:
        if not v:
            raise ValueError("Phase must declare at least one agent.")
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return [str(v).strip()]

    @validator("tools", pre=True, each_item=False)
    def normalize_tools(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return [str(v).strip()]


class ScenarioSpec(BaseModel):
    name: str
    objective: str
    phases: List[PhaseSpec]

    @validator("phases")
    def ensure_phases(cls, value: List[PhaseSpec]) -> List[PhaseSpec]:
        if not value:
            raise ValueError("Scenario must define at least one phase.")
        return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        return {}


@lru_cache(maxsize=1)
def load_scenarios(path: Optional[Path] = None) -> Dict[str, ScenarioSpec]:
    cfg_path = path or SCENARIOS_PATH
    data = _load_yaml(cfg_path)
    raw = data.get("scenarios") or {}
    scenarios: Dict[str, ScenarioSpec] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        try:
            spec = ScenarioSpec(name=name, **entry)
        except Exception:
            continue
        invalid_tools = []
        for phase in spec.phases:
            for tool in phase.tools:
                if tool not in KNOWN_TOOLS:
                    invalid_tools.append(tool)
        if invalid_tools:
            logger.warning("Scenario %s references unknown tools: %s", name, invalid_tools)
            raise ValueError(f"Unknown tool names in scenario '{name}': {invalid_tools}")
        scenarios[name] = spec
    return scenarios


def list_scenarios() -> List[str]:
    return list(load_scenarios().keys())


def get_scenario(name: str) -> Optional[ScenarioSpec]:
    if not name:
        return None
    return load_scenarios().get(name)
