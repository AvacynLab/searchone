import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PARENT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from app.workflows import agents
from app.workflows.scenarios import load_scenarios


def test_scenarios_use_known_tools():
    scenarios = load_scenarios()
    assert scenarios, "Expected scenarios to be defined"
    known_tools = set(agents.TOOL_WHITELIST)
    for scenario in scenarios.values():
        for phase in scenario.phases:
            for tool in phase.tools:
                assert tool in known_tools, f"{scenario.name}.{phase.name} uses unknown tool '{tool}'"
