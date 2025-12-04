import yaml

from app.workflows import scenarios


def test_loads_custom_scenario(tmp_path, monkeypatch):
    payload = {
        "scenarios": {
            "custom_plan": {
                "objective": "Tester le loader",
                "phases": [
                    {
                        "name": "explore",
                        "agents": ["Analyst"],
                        "tools": ["web_search_tool"],
                        "max_iterations": 2,
                        "max_duration_seconds": 120,
                        "exit_criteria": {"coverage": 0.5},
                    }
                ],
            }
        }
    }
    scenario_path = tmp_path / "research_scenarios.yaml"
    scenario_path.write_text(yaml.dump(payload), encoding="utf-8")
    monkeypatch.setattr(scenarios, "SCENARIOS_PATH", scenario_path)
    scenarios.load_scenarios.cache_clear()

    loaded = scenarios.load_scenarios()
    assert "custom_plan" in loaded
    scenario = scenarios.get_scenario("custom_plan")
    assert scenario is not None
    assert scenario.objective.startswith("Tester")
    assert scenario.phases[0].agents == ["Analyst"]
    assert scenario.phases[0].tools == ["web_search_tool"]
    assert "custom_plan" in scenarios.list_scenarios()


def test_list_scenarios_empty(monkeypatch, tmp_path):
    empty_path = tmp_path / "missing.yaml"
    monkeypatch.setattr(scenarios, "SCENARIOS_PATH", empty_path)
    scenarios.load_scenarios.cache_clear()
    assert scenarios.list_scenarios() == []
