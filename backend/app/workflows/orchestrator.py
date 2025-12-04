import asyncio
from typing import List, Dict, Any, Optional

from app.core.config import JOB_TOKEN_BUDGET
from app.workflows.agents import run_agents_job
from app.workflows.runtime import RunContext, MessageBus, ConvergenceController
from app.services.llm import LLMClient
from app.workflows.workflow import WorkflowEngine
from app.workflows.scenarios import get_scenario, list_scenarios


class Orchestrator:
    """Minimal orchestrator that instantiates a run (agents + council) and exposes a unified entrypoint."""

    def __init__(
        self,
        roles: Optional[List[str]] = None,
        max_iterations: int = 5,
        max_duration_seconds: int = 300,
        max_token_budget: int = JOB_TOKEN_BUDGET,
    ):
        self.roles = roles or []
        self.max_iterations = max_iterations
        self.max_duration_seconds = max_duration_seconds
        self.max_token_budget = max_token_budget

    async def run(
        self,
        job_id: int,
        query: str,
        llm_client: Optional[LLMClient] = None,
        pipeline: dict = None,
    ) -> Dict[str, Any]:
        """Execute a research run with the configured agents."""
        bus = MessageBus()
        ctx = RunContext(job_id=job_id, query=query, roles=self.roles)
        controller = ConvergenceController()
        wf = pipeline or {
            "steps": [
                {"name": "agents_run", "fn": run_agents_job, "args": [], "kwargs": {"job_id": job_id, "query": query, "roles": self.roles, "llm_client": llm_client, "max_iterations": self.max_iterations, "max_duration_seconds": self.max_duration_seconds, "max_token_budget": self.max_token_budget, "bus": bus, "run_ctx": ctx, "controller": controller}}
            ]
        }
        engine = WorkflowEngine(steps=wf["steps"])
        result = await engine.run(context={"job_id": job_id, "query": query, "roles": self.roles})
        return result[-1].get("result") if result else {}

    async def run_with_scenario(
        self,
        job_id: int,
        query: str,
        scenario_name: str,
        llm_client: Optional[LLMClient] = None,
    ) -> Dict[str, Any]:
        scenario = get_scenario(scenario_name)
        if scenario is None:
            raise ValueError(f"Scenario '{scenario_name}' not found.")
        base_llm = llm_client or LLMClient()
        bus = MessageBus()
        ctx = RunContext(job_id=job_id, query=query, roles=self.roles)
        controller = ConvergenceController()
        base_kwargs = {
            "job_id": job_id,
            "query": query,
            "llm_client": base_llm,
            "bus": bus,
            "run_ctx": ctx,
            "controller": controller,
            "max_token_budget": self.max_token_budget,
            "max_iterations": self.max_iterations,
            "max_duration_seconds": self.max_duration_seconds,
        }
        entries = []
        seen_groups = set()
        for phase in scenario.phases:
            group_key = phase.parallel_group
            if group_key:
                if group_key in seen_groups:
                    continue
                group_phases = [p for p in scenario.phases if p.parallel_group == group_key]
                seen_groups.add(group_key)
                entries.append(("group", group_key, group_phases))
            else:
                entries.append(("single", phase.name, phase))
        steps = []
        for idx, entry in enumerate(entries):
            if entry[0] == "group":
                _, group_name, phases = entry
                steps.append(
                    {
                        "name": f"group_{group_name}",
                        "fn": self._run_phase_group,
                        "kwargs": {"phases": phases, "group_name": group_name, "base_kwargs": base_kwargs},
                    }
                )
            else:
                _, _, phase = entry
                steps.append(
                    {
                        "name": f"phase_{phase.name}",
                        "fn": self._run_phase,
                        "kwargs": {"phase": phase, "base_kwargs": base_kwargs},
                    }
                )
        engine = WorkflowEngine(steps=steps)
        context = {
            "scenario": scenario.name,
            "objective": scenario.objective,
            "job_id": job_id,
            "query": query,
        }
        results = await engine.run(context=context)
        return {
            "scenario": scenario.name,
            "objective": scenario.objective,
            "results": results,
            "phase_history": context.get("phase_runs", []),
        }

    async def _run_phase(
        self,
        phase,
        base_kwargs: Dict[str, Any],
        context: Dict[str, Any],
        **_: Any,
    ) -> Dict[str, Any]:
        kwargs = self._prepare_phase_kwargs(phase, base_kwargs)
        phase_result = await run_agents_job(**kwargs)
        context.setdefault("phase_runs", []).append({"phase": phase.name, "result": phase_result, "meta": kwargs.get("phase_meta")})
        return {"phase": phase.name, "result": phase_result}

    async def _run_phase_group(
        self,
        phases,
        group_name: str,
        base_kwargs: Dict[str, Any],
        context: Dict[str, Any],
        **_: Any,
    ) -> Dict[str, Any]:
        group_results = []
        for phase in phases:
            res = await self._run_phase(phase, base_kwargs, context=context)
            group_results.append(res)
        return {"group": group_name, "results": group_results}

    def _prepare_phase_kwargs(self, phase, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        max_iters = max(base_kwargs.get("max_iterations") or 1, 1)
        max_duration = max(base_kwargs.get("max_duration_seconds") or 1, 1)
        kwargs = {
            **base_kwargs,
            "roles": phase.agents,
            "max_iterations": phase.max_iterations or max_iters,
            "max_duration_seconds": phase.max_duration_seconds or max_duration,
            "tool_allowlist": phase.tools or None,
            "phase_meta": {
                "name": phase.name,
                "description": phase.description,
                "domain": phase.domain,
                "exit_criteria": phase.exit_criteria,
            },
        }
        return kwargs

    def get_available_scenarios(self) -> List[str]:
        return list_scenarios()

    def run_sync(
        self,
        job_id: int,
        query: str,
        llm_client: Optional[LLMClient] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for environments without an event loop."""
        return asyncio.get_event_loop().run_until_complete(self.run(job_id, query, llm_client=llm_client))
