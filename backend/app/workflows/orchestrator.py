import asyncio
from typing import List, Dict, Any, Optional

from app.workflows.agents import run_agents_job
from app.workflows.runtime import RunContext, MessageBus, AgentSpec, ConvergenceController
from app.services.llm import LLMClient
from app.workflows.workflow import WorkflowEngine


class Orchestrator:
    """Minimal orchestrator that instantiates a run (agents + council) and exposes a unified entrypoint."""

    def __init__(
        self,
        roles: Optional[List[str]] = None,
        max_iterations: int = 5,
        max_duration_seconds: int = 300,
        max_token_budget: int = 0,
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

    def run_sync(
        self,
        job_id: int,
        query: str,
        llm_client: Optional[LLMClient] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for environments without an event loop."""
        return asyncio.get_event_loop().run_until_complete(self.run(job_id, query, llm_client=llm_client))
