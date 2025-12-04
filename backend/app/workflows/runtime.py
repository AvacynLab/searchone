import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol


@dataclass
class AgentSpec:
    name: str
    role_description: str
    capabilities: List[str]
    model_profile: str
    memory_scopes: List[str]
    allowed_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role_description": self.role_description,
            "capabilities": list(self.capabilities),
            "model_profile": self.model_profile,
            "memory_scopes": list(self.memory_scopes),
            "allowed_tools": list(self.allowed_tools),
        }


class AgentInterface(Protocol):
    """Lightweight interface for agent behaviors (plan/act/reflect)."""

    async def plan(self, observation: str) -> str: ...
    async def act(self, shared_context: str, council_inbox) -> None: ...
    async def reflect(self) -> str: ...


class MessageBus:
    """Minimal in-memory bus for intra-run messages (topic-based)."""

    def __init__(self):
        self.topics: Dict[str, List[Dict[str, Any]]] = {}

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        self.topics.setdefault(topic, []).append(payload)

    def drain(self, topic: str) -> List[Dict[str, Any]]:
        msgs = self.topics.get(topic, [])
        self.topics[topic] = []
        return msgs

    def peek(self, topic: str) -> List[Dict[str, Any]]:
        return list(self.topics.get(topic, []))

    def snapshot(self) -> Dict[str, Any]:
        return {k: list(v) for k, v in self.topics.items()}


class RunContext:
    """Lightweight run context holding timeline and messages for observability."""

    def __init__(self, job_id: int, query: str, roles: List[str]):
        self.job_id = job_id
        self.query = query
        self.roles = list(roles)
        self.created_at = time.time()
        self.status = "running"
        self.timeline: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.coverage_history: List[float] = []
        self.hypothesis_history: List[str] = []
        self.evidence_history: List[int] = []

    def add_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.timeline.append({"type": event_type, "ts": time.time(), "payload": payload})

    def record_message(self, sender: Optional[str], payload: Dict[str, Any]) -> None:
        self.messages.append({"sender": sender, "ts": time.time(), "payload": payload})

    def track_iteration(self, coverage: float, evidence_count: int, hypotheses: List[str]) -> None:
        self.coverage_history.append(coverage)
        self.evidence_history.append(evidence_count)
        self.hypothesis_history.extend(hypotheses)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "query": self.query,
            "roles": self.roles,
            "created_at": self.created_at,
            "status": self.status,
            "timeline": self.timeline[-200:],  # prevent unbounded growth
            "messages": self.messages[-200:],
            "metrics": self.metrics,
            "coverage_history": self.coverage_history[-20:],
            "evidence_history": self.evidence_history[-20:],
            "hypothesis_history": self.hypothesis_history[-50:],
        }


class ConvergenceController:
    """Detect stagnation/cycles and suggest stop or re-plan."""

    def __init__(self, window: int = 3, min_delta: float = 0.01):
        self.window = max(window, 1)
        self.min_delta = min_delta
        self.coverage_history: List[float] = []
        self.evidence_history: List[int] = []
        self.hypothesis_history: List[str] = []

    def record_iteration(self, coverage: float, evidence_count: int, hypotheses: List[str]) -> None:
        self.coverage_history.append(coverage)
        self.evidence_history.append(evidence_count)
        self.hypothesis_history.extend([h or "" for h in hypotheses])
        # trim to window
        self.coverage_history = self.coverage_history[-(self.window + 1):]
        self.evidence_history = self.evidence_history[-(self.window + 1):]
        self.hypothesis_history = self.hypothesis_history[-(self.window * 3):]

    def check(self) -> Optional[str]:
        """Return reason for stagnation if detected, else None."""
        if len(self.coverage_history) < self.window + 1:
            return None
        recent = self.coverage_history[-self.window - 1:]
        if max(recent) - min(recent) < self.min_delta:
            return "coverage_stagnation"
        if len(self.evidence_history) >= self.window and all(e == 0 for e in self.evidence_history[-self.window:]):
            return "no_evidence_window"
        # simple cycle detection: repeated top hypothesis
        if len(self.hypothesis_history) >= self.window * 2:
            tail = self.hypothesis_history[-self.window:]
            if len(set(tail)) == 1 and tail[0]:
                return "repeated_hypothesis"
        return None
