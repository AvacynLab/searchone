from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional


@dataclass
class ResearchScore:
    """Lightweight internal score tracking coherence, coverage, robustness, novelty."""

    coherence: float = 0.5
    coverage: float = 0.5
    robustness: float = 0.5
    novelty: float = 0.5
    graph_stats: Dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        evidence_count: int,
        unique_sources: int,
        hypotheses: List[str],
        graph_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        # simple heuristics: more evidence/sources -> better coverage/robustness; new hypotheses -> novelty
        self.coverage = min(1.0, self.coverage * 0.7 + min(unique_sources, 10) / 10.0 * 0.3)
        self.robustness = min(1.0, self.robustness * 0.7 + min(evidence_count, 12) / 12.0 * 0.3)
        if hypotheses:
            unique_h = len(set(hypotheses))
            self.novelty = min(1.0, self.novelty * 0.6 + min(unique_h, 5) / 5.0 * 0.4)
        # coherence heuristic: penalize when no evidence
        if evidence_count == 0:
            self.coherence = max(0.0, self.coherence * 0.8)
        else:
            self.coherence = min(1.0, self.coherence * 0.7 + 0.3)
        if graph_stats:
            self.graph_stats = graph_stats
            avg_deg = float(graph_stats.get("avg_degree") or 0.0)
            components = int(graph_stats.get("component_count") or 0)
            self.robustness = min(
                1.0,
                self.robustness * 0.8 + min(avg_deg, 6) / 6.0 * 0.2,
            )
            if components > 1:
                self.coherence = max(0.0, self.coherence - 0.05)
            else:
                self.coherence = min(1.0, self.coherence + 0.05)
            hub_bonus = len(graph_stats.get("hubs") or []) * 0.02
            self.novelty = min(1.0, self.novelty + hub_bonus)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
