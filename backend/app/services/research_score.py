from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass
class ResearchScore:
    """Lightweight internal score tracking coherence, coverage, robustness, novelty."""

    coherence: float = 0.5
    coverage: float = 0.5
    robustness: float = 0.5
    novelty: float = 0.5

    def update(self, evidence_count: int, unique_sources: int, hypotheses: List[str]) -> None:
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
