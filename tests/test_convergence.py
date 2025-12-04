import pytest
from runtime import ConvergenceController

def test_convergence_detects_stagnation_by_coverage():
    c = ConvergenceController(window=3, min_delta=0.01)
    for _ in range(4):
        c.record_iteration(coverage=0.1, evidence_count=1, hypotheses=["h1"])
    assert c.check() == "coverage_stagnation"


def test_convergence_detects_repeated_hypothesis():
    c = ConvergenceController(window=2, min_delta=0.0)
    c.record_iteration(coverage=0.5, evidence_count=1, hypotheses=["h1"])
    c.record_iteration(coverage=0.5, evidence_count=1, hypotheses=["h1"])
    assert c.check() == "repeated_hypothesis"


def test_convergence_detects_no_evidence_window():
    c = ConvergenceController(window=2, min_delta=0.0)
    c.record_iteration(coverage=0.5, evidence_count=0, hypotheses=["h1"])
    c.record_iteration(coverage=0.6, evidence_count=0, hypotheses=["h2"])
    assert c.check() == "no_evidence_window"
