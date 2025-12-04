from typing import Callable, Dict, Any, List, Optional, Tuple
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable, Any, List, Dict, Optional, Tuple
import math
import re


@dataclass
class PipelineNode:
    name: str
    fn: Callable
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    role: Optional[str] = None

    def to_step(self) -> Dict[str, Any]:
        return {"name": self.name, "fn": self.fn, "args": self.args, "kwargs": self.kwargs}


@dataclass
class Pipeline:
    nodes: List[PipelineNode]
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (from, to)

    def linearize(self) -> List[Dict[str, Any]]:
        """Return a simple topological ordering as steps (ignores cycles)."""
        indeg = defaultdict(int)
        graph = defaultdict(list)
        for src, dst in self.edges:
            graph[src].append(dst)
            indeg[dst] += 1
            if src not in indeg:
                indeg[src] = indeg.get(src, 0)
        name_to_node = {n.name: n for n in self.nodes}
        ready = [n for n in self.nodes if indeg.get(n.name, 0) == 0]
        steps: List[Dict[str, Any]] = []
        while ready:
            node = ready.pop(0)
            steps.append(node.to_step())
            for nxt in graph.get(node.name, []):
                indeg[nxt] -= 1
                if indeg[nxt] == 0 and nxt in name_to_node:
                    ready.append(name_to_node[nxt])
        # fallback append remaining nodes if graph not fully connected
        for n in self.nodes:
            if all(s.get("name") != n.name for s in steps):
                steps.append(n.to_step())
        return steps

    def inject_subpipeline(self, sub_nodes: List[PipelineNode], attach_after: str) -> None:
        """Insert sub-pipeline edges after a given node."""
        self.nodes.extend(sub_nodes)
        # Attach each sub-node sequentially
        for i in range(len(sub_nodes) - 1):
            self.edges.append((sub_nodes[i].name, sub_nodes[i + 1].name))
        if sub_nodes:
            self.edges.append((attach_after, sub_nodes[0].name))


class WorkflowEngine:
    """Very small pipeline executor: runs steps sequentially or in simple parallel groups."""

    def __init__(self, steps: List[Dict[str, Any]]):
        """
        steps: list of dicts with keys:
          - name: str
          - fn: callable async or sync
          - parallel_group: optional str to run in parallel with others of same group
          - args/kwargs: optional args for the callable
        """
        self.steps = steps

    async def run(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        groups: Dict[str, List[Dict[str, Any]]] = {}
        sequential = []
        for step in self.steps:
            group = step.get("parallel_group")
            if group:
                groups.setdefault(group, []).append(step)
            else:
                sequential.append(step)

        # execute sequential steps
        for step in sequential:
            res = await self._execute_step(step, context)
            results.append(res)

        # execute parallel groups
        for group, steps in groups.items():
            coros = [self._execute_step(s, context) for s in steps]
            group_results = await asyncio.gather(*coros, return_exceptions=False)
            results.extend(group_results)
        return results

    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        fn: Callable = step.get("fn")
        args = step.get("args") or []
        kwargs = step.get("kwargs") or {}
        label = step.get("name") or getattr(fn, "__name__", "step")
        if asyncio.iscoroutinefunction(fn):
            out = await fn(*args, **kwargs, context=context)
        else:
            out = fn(*args, **kwargs, context=context)
        return {"name": label, "result": out}


# ---------- Example pipeline definitions ----------

def _merge_result(context: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    context = context or {}
    if key:
        context[key] = value
    return context


def _keywords(text: str, top_k: int = 6) -> List[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+", text.lower())
    counts = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]


def clarify_question(context: Dict[str, Any], reason: str = "", **_: Any) -> Dict[str, Any]:
    """Refine the query by extracting keywords and a concise objective."""
    query = (context.get("query") or "").strip()
    refined = query.rstrip("?")
    if reason:
        refined = f"{refined} (raison: {reason})"
    keywords = _keywords(refined)
    objective = refined if refined else query
    context = _merge_result(context, "clarified_query", objective)
    context["keywords"] = keywords
    return context


def review_literature(context: Dict[str, Any], force_query: str = "", reason: str = "", **_: Any) -> Dict[str, Any]:
    """Simulate a literature review by generating structured sources from keywords."""
    query = force_query or context.get("clarified_query") or context.get("query") or ""
    keywords = context.get("keywords") or _keywords(query)
    sources = []
    for i, kw in enumerate(keywords[:5]):
        reliability = round(0.55 + (i * 0.07), 2)
        sources.append(
            {
                "title": f"Analyse de {kw}",
                "url": f"https://example.org/{kw}",
                "reliability": reliability,
                "domain": "example.org",
                "snippet": f"{kw} etat de l'art",
            }
        )
    reviewed = {
        "status": "done",
        "query": query,
        "reason": reason or "standard_review",
        "sources": sources,
        "coverage_score": round(sum(s["reliability"] for s in sources) / (len(sources) or 1), 2),
    }
    return _merge_result(context, "literature_review", reviewed)


def synthesize_claims(context: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Derive claims from reviewed literature and keywords."""
    sources = context.get("literature_review", {}).get("sources") or []
    keywords = context.get("keywords") or []
    claims = []
    for s in sources:
        kw = s.get("title", "").split()[-1]
        claims.append(f"{kw} montre un impact mesurable (score {s.get('reliability', 0)})")
    if not claims and keywords:
        claims = [f"{k} influence les resultats observes" for k in keywords[:3]]
    if not claims:
        claims = ["Aucune source exploitable - besoin de collecte supplementaire."]
    context["claims"] = claims
    return context


def generate_hypotheses(context: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Generate hypotheses from claims with simple transformations."""
    claims = context.get("claims") or []
    hypos = []
    for idx, c in enumerate(claims[:4]):
        hypos.append(f"H{idx+1}: Si {c.lower()}, alors performance amelioree de {5+idx*2}%")
    if not hypos:
        hypos.append("H1: Collecte complementaire requise pour formuler une hypothese.")
    return _merge_result(context, "hypotheses", hypos)


def plan_experiments(context: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Create concrete test plans for each hypothesis."""
    hypos = context.get("hypotheses") or []
    plans = []
    for h in hypos:
        plans.append(
            {
                "hypothesis": h,
                "metric": "accuracy_delta",
                "design": "A/B sur corpus de test",
                "sample_size": 50,
            }
        )
    return _merge_result(context, "experiment_plans", plans)


def debate_and_decide(context: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Compute a decision based on evidence coverage and hypotheses richness."""
    coverage = context.get("literature_review", {}).get("coverage_score", 0.0)
    hypo_count = len(context.get("hypotheses") or [])
    confidence = round(min(1.0, 0.4 + coverage * 0.4 + math.log(hypo_count + 1, 5)), 2)
    status = "adopt" if confidence >= 0.65 else "replan" if coverage < 0.4 else "investigate"
    decision = {
        "status": status,
        "confidence": confidence,
        "coverage": coverage,
        "hypotheses": hypo_count,
    }
    return _merge_result(context, "decision", decision)


def draft_report(context: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    """Assemble a short draft from claims/hypotheses/plan."""
    claims = context.get("claims") or []
    hypos = context.get("hypotheses") or []
    plans = context.get("experiment_plans") or []
    lines = [
        "# Synthese rapide",
        "## Claims",
        "\n".join(f"- {c}" for c in claims[:5]),
        "## Hypotheses",
        "\n".join(f"- {h}" for h in hypos[:4]),
        "## Plans d'experiences",
        "\n".join(f"- {p.get('hypothesis')}: {p.get('design')} ({p.get('metric')})" for p in plans[:3]),
    ]
    report = {"status": "draft", "notes": "\n".join(lines)}
    return _merge_result(context, "report", report)


def collect_complementary(context: Dict[str, Any], reason: str = "", **_: Any) -> Dict[str, Any]:
    """Collecte ciblée lors d'un replan."""
    query = context.get("query") or context.get("clarified_query") or ""
    note = f"collecte complementaire pour '{query}'"
    if reason:
        note = f"{note} (raison={reason})"
    harvested = [{"title": f"Source additionnelle sur {query[:30]}", "reliability": 0.6, "reason": reason or "replan"}]
    return _merge_result(context, "collection", {"status": "done", "note": note, "sources": harvested})


def build_collect_subpipeline(query: str, reason: str = "") -> Pipeline:
    """Small pipeline used when the coordinateur insère une collecte additionnelle."""
    nodes = [
        PipelineNode(name="clarify_question", fn=clarify_question, kwargs={"reason": reason}),
        PipelineNode(name="collect_complementary", fn=collect_complementary, kwargs={"reason": reason}),
        PipelineNode(name="synthesize_claims", fn=synthesize_claims),
    ]
    edges = [("clarify_question", "collect_complementary"), ("collect_complementary", "synthesize_claims")]
    return Pipeline(nodes=nodes, edges=edges)


def build_conflict_subpipeline(query: str, reason: str = "") -> Pipeline:
    nodes = [
        PipelineNode(name="clarify_question", fn=clarify_question, kwargs={"reason": reason}),
        PipelineNode(name="collect_complementary", fn=collect_complementary, kwargs={"reason": reason}),
        PipelineNode(name="debate_and_decide", fn=debate_and_decide),
    ]
    edges = [
        ("clarify_question", "collect_complementary"),
        ("collect_complementary", "debate_and_decide"),
    ]
    return Pipeline(nodes=nodes, edges=edges)


def build_seek_sources_subpipeline(query: str, reason: str = "") -> Pipeline:
    nodes = [
        PipelineNode(name="clarify_question", fn=clarify_question, kwargs={"reason": reason}),
        PipelineNode(name="collect_complementary", fn=collect_complementary, kwargs={"reason": reason}),
        PipelineNode(name="synthesize_claims", fn=synthesize_claims),
    ]
    edges = [("clarify_question", "collect_complementary"), ("collect_complementary", "synthesize_claims")]
    return Pipeline(nodes=nodes, edges=edges)


async def execute_actions_from_bus(bus: Any, state: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """
    Drain bus actions (ex: replan_collect) and run lightweight sub-pipelines.
    Returns the list of executed action results.
    """
    actions = bus.drain("actions") if bus else []
    executed: List[Dict[str, Any]] = []
    for action in actions:
        a_type = action.get("type")
        if a_type == "replan_collect":
            sub_query = action.get("query") or query
            reason = action.get("reason") or ""
            sub_ctx: Dict[str, Any] = {"query": sub_query, "reason": reason}
            sub_pipe = build_collect_subpipeline(sub_query, reason=reason)
            engine = WorkflowEngine(sub_pipe.linearize())
            results = await engine.run(context=sub_ctx)
            executed.append({"action": action, "context": sub_ctx, "results": results})
        elif a_type == "focus_conflicts":
            sub_query = action.get("query") or query
            reason = action.get("reason") or ""
            sub_ctx = {"query": sub_query, "reason": reason}
            sub_pipe = build_conflict_subpipeline(sub_query, reason=reason)
            engine = WorkflowEngine(sub_pipe.linearize())
            results = await engine.run(context=sub_ctx)
            executed.append({"action": action, "context": sub_ctx, "results": results})
        elif a_type == "seek_additional_sources":
            sub_query = action.get("query") or query
            reason = action.get("reason") or ""
            sub_ctx = {"query": sub_query, "reason": reason}
            sub_pipe = build_seek_sources_subpipeline(sub_query, reason=reason)
            engine = WorkflowEngine(sub_pipe.linearize())
            results = await engine.run(context=sub_ctx)
            executed.append({"action": action, "context": sub_ctx, "results": results})
        elif a_type == "downgrade_low_quality_sources":
            sub_ctx = {"query": query, "note": "sources downgraded", "marked": action.get("marked")}
            executed.append({"action": action, "context": sub_ctx, "results": []})
    if executed:
        state.setdefault("pipeline_events", []).extend(executed)
    return executed


def build_full_research_pipeline() -> Pipeline:
    """Provide an example pipeline from question to draft report."""
    nodes = [
        PipelineNode(name="clarify_question", fn=clarify_question),
        PipelineNode(name="literature_review", fn=review_literature),
        PipelineNode(name="synthesize_claims", fn=synthesize_claims),
        PipelineNode(name="generate_hypotheses", fn=generate_hypotheses),
        PipelineNode(name="plan_experiments", fn=plan_experiments),
        PipelineNode(name="debate_and_decide", fn=debate_and_decide),
        PipelineNode(name="draft_report", fn=draft_report),
    ]
    edges = [
        ("clarify_question", "literature_review"),
        ("literature_review", "synthesize_claims"),
        ("synthesize_claims", "generate_hypotheses"),
        ("generate_hypotheses", "plan_experiments"),
        ("plan_experiments", "debate_and_decide"),
        ("debate_and_decide", "draft_report"),
    ]
    return Pipeline(nodes=nodes, edges=edges)
