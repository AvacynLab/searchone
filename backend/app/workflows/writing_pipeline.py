from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.references import ReferenceManager
from app.services.writing import (
    OutlineGenerator,
    SectionWriter,
    StyleCritic,
    FinalComposer,
    GlobalCritic,
)


def _collect_hypotheses(state: Dict[str, Any]) -> List[str]:
    hypotheses: List[str] = []
    for agent in (state.get("agents") or {}).values():
        for h in agent.get("hypotheses", []):
            if h and h not in hypotheses:
                hypotheses.append(h)
    if not hypotheses and state.get("hypotheses"):
        for h in state.get("hypotheses") or []:
            if h and h not in hypotheses:
                hypotheses.append(h)
    return hypotheses


def _collect_evidence_texts(state: Dict[str, Any], limit: int = 6) -> List[str]:
    texts: List[str] = []
    for entry in state.get("timeline") or []:
        for msg in entry.get("messages") or []:
            for ev in msg.get("evidence") or []:
                text = ev.get("text")
                if text and len(text) > 10:
                    texts.append(text.strip())
                    if len(texts) >= limit:
                        return texts
    return texts


def _collect_figures(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    figures = []
    seen = set()
    for entry in state.get("timeline") or []:
        for msg in entry.get("messages") or []:
            for ev in msg.get("evidence") or []:
                meta = ev.get("meta") or {}
                if meta.get("source_type") != "plot":
                    continue
                figure_meta = meta.get("figure") or {}
                path = figure_meta.get("path") or meta.get("path")
                title = figure_meta.get("title") or ev.get("text") or "figure"
                key = path or title
                if not key or key in seen:
                    continue
                seen.add(key)
                vectors = figure_meta.get("vector_paths") or {}
                figures.append(
                    {
                        "title": title,
                        "description": figure_meta.get("description"),
                        "caption": figure_meta.get("description"),
                        "variables": figure_meta.get("variables"),
                        "plot_type": figure_meta.get("plot_type"),
                        "generated_at": figure_meta.get("generated_at"),
                        "path": path,
                        "svg_path": figure_meta.get("svg_path"),
                        "source": figure_meta.get("metadata", {}).get("source") or meta.get("source"),
                        "vectors": vectors,
                        "metadata": figure_meta.get("metadata"),
                    }
                )
    return figures


def _collect_references(state: Dict[str, Any], ref_manager: ReferenceManager) -> None:
    for entry in state.get("timeline") or []:
        for msg in entry.get("messages") or []:
            for ev in msg.get("evidence") or []:
                meta = ev.get("meta") or {}
                title = meta.get("title") or meta.get("source") or meta.get("domain") or ""
                if not title:
                    continue
                ref_manager.add(
                    title=title,
                    author=meta.get("domain") or "",
                    year=meta.get("published_at", "").split("T", 1)[0] if meta.get("published_at") else "",
                    doi=meta.get("doi") or "",
                    url=meta.get("source") or "",
                )


def _markdown_to_latex(sections: List[Dict[str, Any]], title: str, bibliography: List[str], figures: List[Dict[str, Any]]) -> str:
    lines = [
        "\\documentclass{article}",
        "\\usepackage{graphicx}",
        "\\usepackage[french]{babel}",
        "\\usepackage[utf8]{inputenc}",
        "\\title{" + title + "}",
        "\\begin{document}",
        "\\maketitle",
    ]
    for section in sections:
        lines.append(f"\\section{{{section['title']}}}")
        body = section.get("body", "").replace("\n", " ")
        lines.append(body)
        if section.get("citations"):
            lines.append(" ".join(section["citations"]))
    if figures:
        lines.append("\\section*{Figures}")
        for fig in figures:
            if fig.get("path"):
                lines.append("\\begin{figure}[h]")
                lines.append("\\centering")
                lines.append(f"\\includegraphics[width=\\linewidth]{{{fig['path']}}}")
                if fig.get("title"):
                    lines.append(f"\\caption{{{fig['title']}}}")
                lines.append("\\end{figure}")
    if bibliography:
        lines.append("\\section*{Bibliographie}")
        for ref in bibliography:
            lines.append(ref)
    lines.append("\\end{document}")
    return "\n".join(lines)


def build_scientific_article(
    job_state: Dict[str, Any],
    format: str = "markdown",
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate an IMRaD-style article draft plus metadata from the completed job state."""
    summaries = summary or {}
    claims = summaries.get("hypotheses") or _collect_hypotheses(job_state)
    evidence_texts = summaries.get("evidence_texts") or _collect_evidence_texts(job_state)
    figures = summaries.get("figures") or _collect_figures(job_state)
    knowledge_graph_exports = job_state.get("knowledge_graph_exports") or []
    existing_paths = {fig.get("path") for fig in figures if fig.get("path")}
    for export in knowledge_graph_exports:
        export_path = export.get("path") or export.get("dot_path")
        if not export_path or export_path in existing_paths:
            continue
        existing_paths.add(export_path)
        figures.append(
            {
                "title": export.get("description") or f"Graphe de connaissances ({export.get('format')})",
                "description": export.get("description") or "Graph-export généré par knowledge_graph_tool.",
                "path": export_path,
                "metadata": export,
            }
        )
    topic = job_state.get("query") or "Recherche scientifique"

    outline_generator = OutlineGenerator()
    outline = outline_generator.generate(topic, claims, evidence_texts)
    plan = outline_generator.select_plan(outline)

    section_writer = SectionWriter()
    style_critic = StyleCritic()
    global_critic = GlobalCritic()
    final_composer = FinalComposer()
    ref_manager = ReferenceManager()
    _collect_references(job_state, ref_manager)

    all_refs = ref_manager.list()
    sections_output = []
    for idx, sec in enumerate(plan.get("sections", [])):
        bullets = sec.get("bullets") or []
        claims_snippet = claims[idx : idx + 2]
        evidence_snippet = evidence_texts[idx : idx + 2]
        body = section_writer.draft_section(
            title=sec.get("title") or f"Section {idx + 1}",
            bullets=bullets,
            claims=claims_snippet,
            evidence_snippets=evidence_snippet,
            citations=[ref_manager.cite_inline(i) for i in range(min(2, len(all_refs)))]
        )
        critique = style_critic.critique(body)
        sections_output.append(
            {
                "title": sec.get("title") or f"Section {idx + 1}",
                "body": body,
                "critique": critique,
                "citations": [ref_manager.cite_inline(i) for i in range(min(1, len(all_refs)))]
            }
        )

    draft_sections = [sec["body"] for sec in sections_output]
    final_article = final_composer.compose(
        title=topic,
        sections=draft_sections,
        bibliography=ref_manager.bibliography(),
    )

    latex_text = _markdown_to_latex(
        sections_output,
        title=topic,
        bibliography=ref_manager.bibliography(),
        figures=figures,
    )

    critic = global_critic.review(
        final_article.get("article") or "",
        claims,
        job_state.get("run_metrics") or {},
    )

    kg_stats = job_state.get("knowledge_graph_stats") or {}
    if kg_stats:
        nodes = kg_stats.get("node_count", 0)
        edges = kg_stats.get("edge_count", 0)
        components = kg_stats.get("component_count", 0)
        hubs = kg_stats.get("hubs") or []
        hubs_text = ", ".join(f"{hub.get('node')} ({hub.get('degree')})" for hub in hubs[:3] if hub.get("node"))
        graph_paragraph = f"Structure du graphe de connaissances utilisé: {nodes} nœuds, {edges} liens, {components} composantes."
        if hubs_text:
            graph_paragraph += f" Hubs dominants: {hubs_text}."
        target_section = None
        for section in sections_output:
            title = section.get("title", "").lower()
            if "discussion" in title or "méthod" in title or "method" in title:
                target_section = section
                break
        if target_section:
            body_text = target_section.get("body", "") or ""
            separator = "\n" if body_text else ""
            target_section["body"] = f"{body_text}{separator}\n{graph_paragraph}".strip()
        else:
            sections_output.append(
                {
                    "title": "Topologie du graphe de connaissances",
                    "body": graph_paragraph,
                    "citations": [],
                }
            )

    return {
        "title": topic,
        "outline": plan,
        "sections": sections_output,
        "article": final_article.get("article"),
        "summary": final_article.get("summary"),
        "bibliography": final_article.get("bibliography"),
        "critic": critic,
        "format": format,
        "latex": latex_text,
        "figures": figures,
        "research_score": job_state.get("research_score") or {},
        "metrics": job_state.get("run_metrics") or {},
    }
