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
                title = meta.get("figure", {}).get("title") or ev.get("text") or "figure"
                path = meta.get("path") or meta.get("figure", {}).get("path")
                key = path or title
                if not key or key in seen:
                    continue
                seen.add(key)
                figures.append(
                    {
                        "title": title,
                        "description": meta.get("figure", {}).get("description"),
                        "path": path,
                        "vectors": meta.get("vectors") or {},
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
