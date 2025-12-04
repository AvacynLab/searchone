from pathlib import Path
from datetime import datetime
from app.core.config import DATA_DIR
import json
import subprocess
import shutil
import logging
from app.core.logging_config import configure_logging
from app.core.observability import compute_run_metrics
from typing import Dict, Any, List, Optional

from app.workflows.writing_pipeline import build_scientific_article

configure_logging()
logger = logging.getLogger(__name__)

REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_article_report(job_state: Dict[str, Any], job_name: str) -> Dict[str, Any]:
    summary = build_structured_summary(job_state)
    writing = build_scientific_article(job_state, format="markdown", summary=summary)
    research_score = job_state.get("research_score") or {}
    run_metrics = job_state.get("run_metrics") or {}
    return {
        "title": job_name,
        "summary_text": render_summary_block(summary),
        "summary": summary,
        "writing": writing,
        "figures": writing.get("figures") or summary.get("figures", []),
        "knowledge_topology": summary.get("knowledge_topology") or {},
        "research_score": research_score,
        "run_metrics": run_metrics,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def export_markdown(report_struct: Dict[str, Any]) -> str:
    writing = report_struct.get("writing") or {}
    lines: List[str] = []
    title = report_struct.get("title") or "Rapport scientifique"
    lines.append(f"# {title}\n")
    lines.append(f"Genere le: {report_struct.get('generated_at')}\n")
    summary_text = report_struct.get("summary_text")
    if summary_text:
        lines.append("## Executive Summary\n")
        lines.append(summary_text)
    lines.append("\n## Plan IMRaD\n")
    outline = writing.get("outline", {})
    for section in outline.get("sections", []):
        lines.append(f"- **{section.get('title')}**: {', '.join(section.get('bullets') or [])}")
    lines.append("\n## Sections detaillees\n")
    for section in writing.get("sections", []):
        lines.append(f"### {section.get('title')}\n")
        lines.append(section.get("body") or "")
        if section.get("citations"):
            lines.append("Citations: " + "; ".join(section["citations"]))
        critique = section.get("critique") or {}
        issues = critique.get("issues") or []
        if issues:
            lines.append("Critique: " + "; ".join(issues))
    if writing.get("critic"):
        lines.append("\n## Relecture Globale\n")
        crit = writing["critic"]
        lines.append(f"- Score: {crit.get('score')}")
        if crit.get("issues"):
            lines.append("- Issues: " + "; ".join(crit["issues"]))
    figures = report_struct.get("figures") or []
    if figures:
        lines.append("\n## Figures clés\n")
        for fig in figures:
            lines.append(f"- {fig.get('title') or 'figure'}")
            if fig.get("description"):
                lines.append(f"  - {fig['description']}")
            if fig.get("path"):
                lines.append(f"  - PNG: {fig['path']}")
            for fmt, path in (fig.get("vectors") or {}).items():
                lines.append(f"  - {fmt.upper()}: {path}")
    lines.append("\n## Résultats et scores\n")
    rs = report_struct.get("research_score") or {}
    metrics = report_struct.get("run_metrics") or {}
    lines.append("| Indicateur | Valeur |")
    lines.append("| --- | --- |")
    for label in ("coherence", "coverage", "robustness", "novelty"):
        if label in rs:
            lines.append(f"| {label} | {rs[label]:.3f} |")
    for label in ("coverage_score", "evidence_count", "iterations"):
        if label in metrics:
            lines.append(f"| {label} | {metrics[label]} |")
    lines.append("\n## Bibliographie\n")
    for ref in writing.get("bibliography") or []:
        lines.append(f"- {ref}")
    return "\n".join(lines)


def export_latex(report_struct: Dict[str, Any]) -> str:
    writing = report_struct.get("writing") or {}
    figures = report_struct.get("figures") or []
    bibliography = writing.get("bibliography") or []
    lines = [
        "\\documentclass{article}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[french]{babel}",
        "\\usepackage{graphicx}",
        "\\title{" + (report_struct.get("title") or "Rapport scientifique") + "}",
        "\\begin{document}",
        "\\maketitle",
    ]
    for section in writing.get("sections", []):
        lines.append(f"\\section{{{section.get('title')}}}")
        lines.append(section.get("body") or "")
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
    metrics = report_struct.get("research_score") or {}
    if metrics:
        lines.append("\\section*{Scores de recherche}")
        lines.append("\\begin{tabular}{ll}")
        for label in ("coherence", "coverage", "robustness", "novelty"):
            if label in metrics:
                lines.append(f"{label} & {metrics[label]:.3f} \\\\")
        lines.append("\\end{tabular}")
    if bibliography:
        lines.append("\\section*{Bibliographie}")
        for ref in bibliography:
            lines.append(ref)
    lines.append("\\end{document}")
    rendered = "\n".join(lines)
    return rendered


def assemble_markdown_report(job_name: str, job_state: dict) -> str:
    report_struct = build_article_report(job_state, job_name)
    return export_markdown(report_struct)


def _extract_figures_from_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    figures = []
    seen = set()
    for ev in evidence or []:
        meta = ev.get("meta") or {}
        if meta.get("source_type") != "plot":
            continue
        figure_meta = meta.get("figure") or {}
        path = meta.get("path") or figure_meta.get("path")
        title = figure_meta.get("title") or ev.get("text") or "figure"
        key = path or title
        if not key or key in seen:
            continue
        seen.add(key)
        figures.append(
            {
                "title": title,
                "path": path,
                "description": figure_meta.get("description"),
                "vectors": meta.get("vectors") or figure_meta.get("vector_paths") or {},
                "metadata": figure_meta,
            }
        )
    return figures


def _extract_stats_from_evidence(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    for ev in evidence or []:
        meta = ev.get("meta") or {}
        if meta.get("source_type") == "knowledge_graph":
            stats = meta.get("stats") or {}
            exports = {}
            for key in ("png_path", "dot_path"):
                if meta.get(key):
                    exports[key] = meta.get(key)
            if exports:
                stats["exports"] = exports
            return stats
    return {}


def _extract_knowledge_topology(job_state: Dict[str, Any]) -> Dict[str, Any]:
    stats = job_state.get("knowledge_graph_stats") or {}
    exports = job_state.get("knowledge_graph_exports") or {}
    if not stats:
        for entry in job_state.get("timeline") or []:
            msg_stats = _extract_stats_from_evidence(entry.get("messages") or [])
            if msg_stats:
                stats = msg_stats
                break
            for msg in entry.get("messages") or []:
                msg_stats = _extract_stats_from_evidence(msg.get("evidence") or [])
                if msg_stats:
                    stats = msg_stats
                    break
            if stats:
                break
    if stats and exports:
        stats.setdefault("exports", exports)
    return stats


def build_structured_summary(job_state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a structured summary from the job_state."""
    hypotheses: List[str] = []
    evidence: List[Dict[str, Any]] = []
    counterpoints: List[str] = []
    recommendations: List[str] = []
    notes: List[str] = []

    # collect hypotheses from agents
    agents_state = job_state.get("agents") or {}
    for ag in agents_state.values():
        for h in ag.get("hypotheses", []):
            if h not in hypotheses:
                hypotheses.append(h)
        if ag.get("notes"):
            notes.append(ag["notes"])
    # fallback for simple agent loop
    if not hypotheses and "hypotheses" in job_state:
        for h in job_state.get("hypotheses") or []:
            if h not in hypotheses:
                hypotheses.append(h)
    # evidence from timeline / council messages
    timeline = job_state.get("timeline") or []
    for entry in timeline:
        for msg in entry.get("messages") or []:
            for ev in msg.get("evidence") or []:
                if ev not in evidence:
                    evidence.append(ev)
        # derive counterpoints from votes disagreements
        for vote_block in entry.get("votes") or []:
            hyp = vote_block.get("hypothesis")
            disagree_agents = [v.get("agent") for v in (vote_block.get("votes") or []) if (v.get("vote") or "").startswith("disagree")]
            if disagree_agents and hyp:
                counterpoints.append(f"Desaccord sur '{hyp}' par {', '.join(disagree_agents)}")
        if entry.get("summary"):
            recommendations.append(entry["summary"])
    # fallback evidence
    if not evidence and job_state.get("evidence"):
        evidence = job_state["evidence"]
    # trim recommendations to keep top few
    recommendations = recommendations[-3:]
    figures = _extract_figures_from_evidence(evidence)
    knowledge_topology = _extract_knowledge_topology(job_state)
    return {
        "hypotheses": hypotheses,
        "evidence": evidence,
        "counterpoints": counterpoints,
        "recommendations": recommendations,
        "notes": notes,
        "figures": figures,
        "knowledge_topology": knowledge_topology,
    }


def build_diagnostic(state: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a lightweight internal diagnostic report for end-of-run."""
    metrics = compute_run_metrics(state)
    timeline = state.get("timeline") or []
    last = timeline[-1] if timeline else {}
    return {
        "status": state.get("status"),
        "last_error": state.get("last_error"),
        "iterations": metrics.get("iterations"),
        "avg_evidence_per_iter": metrics.get("avg_evidence_per_iter"),
        "token_spent": metrics.get("token_spent"),
        "usage": state.get("usage"),
        "research_score": state.get("research_score"),
        "last_summary": last.get("summary"),
        "last_votes": last.get("votes"),
    }


def render_summary_block(summary: Dict[str, Any]) -> str:
    """Render a compact executive summary block."""
    lines = []
    if summary.get("hypotheses"):
        lines.append("Hypotheses principales:")
        for h in summary["hypotheses"][:3]:
            lines.append(f"- {h}")
    if summary.get("evidence"):
        lines.append("\nPoints de preuve saillants:")
        for ev in summary["evidence"][:3]:
            src = ev.get("source") or ev.get("meta", {}).get("source") or f"doc {ev.get('document_id')}"
            lines.append(f"- {src} (score={ev.get('score')})")
    if summary.get("recommendations"):
        lines.append("\nRecommandations:")
        for r in summary["recommendations"][:3]:
            lines.append(f"- {r}")
    if summary.get("counterpoints"):
        lines.append("\nContrepoints:")
        for c in summary["counterpoints"][:2]:
            lines.append(f"- {c}")
    if summary.get("figures"):
        lines.append("\nFigures cles:")
        for fig in summary["figures"][:2]:
            label = fig.get("title") or "figure"
            path = fig.get("path")
            if path:
                lines.append(f"- {label}: {path}")
            else:
                lines.append(f"- {label}")
    if summary.get("knowledge_topology"):
        topology = summary["knowledge_topology"]
        lines.append("\nTopologie de la connaissance utilisée:")
        lines.append(f"- Degré moyen: {topology.get('avg_degree')}")
        lines.append(f"- Composantes: {topology.get('component_count')}")
        hubs = topology.get("hubs") or []
        if hubs:
            hub_labels = ", ".join([f"{h.get('node')}({h.get('degree')})" for h in hubs])
            lines.append(f"- Hubs: {hub_labels}")
    return "\n".join(lines)


def save_report(job_id: int, job_name: str, job_state: dict) -> Path:
    report_struct = build_article_report(job_state, job_name)
    md = export_markdown(report_struct)
    out = REPORTS_DIR / f"report_job_{job_id}.md"
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    # also save raw state
    with open(REPORTS_DIR / f"job_{job_id}_state.json", 'w', encoding='utf-8') as f:
        json.dump(job_state, f, ensure_ascii=False, indent=2)
    logger.info("Saved report markdown for job %s at %s", job_id, out)
    tex_path = REPORTS_DIR / f"report_job_{job_id}.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(export_latex(report_struct))
    logger.info("Saved LaTeX report for job %s at %s", job_id, tex_path)
    # try to export PDF as convenience
    pdf_path = REPORTS_DIR / f"report_job_{job_id}.pdf"
    ok = export_report_pdf(out, pdf_path)
    if ok:
        logger.info("Exported PDF report for job %s at %s", job_id, pdf_path)
    else:
        logger.debug("Pandoc not available or export failed for job %s", job_id)
    return out


def export_report_pdf(md_path: Path, out_pdf: Path) -> bool:
    """Try to export Markdown to PDF using pandoc if available.

    Returns True on success, False otherwise.
    """
    pandoc = shutil.which('pandoc')
    if not pandoc:
        return False
    try:
        subprocess.run([pandoc, str(md_path), '-o', str(out_pdf)], check=True)
        return True
    except Exception:
        return False
