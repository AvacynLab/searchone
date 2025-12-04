from pathlib import Path
from datetime import datetime
from app.core.config import DATA_DIR
import json
import subprocess
import shutil
import logging
from app.core.logging_config import configure_logging
from app.core.observability import compute_run_metrics
from typing import Dict, Any, List

configure_logging()
logger = logging.getLogger(__name__)

REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def assemble_markdown_report(job_name: str, job_state: dict) -> str:
    """Assemble a structured markdown report with sections: hypotheses, preuves, contrepoints, recommandations."""
    summary = build_structured_summary(job_state)
    lines: List[str] = []
    lines.append(f"# Rapport - {job_name}\n")
    lines.append(f"Genere le: {datetime.utcnow().isoformat()}Z\n")
    # Executive summary
    lines.append("## Executive Summary\n")
    lines.append(render_summary_block(summary))
    # Annexes techniques
    lines.append("\n## Annexes techniques\n")
    lines.append("### Hypotheses\n")
    for i, h in enumerate(summary.get("hypotheses", []), 1):
        lines.append(f"{i}. {h}")
    lines.append("\n### Preuves\n")
    for i, ev in enumerate(summary.get("evidence", []), 1):
        src = ev.get("source") or f"doc {ev.get('document_id')} chunk {ev.get('chunk_index')}"
        score = ev.get("score")
        lines.append(f"- ({i}) {src} | score={score} | {ev.get('text','')[:400]}")
    lines.append("\n### Contrepoints\n")
    cps = summary.get("counterpoints") or []
    if not cps:
        lines.append("- Aucun contrepoint explicite.")
    else:
        for cp in cps:
            lines.append(f"- {cp}")
    lines.append("\n### Recommandations\n")
    recs = summary.get("recommendations") or []
    if not recs:
        lines.append("- Aucune recommandation generee.")
    else:
        for r in recs:
            lines.append(f"- {r}")
    lines.append("\n### Notes\n")
    notes = summary.get("notes") or job_state.get("notes") or ""
    if isinstance(notes, list):
        lines.extend([f"- {n}" for n in notes])
    elif notes:
        lines.append(str(notes))
    return "\n".join(lines)


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
    return {
        "hypotheses": hypotheses,
        "evidence": evidence,
        "counterpoints": counterpoints,
        "recommendations": recommendations,
        "notes": notes,
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
    return "\n".join(lines)


def save_report(job_id: int, job_name: str, job_state: dict) -> Path:
    md = assemble_markdown_report(job_name, job_state)
    out = REPORTS_DIR / f"report_job_{job_id}.md"
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    # also save raw state
    with open(REPORTS_DIR / f"job_{job_id}_state.json", 'w', encoding='utf-8') as f:
        json.dump(job_state, f, ensure_ascii=False, indent=2)
    logger.info("Saved report markdown for job %s at %s", job_id, out)
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
