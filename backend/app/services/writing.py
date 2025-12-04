"""
Lightweight writing helpers usable sans LLM pour generer plans et brouillons courts.
"""
from typing import List, Dict, Any


class OutlineGenerator:
    """Generate simple IMRaD outlines from claims/evidence."""

    def generate(self, topic: str, claims: List[str], evidence: List[str] = None) -> Dict[str, Any]:
        evidence = evidence or []
        base_sections = [
            {"title": "Introduction", "bullets": [topic]},
            {"title": "Methodes", "bullets": ["Sources clés", "Approche d'analyse"]},
            {"title": "Resultats", "bullets": claims[:3] or ["Resultats attendus"]},
            {"title": "Discussion", "bullets": ["Limites", "Pistes futures"]},
        ]
        alt_sections = [
            {"title": "Context", "bullets": [topic, "Etat de l'art"]},
            {"title": "Approche", "bullets": ["Recherche documentaire", "Synthèse analytique"]},
            {"title": "Insights", "bullets": claims[:2] or ["Insight clé"]},
            {"title": "Prochaines etapes", "bullets": ["Tests / experiences", "Questions ouvertes"]},
        ]
        plans = [
            {"label": "imrad", "sections": base_sections},
            {"label": "insights", "sections": alt_sections},
        ]
        return {"topic": topic, "plans": plans, "evidence_refs": evidence[:5]}

    def select_plan(self, outline: Dict[str, Any], strategy: str = "first") -> Dict[str, Any]:
        plans = outline.get("plans") or []
        if not plans:
            return {"sections": []}
        if strategy == "longest":
            return max(plans, key=lambda p: len(p.get("sections", [])))
        return plans[0]


class SectionWriter:
    """Produce a short section draft from inputs."""

    def draft_section(
        self,
        title: str,
        bullets: List[str],
        claims: List[str] = None,
        evidence_snippets: List[str] = None,
        citations: List[str] = None,
    ) -> str:
        claims = claims or []
        evidence_snippets = evidence_snippets or []
        citations = citations or []
        body_parts = []
        if bullets:
            body_parts.append(" ".join(bullets))
        if claims:
            body_parts.append("Claims: " + "; ".join(claims))
        if evidence_snippets:
            body_parts.append("Evidence: " + "; ".join(evidence_snippets))
        cites = ""
        if citations:
            cites = " Citations: " + "; ".join(citations)
        return f"{title}: {' '.join(body_parts)}.{cites}"


class StyleCritic:
    """Lightweight style critic heuristic."""

    def critique(self, text: str) -> Dict[str, Any]:
        issues = []
        if len(text.split()) > 150:
            issues.append("Section trop longue, considerer un resume.")
        if not text:
            issues.append("Section vide.")
        score = max(0, 10 - len(issues) * 2)
        return {"score": score, "issues": issues, "note": "Heuristique locale (sans LLM)."}


class FinalComposer:
    """Compose a simple final article and summary from drafted sections."""

    def compose(self, title: str, sections: List[str], bibliography: List[str]) -> Dict[str, Any]:
        article = f"# {title}\n\n" + "\n\n".join(sections)
        biblio_block = ""
        if bibliography:
            biblio_block = "\n\n## Bibliographie\n" + "\n".join(f"- {b}" for b in bibliography)
        article += biblio_block
        summary = " ".join(s[:200] for s in sections[:2]) or "Resume indisponible"
        return {"article": article, "summary": summary, "bibliography": bibliography}


class GlobalCritic:
    """Review a full article for basic coherence and structure."""

    def review(self, article: str, claims: List[str], decision: Dict[str, Any]) -> Dict[str, Any]:
        issues = []
        if "# Synthese rapide" not in article and "# " not in article:
            issues.append("Titre ou section manquante")
        if len(claims) < 1:
            issues.append("Aucun claim consolide")
        if "## Plans d'experiences" not in article:
            issues.append("Plan d'experiences absent")
        conf = decision.get("confidence") if isinstance(decision, dict) else None
        score = max(0, 10 - len(issues) * 2)
        return {"score": score, "issues": issues, "decision_confidence": conf}
