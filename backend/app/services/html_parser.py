from __future__ import annotations

from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

try:
    from markdownify import markdownify as _markdownify
    _HAS_MARKDOWNIFY = True
except ImportError:  # pragma: no cover - optional dependency
    _markdownify = None  # type: ignore[assignment]
    _HAS_MARKDOWNIFY = False


def clean_html(html: str) -> str:
    """Strip obvious chrome/boilerplate and disable scripts/styles for later processing."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    for tag_name in ("header", "footer", "nav", "aside", "form", "button"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for tag in soup.find_all(True):
        if tag.has_attr("class"):
            classes = [c.lower() for c in tag.get("class") if isinstance(c, str)]
            if any("ad" in cls or "promo" in cls for cls in classes):
                tag.decompose()
                continue
        if tag.has_attr("style"):
            del tag["style"]
    return str(soup)


def html_to_markdown(html: str) -> str:
    """Fallback to markdownify if available, else collapse text while keeping paragraph breaks."""
    cleaned = clean_html(html)
    if _HAS_MARKDOWNIFY and _markdownify:
        try:
            return _markdownify(cleaned, heading_style="ATX")
        except Exception:
            pass
    soup = BeautifulSoup(cleaned, "html.parser")
    return "\n\n".join(
        part.strip() for part in soup.get_text(separator="\n").splitlines() if part.strip()
    )


def extract_main_content(html: str) -> Dict[str, Any]:
    """Return document title, structured sections, links, and cleaned HTML snippet."""
    cleaned_html = clean_html(html)
    soup = BeautifulSoup(cleaned_html, "html.parser")
    doc_title = (soup.title.string or "").strip() if soup.title else ""
    main = soup.find("main") or soup.find("article") or soup.body or soup
    sections: List[Dict[str, str]] = []
    current = {"title": doc_title or "body", "content": []}
    for element in main.find_all(["h1", "h2", "h3", "p"], recursive=True):
        if element.name in {"h1", "h2", "h3"}:
            if any(part.strip() for part in current["content"]):
                sections.append({"title": current["title"], "content": "\n\n".join(current["content"]).strip()})
            heading = element.get_text(separator=" ", strip=True)
            current = {"title": heading or current["title"], "content": []}
            continue
        text = element.get_text(separator=" ", strip=True)
        if text:
            current["content"].append(text)
    if any(part.strip() for part in current["content"]):
        sections.append({"title": current["title"], "content": "\n\n".join(current["content"]).strip()})
    if not sections:
        fallback = soup.get_text(separator="\n", strip=True)
        if fallback:
            sections.append({"title": doc_title or "body", "content": fallback})
    links = [
        {"href": a["href"], "text": a.get_text(strip=True)}
        for a in soup.find_all("a", href=True)
        if a["href"].strip()
    ]
    return {
        "title": doc_title or "",
        "sections": sections,
        "links": links,
        "raw_html": cleaned_html,
    }
