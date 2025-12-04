"""
Lightweight in-memory reference manager with deduplication.
"""
from typing import List, Dict, Any, Optional
import hashlib


class ReferenceManager:
    def __init__(self):
        self.refs: List[Dict[str, Any]] = []
        self._hashes = set()

    def _fingerprint(self, title: str, doi: str = "", url: str = "") -> str:
        raw = (title or "") + (doi or "") + (url or "")
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def add(self, title: str, author: str = "", year: str = "", doi: str = "", url: str = "") -> Dict[str, Any]:
        fp = self._fingerprint(title, doi, url)
        if fp in self._hashes:
            return self.find(title=title, doi=doi, url=url) or {}
        ref = {"title": title, "author": author, "year": year, "doi": doi, "url": url}
        self.refs.append(ref)
        self._hashes.add(fp)
        return ref

    def list(self) -> List[Dict[str, Any]]:
        return list(self.refs)

    def find(self, title: Optional[str] = None, doi: Optional[str] = None, url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        for r in self.refs:
            if title and r.get("title") == title:
                return r
            if doi and r.get("doi") == doi:
                return r
            if url and r.get("url") == url:
                return r
        return None

    def cite_inline(self, idx: int) -> str:
        if 0 <= idx < len(self.refs):
            r = self.refs[idx]
            parts = [p for p in [r.get("author"), r.get("year")] if p]
            return "(" + ", ".join(parts) + ")" if parts else "(ref)"
        return "(ref)"

    def bibliography(self) -> List[str]:
        out = []
        for r in self.refs:
            parts = [p for p in [r.get("author"), r.get("year"), r.get("title"), r.get("doi") or r.get("url")] if p]
            out.append(" - ".join(parts))
        return out
