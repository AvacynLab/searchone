from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

try:
    import spacy
except Exception:
    spacy = None  # type: ignore

_SPACY_MODEL = None


def _load_spacy_model() -> Optional[Any]:
    global _SPACY_MODEL
    if _SPACY_MODEL is not None:
        return _SPACY_MODEL
    if not spacy:
        return None
    for model in ("en_core_web_sm", "fr_core_news_sm"):
        try:
            _SPACY_MODEL = spacy.load(model)
            return _SPACY_MODEL
        except Exception:
            continue
    return None


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract entity spans using spaCy when available, with a fallback heuristic."""
    model = _load_spacy_model()
    entities: List[Dict[str, Any]] = []
    if model:
        doc = model(text)
        for ent in doc.ents:
            entities.append(
                {"text": ent.text, "label": ent.label_, "span": [ent.start_char, ent.end_char], "source": "spacy"}
            )
        return entities
    pattern = re.compile(r"\b[A-Z][\w]+(?:\s+[A-Z][\w]+)*\b")
    for match in pattern.finditer(text):
        span = match.span()
        entities.append({"text": match.group(0), "label": "CONCEPT", "span": [span[0], span[1]], "source": "heuristic"})
    return entities


def extract_relations(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Infer light relations by spotting keywords in sentences containing multiple entities."""
    if not entities:
        return []
    relations = []
    sentences = re.split(r"[.!?]\s*", text)
    keywords = {
        "causes": ["cause", "leads to", "results in", "induce", "trigger"],
        "part_of": ["is part of", "part of", "component of", "belongs to"],
        "uses": ["uses", "utilizes", "employs", "leverages"],
    }
    entity_texts = {ent["text"]: ent for ent in entities}
    for sentence in sentences:
        sent_lower = sentence.lower()
        present_entities = [ent for ent in entities if ent["text"] in sentence]
        if len(present_entities) < 2:
            continue
        for rel_type, patterns in keywords.items():
            for pattern in patterns:
                if pattern in sent_lower:
                    for src in present_entities:
                        for tgt in present_entities:
                            if src is tgt:
                                continue
                            relations.append(
                                {"type": rel_type, "from": src["text"], "to": tgt["text"], "context": sentence.strip()}
                            )
                    break
    return relations


def normalize_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize entity text for deduplication (lowercasing, stripping diacritics)."""
    normalized = []
    for ent in entities:
        text = ent.get("text", "")
        norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        norm = norm.lower().strip()
        normalized.append({**ent, "normalized": norm})
    return normalized
