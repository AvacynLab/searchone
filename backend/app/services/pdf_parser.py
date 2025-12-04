from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore[assignment]


TableRow = List[str]


def _split_table_line(line: str) -> List[str]:
    """Split a line into columns using common table separators."""
    parts = [part.strip() for part in re.split(r"\s{2,}|\t|\||;", line) if part.strip()]
    return parts


def _detect_tables(page_text: str, page_number: int, source_path: str) -> List[Dict[str, Any]]:
    """Naively detect tables by looking for multi-column rows with consistent widths."""
    tables: List[Dict[str, Any]] = []
    lines = [line for line in page_text.splitlines() if line.strip()]
    current_rows: List[TableRow] = []
    current_col_count: Optional[int] = None
    table_idx = 0

    def emit_table():
        nonlocal table_idx, current_rows, current_col_count
        if current_rows and current_col_count and len(current_rows) >= 2:
            table_id = f"{source_path}:{page_number}:{table_idx}"
            tables.append(
                {
                    "id": table_id,
                    "page": page_number,
                    "rows": current_rows.copy(),
                    "columns": current_col_count,
                }
            )
            table_idx += 1
        current_rows = []
        current_col_count = None

    for line in lines:
        parts = _split_table_line(line)
        if len(parts) < 2:
            emit_table()
            continue
        if current_col_count is None:
            current_col_count = len(parts)
        if len(parts) == current_col_count:
            current_rows.append(parts)
            continue
        emit_table()
        current_col_count = len(parts)
        if current_col_count >= 2:
            current_rows = [parts]
        else:
            current_rows = []
            current_col_count = None
    emit_table()
    return tables


def parse_pdf(path: str) -> Dict[str, Any]:
    """Parse a PDF file into pages, text, metadata, and detect simple tables."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDF parsing.")
    doc_path = Path(path)
    doc = fitz.open(path)
    pages: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    tables: List[Dict[str, Any]] = []

    for idx, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        pages.append({"number": idx, "text": text})
        full_text_parts.append(text)
        tables.extend(_detect_tables(text, idx, str(doc_path)))

    metadata = doc.metadata or {}
    metadata.update(
        {
            "path": str(doc_path),
            "page_count": doc.page_count,
            "title": metadata.get("title") or doc_path.stem,
        }
    )

    return {
        "full_text": "\n".join(full_text_parts).strip(),
        "pages": pages,
        "tables": tables,
        "metadata": metadata,
    }
