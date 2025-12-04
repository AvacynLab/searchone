from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List


def _normalize_number(value: str) -> Any:
    cleaned = value.replace(",", ".").replace(" ", "")
    try:
        if cleaned.endswith("%"):
            return float(cleaned.strip("%")) / 100.0
        return float(cleaned)
    except ValueError:
        return value


def _looks_like_date(value: str) -> bool:
    if not value:
        return False
    value = value.strip()
    patterns = [
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{2}/\d{2}/\d{2,4}$",
        r"^[A-Za-z]{3,}\s+\d{1,2},\s*\d{4}$",
    ]
    return any(re.match(p, value) for p in patterns)


def _normalize_value(value: str, type_hint: str) -> Any:
    if type_hint == "number":
        return _normalize_number(value)
    if type_hint == "date":
        if _looks_like_date(value):
            try:
                return datetime.fromisoformat(value.replace("/", "-")).isoformat()
            except Exception:
                return value
        return value
    return value.strip()


def infer_schema(table: Dict[str, Any]) -> Dict[str, Any]:
    rows = table.get("rows") or []
    header = rows[0] if rows else []
    schema_columns = []
    for idx, raw_label in enumerate(header):
        label = (raw_label or f"col_{idx + 1}").strip()
        unit = None
        match = re.search(r"\(([^)]+)\)", label)
        if match:
            unit = match.group(1).strip()
            label = label.replace(f"({unit})", "").strip()
        values = [row[idx] for row in rows[1:] if len(row) > idx]
        type_hint = "string"
        if values and all(_looks_like_date(str(v)) for v in values):
            type_hint = "date"
        elif values and all(_is_number_like(str(v)) for v in values):
            type_hint = "number"
        schema_columns.append({"name": label or f"col_{idx + 1}", "type": type_hint, "unit": unit})
    return {"columns": schema_columns, "source": table.get("id")}


def _is_number_like(value: str) -> bool:
    try:
        _normalize_number(value)
        return True
    except Exception:
        return False


def table_to_records(table: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = table.get("rows") or []
    columns = schema.get("columns", [])
    header = [col.get("name") or f"col_{idx + 1}" for idx, col in enumerate(columns)]
    records = []
    for row in rows[1:]:
        entry: Dict[str, Any] = {}
        for idx, col_name in enumerate(header):
            if idx >= len(row):
                continue
            type_hint = columns[idx].get("type", "string")
            value = row[idx]
            entry[col_name] = _normalize_value(str(value), type_hint)
        if entry:
            records.append(entry)
    return records


def summarize_table(table: Dict[str, Any]) -> str:
    schema = infer_schema(table)
    rows = len(table.get("rows") or []) - 1
    cols = len(schema.get("columns", []))
    return f"Table {table.get('id')} ({cols} cols × {max(rows,0)} rows)"
