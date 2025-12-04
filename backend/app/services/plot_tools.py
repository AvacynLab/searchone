from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union
import uuid

import matplotlib.pyplot as plt

from app.core.config import DATA_DIR
from app.core.logging_config import configure_logging

configure_logging()

PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PlotArtifact:
    png_path: Path
    metadata: Dict[str, Any]
    vector_paths: Dict[str, Path] = field(default_factory=dict)


def _ensure_iterable(value: Union[Sequence[Any], Iterable[Any], Any]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_series(raw_series: Union[Mapping[str, Any], List[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(raw_series, Mapping):
        series = [raw_series]
    else:
        series = raw_series

    normalized = []
    for entry in series:
        y_values = entry.get("values") or entry.get("y") or []
        if y_values is None:
            y_values = []
        y_values = list(y_values)
        x_values = entry.get("x")
        if x_values is None:
            x_values = list(range(len(y_values)))
        else:
            x_values = list(x_values)
        if len(x_values) != len(y_values):
            # fallback to index alignment if lengths mismatch
            x_values = list(range(len(y_values)))
        normalized.append(
            {
                "label": entry.get("label") or entry.get("name") or "series",
                "x": x_values,
                "y": y_values,
                "errors": list(entry.get("errors")) if entry.get("errors") else None,
                "color": entry.get("color"),
                "style": entry.get("style"),
            }
        )
    return normalized


def generate_plot(data: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]], spec: Mapping[str, Any]) -> PlotArtifact:
    """Generate a plot from structured data and spec, return the artifact metadata."""
    series = _normalize_series(data)
    plot_type = spec.get("type", "line")
    job_id = spec.get("job_id") or spec.get("context", {}).get("job_id")
    folder = PLOTS_DIR / (f"job_{job_id}" if job_id else "misc")
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base_name = spec.get("filename") or spec.get("title", "plot")
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in base_name).strip("_")
    if not safe_name:
        safe_name = "plot"
    stem = f"{safe_name}_{timestamp}_{uuid.uuid4().hex[:6]}"

    fig, ax = plt.subplots()
    for entry in series:
        label = entry["label"]
        if plot_type == "hist":
            ax.hist(entry["y"], label=label, color=entry.get("color"))
        elif plot_type == "bar":
            ax.bar(entry["x"], entry["y"], label=label, color=entry.get("color"), alpha=0.8)
        else:  # line or scatter (default to line)
            plot_kwargs = {"color": entry.get("color"), "label": label}
            if plot_type == "scatter":
                ax.scatter(entry["x"], entry["y"], **{k: v for k, v in plot_kwargs.items() if v is not None})
            else:
                if entry.get("errors"):
                    ax.errorbar(entry["x"], entry["y"], yerr=entry["errors"], fmt="-o", **{k: v for k, v in plot_kwargs.items() if v is not None})
                else:
                    ax.plot(entry["x"], entry["y"], **{k: v for k, v in plot_kwargs.items() if v is not None})
    title = spec.get("title")
    if title:
        ax.set_title(title)
    x_label = spec.get("x_label")
    y_label = spec.get("y_label")
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    scale = spec.get("scale", {})
    for axis in ("x", "y"):
        axis_scale = scale.get(axis)
        if axis_scale:
            if axis == "x":
                ax.set_xscale(axis_scale)
            else:
                ax.set_yscale(axis_scale)
    if spec.get("legend", True):
        ax.legend()
    ax.grid(spec.get("grid", True))
    png_path = folder / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    vector_paths: Dict[str, Path] = {}
    vector_formats = spec.get("vector_formats") or _ensure_iterable(spec.get("vector_format"))
    allowed_vector_formats = {"svg", "pdf"}
    for fmt in vector_formats:
        fmt = fmt.lower()
        if fmt not in allowed_vector_formats:
            continue
        path = folder / f"{stem}.{fmt}"
        fig.savefig(path)
        vector_paths[fmt] = path
    plt.close(fig)
    metadata: Dict[str, Any] = {
        "job_id": job_id,
        "title": title,
        "description": spec.get("description"),
        "variables": spec.get("variables"),
        "plot_type": plot_type,
        "series": [{"label": entry["label"], "points": len(entry["y"])} for entry in series],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return PlotArtifact(png_path=png_path, vector_paths=vector_paths, metadata=metadata)
