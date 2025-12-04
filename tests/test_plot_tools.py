import matplotlib
matplotlib.use("Agg")

from app.services import plot_tools


def test_generate_plot_artifact(tmp_path, monkeypatch):
    plot_dir = tmp_path / "plots"
    monkeypatch.setattr(plot_tools, "PLOTS_DIR", plot_dir)
    data = [{
        "label": "series-a",
        "x": [0, 1, 2],
        "y": [1.2, 2.4, 3.6],
        "errors": [0.1, 0.2, 0.15],
    }]
    spec = {
        "job_id": "42",
        "title": "Test Plot",
        "description": "Coverage vs iterations",
        "type": "line",
        "x_label": "Iterations",
        "y_label": "Coverage",
        "vector_formats": ["svg", "pdf"],
        "scale": {"y": "linear"},
    }
    artifact = plot_tools.generate_plot(data, spec)

    assert artifact.png_path.exists()
    assert artifact.metadata["plot_type"] == "line"
    assert artifact.metadata["job_id"] == "42"
    assert len(artifact.metadata["series"]) == 1
    assert "svg" in artifact.vector_paths
    assert artifact.vector_paths["svg"].exists()
