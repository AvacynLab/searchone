from app.workflows import writing_pipeline


def test_build_scientific_article_with_figures():
    state = {
        "query": "Impact du climat sur les vagues",
        "timeline": [
            {
                "messages": [
                    {
                        "evidence": [
                            {
                                "text": "Plot shows coverage",
                                "meta": {
                                    "source_type": "plot",
                                    "figure": {
                                        "title": "Couverture vs iterations",
                                        "path": "/tmp/figures/coverage.png",
                                        "description": "Evolution de la couverture"
                                    },
                                    "vectors": {"svg": "/tmp/figures/coverage.svg"},
                                },
                            },
                            {
                                "text": "Reference data extracted",
                                "meta": {
                                    "title": "Study X",
                                    "domain": "example.org",
                                    "published_at": "2024-01-01T00:00:00Z",
                                },
                            },
                        ],
                    }
                ]
            }
        ],
        "research_score": {"coverage": 0.78, "coherence": 0.8},
        "run_metrics": {"coverage_score": 0.72, "evidence_count": 4},
    }

    result = writing_pipeline.build_scientific_article(state, format="markdown")
    assert result["sections"]
    assert "latex" in result and "\\documentclass" in result["latex"]
    assert result["figures"]
    assert result["figures"][0]["title"] == "Couverture vs iterations"
    assert result["research_score"]["coverage"] == 0.78
    assert result["metrics"]["coverage_score"] == 0.72
