"""
Bench pipeline vs baseline (LLM direct) on a small fixed set of queries.
Placeholder: uses configuration only; actual LLM calls should be wired in if desired.
"""
import json
from pathlib import Path
from hashlib import md5

QUERIES = [
    "Comparer deux approches de compression de modeles",
    "Impact des NBS sur l'adaptation climatique en Europe",
]


def _heuristic_cost(query: str) -> dict:
    """Deterministic pseudo-benchmark to compare pipeline vs baseline without LLM calls."""
    h = int(md5(query.encode("utf-8")).hexdigest(), 16)
    length_factor = len(query.split())
    pipeline_time = round((length_factor * 0.4) + (h % 7) * 0.1, 2)
    baseline_time = round((length_factor * 0.3) + (h % 5) * 0.1, 2)
    pipeline_tokens = length_factor * 150 + (h % 50)
    baseline_tokens = length_factor * 90 + (h % 30)
    score_delta = pipeline_time - baseline_time
    return {
        "pipeline_time_s": pipeline_time,
        "baseline_time_s": baseline_time,
        "pipeline_tokens": pipeline_tokens,
        "baseline_tokens": baseline_tokens,
        "score_delta": score_delta,
    }


def main():
    results = []
    overrides = []
    for q in QUERIES:
        metrics = _heuristic_cost(q)
        results.append({"query": q, **metrics})
        # Simple recommendation: if pipeline slower, drop temperature / switch to fast model profile
        overrides.append(
            {
                "query": q,
                "recommended_profile": "brouillon rapide" if metrics["score_delta"] > 0 else "cerveau lourd",
                "temperature": 0.2 if metrics["score_delta"] > 0 else 0.35,
            }
        )
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bench_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    rec_path = out_dir / "bench_overrides.json"
    rec_path.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote bench results to {out_path}")
    print(f"Wrote prompt/model overrides suggestions to {rec_path}")


if __name__ == "__main__":
    main()
