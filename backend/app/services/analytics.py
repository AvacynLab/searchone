from typing import List, Dict
import math


def stats_summary(values: List[float]) -> Dict[str, float]:
    """Return basic stats (mean, median, std, min, max) for a list of numbers."""
    if not values:
        return {}
    vals = [float(v) for v in values]
    n = len(vals)
    mean = sum(vals) / n
    sorted_vals = sorted(vals)
    mid = n // 2
    if n % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = sorted_vals[mid]
    variance = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(variance)
    return {
        "count": n,
        "mean": mean,
        "median": median,
        "std": std,
        "min": min(vals),
        "max": max(vals),
    }


def ttest_independent(a: List[float], b: List[float]) -> Dict[str, float]:
    """Simple Welch's t-test (unequal variances)."""
    if not a or not b:
        return {}
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    na, nb = len(a), len(b)
    mean_a, mean_b = sum(a) / na, sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    # Welch-Satterthwaite
    t_num = mean_a - mean_b
    t_den = math.sqrt(var_a / na + var_b / nb) if (var_a or var_b) else 0.0
    if t_den == 0.0:
        return {"t": 0.0, "df": 0.0}
    t_stat = t_num / t_den
    df_num = (var_a / na + var_b / nb) ** 2
    df_den = ((var_a / na) ** 2 / (na - 1)) + ((var_b / nb) ** 2 / (nb - 1))
    df = df_num / df_den if df_den else 0.0
    return {"t": t_stat, "df": df, "mean_a": mean_a, "mean_b": mean_b}
