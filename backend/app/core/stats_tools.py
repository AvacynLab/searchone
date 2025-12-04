import numpy as np
from typing import List, Dict


def correlation_matrix(values: List[List[float]]) -> Dict[str, List[List[float]]]:
    """Compute Pearson correlation matrix for a 2D list of floats."""
    arr = np.array(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    corr = np.corrcoef(arr, rowvar=False).tolist()
    return {"corr": corr}
