from __future__ import annotations

import numpy as np
import pandas as pd



def brier_decompose(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pr = np.asarray(p, dtype=float)
    if y.size == 0:
        return {"brier": float("nan"), "reliability": float("nan"), "resolution": float("nan"), "uncertainty": float("nan")}

    base = y.mean()
    bins_arr = np.linspace(0, 1, bins + 1)
    idx = np.digitize(pr, bins_arr, right=True)

    reliability = 0.0
    resolution = 0.0
    for b in range(1, bins + 1):
        m = idx == b
        if m.sum() == 0:
            continue
        pk = pr[m].mean()
        ok = y[m].mean()
        w = m.mean()
        reliability += w * (pk - ok) ** 2
        resolution += w * (ok - base) ** 2

    uncertainty = base * (1 - base)
    brier = np.mean((pr - y) ** 2)
    return {
        "brier": float(brier),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }
