from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score



def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pr = np.asarray(p, dtype=float)
    return float(np.mean((pr - y) ** 2))



def metric_bundle(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pr = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "log_loss": float(log_loss(y, pr, labels=[0, 1])),
        "brier": brier_score(y, pr),
        "accuracy": float(accuracy_score(y, (pr >= 0.5).astype(int))),
    }
    try:
        out["auc"] = float(roc_auc_score(y, pr))
    except Exception:
        out["auc"] = float("nan")
    return out



def per_game_scores(y_true: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    pr = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    ll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    br = (pr - y) ** 2
    acc = ((pr >= 0.5).astype(int) == y).astype(int)
    return pd.DataFrame({"log_loss": ll, "brier": br, "accuracy": acc})
