from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression



def reliability_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    pr = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(pr, bins, right=True)
    rows = []
    for b in range(1, n_bins + 1):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "pred_mean": float(pr[mask].mean()),
                "obs_rate": float(y[mask].mean()),
                "abs_gap": float(abs(pr[mask].mean() - y[mask].mean())),
            }
        )
    return pd.DataFrame(rows)



def ece_mce(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    tab = reliability_table(y_true, p, n_bins=n_bins)
    if tab.empty:
        return {"ece": float("nan"), "mce": float("nan")}
    w = tab["count"] / tab["count"].sum()
    ece = float((w * tab["abs_gap"]).sum())
    mce = float(tab["abs_gap"].max())
    return {"ece": ece, "mce": mce}



def calibration_alpha_beta(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pr = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(pr / (1 - pr)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"calibration_alpha": float("nan"), "calibration_beta": float("nan")}

    try:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(logits, y)
        alpha = float(lr.intercept_[0])
        beta = float(lr.coef_[0][0])
    except Exception:
        alpha, beta = float("nan"), float("nan")
    return {"calibration_alpha": alpha, "calibration_beta": beta}



def calibrate_platt(train_p: np.ndarray, train_y: np.ndarray, test_p: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(train_p, 1e-6, 1 - 1e-6) / np.clip(1 - train_p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logits, train_y)
    logits_test = np.log(np.clip(test_p, 1e-6, 1 - 1e-6) / np.clip(1 - test_p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return lr.predict_proba(logits_test)[:, 1]



def calibrate_isotonic(train_p: np.ndarray, train_y: np.ndarray, test_p: np.ndarray) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(train_p, train_y)
    return np.clip(ir.predict(test_p), 1e-6, 1 - 1e-6)
