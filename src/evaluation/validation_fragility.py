from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss



def missingness_stress_test(model, df: pd.DataFrame, feature_cols: list[str], target_col: str = "home_win") -> pd.DataFrame:
    eval_df = df[df[target_col].notna()].copy()
    if eval_df.empty:
        return pd.DataFrame()

    y = eval_df[target_col].astype(int).to_numpy()
    base_p = np.clip(model.predict_proba(eval_df), 1e-6, 1 - 1e-6)
    base_ll = float(log_loss(y, base_p, labels=[0, 1]))

    scenarios = {
        "no_xg": [c for c in feature_cols if "xg" in c.lower()],
        "unknown_goalie": [c for c in feature_cols if "goalie" in c.lower()],
        "no_injuries": [c for c in feature_cols if "lineup" in c.lower() or "man_games" in c.lower()],
    }

    rows = []
    for name, cols in scenarios.items():
        scen = eval_df.copy()
        for c in cols:
            if c in scen.columns:
                scen[c] = scen[c].median()
        p = np.clip(model.predict_proba(scen), 1e-6, 1 - 1e-6)
        ll = float(log_loss(y, p, labels=[0, 1]))
        rows.append({"scenario": name, "log_loss": ll, "delta_log_loss": ll - base_ll})

    return pd.DataFrame(rows)



def perturbation_sensitivity(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    jitter_frac: float = 0.05,
    n_draws: int = 30,
) -> dict:
    base = np.clip(model.predict_proba(df), 1e-6, 1 - 1e-6)
    rng = np.random.default_rng(42)
    deltas = []

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return {"mean_abs_delta": 0.0, "p95_abs_delta": 0.0}

    for _ in range(n_draws):
        pert = df.copy()
        for c in numeric_cols:
            scale = df[c].std() if df[c].std() > 0 else 1.0
            pert[c] = pert[c] + rng.normal(0, jitter_frac * scale, size=len(pert))
        p = np.clip(model.predict_proba(pert), 1e-6, 1 - 1e-6)
        deltas.append(np.abs(p - base))

    arr = np.vstack(deltas)
    return {
        "mean_abs_delta": float(arr.mean()),
        "p95_abs_delta": float(np.quantile(arr, 0.95)),
    }
