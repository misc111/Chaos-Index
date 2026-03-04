from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm



def influence_diagnostics(df: pd.DataFrame, features: list[str], target_col: str = "home_win", top_k: int = 10) -> tuple[pd.DataFrame, dict]:
    work = df[df[target_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(), {}

    # Remove constant/degenerate columns to improve numerical stability.
    valid_features = [c for c in features if c in work.columns and work[c].nunique(dropna=False) > 1]
    if not valid_features:
        return pd.DataFrame(), {"error": "no_valid_features_for_influence"}

    y = work[target_col].astype(int)
    try:
        x = sm.add_constant(work[valid_features].astype(float))
        model = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        infl = model.get_influence(observed=True)
    except Exception as exc:
        return pd.DataFrame(), {"error": f"influence_failed: {exc.__class__.__name__}", "message": str(exc)}

    leverage = infl.hat_matrix_diag
    cooks = infl.cooks_distance[0]
    dfbetas = infl.dfbetas

    diag = pd.DataFrame(
        {
            "row_id": work.index,
            "leverage": leverage,
            "cooks_distance": cooks,
            "abs_dfbeta_mean": np.abs(dfbetas).mean(axis=1),
        }
    )
    top = diag.sort_values("cooks_distance", ascending=False).head(top_k)

    keep_ix = [i for i in work.index if i not in set(top["row_id"])]
    refit = sm.GLM(y.loc[keep_ix], x.loc[keep_ix], family=sm.families.Binomial()).fit()
    coef_shift = float(np.linalg.norm(model.params.values - refit.params.reindex(model.params.index).values))

    summary = {
        "coef_shift_without_topk": coef_shift,
        "top_k": top_k,
        "n_total": int(len(work)),
    }
    return top.reset_index(drop=True), summary
