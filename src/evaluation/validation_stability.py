from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression



def coefficient_paths(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "home_win",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [30, 60, 90]

    work = df[df[target_col].notna()].copy().sort_values("game_date_utc")
    if work.empty:
        return pd.DataFrame()

    out = []
    for w in windows:
        for i in range(w, len(work) + 1):
            seg = work.iloc[i - w : i]
            y = seg[target_col].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            x = seg[features].to_numpy(dtype=float)
            m = LogisticRegression(max_iter=1500, C=1.0)
            m.fit(x, y)
            for f, c in zip(features, m.coef_[0]):
                out.append(
                    {
                        "window": w,
                        "as_of": seg.iloc[-1]["game_date_utc"],
                        "feature": f,
                        "coef": float(c),
                    }
                )
    return pd.DataFrame(out)



def vif_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    x = df[features].dropna().copy()
    if x.empty:
        return pd.DataFrame(columns=["feature", "vif"])
    x = sm.add_constant(x)
    vals = []
    for i, col in enumerate(x.columns):
        if col == "const":
            continue
        vals.append({"feature": col, "vif": float(variance_inflation_factor(x.values, i))})
    out = pd.DataFrame(vals)
    out["condition_number"] = float(np.linalg.cond(x.values))
    return out



def break_test_trade_deadline(df: pd.DataFrame, features: list[str], target_col: str = "home_win") -> dict:
    work = df[df[target_col].notna()].copy()
    work["game_date_utc"] = pd.to_datetime(work["game_date_utc"])
    if work.empty:
        return {"delta_coef_l2": float("nan"), "n_pre": 0, "n_post": 0}

    deadline = pd.Timestamp(year=work["game_date_utc"].dt.year.max(), month=3, day=7)
    pre = work[work["game_date_utc"] < deadline]
    post = work[work["game_date_utc"] >= deadline]
    if len(pre) < 20 or len(post) < 20:
        return {"delta_coef_l2": float("nan"), "n_pre": len(pre), "n_post": len(post)}

    m1 = LogisticRegression(max_iter=1500).fit(pre[features], pre[target_col])
    m2 = LogisticRegression(max_iter=1500).fit(post[features], post[target_col])
    delta = float(np.linalg.norm(m1.coef_[0] - m2.coef_[0]))
    return {"delta_coef_l2": delta, "n_pre": len(pre), "n_post": len(post)}
