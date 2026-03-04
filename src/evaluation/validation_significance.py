from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.evaluation.metrics import brier_score



def _fit_logit_ll(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> tuple[LogisticRegression, float]:
    m = LogisticRegression(max_iter=2000, C=c)
    m.fit(x, y)
    p = np.clip(m.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    ll = -float(log_loss(y, p, labels=[0, 1], normalize=False))
    return m, ll



def _ame(model: LogisticRegression, x: np.ndarray, cols: list[str], block_cols: list[str]) -> float:
    p = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    deriv = p * (1 - p)
    coef = dict(zip(cols, model.coef_[0]))
    return float(np.mean(np.abs([coef.get(c, 0.0) for c in block_cols])) * deriv.mean())



def blockwise_nested_lrt(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_blocks: dict[str, list[str]],
    all_features: list[str],
    target_col: str = "home_win",
    n_boot: int = 100,
) -> pd.DataFrame:
    y_train = train_df[target_col].astype(int).to_numpy()
    y_test = test_df[target_col].astype(int).to_numpy()
    x_full_train = train_df[all_features].to_numpy(dtype=float)
    x_full_test = test_df[all_features].to_numpy(dtype=float)

    full_model, ll_full = _fit_logit_ll(x_full_train, y_train)
    p_full = np.clip(full_model.predict_proba(x_full_test)[:, 1], 1e-6, 1 - 1e-6)
    full_logloss = float(log_loss(y_test, p_full, labels=[0, 1]))
    full_brier = brier_score(y_test, p_full)

    rows = []
    for block_name, block_cols in feature_blocks.items():
        reduced_cols = [c for c in all_features if c not in block_cols]
        if not reduced_cols:
            continue
        x_red_train = train_df[reduced_cols].to_numpy(dtype=float)
        x_red_test = test_df[reduced_cols].to_numpy(dtype=float)
        red_model, ll_red = _fit_logit_ll(x_red_train, y_train)
        p_red = np.clip(red_model.predict_proba(x_red_test)[:, 1], 1e-6, 1 - 1e-6)

        lrt = 2 * (ll_full - ll_red)
        df_diff = max(len(all_features) - len(reduced_cols), 1)
        pval = float(1 - chi2.cdf(max(lrt, 0), df=df_diff))
        delta_logloss = float(log_loss(y_test, p_red, labels=[0, 1]) - full_logloss)
        delta_brier = float(brier_score(y_test, p_red) - full_brier)

        rng = np.random.default_rng(42)
        ame_samples = []
        for _ in range(n_boot):
            ix = rng.choice(len(train_df), size=len(train_df), replace=True)
            xb = x_full_train[ix]
            yb = y_train[ix]
            try:
                bm, _ = _fit_logit_ll(xb, yb)
                ame_samples.append(_ame(bm, xb, all_features, block_cols))
            except Exception:
                continue
        ame_mean = float(np.mean(ame_samples)) if ame_samples else float("nan")
        ame_lo = float(np.quantile(ame_samples, 0.05)) if ame_samples else float("nan")
        ame_hi = float(np.quantile(ame_samples, 0.95)) if ame_samples else float("nan")

        rows.append(
            {
                "block": block_name,
                "lrt_stat": float(lrt),
                "df": int(df_diff),
                "p_value": pval,
                "delta_log_loss": delta_logloss,
                "delta_brier": delta_brier,
                "ame_mean": ame_mean,
                "ame_ci_low": ame_lo,
                "ame_ci_high": ame_hi,
            }
        )

    return pd.DataFrame(rows)
