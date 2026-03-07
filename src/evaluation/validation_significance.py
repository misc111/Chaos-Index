from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist
from sklearn.metrics import log_loss

from src.evaluation.metrics import brier_score


@dataclass(frozen=True, slots=True)
class _GLMPack:
    fit: sm.GLM
    medians: pd.Series
    features: list[str]
    exog_names: list[str]
    n_obs: int


def _safe_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=df.index)
    return df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _candidate_features(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    numeric = _safe_numeric_frame(df, features)
    medians = numeric.median(numeric_only=True).fillna(0.0)
    filled = numeric.fillna(medians).fillna(0.0)
    valid = [col for col in filled.columns if filled[col].nunique(dropna=False) > 1]
    return filled, medians, valid


def _design_matrix(
    df: pd.DataFrame,
    features: list[str],
    *,
    medians: pd.Series | None = None,
    valid_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    numeric = _safe_numeric_frame(df, features)
    if medians is None:
        medians = numeric.median(numeric_only=True).fillna(0.0)
    filled = numeric.fillna(medians).fillna(0.0)
    if valid_features is None:
        valid_features = [col for col in filled.columns if filled[col].nunique(dropna=False) > 1]
    x = filled[valid_features].copy() if valid_features else pd.DataFrame(index=filled.index)
    x = sm.add_constant(x, has_constant="add")
    return x, medians, valid_features


def _fit_binomial_glm(
    df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = "home_win",
) -> _GLMPack:
    work = df[df[target_col].notna()].copy()
    y = work[target_col].astype(int).to_numpy()
    _, medians, valid_features = _candidate_features(work, features)
    x, _, _ = _design_matrix(work, features, medians=medians, valid_features=valid_features)
    fit = sm.GLM(y, x, family=sm.families.Binomial()).fit()
    return _GLMPack(
        fit=fit,
        medians=medians,
        features=valid_features,
        exog_names=list(x.columns),
        n_obs=int(len(work)),
    )


def _predict_probability(pack: _GLMPack, df: pd.DataFrame, *, feature_scope: list[str]) -> np.ndarray:
    x, _, _ = _design_matrix(df, feature_scope, medians=pack.medians, valid_features=pack.features)
    x = x.reindex(columns=pack.exog_names, fill_value=1.0)
    p = pack.fit.predict(x)
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def _dispersion_estimate(pack: _GLMPack) -> float:
    df_resid = float(getattr(pack.fit, "df_resid", 0.0))
    pearson = float(getattr(pack.fit, "pearson_chi2", np.nan))
    if df_resid > 0 and np.isfinite(pearson):
        return max(pearson / df_resid, 1e-9)
    return 1.0


def _holdout_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=int)
    return float(log_loss(y, p, labels=[0, 1])), float(brier_score(y, p))


def blockwise_nested_deviance_f_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_blocks: dict[str, list[str]],
    all_features: list[str],
    target_col: str = "home_win",
) -> pd.DataFrame:
    if train_df.empty or test_df.empty:
        return pd.DataFrame(
            columns=[
                "block",
                "small_model_features",
                "big_model_features",
                "deviance_small",
                "deviance_big",
                "deviance_drop",
                "added_parameters",
                "dispersion_big",
                "f_stat",
                "df_num",
                "df_den",
                "p_value",
                "full_holdout_log_loss",
                "reduced_holdout_log_loss",
                "delta_log_loss",
                "full_holdout_brier",
                "reduced_holdout_brier",
                "delta_brier",
            ]
        )

    y_test = test_df[target_col].astype(int).to_numpy()
    full_pack = _fit_binomial_glm(train_df, all_features, target_col=target_col)
    full_holdout_p = _predict_probability(full_pack, test_df, feature_scope=all_features)
    full_logloss, full_brier = _holdout_metrics(y_test, full_holdout_p)

    rows: list[dict[str, float | int | str]] = []
    for block_name, block_cols in feature_blocks.items():
        reduced_cols = [col for col in all_features if col not in block_cols]
        if len(reduced_cols) == len(all_features):
            continue

        reduced_pack = _fit_binomial_glm(train_df, reduced_cols, target_col=target_col)
        reduced_holdout_p = _predict_probability(reduced_pack, test_df, feature_scope=reduced_cols)
        reduced_logloss, reduced_brier = _holdout_metrics(y_test, reduced_holdout_p)

        big_params = len(full_pack.exog_names)
        small_params = len(reduced_pack.exog_names)
        df_num = max(big_params - small_params, 1)
        df_den = max(int(round(getattr(full_pack.fit, "df_resid", 0.0))), 1)
        dev_small = float(reduced_pack.fit.deviance)
        dev_big = float(full_pack.fit.deviance)
        deviance_drop = max(dev_small - dev_big, 0.0)
        dispersion_big = _dispersion_estimate(full_pack)
        f_stat = max(deviance_drop / (df_num * dispersion_big), 0.0)
        p_value = float(f_dist.sf(f_stat, df_num, df_den))

        rows.append(
            {
                "block": block_name,
                "small_model_features": int(max(small_params - 1, 0)),
                "big_model_features": int(max(big_params - 1, 0)),
                "deviance_small": dev_small,
                "deviance_big": dev_big,
                "deviance_drop": deviance_drop,
                "added_parameters": int(df_num),
                "dispersion_big": float(dispersion_big),
                "f_stat": float(f_stat),
                "df_num": int(df_num),
                "df_den": int(df_den),
                "p_value": p_value,
                "full_holdout_log_loss": float(full_logloss),
                "reduced_holdout_log_loss": float(reduced_logloss),
                "delta_log_loss": float(reduced_logloss - full_logloss),
                "full_holdout_brier": float(full_brier),
                "reduced_holdout_brier": float(reduced_brier),
                "delta_brier": float(reduced_brier - full_brier),
            }
        )

    return pd.DataFrame(rows)


def information_criteria_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_blocks: dict[str, list[str]],
    all_features: list[str],
    target_col: str = "home_win",
) -> dict[str, pd.DataFrame | dict[str, float | int | str]]:
    candidates: list[tuple[str, list[str]]] = [("full_model", list(all_features))]
    seen = {tuple(all_features)}
    for block_name, block_cols in feature_blocks.items():
        reduced_cols = [col for col in all_features if col not in block_cols]
        key = tuple(reduced_cols)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((f"without_{block_name}", reduced_cols))
    if tuple() not in seen:
        candidates.append(("null_model", []))

    y_test = test_df[target_col].astype(int).to_numpy() if not test_df.empty else np.array([], dtype=int)
    rows: list[dict[str, float | int | str]] = []
    for candidate_name, features in candidates:
        pack = _fit_binomial_glm(train_df, features, target_col=target_col)
        holdout_logloss = float("nan")
        holdout_brier = float("nan")
        if len(y_test):
            p_test = _predict_probability(pack, test_df, feature_scope=features)
            holdout_logloss, holdout_brier = _holdout_metrics(y_test, p_test)

        k_params = len(pack.exog_names)
        bic = float(-2.0 * pack.fit.llf + k_params * np.log(max(pack.n_obs, 1)))
        rows.append(
            {
                "candidate": candidate_name,
                "feature_count": int(len(pack.features)),
                "parameter_count": int(k_params),
                "log_likelihood": float(pack.fit.llf),
                "deviance": float(pack.fit.deviance),
                "aic": float(pack.fit.aic),
                "bic": bic,
                "holdout_log_loss": holdout_logloss,
                "holdout_brier": holdout_brier,
            }
        )

    frame = pd.DataFrame(rows).sort_values(["aic", "bic", "holdout_log_loss"], na_position="last").reset_index(drop=True)
    if frame.empty:
        summary = {
            "status": "insufficient_data",
            "candidate_count": 0,
            "best_aic_candidate": "",
            "best_bic_candidate": "",
        }
    else:
        summary = {
            "status": "ok",
            "candidate_count": int(len(frame)),
            "best_aic_candidate": str(frame.sort_values("aic").iloc[0]["candidate"]),
            "best_bic_candidate": str(frame.sort_values("bic").iloc[0]["candidate"]),
            "best_holdout_log_loss_candidate": str(frame.sort_values("holdout_log_loss").iloc[0]["candidate"])
            if frame["holdout_log_loss"].notna().any()
            else "",
        }
        frame["delta_aic"] = frame["aic"] - float(frame["aic"].min())
        frame["delta_bic"] = frame["bic"] - float(frame["bic"].min())

    return {"summary": summary, "candidates": frame}


# Backward-compatible alias for older callers. The implementation now uses the paper's
# nested-model deviance F-test rather than the prior likelihood-ratio style report.
blockwise_nested_lrt = blockwise_nested_deviance_f_test
