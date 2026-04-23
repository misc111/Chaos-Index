from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.training.contracts import LassoCredibilityMetadata


def _columns_matching(feature_cols: list[str], needles: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for feature in feature_cols:
        token = str(feature or "").strip().lower()
        if any(needle in token for needle in needles):
            out.append(feature)
    return out


def _scenario_specs(
    feature_cols: list[str],
    *,
    league: str,
    credibility: LassoCredibilityMetadata | None,
) -> list[dict[str, object]]:
    league_code = str(league or "").strip().upper()
    market_cols = _columns_matching(feature_cols, ("market", "vig", "implied", "odds", "price", "spread", "total", "offset"))
    if credibility is not None and credibility.complement_column in feature_cols:
        market_cols = list(dict.fromkeys([credibility.complement_column, *market_cols]))

    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    return [
        {
            "scenario": "market_complement_removed",
            "columns": market_cols,
            "fill_strategy": "zero" if credibility is not None and credibility.offset_scale in {"log", "logit"} else "median",
        },
        {
            "scenario": "unknown_starter_context",
            "columns": _columns_matching(feature_cols, ("starter", "starting_pitcher", "pitcher_hand")),
            "fill_strategy": "median",
        },
        {
            "scenario": "bullpen_context_removed",
            "columns": _columns_matching(feature_cols, ("bullpen", "pitcher_out_count")),
            "fill_strategy": "median",
        },
        {
            "scenario": "lineup_context_removed",
            "columns": _columns_matching(feature_cols, ("lineup", "position_player", "slugging")),
            "fill_strategy": "median",
        },
        {
            "scenario": "park_weather_removed",
            "columns": _columns_matching(feature_cols, ("park", "weather", "wind", "temperature", "humidity", "umpire")),
            "fill_strategy": "median",
        },
    ]


def _apply_fill_value(frame: pd.DataFrame, column: str, *, fill_strategy: str) -> None:
    if column not in frame.columns:
        return
    if fill_strategy == "zero":
        frame[column] = 0.0
        return
    filled = pd.to_numeric(frame[column], errors="coerce")
    frame[column] = filled.fillna(filled.median()).fillna(0.0)


def missingness_stress_test(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "home_win",
    *,
    league: str = "MLB",
    credibility: LassoCredibilityMetadata | None = None,
) -> pd.DataFrame:
    eval_df = df[df[target_col].notna()].copy()
    if eval_df.empty:
        return pd.DataFrame()

    y = eval_df[target_col].astype(int).to_numpy()
    base_p = np.clip(model.predict_proba(eval_df), 1e-6, 1 - 1e-6)
    base_ll = float(log_loss(y, base_p, labels=[0, 1]))

    rows = []
    for spec in _scenario_specs(feature_cols, league=league, credibility=credibility):
        name = str(spec["scenario"])
        cols = [str(col) for col in spec.get("columns", []) if str(col).strip()]
        fill_strategy = str(spec.get("fill_strategy") or "median")
        scen = eval_df.copy()
        for column in cols:
            _apply_fill_value(scen, column, fill_strategy=fill_strategy)
        if cols:
            p = np.clip(model.predict_proba(scen), 1e-6, 1 - 1e-6)
            ll = float(log_loss(y, p, labels=[0, 1]))
            scenario_status = "ok"
        else:
            ll = base_ll
            scenario_status = "not_applicable"
        rows.append(
            {
                "scenario": name,
                "scenario_status": scenario_status,
                "applied_feature_count": int(len(cols)),
                "applied_features": "|".join(cols),
                "fill_strategy": fill_strategy,
                "log_loss": ll,
                "delta_log_loss": ll - base_ll,
            }
        )

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
