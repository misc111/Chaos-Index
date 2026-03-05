from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.time import utc_now_iso
from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.metrics import metric_bundle
from src.training.backtest import run_walk_forward_backtest
from src.training.model_feature_research import (
    MODEL_FEATURE_MAP_PATH_TEMPLATE,
    RESEARCHABLE_MODELS,
    _model_feature_pruning_config,
    load_model_feature_map,
    rank_model_features,
    save_model_feature_map,
    select_model_features,
)
from src.training.train import select_feature_columns


@dataclass(frozen=True)
class FeatureWidthEvalResult:
    league: str
    model_name: str
    summary_path: str
    summary_rows: list[dict[str, object]]
    best_width: int
    best_features: list[str]
    registry_path: str
    registry_updated: bool


def default_width_candidates(model_name: str, *, current_width: int | None = None, step: int = 4) -> list[int]:
    min_features, max_features, _ = _model_feature_pruning_config(model_name)
    upper = max_features if current_width is None else max(min_features, min(int(current_width), max_features))
    widths = list(range(upper, min_features - 1, -abs(int(step or 4))))
    if widths[-1] != min_features:
        widths.append(min_features)
    return list(dict.fromkeys(widths))


def _normalize_width_candidates(model_name: str, candidate_widths: list[int] | None, *, current_width: int | None = None) -> list[int]:
    min_features, max_features, _ = _model_feature_pruning_config(model_name)
    if candidate_widths:
        widths = [max(min_features, min(int(width), max_features)) for width in candidate_widths]
        if current_width is not None:
            widths.append(max(min_features, min(int(current_width), max_features)))
        return sorted(set(widths), reverse=True)
    return default_width_candidates(model_name, current_width=current_width)


def _fold_metric_summary(oof: pd.DataFrame, model_name: str) -> dict[str, float]:
    fold_rows: list[dict[str, float]] = []
    work = oof[["fold", "home_win", model_name]].dropna().copy()
    for _, grp in work.groupby("fold"):
        y = grp["home_win"].astype(int).to_numpy()
        p = grp[model_name].astype(float).to_numpy()
        fold_rows.append(metric_bundle(y, p))
    if not fold_rows:
        return {
            "fold_log_loss_mean": float("nan"),
            "fold_log_loss_sd": float("nan"),
            "fold_auc_mean": float("nan"),
            "fold_auc_sd": float("nan"),
        }
    fold_df = pd.DataFrame(fold_rows)
    return {
        "fold_log_loss_mean": float(fold_df["log_loss"].mean()),
        "fold_log_loss_sd": float(fold_df["log_loss"].std(ddof=0)),
        "fold_auc_mean": float(fold_df["auc"].mean()),
        "fold_auc_sd": float(fold_df["auc"].std(ddof=0)),
    }


def run_feature_width_eval(
    features_df: pd.DataFrame,
    *,
    league: str,
    model_name: str,
    artifacts_dir: str,
    bayes_cfg: dict,
    n_splits: int = 5,
    feature_columns: list[str] | None = None,
    candidate_widths: list[int] | None = None,
    path_template: str = MODEL_FEATURE_MAP_PATH_TEMPLATE,
    approve_changes: bool = False,
) -> FeatureWidthEvalResult:
    league_code = str(league or "NHL").strip().upper()
    model_code = str(model_name or "").strip()
    if model_code not in RESEARCHABLE_MODELS:
        raise ValueError(f"Unsupported model_name '{model_name}'. Expected one of {RESEARCHABLE_MODELS}.")

    train_df = features_df[features_df["home_win"].notna()].copy().sort_values("start_time_utc")
    if train_df.empty:
        raise RuntimeError("Feature-width evaluation requires finalized games with non-null home_win.")

    all_feature_columns = list(feature_columns) if feature_columns is not None else select_feature_columns(features_df)
    ranked_rows = rank_model_features(
        train_df,
        model_name=model_code,
        feature_columns=all_feature_columns,
        league=league_code,
    )
    if not ranked_rows:
        raise RuntimeError(f"No eligible ranked features found for {model_code} in {league_code}.")
    ranked_features = [str(row["feature"]) for row in ranked_rows]

    current_model_map = load_model_feature_map(league_code, path_template=path_template)
    current_width = len(current_model_map.get(model_code, [])) or None
    widths = _normalize_width_candidates(model_code, candidate_widths, current_width=current_width)

    timestamp = utc_now_iso().replace(":", "").replace("+00:00", "Z")
    eval_root = Path(artifacts_dir) / "research" / f"{model_code}_width_eval_{timestamp}"
    eval_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    best_features_by_width: dict[int, list[str]] = {}
    allow_nn = model_code == "nn_mlp"
    min_train_size = max(350, min(220, max(80, len(train_df) // 2))) if allow_nn else None

    for width in widths:
        selected_features = select_model_features(
            train_df,
            model_name=model_code,
            league=league_code,
            ranked_features=ranked_features,
            target_width=width,
        )
        actual_width = len(selected_features)
        best_features_by_width[actual_width] = selected_features

        width_dir = eval_root / f"width_{actual_width}"
        width_dir.mkdir(parents=True, exist_ok=True)
        bt = run_walk_forward_backtest(
            features_df,
            artifacts_dir=str(width_dir),
            bayes_cfg=bayes_cfg,
            n_splits=n_splits,
            selected_models=[model_code],
            selected_feature_columns=all_feature_columns,
            selected_model_feature_columns={model_code: selected_features},
            allow_nn=allow_nn,
            min_train_size=min_train_size,
        )
        metrics_df = bt["metrics"]
        if metrics_df.empty or model_code not in set(metrics_df["model_name"]):
            raise RuntimeError(f"Backtest did not produce metrics for {model_code} at width {actual_width}.")
        metric_row = metrics_df.loc[metrics_df["model_name"] == model_code].iloc[0].to_dict()
        oof = bt["oof_predictions"][["fold", "home_win", model_code]].dropna().copy()
        y = oof["home_win"].astype(int).to_numpy()
        p = oof[model_code].astype(float).to_numpy()
        summary_rows.append(
            {
                "width": int(width),
                "feature_count": int(actual_width),
                "log_loss": float(metric_row["log_loss"]),
                "brier": float(metric_row["brier"]),
                "accuracy": float(metric_row["accuracy"]),
                "auc": float(metric_row["auc"]),
                **ece_mce(y, p),
                **calibration_alpha_beta(y, p),
                **_fold_metric_summary(oof, model_code),
                "top_features": selected_features[:12],
            }
        )

    summary_rows.sort(
        key=lambda row: (
            float(row["log_loss"]),
            float(row["brier"]),
            -float(row["auc"]),
            int(row["feature_count"]),
        )
    )
    summary_path = eval_root / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))

    best = summary_rows[0]
    best_width = int(best["feature_count"])
    best_features = best_features_by_width[best_width]

    registry_path = str(Path(path_template.replace("{league}", league_code.lower())))
    registry_updated = False
    if approve_changes:
        merged = load_model_feature_map(league_code, path_template=path_template)
        merged[model_code] = best_features
        registry_path = str(save_model_feature_map(league_code, merged, path_template=path_template))
        registry_updated = True

    return FeatureWidthEvalResult(
        league=league_code,
        model_name=model_code,
        summary_path=str(summary_path),
        summary_rows=summary_rows,
        best_width=best_width,
        best_features=best_features,
        registry_path=registry_path,
        registry_updated=registry_updated,
    )
