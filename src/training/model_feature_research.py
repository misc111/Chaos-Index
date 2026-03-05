from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.time import utc_now_iso


MODEL_FEATURE_MAP_PATH_TEMPLATE = "configs/model_feature_map_{league}.yaml"
RESEARCHABLE_MODELS = ["glm_logit", "gbdt", "rf", "two_stage", "bayes_bt_state_space", "nn_mlp"]


@dataclass(frozen=True)
class ModelFeatureResearchResult:
    league: str
    registry_path: str
    report_path: str
    score_table_path: str
    approved_model_features: dict[str, list[str]]
    registry_updated: bool


def resolve_model_feature_map_path(path_template: str, league: str) -> Path:
    league_token = str(league or "unknown").strip().lower()
    rendered = str(path_template).replace("{league}", league_token)
    return Path(rendered)


def load_model_feature_map(league: str, path_template: str = MODEL_FEATURE_MAP_PATH_TEMPLATE) -> dict[str, list[str]]:
    path = resolve_model_feature_map_path(path_template, league=league)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    models = raw.get("models", {})
    if not isinstance(models, dict):
        return {}
    out: dict[str, list[str]] = {}
    for model_name, payload in models.items():
        features = payload.get("active_features", []) if isinstance(payload, dict) else []
        out[str(model_name)] = [str(col) for col in features if str(col).strip()]
    return out


def save_model_feature_map(
    league: str,
    model_features: dict[str, list[str]],
    *,
    path_template: str = MODEL_FEATURE_MAP_PATH_TEMPLATE,
) -> Path:
    league_code = str(league or "NHL").strip().upper()
    registry_path = resolve_model_feature_map_path(path_template, league=league_code)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "league": league_code,
        "updated_at_utc": utc_now_iso(),
        "models": {
            model_name: {
                "active_features": list(features),
                "feature_count": len(features),
            }
            for model_name, features in model_features.items()
        },
    }
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return registry_path


def _safe_abs_corr(feature: pd.Series, target: pd.Series) -> float:
    joined = pd.concat([feature, target], axis=1).dropna()
    if len(joined) < 20:
        return 0.0
    x = joined.iloc[:, 0]
    y = joined.iloc[:, 1]
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return 0.0
    value = x.corr(y, method="spearman")
    if pd.isna(value):
        return 0.0
    return float(abs(value))


def _eligible_features_for_model(model_name: str, feature_columns: list[str], league: str) -> list[str]:
    league_code = str(league or "NHL").strip().upper()
    cols = [str(c) for c in feature_columns]

    if league_code == "NBA":
        glm_pool = [
            c
            for c in cols
            if c.startswith(("diff_", "discipline_", "availability_", "elo_", "dyn_"))
            or c in {"travel_diff", "rest_diff", "arena_margin_effect", "arena_shot_volume_effect"}
        ]
        bayes_pool = [
            c
            for c in cols
            if c.startswith(("diff_", "availability_", "elo_", "dyn_"))
            or c in {"travel_diff", "rest_diff", "arena_margin_effect", "arena_shot_volume_effect"}
        ]
        two_stage_pool = [
            c
            for c in cols
            if c.startswith(
                (
                    "home_ewm_",
                    "away_ewm_",
                    "home_r5_",
                    "away_r5_",
                    "home_r14_",
                    "away_r14_",
                    "diff_",
                    "discipline_",
                    "availability_",
                    "elo_",
                    "dyn_",
                )
            )
            or c
            in {
                "travel_diff",
                "rest_diff",
                "arena_margin_effect",
                "arena_shot_volume_effect",
                "home_post_all_star_break",
                "away_post_all_star_break",
                "home_post_trade_deadline",
                "away_post_trade_deadline",
            }
        ]
        tree_pool = [
            c
            for c in cols
            if not c.startswith(("home_season", "away_season"))
            and not c.endswith(("_team", "_name"))
        ]

        if model_name == "glm_logit":
            return glm_pool
        if model_name == "bayes_bt_state_space":
            return bayes_pool
        if model_name == "two_stage":
            return two_stage_pool
        if model_name in {"gbdt", "rf", "nn_mlp"}:
            return tree_pool
        return cols

    if model_name == "glm_logit":
        return [
            c
            for c in cols
            if c.startswith(("diff_", "special_", "goalie_", "elo_", "dyn_"))
            or c in {"travel_diff", "rest_diff", "rink_goal_effect", "rink_shot_effect"}
        ]
    if model_name == "bayes_bt_state_space":
        return [
            c
            for c in cols
            if c.startswith(("diff_", "goalie_", "elo_", "dyn_"))
            or c in {"travel_diff", "rest_diff", "rink_goal_effect", "rink_shot_effect"}
        ]
    if model_name == "two_stage":
        return [
            c
            for c in cols
            if c.startswith(("home_ewm_", "away_ewm_", "home_r5_", "away_r5_", "home_r14_", "away_r14_", "diff_", "special_", "goalie_", "elo_", "dyn_"))
            or c in {"travel_diff", "rest_diff", "rink_goal_effect", "rink_shot_effect"}
        ]
    return cols


def _model_feature_pruning_config(model_name: str) -> tuple[int, int, float]:
    limits = {
        "glm_logit": (14, 24, 0.92),
        "gbdt": (24, 40, 0.88),
        "rf": (20, 44, 0.92),
        "two_stage": (16, 30, 0.92),
        "bayes_bt_state_space": (12, 20, 0.92),
        "nn_mlp": (18, 34, 0.92),
    }
    return limits.get(model_name, (12, 24, 0.92))


def _anchor_features(model_name: str, league: str) -> list[str]:
    if str(league or "NHL").strip().upper() == "NBA":
        anchors = {
            "glm_logit": [
                "diff_form_point_margin",
                "diff_form_win_rate",
                "travel_diff",
                "rest_diff",
                "discipline_free_throw_pressure_diff",
                "discipline_foul_margin_diff",
                "elo_home_prob",
                "dyn_home_prob",
                "arena_margin_effect",
            ],
            "gbdt": [
                "diff_form_point_margin",
                "travel_diff",
                "rest_diff",
                "elo_home_prob",
                "dyn_home_prob",
                "home_ewm_points_for",
                "away_ewm_points_for",
                "home_ewm_point_margin",
                "away_ewm_point_margin",
            ],
            "rf": [
                "diff_form_point_margin",
                "travel_diff",
                "rest_diff",
                "elo_home_prob",
                "dyn_home_prob",
            ],
            "two_stage": [
                "home_ewm_shot_volume_share",
                "away_ewm_shot_volume_share",
                "home_ewm_free_throw_pressure",
                "away_ewm_free_throw_pressure",
                "home_ewm_possession_proxy",
                "away_ewm_possession_proxy",
                "travel_diff",
                "rest_diff",
            ],
            "bayes_bt_state_space": [
                "diff_form_point_margin",
                "diff_form_win_rate",
                "travel_diff",
                "rest_diff",
                "elo_home_prob",
                "dyn_home_prob",
            ],
            "nn_mlp": [
                "diff_form_point_margin",
                "travel_diff",
                "rest_diff",
                "elo_home_prob",
                "dyn_home_prob",
            ],
        }
        return anchors.get(model_name, [])

    anchors = {
        "glm_logit": ["diff_form_goal_diff", "diff_form_win_rate", "travel_diff", "rest_diff", "elo_home_prob", "dyn_home_prob"],
        "bayes_bt_state_space": ["diff_form_goal_diff", "travel_diff", "rest_diff", "elo_home_prob", "dyn_home_prob"],
    }
    return anchors.get(model_name, [])


def _prune_correlated_features(
    train_df: pd.DataFrame,
    ranked_features: list[str],
    *,
    seed_features: list[str],
    min_features: int,
    max_features: int,
    max_abs_corr: float = 0.92,
) -> list[str]:
    selected: list[str] = []

    def _too_correlated(candidate: str) -> bool:
        for existing in selected:
            corr = _safe_abs_corr(train_df[candidate], train_df[existing])
            if corr >= max_abs_corr:
                return True
        return False

    for feature in seed_features:
        if feature in train_df.columns and feature in ranked_features and feature not in selected:
            selected.append(feature)

    for feature in ranked_features:
        if feature in selected:
            continue
        if len(selected) >= max_features:
            break
        if _too_correlated(feature):
            continue
        selected.append(feature)

    if len(selected) < min_features:
        for feature in ranked_features:
            if feature in selected:
                continue
            selected.append(feature)
            if len(selected) >= min_features:
                break
    return selected[:max_features]


def rank_model_features(
    train_df: pd.DataFrame,
    *,
    model_name: str,
    feature_columns: list[str],
    league: str,
) -> list[dict[str, object]]:
    league_code = str(league or "NHL").strip().upper()
    eligible = [c for c in _eligible_features_for_model(model_name, feature_columns, league_code) if c in train_df.columns]
    if not eligible:
        return []

    y = train_df["home_win"].astype(float)
    stage_targets = [
        c
        for c in [
            "target_xg_share",
            "target_penalty_diff",
            "target_pace",
            "target_shot_volume_share",
            "target_free_throw_pressure",
            "target_possession_volume",
        ]
        if c in train_df.columns
    ]

    scored_rows: list[dict[str, object]] = []
    for feature in eligible:
        feature_series = pd.to_numeric(train_df[feature], errors="coerce")
        outcome_score = _safe_abs_corr(feature_series, y)
        stage_score = 0.0
        if stage_targets:
            stage_score = max(_safe_abs_corr(feature_series, pd.to_numeric(train_df[target], errors="coerce")) for target in stage_targets)
        non_null_share = float(feature_series.notna().mean())
        combined_score = outcome_score + 0.35 * stage_score + 0.05 * non_null_share
        scored_rows.append(
            {
                "model_name": model_name,
                "feature": feature,
                "outcome_score": outcome_score,
                "stage_score": stage_score,
                "non_null_share": non_null_share,
                "combined_score": combined_score,
            }
        )

    scored_rows.sort(key=lambda row: (float(row["combined_score"]), float(row["outcome_score"])), reverse=True)
    return scored_rows


def select_model_features(
    train_df: pd.DataFrame,
    *,
    model_name: str,
    league: str,
    ranked_features: list[str],
    target_width: int | None = None,
) -> list[str]:
    league_code = str(league or "NHL").strip().upper()
    min_features, max_features, max_abs_corr = _model_feature_pruning_config(model_name)
    width = max_features if target_width is None else int(target_width)
    width = max(min_features, min(width, max_features))
    return _prune_correlated_features(
        train_df,
        ranked_features,
        seed_features=_anchor_features(model_name, league_code),
        min_features=width,
        max_features=width,
        max_abs_corr=max_abs_corr,
    )


def research_model_feature_map(
    features_df: pd.DataFrame,
    *,
    league: str,
    artifacts_dir: str,
    feature_columns: list[str],
    selected_models: list[str] | None = None,
    approve_changes: bool = False,
    path_template: str = MODEL_FEATURE_MAP_PATH_TEMPLATE,
) -> ModelFeatureResearchResult:
    league_code = str(league or "NHL").strip().upper()
    train_df = features_df[features_df["home_win"].notna()].copy().sort_values("start_time_utc")
    if train_df.empty:
        raise RuntimeError("Model feature research requires finalized games with non-null home_win.")

    model_names = [m for m in (selected_models or RESEARCHABLE_MODELS) if m in RESEARCHABLE_MODELS]
    report_rows: list[dict[str, object]] = []
    approved_model_features: dict[str, list[str]] = {}

    for model_name in model_names:
        scored_rows = rank_model_features(
            train_df,
            model_name=model_name,
            feature_columns=feature_columns,
            league=league_code,
        )
        if not scored_rows:
            approved_model_features[model_name] = []
            continue

        ranked_features = [str(row["feature"]) for row in scored_rows]
        _, max_features, _ = _model_feature_pruning_config(model_name)
        selected = select_model_features(
            train_df,
            model_name=model_name,
            league=league_code,
            ranked_features=ranked_features,
            target_width=max_features,
        )
        approved_model_features[model_name] = selected

        selected_set = set(selected)
        for rank, row in enumerate(scored_rows, start=1):
            report_rows.append(
                row
                | {
                    "league": league_code,
                    "rank": rank,
                    "selected": int(row["feature"] in selected_set),
                }
            )

    research_dir = Path(artifacts_dir) / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now_iso().replace(":", "").replace("+00:00", "Z")
    score_table_path = research_dir / f"{league_code.lower()}_model_feature_scores_{timestamp}.csv"
    report_path = research_dir / f"{league_code.lower()}_model_feature_research_{timestamp}.json"

    score_df = pd.DataFrame(report_rows)
    if not score_df.empty:
        score_df = score_df.sort_values(["model_name", "selected", "combined_score"], ascending=[True, False, False])
    score_df.to_csv(score_table_path, index=False)
    report_payload = {
        "league": league_code,
        "created_at_utc": utc_now_iso(),
        "selected_models": model_names,
        "approved_model_features": approved_model_features,
        "score_table_path": str(score_table_path),
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True))

    registry_path = resolve_model_feature_map_path(path_template, league=league_code)
    registry_updated = False
    if approve_changes:
        merged_model_features = load_model_feature_map(league_code, path_template=path_template)
        merged_model_features.update(approved_model_features)
        registry_path = save_model_feature_map(league_code, merged_model_features, path_template=path_template)
        registry_updated = True

    return ModelFeatureResearchResult(
        league=league_code,
        registry_path=str(registry_path),
        report_path=str(report_path),
        score_table_path=str(score_table_path),
        approved_model_features=approved_model_features,
        registry_updated=registry_updated,
    )
