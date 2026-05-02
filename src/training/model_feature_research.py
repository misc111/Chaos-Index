from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from src.common.time import utc_now_iso
from src.training.model_feature_guardrails import (
    apply_model_feature_guardrails,
    default_guardrails_path_template,
    find_model_feature_guardrail_conflicts,
    resolve_model_feature_guardrails_path,
)


MODEL_FEATURE_MAP_PATH_TEMPLATE = "configs/model_feature_map_{league}.yaml"
MODEL_FEATURE_MAP_JSON_TEMPLATE = "configs/generated/model_feature_map_{league}.json"
RESEARCHABLE_MODELS = [
    "glm_ridge",
    "glm_elastic_net",
    "glm_lasso",
    "glm_vanilla",
    "gam_spline",
    "glmm_logit",
    "dglm_margin",
    "goals_poisson",
]


@dataclass(frozen=True)
class ModelFeatureResearchResult:
    league: str
    registry_path: str
    report_path: str
    score_table_path: str
    approved_model_features: dict[str, list[str]]
    registry_updated: bool


def _ordered_model_feature_items(model_features: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """Return model feature-map items in deterministic model-key order."""

    return [(str(model_name), list(features)) for model_name, features in sorted(model_features.items())]


def resolve_model_feature_map_path(path_template: str, league: str) -> Path:
    league_token = str(league or "unknown").strip().lower()
    rendered = str(path_template).replace("{league}", league_token)
    return Path(rendered)


def default_model_feature_map_json_template(path_template: str) -> str:
    rendered = str(path_template or MODEL_FEATURE_MAP_PATH_TEMPLATE)
    if rendered == MODEL_FEATURE_MAP_PATH_TEMPLATE:
        return MODEL_FEATURE_MAP_JSON_TEMPLATE
    if rendered.endswith(".yaml"):
        return f"{rendered[:-5]}.json"
    if rendered.endswith(".yml"):
        return f"{rendered[:-4]}.json"
    return f"{rendered}.json"


def export_model_feature_map_json(
    league: str,
    model_features: dict[str, list[str]],
    *,
    path_template: str = MODEL_FEATURE_MAP_JSON_TEMPLATE,
) -> Path:
    path = resolve_model_feature_map_path(path_template, league=league)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "league": str(league or "MLB").strip().upper(),
        "updated_at_utc": utc_now_iso(),
        "models": {
            model_name: {
                "active_features": list(features),
                "feature_count": len(features),
            }
            for model_name, features in _ordered_model_feature_items(model_features)
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_model_feature_map(
    league: str,
    path_template: str = MODEL_FEATURE_MAP_PATH_TEMPLATE,
    guardrails_path_template: str | None = None,
) -> dict[str, list[str]]:
    return load_model_feature_map_with_guardrails(
        league,
        path_template=path_template,
        guardrails_path_template=guardrails_path_template,
    )


def load_model_feature_map_with_guardrails(
    league: str,
    *,
    path_template: str = MODEL_FEATURE_MAP_PATH_TEMPLATE,
    guardrails_path_template: str | None = None,
) -> dict[str, list[str]]:
    path = resolve_model_feature_map_path(path_template, league=league)
    if not path.exists():
        return {}
    guardrails_template = guardrails_path_template or default_guardrails_path_template(path_template)
    guardrails_path = resolve_model_feature_guardrails_path(guardrails_template, league=league)
    raw = yaml.safe_load(path.read_text()) or {}
    models = raw.get("models", {})
    if not isinstance(models, dict):
        return {}
    out: dict[str, list[str]] = {}
    for model_name, payload in models.items():
        features = payload.get("active_features", []) if isinstance(payload, dict) else []
        ordered = [str(col) for col in features if str(col).strip()]
        blocked_hits = find_model_feature_guardrail_conflicts(
            ordered,
            league=league,
            model_name=str(model_name),
            path_template=guardrails_template,
        )
        if blocked_hits:
            raise RuntimeError(
                "Blocked features found in active model feature map "
                f"{path} for {str(league).upper()}/{model_name}: {blocked_hits}. "
                f"Update {guardrails_path} if this reintroduction is intentional, or remove them from the active map."
            )
        out[str(model_name)] = ordered
    return out


def save_model_feature_map(
    league: str,
    model_features: dict[str, list[str]],
    *,
    path_template: str = MODEL_FEATURE_MAP_PATH_TEMPLATE,
    guardrails_path_template: str | None = None,
) -> Path:
    league_code = str(league or "MLB").strip().upper()
    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    registry_path = resolve_model_feature_map_path(path_template, league=league_code)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    guardrails_template = guardrails_path_template or default_guardrails_path_template(path_template)
    json_path_template = default_model_feature_map_json_template(path_template)
    sanitized_model_features = {
        model_name: apply_model_feature_guardrails(
            list(features),
            league=league_code,
            model_name=model_name,
            path_template=guardrails_template,
        )[0]
        for model_name, features in _ordered_model_feature_items(model_features)
    }
    payload = {
        "version": 1,
        "league": league_code,
        "updated_at_utc": utc_now_iso(),
        "models": {
            model_name: {
                "active_features": list(features),
                "feature_count": len(features),
            }
            for model_name, features in sanitized_model_features.items()
        },
    }
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    export_model_feature_map_json(league_code, sanitized_model_features, path_template=json_path_template)
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
    league_code = str(league or "MLB").strip().upper()
    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    cols = [str(c) for c in feature_columns]

    if league_code == "MLB":
        glm_pool = [
            c
            for c in cols
            if c.startswith("diff_")
            or c
            in {
                "starting_pitcher_hand_matchup",
                "lineup_stability_diff",
                "park_run_effect",
                "park_weather_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
                "series_context_index",
                "day_game_indicator",
                "umpire_run_effect",
                "weather_wind_run_effect",
                "weather_temperature_run_effect",
                "weather_humidity_run_effect",
            }
        ]
        bayes_pool = [
            c
            for c in cols
            if c.startswith(("diff_",))
            or c
            in {
                "park_run_effect",
                "park_weather_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
                "series_context_index",
                "day_game_indicator",
                "umpire_run_effect",
            }
        ]
        two_stage_pool = [
            c
            for c in cols
            if c.startswith(("home_", "away_", "diff_"))
            or c
            in {
                "starting_pitcher_hand_matchup",
                "lineup_stability_diff",
                "park_run_effect",
                "park_weather_run_effect",
                "weather_wind_run_effect",
                "weather_temperature_run_effect",
                "weather_humidity_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
                "series_context_index",
                "day_game_indicator",
                "umpire_run_effect",
            }
        ]
        tree_pool = [
            c
            for c in cols
            if not c.endswith(("_team", "_name", "_id"))
            and c not in {"home_starting_pitcher_hand", "away_starting_pitcher_hand"}
        ]

        if model_name in {"glm_ridge", "glm_elastic_net", "glm_lasso"}:
            return glm_pool
        if model_name == "bayes_bt_state_space":
            return bayes_pool
        if model_name == "two_stage":
            return two_stage_pool
        if model_name in {"gbdt", "rf", "nn_mlp"}:
            return tree_pool
        return cols

    return cols


def _model_feature_pruning_config(model_name: str, league: str | None = None) -> tuple[int, int, float]:
    league_code = str(league or "MLB").strip().upper()
    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    limits = {
        "glm_ridge": (8, 14, 0.92),
        "glm_elastic_net": (8, 14, 0.92),
        "glm_lasso": (6, 10, 0.92),
        "gbdt": (18, 32, 0.88),
        "rf": (16, 28, 0.92),
        "two_stage": (12, 20, 0.92),
        "bayes_bt_state_space": (8, 14, 0.92),
        "nn_mlp": (18, 32, 0.92),
    }
    return limits.get(model_name, (12, 24, 0.92))


def _default_model_feature_target_width(model_name: str, league: str) -> int:
    league_code = str(league or "MLB").strip().upper()
    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    defaults = {
        "glm_ridge": 11,
        "glm_elastic_net": 12,
        "glm_lasso": 8,
        "two_stage": 16,
        "bayes_bt_state_space": 12,
        "gbdt": 25,
        "rf": 20,
        "nn_mlp": 26,
    }
    if model_name in defaults:
        return defaults[model_name]
    _, max_features, _ = _model_feature_pruning_config(model_name, league=league_code)
    return max_features


def _anchor_features(model_name: str, league: str) -> list[str]:
    league_code = str(league or "MLB").strip().upper()
    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    anchors = {
            "glm_ridge": [
                "diff_starting_pitcher_quality",
                "starting_pitcher_hand_matchup",
                "diff_bullpen_quality",
                "diff_lineup_talent",
                "park_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
            ],
            "glm_elastic_net": [
                "diff_starting_pitcher_quality",
                "starting_pitcher_hand_matchup",
                "diff_bullpen_quality",
                "diff_bullpen_fatigue",
                "diff_lineup_talent",
                "park_run_effect",
                "park_weather_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
            ],
            "glm_lasso": [
                "diff_starting_pitcher_quality",
                "diff_bullpen_quality",
                "diff_lineup_talent",
                "park_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
            ],
            "two_stage": [
                "diff_starting_pitcher_quality",
                "starting_pitcher_hand_matchup",
                "diff_bullpen_quality",
                "diff_bullpen_fatigue",
                "diff_lineup_talent",
                "lineup_stability_diff",
                "park_run_effect",
                "park_weather_run_effect",
                "weather_wind_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
                "series_context_index",
                "umpire_run_effect",
                "diff_offense_form",
                "diff_pitching_form",
            ],
            "gbdt": [
                "diff_starting_pitcher_quality",
                "diff_bullpen_quality",
                "diff_lineup_talent",
                "park_run_effect",
                "weather_wind_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
                "series_context_index",
            ],
            "rf": [
                "diff_starting_pitcher_quality",
                "diff_bullpen_quality",
                "diff_lineup_talent",
                "park_run_effect",
                "weather_wind_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
            ],
            "bayes_bt_state_space": [
                "diff_starting_pitcher_quality",
                "diff_bullpen_quality",
                "diff_lineup_talent",
                "park_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
                "series_context_index",
            ],
            "nn_mlp": [
                "diff_starting_pitcher_quality",
                "diff_bullpen_quality",
                "diff_lineup_talent",
                "park_run_effect",
                "weather_wind_run_effect",
                "rest_diff",
                "travel_diff",
                "home_field_advantage",
                "series_context_index",
            ],
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
    guardrails_path_template: str | None = None,
) -> list[dict[str, object]]:
    league_code = str(league or "MLB").strip().upper()
    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    eligible = [c for c in _eligible_features_for_model(model_name, feature_columns, league_code) if c in train_df.columns]
    eligible, _ = apply_model_feature_guardrails(
        eligible,
        league=league_code,
        model_name=model_name,
        path_template=guardrails_path_template or default_guardrails_path_template(MODEL_FEATURE_MAP_PATH_TEMPLATE),
    )
    if not eligible:
        return []

    y = train_df["home_win"].astype(float)
    stage_targets = [
        c
        for c in [
            "target_home_runs",
            "target_away_runs",
            "target_total_runs",
            "target_run_differential",
            "target_run_environment",
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
    league_code = str(league or "MLB").strip().upper()
    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    min_features, max_features, max_abs_corr = _model_feature_pruning_config(model_name, league=league_code)
    width = _default_model_feature_target_width(model_name, league_code) if target_width is None else int(target_width)
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
    guardrails_path_template: str | None = None,
) -> ModelFeatureResearchResult:
    league_code = str(league or "MLB").strip().upper()
    if league_code != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    guardrails_template = guardrails_path_template or default_guardrails_path_template(path_template)
    train_df = features_df[features_df["home_win"].notna()].copy().sort_values("start_time_utc")
    if train_df.empty:
        raise RuntimeError("Model feature research requires finalized games with non-null home_win.")

    model_names = [m for m in (selected_models or RESEARCHABLE_MODELS) if m in RESEARCHABLE_MODELS]
    report_rows: list[dict[str, object]] = []
    approved_model_features: dict[str, list[str]] = {}
    guardrail_exclusions: dict[str, list[str]] = {}

    for model_name in model_names:
        _, blocked_candidate_hits = apply_model_feature_guardrails(
            _eligible_features_for_model(model_name, feature_columns, league_code),
            league=league_code,
            model_name=model_name,
            path_template=guardrails_template,
        )
        if blocked_candidate_hits:
            guardrail_exclusions[model_name] = list(blocked_candidate_hits)

        scored_rows = rank_model_features(
            train_df,
            model_name=model_name,
            feature_columns=feature_columns,
            league=league_code,
            guardrails_path_template=guardrails_template,
        )
        if not scored_rows:
            approved_model_features[model_name] = []
            continue

        ranked_features = [str(row["feature"]) for row in scored_rows]
        target_width = _default_model_feature_target_width(model_name, league_code)
        selected = select_model_features(
            train_df,
            model_name=model_name,
            league=league_code,
            ranked_features=ranked_features,
            target_width=target_width,
        )
        selected, blocked_hits = apply_model_feature_guardrails(
            selected,
            league=league_code,
            model_name=model_name,
            path_template=guardrails_template,
        )
        if blocked_hits:
            existing = guardrail_exclusions.get(model_name, [])
            guardrail_exclusions[model_name] = list(dict.fromkeys(existing + blocked_hits))
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
        "guardrail_exclusions": guardrail_exclusions,
        "model_feature_guardrails_path": str(resolve_model_feature_guardrails_path(guardrails_template, league=league_code)),
        "score_table_path": str(score_table_path),
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True))

    registry_path = resolve_model_feature_map_path(path_template, league=league_code)
    registry_updated = False
    if approve_changes:
        merged_model_features = load_model_feature_map_with_guardrails(
            league_code,
            path_template=path_template,
            guardrails_path_template=guardrails_template,
        )
        merged_model_features.update(approved_model_features)
        registry_path = save_model_feature_map(
            league_code,
            merged_model_features,
            path_template=path_template,
            guardrails_path_template=guardrails_template,
        )
        registry_updated = True

    return ModelFeatureResearchResult(
        league=league_code,
        registry_path=str(registry_path),
        report_path=str(report_path),
        score_table_path=str(score_table_path),
        approved_model_features=approved_model_features,
        registry_updated=registry_updated,
    )
