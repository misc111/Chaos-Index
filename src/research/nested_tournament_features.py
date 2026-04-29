"""Feature-pool and feature-variant helpers for nested MLB tournaments.

These helpers sit between raw pregame feature frames and candidate model specs.
They enforce leakage checks, resolve structured GLM slates, and produce explicit
market-feature coverage rows.  That keeps feature governance separate from model
fitting and tournament orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common.config import AppConfig
from src.features.leakage_checks import run_leakage_checks
from src.research.mlb_targets import TargetDefinition
from src.research.model_comparison import (
    FEATURE_POOL_FULL_SCREENED,
    FEATURE_POOL_PRODUCTION_MODEL_MAP,
    FEATURE_POOL_RESEARCH_BROAD,
    _select_feature_sets,
)
from src.research.nested_tournament_contracts import DEFAULT_STRUCTURED_SPEC_PATH, FeatureVariant
from src.services.train import load_features_dataframe
from src.training.feature_selection import select_feature_columns
from src.training.model_feature_research import load_model_feature_map

def _raw_features_for_pool(cfg: AppConfig, features_df: pd.DataFrame, *, feature_pool: str, feature_map_model: str) -> tuple[list[str], str]:
    all_raw_features = select_feature_columns(features_df)
    leakage_issues = run_leakage_checks(features_df, feature_columns=all_raw_features)
    if leakage_issues:
        raise RuntimeError(f"Leakage checks failed before nested tournament: {leakage_issues}")
    token = str(feature_pool or FEATURE_POOL_FULL_SCREENED).strip().lower()
    if token == FEATURE_POOL_PRODUCTION_MODEL_MAP:
        model_feature_map = load_model_feature_map(cfg.data.league)
        requested = [str(col) for col in model_feature_map.get(str(feature_map_model), []) if str(col).strip()]
        missing = [feature for feature in requested if feature not in all_raw_features]
        if missing:
            raise ValueError(f"Production feature map `{feature_map_model}` includes unavailable columns: {missing}")
        return requested, f"production_model_map:{feature_map_model}"
    if token in {FEATURE_POOL_FULL_SCREENED, FEATURE_POOL_RESEARCH_BROAD}:
        return all_raw_features, token
    raise ValueError(f"Unknown nested tournament feature pool `{feature_pool}`")


def _raw_features_for_target(target_df: pd.DataFrame, base_raw_features: list[str]) -> list[str]:
    target_features = select_feature_columns(target_df)
    combined = list(dict.fromkeys(list(base_raw_features) + target_features))
    leakage_issues = run_leakage_checks(target_df, feature_columns=combined)
    if leakage_issues:
        raise RuntimeError(f"Leakage checks failed after target/market enrichment: {leakage_issues}")
    return combined


def _load_features(cfg: AppConfig, *, feature_pool: str) -> pd.DataFrame:
    if str(feature_pool or "").strip().lower() == FEATURE_POOL_RESEARCH_BROAD:
        from src.common.research import resolve_research_paths

        research_paths = resolve_research_paths(cfg)
        return load_features_dataframe(str(research_paths.processed_dir))
    return load_features_dataframe(cfg.paths.processed_dir)


def _screened_feature_sets(fit_df: pd.DataFrame, *, target_col: str, raw_features: list[str]):
    work = fit_df.copy()
    work["home_win"] = pd.to_numeric(work[target_col], errors="coerce")
    return _select_feature_sets(work, raw_features)


def _resolve_spec_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path if path.exists() else None


def _structured_variants(feature_sets: Any, *, spec_path: str | Path | None) -> list[FeatureVariant]:
    path = _resolve_spec_path(spec_path or DEFAULT_STRUCTURED_SPEC_PATH)
    screened_set = set(feature_sets.screened_features)
    screening_frame = getattr(feature_sets, "screening_frame", pd.DataFrame())
    screening_by_feature = (
        {
            str(row["feature"]): row
            for row in screening_frame.to_dict(orient="records")
            if str(row.get("feature") or "").strip()
        }
        if isinstance(screening_frame, pd.DataFrame) and not screening_frame.empty
        else {}
    )
    variants: list[FeatureVariant] = []

    def structured_feature(feature: str) -> str | None:
        if feature in screened_set:
            return feature
        row = screening_by_feature.get(feature)
        if not row:
            return None
        if str(row.get("reason") or "") == "constant_or_singleton_on_fit_window":
            return None
        retained = str(row.get("retained_as") or feature).strip()
        return retained if retained in screened_set else None

    if path is not None:
        payload = yaml.safe_load(path.read_text()) or {}
        slates = payload.get("slates") if isinstance(payload.get("slates"), dict) else {}
        for slate_name, slate_payload in slates.items():
            if not isinstance(slate_payload, dict):
                continue
            feature_order = [str(feature) for feature in slate_payload.get("feature_order", []) if str(feature).strip()]
            width_variants = slate_payload.get("width_variants") if isinstance(slate_payload.get("width_variants"), dict) else {}
            for width_name, width_payload in width_variants.items():
                count = int(width_payload.get("feature_count", 0) if isinstance(width_payload, dict) else 0)
                if count <= 0:
                    continue
                features = tuple(
                    dict.fromkeys(
                        resolved
                        for feature in feature_order[:count]
                        for resolved in [structured_feature(feature)]
                        if resolved
                    )
                )
                if features:
                    key = f"{slate_name}_{width_name}"
                    variants.append(
                        FeatureVariant(
                            variant_key=key,
                            display_name=key.replace("_", " ").title(),
                            source=f"structured:{path.name}",
                            features=features,
                        )
                    )

    ranked = list(feature_sets.ranking_frame["feature"]) if not feature_sets.ranking_frame.empty else list(feature_sets.screened_features)
    for count in (8, 12, 18):
        features = tuple(feature for feature in ranked[: min(count, len(ranked))] if feature in screened_set)
        if features:
            variants.append(
                FeatureVariant(
                    variant_key=f"screened_top_{count}",
                    display_name=f"Screened Top {count}",
                    source="screened_rank",
                    features=features,
                )
            )
    no_market = tuple(
        feature
        for feature in ranked
        if feature in screened_set
        and not any(token in feature.lower() for token in ("market", "vig", "odds", "price", "bookmaker"))
    )[:12]
    if no_market:
        variants.append(
            FeatureVariant(
                variant_key="no_market_top_12",
                display_name="No Market Top 12",
                source="screened_rank_no_market",
                features=no_market,
            )
        )

    deduped: dict[str, FeatureVariant] = {}
    for variant in variants:
        deduped.setdefault(variant.variant_key, variant)
    return list(deduped.values())


def _market_probability_feature(target: TargetDefinition) -> str:
    if target.target_name == "moneyline_home_win":
        return "market_vig_free_home_prob"
    return f"market_{target.market}_vig_free_positive_prob"


def _market_offset_feature(target: TargetDefinition) -> str:
    if target.target_name == "moneyline_home_win":
        return "market_offset_logit"
    return f"market_{target.market}_logit_positive_prob"


def _market_baseline_variants(target: TargetDefinition, feature_sets: Any) -> list[FeatureVariant]:
    market_prob = _market_probability_feature(target)
    market_offset = _market_offset_feature(target)
    ranked = list(feature_sets.ranking_frame["feature"]) if not feature_sets.ranking_frame.empty else list(feature_sets.screened_features)
    screened = set(feature_sets.screened_features)
    non_market = tuple(
        feature
        for feature in ranked
        if feature in screened
        and feature not in {market_prob, market_offset}
        and not any(token in feature.lower() for token in ("market", "vig", "odds", "price", "bookmaker"))
    )[:8]
    plus_features = tuple(dict.fromkeys((market_prob,) + non_market[:7]))
    variants = [
        FeatureVariant(
            variant_key="market_implied_only",
            display_name="Market Implied Only",
            source="market_baseline",
            features=(market_prob,),
        )
    ]
    if non_market:
        variants.append(
            FeatureVariant(
                variant_key="market_offset_glm",
                display_name="Market Offset GLM",
                source="market_baseline",
                features=(market_offset,) + non_market,
            )
        )
    if plus_features:
        variants.append(
            FeatureVariant(
                variant_key="market_plus_features_glm",
                display_name="Market Plus Features GLM",
                source="market_baseline",
                features=plus_features,
            )
        )
    return variants

def _target_prediction_frame(eval_df: pd.DataFrame, *, target: TargetDefinition) -> pd.DataFrame:
    columns = [column for column in ("game_id", "game_date_utc", "start_time_utc", "home_team", "away_team") if column in eval_df.columns]
    out = eval_df[columns].copy().reset_index(drop=True)
    out[f"{target.target_name}__actual"] = eval_df[target.target_col].reset_index(drop=True)
    return out


def _feature_coverage_rows(
    *,
    target: TargetDefinition,
    split_frames: dict[str, pd.DataFrame],
    features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    market_prob = _market_probability_feature(target)
    market_offset = _market_offset_feature(target)
    tracked = list(dict.fromkeys([market_prob, market_offset] + list(features)))
    for split_name, frame in split_frames.items():
        total_rows = int(len(frame))
        for feature in tracked:
            present = feature in frame.columns
            non_null = int(pd.to_numeric(frame[feature], errors="coerce").notna().sum()) if present else 0
            coverage = float(non_null / total_rows) if total_rows else 0.0
            rows.append(
                {
                    "target_name": target.target_name,
                    "market": target.market,
                    "split": split_name,
                    "feature": feature,
                    "present": bool(present),
                    "non_null_rows": non_null,
                    "total_rows": total_rows,
                    "coverage": coverage,
                    "is_market_feature": bool(
                        feature in {market_prob, market_offset}
                        or any(token in feature.lower() for token in ("market", "vig", "odds", "price", "bookmaker"))
                    ),
                }
            )
    return rows


def _feature_coverage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    grouped = (
        frame.groupby(["target_name", "split"], as_index=False)
        .agg(
            tracked_features=("feature", "count"),
            present_features=("present", "sum"),
            full_coverage_features=("coverage", lambda values: int((pd.Series(values) >= 0.999).sum())),
            market_features=("is_market_feature", "sum"),
        )
    )
    market_present = (
        frame[frame["is_market_feature"]]
        .groupby(["target_name", "split"], as_index=False)
        .agg(market_features_present=("present", "sum"), market_min_coverage=("coverage", "min"))
    )
    grouped = grouped.merge(market_present, on=["target_name", "split"], how="left")
    grouped["market_features_present"] = grouped["market_features_present"].fillna(0).astype(int)
    grouped["market_min_coverage"] = grouped["market_min_coverage"].fillna(0.0)
    return grouped.to_dict(orient="records")
