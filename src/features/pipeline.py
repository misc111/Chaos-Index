"""Shared feature-pipeline stages with league strategy hooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.time import utc_now_iso
from src.common.utils import stable_hash
from src.features.base import FeatureBuildResult
from src.features.strategies.base import FeatureStrategy


def load_interim(name: str, interim_dir: str) -> pd.DataFrame:
    parquet_path = Path(interim_dir) / f"{name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = Path(interim_dir) / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _aggregate_stats(stats_df: pd.DataFrame, aggregations: dict[str, tuple[str, str]]) -> pd.DataFrame:
    if stats_df.empty or not aggregations:
        return pd.DataFrame(columns=["game_id", "team", *aggregations.keys()])
    return stats_df.groupby(["game_id", "team"], dropna=False).agg(**aggregations).reset_index()


def expand_team_games(games_df: pd.DataFrame, stats_df: pd.DataFrame, strategy: FeatureStrategy) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    summary_rows = stats_df[stats_df["goalie_id"].isna()].copy() if not stats_df.empty else pd.DataFrame()
    starter_rows = (
        stats_df[(stats_df["starter_status"] == "confirmed") & stats_df["goalie_id"].notna()].copy()
        if not stats_df.empty and strategy.starter_aggregations
        else pd.DataFrame()
    )

    summary = _aggregate_stats(summary_rows, strategy.summary_aggregations)
    starters = _aggregate_stats(starter_rows, strategy.starter_aggregations)
    team_extra = summary.merge(starters, on=["game_id", "team"], how="outer")

    base_cols = [
        "game_id",
        "season",
        "game_date_utc",
        "start_time_utc",
        "venue",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "status_final",
        "home_win",
        "as_of_utc",
    ]

    home = games_df[base_cols].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = 1
    home[strategy.team_value_for_column] = home["home_score"]
    home[strategy.team_value_against_column] = home["away_score"]
    home[strategy.team_result_column] = home["home_win"]

    away = games_df[base_cols].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = 0
    away[strategy.team_value_for_column] = away["away_score"]
    away[strategy.team_value_against_column] = away["home_score"]
    away[strategy.team_result_column] = np.where(away["home_win"].isna(), np.nan, 1 - away["home_win"].astype(float))

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.merge(team_extra, on=["game_id", "team"], how="left")
    if "starter_status" in team_games.columns:
        team_games["starter_status"] = team_games["starter_status"].fillna("unknown")
    return team_games


def apply_team_rolling_windows(
    team_games: pd.DataFrame,
    players_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
    strategy: FeatureStrategy,
) -> pd.DataFrame:
    if team_games.empty:
        return team_games

    prepared = strategy.prepare_team_games(team_games, players_df=players_df, injuries_df=injuries_df)

    def _per_team(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["team"] = group.name
        g["start_dt"] = pd.to_datetime(g["start_time_utc"], errors="coerce")
        g["prev_start_dt"] = g["start_dt"].shift(1)
        g["rest_days"] = (g["start_dt"] - g["prev_start_dt"]).dt.total_seconds() / 86400
        g["rest_days"] = g["rest_days"].fillna(7).clip(lower=0)
        g["b2b"] = (g["rest_days"] <= 1.1).astype(int)
        g["games_played_prior"] = range(len(g))
        for col in strategy.rolling_value_columns:
            g[f"ewm_{col}"] = g[col].shift(1).ewm(alpha=0.2, adjust=False).mean()
            g[f"r5_{col}"] = g[col].shift(1).rolling(5, min_periods=1).mean()
            g[f"r14_{col}"] = g[col].shift(1).rolling(14, min_periods=1).mean()
        g["win_rate_ewm"] = g[strategy.team_result_column].shift(1).ewm(alpha=0.2, adjust=False).mean()
        return g

    rolled = prepared.groupby("team", group_keys=False).apply(_per_team, include_groups=False)
    return strategy.finalize_team_games(rolled)


def merge_game_level_frames(team_games: pd.DataFrame, games_df: pd.DataFrame, strategy: FeatureStrategy) -> pd.DataFrame:
    home = team_games[team_games["is_home"] == 1].copy()
    away = team_games[team_games["is_home"] == 0].copy()

    home_cols = [c for c in home.columns if c not in {"opponent", strategy.team_result_column}]
    away_cols = [c for c in away.columns if c not in {"opponent", strategy.team_result_column}]
    home = home[home_cols].add_prefix("home_").rename(columns={"home_game_id": "game_id"})
    away = away[away_cols].add_prefix("away_").rename(columns={"away_game_id": "game_id"})

    base_cols = [
        "game_id",
        "season",
        "game_date_utc",
        "start_time_utc",
        "venue",
        "home_team",
        "away_team",
        "status_final",
        "home_win",
        "as_of_utc",
        "home_score",
        "away_score",
    ]
    merged = games_df[base_cols].copy().merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
    merged = merged.rename(
        columns={
            "home_team_x": "home_team",
            "away_team_x": "away_team",
            "venue_x": "venue",
            "season_x": "season",
            "game_date_utc_x": "game_date_utc",
            "start_time_utc_x": "start_time_utc",
            "status_final_x": "status_final",
            "home_win_x": "home_win",
            "as_of_utc_x": "as_of_utc",
            "home_score_x": "home_score",
            "away_score_x": "away_score",
        }
    )
    drop_suffix_cols = [c for c in merged.columns if c.endswith("_y") and c.split("_")[0] in {"home", "away", "season", "venue"}]
    if drop_suffix_cols:
        merged = merged.drop(columns=drop_suffix_cols)

    for src, name in strategy.diff_pairs:
        hc = f"home_{src}"
        ac = f"away_{src}"
        if hc in merged.columns and ac in merged.columns:
            merged[f"diff_{name}"] = merged[hc] - merged[ac]

    enriched = strategy.enrich_game_level(merged, games_df=games_df, team_games=team_games)
    existing_drop = [c for c in strategy.direct_event_drop_columns if c in enriched.columns]
    if existing_drop:
        enriched = enriched.drop(columns=existing_drop)
    return enriched


def finalize_feature_frame(game_features: pd.DataFrame, processed_dir: str, strategy: FeatureStrategy) -> FeatureBuildResult:
    drop_cols = {
        "game_id",
        "game_date_utc",
        "start_time_utc",
        "home_team",
        "away_team",
        "venue",
        "home_win",
        "status_final",
        "as_of_utc",
        "home_score",
        "away_score",
        "season",
    }
    feature_columns = [c for c in game_features.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(game_features[c])]

    for col in feature_columns:
        game_features[col] = pd.to_numeric(game_features[col], errors="coerce")
    game_features[feature_columns] = game_features[feature_columns].replace([np.inf, -np.inf], np.nan)
    game_features[feature_columns] = game_features[feature_columns].fillna(game_features[feature_columns].median(numeric_only=True)).fillna(0)

    feature_set_version = f"fset_{stable_hash(strategy.feature_hash_payload(feature_columns))}"
    metadata = {
        "feature_set_version": feature_set_version,
        "built_at_utc": utc_now_iso(),
        "league": strategy.league,
        "n_rows": int(len(game_features)),
        "n_features": int(len(feature_columns)),
    }

    out_path = Path(processed_dir) / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = out_path
    try:
        game_features.to_parquet(out_path, index=False)
    except Exception:
        saved_path = Path(processed_dir) / "features.csv"
        game_features.to_csv(saved_path, index=False)
    metadata["saved_path"] = str(saved_path)
    return FeatureBuildResult(
        dataframe=game_features,
        feature_columns=feature_columns,
        feature_set_version=feature_set_version,
        metadata=metadata,
    )


def build_features_from_interim_with_strategy(interim_dir: str, processed_dir: str, strategy: FeatureStrategy) -> FeatureBuildResult:
    games = load_interim("games", interim_dir).sort_values("start_time_utc").reset_index(drop=True)
    stats = load_interim("goalies", interim_dir)
    players = load_interim("players", interim_dir)
    injuries = load_interim("injuries", interim_dir)

    team_games = expand_team_games(games, stats, strategy)
    team_games = apply_team_rolling_windows(team_games, players_df=players, injuries_df=injuries, strategy=strategy)
    game_features = merge_game_level_frames(team_games, games, strategy)
    game_features = strategy.add_model_transforms(game_features)
    return finalize_feature_frame(game_features, processed_dir=processed_dir, strategy=strategy)
