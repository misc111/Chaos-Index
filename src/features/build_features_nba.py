from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.time import utc_now_iso
from src.common.utils import stable_hash
from src.features.base import FeatureBuildResult
from src.features.dynamic_ratings import compute_dynamic_rating_features
from src.features.elo import compute_elo_features
from src.features.travel import build_travel_features


def _load(name: str, interim_dir: str) -> pd.DataFrame:
    parquet_path = Path(interim_dir) / f"{name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = Path(interim_dir) / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _expand_team_games(games: pd.DataFrame, boxscore_stats: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    summary_rows = boxscore_stats[boxscore_stats["goalie_id"].isna()].copy() if not boxscore_stats.empty else pd.DataFrame()
    if not summary_rows.empty:
        summary = (
            summary_rows.groupby(["game_id", "team"], dropna=False)
            .agg(
                field_goal_attempts_for=("shots_for", "max"),
                field_goal_attempts_against=("shots_against", "max"),
                fouls_committed=("penalties_taken", "max"),
                fouls_drawn=("penalties_drawn", "max"),
                free_throws_made=("pp_goals", "max"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "game_id",
                "team",
                "field_goal_attempts_for",
                "field_goal_attempts_against",
                "fouls_committed",
                "fouls_drawn",
                "free_throws_made",
            ]
        )

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

    home = games[base_cols].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = 1
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]
    home["won"] = home["home_win"]

    away = games[base_cols].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = 0
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]
    away["won"] = np.where(away["home_win"].isna(), np.nan, 1 - away["home_win"].astype(float))

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.merge(summary, on=["game_id", "team"], how="left")
    return team_games


def _team_meta(players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
    roster = pd.DataFrame(columns=["team", "roster_depth_index"])
    if not players_df.empty and "team" in players_df.columns:
        roster = (
            players_df[players_df["team"].notna()]
            .groupby("team", as_index=False)
            .agg(active_roster_count=("player_id", "nunique"))
        )
        median_count = float(roster["active_roster_count"].median()) if not roster.empty else 0.0
        scale = median_count if median_count > 0 else 15.0
        roster["roster_depth_index"] = roster["active_roster_count"] / scale
        roster = roster[["team", "roster_depth_index"]]

    if not injuries_df.empty:
        injury_cols = [c for c in ["team", "lineup_uncertainty", "man_games_lost_proxy"] if c in injuries_df.columns]
        injuries = injuries_df[injury_cols].copy()
        injuries = injuries.rename(
            columns={
                "lineup_uncertainty": "availability_uncertainty",
                "man_games_lost_proxy": "absence_load_proxy",
            }
        )
    else:
        injuries = pd.DataFrame(columns=["team", "availability_uncertainty", "absence_load_proxy"])

    meta = roster.merge(injuries, on="team", how="outer")
    if "roster_depth_index" not in meta.columns:
        meta["roster_depth_index"] = 1.0
    if "availability_uncertainty" not in meta.columns:
        meta["availability_uncertainty"] = 1.0
    if "absence_load_proxy" not in meta.columns:
        meta["absence_load_proxy"] = 0.0
    meta["roster_depth_index"] = pd.to_numeric(meta["roster_depth_index"], errors="coerce").fillna(1.0)
    meta["availability_uncertainty"] = pd.to_numeric(meta["availability_uncertainty"], errors="coerce").fillna(1.0)
    meta["absence_load_proxy"] = pd.to_numeric(meta["absence_load_proxy"], errors="coerce").fillna(0.0)
    meta["availability_reliability"] = (1.0 - meta["availability_uncertainty"]).clip(lower=0.0, upper=1.0)
    return meta


def _team_rolling(team_games: pd.DataFrame, players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return team_games

    meta = _team_meta(players_df=players_df, injuries_df=injuries_df)
    df = team_games.sort_values(["team", "start_time_utc"]).copy()

    df["point_margin"] = df["points_for"].fillna(0) - df["points_against"].fillna(0)
    df["field_goal_attempts_for"] = df["field_goal_attempts_for"].fillna(df["points_for"] * 0.85)
    df["field_goal_attempts_against"] = df["field_goal_attempts_against"].fillna(df["points_against"] * 0.85)
    df["fouls_committed"] = df["fouls_committed"].fillna(20)
    df["fouls_drawn"] = df["fouls_drawn"].fillna(20)
    df["free_throws_made"] = df["free_throws_made"].fillna(df["points_for"] * 0.18)
    df["shot_volume_share"] = df["field_goal_attempts_for"] / (
        df["field_goal_attempts_for"] + df["field_goal_attempts_against"]
    ).replace(0, np.nan)
    df["shot_volume_share"] = df["shot_volume_share"].fillna(0.5)
    df["free_throw_pressure"] = df["free_throws_made"] / df["field_goal_attempts_for"].replace(0, np.nan)
    df["free_throw_pressure"] = df["free_throw_pressure"].fillna(0.18)
    df["foul_margin"] = df["fouls_drawn"] - df["fouls_committed"]
    df["scoring_efficiency_proxy"] = df["points_for"] / df["field_goal_attempts_for"].replace(0, np.nan)
    df["scoring_efficiency_proxy"] = df["scoring_efficiency_proxy"].fillna(1.1)
    df["possession_proxy"] = df["field_goal_attempts_for"] + df["field_goal_attempts_against"]

    def _per_team(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.copy()
        g["start_dt"] = pd.to_datetime(g["start_time_utc"], errors="coerce")
        g["prev_start_dt"] = g["start_dt"].shift(1)
        g["rest_days"] = (g["start_dt"] - g["prev_start_dt"]).dt.total_seconds() / 86400
        g["rest_days"] = g["rest_days"].fillna(7).clip(lower=0)
        g["b2b"] = (g["rest_days"] <= 1.1).astype(int)
        g["games_played_prior"] = range(len(g))

        rolling_cols = [
            "points_for",
            "points_against",
            "point_margin",
            "field_goal_attempts_for",
            "field_goal_attempts_against",
            "shot_volume_share",
            "free_throw_pressure",
            "foul_margin",
            "scoring_efficiency_proxy",
            "possession_proxy",
        ]
        for col in rolling_cols:
            g[f"ewm_{col}"] = g[col].shift(1).ewm(alpha=0.2, adjust=False).mean()
            g[f"r5_{col}"] = g[col].shift(1).rolling(5, min_periods=1).mean()
            g[f"r14_{col}"] = g[col].shift(1).rolling(14, min_periods=1).mean()
        g["win_rate_ewm"] = g["won"].shift(1).ewm(alpha=0.2, adjust=False).mean()
        g["shot_profile_proxy_used"] = 1
        return g

    df = df.groupby("team", group_keys=False).apply(_per_team)

    season_dates = pd.to_datetime(df["game_date_utc"], errors="coerce")
    season_start = season_dates.min()
    df["days_into_season"] = (season_dates - season_start).dt.days.fillna(0)
    df["days_into_season_spline"] = np.sqrt(df["days_into_season"].clip(lower=0))
    df["season_phase"] = pd.cut(
        df["days_into_season"],
        bins=[-1, 45, 120, 1000],
        labels=["early", "mid", "late"],
    ).astype(str)
    df["season_phase_early"] = (df["season_phase"] == "early").astype(int)
    df["season_phase_mid"] = (df["season_phase"] == "mid").astype(int)
    df["season_phase_late"] = (df["season_phase"] == "late").astype(int)
    df["post_all_star_break"] = (season_dates.dt.month >= 2).astype(int)
    df["post_trade_deadline"] = (season_dates.dt.month >= 2).astype(int)

    df = df.merge(meta, on="team", how="left")
    df["roster_depth_index"] = df["roster_depth_index"].fillna(1.0)
    df["availability_uncertainty"] = df["availability_uncertainty"].fillna(1.0)
    df["absence_load_proxy"] = df["absence_load_proxy"].fillna(0.0)
    df["availability_reliability"] = df["availability_reliability"].fillna(0.0)
    return df


def _compute_arena_effects(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["venue", "arena_margin_effect", "arena_shot_volume_effect"])

    tmp = games_df.copy()
    tmp["point_margin"] = tmp["home_score"].fillna(0) - tmp["away_score"].fillna(0)
    tmp["shot_volume_diff"] = tmp["home_field_goal_attempts_for"].fillna(0) - tmp["away_field_goal_attempts_for"].fillna(0)
    arena_mean = tmp.groupby("venue", dropna=False).agg(
        arena_margin_effect=("point_margin", "mean"),
        arena_shot_volume_effect=("shot_volume_diff", "mean"),
        n=("game_id", "count"),
    )
    arena_mean["arena_margin_effect"] = arena_mean["arena_margin_effect"] * (arena_mean["n"] / (arena_mean["n"] + 25))
    arena_mean["arena_shot_volume_effect"] = arena_mean["arena_shot_volume_effect"] * (arena_mean["n"] / (arena_mean["n"] + 25))
    return arena_mean.reset_index()[["venue", "arena_margin_effect", "arena_shot_volume_effect"]]


def _to_game_level(team_games: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    home = team_games[team_games["is_home"] == 1].copy()
    away = team_games[team_games["is_home"] == 0].copy()

    home_cols = [c for c in home.columns if c not in {"opponent", "won"}]
    away_cols = [c for c in away.columns if c not in {"opponent", "won"}]
    home = home[home_cols].add_prefix("home_")
    away = away[away_cols].add_prefix("away_")
    home = home.rename(columns={"home_game_id": "game_id"})
    away = away.rename(columns={"away_game_id": "game_id"})

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
    merged = games_df[base_cols].copy()
    merged = merged.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
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

    diff_pairs = [
        ("ewm_point_margin", "form_point_margin"),
        ("win_rate_ewm", "form_win_rate"),
        ("ewm_shot_volume_share", "shot_volume_share"),
        ("ewm_scoring_efficiency_proxy", "scoring_efficiency"),
        ("ewm_foul_margin", "foul_margin"),
        ("roster_depth_index", "roster_depth"),
        ("availability_uncertainty", "availability_uncertainty"),
        ("availability_reliability", "availability_reliability"),
    ]
    for src, name in diff_pairs:
        hc = f"home_{src}"
        ac = f"away_{src}"
        if hc in merged.columns and ac in merged.columns:
            merged[f"diff_{name}"] = merged[hc] - merged[ac]

    for col in [
        "home_fouls_drawn",
        "away_fouls_drawn",
        "home_fouls_committed",
        "away_fouls_committed",
        "home_field_goal_attempts_for",
        "away_field_goal_attempts_for",
    ]:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0)

    merged["discipline_free_throw_pressure_diff"] = (
        merged["home_ewm_free_throw_pressure"].fillna(0) - merged["away_ewm_free_throw_pressure"].fillna(0)
    )
    merged["discipline_foul_margin_diff"] = merged["home_ewm_foul_margin"].fillna(0) - merged["away_ewm_foul_margin"].fillna(0)
    merged["discipline_foul_pressure_diff"] = (
        merged["away_fouls_committed"].fillna(0) - merged["home_fouls_committed"].fillna(0)
    )
    merged["availability_depth_diff"] = merged["home_roster_depth_index"].fillna(1) - merged["away_roster_depth_index"].fillna(1)
    merged["availability_stress_diff"] = merged["home_absence_load_proxy"].fillna(0) - merged["away_absence_load_proxy"].fillna(0)

    total_fga = (merged["home_field_goal_attempts_for"] + merged["away_field_goal_attempts_for"]).replace(0, np.nan)
    merged["target_shot_volume_share"] = (merged["home_field_goal_attempts_for"] / total_fga).fillna(0.5)
    merged["target_free_throw_pressure"] = merged["home_free_throws_made"].fillna(0) - merged["away_free_throws_made"].fillna(0)
    merged["target_possession_volume"] = (
        merged["home_field_goal_attempts_for"] + merged["away_field_goal_attempts_for"]
    ).fillna(0)

    travel = build_travel_features(games_df, league="NBA")
    if not travel.empty:
        merged = merged.merge(travel, on="game_id", how="left")
        for stem in ["home_rest_days", "home_b2b", "away_rest_days", "away_b2b"]:
            left = f"{stem}_x"
            right = f"{stem}_y"
            if left in merged.columns or right in merged.columns:
                merged[stem] = merged.get(right, pd.Series(index=merged.index, dtype=float)).fillna(
                    merged.get(left, pd.Series(index=merged.index, dtype=float))
                )
        merged = merged.drop(
            columns=[c for c in ["home_rest_days_x", "home_rest_days_y", "home_b2b_x", "home_b2b_y", "away_rest_days_x", "away_rest_days_y", "away_b2b_x", "away_b2b_y"] if c in merged.columns]
        )

    arena = _compute_arena_effects(merged[merged["status_final"] == 1])
    merged = merged.merge(arena, on="venue", how="left")
    merged["arena_margin_effect"] = merged["arena_margin_effect"].fillna(0)
    merged["arena_shot_volume_effect"] = merged["arena_shot_volume_effect"].fillna(0)

    merged["fallback_shot_profile_proxy_used"] = 1
    merged["fallback_availability_proxy_used"] = 1

    direct_event_cols = [
        "home_home_score",
        "home_away_score",
        "away_home_score",
        "away_away_score",
        "home_points_for",
        "away_points_for",
        "home_points_against",
        "away_points_against",
        "home_point_margin",
        "away_point_margin",
        "home_field_goal_attempts_for",
        "away_field_goal_attempts_for",
        "home_field_goal_attempts_against",
        "away_field_goal_attempts_against",
        "home_fouls_committed",
        "away_fouls_committed",
        "home_fouls_drawn",
        "away_fouls_drawn",
        "home_foul_margin",
        "away_foul_margin",
        "home_free_throws_made",
        "away_free_throws_made",
        "home_shot_volume_share",
        "away_shot_volume_share",
        "home_free_throw_pressure",
        "away_free_throw_pressure",
        "home_scoring_efficiency_proxy",
        "away_scoring_efficiency_proxy",
        "home_possession_proxy",
        "away_possession_proxy",
        "home_home_win",
        "away_home_win",
        "home_status_final",
        "away_status_final",
    ]
    existing_drop = [c for c in direct_event_cols if c in merged.columns]
    if existing_drop:
        merged = merged.drop(columns=existing_drop)

    return merged


def build_nba_features_from_interim(interim_dir: str, processed_dir: str) -> FeatureBuildResult:
    games = _load("games", interim_dir)
    boxscore_stats = _load("goalies", interim_dir)
    players = _load("players", interim_dir)
    injuries = _load("injuries", interim_dir)

    games = games.sort_values("start_time_utc").reset_index(drop=True)
    team_games = _expand_team_games(games, boxscore_stats)
    team_games = _team_rolling(team_games, players_df=players, injuries_df=injuries)
    game_features = _to_game_level(team_games, games)

    elo = compute_elo_features(games)
    dyn = compute_dynamic_rating_features(games)
    game_features = game_features.merge(elo, on="game_id", how="left").merge(dyn, on="game_id", how="left")

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

    for c in feature_columns:
        game_features[c] = pd.to_numeric(game_features[c], errors="coerce")
    game_features[feature_columns] = game_features[feature_columns].replace([np.inf, -np.inf], np.nan)
    game_features[feature_columns] = game_features[feature_columns].fillna(
        game_features[feature_columns].median(numeric_only=True)
    ).fillna(0)

    feature_set_version = f"fset_{stable_hash({'league': 'NBA', 'n_features': len(feature_columns), 'cols': feature_columns})}"
    metadata = {
        "feature_set_version": feature_set_version,
        "built_at_utc": utc_now_iso(),
        "league": "NBA",
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
