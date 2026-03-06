from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.time import utc_now_iso
from src.common.utils import stable_hash
from src.features.base import FeatureBuildResult
from src.features.build_features_nba import build_nba_features_from_interim
from src.features.dynamic_ratings import compute_dynamic_rating_features
from src.features.elo import compute_elo_features
from src.features.goalie_features import add_goalie_features, combine_goalie_game_features
from src.features.intermediates import add_intermediate_targets
from src.features.rink_adjustments import compute_rink_effects
from src.features.special_teams import add_special_teams_features, combine_special_teams_game_features
from src.features.travel import build_travel_features

NHL_GLM_HINGE_KNOTS = {
    "diff_form_goal_diff": (-1.0, 1.0),
    "dyn_home_prob": 0.55,
    "dyn_home_mean": 0.0,
    "elo_home_prob": 0.54,
}


def _load(name: str, interim_dir: str) -> pd.DataFrame:
    p = Path(interim_dir) / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    csv_p = Path(interim_dir) / f"{name}.csv"
    if csv_p.exists():
        return pd.read_csv(csv_p)
    return pd.DataFrame()


def _expand_team_games(games: pd.DataFrame, goalie_stats: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    summary_rows = goalie_stats[goalie_stats["goalie_id"].isna()].copy() if not goalie_stats.empty else pd.DataFrame()
    starter_rows = goalie_stats[(goalie_stats["starter_status"] == "confirmed") & goalie_stats["goalie_id"].notna()].copy() if not goalie_stats.empty else pd.DataFrame()

    if not summary_rows.empty:
        summary = (
            summary_rows.groupby(["game_id", "team"], dropna=False)
            .agg(
                shots_for=("shots_for", "max"),
                shots_against=("shots_against", "max"),
                penalties_taken=("penalties_taken", "max"),
                penalties_drawn=("penalties_drawn", "max"),
                pp_goals=("pp_goals", "max"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["game_id", "team", "shots_for", "shots_against", "penalties_taken", "penalties_drawn", "pp_goals"])

    if not starter_rows.empty:
        starters = (
            starter_rows.groupby(["game_id", "team"], dropna=False)
            .agg(
                starter_goalie_id=("goalie_id", "first"),
                starter_name=("goalie_name", "first"),
                starter_save_pct=("save_pct", "mean"),
                starter_status=("starter_status", "first"),
            )
            .reset_index()
        )
    else:
        starters = pd.DataFrame(columns=["game_id", "team", "starter_goalie_id", "starter_name", "starter_save_pct", "starter_status"])

    team_extra = summary.merge(starters, on=["game_id", "team"], how="outer")

    home = games[
        [
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
    ].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = 1
    home["goals_for"] = home["home_score"]
    home["goals_against"] = home["away_score"]
    home["won"] = home["home_win"]

    away = games[
        [
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
    ].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = 0
    away["goals_for"] = away["away_score"]
    away["goals_against"] = away["home_score"]
    away["won"] = np.where(away["home_win"].isna(), np.nan, 1 - away["home_win"].astype(float))

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.merge(team_extra, on=["game_id", "team"], how="left")
    team_games["starter_status"] = team_games["starter_status"].fillna("unknown")
    return team_games


def _positive_part(series: pd.Series, knot: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values - float(knot)).clip(lower=0.0)


def _add_nhl_glm_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "diff_xg_share" in out.columns:
        xg_share = pd.to_numeric(out["diff_xg_share"], errors="coerce")
        out["diff_xg_share_cubic"] = xg_share.pow(3)

    if "diff_form_goal_diff" in out.columns:
        left_knot, right_knot = NHL_GLM_HINGE_KNOTS["diff_form_goal_diff"]
        out["diff_form_goal_diff_hinge_m1"] = _positive_part(out["diff_form_goal_diff"], left_knot)
        out["diff_form_goal_diff_hinge_p1"] = _positive_part(out["diff_form_goal_diff"], right_knot)

    if "dyn_home_prob" in out.columns:
        out["dyn_home_prob_hinge_055"] = _positive_part(out["dyn_home_prob"], NHL_GLM_HINGE_KNOTS["dyn_home_prob"])

    if "dyn_home_mean" in out.columns:
        out["dyn_home_mean_hinge_000"] = _positive_part(out["dyn_home_mean"], NHL_GLM_HINGE_KNOTS["dyn_home_mean"])

    if "elo_home_prob" in out.columns:
        out["elo_home_prob_hinge_054"] = _positive_part(out["elo_home_prob"], NHL_GLM_HINGE_KNOTS["elo_home_prob"])

    return out


def _team_rolling(team_games: pd.DataFrame, players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return team_games

    lineup_strength = pd.DataFrame(columns=["team", "roster_strength_index", "lineup_uncertainty"])
    if not players_df.empty:
        p = players_df.copy()
        p["games_played"] = pd.to_numeric(p["games_played"], errors="coerce").fillna(0)
        p["points"] = pd.to_numeric(p["points"], errors="coerce").fillna(0)
        p = p[p["games_played"] > 0]
        if not p.empty:
            p["pts_per_game"] = p["points"] / p["games_played"]
            lineup_strength = p.sort_values("pts_per_game", ascending=False).groupby("team", as_index=False).head(6)
            lineup_strength = lineup_strength.groupby("team", as_index=False).agg(roster_strength_index=("pts_per_game", "mean"))
            lineup_strength["lineup_uncertainty"] = 0

    if not injuries_df.empty:
        inj = injuries_df[["team", "lineup_uncertainty", "man_games_lost_proxy"]].copy()
    else:
        inj = pd.DataFrame(columns=["team", "lineup_uncertainty", "man_games_lost_proxy"])

    team_meta = lineup_strength.merge(inj, on="team", how="outer")
    if "lineup_uncertainty_x" in team_meta.columns:
        team_meta["lineup_uncertainty"] = team_meta["lineup_uncertainty_x"].fillna(team_meta["lineup_uncertainty_y"])
        team_meta = team_meta.drop(columns=["lineup_uncertainty_x", "lineup_uncertainty_y"])
    team_meta["roster_strength_index"] = team_meta.get("roster_strength_index", pd.Series(dtype=float)).fillna(0.1)
    team_meta["lineup_uncertainty"] = team_meta.get("lineup_uncertainty", pd.Series(dtype=float)).fillna(1)
    team_meta["man_games_lost_proxy"] = team_meta.get("man_games_lost_proxy", pd.Series(dtype=float)).fillna(0)

    df = team_games.sort_values(["team", "start_time_utc"]).copy()
    df["goal_diff"] = df["goals_for"].fillna(0) - df["goals_against"].fillna(0)
    df["shots_for"] = df["shots_for"].fillna(df["goals_for"] * 6 + 25)
    df["shots_against"] = df["shots_against"].fillna(df["goals_against"] * 6 + 25)
    df["team_save_pct_proxy"] = 1 - (df["goals_against"].fillna(0) / df["shots_against"].replace(0, np.nan))
    df["team_save_pct_proxy"] = df["team_save_pct_proxy"].fillna(0.905)

    def _roll(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.copy()
        g["start_dt"] = pd.to_datetime(g["start_time_utc"], errors="coerce")
        g["prev_start_dt"] = g["start_dt"].shift(1)
        g["rest_days"] = (g["start_dt"] - g["prev_start_dt"]).dt.total_seconds() / 86400
        g["rest_days"] = g["rest_days"].fillna(7).clip(lower=0)
        g["b2b"] = (g["rest_days"] <= 1.1).astype(int)
        g["games_played_prior"] = range(len(g))
        for col in ["goals_for", "goals_against", "goal_diff", "shots_for", "shots_against", "team_save_pct_proxy"]:
            g[f"ewm_{col}"] = g[col].shift(1).ewm(alpha=0.2, adjust=False).mean()
            g[f"r5_{col}"] = g[col].shift(1).rolling(5, min_periods=1).mean()
            g[f"r14_{col}"] = g[col].shift(1).rolling(14, min_periods=1).mean()
        g["win_rate_ewm"] = g["won"].shift(1).ewm(alpha=0.2, adjust=False).mean()
        g["xg_available"] = 0
        g["xg_for_ewm"] = g["ewm_shots_for"] / 10.0
        g["xg_against_ewm"] = g["ewm_shots_against"] / 10.0
        g["xg_share_ewm"] = g["xg_for_ewm"] / (g["xg_for_ewm"] + g["xg_against_ewm"]).replace(0, np.nan)
        g["xg_share_ewm"] = g["xg_share_ewm"].fillna(0.5)
        return g

    df = df.groupby("team", group_keys=False).apply(_roll)
    df = add_special_teams_features(df)
    df = add_goalie_features(df)
    df = add_intermediate_targets(df)

    season_start = pd.to_datetime(df["game_date_utc"]).min()
    d = pd.to_datetime(df["game_date_utc"])
    df["days_into_season"] = (d - season_start).dt.days.fillna(0)
    df["days_into_season_spline"] = np.sqrt(df["days_into_season"].clip(lower=0))
    df["season_phase"] = pd.cut(
        df["days_into_season"],
        bins=[-1, 45, 120, 1000],
        labels=["early", "mid", "late"],
    ).astype(str)
    df["season_phase_early"] = (df["season_phase"] == "early").astype(int)
    df["season_phase_mid"] = (df["season_phase"] == "mid").astype(int)
    df["season_phase_late"] = (df["season_phase"] == "late").astype(int)
    df["post_trade_deadline"] = (pd.to_datetime(df["game_date_utc"]).dt.month >= 3).astype(int)
    df["coaching_change_indicator"] = 0

    df = df.merge(team_meta, on="team", how="left")
    df["roster_strength_index"] = df["roster_strength_index"].fillna(0.1)
    df["lineup_uncertainty"] = df["lineup_uncertainty"].fillna(1)
    df["man_games_lost_proxy"] = df["man_games_lost_proxy"].fillna(0)
    return df


def _to_game_level(team_games: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    home = team_games[team_games["is_home"] == 1].copy()
    away = team_games[team_games["is_home"] == 0].copy()

    home_cols = [c for c in home.columns if c not in {"opponent", "won"}]
    away_cols = [c for c in away.columns if c not in {"opponent", "won"}]
    home = home[home_cols].add_prefix("home_")
    away = away[away_cols].add_prefix("away_")

    home = home.rename(columns={"home_game_id": "game_id"})
    away = away.rename(columns={"away_game_id": "game_id"})

    base = games_df[
        [
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
    ].copy()

    merged = base.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
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

    # Convenience diffs.
    diff_pairs = [
        ("ewm_goal_diff", "form_goal_diff"),
        ("win_rate_ewm", "form_win_rate"),
        ("xg_share_ewm", "xg_share"),
        ("ewm_shots_for", "shots_for"),
        ("ewm_shots_against", "shots_against"),
        ("penalty_diff_ewm", "penalty_diff"),
        ("roster_strength_index", "roster_strength"),
        ("lineup_uncertainty", "lineup_uncertainty"),
    ]
    for src, name in diff_pairs:
        hc = f"home_{src}"
        ac = f"away_{src}"
        if hc in merged.columns and ac in merged.columns:
            merged[f"diff_{name}"] = merged[hc] - merged[ac]

    merged["home_shots_for"] = merged["home_shots_for"].fillna(0)
    merged["away_shots_for"] = merged["away_shots_for"].fillna(0)
    merged["home_penalties_drawn"] = merged["home_penalties_drawn"].fillna(0)
    merged["away_penalties_drawn"] = merged["away_penalties_drawn"].fillna(0)
    merged["home_penalties_taken"] = merged["home_penalties_taken"].fillna(0)
    merged["away_penalties_taken"] = merged["away_penalties_taken"].fillna(0)

    total_shots = (merged["home_shots_for"] + merged["away_shots_for"]).replace(0, np.nan)
    merged["target_xg_share"] = (merged["home_shots_for"] / total_shots).fillna(0.5)
    merged["target_penalty_diff"] = (merged["home_penalties_drawn"] - merged["home_penalties_taken"]) - (
        merged["away_penalties_drawn"] - merged["away_penalties_taken"]
    )
    merged["target_pace"] = (merged["home_shots_for"] + merged["away_shots_for"]).fillna(0)

    merged = combine_special_teams_game_features(merged)
    merged = combine_goalie_game_features(merged)

    travel = build_travel_features(base)
    if not travel.empty:
        merged = merged.merge(travel, on="game_id", how="left")

    rink = compute_rink_effects(merged[merged["status_final"] == 1])
    merged = merged.merge(rink, on="venue", how="left")
    merged["rink_goal_effect"] = merged["rink_goal_effect"].fillna(0)
    merged["rink_shot_effect"] = merged["rink_shot_effect"].fillna(0)

    merged["fallback_xg_proxy_used"] = 1
    merged["fallback_goalie_unknown"] = (
        merged["home_goalie_uncertainty_feature"].fillna(1) + merged["away_goalie_uncertainty_feature"].fillna(1) > 0
    ).astype(int)
    merged["fallback_lineup_proxy_used"] = 1

    # Drop direct single-game outcome columns from feature frame to enforce temporal integrity.
    direct_outcome_cols = [
        "home_home_score",
        "home_away_score",
        "away_home_score",
        "away_away_score",
        "home_goal_diff",
        "away_goal_diff",
        "home_home_win",
        "away_home_win",
        "home_status_final",
        "away_status_final",
        "home_goals_for",
        "away_goals_for",
        "home_goals_against",
        "away_goals_against",
    ]
    existing_drop = [c for c in direct_outcome_cols if c in merged.columns]
    if existing_drop:
        merged = merged.drop(columns=existing_drop)

    return merged


def build_features_from_interim(interim_dir: str, processed_dir: str, league: str = "NHL") -> FeatureBuildResult:
    if str(league or "NHL").strip().upper() == "NBA":
        return build_nba_features_from_interim(interim_dir=interim_dir, processed_dir=processed_dir)

    games = _load("games", interim_dir)
    goalies = _load("goalies", interim_dir)
    players = _load("players", interim_dir)
    injuries = _load("injuries", interim_dir)

    games = games.sort_values("start_time_utc").reset_index(drop=True)
    team_games = _expand_team_games(games, goalies)
    team_games = _team_rolling(team_games, players_df=players, injuries_df=injuries)
    game_features = _to_game_level(team_games, games)

    elo = compute_elo_features(games)
    dyn = compute_dynamic_rating_features(games)
    game_features = game_features.merge(elo, on="game_id", how="left").merge(dyn, on="game_id", how="left")
    game_features = _add_nhl_glm_transforms(game_features)

    # Feature columns exclude identifiers and labels.
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
    game_features[feature_columns] = game_features[feature_columns].fillna(game_features[feature_columns].median(numeric_only=True)).fillna(0)

    feature_set_version = f"fset_{stable_hash({'n_features': len(feature_columns), 'cols': feature_columns})}"
    metadata = {
        "feature_set_version": feature_set_version,
        "built_at_utc": utc_now_iso(),
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
