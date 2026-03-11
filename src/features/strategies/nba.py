"""NBA-specific feature transforms for the shared pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.contextual_effects import compute_causal_group_effects
from src.features.dynamic_ratings import compute_dynamic_rating_features
from src.features.elo import compute_elo_features
from src.features.strategies.base import BaseFeatureStrategy
from src.features.travel import build_travel_features


NBA_GLM_HINGE_KNOTS = {
    "diff_darko_like_total": 0.0,
    "discipline_foul_margin_diff": 0.0,
    "discipline_free_throw_pressure_diff": 0.0,
    "diff_form_point_margin": 0.0,
    "diff_shot_volume_share": 0.01,
    "elo_home_prob": 0.55,
}


def _positive_part(series: pd.Series, knot: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values - float(knot)).clip(lower=0.0)


def _shrink_to_mean(series: pd.Series, counts: pd.Series, prior_mean: float, k: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    n = pd.to_numeric(counts, errors="coerce").fillna(0.0)
    numer = values.fillna(prior_mean) * n + float(prior_mean) * float(k)
    denom = n + float(k)
    return numer / denom.replace(0, np.nan)


def _status_to_minutes_factor(status: str | None) -> float:
    token = str(status or "").strip().lower()
    if not token:
        return 1.0
    if "out" in token or "suspend" in token:
        return 0.0
    if "doubt" in token:
        return 0.15
    if "question" in token:
        return 0.45
    if "day-to-day" in token or "day to day" in token:
        return 0.65
    if "probable" in token:
        return 0.9
    return 1.0


def _prepare_player_projection_history(players_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "player_id",
        "team",
        "start_time_utc",
        "minutes",
        "points",
        "assists",
        "rebounds_offensive",
        "rebounds_defensive",
        "steals",
        "blocks",
        "turnovers",
        "fouls_personal",
        "field_goals_made",
        "field_goals_attempted",
        "free_throws_made",
        "free_throws_attempted",
        "three_pointers_made",
        "plus_minus_points",
    }
    if players_df.empty or not required.issubset(players_df.columns):
        return pd.DataFrame()

    p = players_df.copy()
    p["start_dt"] = pd.to_datetime(p["start_time_utc"], errors="coerce")
    p = p[p["player_id"].notna() & p["start_dt"].notna()].copy()
    if p.empty:
        return p

    numeric_cols = [
        "minutes",
        "points",
        "assists",
        "rebounds_offensive",
        "rebounds_defensive",
        "rebounds_total",
        "steals",
        "blocks",
        "turnovers",
        "fouls_personal",
        "field_goals_made",
        "field_goals_attempted",
        "free_throws_made",
        "free_throws_attempted",
        "three_pointers_made",
        "plus_minus_points",
        "played",
        "starter",
    ]
    for col in numeric_cols:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    p["minutes"] = p["minutes"].fillna(0.0).clip(lower=0.0)
    p["played"] = p["played"].fillna((p["minutes"] > 0).astype(float)).clip(lower=0.0, upper=1.0)
    p["starter"] = p["starter"].fillna(0.0).clip(lower=0.0, upper=1.0)

    p["off_box_score_raw"] = (
        p["points"].fillna(0.0)
        + 0.4 * p["field_goals_made"].fillna(0.0)
        - 0.7 * p["field_goals_attempted"].fillna(0.0)
        - 0.4 * (p["free_throws_attempted"].fillna(0.0) - p["free_throws_made"].fillna(0.0))
        + 0.7 * p["rebounds_offensive"].fillna(0.0)
        + 0.7 * p["assists"].fillna(0.0)
        + 0.3 * p["three_pointers_made"].fillna(0.0)
        - p["turnovers"].fillna(0.0)
    )
    p["def_box_score_raw"] = (
        0.3 * p["rebounds_defensive"].fillna(0.0)
        + p["steals"].fillna(0.0)
        + 0.7 * p["blocks"].fillna(0.0)
        - 0.4 * p["fouls_personal"].fillna(0.0)
    )
    p["overall_box_score_raw"] = p["off_box_score_raw"] + p["def_box_score_raw"] + 0.08 * p["plus_minus_points"].fillna(0.0)

    minutes_denom = p["minutes"].clip(lower=6.0)
    for raw_col, out_col in [
        ("off_box_score_raw", "offense_per36"),
        ("def_box_score_raw", "defense_per36"),
        ("overall_box_score_raw", "overall_per36"),
    ]:
        p[out_col] = p[raw_col] * 36.0 / minutes_denom

    league_means = {
        "offense": float(p["offense_per36"].mean()) if not p.empty else 0.0,
        "defense": float(p["defense_per36"].mean()) if not p.empty else 0.0,
        "overall": float(p["overall_per36"].mean()) if not p.empty else 0.0,
    }
    league_minutes_mean = float(p.loc[p["minutes"] > 0, "minutes"].mean()) if (p["minutes"] > 0).any() else 18.0

    def _per_player(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("start_dt").copy()
        g["player_id"] = group.name
        g["games_played_prior"] = np.arange(len(g), dtype=float)
        for source_col, stem in [("offense_per36", "offense"), ("defense_per36", "defense"), ("overall_per36", "overall")]:
            g[f"ewm_{stem}_per36"] = g[source_col].shift(1).ewm(alpha=0.25, adjust=False).mean()
            g[f"r5_{stem}_per36"] = g[source_col].shift(1).rolling(5, min_periods=1).mean()
        g["ewm_minutes"] = g["minutes"].shift(1).ewm(alpha=0.35, adjust=False).mean()
        g["r5_minutes"] = g["minutes"].shift(1).rolling(5, min_periods=1).mean()
        g["start_rate_ewm"] = g["starter"].shift(1).ewm(alpha=0.3, adjust=False).mean()
        g["played_rate_ewm"] = g["played"].shift(1).ewm(alpha=0.3, adjust=False).mean()
        return g

    p = p.groupby("player_id", group_keys=False).apply(_per_player, include_groups=False)
    for stem, prior_mean in league_means.items():
        blended = 0.7 * p[f"ewm_{stem}_per36"].fillna(prior_mean) + 0.3 * p[f"r5_{stem}_per36"].fillna(prior_mean)
        p[f"{stem}_projection"] = _shrink_to_mean(blended, p["games_played_prior"], prior_mean, k=8.0)

    blended_minutes = 0.65 * p["ewm_minutes"].fillna(league_minutes_mean) + 0.35 * p["r5_minutes"].fillna(league_minutes_mean) + 4.0 * p["start_rate_ewm"].fillna(0.0)
    p["minutes_projection"] = _shrink_to_mean(blended_minutes, p["games_played_prior"], league_minutes_mean, k=6.0)
    p["minutes_projection"] = p["minutes_projection"].clip(lower=0.0, upper=42.0)
    p["projection_confidence"] = (p["games_played_prior"] / (p["games_played_prior"] + 6.0)).fillna(0.0)

    for col, lower, upper in [("offense_projection", -18.0, 18.0), ("defense_projection", -12.0, 12.0), ("overall_projection", -24.0, 24.0)]:
        p[col] = p[col].clip(lower=lower, upper=upper)
    return p


def _player_projection_meta(team_games: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    projections = _prepare_player_projection_history(players_df)
    if team_games.empty or projections.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team",
                "roster_depth_index",
                "availability_uncertainty",
                "availability_reliability",
                "absence_load_proxy",
                "darko_like_total",
                "darko_like_offense",
                "darko_like_defense",
                "projected_minutes_known_share",
                "projected_absence_pressure",
                "rotation_top6_share",
                "rotation_stability",
                "player_projection_confidence",
            ]
        )

    rows: list[dict] = []
    team_schedule = team_games.sort_values(["team", "start_time_utc"]).copy()
    team_schedule["start_dt"] = pd.to_datetime(team_schedule["start_time_utc"], errors="coerce")

    for team, games_grp in team_schedule.groupby("team", sort=False):
        hist_pool = projections[projections["team"] == team].copy()
        future_pool = projections[projections["current_team"].fillna(projections["team"]) == team].copy()
        if hist_pool.empty and future_pool.empty:
            continue

        for game in games_grp.itertuples(index=False):
            game_dt = getattr(game, "start_dt")
            if pd.isna(game_dt):
                continue

            is_upcoming = pd.isna(getattr(game, "won"))
            pool = future_pool if is_upcoming and not future_pool.empty else hist_pool
            if pool.empty:
                continue

            prior = pool[pool["start_dt"] < game_dt].copy()
            if prior.empty:
                rows.append(
                    {
                        "game_id": game.game_id,
                        "team": team,
                        "roster_depth_index": 0.0,
                        "availability_uncertainty": 1.0,
                        "availability_reliability": 0.0,
                        "absence_load_proxy": 1.0,
                        "darko_like_total": 0.0,
                        "darko_like_offense": 0.0,
                        "darko_like_defense": 0.0,
                        "projected_minutes_known_share": 0.0,
                        "projected_absence_pressure": 1.0,
                        "rotation_top6_share": 0.0,
                        "rotation_stability": 0.0,
                        "player_projection_confidence": 0.0,
                    }
                )
                continue

            latest = prior.groupby("player_id", as_index=False).tail(1).copy()
            latest["days_since_last_game"] = (game_dt - latest["start_dt"]).dt.total_seconds() / 86400.0
            gap = latest["days_since_last_game"].clip(lower=0.0)
            latest["recency_factor"] = np.where(gap <= 7.0, 1.0, np.exp(-(gap - 7.0) / 14.0))
            latest["availability_factor"] = latest["played_rate_ewm"].fillna(0.6).clip(lower=0.1, upper=1.0)
            latest["current_status_factor"] = 1.0
            if is_upcoming and "current_injury_status" in latest.columns:
                latest["current_status_factor"] = latest["current_injury_status"].map(_status_to_minutes_factor).fillna(1.0)

            latest["expected_minutes"] = (
                latest["minutes_projection"].fillna(0.0).clip(lower=0.0, upper=42.0)
                * latest["availability_factor"]
                * latest["recency_factor"]
                * latest["current_status_factor"]
            )
            latest = latest[latest["expected_minutes"] > 0.5].copy()
            if latest.empty:
                continue

            latest = latest.sort_values("expected_minutes", ascending=False).head(12)
            total_known_minutes = float(latest["expected_minutes"].sum())
            if total_known_minutes > 240.0:
                latest["expected_minutes"] = latest["expected_minutes"] * (240.0 / total_known_minutes)
                total_known_minutes = 240.0

            known_share = total_known_minutes / 240.0
            projected_absence_pressure = max(0.0, 1.0 - known_share)
            rotation_players = int((latest["expected_minutes"] >= 8.0).sum())
            roster_depth_index = (rotation_players / 8.0) * known_share
            weighted_minutes = max(total_known_minutes, 1.0)
            player_projection_confidence = float((latest["projection_confidence"].fillna(0.0) * latest["expected_minutes"]).sum() / weighted_minutes)
            recency_stability = 1.0 - latest["days_since_last_game"].clip(lower=0.0, upper=21.0) / 21.0
            rotation_stability = float((recency_stability * latest["expected_minutes"]).sum() / weighted_minutes)
            availability_reliability = float(known_share * player_projection_confidence)
            availability_uncertainty = float(min(max(1.0 - availability_reliability, 0.0), 1.0))

            rows.append(
                {
                    "game_id": game.game_id,
                    "team": team,
                    "roster_depth_index": roster_depth_index,
                    "availability_uncertainty": availability_uncertainty,
                    "availability_reliability": availability_reliability,
                    "absence_load_proxy": projected_absence_pressure,
                    "darko_like_total": float((latest["overall_projection"].fillna(0.0) * latest["expected_minutes"]).sum() / 240.0),
                    "darko_like_offense": float((latest["offense_projection"].fillna(0.0) * latest["expected_minutes"]).sum() / 240.0),
                    "darko_like_defense": float((latest["defense_projection"].fillna(0.0) * latest["expected_minutes"]).sum() / 240.0),
                    "projected_minutes_known_share": known_share,
                    "projected_absence_pressure": projected_absence_pressure,
                    "rotation_top6_share": float(latest.nlargest(6, "expected_minutes")["expected_minutes"].sum() / 240.0),
                    "rotation_stability": rotation_stability,
                    "player_projection_confidence": player_projection_confidence,
                }
            )

    return pd.DataFrame(rows)


def _compute_arena_effects(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(columns=["game_id", "arena_margin_effect", "arena_shot_volume_effect"])

    tmp = games_df.copy()
    tmp["point_margin"] = tmp["home_score"].fillna(0) - tmp["away_score"].fillna(0)
    tmp["shot_volume_diff"] = tmp["home_field_goal_attempts_for"].fillna(0) - tmp["away_field_goal_attempts_for"].fillna(0)
    # This must stay causal: each row can only see prior finalized games at the
    # same arena. Using full-season venue means lets future home dominance leak
    # backward and rewrite historical "expected" probabilities.
    return compute_causal_group_effects(
        tmp,
        group_col="venue",
        metric_columns={
            "arena_margin_effect": "point_margin",
            "arena_shot_volume_effect": "shot_volume_diff",
        },
        shrinkage=25.0,
    )


class NbaFeatureStrategy(BaseFeatureStrategy):
    def __init__(self) -> None:
        super().__init__(
            league="NBA",
            summary_aggregations={
                "field_goal_attempts_for": ("shots_for", "max"),
                "field_goal_attempts_against": ("shots_against", "max"),
                "fouls_committed": ("penalties_taken", "max"),
                "fouls_drawn": ("penalties_drawn", "max"),
                "free_throws_made": ("pp_goals", "max"),
            },
            starter_aggregations={},
            team_value_for_column="points_for",
            team_value_against_column="points_against",
            team_result_column="won",
            rolling_value_columns=[
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
            ],
            diff_pairs=[
                ("ewm_point_margin", "form_point_margin"),
                ("win_rate_ewm", "form_win_rate"),
                ("ewm_shot_volume_share", "shot_volume_share"),
                ("ewm_scoring_efficiency_proxy", "scoring_efficiency"),
                ("ewm_foul_margin", "foul_margin"),
                ("darko_like_total", "darko_like_total"),
                ("projected_minutes_known_share", "projected_minutes_known_share"),
                ("projected_absence_pressure", "projected_absence_pressure"),
                ("rotation_top6_share", "rotation_top6_share"),
                ("rotation_stability", "rotation_stability"),
                ("player_projection_confidence", "player_projection_confidence"),
                ("roster_depth_index", "roster_depth"),
                ("availability_uncertainty", "availability_uncertainty"),
                ("availability_reliability", "availability_reliability"),
            ],
            direct_event_drop_columns=[
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
            ],
        )

    def prepare_team_games(self, team_games: pd.DataFrame, players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
        df = team_games.sort_values(["team", "start_time_utc"]).copy()
        meta = _player_projection_meta(team_games=team_games, players_df=players_df)
        df = df.merge(meta, on=["game_id", "team"], how="left")
        df["point_margin"] = df["points_for"].fillna(0) - df["points_against"].fillna(0)
        current_shot_profile_proxy_used = (
            pd.to_numeric(df.get("field_goal_attempts_for", pd.Series(index=df.index, dtype=float)), errors="coerce").isna()
            | pd.to_numeric(df.get("field_goal_attempts_against", pd.Series(index=df.index, dtype=float)), errors="coerce").isna()
            | pd.to_numeric(df.get("free_throws_made", pd.Series(index=df.index, dtype=float)), errors="coerce").isna()
        ).astype(int)
        df["field_goal_attempts_for"] = df["field_goal_attempts_for"].fillna(df["points_for"] * 0.85)
        df["field_goal_attempts_against"] = df["field_goal_attempts_against"].fillna(df["points_against"] * 0.85)
        df["fouls_committed"] = df["fouls_committed"].fillna(20)
        df["fouls_drawn"] = df["fouls_drawn"].fillna(20)
        df["free_throws_made"] = df["free_throws_made"].fillna(df["points_for"] * 0.18)
        df["shot_volume_share"] = df["field_goal_attempts_for"] / (df["field_goal_attempts_for"] + df["field_goal_attempts_against"]).replace(0, np.nan)
        df["shot_volume_share"] = df["shot_volume_share"].fillna(0.5)
        df["free_throw_pressure"] = df["free_throws_made"] / df["field_goal_attempts_for"].replace(0, np.nan)
        df["free_throw_pressure"] = df["free_throw_pressure"].fillna(0.18)
        df["foul_margin"] = df["fouls_drawn"] - df["fouls_committed"]
        df["scoring_efficiency_proxy"] = df["points_for"] / df["field_goal_attempts_for"].replace(0, np.nan)
        df["scoring_efficiency_proxy"] = df["scoring_efficiency_proxy"].fillna(1.1)
        df["possession_proxy"] = df["field_goal_attempts_for"] + df["field_goal_attempts_against"]
        for col in [
            "roster_depth_index",
            "availability_uncertainty",
            "absence_load_proxy",
            "availability_reliability",
            "darko_like_total",
            "darko_like_offense",
            "darko_like_defense",
            "projected_minutes_known_share",
            "projected_absence_pressure",
            "rotation_top6_share",
            "rotation_stability",
            "player_projection_confidence",
        ]:
            df[col] = pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce").fillna(0.0 if col != "availability_uncertainty" else 1.0)
        df["shot_profile_proxy_used"] = current_shot_profile_proxy_used.groupby(df["team"], sort=False).shift(1).fillna(0).astype(int)
        return df

    def finalize_team_games(self, team_games: pd.DataFrame) -> pd.DataFrame:
        df = team_games.copy()
        season_dates = pd.to_datetime(df["game_date_utc"], errors="coerce")
        season_start = season_dates.min()
        df["days_into_season"] = (season_dates - season_start).dt.days.fillna(0)
        df["days_into_season_spline"] = np.sqrt(df["days_into_season"].clip(lower=0))
        df["season_phase"] = pd.cut(df["days_into_season"], bins=[-1, 45, 120, 1000], labels=["early", "mid", "late"]).astype(str)
        df["season_phase_early"] = (df["season_phase"] == "early").astype(int)
        df["season_phase_mid"] = (df["season_phase"] == "mid").astype(int)
        df["season_phase_late"] = (df["season_phase"] == "late").astype(int)
        df["post_all_star_break"] = (season_dates.dt.month >= 2).astype(int)
        df["post_trade_deadline"] = (season_dates.dt.month >= 2).astype(int)
        df["shot_profile_proxy_used"] = (
            pd.to_numeric(df.get("shot_profile_proxy_used", pd.Series(index=df.index, dtype=float)), errors="coerce").fillna(0).clip(lower=0, upper=1).astype(int)
        )
        return df

    def enrich_game_level(self, merged: pd.DataFrame, games_df: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
        out = merged.copy()
        for col in [
            "home_fouls_drawn",
            "away_fouls_drawn",
            "home_fouls_committed",
            "away_fouls_committed",
            "home_field_goal_attempts_for",
            "away_field_goal_attempts_for",
        ]:
            if col not in out.columns:
                out[col] = 0.0
            else:
                out[col] = out[col].fillna(0)

        out["discipline_free_throw_pressure_diff"] = out["home_ewm_free_throw_pressure"].fillna(0) - out["away_ewm_free_throw_pressure"].fillna(0)
        out["discipline_foul_margin_diff"] = out["home_ewm_foul_margin"].fillna(0) - out["away_ewm_foul_margin"].fillna(0)
        out["discipline_foul_pressure_diff"] = out["away_fouls_committed"].fillna(0) - out["home_fouls_committed"].fillna(0)
        out["availability_depth_diff"] = out["home_roster_depth_index"].fillna(1) - out["away_roster_depth_index"].fillna(1)
        out["availability_stress_diff"] = out["home_absence_load_proxy"].fillna(0) - out["away_absence_load_proxy"].fillna(0)
        out["diff_projected_absence_pressure"] = out["away_projected_absence_pressure"].fillna(1) - out["home_projected_absence_pressure"].fillna(1)
        out["darko_like_off_matchup"] = out["home_darko_like_offense"].fillna(0) - out["away_darko_like_defense"].fillna(0)
        out["darko_like_def_matchup"] = out["home_darko_like_defense"].fillna(0) - out["away_darko_like_offense"].fillna(0)

        total_fga = (out["home_field_goal_attempts_for"] + out["away_field_goal_attempts_for"]).replace(0, np.nan)
        out["target_shot_volume_share"] = (out["home_field_goal_attempts_for"] / total_fga).fillna(0.5)
        out["target_free_throw_pressure"] = out["home_free_throws_made"].fillna(0) - out["away_free_throws_made"].fillna(0)
        out["target_possession_volume"] = (out["home_field_goal_attempts_for"] + out["away_field_goal_attempts_for"]).fillna(0)

        travel = build_travel_features(games_df, league="NBA")
        if not travel.empty:
            out = out.merge(travel, on="game_id", how="left")
            for stem in ["home_rest_days", "home_b2b", "away_rest_days", "away_b2b"]:
                left = f"{stem}_x"
                right = f"{stem}_y"
                if left in out.columns or right in out.columns:
                    out[stem] = out.get(right, pd.Series(index=out.index, dtype=float)).fillna(out.get(left, pd.Series(index=out.index, dtype=float)))
            out = out.drop(columns=[c for c in ["home_rest_days_x", "home_rest_days_y", "home_b2b_x", "home_b2b_y", "away_rest_days_x", "away_rest_days_y", "away_b2b_x", "away_b2b_y"] if c in out.columns])

        arena = _compute_arena_effects(out)
        out = out.merge(arena, on="game_id", how="left")
        out["arena_margin_effect"] = out["arena_margin_effect"].fillna(0)
        out["arena_shot_volume_effect"] = out["arena_shot_volume_effect"].fillna(0)
        home_shot_profile_proxy = pd.to_numeric(
            out.get("home_shot_profile_proxy_used", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        ).fillna(0)
        away_shot_profile_proxy = pd.to_numeric(
            out.get("away_shot_profile_proxy_used", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        ).fillna(0)
        out["fallback_shot_profile_proxy_used"] = ((home_shot_profile_proxy > 0) | (away_shot_profile_proxy > 0)).astype(int)
        out["fallback_availability_proxy_used"] = (
            (out.get("home_player_projection_confidence", pd.Series(index=out.index, dtype=float)).fillna(0)
            + out.get("away_player_projection_confidence", pd.Series(index=out.index, dtype=float)).fillna(0))
            <= 0
        ).astype(int)

        elo = compute_elo_features(games_df)
        dyn = compute_dynamic_rating_features(games_df)
        return out.merge(elo, on="game_id", how="left").merge(dyn, on="game_id", how="left")

    def add_model_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "diff_darko_like_total" in out.columns:
            out["diff_darko_like_total_hinge_000"] = _positive_part(out["diff_darko_like_total"], NBA_GLM_HINGE_KNOTS["diff_darko_like_total"])
        if "discipline_foul_margin_diff" in out.columns:
            out["discipline_foul_margin_diff_hinge_000"] = _positive_part(out["discipline_foul_margin_diff"], NBA_GLM_HINGE_KNOTS["discipline_foul_margin_diff"])
        if "discipline_free_throw_pressure_diff" in out.columns:
            pressure_diff = pd.to_numeric(out["discipline_free_throw_pressure_diff"], errors="coerce")
            out["discipline_free_throw_pressure_diff_hinge_000"] = _positive_part(pressure_diff, NBA_GLM_HINGE_KNOTS["discipline_free_throw_pressure_diff"])
            out["discipline_free_throw_pressure_diff_is_zero"] = pressure_diff.eq(0).astype(float)
        if "elo_home_prob" in out.columns:
            out["elo_home_prob_hinge_055"] = _positive_part(out["elo_home_prob"], NBA_GLM_HINGE_KNOTS["elo_home_prob"])
        if "diff_shot_volume_share" in out.columns:
            out["diff_shot_volume_share_hinge_001"] = _positive_part(out["diff_shot_volume_share"], NBA_GLM_HINGE_KNOTS["diff_shot_volume_share"])
        if "diff_form_point_margin" in out.columns:
            out["diff_form_point_margin_hinge_000"] = _positive_part(out["diff_form_point_margin"], NBA_GLM_HINGE_KNOTS["diff_form_point_margin"])
        return out

    def feature_hash_payload(self, feature_columns: list[str]) -> dict:
        return {"league": self.league, "n_features": len(feature_columns), "cols": feature_columns}
