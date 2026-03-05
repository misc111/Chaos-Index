from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.common.config import AppConfig, load_config
from src.common.logging import get_logger, setup_logging
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir, to_json
from src.data_sources.base import HttpClient, SourceFetchResult
from src.evaluation.brier_decomposition import brier_decompose
from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.diagnostics_glm import save_glm_diagnostics
from src.evaluation.diagnostics_ml import permutation_importance_report
from src.evaluation.validation_backtest_integrity import run_backtest_integrity_checks
from src.evaluation.validation_fragility import missingness_stress_test, perturbation_sensitivity
from src.evaluation.validation_influence import influence_diagnostics
from src.evaluation.validation_significance import blockwise_nested_lrt
from src.evaluation.validation_stability import break_test_trade_deadline, coefficient_paths, vif_table
from src.features.build_features import build_features_from_interim
from src.storage.db import Database
from src.storage.tracker import RunTracker
from src.training.backtest import run_walk_forward_backtest
from src.training.feature_policy import apply_feature_policy
from src.training.prequential import score_predictions
from src.training.train import normalize_selected_models, select_feature_columns, train_and_predict

logger = get_logger(__name__)


INTERIM_FILES = {
    "games": "games.parquet",
    "schedule": "schedule.parquet",
    "teams": "teams.parquet",
    "players": "players.parquet",
    "goalies": "goalies.parquet",
    "injuries": "injuries.parquet",
    "odds": "odds.parquet",
    "xg": "xg.parquet",
}


def _canonical_league(league: str | None) -> str:
    token = str(league or "").strip().upper()
    if token in {"NHL", "NBA"}:
        return token
    raise ValueError(f"Unsupported league '{league}'. Expected one of: NHL, NBA.")


def _league_sources(league: str) -> dict[str, Any]:
    code = _canonical_league(league)
    if code == "NHL":
        from src.data_sources.nhl.games import fetch_games
        from src.data_sources.nhl.goalies import fetch_goalie_game_stats
        from src.data_sources.nhl.injuries import fetch_injuries_proxy
        from src.data_sources.nhl.odds import fetch_public_odds_optional
        from src.data_sources.nhl.players import fetch_players
        from src.data_sources.nhl.results import build_results_from_games
        from src.data_sources.nhl.schedule import fetch_upcoming_schedule
        from src.data_sources.nhl.teams import fetch_teams
        from src.data_sources.nhl.xg import fetch_xg_optional
    else:
        from src.data_sources.nba.games import fetch_games
        from src.data_sources.nba.goalies import fetch_goalie_game_stats
        from src.data_sources.nba.injuries import fetch_injuries_proxy
        from src.data_sources.nba.odds import fetch_public_odds_optional
        from src.data_sources.nba.players import fetch_players
        from src.data_sources.nba.results import build_results_from_games
        from src.data_sources.nba.schedule import fetch_upcoming_schedule
        from src.data_sources.nba.teams import fetch_teams
        from src.data_sources.nba.xg import fetch_xg_optional

    return {
        "fetch_games": fetch_games,
        "fetch_goalie_game_stats": fetch_goalie_game_stats,
        "fetch_injuries_proxy": fetch_injuries_proxy,
        "fetch_public_odds_optional": fetch_public_odds_optional,
        "fetch_players": fetch_players,
        "build_results_from_games": build_results_from_games,
        "fetch_upcoming_schedule": fetch_upcoming_schedule,
        "fetch_teams": fetch_teams,
        "fetch_xg_optional": fetch_xg_optional,
    }


def _parse_models_arg(models_arg: str | None) -> list[str] | None:
    if models_arg is None:
        return None
    tokens = [t.strip() for t in str(models_arg).split(",") if t.strip()]
    return normalize_selected_models(tokens)


def _apply_model_feature_policy(
    cfg: AppConfig,
    features_df: pd.DataFrame,
    *,
    approve_feature_changes: bool,
    run_context: str,
) -> list[str]:
    league = _canonical_league(cfg.data.league)
    raw_feature_cols = select_feature_columns(features_df)
    policy = apply_feature_policy(
        raw_feature_cols,
        league=league,
        mode=cfg.feature_policy.mode,
        registry_path_template=cfg.feature_policy.registry_path,
        approve_changes=approve_feature_changes,
    )
    logger.info(
        "Feature policy | context=%s mode=%s registry=%s selected=%d added=%d removed=%d candidates_added=%d updated=%s",
        run_context,
        policy.mode,
        policy.registry_path,
        len(policy.approved_feature_columns),
        len(policy.added_features),
        len(policy.removed_features),
        len(policy.candidates_added),
        policy.registry_updated,
    )
    return policy.approved_feature_columns


def _load_dotenv_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        env_key = key.strip()
        if not env_key or env_key in os.environ:
            continue
        env_value = value.strip()
        if len(env_value) >= 2 and env_value[0] == env_value[-1] and env_value[0] in {"'", '"'}:
            env_value = env_value[1:-1]
        os.environ[env_key] = env_value


def _client(cfg: AppConfig) -> HttpClient:
    return HttpClient(
        raw_dir=cfg.paths.raw_dir,
        timeout_seconds=cfg.data.timeout_seconds,
        max_retries=cfg.data.max_retries,
        backoff_seconds=cfg.data.backoff_seconds,
        offline_mode=cfg.data.offline_mode,
    )



def _save_interim(df: pd.DataFrame, interim_dir: str, name: str) -> str:
    path = Path(interim_dir) / INTERIM_FILES[name]
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
        return str(path)
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)



def _insert_snapshot(db: Database, res: SourceFetchResult) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO raw_snapshots(snapshot_id, source, extracted_at_utc, raw_path, metadata_json, freshness_utc, row_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            res.snapshot_id,
            res.source,
            res.extracted_at_utc,
            res.raw_path,
            to_json(res.metadata),
            res.extracted_at_utc,
            int(len(res.dataframe)),
        ),
    )



def _upsert_games(db: Database, games_df: pd.DataFrame) -> None:
    if games_df.empty:
        return
    rows = [
        (
            int(r.game_id),
            int(r.season) if pd.notna(r.season) else None,
            r.game_date_utc,
            r.start_time_utc,
            r.game_state,
            r.home_team,
            r.away_team,
            int(r.home_team_id) if pd.notna(r.home_team_id) else None,
            int(r.away_team_id) if pd.notna(r.away_team_id) else None,
            r.venue,
            int(r.is_neutral_site) if pd.notna(r.is_neutral_site) else 0,
            int(r.home_score) if pd.notna(r.home_score) else None,
            int(r.away_score) if pd.notna(r.away_score) else None,
            int(r.went_ot) if pd.notna(r.went_ot) else 0,
            int(r.went_so) if pd.notna(r.went_so) else 0,
            int(r.home_win) if pd.notna(r.home_win) else None,
            int(r.status_final) if pd.notna(r.status_final) else 0,
            str(r.as_of_utc),
        )
        for r in games_df.itertuples(index=False)
    ]
    db.executemany(
        """
        INSERT OR REPLACE INTO games(
          game_id, season, game_date_utc, start_time_utc, game_state,
          home_team, away_team, home_team_id, away_team_id, venue,
          is_neutral_site, home_score, away_score, went_ot, went_so,
          home_win, status_final, as_of_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )



def _upsert_results(db: Database, results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return
    rows = [
        (
            int(r.game_id),
            int(r.season) if pd.notna(r.season) else None,
            r.game_date_utc,
            r.final_utc,
            r.home_team,
            r.away_team,
            int(r.home_score) if pd.notna(r.home_score) else None,
            int(r.away_score) if pd.notna(r.away_score) else None,
            int(r.home_win) if pd.notna(r.home_win) else None,
            r.ingested_at_utc,
        )
        for r in results_df.itertuples(index=False)
    ]
    db.executemany(
        """
        INSERT OR REPLACE INTO results(
          game_id, season, game_date_utc, final_utc, home_team, away_team,
          home_score, away_score, home_win, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )



def _upsert_teams(
    db: Database,
    teams_df: pd.DataFrame,
    league: str,
    snapshot_id: str | None,
    as_of_utc: str,
) -> None:
    if teams_df.empty:
        return

    rows = []
    for r in teams_df.itertuples(index=False):
        team_abbrev = getattr(r, "team_abbrev", None)
        if team_abbrev is None or str(team_abbrev).strip() == "":
            continue
        rows.append(
            (
                league,
                str(team_abbrev),
                getattr(r, "team_name", None),
                getattr(r, "conference", None),
                getattr(r, "division", None),
                getattr(r, "as_of_date", None),
                as_of_utc,
                snapshot_id,
                to_json({}),
            )
        )

    db.executemany(
        """
        INSERT INTO teams(
          league, team_abbrev, team_name, conference, division,
          as_of_date, as_of_utc, snapshot_id, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _parse_iso_or_none(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _map_odds_rows_to_games(db: Database, odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        mapped = odds_df.copy()
        mapped["game_id"] = None
        return mapped

    required = {"odds_event_id", "home_team", "away_team"}
    if not required.issubset(set(odds_df.columns)):
        mapped = odds_df.copy()
        mapped["game_id"] = None
        return mapped

    games = pd.DataFrame(
        db.query(
            """
            SELECT game_id, game_date_utc, start_time_utc, home_team, away_team
            FROM games
            WHERE home_team IS NOT NULL
              AND away_team IS NOT NULL
            """
        )
    )

    if games.empty:
        mapped = odds_df.copy()
        mapped["game_id"] = None
        return mapped

    event_keys = (
        odds_df[["odds_event_id", "home_team", "away_team", "commence_date_central", "commence_time_utc"]]
        .drop_duplicates(subset=["odds_event_id"])
        .to_dict(orient="records")
    )

    event_to_game: dict[str, int | None] = {}
    for event in event_keys:
        event_id = str(event.get("odds_event_id") or "").strip()
        if not event_id:
            continue
        home_team = str(event.get("home_team") or "").strip()
        away_team = str(event.get("away_team") or "").strip()
        if not home_team or not away_team:
            event_to_game[event_id] = None
            continue

        candidates = games[(games["home_team"] == home_team) & (games["away_team"] == away_team)].copy()
        if candidates.empty:
            event_to_game[event_id] = None
            continue

        preferred_date = str(event.get("commence_date_central") or "").strip()
        if preferred_date:
            by_date = candidates[candidates["game_date_utc"].astype(str) == preferred_date]
            if len(by_date) == 1:
                event_to_game[event_id] = int(by_date.iloc[0]["game_id"])
                continue
            if len(by_date) > 1:
                candidates = by_date

        commence_dt = _parse_iso_or_none(event.get("commence_time_utc"))
        if commence_dt is not None and not candidates.empty:
            candidates = candidates.copy()
            candidates["start_dt"] = candidates["start_time_utc"].apply(_parse_iso_or_none)
            candidates = candidates[candidates["start_dt"].notna()]
            if not candidates.empty:
                candidates["abs_diff_seconds"] = candidates["start_dt"].apply(lambda dt: abs((dt - commence_dt).total_seconds()))
                winner = candidates.sort_values("abs_diff_seconds").iloc[0]
                if float(winner["abs_diff_seconds"]) <= 18 * 3600:
                    event_to_game[event_id] = int(winner["game_id"])
                    continue

        event_to_game[event_id] = int(candidates.iloc[0]["game_id"]) if len(candidates) == 1 else None

    mapped = odds_df.copy()
    mapped["game_id"] = mapped["odds_event_id"].map(event_to_game)
    return mapped


def _insert_odds_snapshot_and_lines(
    db: Database,
    league: str,
    odds_res: SourceFetchResult,
) -> None:
    def _nullable_text(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return text

    metadata = odds_res.metadata or {}
    rows_df = _map_odds_rows_to_games(db, odds_res.dataframe)

    db.execute(
        """
        INSERT OR REPLACE INTO odds_snapshots(
          odds_snapshot_id, source, league, as_of_utc, raw_path,
          regions, markets, odds_format, date_format,
          event_count, row_count,
          requests_last, requests_used, requests_remaining,
          from_cache, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            odds_res.snapshot_id,
            odds_res.source,
            league,
            odds_res.extracted_at_utc,
            odds_res.raw_path,
            str(metadata.get("regions") or ""),
            str(metadata.get("markets") or ""),
            str(metadata.get("odds_format") or ""),
            str(metadata.get("date_format") or ""),
            int(metadata.get("n_events") or 0),
            int(len(rows_df)),
            int(metadata["requests_last"]) if metadata.get("requests_last") is not None else None,
            int(metadata["requests_used"]) if metadata.get("requests_used") is not None else None,
            int(metadata["requests_remaining"]) if metadata.get("requests_remaining") is not None else None,
            int(metadata.get("from_cache") or 0),
            to_json(metadata),
        ),
    )

    if rows_df.empty:
        return

    created_at_utc = utc_now_iso()
    insert_rows: list[tuple[Any, ...]] = []
    for row in rows_df.itertuples(index=False):
        game_id_raw = getattr(row, "game_id", None)
        game_id = int(game_id_raw) if pd.notna(game_id_raw) else None
        outcome_price = pd.to_numeric(getattr(row, "outcome_price", None), errors="coerce")
        outcome_point = pd.to_numeric(getattr(row, "outcome_point", None), errors="coerce")
        implied_probability = pd.to_numeric(getattr(row, "implied_probability", None), errors="coerce")

        insert_rows.append(
            (
                odds_res.snapshot_id,
                league,
                game_id,
                _nullable_text(getattr(row, "sport_key", "")),
                _nullable_text(getattr(row, "odds_event_id", "")),
                _nullable_text(getattr(row, "commence_time_utc", "")),
                _nullable_text(getattr(row, "commence_date_central", "")),
                _nullable_text(getattr(row, "api_home_team", "")),
                _nullable_text(getattr(row, "api_away_team", "")),
                _nullable_text(getattr(row, "home_team", "")),
                _nullable_text(getattr(row, "away_team", "")),
                _nullable_text(getattr(row, "bookmaker_key", "")),
                _nullable_text(getattr(row, "bookmaker_title", "")),
                _nullable_text(getattr(row, "bookmaker_last_update_utc", "")),
                _nullable_text(getattr(row, "market_key", "")),
                _nullable_text(getattr(row, "outcome_name", "")),
                _nullable_text(getattr(row, "outcome_side", "")),
                _nullable_text(getattr(row, "outcome_team", "")),
                float(outcome_price) if pd.notna(outcome_price) else None,
                float(outcome_point) if pd.notna(outcome_point) else None,
                float(implied_probability) if pd.notna(implied_probability) else None,
                created_at_utc,
            )
        )

    db.executemany(
        """
        INSERT INTO odds_market_lines(
          odds_snapshot_id, league, game_id, sport_key, odds_event_id,
          commence_time_utc, commence_date_central, api_home_team, api_away_team,
          home_team, away_team, bookmaker_key, bookmaker_title, bookmaker_last_update_utc,
          market_key, outcome_name, outcome_side, outcome_team,
          outcome_price, outcome_point, implied_probability, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        insert_rows,
    )


def _latest_snapshot_id(db: Database) -> str | None:
    rows = db.query("SELECT snapshot_id FROM raw_snapshots ORDER BY extracted_at_utc DESC LIMIT 1")
    return rows[0]["snapshot_id"] if rows else None



def cmd_init_db(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()
    logger.info("Initialized DB schema at %s", cfg.paths.db_path)



def cmd_fetch(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()
    client = _client(cfg)
    league = _canonical_league(cfg.data.league)
    sources = _league_sources(league)

    end_date = datetime.now(timezone.utc) + timedelta(days=cfg.data.upcoming_days)
    start_date = datetime.now(timezone.utc) - timedelta(days=cfg.data.history_days)

    games_res = sources["fetch_games"](client, start_date=start_date, end_date=end_date)
    _save_interim(games_res.dataframe, cfg.paths.interim_dir, "games")
    _insert_snapshot(db, games_res)
    _upsert_games(db, games_res.dataframe)

    schedule_res = sources["fetch_upcoming_schedule"](client, days_ahead=cfg.data.upcoming_days)
    _save_interim(schedule_res.dataframe, cfg.paths.interim_dir, "schedule")
    _insert_snapshot(db, schedule_res)

    teams_res = sources["fetch_teams"](client)
    _save_interim(teams_res.dataframe, cfg.paths.interim_dir, "teams")
    _insert_snapshot(db, teams_res)
    _upsert_teams(
        db,
        teams_df=teams_res.dataframe,
        league=league,
        snapshot_id=teams_res.snapshot_id,
        as_of_utc=teams_res.extracted_at_utc,
    )

    team_abbrevs = sorted(set(teams_res.dataframe.get("team_abbrev", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    if not games_res.dataframe.empty and "season" in games_res.dataframe.columns and games_res.dataframe["season"].notna().any():
        season_guess = str(int(games_res.dataframe["season"].dropna().max()))
    else:
        now = datetime.now(timezone.utc)
        season_end_year = now.year + (1 if now.month >= 7 else 0)
        season_guess = f"{season_end_year - 1}{season_end_year}"

    players_res = sources["fetch_players"](client, team_abbrevs=team_abbrevs, season=season_guess)
    _save_interim(players_res.dataframe, cfg.paths.interim_dir, "players")
    _insert_snapshot(db, players_res)

    final_ids = games_res.dataframe[games_res.dataframe["status_final"] == 1]["game_id"].astype(int).tolist()
    goalies_res = sources["fetch_goalie_game_stats"](client, game_ids=final_ids, max_games=350)
    _save_interim(goalies_res.dataframe, cfg.paths.interim_dir, "goalies")
    _insert_snapshot(db, goalies_res)

    injuries_res = sources["fetch_injuries_proxy"](client, teams=team_abbrevs)
    _save_interim(injuries_res.dataframe, cfg.paths.interim_dir, "injuries")
    _insert_snapshot(db, injuries_res)

    odds_res = sources["fetch_public_odds_optional"](client, teams_df=teams_res.dataframe, league=league)
    _save_interim(odds_res.dataframe, cfg.paths.interim_dir, "odds")
    _insert_snapshot(db, odds_res)
    _insert_odds_snapshot_and_lines(db, league=league, odds_res=odds_res)

    xg_res = sources["fetch_xg_optional"](client)
    _save_interim(xg_res.dataframe, cfg.paths.interim_dir, "xg")
    _insert_snapshot(db, xg_res)

    results_df = sources["build_results_from_games"](games_res.dataframe)
    _upsert_results(db, results_df)

    logger.info(
        "Fetch complete | league=%s games=%d final=%d upcoming=%d players=%d goalie_rows=%d",
        league,
        len(games_res.dataframe),
        int(games_res.dataframe["status_final"].sum()) if not games_res.dataframe.empty else 0,
        len(schedule_res.dataframe),
        len(players_res.dataframe),
        len(goalies_res.dataframe),
    )



def cmd_features(cfg: AppConfig) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()

    res = build_features_from_interim(cfg.paths.interim_dir, cfg.paths.processed_dir)
    db.execute(
        """
        INSERT OR REPLACE INTO feature_sets(feature_set_version, created_at_utc, snapshot_id, feature_columns_json, metadata_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            res.feature_set_version,
            utc_now_iso(),
            _latest_snapshot_id(db),
            to_json(res.feature_columns),
            to_json(res.metadata),
        ),
    )
    logger.info("Features built | rows=%d features=%d version=%s", len(res.dataframe), len(res.feature_columns), res.feature_set_version)



def _persist_predictions(
    db: Database,
    forecasts: pd.DataFrame,
    per_model_probs: pd.DataFrame,
    model_run_id: str,
    feature_set_version: str,
) -> None:
    snapshot_id = _latest_snapshot_id(db)

    pred_rows = []
    forecast_rows = []

    per_model_map = per_model_probs.set_index("game_id").to_dict(orient="index")

    for r in forecasts.itertuples(index=False):
        game_id = int(r.game_id)
        model_probs = per_model_map.get(game_id, {})
        as_of = str(r.as_of_utc)

        for model_name, p in model_probs.items():
            if model_name == "game_id":
                continue
            prob = float(p)
            winner = r.home_team if prob >= 0.5 else r.away_team
            pred_rows.append(
                (
                    game_id,
                    as_of,
                    model_name,
                    f"{model_run_id}__{model_name}",
                    feature_set_version,
                    snapshot_id,
                    r.game_date_utc,
                    r.home_team,
                    r.away_team,
                    prob,
                    winner,
                    None,
                    None,
                    r.uncertainty_flags_json,
                    to_json({"source": "train_upcoming"}),
                )
            )

        ensemble_prob = float(r.ensemble_prob_home_win)
        ensemble_winner = r.home_team if ensemble_prob >= 0.5 else r.away_team
        pred_rows.append(
            (
                game_id,
                as_of,
                "ensemble",
                f"{model_run_id}__ensemble",
                feature_set_version,
                snapshot_id,
                r.game_date_utc,
                r.home_team,
                r.away_team,
                ensemble_prob,
                ensemble_winner,
                r.bayes_ci_low,
                r.bayes_ci_high,
                r.uncertainty_flags_json,
                to_json({"source": "train_upcoming"}),
            )
        )

        forecast_rows.append(
            (
                game_id,
                as_of,
                r.game_date_utc,
                r.home_team,
                r.away_team,
                ensemble_prob,
                ensemble_winner,
                r.per_model_probs_json,
                r.spread_min,
                r.spread_median,
                r.spread_max,
                r.spread_mean,
                r.spread_sd,
                r.spread_iqr,
                r.bayes_ci_low,
                r.bayes_ci_high,
                r.uncertainty_flags_json,
                snapshot_id,
                feature_set_version,
                model_run_id,
            )
        )

    db.executemany(
        """
        INSERT OR REPLACE INTO predictions(
          game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
          game_date_utc, home_team, away_team, prob_home_win, pred_winner, prob_low, prob_high,
          uncertainty_flags_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pred_rows,
    )

    db.executemany(
        """
        INSERT OR REPLACE INTO upcoming_game_forecasts(
          game_id, as_of_utc, game_date_utc, home_team, away_team,
          ensemble_prob_home_win, predicted_winner, per_model_probs_json,
          spread_min, spread_median, spread_max, spread_mean, spread_sd, spread_iqr,
          bayes_ci_low, bayes_ci_high, uncertainty_flags_json, snapshot_id,
          feature_set_version, model_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        forecast_rows,
    )



def _run_validation_outputs(result: dict, cfg: AppConfig) -> None:
    models = result["models"]
    train_df = result["train_df"].copy()
    feature_cols = result["feature_columns"]
    out_val = ensure_dir(Path(cfg.paths.artifacts_dir) / "validation")

    split = int(len(train_df) * 0.8)
    tr = train_df.iloc[:split].copy()
    va = train_df.iloc[split:].copy()

    # GLM diagnostics.
    glm = models.get("glm_logit")
    if glm is not None and not va.empty:
        va = va.copy()
        va["glm_prob"] = glm.predict_proba(va)
        save_glm_diagnostics(
            df=va,
            p_col="glm_prob",
            y_col="home_win",
            feature_cols=glm.feature_columns,
            coefs=glm.model.coef_[0],
            out_dir=str(Path(cfg.paths.artifacts_dir) / "plots"),
            prefix="glm",
        )

    # Permutation importance.
    for model_name in ["gbdt", "rf"]:
        m = models.get(model_name)
        if m is not None and len(va) > 25:
            permutation_importance_report(
                m.model,
                va[feature_cols],
                va["home_win"].astype(int).to_numpy(),
                out_dir=str(Path(cfg.paths.artifacts_dir) / "validation"),
                model_name=model_name,
            )

    # Significance blocks.
    feature_blocks = {
        "goalie_block": [c for c in feature_cols if "goalie" in c],
        "xg_block": [c for c in feature_cols if "xg" in c],
        "special_teams_block": [c for c in feature_cols if "special" in c or "penalty" in c or "pp_" in c],
        "travel_block": [c for c in feature_cols if "travel" in c or "rest" in c or "tz_" in c],
        "lineup_block": [c for c in feature_cols if "lineup" in c or "roster" in c or "man_games" in c],
        "rink_block": [c for c in feature_cols if "rink" in c],
    }
    sig = blockwise_nested_lrt(tr, va, feature_blocks=feature_blocks, all_features=feature_cols)
    sig.to_csv(out_val / "validation_significance.csv", index=False)

    # Stability and multicollinearity.
    coef_path = coefficient_paths(train_df, features=feature_cols)
    coef_path.to_csv(out_val / "validation_coef_paths.csv", index=False)
    vif = vif_table(train_df, features=feature_cols)
    vif.to_csv(out_val / "validation_vif.csv", index=False)
    break_test = break_test_trade_deadline(train_df, features=feature_cols)
    (out_val / "validation_break_test.json").write_text(json.dumps(break_test, indent=2, sort_keys=True))

    # Influence.
    infl_df, infl_summary = influence_diagnostics(train_df, features=feature_cols, top_k=10)
    infl_df.to_csv(out_val / "validation_influence_top.csv", index=False)
    (out_val / "validation_influence_summary.json").write_text(json.dumps(infl_summary, indent=2, sort_keys=True))

    # Fragility.
    if glm is not None:
        stress = missingness_stress_test(glm, va if not va.empty else train_df, feature_cols=feature_cols)
        stress.to_csv(out_val / "validation_fragility_missingness.csv", index=False)
        pert = perturbation_sensitivity(glm, va if not va.empty else train_df, feature_cols=feature_cols)
        (out_val / "validation_fragility_perturbation.json").write_text(json.dumps(pert, indent=2, sort_keys=True))

    # Calibration robustness outputs.
    if glm is not None and not va.empty:
        p = glm.predict_proba(va)
        y = va["home_win"].astype(int).to_numpy()
        cal = calibration_alpha_beta(y, p) | ece_mce(y, p)
        cal |= brier_decompose(y, p)
        (out_val / "validation_calibration_robustness.json").write_text(json.dumps(cal, indent=2, sort_keys=True))



def cmd_train(cfg: AppConfig, models_arg: str | None = None, approve_feature_changes: bool = False) -> None:
    def emit_train_progress(event: dict[str, Any]) -> None:
        print(f"TRAIN_PROGRESS::{json.dumps(event, sort_keys=True)}", flush=True)

    db = Database(cfg.paths.db_path)
    db.init_schema()

    feat_path = Path(cfg.paths.processed_dir) / "features.parquet"
    if not feat_path.exists():
        feat_path = Path(cfg.paths.processed_dir) / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("features.parquet not found. Run features first.")

    if feat_path.suffix == ".parquet":
        features_df = pd.read_parquet(feat_path)
    else:
        features_df = pd.read_csv(feat_path)
    approved_feature_columns = _apply_model_feature_policy(
        cfg,
        features_df,
        approve_feature_changes=approve_feature_changes,
        run_context="train",
    )
    feature_set_rows = db.query("SELECT feature_set_version FROM feature_sets ORDER BY created_at_utc DESC LIMIT 1")
    feature_set_version = feature_set_rows[0]["feature_set_version"] if feature_set_rows else "unknown_feature_set"
    selected_models = _parse_models_arg(models_arg)

    tracker = RunTracker(cfg.paths.artifacts_dir)
    run_id = tracker.start_run(
        "train",
        {
            "feature_set_version": feature_set_version,
            "selected_models": selected_models if selected_models is not None else ["all"],
        },
    )
    emit_train_progress(
        {
            "kind": "pipeline",
            "stage": "train_command",
            "status": "started",
            "message": "Starting cmd_train",
            "feature_set_version": feature_set_version,
            "selected_models": selected_models if selected_models is not None else ["all"],
        }
    )
    result = train_and_predict(
        features_df=features_df,
        feature_set_version=feature_set_version,
        artifacts_dir=cfg.paths.artifacts_dir,
        bayes_cfg=cfg.bayes.model_dump(),
        selected_models=selected_models,
        progress_callback=emit_train_progress,
        selected_feature_columns=approved_feature_columns,
    )
    tracker.log_metrics(
        run_id,
        {
            "n_upcoming": int(len(result["forecasts"])),
            "stack_ready": int(result["stack_ready"]),
            "n_selected_models": int(len(result["run_payload"].get("selected_models", []))),
        },
    )
    tracker.log_artifact(run_id, "train_metrics", result["train_metrics"])

    _persist_predictions(
        db,
        forecasts=result["forecasts"],
        per_model_probs=result["upcoming_model_probs"],
        model_run_id=result["model_run_id"],
        feature_set_version=feature_set_version,
    )

    # Persist model run metadata rows.
    run_rows = []
    for model_name in [c for c in result["upcoming_model_probs"].columns if c != "game_id"] + ["ensemble"]:
        run_rows.append(
            (
                f"{result['model_run_id']}__{model_name}",
                model_name,
                "daily_train",
                utc_now_iso(),
                _latest_snapshot_id(db),
                feature_set_version,
                to_json({"weights": result["weights"]}),
                to_json(result["train_metrics"].get(model_name, {})),
                result["model_dir"],
                result["model_run_id"],
            )
        )
    db.executemany(
        """
        INSERT OR REPLACE INTO model_runs(
          model_run_id, model_name, run_type, created_at_utc, snapshot_id,
          feature_set_version, params_json, metrics_json, artifact_path, model_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        run_rows,
    )

    _run_validation_outputs(result, cfg)

    tracker.end_run(run_id)
    emit_train_progress(
        {
            "kind": "pipeline",
            "stage": "train_command",
            "status": "completed",
            "message": "Completed cmd_train",
            "model_run_id": result["model_run_id"],
        }
    )
    logger.info(
        "Train complete | model_run_id=%s upcoming=%d selected_models=%s",
        result["model_run_id"],
        len(result["forecasts"]),
        result["run_payload"].get("selected_models"),
    )



def cmd_backtest(cfg: AppConfig, models_arg: str | None = None, approve_feature_changes: bool = False) -> None:
    db = Database(cfg.paths.db_path)
    db.init_schema()

    feat_path = Path(cfg.paths.processed_dir) / "features.parquet"
    if not feat_path.exists():
        feat_path = Path(cfg.paths.processed_dir) / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("features.parquet not found. Run features first.")

    if feat_path.suffix == ".parquet":
        features_df = pd.read_parquet(feat_path)
    else:
        features_df = pd.read_csv(feat_path)
    approved_feature_columns = _apply_model_feature_policy(
        cfg,
        features_df,
        approve_feature_changes=approve_feature_changes,
        run_context="backtest",
    )
    selected_models = _parse_models_arg(models_arg)
    bt = run_walk_forward_backtest(
        features_df,
        artifacts_dir=cfg.paths.artifacts_dir,
        bayes_cfg=cfg.bayes.model_dump(),
        n_splits=cfg.modeling.cv_splits,
        selected_models=selected_models,
        selected_feature_columns=approved_feature_columns,
    )

    oof = bt["oof_predictions"]
    if oof.empty:
        logger.warning("Backtest produced no folds.")
        return

    feature_set_rows = db.query("SELECT feature_set_version FROM feature_sets ORDER BY created_at_utc DESC LIMIT 1")
    feature_set_version = feature_set_rows[0]["feature_set_version"] if feature_set_rows else "unknown_feature_set"
    model_run_id = f"backtest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    snapshot_id = _latest_snapshot_id(db)

    pred_rows = []
    for r in oof.itertuples(index=False):
        game_date = pd.to_datetime(r.game_date_utc)
        as_of = (game_date - pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0).isoformat()
        for model_name in [c for c in oof.columns if c not in {"fold", "home_win", "game_id", "game_date_utc"}]:
            p = float(getattr(r, model_name))
            pred_rows.append(
                (
                    int(r.game_id),
                    as_of,
                    model_name,
                    f"{model_run_id}__{model_name}",
                    feature_set_version,
                    snapshot_id,
                    str(r.game_date_utc),
                    None,
                    None,
                    p,
                    None,
                    None,
                    None,
                    to_json({"backtest_fold": int(r.fold)}),
                    to_json({"source": "walk_forward_backtest"}),
                )
            )

    db.executemany(
        """
        INSERT OR REPLACE INTO predictions(
          game_id, as_of_utc, model_name, model_run_id, feature_set_version, snapshot_id,
          game_date_utc, home_team, away_team, prob_home_win, pred_winner, prob_low, prob_high,
          uncertainty_flags_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pred_rows,
    )

    # Score prequentially after inserting backtest predictions.
    score_info = score_predictions(db, windows_days=cfg.modeling.rolling_windows_days)

    # Integrity checks.
    pred_df = pd.DataFrame(db.query("SELECT * FROM predictions"))
    res_df = pd.DataFrame(db.query("SELECT * FROM results"))
    integrity = run_backtest_integrity_checks(pred_df, res_df, embargo_days=cfg.runtime.embargo_days)
    out_path = Path(cfg.paths.artifacts_dir) / "validation" / "backtest_integrity.json"
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(integrity, indent=2, sort_keys=True))

    logger.info(
        "Backtest complete | oof_rows=%d scored=%d selected_models=%s",
        len(oof),
        score_info.get("n_scored", 0),
        selected_models if selected_models is not None else ["all"],
    )



def cmd_run_daily(cfg: AppConfig, models_arg: str | None = None, approve_feature_changes: bool = False) -> None:
    cmd_fetch(cfg)
    cmd_features(cfg)
    if cfg.runtime.retrain_daily:
        cmd_train(cfg, models_arg=models_arg, approve_feature_changes=approve_feature_changes)

    db = Database(cfg.paths.db_path)
    score_info = score_predictions(db, windows_days=cfg.modeling.rolling_windows_days)

    # Export leaderboard/performance artifacts for dashboard convenience.
    perf = pd.DataFrame(db.query("SELECT * FROM performance_aggregates ORDER BY as_of_utc DESC"))
    if not perf.empty:
        out = Path(cfg.paths.artifacts_dir) / "reports" / "performance_aggregates_latest.csv"
        ensure_dir(out.parent)
        perf.to_csv(out, index=False)

    logger.info("Daily run complete | scored=%s", score_info)



def cmd_smoke(cfg: AppConfig) -> None:
    # Limited smoke with shortened history for speed.
    old_hist = cfg.data.history_days
    old_upc = cfg.data.upcoming_days
    cfg.data.history_days = min(60, old_hist)
    cfg.data.upcoming_days = min(7, old_upc)

    cmd_init_db(cfg)
    cmd_fetch(cfg)
    cmd_features(cfg)
    cmd_train(cfg, approve_feature_changes=True)
    score_info = score_predictions(Database(cfg.paths.db_path), windows_days=cfg.modeling.rolling_windows_days)

    logger.info("Smoke scoring info: %s", score_info)
    logger.info("Smoke query checks:")
    from src.query.answer import answer_question

    league = _canonical_league(cfg.data.league)
    team_prompt = "Leafs" if league == "NHL" else "Raptors"

    db = Database(cfg.paths.db_path)
    for q in [
        f"What's the chance the {team_prompt} win their next game?",
        "Which model has performed best the last 60 days?",
    ]:
        ans, payload = answer_question(db, q, default_league=league)
        logger.info("Q: %s", q)
        logger.info("A: %s", ans)
        logger.info("Payload keys: %s", list(payload.keys()))

    cfg.data.history_days = old_hist
    cfg.data.upcoming_days = old_upc



def main() -> None:
    _load_dotenv_file(Path(".env"))
    _load_dotenv_file(Path("web/.env.local"))

    parser = argparse.ArgumentParser(description="NHL/NBA probabilistic forecasting pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ["init-db", "fetch", "features", "train", "backtest", "run-daily", "smoke"]:
        p = sub.add_parser(cmd)
        p.add_argument("--config", default="configs/nhl.yaml")
        if cmd in {"train", "backtest", "run-daily"}:
            p.add_argument(
                "--models",
                default="all",
                help="Comma-separated model list (e.g. glm_logit,rf) or 'all'",
            )
            p.add_argument(
                "--approve-feature-changes",
                action="store_true",
                help="Explicitly accept and persist model feature-contract changes in the registry.",
            )

    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging("INFO")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="X has feature names")

    if args.command == "init-db":
        cmd_init_db(cfg)
    elif args.command == "fetch":
        cmd_fetch(cfg)
    elif args.command == "features":
        cmd_features(cfg)
    elif args.command == "train":
        cmd_train(cfg, models_arg=args.models, approve_feature_changes=bool(args.approve_feature_changes))
    elif args.command == "backtest":
        cmd_backtest(cfg, models_arg=args.models, approve_feature_changes=bool(args.approve_feature_changes))
    elif args.command == "run-daily":
        cmd_run_daily(
            cfg,
            models_arg=args.models,
            approve_feature_changes=bool(args.approve_feature_changes),
        )
    elif args.command == "smoke":
        cmd_smoke(cfg)


if __name__ == "__main__":
    main()
