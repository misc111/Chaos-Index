from __future__ import annotations

import pandas as pd

from src.common.market_transforms import vig_free_two_way_logit_offsets
from src.common.time import utc_now_iso
from src.common.utils import to_json
from src.data_sources.base import SourceFetchResult
from src.services.history_import import _filter_odds_to_recent_seasons
from src.services.ingest import insert_odds_snapshot_and_lines, upsert_games, upsert_results
from src.storage.db import Database


def test_init_schema_exposes_mlb_contract_surfaces(tmp_path) -> None:
    db = Database(str(tmp_path / "mlb_contract.db"))
    db.init_schema()

    objects = {
        row["name"]: row["type"]
        for row in db.query(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE name LIKE 'mlb_%'
            ORDER BY name
            """
        )
    }

    assert objects["mlb_games"] == "table"
    assert objects["mlb_results"] == "table"
    assert objects["mlb_odds_snapshots"] == "table"
    assert objects["mlb_odds_market_lines"] == "table"
    assert objects["mlb_feature_sets"] == "view"
    assert objects["mlb_model_runs"] == "view"
    assert objects["mlb_model_predictions"] == "view"
    assert objects["mlb_validation_results"] == "view"
    assert objects["mlb_backtest_bets"] == "view"
    assert objects["mlb_performance_aggregates"] == "view"
    assert objects["mlb_theory_trace"] == "table"
    assert objects["mlb_prediction_components"] == "table"


def test_mlb_contract_surfaces_reflect_ingest_writes(tmp_path) -> None:
    db = Database(str(tmp_path / "mlb_ingest.db"))
    db.init_schema()

    games_df = pd.DataFrame(
        [
            {
                "game_id": 401815019,
                "season": 2026,
                "game_date_utc": "2026-04-19",
                "start_time_utc": "2026-04-19T18:35:00Z",
                "game_state": "Final",
                "home_team": "CHC",
                "away_team": "LAD",
                "home_team_id": 16,
                "away_team_id": 19,
                "venue": "Wrigley Field",
                "is_neutral_site": 0,
                "home_score": 5,
                "away_score": 3,
                "went_ot": 0,
                "went_so": 0,
                "home_win": 1,
                "status_final": 1,
                "as_of_utc": "2026-04-19T22:00:00Z",
            }
        ]
    )
    upsert_games(db, games_df, league="MLB")

    results_df = pd.DataFrame(
        [
            {
                "game_id": 401815019,
                "season": 2026,
                "game_date_utc": "2026-04-19",
                "final_utc": "2026-04-19T18:35:00Z",
                "home_team": "CHC",
                "away_team": "LAD",
                "home_score": 5,
                "away_score": 3,
                "home_win": 1,
                "ingested_at_utc": "2026-04-19T22:00:00Z",
            }
        ]
    )
    upsert_results(db, results_df, league="MLB")

    odds_res = SourceFetchResult(
        source="mlb_odds",
        snapshot_id="mlb_odds_snapshot_1",
        extracted_at_utc="2026-04-19T16:00:00Z",
        raw_path=str(tmp_path / "mlb_odds_snapshot_1.json"),
        metadata={
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "odds_format": "american",
            "date_format": "iso",
            "n_events": 1,
            "requests_last": 1,
            "requests_used": 3,
            "requests_remaining": 97,
            "from_cache": 0,
        },
        dataframe=pd.DataFrame(
            [
                {
                    "sport_key": "baseball_mlb",
                    "odds_event_id": "401815019",
                    "commence_time_utc": "2026-04-19T18:35:00Z",
                    "commence_date_central": "2026-04-19",
                    "api_home_team": "Chicago Cubs",
                    "api_away_team": "Los Angeles Dodgers",
                    "home_team": "CHC",
                    "away_team": "LAD",
                    "bookmaker_key": "draftkings",
                    "bookmaker_title": "DraftKings",
                    "bookmaker_last_update_utc": "2026-04-19T16:00:00Z",
                    "market_key": "h2h",
                    "outcome_name": "CHC",
                    "outcome_side": "home",
                    "outcome_team": "CHC",
                    "outcome_price": -120,
                    "outcome_point": None,
                    "implied_probability": 0.5454545,
                },
                {
                    "sport_key": "baseball_mlb",
                    "odds_event_id": "401815019",
                    "commence_time_utc": "2026-04-19T18:35:00Z",
                    "commence_date_central": "2026-04-19",
                    "api_home_team": "Chicago Cubs",
                    "api_away_team": "Los Angeles Dodgers",
                    "home_team": "CHC",
                    "away_team": "LAD",
                    "bookmaker_key": "draftkings",
                    "bookmaker_title": "DraftKings",
                    "bookmaker_last_update_utc": "2026-04-19T16:00:00Z",
                    "market_key": "h2h",
                    "outcome_name": "LAD",
                    "outcome_side": "away",
                    "outcome_team": "LAD",
                    "outcome_price": 110,
                    "outcome_point": None,
                    "implied_probability": 0.4761905,
                },
            ]
        ),
    )
    insert_odds_snapshot_and_lines(db, league="MLB", odds_res=odds_res)

    game_rows = db.query("SELECT game_id, season, home_team, away_team FROM mlb_games")
    result_rows = db.query("SELECT game_id, season, home_score, away_score FROM mlb_results")
    snapshot_rows = db.query("SELECT odds_snapshot_id, league, event_count, row_count FROM mlb_odds_snapshots")
    line_rows = db.query(
        """
        SELECT game_id, market_key, outcome_side, outcome_team
        FROM mlb_odds_market_lines
        ORDER BY line_id
        """
    )

    assert game_rows == [{"game_id": 401815019, "season": 2026, "home_team": "CHC", "away_team": "LAD"}]
    assert result_rows == [{"game_id": 401815019, "season": 2026, "home_score": 5, "away_score": 3}]
    assert snapshot_rows == [
        {
            "odds_snapshot_id": "mlb_odds_snapshot_1",
            "league": "MLB",
            "event_count": 1,
            "row_count": 2,
        }
    ]
    assert line_rows == [
        {"game_id": 401815019, "market_key": "h2h", "outcome_side": "home", "outcome_team": "CHC"},
        {"game_id": 401815019, "market_key": "h2h", "outcome_side": "away", "outcome_team": "LAD"},
    ]


def test_mlb_theory_trace_and_prediction_components_accept_contract_rows(tmp_path) -> None:
    db = Database(str(tmp_path / "mlb_contract_rows.db"))
    db.init_schema()
    market_offset_logit, _ = vig_free_two_way_logit_offsets(-120, +110, input_scale="american_price")

    db.execute(
        """
        INSERT INTO mlb_theory_trace(
          trace_id, implementation_key, category, theory_source, theory_reference,
          implementation_label, implementation_type, notes, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "trace_1",
            "glm_logit_moneyline",
            "direct theory implementation",
            "09_GLM_Generalized_Linear_Models_for_Insurance_Rating",
            "logit-link logistic GLM",
            "moneyline_home_win_logit",
            "model_family",
            "Home-win Bernoulli lane",
            utc_now_iso(),
        ),
    )
    db.execute(
        """
        INSERT INTO mlb_prediction_components(
          component_id, game_id, as_of_utc, model_name, component_name,
          component_role, component_value, metadata_json, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "component_1",
            401815019,
            "2026-04-19T16:00:00Z",
            "glm_logit_moneyline",
            "market_offset_logit",
            "offset",
            float(market_offset_logit),
            to_json({"source": "vig_free_market"}),
            utc_now_iso(),
        ),
    )

    assert db.query("SELECT COUNT(*) AS count FROM mlb_theory_trace")[0]["count"] == 1
    assert db.query("SELECT COUNT(*) AS count FROM mlb_prediction_components")[0]["count"] == 1


def test_historical_bundle_effective_odds_as_of_is_strictly_pregame(tmp_path) -> None:
    db = Database(str(tmp_path / "mlb_historical_bundle_odds.db"))
    db.init_schema()
    odds_res = SourceFetchResult(
        source="mlb_historical_odds_backfill",
        snapshot_id="historical_bundle_1",
        extracted_at_utc="2026-04-24T01:00:00Z",
        raw_path=str(tmp_path / "odds.csv"),
        metadata={"import_mode": "historical_bundle", "n_events": 1},
        dataframe=pd.DataFrame(
            [
                {
                    "sport_key": "baseball_mlb",
                    "odds_event_id": "401815019",
                    "commence_time_utc": "2026-04-19T18:35:00Z",
                    "commence_date_central": "2026-04-19",
                    "api_home_team": "Chicago Cubs",
                    "api_away_team": "Los Angeles Dodgers",
                    "home_team": "CHC",
                    "away_team": "LAD",
                    "bookmaker_key": "espn",
                    "bookmaker_title": "ESPN",
                    "bookmaker_last_update_utc": "2026-04-24T01:00:00Z",
                    "market_key": "spreads",
                    "outcome_name": "Chicago Cubs",
                    "outcome_side": "home",
                    "outcome_team": "CHC",
                    "outcome_price": -110,
                    "outcome_point": -1.5,
                    "implied_probability": 0.5238095,
                }
            ]
        ),
    )
    insert_odds_snapshot_and_lines(db, league="MLB", odds_res=odds_res)

    rows = db.query(
        """
        SELECT commence_time_utc, effective_odds_as_of_utc
        FROM odds_market_lines_effective_as_of
        """
    )

    assert rows == [
        {
            "commence_time_utc": "2026-04-19T18:35:00Z",
            "effective_odds_as_of_utc": "2026-04-19T18:34:59Z",
        }
    ]


def test_mlb_historical_odds_filter_uses_single_year_seasons() -> None:
    odds_df = pd.DataFrame(
        [
            {"commence_time_utc": "2025-09-01T18:05:00Z", "odds_event_id": "g1"},
            {"commence_time_utc": "2026-04-19T18:35:00Z", "odds_event_id": "g2"},
        ]
    )

    filtered = _filter_odds_to_recent_seasons(odds_df, {2026}, league="MLB")

    assert filtered["odds_event_id"].tolist() == ["g2"]
