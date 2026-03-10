from pathlib import Path

import pytest

from src.league_registry import get_league_adapter, supported_leagues
from src.orchestration.refresh_pipeline import build_data_refresh_steps
from src.query.answer import answer_question
from src.storage.db import Database


@pytest.mark.parametrize(
    ("league", "slug", "config_path", "championship_name", "probability_key"),
    [
        ("NHL", "nhl", "configs/nhl.yaml", "Stanley Cup", "stanley_cup_prob"),
        ("NBA", "nba", "configs/nba.yaml", "NBA Finals", "nba_finals_prob"),
        ("NCAAM", "ncaam", "configs/ncaam.yaml", "NCAA Tournament", "ncaa_tournament_prob"),
    ],
)
def test_league_adapter_contracts_stay_in_sync(
    league: str,
    slug: str,
    config_path: str,
    championship_name: str,
    probability_key: str,
) -> None:
    adapter = get_league_adapter(league)

    assert adapter.code == league
    assert adapter.metadata.slug == slug
    assert adapter.metadata.default_config_path == config_path
    assert adapter.metadata.championship_name == championship_name
    assert adapter.metadata.championship_probability_key == probability_key

    for attr in (
        "fetch_games",
        "fetch_goalie_game_stats",
        "fetch_injuries_proxy",
        "fetch_public_odds_optional",
        "fetch_players",
        "build_results_from_games",
        "fetch_upcoming_schedule",
        "fetch_teams",
        "fetch_xg_optional",
    ):
        assert callable(getattr(adapter, attr))


def test_supported_leagues_are_explicit_and_stable() -> None:
    assert supported_leagues() == ("NHL", "NBA", "NCAAM")


@pytest.mark.parametrize(
    ("league", "slug", "config_path"),
    [
        ("NHL", "nhl", "configs/nhl.yaml"),
        ("NBA", "nba", "configs/nba.yaml"),
        ("NCAAM", "ncaam", "configs/ncaam.yaml"),
    ],
)
def test_data_refresh_steps_cover_all_supported_leagues_with_registry_defaults(league: str, slug: str, config_path: str) -> None:
    steps = {step.name: step for step in build_data_refresh_steps()}

    assert steps[f"{slug}:fetch"].command[-1] == config_path
    assert steps[f"{slug}:fetch-odds"].command[-1] == config_path
    assert steps[f"{slug}:fetch"].name.startswith(league.lower())


def _seed_query_db(tmp_path: Path) -> Database:
    db = Database(str(tmp_path / "parity.db"))
    db.init_schema()

    db.executemany(
        """
        INSERT INTO upcoming_game_forecasts(
          game_id, as_of_utc, game_date_utc, home_team, away_team,
          ensemble_prob_home_win, predicted_winner, per_model_probs_json,
          spread_min, spread_median, spread_max, spread_mean, spread_sd, spread_iqr,
          bayes_ci_low, bayes_ci_high, uncertainty_flags_json,
          snapshot_id, feature_set_version, model_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                10,
                "2026-03-01T00:00:00Z",
                "2026-03-05",
                "TOR",
                "MTL",
                0.62,
                "TOR",
                '{"glm_ridge":0.61}',
                0.55,
                0.61,
                0.66,
                0.61,
                0.03,
                0.04,
                0.54,
                0.69,
                '{"starter_unknown":true}',
                "s1",
                "f1",
                "r1",
            ),
            (
                20,
                "2026-03-01T00:00:00Z",
                "2026-03-05",
                "NYK",
                "CHI",
                0.59,
                "NYK",
                '{"glm_ridge":0.60}',
                0.52,
                0.59,
                0.64,
                0.58,
                0.02,
                0.03,
                0.51,
                0.66,
                '{"injury_noise":false}',
                "s1",
                "f1",
                "r1",
            ),
            (
                30,
                "2026-03-01T00:00:00Z",
                "2026-03-05",
                "DUKE",
                "UNC",
                0.64,
                "DUKE",
                '{"glm_ridge":0.63}',
                0.57,
                0.64,
                0.70,
                0.64,
                0.03,
                0.05,
                0.56,
                0.72,
                '{"tournament_noise":false}',
                "s1",
                "f1",
                "r1",
            ),
        ],
    )

    db.executemany(
        """
        INSERT INTO results(
          game_id, season, game_date_utc, final_utc, home_team, away_team,
          home_score, away_score, home_win, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (101, 20252026, "2026-01-05", "2026-01-05T04:00:00Z", "LAK", "SJS", 4, 1, 1, "2026-01-05T05:00:00Z"),
            (102, 20252026, "2026-01-06", "2026-01-06T04:00:00Z", "LAK", "ANA", 3, 2, 1, "2026-01-06T05:00:00Z"),
            (103, 20252026, "2026-01-07", "2026-01-07T04:00:00Z", "BOS", "TOR", 4, 2, 1, "2026-01-07T05:00:00Z"),
            (104, 20252026, "2026-01-08", "2026-01-08T04:00:00Z", "TBL", "FLA", 5, 3, 1, "2026-01-08T05:00:00Z"),
            (201, 20252026, "2026-01-05", "2026-01-05T05:00:00Z", "NYK", "CHI", 108, 101, 1, "2026-01-05T06:00:00Z"),
            (202, 20252026, "2026-01-06", "2026-01-06T05:00:00Z", "BOS", "NYK", 112, 104, 1, "2026-01-06T06:00:00Z"),
            (203, 20252026, "2026-01-07", "2026-01-07T05:00:00Z", "MIA", "ATL", 111, 103, 1, "2026-01-07T06:00:00Z"),
            (204, 20252026, "2026-01-08", "2026-01-08T05:00:00Z", "LAL", "DEN", 99, 105, 0, "2026-01-08T06:00:00Z"),
            (301, 20252026, "2026-01-05", "2026-01-05T03:00:00Z", "DUKE", "UNC", 82, 75, 1, "2026-01-05T04:00:00Z"),
            (302, 20252026, "2026-01-07", "2026-01-07T03:00:00Z", "DUKE", "UVA", 77, 69, 1, "2026-01-07T04:00:00Z"),
            (303, 20252026, "2026-01-10", "2026-01-10T03:00:00Z", "UNC", "DUKE", 70, 76, 0, "2026-01-10T04:00:00Z"),
        ],
    )

    return db


@pytest.mark.parametrize(
    ("question", "intent", "team", "league", "competition", "probability_key"),
    [
        ("What's the chance the Leafs win their next game?", "team_next_game", "TOR", "NHL", None, None),
        ("What's the chance the Knicks win the next game?", "team_next_game", "NYK", "NBA", None, None),
        ("What's the chance Duke wins the next game?", "team_next_game", "DUKE", "NCAAM", None, None),
        ("What's the probability the Kings win the Stanley Cup?", "team_championship", "LAK", "NHL", "Stanley Cup", "stanley_cup_prob"),
        ("What's the probability the Knicks win the NBA Finals?", "team_championship", "NYK", "NBA", "NBA Finals", "nba_finals_prob"),
        ("What's the probability Duke wins March Madness?", "team_championship", "DUKE", "NCAAM", "NCAA Tournament", "ncaa_tournament_prob"),
    ],
)
def test_query_handlers_hold_shared_parity_contract(
    tmp_path: Path,
    question: str,
    intent: str,
    team: str,
    league: str,
    competition: str | None,
    probability_key: str | None,
) -> None:
    db = _seed_query_db(tmp_path)

    answer, payload = answer_question(db, question)

    assert payload["intent"] == intent
    assert payload["team"] == team
    assert payload["league"] == league
    assert team in answer

    if competition is None:
        assert 0 < payload["ensemble_prob_team_win"] < 1
    else:
        assert payload["competition"] == competition
        assert probability_key is not None
        assert 0 < payload[probability_key] < 1
        assert "Heuristic estimate" in answer
