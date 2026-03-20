from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.query.answer import answer_question
from src.storage.db import Database



def test_query_answers(tmp_path: Path):
    db = Database(str(tmp_path / "q.db"))
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
            (10, "2026-03-01T00:00:00Z", "2026-03-05", "TOR", "MTL", 0.62, "TOR", '{"glm_ridge":0.61}', 0.55, 0.61, 0.66, 0.61, 0.03, 0.04, 0.54, 0.69, '{"starter_unknown":true}', "s1", "f1", "r1"),
            (11, "2026-03-01T00:00:00Z", "2026-03-06", "BOS", "NJD", 0.57, "BOS", '{"glm_ridge":0.58}', 0.50, 0.57, 0.61, 0.56, 0.02, 0.03, 0.49, 0.64, '{"starter_unknown":false}', "s1", "f1", "r1"),
            (12, "2026-03-01T00:00:00Z", "2026-03-07", "TBL", "FLA", 0.55, "TBL", '{"glm_ridge":0.56}', 0.49, 0.55, 0.60, 0.55, 0.02, 0.03, 0.48, 0.62, '{"starter_unknown":false}', "s1", "f1", "r1"),
            (13, "2026-03-01T00:00:00Z", "2026-03-06", "BOS", "TOR", 0.57, "BOS", '{"glm_ridge":0.57}', 0.50, 0.56, 0.61, 0.56, 0.03, 0.04, 0.49, 0.63, '{"starter_unknown":false}', "s1", "f1", "r1"),
            (14, "2026-03-01T00:00:00Z", "2026-03-07", "TOR", "OTT", 0.54, "TOR", '{"glm_ridge":0.55}', 0.49, 0.54, 0.60, 0.54, 0.03, 0.04, 0.47, 0.61, '{"starter_unknown":false}', "s1", "f1", "r1"),
            (20, "2026-03-01T00:00:00Z", "2026-03-05", "NYK", "CHI", 0.59, "NYK", '{"glm_ridge":0.60}', 0.52, 0.59, 0.64, 0.58, 0.02, 0.03, 0.51, 0.66, '{"injury_noise":false}', "s1", "f1", "r1"),
            (21, "2026-03-01T00:00:00Z", "2026-03-06", "MIA", "NYK", 0.56, "MIA", '{"glm_ridge":0.57}', 0.50, 0.56, 0.61, 0.56, 0.02, 0.03, 0.49, 0.63, '{"injury_noise":false}', "s1", "f1", "r1"),
            (30, "2026-03-01T00:00:00Z", "2026-03-05", "DUKE", "UNC", 0.64, "DUKE", '{"glm_ridge":0.63}', 0.57, 0.64, 0.70, 0.64, 0.03, 0.05, 0.56, 0.72, '{"tournament_noise":false}', "s1", "f1", "r1"),
        ],
    )

    db.executemany(
        """
        INSERT INTO model_scores(
          game_id, model_name, model_run_id, as_of_utc, game_date_utc,
          prob_home_win, outcome_home_win, log_loss, brier, accuracy, scored_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, "glm_ridge", "r1", "2026-02-01T00:00:00Z", "2026-02-02", 0.6, 1, 0.51, 0.16, 1, "2026-02-03T00:00:00Z"),
            (2, "rf", "r2", "2026-02-01T00:00:00Z", "2026-02-03", 0.4, 0, 0.55, 0.18, 1, "2026-02-04T00:00:00Z"),
        ],
    )

    db.executemany(
        """
        INSERT INTO teams(
          league, team_abbrev, team_name, conference, division,
          as_of_date, as_of_utc, snapshot_id, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("NHL", "BOS", "Boston Bruins", "E", "Atlantic", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NHL", "FLA", "Florida Panthers", "E", "Atlantic", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NHL", "MTL", "Montreal Canadiens", "E", "Atlantic", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NHL", "NJD", "New Jersey Devils", "E", "Metropolitan", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NHL", "TOR", "Toronto Maple Leafs", "E", "Atlantic", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NHL", "TBL", "Tampa Bay Lightning", "E", "Atlantic", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NHL", "WPG", "Winnipeg Jets", "W", "Central", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NBA", "BOS", "Boston Celtics", "East", "Atlantic", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NBA", "CHI", "Chicago Bulls", "East", "Central", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NBA", "MIA", "Miami Heat", "East", "Southeast", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NBA", "NYK", "New York Knicks", "East", "Atlantic", "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NCAAM", "DUKE", "Duke Blue Devils", "ACC", None, "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
            ("NCAAM", "UNC", "North Carolina Tar Heels", "ACC", None, "2026-03-01", "2026-03-01T00:00:00Z", "s1", "{}"),
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
            (103, 20252026, "2026-01-07", "2026-01-07T04:00:00Z", "LAK", "EDM", 1, 2, 0, "2026-01-07T05:00:00Z"),
            (104, 20252026, "2026-01-08", "2026-01-08T04:00:00Z", "TBL", "FLA", 5, 3, 1, "2026-01-08T05:00:00Z"),
            (105, 20252026, "2026-01-09", "2026-01-09T04:00:00Z", "NJD", "NYR", 3, 1, 1, "2026-01-09T05:00:00Z"),
            (106, 20252026, "2026-01-10", "2026-01-10T04:00:00Z", "BOS", "MTL", 4, 2, 1, "2026-01-10T05:00:00Z"),
            (107, 20252026, "2026-01-11", "2026-01-11T04:00:00Z", "TOR", "DET", 3, 2, 1, "2026-01-11T05:00:00Z"),
            (108, 20252026, "2026-01-12", "2026-01-12T04:00:00Z", "LAK", "VGK", 4, 2, 1, "2026-01-12T05:00:00Z"),
            (109, 20252026, "2026-01-13", "2026-01-13T04:00:00Z", "USA", "CAN", 3, 2, 1, "2026-01-13T05:00:00Z"),
            (201, 20252026, "2026-01-05", "2026-01-05T05:00:00Z", "NYK", "CHI", 108, 101, 1, "2026-01-05T06:00:00Z"),
            (202, 20252026, "2026-01-06", "2026-01-06T05:00:00Z", "BOS", "NYK", 112, 104, 1, "2026-01-06T06:00:00Z"),
            (203, 20252026, "2026-01-07", "2026-01-07T05:00:00Z", "LAL", "NYK", 99, 105, 0, "2026-01-07T06:00:00Z"),
            (204, 20252026, "2026-01-08", "2026-01-08T05:00:00Z", "NYK", "BKN", 110, 103, 1, "2026-01-08T06:00:00Z"),
            (205, 20252026, "2026-01-09", "2026-01-09T05:00:00Z", "BOS", "LAL", 115, 111, 1, "2026-01-09T06:00:00Z"),
            (301, 20252026, "2026-01-05", "2026-01-05T03:00:00Z", "DUKE", "UNC", 82, 75, 1, "2026-01-05T04:00:00Z"),
            (302, 20252026, "2026-01-07", "2026-01-07T03:00:00Z", "DUKE", "UVA", 77, 69, 1, "2026-01-07T04:00:00Z"),
            (303, 20252026, "2026-01-10", "2026-01-10T03:00:00Z", "DUKE", "WAKE", 74, 71, 1, "2026-01-10T04:00:00Z"),
            (304, 20252026, "2026-01-12", "2026-01-12T03:00:00Z", "UNC", "DUKE", 70, 76, 0, "2026-01-12T04:00:00Z"),
        ],
    )

    ans1, payload1 = answer_question(db, "What's the chance the Leafs win their next game?")
    assert "TOR" in ans1
    assert payload1["intent"] == "team_next_game"
    assert payload1["team"] == "TOR"
    assert payload1["league"] == "NHL"

    ans1b, payload1b = answer_question(db, "What's the chance that Toronto wins the next game?", default_league="NHL")
    assert "TOR" in ans1b
    assert payload1b["intent"] == "team_next_game"
    assert payload1b["team"] == "TOR"

    ans1c, payload1c = answer_question(db, "What are New Jersey's odds?")
    assert "NJD" in ans1c
    assert payload1c["intent"] == "team_next_game"
    assert payload1c["team"] == "NJD"

    ans1d, payload1d = answer_question(db, "lightning tonight?")
    assert "TBL" in ans1d
    assert payload1d["intent"] == "team_next_game"
    assert payload1d["team"] == "TBL"

    _, payload1e = answer_question(db, "What's the probability the Leafs win the next three games?")
    assert payload1e["intent"] == "team_next_n_games"
    assert payload1e["team"] == "TOR"
    assert payload1e["n_games_requested"] == 3
    assert payload1e["n_games_returned"] == 3
    assert len(payload1e["games"]) == 3
    assert 0 < payload1e["aggregate"]["prob_win_all"] < 1

    ans1f, payload1f = answer_question(db, "What's the probability the Kings win the Stanley Cup?")
    assert payload1f["intent"] == "team_championship"
    assert payload1f["league"] == "NHL"
    assert payload1f["competition"] == "Stanley Cup"
    assert payload1f["team"] == "LAK"
    assert 0 < payload1f["stanley_cup_prob"] < 1
    assert payload1f["interval_90"]["low"] <= payload1f["stanley_cup_prob"] <= payload1f["interval_90"]["high"]
    assert all(t["team"] not in {"USA", "CAN"} for t in payload1f["top_teams"])
    assert "Heuristic estimate" in ans1f

    ans_nba, payload_nba = answer_question(db, "What's the chance the Knicks win the next game?")
    assert payload_nba["intent"] == "team_next_game"
    assert payload_nba["team"] == "NYK"
    assert payload_nba["league"] == "NBA"
    assert "NYK" in ans_nba

    ans_nba_finals, payload_nba_finals = answer_question(db, "What's the probability the Knicks win the NBA Finals?")
    assert payload_nba_finals["intent"] == "team_championship"
    assert payload_nba_finals["league"] == "NBA"
    assert payload_nba_finals["competition"] == "NBA Finals"
    assert payload_nba_finals["team"] == "NYK"
    assert 0 < payload_nba_finals["nba_finals_prob"] < 1
    assert payload_nba_finals["interval_90"]["low"] <= payload_nba_finals["nba_finals_prob"] <= payload_nba_finals["interval_90"]["high"]
    assert "Heuristic estimate" in ans_nba_finals

    ans_ncaam, payload_ncaam = answer_question(db, "What's the chance Duke wins the next game?", default_league="NCAAM")
    assert payload_ncaam["intent"] == "team_next_game"
    assert payload_ncaam["team"] == "DUKE"
    assert payload_ncaam["league"] == "NCAAM"
    assert "DUKE" in ans_ncaam

    ans_ncaam_title, payload_ncaam_title = answer_question(db, "What are the odds Duke wins March Madness?", default_league="NCAAM")
    assert payload_ncaam_title["intent"] == "team_championship"
    assert payload_ncaam_title["league"] == "NCAAM"
    assert payload_ncaam_title["competition"] == "NCAA Tournament"
    assert payload_ncaam_title["team"] == "DUKE"
    assert 0 < payload_ncaam_title["ncaa_tournament_prob"] < 1
    assert "Heuristic estimate" in ans_ncaam_title

    _, payload2 = answer_question(db, "Which model has performed best the last 60 days?")
    assert payload2["intent"] == "best_model"
    assert len(payload2["leaderboard"]) >= 1

    report_answer, report_payload = answer_question(db, "Give me the report of all teams in a table")
    assert report_payload["intent"] == "league_report"
    assert report_payload["league"] == "NBA"
    assert report_payload["as_of_utc"] == "2026-03-01T00:00:00Z"
    assert report_payload["model_columns"][:2] == ["ensemble", "glm_ridge"]
    assert "Model trust guide (super brief)" in report_answer
    assert "Home Team | Away Team | Date" in report_answer
    assert report_answer.count("| NYK | CHI | 2026-03-05 | 59.0% | 60.0% |") == 1

    nyk_row = next(r for r in report_payload["rows"] if r["team"] == "NYK")
    assert nyk_row["division"] == "Atlantic"
    assert nyk_row["next_opponent"] == "CHI"
    assert nyk_row["home_team"] == "NYK"
    assert nyk_row["away_team"] == "CHI"
    assert nyk_row["next_game_date_utc"] == "2026-03-05"
    assert nyk_row["home_or_away"] == "Home"
    assert 0 < nyk_row["model_win_probabilities"]["ensemble"] < 1

    bos_row = next(r for r in report_payload["rows"] if r["team"] == "BOS")
    assert bos_row["next_opponent"] is None


def test_query_answers_bet_history_summary_and_cumulative(tmp_path: Path):
    db = Database(str(tmp_path / "bets.db"))
    db.init_schema()

    today_central = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Chicago")).date()
    yesterday = (today_central - timedelta(days=1)).isoformat()
    two_days_ago = (today_central - timedelta(days=2)).isoformat()

    db.executemany(
        """
        INSERT INTO results(
          game_id, season, game_date_utc, final_utc, home_team, away_team,
          home_score, away_score, home_win, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, 20252026, yesterday, f"{yesterday}T05:00:00Z", "NYK", "CHI", 108, 101, 1, f"{yesterday}T06:00:00Z"),
            (2, 20252026, yesterday, f"{yesterday}T05:30:00Z", "MIA", "BOS", 104, 109, 0, f"{yesterday}T06:30:00Z"),
            (3, 20252026, yesterday, f"{yesterday}T06:00:00Z", "LAL", "DEN", 99, 105, 0, f"{yesterday}T07:00:00Z"),
            (4, 20252026, two_days_ago, f"{two_days_ago}T05:00:00Z", "DAL", "PHX", 112, 108, 1, f"{two_days_ago}T06:00:00Z"),
        ],
    )
    db.executemany(
        """
        INSERT INTO historical_bet_decisions_by_profile(
          strategy, sizing_style, game_id, date_central, forecast_as_of_utc, forecast_model_run_id,
          odds_as_of_utc, odds_snapshot_id, home_team, away_team, home_win_probability,
          home_moneyline, away_moneyline, bet_label, reason, side, team, stake, odds,
          model_probability, market_probability, edge, expected_value, stake_unit_dollars,
          strategy_config_signature, decision_logic_version, materialization_version, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("riskAdjusted", "continuous", 1, yesterday, f"{yesterday}T00:00:00Z", "r1", f"{yesterday}T00:10:00Z", "s1", "NYK", "CHI", 0.59, -120, 105, "$50 NYK", "Favorite underpriced", "home", "NYK", 50.0, -120.0, 0.59, 0.5455, 0.0445, 0.031, 100.0, "sig1", "v1", "m1", f"{yesterday}T00:20:00Z"),
            ("riskAdjusted", "continuous", 2, yesterday, f"{yesterday}T00:00:00Z", "r1", f"{yesterday}T00:10:00Z", "s1", "MIA", "BOS", 0.44, 110, -130, "$50 BOS", "Underdog underpriced", "away", "BOS", 50.0, -130.0, 0.56, 0.4348, 0.1252, 0.062, 100.0, "sig1", "v1", "m1", f"{yesterday}T00:20:00Z"),
            ("riskAdjusted", "continuous", 3, yesterday, f"{yesterday}T00:00:00Z", "r1", f"{yesterday}T00:10:00Z", "s1", "LAL", "DEN", 0.51, -105, -105, "$0", "Too close", "none", None, 0.0, None, None, None, None, None, 100.0, "sig1", "v1", "m1", f"{yesterday}T00:20:00Z"),
            ("riskAdjusted", "continuous", 4, two_days_ago, f"{two_days_ago}T00:00:00Z", "r1", f"{two_days_ago}T00:10:00Z", "s1", "DAL", "PHX", 0.57, -110, 100, "$40 DAL", "Favorite underpriced", "home", "DAL", 40.0, -110.0, 0.57, 0.5238, 0.0462, 0.028, 100.0, "sig1", "v1", "m1", f"{two_days_ago}T00:20:00Z"),
        ],
    )

    answer, payload = answer_question(
        db,
        "How much money did I win/lose last night? Be brief in your summary.",
    )
    assert payload["intent"] == "bet_history_summary"
    assert payload["league"] == "NBA"
    assert payload["period"] == "yesterday"
    assert payload["strategy"] == "riskAdjusted"
    assert payload["sizing_style"] == "continuous"
    assert payload["summary"]["tracked_games"] == 3
    assert payload["summary"]["settled_bets"] == 2
    assert payload["summary"]["wins"] == 2
    assert payload["summary"]["losses"] == 0
    assert round(payload["summary"]["total_risked"], 2) == 100.00
    assert round(payload["summary"]["total_profit"], 2) == 80.13
    assert len(payload["games"]) == 3
    assert answer.startswith(f"NBA last night ({yesterday}): +$80.13 net, $100.00 risked, 2-0 on 2 bets.")
    assert "| Game | Bet on | Winner | P/L | Bet rationale |" in answer
    assert "| CHI @ NYK | NYK | NYK | +$41.67 | New York was the favorite but underpriced. |" in answer
    assert "| BOS @ MIA | BOS | BOS | +$38.46 | Boston was the favorite but underpriced. |" in answer
    assert any(game["reason"] == "Too close" and game["outcome"] == "no_bet" for game in payload["games"])
    assert any(game["bet_rationale"] == "No bet because the game was too close." for game in payload["games"])

    cumulative_answer, cumulative_payload = answer_question(
        db,
        "What are my cumulative net profits or losses and how much have I risked since the beginning of tracking?",
    )
    assert cumulative_payload["intent"] == "bet_history_summary"
    assert cumulative_payload["league"] == "NBA"
    assert cumulative_payload["period"] == "all_time"
    assert round(cumulative_payload["summary"]["total_risked"], 2) == 140.00
    assert round(cumulative_payload["summary"]["total_profit"], 2) == 116.49
    assert cumulative_payload["games"] == []
    assert "since tracking started" in cumulative_answer

    recap_answer, recap_payload = answer_question(
        db,
        "How'd I do last night on my bets?",
    )
    assert recap_payload["intent"] == "bet_history_summary"
    assert recap_payload["period"] == "yesterday"
    assert recap_answer.startswith(f"NBA last night ({yesterday}): +$80.13 net, $100.00 risked, 2-0 on 2 bets.")
    assert "| Game | Bet on | Winner | P/L | Bet rationale |" in recap_answer

    net_profit_answer, net_profit_payload = answer_question(
        db,
        "What was my net profit last night?",
    )
    assert net_profit_payload["intent"] == "bet_history_summary"
    assert net_profit_payload["period"] == "yesterday"
    assert "| Game | Bet on | Winner | P/L | Bet rationale |" in net_profit_answer


def test_query_answers_bet_history_prefers_capital_preservation_default_profile_for_nba(tmp_path: Path):
    db = Database(str(tmp_path / "bets-v2.db"))
    db.init_schema()

    today_central = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Chicago")).date()
    yesterday = (today_central - timedelta(days=1)).isoformat()

    db.executemany(
        """
        INSERT INTO results(
          game_id, season, game_date_utc, final_utc, home_team, away_team,
          home_score, away_score, home_win, ingested_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, 20252026, yesterday, f"{yesterday}T05:00:00Z", "NYK", "CHI", 108, 101, 1, f"{yesterday}T06:00:00Z"),
            (2, 20252026, yesterday, f"{yesterday}T05:30:00Z", "MIA", "BOS", 104, 109, 0, f"{yesterday}T06:30:00Z"),
        ],
    )
    db.executemany(
        """
        INSERT INTO historical_bet_decisions_by_profile_v2(
          strategy, sizing_style, strategy_config_signature, game_id, date_central, forecast_as_of_utc, forecast_model_run_id,
          odds_as_of_utc, odds_snapshot_id, home_team, away_team, home_win_probability,
          home_moneyline, away_moneyline, bet_label, reason, side, team, stake, odds,
          model_probability, market_probability, edge, expected_value, stake_unit_dollars,
          decision_logic_version, materialization_version, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "riskAdjusted",
                "default",
                "sig-risk",
                1,
                yesterday,
                f"{yesterday}T00:00:00Z",
                "r1",
                f"{yesterday}T00:10:00Z",
                "s1",
                "NYK",
                "CHI",
                0.59,
                -120,
                105,
                "$50 NYK",
                "Favorite underpriced",
                "home",
                "NYK",
                50.0,
                -120.0,
                0.59,
                0.5455,
                0.0445,
                0.031,
                100.0,
                "v1",
                "m1",
                f"{yesterday}T00:20:00Z",
            ),
            (
                "capitalPreservation",
                "default",
                "sig-conservative",
                2,
                yesterday,
                f"{yesterday}T00:00:00Z",
                "r1",
                f"{yesterday}T00:10:00Z",
                "s1",
                "MIA",
                "BOS",
                0.44,
                110,
                -130,
                "$75 BOS",
                "Favorite underpriced",
                "away",
                "BOS",
                75.0,
                -130.0,
                0.56,
                0.4348,
                0.1252,
                0.062,
                100.0,
                "v1",
                "m1",
                f"{yesterday}T00:20:01Z",
            ),
        ],
    )

    answer, payload = answer_question(db, "How much money did I win or lose last night?")

    assert payload["intent"] == "bet_history_summary"
    assert payload["league"] == "NBA"
    assert payload["period"] == "yesterday"
    assert payload["strategy"] == "capitalPreservation"
    assert payload["sizing_style"] == "default"
    assert payload["source_table"] == "historical_bet_decisions_by_profile_v2"
    assert payload["summary"]["tracked_games"] == 1
    assert payload["summary"]["settled_bets"] == 1
    assert payload["summary"]["wins"] == 1
    assert payload["summary"]["losses"] == 0
    assert round(payload["summary"]["total_risked"], 2) == 75.00
    assert round(payload["summary"]["total_profit"], 2) == 57.69
    assert answer.startswith(f"NBA last night ({yesterday}): +$57.69 net, $75.00 risked, 1-0 on 1 bets.")
