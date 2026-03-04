from pathlib import Path

from src.query.answer import answer_question
from src.storage.db import Database



def test_query_answers(tmp_path: Path):
    db = Database(str(tmp_path / "q.db"))
    db.init_schema()

    db.execute(
        "INSERT INTO upcoming_game_forecasts(game_id, as_of_utc, game_date_utc, home_team, away_team, ensemble_prob_home_win, predicted_winner, per_model_probs_json, spread_min, spread_median, spread_max, spread_mean, spread_sd, spread_iqr, bayes_ci_low, bayes_ci_high, uncertainty_flags_json, snapshot_id, feature_set_version, model_run_id) VALUES (10, '2026-03-01T00:00:00Z', '2026-03-05', 'TOR', 'MTL', 0.62, 'TOR', '{\"glm_logit\":0.61}', 0.55, 0.61, 0.66, 0.61, 0.03, 0.04, 0.54, 0.69, '{\"starter_unknown\":true}', 's1', 'f1', 'r1')"
    )
    db.execute(
        "INSERT INTO upcoming_game_forecasts(game_id, as_of_utc, game_date_utc, home_team, away_team, ensemble_prob_home_win, predicted_winner, per_model_probs_json, spread_min, spread_median, spread_max, spread_mean, spread_sd, spread_iqr, bayes_ci_low, bayes_ci_high, uncertainty_flags_json, snapshot_id, feature_set_version, model_run_id) VALUES (11, '2026-03-01T00:00:00Z', '2026-03-06', 'BOS', 'NJD', 0.57, 'BOS', '{\"glm_logit\":0.58}', 0.50, 0.57, 0.61, 0.56, 0.02, 0.03, 0.49, 0.64, '{\"starter_unknown\":false}', 's1', 'f1', 'r1')"
    )
    db.execute(
        "INSERT INTO upcoming_game_forecasts(game_id, as_of_utc, game_date_utc, home_team, away_team, ensemble_prob_home_win, predicted_winner, per_model_probs_json, spread_min, spread_median, spread_max, spread_mean, spread_sd, spread_iqr, bayes_ci_low, bayes_ci_high, uncertainty_flags_json, snapshot_id, feature_set_version, model_run_id) VALUES (12, '2026-03-01T00:00:00Z', '2026-03-07', 'TBL', 'FLA', 0.55, 'TBL', '{\"glm_logit\":0.56}', 0.49, 0.55, 0.60, 0.55, 0.02, 0.03, 0.48, 0.62, '{\"starter_unknown\":false}', 's1', 'f1', 'r1')"
    )
    db.execute(
        "INSERT INTO upcoming_game_forecasts(game_id, as_of_utc, game_date_utc, home_team, away_team, ensemble_prob_home_win, predicted_winner, per_model_probs_json, spread_min, spread_median, spread_max, spread_mean, spread_sd, spread_iqr, bayes_ci_low, bayes_ci_high, uncertainty_flags_json, snapshot_id, feature_set_version, model_run_id) VALUES (13, '2026-03-01T00:00:00Z', '2026-03-06', 'BOS', 'TOR', 0.57, 'BOS', '{\"glm_logit\":0.57}', 0.50, 0.56, 0.61, 0.56, 0.03, 0.04, 0.49, 0.63, '{\"starter_unknown\":false}', 's1', 'f1', 'r1')"
    )
    db.execute(
        "INSERT INTO upcoming_game_forecasts(game_id, as_of_utc, game_date_utc, home_team, away_team, ensemble_prob_home_win, predicted_winner, per_model_probs_json, spread_min, spread_median, spread_max, spread_mean, spread_sd, spread_iqr, bayes_ci_low, bayes_ci_high, uncertainty_flags_json, snapshot_id, feature_set_version, model_run_id) VALUES (14, '2026-03-01T00:00:00Z', '2026-03-07', 'TOR', 'OTT', 0.54, 'TOR', '{\"glm_logit\":0.55}', 0.49, 0.54, 0.60, 0.54, 0.03, 0.04, 0.47, 0.61, '{\"starter_unknown\":false}', 's1', 'f1', 'r1')"
    )
    db.execute(
        "INSERT INTO model_scores(game_id, model_name, model_run_id, as_of_utc, game_date_utc, prob_home_win, outcome_home_win, log_loss, brier, accuracy, scored_at_utc) VALUES (1, 'glm_logit', 'r1', '2026-02-01T00:00:00Z', '2026-02-02', 0.6, 1, 0.51, 0.16, 1, '2026-02-03T00:00:00Z')"
    )
    db.execute(
        "INSERT INTO model_scores(game_id, model_name, model_run_id, as_of_utc, game_date_utc, prob_home_win, outcome_home_win, log_loss, brier, accuracy, scored_at_utc) VALUES (2, 'rf', 'r2', '2026-02-01T00:00:00Z', '2026-02-03', 0.4, 0, 0.55, 0.18, 1, '2026-02-04T00:00:00Z')"
    )
    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (101, 20252026, '2026-01-05', '2026-01-05T04:00:00Z', 'LAK', 'SJS', 4, 1, 1, '2026-01-05T05:00:00Z')"
    )
    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (102, 20252026, '2026-01-06', '2026-01-06T04:00:00Z', 'LAK', 'ANA', 3, 2, 1, '2026-01-06T05:00:00Z')"
    )
    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (103, 20252026, '2026-01-07', '2026-01-07T04:00:00Z', 'LAK', 'EDM', 1, 2, 0, '2026-01-07T05:00:00Z')"
    )
    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (104, 20252026, '2026-01-08', '2026-01-08T04:00:00Z', 'TBL', 'FLA', 5, 3, 1, '2026-01-08T05:00:00Z')"
    )
    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (105, 20252026, '2026-01-09', '2026-01-09T04:00:00Z', 'NJD', 'NYR', 3, 1, 1, '2026-01-09T05:00:00Z')"
    )
    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (106, 20252026, '2026-01-10', '2026-01-10T04:00:00Z', 'BOS', 'MTL', 4, 2, 1, '2026-01-10T05:00:00Z')"
    )
    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (107, 20252026, '2026-01-11', '2026-01-11T04:00:00Z', 'TOR', 'DET', 3, 2, 1, '2026-01-11T05:00:00Z')"
    )
    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (108, 20252026, '2026-01-12', '2026-01-12T04:00:00Z', 'LAK', 'VGK', 4, 2, 1, '2026-01-12T05:00:00Z')"
    )
    db.execute(
        "INSERT INTO results(game_id, season, game_date_utc, final_utc, home_team, away_team, home_score, away_score, home_win, ingested_at_utc) VALUES (109, 20252026, '2026-01-13', '2026-01-13T04:00:00Z', 'USA', 'CAN', 3, 2, 1, '2026-01-13T05:00:00Z')"
    )

    ans1, payload1 = answer_question(db, "What's the chance the Leafs win their next game?")
    assert "TOR" in ans1
    assert payload1["intent"] == "team_next_game"
    assert payload1["team"] == "TOR"

    ans1b, payload1b = answer_question(db, "What's the chance that Toronto wins the next game?")
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

    ans1e, payload1e = answer_question(db, "What's the probability Toronto wins the next three games?")
    assert payload1e["intent"] == "team_next_n_games"
    assert payload1e["team"] == "TOR"
    assert payload1e["n_games_requested"] == 3
    assert payload1e["n_games_returned"] == 3
    assert len(payload1e["games"]) == 3
    assert 0 < payload1e["aggregate"]["prob_win_all"] < 1

    ans1f, payload1f = answer_question(db, "What's the probability the Kings win the Stanley Cup?")
    assert payload1f["intent"] == "team_stanley_cup"
    assert payload1f["team"] == "LAK"
    assert 0 < payload1f["stanley_cup_prob"] < 1
    assert payload1f["interval_90"]["low"] <= payload1f["stanley_cup_prob"] <= payload1f["interval_90"]["high"]
    assert all(t["team"] not in {"USA", "CAN"} for t in payload1f["top_teams"])
    assert "Heuristic estimate" in ans1f

    ans2, payload2 = answer_question(db, "Which model has performed best the last 60 days?")
    assert payload2["intent"] == "best_model"
    assert len(payload2["leaderboard"]) >= 1
