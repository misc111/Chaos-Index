import json
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
        "INSERT INTO model_scores(game_id, model_name, model_run_id, as_of_utc, game_date_utc, prob_home_win, outcome_home_win, log_loss, brier, accuracy, scored_at_utc) VALUES (1, 'glm_logit', 'r1', '2026-02-01T00:00:00Z', '2026-02-02', 0.6, 1, 0.51, 0.16, 1, '2026-02-03T00:00:00Z')"
    )
    db.execute(
        "INSERT INTO model_scores(game_id, model_name, model_run_id, as_of_utc, game_date_utc, prob_home_win, outcome_home_win, log_loss, brier, accuracy, scored_at_utc) VALUES (2, 'rf', 'r2', '2026-02-01T00:00:00Z', '2026-02-03', 0.4, 0, 0.55, 0.18, 1, '2026-02-04T00:00:00Z')"
    )

    ans1, payload1 = answer_question(db, "What's the chance the Leafs win their next game?")
    assert "TOR" in ans1
    assert payload1["intent"] == "team_next_game"

    ans2, payload2 = answer_question(db, "Which model has performed best the last 60 days?")
    assert payload2["intent"] == "best_model"
    assert len(payload2["leaderboard"]) >= 1
