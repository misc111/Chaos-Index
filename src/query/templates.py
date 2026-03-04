from __future__ import annotations


def team_prob_answer(team: str, opponent: str, date: str, win_prob: float) -> str:
    return f"{team} win probability vs {opponent} on {date}: {win_prob:.1%}."


def team_multi_game_answer(
    team: str,
    n_games_requested: int,
    n_games_returned: int,
    prob_win_all: float,
    expected_wins: float,
    games_summary: str,
) -> str:
    prefix = ""
    if n_games_returned < n_games_requested:
        prefix = f"Only {n_games_returned} upcoming forecasted games are currently stored for {team}. "
    return (
        f"{prefix}{team} win probabilities over next {n_games_returned} game(s): {games_summary}. "
        f"Probability of winning all {n_games_returned}: {prob_win_all:.1%}. "
        f"Expected wins: {expected_wins:.2f}."
    )


def stanley_cup_answer(team: str, cup_prob: float, low_90: float, high_90: float, as_of_date: str) -> str:
    return (
        f"Estimated {team} Stanley Cup win probability as of {as_of_date}: {cup_prob:.1%} "
        f"(90% interval {low_90:.1%}-{high_90:.1%}). "
        "Heuristic estimate from current-season results, not a full bracket simulation."
    )


def best_model_answer(model: str, window_days: int, log_loss: float) -> str:
    return f"Best model over the last {window_days} days by log loss: {model} (log loss {log_loss:.4f})."


def help_answer() -> str:
    return (
        "I can answer forecast and performance questions from local SQLite data, for example: "
        "'What's the chance the Leafs win their next game?', "
        "'What are Toronto's odds in the next three games?', "
        "'What's the probability the Kings win the Stanley Cup?', or "
        "'Which model has performed best the last 60 days?'"
    )
