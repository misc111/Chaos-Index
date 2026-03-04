from __future__ import annotations


def team_prob_answer(team: str, opponent: str, date: str, win_prob: float) -> str:
    return f"{team} win probability vs {opponent} on {date}: {win_prob:.1%}."



def best_model_answer(model: str, window_days: int, log_loss: float) -> str:
    return f"Best model over the last {window_days} days by log loss: {model} (log loss {log_loss:.4f})."



def help_answer() -> str:
    return (
        "I can answer forecast and performance questions from local SQLite data, for example: "
        "'What's the chance the Leafs win their next game?' or 'Which model has performed best the last 60 days?'"
    )
