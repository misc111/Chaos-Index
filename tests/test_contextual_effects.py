import pandas as pd
import pytest

from src.features.contextual_effects import compute_causal_group_effects


def test_compute_causal_group_effects_uses_only_prior_finalized_games():
    games = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "venue": ["A", "B", "A", "A"],
            "start_time_utc": [
                "2025-11-01T00:00:00Z",
                "2025-11-02T00:00:00Z",
                "2025-11-03T00:00:00Z",
                "2025-11-04T00:00:00Z",
            ],
            "status_final": [1, 1, 1, 0],
            "margin": [10.0, -4.0, 2.0, 0.0],
        }
    )

    effects = compute_causal_group_effects(
        games,
        group_col="venue",
        metric_columns={"arena_margin_effect": "margin"},
        shrinkage=20.0,
    ).set_index("game_id")

    assert float(effects.loc[1, "arena_margin_effect"]) == 0.0
    assert float(effects.loc[3, "arena_margin_effect"]) == pytest.approx(10.0 / 21.0)
    assert float(effects.loc[4, "arena_margin_effect"]) == pytest.approx(12.0 / 22.0)
