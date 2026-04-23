from __future__ import annotations

import pytest

from src.common.market_transforms import (
    american_price_to_implied_probability,
    probability_to_logit,
    vig_free_two_way_logit_offsets,
    vig_free_two_way_probabilities,
)


def test_vig_free_two_way_probabilities_and_logits_from_implied_probabilities() -> None:
    home_vig_free, away_vig_free = vig_free_two_way_probabilities(
        0.5454545,
        0.4761905,
        input_scale="implied_probability",
    )
    home_logit, away_logit = vig_free_two_way_logit_offsets(
        0.5454545,
        0.4761905,
        input_scale="implied_probability",
    )

    assert home_vig_free + away_vig_free == pytest.approx(1.0)
    assert home_vig_free == pytest.approx(0.5338983, rel=1e-5)
    assert away_vig_free == pytest.approx(0.4661017, rel=1e-5)
    assert home_logit == pytest.approx(probability_to_logit(home_vig_free))
    assert away_logit == pytest.approx(probability_to_logit(away_vig_free))
    assert home_logit == pytest.approx(-away_logit)


def test_vig_free_probabilities_from_american_prices_match_implied_probability_path() -> None:
    home_from_price = american_price_to_implied_probability(-120)
    away_from_price = american_price_to_implied_probability(+110)
    home_vig_free_price, away_vig_free_price = vig_free_two_way_probabilities(-120, +110, input_scale="american_price")
    home_vig_free_implied, away_vig_free_implied = vig_free_two_way_probabilities(
        home_from_price,
        away_from_price,
        input_scale="implied_probability",
    )

    assert home_from_price == pytest.approx(0.5454545, rel=1e-5)
    assert away_from_price == pytest.approx(0.4761905, rel=1e-5)
    assert home_vig_free_price == pytest.approx(home_vig_free_implied)
    assert away_vig_free_price == pytest.approx(away_vig_free_implied)


def test_vig_free_helpers_fail_on_invalid_probability_inputs() -> None:
    with pytest.raises(ValueError, match="must stay within \\[0, 1\\]"):
        vig_free_two_way_probabilities(1.2, 0.4, input_scale="implied_probability")
