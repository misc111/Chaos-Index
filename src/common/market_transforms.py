from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PROBABILITY_EPS = 1e-6
_INPUT_SCALE_ALIASES = {
    "implied_probability": "implied_probability",
    "probability": "implied_probability",
    "prob": "implied_probability",
    "american_price": "american_price",
    "american": "american_price",
    "moneyline": "american_price",
}


def _normalize_input_scale(input_scale: str) -> str:
    token = str(input_scale or "").strip().lower().replace("-", "_")
    normalized = _INPUT_SCALE_ALIASES.get(token)
    if normalized is None:
        raise ValueError(
            f"Unsupported market input scale '{input_scale}'. "
            f"Valid={sorted(_INPUT_SCALE_ALIASES)}"
        )
    return normalized


def _to_numeric_array(values: Any, *, label: str) -> tuple[np.ndarray, bool, pd.Index | None]:
    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return arr, True, values.index
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(float), False, None


def _restore_shape(values: np.ndarray, *, was_series: bool, index: pd.Index | None, as_scalar: bool) -> np.ndarray | pd.Series | float:
    if as_scalar:
        return float(values[0])
    if was_series and index is not None:
        return pd.Series(values, index=index)
    return values


def american_price_to_implied_probability(prices: Any) -> np.ndarray | pd.Series | float:
    arr, was_series, index = _to_numeric_array(prices, label="american prices")
    if not np.isfinite(arr).all():
        raise ValueError("American prices must be finite numeric values.")
    if bool((arr == 0.0).any()):
        raise ValueError("American prices cannot include zero.")
    positive = arr > 0.0
    implied = np.empty_like(arr, dtype=float)
    implied[positive] = 100.0 / (arr[positive] + 100.0)
    implied[~positive] = (-arr[~positive]) / ((-arr[~positive]) + 100.0)
    return _restore_shape(implied, was_series=was_series, index=index, as_scalar=np.isscalar(prices))


def _validate_probability(probabilities: np.ndarray, *, label: str) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{label} must be finite numeric values.")
    below = int((probabilities < 0.0).sum())
    above = int((probabilities > 1.0).sum())
    if below or above:
        raise ValueError(
            f"{label} must stay within [0, 1], but found {below} rows below 0 and {above} rows above 1."
        )


def vig_free_two_way_probabilities(
    home_values: Any,
    away_values: Any,
    *,
    input_scale: str = "implied_probability",
) -> tuple[np.ndarray | pd.Series | float, np.ndarray | pd.Series | float]:
    normalized_scale = _normalize_input_scale(input_scale)
    home_arr, home_series, home_index = _to_numeric_array(home_values, label="home inputs")
    away_arr, away_series, away_index = _to_numeric_array(away_values, label="away inputs")
    try:
        home_arr, away_arr = np.broadcast_arrays(home_arr, away_arr)
    except ValueError as exc:
        raise ValueError("Home and away market inputs must be broadcastable to the same shape.") from exc

    if normalized_scale == "implied_probability":
        home_implied = home_arr.astype(float)
        away_implied = away_arr.astype(float)
        _validate_probability(home_implied, label="home implied probabilities")
        _validate_probability(away_implied, label="away implied probabilities")
    else:
        home_implied = np.asarray(american_price_to_implied_probability(home_arr), dtype=float)
        away_implied = np.asarray(american_price_to_implied_probability(away_arr), dtype=float)

    total = home_implied + away_implied
    if not np.isfinite(total).all():
        raise ValueError("Market implied-probability sums must be finite.")
    if bool((total <= 0.0).any()):
        raise ValueError("Market implied-probability sums must be positive for vig-free normalization.")

    home_vig_free = home_implied / total
    away_vig_free = away_implied / total
    home_out = _restore_shape(
        home_vig_free,
        was_series=home_series,
        index=home_index if home_series else away_index if away_series else None,
        as_scalar=np.isscalar(home_values) and np.isscalar(away_values),
    )
    away_out = _restore_shape(
        away_vig_free,
        was_series=away_series,
        index=away_index if away_series else home_index if home_series else None,
        as_scalar=np.isscalar(home_values) and np.isscalar(away_values),
    )
    return home_out, away_out


def probability_to_logit(probabilities: Any, *, eps: float = PROBABILITY_EPS) -> np.ndarray | pd.Series | float:
    if not np.isfinite(float(eps)) or float(eps) <= 0.0 or float(eps) >= 0.5:
        raise ValueError(f"eps must be a finite value in (0, 0.5), got {eps}.")
    arr, was_series, index = _to_numeric_array(probabilities, label="probabilities")
    _validate_probability(arr, label="probabilities")
    clipped = np.clip(arr, float(eps), 1.0 - float(eps))
    logits = np.log(clipped / (1.0 - clipped))
    return _restore_shape(logits, was_series=was_series, index=index, as_scalar=np.isscalar(probabilities))


def vig_free_two_way_logit_offsets(
    home_values: Any,
    away_values: Any,
    *,
    input_scale: str = "implied_probability",
    eps: float = PROBABILITY_EPS,
) -> tuple[np.ndarray | pd.Series | float, np.ndarray | pd.Series | float]:
    home_prob, away_prob = vig_free_two_way_probabilities(home_values, away_values, input_scale=input_scale)
    home_logit = probability_to_logit(home_prob, eps=eps)
    away_logit = probability_to_logit(away_prob, eps=eps)
    return home_logit, away_logit


__all__ = [
    "PROBABILITY_EPS",
    "american_price_to_implied_probability",
    "probability_to_logit",
    "vig_free_two_way_logit_offsets",
    "vig_free_two_way_probabilities",
]
