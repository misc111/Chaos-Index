from __future__ import annotations

from math import isclose, isfinite


_LAMBDA_GRIDS: dict[str, list[float]] = {
    "glm_ridge": [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625],
    "glm_lasso": [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125],
    "glm_elastic_net": [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125],
}
_ELASTIC_NET_L1_RATIOS = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
_PENALTY_FAMILIES = {
    "glm_ridge": "ridge",
    "glm_lasso": "lasso",
    "glm_elastic_net": "elastic_net",
}


def _normalize_model_name(model_name: str) -> str:
    token = str(model_name or "").strip()
    if token not in _LAMBDA_GRIDS:
        raise ValueError(f"Unsupported penalized GLM '{model_name}'")
    return token


def lambda_to_c(value: float) -> float:
    lam = float(value)
    if not isfinite(lam) or lam <= 0.0:
        raise ValueError("lambda must be positive and finite")
    return 1.0 / lam


def c_to_lambda(value: float) -> float:
    c_value = float(value)
    if not isfinite(c_value) or c_value <= 0.0:
        raise ValueError("C must be positive and finite")
    return 1.0 / c_value


def penalty_family(model_name: str) -> str:
    token = _normalize_model_name(model_name)
    return _PENALTY_FAMILIES[token]


def penalty_choice(
    model_name: str,
    *,
    lambda_value: float | None = None,
    c_value: float | None = None,
    l1_ratio: float | None = None,
) -> dict[str, float | str | None]:
    token = _normalize_model_name(model_name)
    if lambda_value is None and c_value is None:
        raise ValueError("Either lambda_value or c_value must be provided")

    resolved_lambda = None if lambda_value is None else float(lambda_value)
    resolved_c = None if c_value is None else float(c_value)
    if resolved_lambda is not None:
        if not isfinite(resolved_lambda) or resolved_lambda <= 0.0:
            raise ValueError("lambda must be positive and finite")
        if resolved_c is None:
            resolved_c = lambda_to_c(resolved_lambda)
    if resolved_c is not None:
        if not isfinite(resolved_c) or resolved_c <= 0.0:
            raise ValueError("C must be positive and finite")
        if resolved_lambda is None:
            resolved_lambda = c_to_lambda(resolved_c)
    implied_lambda = c_to_lambda(float(resolved_c))
    if not isclose(float(resolved_lambda), implied_lambda, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("lambda_value and c_value must be reciprocal")

    return {
        "model_name": token,
        "penalty_family": penalty_family(token),
        "lambda": float(resolved_lambda),
        "c": float(resolved_c),
        "l1_ratio": None if token != "glm_elastic_net" or l1_ratio is None else float(l1_ratio),
    }


def default_lambda_grid(model_name: str) -> list[float]:
    token = _normalize_model_name(model_name)
    return list(_LAMBDA_GRIDS[token])


def default_l1_ratio_grid(model_name: str) -> list[float]:
    token = _normalize_model_name(model_name)
    if token != "glm_elastic_net":
        return []
    return list(_ELASTIC_NET_L1_RATIOS)


def penalized_glm_search_grid(model_name: str) -> list[dict[str, float | str | None]]:
    token = _normalize_model_name(model_name)
    lambdas = default_lambda_grid(token)
    if token == "glm_elastic_net":
        return [
            penalty_choice(token, lambda_value=lam, l1_ratio=ratio)
            for lam in lambdas
            for ratio in default_l1_ratio_grid(token)
        ]
    return [penalty_choice(token, lambda_value=lam) for lam in lambdas]
