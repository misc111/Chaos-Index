from __future__ import annotations

from math import isfinite


_LAMBDA_GRIDS: dict[str, list[float]] = {
    "glm_ridge": [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625],
    "glm_lasso": [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125],
    "glm_elastic_net": [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125],
}
_ELASTIC_NET_L1_RATIOS = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]


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


def default_lambda_grid(model_name: str) -> list[float]:
    token = str(model_name or "").strip()
    if token not in _LAMBDA_GRIDS:
        raise ValueError(f"Unsupported penalized GLM '{model_name}'")
    return list(_LAMBDA_GRIDS[token])


def default_l1_ratio_grid(model_name: str) -> list[float]:
    token = str(model_name or "").strip()
    if token != "glm_elastic_net":
        return []
    return list(_ELASTIC_NET_L1_RATIOS)


def penalized_glm_search_grid(model_name: str) -> list[dict[str, float]]:
    token = str(model_name or "").strip()
    lambdas = default_lambda_grid(token)
    if token == "glm_elastic_net":
        return [
            {
                "lambda": lam,
                "c": lambda_to_c(lam),
                "l1_ratio": ratio,
            }
            for lam in lambdas
            for ratio in default_l1_ratio_grid(token)
        ]
    return [{"lambda": lam, "c": lambda_to_c(lam)} for lam in lambdas]
