"""Candidate-spec construction for nested MLB tournaments.

This module translates screened feature variants into concrete candidate model
specs and parameter grids.  It is intentionally about model-family construction
only; fitting, diagnostics, and champion selection live in separate modules.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.research.candidate_models import (
    BaseCandidateModel,
    CalibratedProbabilityCandidate,
    DGLMMarginCandidate,
    GAMSplineCandidate,
    GLMMLogitCandidate,
    MarketOffsetGLMCandidate,
    MarketProbabilityCandidate,
    PenalizedLogitCandidate,
    VanillaGLMBinomialCandidate,
)
from src.research.mlb_targets import TargetDefinition
from src.research.nested_tournament_contracts import FeatureVariant, NestedCandidateSpec
from src.research.nested_tournament_features import (
    _market_baseline_variants,
    _market_offset_feature,
    _market_probability_feature,
    _structured_variants,
)
from src.training.lambda_search import penalized_glm_search_grid

def _calibrated_variant(variant: FeatureVariant, method: str) -> FeatureVariant:
    return FeatureVariant(
        variant_key=f"{variant.variant_key}_{method}_calibrated",
        display_name=f"{variant.display_name} {method.title()} Calibrated",
        source=f"{variant.source};calibration:{method}",
        features=variant.features,
    )


def _wrap_calibrated(
    builder: Callable[[dict[str, Any]], BaseCandidateModel],
    *,
    method: str,
) -> Callable[[dict[str, Any]], BaseCandidateModel]:
    return lambda params: CalibratedProbabilityCandidate(base_model=builder(params), calibration_method=method)


def _model_specs_for_variant(
    *,
    target: TargetDefinition,
    feature_sets: Any,
    variant: FeatureVariant,
    candidate_models: set[str] | None,
) -> list[NestedCandidateSpec]:
    specs: list[NestedCandidateSpec] = []

    def selected(name: str) -> bool:
        return candidate_models is None or name in candidate_models

    def optional_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return None
        return float(value)

    if selected("glm_vanilla"):
        def base_builder(
            params: dict[str, Any],
            features: tuple[str, ...] = variant.features,
        ) -> BaseCandidateModel:
            return VanillaGLMBinomialCandidate(features=list(features))

        specs.append(
            NestedCandidateSpec(
                target=target,
                model_name="glm_vanilla",
                display_name="Vanilla GLM",
                variant_key=variant.variant_key,
                variant_display_name=variant.display_name,
                feature_source=variant.source,
                features=variant.features,
                param_grid=({},),
                builder=base_builder,
            )
        )
        for method in ("platt", "isotonic"):
            calibrated = _calibrated_variant(variant, method)
            specs.append(
                NestedCandidateSpec(
                    target=target,
                    model_name="glm_vanilla",
                    display_name="Vanilla GLM",
                    variant_key=calibrated.variant_key,
                    variant_display_name=calibrated.display_name,
                    feature_source=calibrated.source,
                    features=calibrated.features,
                    param_grid=({},),
                    builder=_wrap_calibrated(base_builder, method=method),
                )
            )
    for model_name, display_name, penalty, solver in (
        ("glm_ridge", "GLM Ridge", "l2", "lbfgs"),
        ("glm_lasso", "GLM Lasso", "l1", "saga"),
        ("glm_elastic_net", "Elastic Net GLM", "elasticnet", "saga"),
    ):
        if not selected(model_name):
            continue
        grid = tuple(penalized_glm_search_grid(model_name))
        def base_builder(
            params: dict[str, Any],
            model_name: str = model_name,
            display_name: str = display_name,
            penalty: str = penalty,
            solver: str = solver,
            features: tuple[str, ...] = variant.features,
        ) -> BaseCandidateModel:
            return PenalizedLogitCandidate(
                model_name=model_name,
                display_name=display_name,
                features=list(features),
                penalty=penalty,
                c=float(params["c"]),
                l1_ratio=optional_float(params.get("l1_ratio")),
                solver=solver,
            )

        specs.append(
            NestedCandidateSpec(
                target=target,
                model_name=model_name,
                display_name=display_name,
                variant_key=variant.variant_key,
                variant_display_name=variant.display_name,
                feature_source=variant.source,
                features=variant.features,
                param_grid=grid,
                builder=base_builder,
            )
        )
        for method in ("platt", "isotonic"):
            calibrated = _calibrated_variant(variant, method)
            specs.append(
                NestedCandidateSpec(
                    target=target,
                    model_name=model_name,
                    display_name=display_name,
                    variant_key=calibrated.variant_key,
                    variant_display_name=calibrated.display_name,
                    feature_source=calibrated.source,
                    features=calibrated.features,
                    param_grid=grid,
                    builder=_wrap_calibrated(base_builder, method=method),
                )
            )

    if selected("glmm_logit"):
        caps = sorted({cap for cap in (6, 10, 14) if cap <= len(feature_sets.glmm_features)}) or [len(feature_sets.glmm_features)]
        for cap in caps:
            features = tuple(feature_sets.glmm_features[:cap])
            specs.append(
                NestedCandidateSpec(
                    target=target,
                    model_name="glmm_logit",
                    display_name="GLMM Logit",
                    variant_key=f"glmm_cap_{cap}",
                    variant_display_name=f"GLMM Cap {cap}",
                    feature_source="grouped_effects",
                    features=features,
                    param_grid=({"feature_cap": cap},),
                    builder=lambda params, features=features: GLMMLogitCandidate(fixed_features=list(features)),
                )
            )
    if selected("dglm_margin") and target.target_name == "moneyline_home_win":
        caps = sorted({cap for cap in (6, 10, 14) if cap <= len(feature_sets.dglm_features)}) or [len(feature_sets.dglm_features)]
        for cap in caps:
            features = tuple(feature_sets.dglm_features[:cap])
            specs.append(
                NestedCandidateSpec(
                    target=target,
                    model_name="dglm_margin",
                    display_name="DGLM Margin",
                    variant_key=f"dglm_cap_{cap}",
                    variant_display_name=f"DGLM Cap {cap}",
                    feature_source="dispersion",
                    features=features,
                    param_grid=tuple({"feature_cap": cap, "iterations": iterations} for iterations in (1, 2)),
                    builder=lambda params, features=features: DGLMMarginCandidate(
                        features=list(features),
                        iterations=int(params["iterations"]),
                    ),
                )
            )
    if selected("gam_spline"):
        caps = sorted({cap for cap in (3, 5, 6) if cap <= len(feature_sets.gam_features)}) or [len(feature_sets.gam_features)]
        for cap in caps:
            spline_features = tuple(feature_sets.gam_features[:cap])
            linear_features = tuple(
                feature for feature in feature_sets.core_features if feature not in spline_features
            )[: max(4, cap + 2)]
            specs.append(
                NestedCandidateSpec(
                    target=target,
                    model_name="gam_spline",
                    display_name="GAM Spline",
                    variant_key=f"gam_cap_{cap}",
                    variant_display_name=f"GAM Cap {cap}",
                    feature_source="nonlinearity",
                    features=tuple(dict.fromkeys(linear_features + spline_features)),
                    param_grid=tuple(
                        {"feature_cap": cap, "n_knots": n_knots, "c": c}
                        for n_knots in (4, 5)
                        for c in (0.25, 0.5, 1.0)
                    ),
                    builder=lambda params, linear_features=linear_features, spline_features=spline_features: GAMSplineCandidate(
                        linear_features=list(linear_features),
                        spline_features=list(spline_features),
                        n_knots=int(params["n_knots"]),
                        c=float(params["c"]),
                    ),
                )
            )
    return specs


def _candidate_specs(
    *,
    target: TargetDefinition,
    feature_sets: Any,
    candidate_models: set[str] | None,
    structured_glm_spec_path: str | Path | None,
) -> list[NestedCandidateSpec]:
    variants = _structured_variants(feature_sets, spec_path=structured_glm_spec_path)
    specs: list[NestedCandidateSpec] = []
    selected_models = candidate_models

    def selected(name: str) -> bool:
        return selected_models is None or name in selected_models

    if selected("market_baseline"):
        market_prob = _market_probability_feature(target)
        market_offset = _market_offset_feature(target)
        for variant in _market_baseline_variants(target, feature_sets):
            if variant.variant_key == "market_implied_only":
                specs.append(
                    NestedCandidateSpec(
                        target=target,
                        model_name="market_baseline",
                        display_name="Market Baseline",
                        variant_key=variant.variant_key,
                        variant_display_name=variant.display_name,
                        feature_source=variant.source,
                        features=variant.features,
                        param_grid=({},),
                        builder=lambda params, probability_feature=market_prob: MarketProbabilityCandidate(
                            probability_feature=probability_feature,
                        ),
                    )
                )
            elif variant.variant_key == "market_offset_glm":
                process_features = tuple(feature for feature in variant.features if feature != market_offset)
                specs.append(
                    NestedCandidateSpec(
                        target=target,
                        model_name="market_baseline",
                        display_name="Market Baseline",
                        variant_key=variant.variant_key,
                        variant_display_name=variant.display_name,
                        feature_source=variant.source,
                        features=variant.features,
                        param_grid=({},),
                        builder=lambda params, features=process_features, offset_feature=market_offset: MarketOffsetGLMCandidate(
                            features=list(features),
                            offset_feature=offset_feature,
                            display_name="Market Offset GLM",
                        ),
                    )
                )
            elif variant.variant_key == "market_plus_features_glm":
                specs.append(
                    NestedCandidateSpec(
                        target=target,
                        model_name="market_baseline",
                        display_name="Market Baseline",
                        variant_key=variant.variant_key,
                        variant_display_name=variant.display_name,
                        feature_source=variant.source,
                        features=variant.features,
                        param_grid=({},),
                        builder=lambda params, features=variant.features: MarketOffsetGLMCandidate(
                            features=list(features),
                            offset_feature=None,
                            display_name="Market Plus Features GLM",
                        ),
                    )
                )
    for variant in variants:
        specs.extend(
            _model_specs_for_variant(
                target=target,
                feature_sets=feature_sets,
                variant=variant,
                candidate_models=candidate_models,
            )
        )
    return specs


def _params_text(params: dict[str, Any]) -> str:
    if not params:
        return "{}"
    return "; ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _parse_params_text(params_text: Any) -> dict[str, Any]:
    if params_text is None:
        return {}
    if isinstance(params_text, float) and np.isnan(params_text):
        return {}
    text = str(params_text).strip()
    if not text or text == "{}" or text.lower() in {"nan", "none", "null"}:
        return {}
    parsed: dict[str, Any] = {}
    for token in text.split(";"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value_text = value.strip()
        if value_text.lower() in {"none", "nan", "null"}:
            parsed[key] = None
            continue
        try:
            parsed[key] = float(value_text)
        except ValueError:
            parsed[key] = value_text
    return parsed
