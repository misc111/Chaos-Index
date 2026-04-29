"""Model-family validation tasks for significance, stability, influence, and fragility."""

from __future__ import annotations

import pandas as pd

from src.evaluation.validation_artifacts import ValidationOutputs, validation_path as _validation_path
from src.evaluation.validation_context import ValidationContext, holdout_df as _holdout_df
from src.evaluation.validation_fragility import missingness_stress_test, perturbation_sensitivity
from src.evaluation.validation_influence import influence_diagnostics
from src.evaluation.validation_significance import (
    blockwise_nested_deviance_f_test,
    information_criteria_report,
    penalized_block_ablation_report,
    penalized_information_criteria_report,
)
from src.evaluation.validation_stability import (
    bootstrap_glm_coefficients,
    break_test_trade_deadline,
    coefficient_paths,
    cv_glm_stability_report,
)
from src.evaluation.validation_task_support import (
    ValidationTaskResult,
    _feature_blocks_for,
    _is_credibility_primary_model,
    _is_extension_primary_model,
    _is_penalized_primary_model,
    _is_vanilla_primary_model,
    _probability_model_family_applicability,
    _task_result,
    _task_summary_with_model_metadata,
    _task_support_payload,
    _unsupported_task_result,
    _validation_surrogate_penalized_spec,
)

def _task_significance(ctx: ValidationContext) -> ValidationTaskResult:
    holdout = _holdout_df(ctx)
    fit_df = ctx.fit_df if not ctx.fit_df.empty else ctx.tr
    feature_blocks = _feature_blocks_for(ctx)
    if _is_penalized_primary_model(ctx) or _is_credibility_primary_model(ctx):
        glm_model_name, glm_c, glm_l1_ratio, support_payload = _validation_surrogate_penalized_spec(ctx)
        if not glm_model_name or glm_c is None:
            return _unsupported_task_result(
                ctx,
                task_name="significance",
                note="Penalized/credibility significance diagnostics require a valid penalized surrogate model.",
            )
        sig = penalized_block_ablation_report(
            fit_df,
            holdout,
            feature_blocks=feature_blocks,
            all_features=ctx.diagnostic_feature_cols,
            model_name=glm_model_name,
            c=glm_c,
            l1_ratio=glm_l1_ratio,
            credibility=ctx.glm_validation_metadata.credibility,
        )
        ic = penalized_information_criteria_report(
            fit_df,
            holdout,
            feature_blocks=feature_blocks,
            all_features=ctx.diagnostic_feature_cols,
            model_name=glm_model_name,
            c=glm_c,
            l1_ratio=glm_l1_ratio,
            credibility=ctx.glm_validation_metadata.credibility,
        )
        applicability = "partial"
        support_payload = {
            **support_payload,
            "task_name": "significance",
        }
    elif _is_vanilla_primary_model(ctx):
        sig = blockwise_nested_deviance_f_test(
            fit_df,
            holdout,
            feature_blocks=feature_blocks,
            all_features=ctx.diagnostic_feature_cols,
        )
        ic = information_criteria_report(
            fit_df,
            holdout,
            feature_blocks=feature_blocks,
            all_features=ctx.diagnostic_feature_cols,
        )
        applicability = "applicable"
        support_payload = _task_support_payload(
            ctx,
            task_name="significance",
            support_status="applicable",
            route="selected_model_direct",
            note="Using direct vanilla-GLM nested-deviance and information-criteria diagnostics.",
        )
    else:
        return _unsupported_task_result(
            ctx,
            task_name="significance",
            note=(
                "Significance diagnostics currently support vanilla GLM directly and penalized/credibility "
                "families through penalized ablation surrogates."
            ),
        )
    out = ValidationOutputs()
    out.add_csv(
        section="significance",
        file_name=_validation_path("diagnostics", "significance", "validation_significance.csv"),
        rows=sig,
    )
    ic_summary = dict(ic["summary"])
    ic_summary["feature_block_map"] = {block_name: list(cols) for block_name, cols in feature_blocks.items()}
    ic_summary = _task_summary_with_model_metadata(ctx, ic_summary)
    out.add_json(
        section="information_criteria_summary",
        file_name=_validation_path("diagnostics", "significance", "validation_information_criteria_summary.json"),
        payload=ic_summary,
    )
    out.add_csv(
        section="information_criteria_candidates",
        file_name=_validation_path("diagnostics", "significance", "validation_information_criteria_candidates.csv"),
        rows=ic["candidates"],
    )
    summary_payload = dict(ic_summary)
    summary_payload.update(support_payload)
    return _task_result(ctx, out, summary=summary_payload, applicability=applicability)


def _task_stability(ctx: ValidationContext) -> ValidationTaskResult:
    if not (_is_penalized_primary_model(ctx) or _is_credibility_primary_model(ctx)):
        return _unsupported_task_result(
            ctx,
            task_name="stability",
            note="Stability diagnostics currently support penalized GLMs directly and lasso-credibility via the penalized driver surrogate.",
        )
    glm_model_name, glm_c, glm_l1_ratio, support_payload = _validation_surrogate_penalized_spec(ctx)
    if not glm_model_name or glm_c is None:
        return _unsupported_task_result(
            ctx,
            task_name="stability",
            note="No penalized surrogate was available for stability diagnostics.",
        )
    cv_report = cv_glm_stability_report(
        ctx.train_df,
        features=ctx.diagnostic_feature_cols,
        n_splits=max(2, int(getattr(ctx.cfg.modeling, "cv_splits", 5) or 5)),
        model_name=glm_model_name,
        c=glm_c,
        l1_ratio=glm_l1_ratio,
    )
    bootstrap = bootstrap_glm_coefficients(
        ctx.fit_df if not ctx.fit_df.empty else ctx.train_df,
        features=ctx.diagnostic_feature_cols,
        model_name=glm_model_name,
        c=glm_c,
        l1_ratio=glm_l1_ratio,
    )
    out = ValidationOutputs()
    out.add_csv(
        section="coef_paths",
        file_name=_validation_path("diagnostics", "stability", "validation_coef_paths.csv"),
        rows=coefficient_paths(
            ctx.train_df,
            features=ctx.diagnostic_feature_cols,
            model_name=glm_model_name,
            c=glm_c,
            l1_ratio=glm_l1_ratio,
        ),
        tail_rows=200,
    )
    break_test = break_test_trade_deadline(
        ctx.train_df,
        features=ctx.diagnostic_feature_cols,
        league=ctx.league,
        model_name=glm_model_name,
        c=glm_c,
        l1_ratio=glm_l1_ratio,
    )
    out.add_json(
        section="break_test",
        file_name=_validation_path("diagnostics", "stability", "validation_break_test.json"),
        payload=break_test,
    )
    out.add_json(
        section="cv_summary",
        file_name=_validation_path("diagnostics", "stability", "validation_cv_summary.json"),
        payload=cv_report["summary"],
    )
    out.add_csv(
        section="cv_fold_metrics",
        file_name=_validation_path("diagnostics", "stability", "validation_cv_fold_metrics.csv"),
        rows=cv_report["fold_metrics"],
    )
    out.add_csv(
        section="cv_feature_stability",
        file_name=_validation_path("diagnostics", "stability", "validation_cv_feature_stability.csv"),
        rows=cv_report["feature_summary"],
    )
    out.add_csv(
        section="cv_coefficients",
        file_name=_validation_path("diagnostics", "stability", "validation_cv_coefficients.csv"),
        rows=cv_report["coefficients"],
    )
    out.add_json(
        section="bootstrap_summary",
        file_name=_validation_path("diagnostics", "stability", "validation_bootstrap_summary.json"),
        payload=bootstrap["summary"],
    )
    out.add_csv(
        section="bootstrap_feature_summary",
        file_name=_validation_path("diagnostics", "stability", "validation_bootstrap_feature_summary.csv"),
        rows=bootstrap["feature_summary"],
    )
    out.add_csv(
        section="bootstrap_coefficients",
        file_name=_validation_path("diagnostics", "stability", "validation_bootstrap_coefficients.csv"),
        rows=bootstrap["coefficients"],
    )
    summary_payload = {
        "status": "ok",
        "break_test": break_test,
        "cv_summary": cv_report["summary"],
        "bootstrap_summary": bootstrap["summary"],
    }
    summary_payload.update({**support_payload, "task_name": "stability"})
    applicability = "applicable" if _is_penalized_primary_model(ctx) else "partial"
    return _task_result(ctx, out, summary=summary_payload, applicability=applicability)


def _task_influence(ctx: ValidationContext) -> ValidationTaskResult:
    if _is_extension_primary_model(ctx):
        return _unsupported_task_result(
            ctx,
            task_name="influence",
            note="Influence diagnostics are not yet implemented for theory-extension validation families.",
        )
    infl_df, infl_summary = influence_diagnostics(
        ctx.fit_df if not ctx.fit_df.empty else ctx.train_df,
        features=ctx.diagnostic_feature_cols,
        top_k=10,
    )
    out = ValidationOutputs()
    out.add_csv(
        section="influence_top",
        file_name=_validation_path("diagnostics", "influence", "validation_influence_top.csv"),
        rows=infl_df,
    )
    out.add_json(
        section="influence_summary",
        file_name=_validation_path("diagnostics", "influence", "validation_influence_summary.json"),
        payload=infl_summary,
    )
    if _is_vanilla_primary_model(ctx):
        support_payload = _task_support_payload(
            ctx,
            task_name="influence",
            support_status="applicable",
            route="selected_model_direct",
            note="Influence diagnostics are computed with an unpenalized GLM consistent with the vanilla GLM family.",
        )
        applicability = "applicable"
    else:
        support_payload = _task_support_payload(
            ctx,
            task_name="influence",
            support_status="partial",
            route="unpenalized_glm_surrogate",
            note="Influence diagnostics use an unpenalized GLM surrogate and do not capture shrinkage or credibility offsets directly.",
        )
        applicability = "partial"
    infl_summary = dict(infl_summary)
    infl_summary.update(support_payload)
    return _task_result(ctx, out, summary=infl_summary, applicability=applicability)


def _task_fragility(ctx: ValidationContext) -> ValidationTaskResult:
    base_df = _holdout_df(ctx) if not _holdout_df(ctx).empty else ctx.train_df
    missingness = missingness_stress_test(
        ctx.glm,
        base_df,
        feature_cols=ctx.diagnostic_feature_cols,
        league=ctx.league,
        credibility=ctx.glm_validation_metadata.credibility,
    )
    perturbation = perturbation_sensitivity(ctx.glm, base_df, feature_cols=ctx.diagnostic_feature_cols)
    out = ValidationOutputs()
    out.add_csv(
        section="fragility_missingness",
        file_name=_validation_path("diagnostics", "fragility", "validation_fragility_missingness.csv"),
        rows=missingness,
    )
    out.add_json(
        section="fragility_perturbation",
        file_name=_validation_path("diagnostics", "fragility", "validation_fragility_perturbation.json"),
        payload=perturbation,
    )
    summary_payload = dict(perturbation)
    summary_payload["scenario_count"] = int(len(missingness))
    summary_payload["applicable_scenarios"] = int((missingness.get("scenario_status", pd.Series(dtype=str)) == "ok").sum())
    summary_payload["model_family_applicability"] = _probability_model_family_applicability(
        ctx,
        task_name="fragility",
        requires_class_contrast=False,
        class_contrast_available=True,
        n_obs=int(len(base_df)),
    )
    return _task_result(ctx, out, summary=summary_payload)
