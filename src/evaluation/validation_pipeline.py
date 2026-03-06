from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.common.config import AppConfig
from src.common.utils import ensure_dir
from src.evaluation.brier_decomposition import brier_decompose
from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.diagnostics_glm import save_glm_diagnostics
from src.evaluation.diagnostics_ml import permutation_importance_report
from src.evaluation.validation_fragility import missingness_stress_test, perturbation_sensitivity
from src.evaluation.validation_influence import influence_diagnostics
from src.evaluation.validation_significance import blockwise_nested_lrt
from src.evaluation.validation_stability import assess_multicollinearity, break_test_trade_deadline, coefficient_paths

ValidationTaskRunner = Callable[["ValidationContext"], "ValidationOutputs"]
ValidationTaskPredicate = Callable[["ValidationContext"], bool]


def _canonical_league(league: str | None) -> str:
    token = str(league or "").strip().upper()
    if token in {"NHL", "NBA"}:
        return token
    raise ValueError(f"Unsupported league '{league}'. Expected one of: NHL, NBA.")


@dataclass(frozen=True, slots=True)
class ValidationSectionSpec:
    section: str
    file_name: str
    kind: str
    tail_rows: int | None = None


@dataclass(slots=True)
class ValidationOutputs:
    sections: list[ValidationSectionSpec] = field(default_factory=list)
    csv_payloads: dict[str, pd.DataFrame] = field(default_factory=dict)
    json_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_csv(
        self,
        *,
        section: str,
        file_name: str,
        rows: pd.DataFrame,
        tail_rows: int | None = None,
    ) -> None:
        self._register(ValidationSectionSpec(section=section, file_name=file_name, kind="csv", tail_rows=tail_rows))
        self.csv_payloads[section] = rows.copy()

    def add_json(self, *, section: str, file_name: str, payload: dict[str, Any]) -> None:
        self._register(ValidationSectionSpec(section=section, file_name=file_name, kind="json"))
        self.json_payloads[section] = dict(payload)

    def merge(self, other: "ValidationOutputs") -> None:
        for spec in other.sections:
            self._register(spec)
        self.csv_payloads.update({key: value.copy() for key, value in other.csv_payloads.items()})
        self.json_payloads.update({key: dict(value) for key, value in other.json_payloads.items()})

    def write(self, out_dir: Path, *, league: str) -> None:
        root = ensure_dir(out_dir)
        for spec in self.sections:
            path = root / spec.file_name
            if spec.kind == "csv":
                self.csv_payloads[spec.section].to_csv(path, index=False)
            elif spec.kind == "json":
                path.write_text(json.dumps(self.json_payloads[spec.section], indent=2, sort_keys=True))
            else:
                raise ValueError(f"Unsupported validation artifact kind '{spec.kind}' for section '{spec.section}'")

        manifest = {
            "league": league,
            "sections": [asdict(spec) for spec in self.sections],
        }
        (root / "validation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _register(self, spec: ValidationSectionSpec) -> None:
        if spec.section in self.csv_payloads or spec.section in self.json_payloads:
            raise ValueError(f"Duplicate validation section '{spec.section}'")
        if any(existing.file_name == spec.file_name for existing in self.sections):
            raise ValueError(f"Duplicate validation artifact path '{spec.file_name}'")
        self.sections.append(spec)


@dataclass(slots=True)
class ValidationContext:
    cfg: AppConfig
    models: dict[str, object]
    train_df: pd.DataFrame
    feature_cols: list[str]
    out_dir: Path
    plots_dir: Path
    league: str
    tr: pd.DataFrame
    va: pd.DataFrame
    glm: object | None
    diagnostic_feature_cols: list[str]

    @classmethod
    def from_result(cls, result: dict[str, Any], cfg: AppConfig) -> "ValidationContext":
        train_df = result["train_df"].copy()
        feature_cols = list(result["feature_columns"])
        models = result["models"]
        out_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "validation")
        plots_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "plots")
        split = int(len(train_df) * 0.8)
        tr = train_df.iloc[:split].copy()
        va = train_df.iloc[split:].copy()
        glm = models.get("glm_logit")
        diagnostic_feature_cols = list(getattr(glm, "feature_columns", []) or feature_cols[: min(40, len(feature_cols))])

        return cls(
            cfg=cfg,
            models=models,
            train_df=train_df,
            feature_cols=feature_cols,
            out_dir=out_dir,
            plots_dir=plots_dir,
            league=_canonical_league(cfg.data.league),
            tr=tr,
            va=va,
            glm=glm,
            diagnostic_feature_cols=diagnostic_feature_cols,
        )


@dataclass(frozen=True, slots=True)
class ValidationTask:
    name: str
    runner: ValidationTaskRunner
    enabled: ValidationTaskPredicate | None = None

    def should_run(self, ctx: ValidationContext) -> bool:
        return True if self.enabled is None else bool(self.enabled(ctx))


def _is_full_suite_league(ctx: ValidationContext) -> bool:
    return ctx.league != "NBA"


def _has_glm(ctx: ValidationContext) -> bool:
    return ctx.glm is not None


def _has_holdout(ctx: ValidationContext) -> bool:
    return not ctx.va.empty


def _feature_blocks_for(ctx: ValidationContext) -> dict[str, list[str]]:
    if ctx.league == "NBA":
        return {
            "availability_block": [
                c for c in ctx.diagnostic_feature_cols if "availability" in c or "absence" in c or "roster_depth" in c
            ],
            "shot_profile_block": [c for c in ctx.diagnostic_feature_cols if "shot" in c or "scoring_efficiency" in c],
            "discipline_block": [
                c for c in ctx.diagnostic_feature_cols if "discipline" in c or "foul" in c or "free_throw" in c
            ],
            "travel_block": [c for c in ctx.diagnostic_feature_cols if "travel" in c or "rest" in c or "tz_" in c],
            "arena_block": [c for c in ctx.diagnostic_feature_cols if "arena" in c],
        }
    return {
        "goalie_block": [c for c in ctx.diagnostic_feature_cols if "goalie" in c],
        "xg_block": [c for c in ctx.diagnostic_feature_cols if "xg" in c],
        "special_teams_block": [
            c for c in ctx.diagnostic_feature_cols if "special" in c or "penalty" in c or "pp_" in c
        ],
        "travel_block": [c for c in ctx.diagnostic_feature_cols if "travel" in c or "rest" in c or "tz_" in c],
        "lineup_block": [c for c in ctx.diagnostic_feature_cols if "lineup" in c or "roster" in c or "man_games" in c],
        "rink_block": [c for c in ctx.diagnostic_feature_cols if "rink" in c],
    }


def _task_glm_diagnostics(ctx: ValidationContext) -> ValidationOutputs:
    if ctx.glm is None or ctx.va.empty:
        return ValidationOutputs()

    va = ctx.va.copy()
    va["glm_prob"] = ctx.glm.predict_proba(va)
    save_glm_diagnostics(
        df=va,
        p_col="glm_prob",
        y_col="home_win",
        feature_cols=ctx.diagnostic_feature_cols,
        coefs=ctx.glm.model.coef_[0],
        out_dir=str(ctx.plots_dir),
        prefix="glm",
    )
    return ValidationOutputs()


def _task_permutation_importance(ctx: ValidationContext) -> ValidationOutputs:
    if ctx.va.empty:
        return ValidationOutputs()

    for model_name in ["gbdt", "rf"]:
        model = ctx.models.get(model_name)
        if model is None or len(ctx.va) <= 25:
            continue
        model_feature_cols = getattr(model, "feature_columns", ctx.feature_cols) or ctx.feature_cols
        permutation_importance_report(
            model.model,
            ctx.va[model_feature_cols],
            ctx.va["home_win"].astype(int).to_numpy(),
            out_dir=str(ctx.out_dir),
            model_name=model_name,
        )
    return ValidationOutputs()


def _task_collinearity(ctx: ValidationContext) -> ValidationOutputs:
    report = assess_multicollinearity(ctx.train_df, features=ctx.diagnostic_feature_cols)
    out = ValidationOutputs()
    out.add_json(
        section="collinearity_summary",
        file_name="validation_collinearity_summary.json",
        payload=report["summary"],
    )
    out.add_csv(section="vif", file_name="validation_vif.csv", rows=report["vif"])
    out.add_csv(
        section="collinearity_structural",
        file_name="validation_collinearity_structural.csv",
        rows=report["structural"],
    )
    out.add_csv(
        section="collinearity_pairs",
        file_name="validation_collinearity_pairs.csv",
        rows=report["pairwise"],
    )
    out.add_csv(
        section="collinearity_condition",
        file_name="validation_collinearity_condition.csv",
        rows=report["condition"],
    )
    out.add_csv(
        section="collinearity_variance_decomposition",
        file_name="validation_collinearity_variance_decomposition.csv",
        rows=report["variance_decomposition"],
    )
    return out


def _task_significance(ctx: ValidationContext) -> ValidationOutputs:
    sig = blockwise_nested_lrt(
        ctx.tr,
        ctx.va,
        feature_blocks=_feature_blocks_for(ctx),
        all_features=ctx.diagnostic_feature_cols,
    )
    out = ValidationOutputs()
    out.add_csv(section="significance", file_name="validation_significance.csv", rows=sig)
    return out


def _task_stability(ctx: ValidationContext) -> ValidationOutputs:
    out = ValidationOutputs()
    out.add_csv(
        section="coef_paths",
        file_name="validation_coef_paths.csv",
        rows=coefficient_paths(ctx.train_df, features=ctx.diagnostic_feature_cols),
        tail_rows=200,
    )
    out.add_json(
        section="break_test",
        file_name="validation_break_test.json",
        payload=break_test_trade_deadline(ctx.train_df, features=ctx.diagnostic_feature_cols),
    )
    return out


def _task_influence(ctx: ValidationContext) -> ValidationOutputs:
    infl_df, infl_summary = influence_diagnostics(ctx.train_df, features=ctx.diagnostic_feature_cols, top_k=10)
    out = ValidationOutputs()
    out.add_csv(section="influence_top", file_name="validation_influence_top.csv", rows=infl_df)
    out.add_json(section="influence_summary", file_name="validation_influence_summary.json", payload=infl_summary)
    return out


def _task_fragility(ctx: ValidationContext) -> ValidationOutputs:
    base_df = ctx.va if not ctx.va.empty else ctx.train_df
    out = ValidationOutputs()
    out.add_csv(
        section="fragility_missingness",
        file_name="validation_fragility_missingness.csv",
        rows=missingness_stress_test(ctx.glm, base_df, feature_cols=ctx.diagnostic_feature_cols),
    )
    out.add_json(
        section="fragility_perturbation",
        file_name="validation_fragility_perturbation.json",
        payload=perturbation_sensitivity(ctx.glm, base_df, feature_cols=ctx.diagnostic_feature_cols),
    )
    return out


def _task_calibration(ctx: ValidationContext) -> ValidationOutputs:
    p = ctx.glm.predict_proba(ctx.va)
    y = ctx.va["home_win"].astype(int).to_numpy()
    payload = calibration_alpha_beta(y, p) | ece_mce(y, p)
    payload |= brier_decompose(y, p)

    out = ValidationOutputs()
    out.add_json(
        section="calibration_robustness",
        file_name="validation_calibration_robustness.json",
        payload=payload,
    )
    return out


def build_validation_tasks(
    *,
    extra_tasks: Sequence[ValidationTask] | None = None,
) -> list[ValidationTask]:
    tasks = [
        ValidationTask(name="glm_diagnostics", runner=_task_glm_diagnostics, enabled=lambda ctx: _has_glm(ctx) and _has_holdout(ctx)),
        ValidationTask(name="permutation_importance", runner=_task_permutation_importance, enabled=_has_holdout),
        ValidationTask(name="collinearity", runner=_task_collinearity),
        ValidationTask(
            name="significance",
            runner=_task_significance,
            enabled=lambda ctx: _is_full_suite_league(ctx) and _has_holdout(ctx) and not ctx.tr.empty,
        ),
        ValidationTask(name="stability", runner=_task_stability, enabled=_is_full_suite_league),
        ValidationTask(name="influence", runner=_task_influence, enabled=_is_full_suite_league),
        ValidationTask(name="fragility", runner=_task_fragility, enabled=_has_glm),
        ValidationTask(name="calibration", runner=_task_calibration, enabled=lambda ctx: _has_glm(ctx) and _has_holdout(ctx)),
    ]
    if extra_tasks:
        tasks.extend(extra_tasks)
    return tasks


def run_validation_pipeline(
    result: dict[str, Any],
    cfg: AppConfig,
    *,
    tasks: Sequence[ValidationTask] | None = None,
    extra_tasks: Sequence[ValidationTask] | None = None,
) -> ValidationOutputs:
    ctx = ValidationContext.from_result(result, cfg)
    outputs = ValidationOutputs()
    selected_tasks = list(tasks) if tasks is not None else build_validation_tasks(extra_tasks=extra_tasks)

    for task in selected_tasks:
        if not task.should_run(ctx):
            continue
        outputs.merge(task.runner(ctx))

    outputs.write(ctx.out_dir, league=ctx.league)
    return outputs
