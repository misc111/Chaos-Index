from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.common.config import AppConfig
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir
from src.evaluation.brier_decomposition import brier_decompose
from src.evaluation.calibration import calibration_alpha_beta, ece_mce
from src.evaluation.validation_classification import validate_logistic_probability_model
from src.evaluation.diagnostics_glm import save_glm_diagnostics
from src.evaluation.diagnostics_ml import permutation_importance_report
from src.evaluation.validation_fragility import missingness_stress_test, perturbation_sensitivity
from src.evaluation.validation_influence import influence_diagnostics
from src.evaluation.validation_nonlinearity import assess_nonlinearity
from src.evaluation.validation_significance import blockwise_nested_deviance_f_test, information_criteria_report
from src.evaluation.validation_stability import (
    assess_multicollinearity,
    bootstrap_glm_coefficients,
    break_test_trade_deadline,
    coefficient_paths,
    cv_glm_stability_report,
)
from src.models.gbdt import GBDTModel
from src.models.glm_ridge import GLMRidgeModel
from src.models.rf import RFModel
from src.training.tune import quick_tune_glm

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
            ensure_dir(path.parent)
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
    run_payload: dict[str, Any]
    models: dict[str, object]
    train_df: pd.DataFrame
    feature_cols: list[str]
    out_dir: Path
    plots_dir: Path
    league: str
    tr: pd.DataFrame
    va: pd.DataFrame
    te: pd.DataFrame
    split_plan: "ValidationSplitPlan"
    fit_df: pd.DataFrame
    glm: object | None
    diagnostic_feature_cols: list[str]

    @classmethod
    def from_result(cls, result: dict[str, Any], cfg: AppConfig) -> "ValidationContext":
        train_df = result["train_df"].copy()
        feature_cols = list(result["feature_columns"])
        run_payload = result.get("run_payload", {})
        selected_models = _selected_validation_models(result, run_payload)
        split_plan, tr, va, te = _validation_split(train_df, cfg)
        fit_df = _concat_frames(tr, va)
        models = _fit_validation_models(
            fit_df,
            feature_cols=feature_cols,
            selected_models=selected_models,
            run_payload=run_payload,
        )
        league = _canonical_league(cfg.data.league)
        out_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "validation" / league.lower())
        plots_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "plots" / league.lower() / "glm" / "performance")
        glm = models.get("glm_ridge")
        diagnostic_feature_cols = list(
            getattr(glm, "feature_columns", [])
            or _validation_feature_columns(feature_cols, run_payload, "glm_ridge")
            or feature_cols[: min(40, len(feature_cols))]
        )

        return cls(
            cfg=cfg,
            run_payload=run_payload,
            models=models,
            train_df=train_df,
            feature_cols=feature_cols,
            out_dir=out_dir,
            plots_dir=plots_dir,
            league=league,
            tr=tr,
            va=va,
            te=te,
            split_plan=split_plan,
            fit_df=fit_df,
            glm=glm,
            diagnostic_feature_cols=diagnostic_feature_cols,
        )


def _safe_archive_token(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("_")
    return cleaned or "adhoc_validation"


def _artifacts_root(cfg: AppConfig) -> Path:
    return Path(cfg.paths.artifacts_dir)


def _relative_artifact_path(path: Path, *, cfg: AppConfig) -> str:
    try:
        return str(path.relative_to(_artifacts_root(cfg)))
    except ValueError:
        return str(path)


def _list_relative_files(root: Path, *, exclude: set[str] | None = None) -> list[str]:
    if not root.exists():
        return []
    excluded = exclude or set()
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if rel in excluded:
            continue
        files.append(rel)
    return files


def _validation_path(*parts: Any) -> str:
    cleaned: list[str] = []
    for part in parts:
        token = str(part).strip("/").replace("\\", "/")
        if token:
            cleaned.append(token)
    return "/".join(cleaned)


def _archive_root_for(ctx: ValidationContext, *, generated_at_utc: str) -> Path:
    date_bucket = generated_at_utc[:10]
    timestamp_slug = generated_at_utc.replace(":", "-").replace("+00:00", "Z")
    model_run_id = _safe_archive_token(ctx.run_payload.get("model_run_id"))
    base = ensure_dir(_artifacts_root(ctx.cfg) / "validation-runs" / ctx.league.lower() / date_bucket)
    candidate = base / f"{timestamp_slug}_{model_run_id}"
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        alt = base / f"{timestamp_slug}_{model_run_id}_{suffix:02d}"
        if not alt.exists():
            return alt
        suffix += 1


def _validation_run_metadata(
    ctx: ValidationContext,
    outputs: ValidationOutputs,
    *,
    generated_at_utc: str,
    archive_root: Path,
) -> dict[str, Any]:
    validation_files = _list_relative_files(ctx.out_dir, exclude={"validation_run_metadata.json"})
    performance_files = _list_relative_files(ctx.plots_dir)
    root_files = [rel for rel in validation_files if "/" not in rel]
    grouped_validation_files: dict[str, list[str]] = {}
    for rel in validation_files:
        if "/" not in rel:
            continue
        top_level = rel.split("/", 1)[0]
        grouped_validation_files.setdefault(top_level, []).append(rel)
    selected_models = ctx.run_payload.get("selected_models", [])
    if not isinstance(selected_models, list):
        selected_models = []

    artifact_groups = [{"name": "validation_root", "relative_dir": ".", "files": root_files}]
    for group_name in sorted(grouped_validation_files):
        artifact_groups.append(
            {
                "name": group_name,
                "relative_dir": group_name,
                "files": grouped_validation_files[group_name],
            }
        )
    artifact_groups.append({"name": "performance", "relative_dir": "performance", "files": performance_files})

    return {
        "league": ctx.league,
        "generated_at_utc": generated_at_utc,
        "archive_id": archive_root.name,
        "archive_date": generated_at_utc[:10],
        "model_run_id": ctx.run_payload.get("model_run_id"),
        "feature_set_version": ctx.run_payload.get("feature_set_version"),
        "selected_models": [str(model) for model in selected_models if str(model).strip()],
        "latest_validation_dir": _relative_artifact_path(ctx.out_dir, cfg=ctx.cfg),
        "latest_performance_dir": _relative_artifact_path(ctx.plots_dir, cfg=ctx.cfg),
        "archive_dir": _relative_artifact_path(archive_root, cfg=ctx.cfg),
        "registered_sections": [asdict(spec) for spec in outputs.sections],
        "artifact_counts": {
            "validation_root_files": len(root_files),
            "validation_subdir_groups": len(grouped_validation_files),
            "validation_subdir_files": sum(len(files) for files in grouped_validation_files.values()),
            "performance_files": len(performance_files),
        },
        "artifact_groups": artifact_groups,
    }


def _reset_validation_output_dirs(ctx: ValidationContext) -> None:
    if ctx.out_dir.exists():
        shutil.rmtree(ctx.out_dir)
    ensure_dir(ctx.out_dir)
    if ctx.plots_dir.exists():
        shutil.rmtree(ctx.plots_dir)
    ensure_dir(ctx.plots_dir)


def _archive_validation_outputs(ctx: ValidationContext, outputs: ValidationOutputs) -> None:
    generated_at_utc = utc_now_iso()
    archive_root = _archive_root_for(ctx, generated_at_utc=generated_at_utc)
    metadata = _validation_run_metadata(ctx, outputs, generated_at_utc=generated_at_utc, archive_root=archive_root)
    (ctx.out_dir / "validation_run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    ensure_dir(archive_root.parent)
    shutil.copytree(ctx.out_dir, archive_root)
    if archive_root.exists():
        performance_root = archive_root / "performance"
        if performance_root.exists():
            shutil.rmtree(performance_root)
        shutil.copytree(ctx.plots_dir, performance_root)


def _selected_validation_models(result: dict[str, Any], run_payload: dict[str, Any]) -> list[str]:
    def _canonical_model_name(model_name: Any) -> str:
        token = str(model_name).strip()
        if token == "glm_logit":
            return "glm_ridge"
        return token

    payload_models = run_payload.get("selected_models", [])
    if isinstance(payload_models, list):
        selected = [_canonical_model_name(model) for model in payload_models if str(model).strip()]
        if selected:
            return selected

    raw_models = result.get("models", {})
    if isinstance(raw_models, dict):
        return [_canonical_model_name(model) for model in raw_models.keys() if str(model).strip()]

    return []


def _sort_validation_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ("start_time_utc", "game_date_utc"):
        if col in work.columns:
            return work.sort_values(col)
    return work.reset_index(drop=True)


def _concat_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, axis=0)
    return _sort_validation_rows(out)


@dataclass(frozen=True, slots=True)
class ValidationSplitPlan:
    requested_mode: str
    requested_method: str
    resolved_mode: str
    resolved_method: str
    train_fraction: float
    validation_fraction: float
    holdout_fraction: float
    random_seed: int | None
    note: str | None = None


def _slice_train_test(work: pd.DataFrame, *, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_obs = len(work)
    split = int(round(train_fraction * n_obs))
    if split <= 0 or split >= n_obs:
        return work.copy(), work.iloc[0:0].copy(), work.iloc[0:0].copy()
    return work.iloc[:split].copy(), work.iloc[0:0].copy(), work.iloc[split:].copy()


def _slice_train_validation_test(
    work: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_obs = len(work)
    train_end = int(round(train_fraction * n_obs))
    valid_end = int(round((train_fraction + validation_fraction) * n_obs))
    if train_end <= 0 or valid_end <= train_end or valid_end >= n_obs:
        return work.copy(), work.iloc[0:0].copy(), work.iloc[0:0].copy()
    return (
        work.iloc[:train_end].copy(),
        work.iloc[train_end:valid_end].copy(),
        work.iloc[valid_end:].copy(),
    )


def _validation_split(train_df: pd.DataFrame, cfg: AppConfig) -> tuple[ValidationSplitPlan, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = train_df[train_df["home_win"].notna()].copy()
    split_cfg = cfg.validation_split
    train_fraction, validation_fraction, holdout_fraction = split_cfg.fractions()
    random_seed = None
    note: str | None = None
    if work.empty:
        plan = ValidationSplitPlan(
            requested_mode=split_cfg.mode,
            requested_method=split_cfg.method,
            resolved_mode=split_cfg.mode,
            resolved_method=split_cfg.method,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            holdout_fraction=holdout_fraction,
            random_seed=None,
            note="no_finalized_rows",
        )
        return plan, work, work.copy(), work.copy()

    if split_cfg.method == "random":
        random_seed = split_cfg.normalized_random_seed(fallback_seed=int(cfg.modeling.random_seed))
        work = work.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    else:
        work = _sort_validation_rows(work).reset_index(drop=True)

    resolved_mode = split_cfg.mode
    if split_cfg.mode == "train_validation_test":
        tr, va, te = _slice_train_validation_test(
            work,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        if min(len(tr), len(va), len(te)) < 20:
            resolved_mode = "train_test"
            train_fraction, validation_fraction, holdout_fraction = 0.7, 0.0, 0.3
            tr, va, te = _slice_train_test(work, train_fraction=train_fraction)
            note = "requested_train_validation_test_was_too_thin_so_train_test_was_used"
    else:
        tr, va, te = _slice_train_test(work, train_fraction=train_fraction)

    tr = _sort_validation_rows(tr)
    va = _sort_validation_rows(va)
    te = _sort_validation_rows(te)
    plan = ValidationSplitPlan(
        requested_mode=split_cfg.mode,
        requested_method=split_cfg.method,
        resolved_mode=resolved_mode,
        resolved_method=split_cfg.method,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        holdout_fraction=holdout_fraction,
        random_seed=random_seed,
        note=note,
    )
    return plan, tr, va, te


def _holdout_df(ctx: ValidationContext) -> pd.DataFrame:
    return ctx.te if not ctx.te.empty else ctx.va


def _date_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    for col in ("start_time_utc", "game_date_utc"):
        if col in df.columns:
            values = df[col].dropna().astype(str)
            if not values.empty:
                return str(values.min()), str(values.max())
    return None, None


def _validation_feature_columns(
    feature_cols: list[str],
    run_payload: dict[str, Any],
    model_name: str,
) -> list[str]:
    model_feature_map = run_payload.get("model_feature_columns", {})
    if isinstance(model_feature_map, dict):
        candidate_keys = [model_name]
        if model_name == "glm_ridge":
            candidate_keys.append("glm_logit")
        for key in candidate_keys:
            requested = model_feature_map.get(key, [])
            if isinstance(requested, list):
                resolved = [str(col) for col in requested if str(col) in feature_cols]
                if resolved:
                    return resolved

    if model_name == "glm_ridge":
        glm_feature_columns = run_payload.get("glm_feature_columns", [])
        if isinstance(glm_feature_columns, list):
            resolved = [str(col) for col in glm_feature_columns if str(col) in feature_cols]
            if resolved:
                return resolved

    return list(feature_cols)


def _fit_validation_models(
    tr: pd.DataFrame,
    *,
    feature_cols: list[str],
    selected_models: list[str],
    run_payload: dict[str, Any],
) -> dict[str, object]:
    if tr.empty:
        return {}

    selected = set(selected_models)
    models: dict[str, object] = {}

    if "glm_ridge" in selected:
        glm_cols = _validation_feature_columns(feature_cols, run_payload, "glm_ridge")
        glm_tune = {"best_c": 1.0}
        if "start_time_utc" in tr.columns:
            glm_tune = quick_tune_glm(
                tr,
                glm_cols,
                n_splits=3,
                min_train_size=min(140, max(70, len(tr) // 2)),
            )
        glm = GLMRidgeModel(c=float(glm_tune.get("best_c", 1.0)))
        glm.fit(tr, glm_cols)
        models[glm.model_name] = glm

    for model_name, model_cls in (("gbdt", GBDTModel), ("rf", RFModel)):
        if model_name not in selected:
            continue
        model_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        model = model_cls()
        model.fit(tr, model_cols)
        models[model_name] = model

    return models


@dataclass(frozen=True, slots=True)
class ValidationTask:
    name: str
    runner: ValidationTaskRunner
    enabled: ValidationTaskPredicate | None = None

    def should_run(self, ctx: ValidationContext) -> bool:
        return True if self.enabled is None else bool(self.enabled(ctx))


def _has_glm(ctx: ValidationContext) -> bool:
    return ctx.glm is not None


def _has_holdout(ctx: ValidationContext) -> bool:
    return not _holdout_df(ctx).empty


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


def _task_split_summary(ctx: ValidationContext) -> ValidationOutputs:
    train_start, train_end = _date_bounds(ctx.tr)
    valid_start, valid_end = _date_bounds(ctx.va)
    holdout = _holdout_df(ctx)
    holdout_start, holdout_end = _date_bounds(holdout)
    payload = {
        "status": "ok",
        "split_mode": ctx.split_plan.resolved_mode,
        "split_method": ctx.split_plan.resolved_method,
        "requested_split_mode": ctx.split_plan.requested_mode,
        "requested_split_method": ctx.split_plan.requested_method,
        "train_fraction": ctx.split_plan.train_fraction,
        "validation_fraction": ctx.split_plan.validation_fraction,
        "holdout_fraction": ctx.split_plan.holdout_fraction,
        "random_seed": ctx.split_plan.random_seed,
        "split_note": ctx.split_plan.note,
        "n_train": int(len(ctx.tr)),
        "n_validation": int(len(ctx.va)),
        "n_holdout": int(len(holdout)),
        "n_fit": int(len(ctx.fit_df)),
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": valid_start,
        "validation_end": valid_end,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
    }
    out = ValidationOutputs()
    out.add_json(section="split_summary", file_name=_validation_path("split", "validation_split_summary.json"), payload=payload)
    return out


def _task_glm_diagnostics(ctx: ValidationContext) -> ValidationOutputs:
    if ctx.glm is None:
        return ValidationOutputs()

    diagnostic_df = ctx.fit_df if not ctx.fit_df.empty else (ctx.tr if not ctx.tr.empty else ctx.train_df)
    report = save_glm_diagnostics(
        diagnostic_df,
        glm=ctx.glm,
        target_col="home_win",
        out_dir=str(ensure_dir(ctx.out_dir / "glm" / "residuals" / "plots")),
        prefix="glm_validation",
        relative_plot_dir=_validation_path("glm", "residuals", "plots"),
    )
    out = ValidationOutputs()
    out.add_json(
        section="glm_residual_summary",
        file_name=_validation_path("glm", "residuals", "validation_glm_residual_summary.json"),
        payload=report["summary"],
    )
    out.add_csv(
        section="glm_residual_feature_summary",
        file_name=_validation_path("glm", "residuals", "validation_glm_residual_feature_summary.csv"),
        rows=report["feature_summary"],
    )
    out.add_csv(
        section="glm_working_residual_bins_linear_predictor",
        file_name=_validation_path("glm", "residuals", "validation_glm_working_residual_bins_linear_predictor.csv"),
        rows=report["linear_predictor_bins"],
    )
    out.add_csv(
        section="glm_working_residual_bins_features",
        file_name=_validation_path("glm", "residuals", "validation_glm_working_residual_bins_features.csv"),
        rows=report["feature_working_bins"],
    )
    out.add_csv(
        section="glm_working_residual_bins_weight",
        file_name=_validation_path("glm", "residuals", "validation_glm_working_residual_bins_weight.csv"),
        rows=report["weight_bins"],
    )
    out.add_csv(
        section="glm_partial_residual_bins",
        file_name=_validation_path("glm", "residuals", "validation_glm_partial_residual_bins.csv"),
        rows=report["partial_residual_bins"],
    )
    return out


def _task_permutation_importance(ctx: ValidationContext) -> ValidationOutputs:
    holdout = _holdout_df(ctx)
    if holdout.empty:
        return ValidationOutputs()

    for model_name in ["gbdt", "rf"]:
        model = ctx.models.get(model_name)
        if model is None or len(holdout) <= 25:
            continue
        model_feature_cols = getattr(model, "feature_columns", ctx.feature_cols) or ctx.feature_cols
        permutation_importance_report(
            model.model,
            holdout[model_feature_cols],
            holdout["home_win"].astype(int).to_numpy(),
            out_dir=str(ensure_dir(ctx.out_dir / "diagnostics" / "permutation_importance")),
            model_name=model_name,
        )
    return ValidationOutputs()


def _task_collinearity(ctx: ValidationContext) -> ValidationOutputs:
    report = assess_multicollinearity(ctx.train_df, features=ctx.diagnostic_feature_cols)
    out = ValidationOutputs()
    out.add_json(
        section="collinearity_summary",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_summary.json"),
        payload=report["summary"],
    )
    out.add_csv(
        section="vif",
        file_name=_validation_path("diagnostics", "collinearity", "validation_vif.csv"),
        rows=report["vif"],
    )
    out.add_csv(
        section="collinearity_structural",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_structural.csv"),
        rows=report["structural"],
    )
    out.add_csv(
        section="collinearity_pairs",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_pairs.csv"),
        rows=report["pairwise"],
    )
    out.add_csv(
        section="collinearity_condition",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_condition.csv"),
        rows=report["condition"],
    )
    out.add_csv(
        section="collinearity_variance_decomposition",
        file_name=_validation_path("diagnostics", "collinearity", "validation_collinearity_variance_decomposition.csv"),
        rows=report["variance_decomposition"],
    )
    return out


def _task_nonlinearity(ctx: ValidationContext) -> ValidationOutputs:
    report = assess_nonlinearity(
        ctx.tr if not ctx.tr.empty else ctx.train_df,
        ctx.va if not ctx.va.empty else (_holdout_df(ctx) if not _holdout_df(ctx).empty else ctx.train_df),
        features=ctx.diagnostic_feature_cols,
    )
    out = ValidationOutputs()
    out.add_json(
        section="nonlinearity_summary",
        file_name=_validation_path("diagnostics", "nonlinearity", "validation_nonlinearity_summary.json"),
        payload=report["summary"],
    )
    out.add_csv(
        section="nonlinearity_feature_summary",
        file_name=_validation_path("diagnostics", "nonlinearity", "validation_nonlinearity_feature_summary.csv"),
        rows=report["feature_summary"],
    )
    out.add_csv(
        section="nonlinearity_curve_points",
        file_name=_validation_path("diagnostics", "nonlinearity", "validation_nonlinearity_curve_points.csv"),
        rows=report["curve_points"],
    )
    return out


def _task_significance(ctx: ValidationContext) -> ValidationOutputs:
    holdout = _holdout_df(ctx)
    fit_df = ctx.fit_df if not ctx.fit_df.empty else ctx.tr
    sig = blockwise_nested_deviance_f_test(
        fit_df,
        holdout,
        feature_blocks=_feature_blocks_for(ctx),
        all_features=ctx.diagnostic_feature_cols,
    )
    ic = information_criteria_report(
        fit_df,
        holdout,
        feature_blocks=_feature_blocks_for(ctx),
        all_features=ctx.diagnostic_feature_cols,
    )
    out = ValidationOutputs()
    out.add_csv(
        section="significance",
        file_name=_validation_path("diagnostics", "significance", "validation_significance.csv"),
        rows=sig,
    )
    out.add_json(
        section="information_criteria_summary",
        file_name=_validation_path("diagnostics", "significance", "validation_information_criteria_summary.json"),
        payload=ic["summary"],
    )
    out.add_csv(
        section="information_criteria_candidates",
        file_name=_validation_path("diagnostics", "significance", "validation_information_criteria_candidates.csv"),
        rows=ic["candidates"],
    )
    return out


def _task_stability(ctx: ValidationContext) -> ValidationOutputs:
    glm_c = float(getattr(ctx.glm, "c", 1.0)) if ctx.glm is not None else 1.0
    cv_report = cv_glm_stability_report(
        ctx.train_df,
        features=ctx.diagnostic_feature_cols,
        n_splits=max(2, int(getattr(ctx.cfg.modeling, "cv_splits", 5) or 5)),
        c=glm_c,
    )
    bootstrap = bootstrap_glm_coefficients(
        ctx.fit_df if not ctx.fit_df.empty else ctx.train_df,
        features=ctx.diagnostic_feature_cols,
        c=glm_c,
    )
    out = ValidationOutputs()
    out.add_csv(
        section="coef_paths",
        file_name=_validation_path("diagnostics", "stability", "validation_coef_paths.csv"),
        rows=coefficient_paths(ctx.train_df, features=ctx.diagnostic_feature_cols),
        tail_rows=200,
    )
    out.add_json(
        section="break_test",
        file_name=_validation_path("diagnostics", "stability", "validation_break_test.json"),
        payload=break_test_trade_deadline(ctx.train_df, features=ctx.diagnostic_feature_cols, league=ctx.league),
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
    return out


def _task_influence(ctx: ValidationContext) -> ValidationOutputs:
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
    return out


def _task_fragility(ctx: ValidationContext) -> ValidationOutputs:
    base_df = _holdout_df(ctx) if not _holdout_df(ctx).empty else ctx.train_df
    out = ValidationOutputs()
    out.add_csv(
        section="fragility_missingness",
        file_name=_validation_path("diagnostics", "fragility", "validation_fragility_missingness.csv"),
        rows=missingness_stress_test(ctx.glm, base_df, feature_cols=ctx.diagnostic_feature_cols),
    )
    out.add_json(
        section="fragility_perturbation",
        file_name=_validation_path("diagnostics", "fragility", "validation_fragility_perturbation.json"),
        payload=perturbation_sensitivity(ctx.glm, base_df, feature_cols=ctx.diagnostic_feature_cols),
    )
    return out


def _task_calibration(ctx: ValidationContext) -> ValidationOutputs:
    holdout = _holdout_df(ctx)
    p = ctx.glm.predict_proba(holdout)
    y = holdout["home_win"].astype(int).to_numpy()
    payload = calibration_alpha_beta(y, p) | ece_mce(y, p)
    payload |= brier_decompose(y, p)

    out = ValidationOutputs()
    out.add_json(
        section="calibration_robustness",
        file_name=_validation_path("diagnostics", "calibration", "validation_calibration_robustness.json"),
        payload=payload,
    )
    return out


def _task_classification_curves(ctx: ValidationContext) -> ValidationOutputs:
    holdout = _holdout_df(ctx)
    p = ctx.glm.predict_proba(holdout)
    y = holdout["home_win"].astype(int).to_numpy()
    report = validate_logistic_probability_model(
        y,
        p,
        bins=max(2, int(ctx.cfg.modeling.calibration_bins)),
        current_tossup_half_width=0.05,
        plot_dir=ctx.plots_dir,
        plot_prefix="",
    )

    out = ValidationOutputs()
    out.add_json(
        section="logit_quantile_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_quantile_summary.json"),
        payload=report["quantile_summary"],
    )
    out.add_csv(
        section="logit_quantile_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_quantile_curve.csv"),
        rows=report["quantile_curve"],
    )
    out.add_json(
        section="logit_actual_vs_predicted_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_actual_vs_predicted_summary.json"),
        payload=report["actual_vs_predicted_summary"],
    )
    out.add_csv(
        section="logit_actual_vs_predicted_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_actual_vs_predicted_curve.csv"),
        rows=report["actual_vs_predicted_curve"],
    )
    out.add_json(
        section="logit_lift_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_lift_summary.json"),
        payload=report["lift_summary"],
    )
    out.add_csv(
        section="logit_lift_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_lift_curve.csv"),
        rows=report["lift_curve"],
    )
    out.add_json(
        section="logit_lorenz_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_lorenz_summary.json"),
        payload=report["lorenz_summary"],
    )
    out.add_csv(
        section="logit_lorenz_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_lorenz_curve.csv"),
        rows=report["lorenz_curve"],
    )
    out.add_json(
        section="logit_roc_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_roc_summary.json"),
        payload=report["roc_summary"],
    )
    out.add_csv(
        section="logit_roc_curve",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_roc_curve.csv"),
        rows=report["roc_curve"],
    )
    out.add_csv(
        section="logit_operating_points",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_operating_points.csv"),
        rows=report["operating_points"],
    )
    out.add_json(
        section="logit_tossup_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_tossup_summary.json"),
        payload=report["tossup_summary"],
    )
    out.add_csv(
        section="logit_tossup_sweep",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_tossup_sweep.csv"),
        rows=report["tossup_sweep"],
    )
    return out


def build_validation_tasks(
    *,
    extra_tasks: Sequence[ValidationTask] | None = None,
) -> list[ValidationTask]:
    tasks = [
        ValidationTask(name="split_summary", runner=_task_split_summary),
        ValidationTask(name="glm_diagnostics", runner=_task_glm_diagnostics, enabled=_has_glm),
        ValidationTask(name="permutation_importance", runner=_task_permutation_importance, enabled=_has_holdout),
        ValidationTask(name="collinearity", runner=_task_collinearity),
        ValidationTask(name="nonlinearity", runner=_task_nonlinearity, enabled=_has_holdout),
        ValidationTask(
            name="significance",
            runner=_task_significance,
            enabled=lambda ctx: _has_holdout(ctx) and not (ctx.fit_df if not ctx.fit_df.empty else ctx.tr).empty,
        ),
        ValidationTask(name="stability", runner=_task_stability, enabled=_has_glm),
        ValidationTask(name="influence", runner=_task_influence, enabled=_has_glm),
        ValidationTask(name="fragility", runner=_task_fragility, enabled=_has_glm),
        ValidationTask(name="calibration", runner=_task_calibration, enabled=lambda ctx: _has_glm(ctx) and _has_holdout(ctx)),
        ValidationTask(
            name="classification_curves",
            runner=_task_classification_curves,
            enabled=lambda ctx: _has_glm(ctx) and _has_holdout(ctx),
        ),
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
    _reset_validation_output_dirs(ctx)
    outputs = ValidationOutputs()
    selected_tasks = list(tasks) if tasks is not None else build_validation_tasks(extra_tasks=extra_tasks)

    for task in selected_tasks:
        if not task.should_run(ctx):
            continue
        outputs.merge(task.runner(ctx))

    outputs.write(ctx.out_dir, league=ctx.league)
    _archive_validation_outputs(ctx, outputs)
    return outputs
