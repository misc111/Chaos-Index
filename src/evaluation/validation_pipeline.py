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
from src.evaluation.validation_contract import (
    ValidationModelMetadata,
    build_validation_artifact_record,
    resolve_validation_model_metadata,
)
from src.evaluation.diagnostics_glm import save_glm_diagnostics
from src.evaluation.diagnostics_ml import permutation_importance_report
from src.evaluation.validation_groups import build_feature_blocks
from src.evaluation.mlb_data_quality import assess_mlb_data_quality
from src.evaluation.validation_fragility import missingness_stress_test, perturbation_sensitivity
from src.evaluation.validation_influence import influence_diagnostics
from src.evaluation.validation_nonlinearity import assess_nonlinearity
from src.evaluation.validation_significance import (
    blockwise_nested_deviance_f_test,
    information_criteria_report,
    penalized_block_ablation_report,
    penalized_information_criteria_report,
)
from src.evaluation.validation_stability import (
    assess_multicollinearity,
    bootstrap_glm_coefficients,
    break_test_trade_deadline,
    coefficient_paths,
    cv_glm_stability_report,
)
from src.models.gbdt import GBDTModel
from src.models.glm_penalized import build_penalized_glm
from src.models.rf import RFModel
from src.training.model_catalog import LEGACY_MODEL_KEYS, MODEL_ALIASES
from src.training.penalized_glm import primary_penalized_glm_name, selected_penalized_glm_models
from src.training.tune import quick_tune_penalized_glm

ValidationTaskRunner = Callable[["ValidationContext"], "ValidationOutputs | ValidationTaskResult"]
ValidationTaskPredicate = Callable[["ValidationContext"], bool]


def _canonical_league(league: str | None) -> str:
    token = str(league or "").strip().upper()
    if token in {"MLB", "NHL", "NBA"}:
        return token
    raise ValueError(f"Unsupported league '{league}'. Expected one of: MLB, NHL, NBA.")


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
    task_records: list[dict[str, Any]] = field(default_factory=list)

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
        self.task_records.extend([dict(record) for record in other.task_records])

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
        (root / "validation_outputs_contract.json").write_text(
            json.dumps({"validation_outputs": list(self.task_records)}, indent=2, sort_keys=True)
        )

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
    glm_validation_metadata: ValidationModelMetadata

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
        glm_name = primary_penalized_glm_name(selected_models, models)
        glm = models.get(glm_name) if glm_name else None
        diagnostic_feature_cols = list(
            getattr(glm, "feature_columns", [])
            or _validation_feature_columns(feature_cols, run_payload, glm_name or "glm_ridge")
            or feature_cols[: min(40, len(feature_cols))]
        )
        glm_validation_metadata = resolve_validation_model_metadata(run_payload=run_payload, model=glm)

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
            glm_validation_metadata=glm_validation_metadata,
        )


@dataclass(slots=True)
class ValidationTaskResult:
    outputs: ValidationOutputs = field(default_factory=ValidationOutputs)
    applicability: str = "applicable"
    summary: dict[str, Any] = field(default_factory=dict)


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
        "validation_task_count": int(len(outputs.task_records)),
        "validation_contract_file": "validation_outputs_contract.json",
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
        token = str(model_name).strip().lower()
        if not token:
            return ""
        canonical = MODEL_ALIASES.get(token, token)
        if not canonical:
            return ""
        token = str(canonical).strip()
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
        candidate_keys.extend(LEGACY_MODEL_KEYS.get(model_name, ()))
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

    for model_name in selected_penalized_glm_models(selected_models):
        if model_name not in selected:
            continue
        glm_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        glm_tune = dict(run_payload.get("glm_tuning_by_model", {}).get(model_name, {}))
        if "start_time_utc" in tr.columns:
            glm_tune = quick_tune_penalized_glm(
                tr,
                glm_cols,
                model_name=model_name,
                n_splits=3,
                min_train_size=min(140, max(70, len(tr) // 2)),
            )
        glm = build_penalized_glm(
            model_name,
            c=float(glm_tune.get("best_c", 1.0)),
            l1_ratio=glm_tune.get("best_l1_ratio"),
        )
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
    family: str = "diagnostics"

    def should_run(self, ctx: ValidationContext) -> bool:
        return True if self.enabled is None else bool(self.enabled(ctx))


def _has_glm(ctx: ValidationContext) -> bool:
    return ctx.glm is not None


def _has_holdout(ctx: ValidationContext) -> bool:
    return not _holdout_df(ctx).empty


def _feature_blocks_for(ctx: ValidationContext) -> dict[str, list[str]]:
    return build_feature_blocks(
        ctx.league,
        ctx.diagnostic_feature_cols,
        credibility=ctx.glm_validation_metadata.credibility,
    )


def _task_summary_with_model_metadata(ctx: ValidationContext, payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload)
    summary |= ctx.glm_validation_metadata.to_summary()
    return summary


def _task_result(
    ctx: ValidationContext,
    outputs: ValidationOutputs,
    *,
    summary: dict[str, Any] | None = None,
    applicability: str = "applicable",
) -> ValidationTaskResult:
    resolved_summary = _task_summary_with_model_metadata(ctx, summary or {}) if summary else {}
    return ValidationTaskResult(outputs=outputs, applicability=applicability, summary=resolved_summary)


def _task_split_summary(ctx: ValidationContext) -> ValidationTaskResult:
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
    return _task_result(ctx, out, summary=payload)


def _task_glm_diagnostics(ctx: ValidationContext) -> ValidationTaskResult:
    if ctx.glm is None:
        return ValidationTaskResult()

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
    return _task_result(ctx, out, summary=report["summary"])


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


def _task_collinearity(ctx: ValidationContext) -> ValidationTaskResult:
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
    return _task_result(ctx, out, summary=report["summary"])


def _task_nonlinearity(ctx: ValidationContext) -> ValidationTaskResult:
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
    return _task_result(ctx, out, summary=report["summary"])


def _task_significance(ctx: ValidationContext) -> ValidationTaskResult:
    holdout = _holdout_df(ctx)
    fit_df = ctx.fit_df if not ctx.fit_df.empty else ctx.tr
    feature_blocks = _feature_blocks_for(ctx)
    if ctx.glm_validation_metadata.uses_penalized_glm and ctx.glm is not None:
        glm_c = float(getattr(ctx.glm, "c", 1.0))
        glm_model_name = str(getattr(ctx.glm, "model_name", ctx.glm_validation_metadata.model_name or "glm_ridge"))
        glm_l1_ratio = getattr(ctx.glm, "l1_ratio", None)
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
    else:
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
    return _task_result(ctx, out, summary=summary_payload, applicability=applicability)


def _task_stability(ctx: ValidationContext) -> ValidationTaskResult:
    glm_c = float(getattr(ctx.glm, "c", 1.0)) if ctx.glm is not None else 1.0
    glm_model_name = str(getattr(ctx.glm, "model_name", "glm_ridge")) if ctx.glm is not None else "glm_ridge"
    glm_l1_ratio = getattr(ctx.glm, "l1_ratio", None) if ctx.glm is not None else None
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
    return _task_result(ctx, out, summary=summary_payload)


def _task_influence(ctx: ValidationContext) -> ValidationTaskResult:
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
    return _task_result(ctx, out, summary=infl_summary)


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
    return _task_result(ctx, out, summary=summary_payload)


def _task_calibration(ctx: ValidationContext) -> ValidationTaskResult:
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
    return _task_result(ctx, out, summary=payload)


def _task_classification_curves(ctx: ValidationContext) -> ValidationTaskResult:
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
    summary_payload = dict(report["roc_summary"])
    summary_payload["quantile_summary"] = report["quantile_summary"]
    summary_payload["tossup_summary"] = report["tossup_summary"]
    return _task_result(ctx, out, summary=summary_payload)


def _task_mlb_data_quality(ctx: ValidationContext) -> ValidationTaskResult:
    summary, issues = assess_mlb_data_quality(ctx.cfg.paths.db_path)
    out = ValidationOutputs()
    out.add_json(
        section="mlb_data_quality_summary",
        file_name=_validation_path("diagnostics", "data_quality", "validation_mlb_data_quality_summary.json"),
        payload=summary,
    )
    out.add_csv(
        section="mlb_data_quality_issues",
        file_name=_validation_path("diagnostics", "data_quality", "validation_mlb_data_quality_issues.csv"),
        rows=issues,
    )
    return _task_result(ctx, out, summary=summary)


def build_validation_tasks(
    *,
    extra_tasks: Sequence[ValidationTask] | None = None,
) -> list[ValidationTask]:
    tasks = [
        ValidationTask(name="split_summary", runner=_task_split_summary, family="split"),
        ValidationTask(name="mlb_data_quality", runner=_task_mlb_data_quality, enabled=lambda ctx: ctx.league == "MLB", family="data_quality"),
        ValidationTask(name="glm_diagnostics", runner=_task_glm_diagnostics, enabled=_has_glm, family="glm_diagnostics"),
        ValidationTask(name="permutation_importance", runner=_task_permutation_importance, enabled=_has_holdout, family="permutation_importance"),
        ValidationTask(name="collinearity", runner=_task_collinearity, family="collinearity"),
        ValidationTask(name="nonlinearity", runner=_task_nonlinearity, enabled=_has_holdout, family="nonlinearity"),
        ValidationTask(
            name="significance",
            runner=_task_significance,
            enabled=lambda ctx: _has_holdout(ctx) and not (ctx.fit_df if not ctx.fit_df.empty else ctx.tr).empty,
            family="significance",
        ),
        ValidationTask(name="stability", runner=_task_stability, enabled=_has_glm, family="stability"),
        ValidationTask(name="influence", runner=_task_influence, enabled=_has_glm, family="influence"),
        ValidationTask(name="fragility", runner=_task_fragility, enabled=_has_glm, family="fragility"),
        ValidationTask(
            name="calibration",
            runner=_task_calibration,
            enabled=lambda ctx: _has_glm(ctx) and _has_holdout(ctx),
            family="calibration",
        ),
        ValidationTask(
            name="classification_curves",
            runner=_task_classification_curves,
            enabled=lambda ctx: _has_glm(ctx) and _has_holdout(ctx),
            family="classification",
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
        task_result = task.runner(ctx)
        if isinstance(task_result, ValidationOutputs):
            normalized = ValidationTaskResult(outputs=task_result)
        else:
            normalized = task_result
        outputs.merge(normalized.outputs)
        if normalized.outputs.sections or normalized.summary:
            outputs.task_records.append(
                build_validation_artifact_record(
                    task_name=task.name,
                    family=task.family,
                    applicability=normalized.applicability,
                    artifacts=[spec.file_name for spec in normalized.outputs.sections],
                    summary=normalized.summary,
                ).to_dict()
            )

    outputs.write(ctx.out_dir, league=ctx.league)
    _archive_validation_outputs(ctx, outputs)
    return outputs
