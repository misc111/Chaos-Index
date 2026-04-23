from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
from src.evaluation.market_truth import build_market_truth, load_market_truth_moneyline_odds
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
from src.models.experimental.gbdt import GBDTModel
from src.models.extensions.glm_goals import GoalsPoissonModel
from src.models.extensions.glm_variants import DGLMMarginModel, GAMSplineModel, GLMMLogitModel, VanillaGLMModel
from src.models.lasso_credibility import LassoCredibilityModel
from src.models.glm_penalized import build_penalized_glm
from src.models.experimental.rf import RFModel
from src.registry.models import planned_credibility_model_catalog
from src.training.model_catalog import LEGACY_MODEL_KEYS, MODEL_ALIASES
from src.training.lasso_credibility import selected_lasso_credibility_models
from src.training.penalized_glm import selected_penalized_glm_models
from src.training.tune import quick_tune_penalized_glm

ValidationTaskRunner = Callable[["ValidationContext"], "ValidationOutputs | ValidationTaskResult"]
ValidationTaskPredicate = Callable[["ValidationContext"], bool]


def _canonical_league(league: str | None) -> str:
    token = str(league or "").strip().upper()
    if token == "MLB":
        return token
    raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")


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

    def write(
        self,
        out_dir: Path,
        *,
        league: str,
        manifest_metadata: Mapping[str, Any] | None = None,
    ) -> None:
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
        if manifest_metadata:
            manifest["metadata"] = dict(manifest_metadata)
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
    primary_model_resolution: dict[str, Any]

    @classmethod
    def from_result(cls, result: dict[str, Any], cfg: AppConfig) -> "ValidationContext":
        train_df = result["train_df"].copy()
        feature_cols = list(result["feature_columns"])
        run_payload = result.get("run_payload", {})
        selected_models = _selected_validation_models(result, run_payload)
        split_plan, tr, va, te = _validation_split(train_df, cfg)
        fit_df = _concat_frames(tr, va)
        models, model_fit_status = _fit_validation_models(
            fit_df,
            feature_cols=feature_cols,
            selected_models=selected_models,
            run_payload=run_payload,
        )
        league = _canonical_league(cfg.data.league)
        out_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "validation" / league.lower())
        plots_dir = ensure_dir(Path(cfg.paths.artifacts_dir) / "plots" / league.lower() / "glm" / "performance")
        glm_name, glm, primary_model_resolution = _resolve_primary_validation_model(
            selected_models=selected_models,
            models=models,
            model_fit_status=model_fit_status,
            run_payload=run_payload,
        )
        diagnostic_feature_cols = list(
            getattr(glm, "feature_columns", [])
            or _validation_feature_columns(feature_cols, run_payload, glm_name or "glm_vanilla")
            or feature_cols[: min(40, len(feature_cols))]
        )
        glm_validation_metadata = resolve_validation_model_metadata(
            run_payload={**run_payload, "glm_primary_model": glm_name or run_payload.get("glm_primary_model")},
            model=glm,
            model_name=glm_name,
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
            glm_validation_metadata=glm_validation_metadata,
            primary_model_resolution=primary_model_resolution,
        )


@dataclass(slots=True)
class ValidationTaskResult:
    outputs: ValidationOutputs = field(default_factory=ValidationOutputs)
    applicability: str = "applicable"
    summary: dict[str, Any] = field(default_factory=dict)


def _safe_archive_token(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("_")
    return cleaned or "adhoc_validation"


def _non_empty_string(value: Any) -> str | None:
    token = str(value or "").strip()
    return token or None


def _metadata_scopes(run_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    scopes: list[Mapping[str, Any]] = [run_payload]
    for key in ("metadata", "run_metadata"):
        value = run_payload.get(key)
        if isinstance(value, Mapping):
            scopes.append(value)
    for key in ("run_contract", "model_run_contract"):
        contract = run_payload.get(key)
        if not isinstance(contract, Mapping):
            continue
        scopes.append(contract)
        contract_metadata = contract.get("metadata")
        if isinstance(contract_metadata, Mapping):
            scopes.append(contract_metadata)
    return scopes


def _first_metadata_value(scopes: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str | None:
    for scope in scopes:
        for key in keys:
            value = _non_empty_string(scope.get(key))
            if value:
                return value
    return None


def _resolve_execution_scope_labels(run_payload: Mapping[str, Any]) -> dict[str, str]:
    scopes = _metadata_scopes(run_payload)
    execution_data_scope = _first_metadata_value(scopes, ("execution_data_scope", "execution_scope", "data_scope"))
    execution_source_label = _first_metadata_value(
        scopes,
        ("execution_source_label", "source_label", "execution_source", "source"),
    )
    if not execution_data_scope and not execution_source_label:
        return {}

    payload: dict[str, str] = {}
    if execution_data_scope:
        payload["execution_data_scope"] = execution_data_scope
    if execution_source_label:
        payload["execution_source_label"] = execution_source_label
    combined = f"{execution_data_scope or ''} {execution_source_label or ''}".lower()
    if "fixture" in combined or "demo" in combined:
        payload["execution_label_class"] = "fixture_or_demo"
    return payload


def _fixture_demo_evidence_labels(scoped_labels: Mapping[str, Any]) -> dict[str, Any]:
    label_class = str(scoped_labels.get("execution_label_class") or "").strip().lower()
    if label_class != "fixture_or_demo":
        return {}
    return {
        "evidence_scope": "bounded_non_production",
        "evidence_grade": "non_production_fixture_or_demo",
        "evidence_boundary_note": "Validation evidence is fixture/demo bounded and should not be treated as production-grade.",
    }


def _has_execution_metadata_labels(scope: Mapping[str, Any]) -> bool:
    return bool(
        _first_metadata_value(
            [scope],
            (
                "execution_data_scope",
                "execution_scope",
                "data_scope",
                "execution_source_label",
                "source_label",
                "execution_source",
                "source",
            ),
        )
    ) or any(
        key in scope
        for key in (
            "production_grade",
            "fixture_rows",
            "feature_count",
            "artifact_purpose",
            "full_mlb_blocker",
        )
    )


def _execution_metadata_from_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    nested = scope.get("execution_metadata")
    if isinstance(nested, Mapping) and nested:
        return dict(nested)
    if _has_execution_metadata_labels(scope):
        metadata = dict(scope)
        metadata.pop("execution_metadata", None)
        return metadata
    return {}


def _resolve_execution_metadata(run_payload: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("metadata", "run_metadata"):
        value = run_payload.get(key)
        if not isinstance(value, Mapping):
            continue
        metadata = _execution_metadata_from_scope(value)
        if metadata:
            return metadata

    for key in ("run_contract", "model_run_contract"):
        contract = run_payload.get(key)
        if not isinstance(contract, Mapping):
            continue
        for metadata_key in ("metadata", "run_metadata"):
            value = contract.get(metadata_key)
            if not isinstance(value, Mapping):
                continue
            metadata = _execution_metadata_from_scope(value)
            if metadata:
                return metadata
        nested = contract.get("execution_metadata")
        if isinstance(nested, Mapping):
            return dict(nested)

    payload: dict[str, Any] = {}
    for key in (
        "execution_data_scope",
        "execution_scope",
        "data_scope",
        "execution_source_label",
        "source_label",
        "execution_source",
        "production_grade",
        "fixture_rows",
        "feature_count",
        "artifact_purpose",
        "full_mlb_blocker",
    ):
        if key in run_payload:
            payload[key] = run_payload[key]
    return payload


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
        top_level, remainder = rel.split("/", 1)
        grouped_validation_files.setdefault(top_level, []).append(remainder)
    selected_models_payload = ctx.run_payload.get("selected_models", [])
    selected_models: list[str] = []
    if isinstance(selected_models_payload, Sequence) and not isinstance(selected_models_payload, (str, bytes)):
        selected_models = [str(model) for model in selected_models_payload if str(model).strip()]
    if not selected_models:
        selected_models = [str(model) for model in ctx.models.keys() if str(model).strip()]
    model_run_id = _non_empty_string(ctx.run_payload.get("model_run_id")) or "adhoc_validation"
    execution_metadata = _resolve_execution_metadata(ctx.run_payload)
    scoped_labels = _resolve_execution_scope_labels(ctx.run_payload)
    if not scoped_labels and execution_metadata:
        scoped_labels = _resolve_execution_scope_labels(execution_metadata)
    evidence_labels = _fixture_demo_evidence_labels(scoped_labels)

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

    metadata = {
        "league": ctx.league,
        "generated_at_utc": generated_at_utc,
        "archive_id": archive_root.name,
        "archive_date": generated_at_utc[:10],
        "model_run_id": model_run_id,
        "feature_set_version": ctx.run_payload.get("feature_set_version"),
        "selected_models": selected_models,
        "execution_metadata": dict(execution_metadata),
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
            "total_files": len(root_files)
            + sum(len(files) for files in grouped_validation_files.values())
            + len(performance_files),
        },
        "artifact_groups": artifact_groups,
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }
    metadata.update(scoped_labels)
    metadata.update(evidence_labels)
    if "production_grade" in execution_metadata:
        metadata["production_grade"] = execution_metadata["production_grade"]
    if "data_origin" in execution_metadata:
        metadata["data_origin"] = execution_metadata["data_origin"]
    return metadata


def _reset_validation_output_dirs(ctx: ValidationContext) -> None:
    if ctx.out_dir.exists():
        shutil.rmtree(ctx.out_dir)
    ensure_dir(ctx.out_dir)
    if ctx.plots_dir.exists():
        shutil.rmtree(ctx.plots_dir)
    ensure_dir(ctx.plots_dir)


def _validation_manifest_metadata(ctx: ValidationContext) -> dict[str, Any]:
    execution_metadata = _resolve_execution_metadata(ctx.run_payload)
    scoped_labels = _resolve_execution_scope_labels(ctx.run_payload)
    if not scoped_labels and execution_metadata:
        scoped_labels = _resolve_execution_scope_labels(execution_metadata)
    payload: dict[str, Any] = {
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }
    payload.update(scoped_labels)
    payload.update(_fixture_demo_evidence_labels(scoped_labels))
    return payload


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

    if model_name in {"glm_ridge", "glm_lasso", "glm_elastic_net", "glm_vanilla"}:
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
) -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
    fit_status: dict[str, dict[str, Any]] = {}
    selected = [str(model) for model in selected_models if str(model).strip()]
    if tr.empty:
        for model_name in selected:
            fit_status[model_name] = {"status": "not_applicable", "reason": "empty_training_frame"}
        return {}, fit_status

    selected_set = set(selected)
    models: dict[str, object] = {}
    credibility_catalog = planned_credibility_model_catalog()

    def _mark_status(model_name: str, *, status: str, **payload: Any) -> None:
        fit_status[str(model_name)] = {"status": str(status), **payload}

    for model_name in selected_penalized_glm_models(selected):
        if model_name not in selected_set:
            continue
        glm_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        try:
            glm_tuning_by_model = run_payload.get("glm_tuning_by_model")
            if isinstance(glm_tuning_by_model, Mapping):
                glm_tune = dict(glm_tuning_by_model.get(model_name, {}))
            else:
                glm_tune = {}
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
            _mark_status(model_name, status="fit_ok", model_kind="penalized_glm", feature_count=len(glm_cols))
        except Exception as exc:
            _mark_status(
                model_name,
                status="fit_failed",
                model_kind="penalized_glm",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    credibility_tuning_by_model: dict[str, Any] = {}
    for key in ("credibility_tuning_by_model", "lasso_credibility_tuning_by_model"):
        payload = run_payload.get(key)
        if isinstance(payload, Mapping):
            credibility_tuning_by_model |= dict(payload)
    for model_name in selected_lasso_credibility_models(selected):
        if model_name not in selected_set:
            continue
        payload = dict(credibility_catalog.get(model_name, {}))
        if not payload:
            _mark_status(
                model_name,
                status="not_supported",
                model_kind="lasso_credibility",
                reason="missing_credibility_catalog_payload",
            )
            continue
        params = dict(credibility_tuning_by_model.get(model_name, {}))
        lambda_value = float(params.get("best_lambda", params.get("lambda_value", 1.0)) or 1.0)
        if not np.isfinite(lambda_value) or lambda_value <= 0:
            lambda_value = 1.0
        credibility_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        try:
            model = LassoCredibilityModel(
                model_name=model_name,
                lambda_value=lambda_value,
                complement_kind=str(payload.get("complement_kind") or ""),
                complement_column=str(payload.get("complement_column") or ""),
                complement_label=str(payload.get("complement_label") or ""),
                complement_input_scale=str(payload.get("offset_scale") or "logit"),
            )
            model.fit(tr, credibility_cols)
            models[model_name] = model
            _mark_status(model_name, status="fit_ok", model_kind="lasso_credibility", feature_count=len(credibility_cols))
        except Exception as exc:
            _mark_status(
                model_name,
                status="fit_failed",
                model_kind="lasso_credibility",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    extension_builders: dict[str, type] = {
        "glm_vanilla": VanillaGLMModel,
        "gam_spline": GAMSplineModel,
        "glmm_logit": GLMMLogitModel,
        "dglm_margin": DGLMMarginModel,
    }
    for model_name, model_cls in extension_builders.items():
        if model_name not in selected_set:
            continue
        model_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        try:
            model = model_cls()
            model.fit(tr, model_cols)
            models[model_name] = model
            _mark_status(model_name, status="fit_ok", model_kind="extension_or_vanilla", feature_count=len(model_cols))
        except Exception as exc:
            _mark_status(
                model_name,
                status="fit_failed",
                model_kind="extension_or_vanilla",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    if "goals_poisson" in selected_set:
        try:
            model = GoalsPoissonModel()
            model.fit(tr)
            models[model.model_name] = model
            _mark_status("goals_poisson", status="fit_ok", model_kind="extension_or_vanilla", feature_count=0)
        except Exception as exc:
            _mark_status(
                "goals_poisson",
                status="fit_failed",
                model_kind="extension_or_vanilla",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    for model_name, model_cls in (("gbdt", GBDTModel), ("rf", RFModel)):
        if model_name not in selected_set:
            continue
        model_cols = _validation_feature_columns(feature_cols, run_payload, model_name)
        try:
            model = model_cls()
            model.fit(tr, model_cols)
            models[model_name] = model
            _mark_status(model_name, status="fit_ok", model_kind="ml_baseline", feature_count=len(model_cols))
        except Exception as exc:
            _mark_status(
                model_name,
                status="fit_failed",
                model_kind="ml_baseline",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    for model_name in selected:
        fit_status.setdefault(model_name, {"status": "not_supported", "reason": "validation_wrapper_unavailable"})

    return models, fit_status


def _resolve_primary_validation_model(
    *,
    selected_models: list[str],
    models: Mapping[str, object],
    model_fit_status: Mapping[str, Mapping[str, Any]],
    run_payload: Mapping[str, Any],
) -> tuple[str | None, object | None, dict[str, Any]]:
    preferred_default_order = [
        "glm_ridge",
        "glm_elastic_net",
        "glm_lasso",
        "glm_lasso_market_credibility",
        "glm_lasso_prior_credibility",
        "glm_vanilla",
        "gam_spline",
        "glmm_logit",
        "dglm_margin",
        "goals_poisson",
        "gbdt",
        "rf",
    ]
    requested_primary_raw = str(run_payload.get("glm_primary_model") or "").strip().lower()
    requested_primary = MODEL_ALIASES.get(requested_primary_raw, requested_primary_raw) if requested_primary_raw else ""
    candidate_order: list[str] = []
    if requested_primary:
        candidate_order.append(str(requested_primary))
    candidate_order.extend([str(name) for name in selected_models if str(name).strip()])
    candidate_order.extend(preferred_default_order)

    resolved_model_name: str | None = None
    resolved_model: object | None = None
    for model_name in list(dict.fromkeys(candidate_order)):
        model = models.get(model_name)
        if model is None or not callable(getattr(model, "predict_proba", None)):
            continue
        resolved_model_name = model_name
        resolved_model = model
        break

    if resolved_model_name is None:
        resolution_status = "not_supported" if selected_models else "not_applicable"
        resolution_note = "No selected validation model could be fit with a probability interface."
    elif requested_primary and resolved_model_name != requested_primary:
        resolution_status = "partial"
        resolution_note = (
            f"Requested primary model '{requested_primary}' was unavailable for validation; "
            f"fell back to '{resolved_model_name}'."
        )
    else:
        resolution_status = "supported"
        resolution_note = f"Resolved primary validation model '{resolved_model_name}'."

    resolution = {
        "status": resolution_status,
        "note": resolution_note,
        "requested_primary_model": requested_primary or None,
        "resolved_primary_model": resolved_model_name,
        "selected_models": [str(name) for name in selected_models if str(name).strip()],
        "model_fit_status": {str(key): dict(value) for key, value in model_fit_status.items()},
    }
    return resolved_model_name, resolved_model, resolution


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


_PENALIZED_VALIDATION_MODEL_KEYS = {"glm_ridge", "glm_lasso", "glm_elastic_net"}
_CREDIBILITY_VALIDATION_MODEL_KEYS = {"glm_lasso_market_credibility", "glm_lasso_prior_credibility"}
_EXTENSION_VALIDATION_MODEL_KEYS = {"gam_spline", "glmm_logit", "dglm_margin", "goals_poisson"}


def _primary_model_key(ctx: ValidationContext) -> str:
    return str(ctx.glm_validation_metadata.model_key or ctx.glm_validation_metadata.model_name or "").strip()


def _is_penalized_primary_model(ctx: ValidationContext) -> bool:
    return _primary_model_key(ctx) in _PENALIZED_VALIDATION_MODEL_KEYS


def _is_credibility_primary_model(ctx: ValidationContext) -> bool:
    return _primary_model_key(ctx) in _CREDIBILITY_VALIDATION_MODEL_KEYS or ctx.glm_validation_metadata.credibility is not None


def _is_vanilla_primary_model(ctx: ValidationContext) -> bool:
    return _primary_model_key(ctx) == "glm_vanilla"


def _is_extension_primary_model(ctx: ValidationContext) -> bool:
    return _primary_model_key(ctx) in _EXTENSION_VALIDATION_MODEL_KEYS or str(ctx.glm_validation_metadata.model_lane or "") == "extension"


def _task_support_payload(
    ctx: ValidationContext,
    *,
    task_name: str,
    support_status: str,
    route: str,
    note: str,
) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "task_support_status": support_status,
        "task_support_route": route,
        "task_support_note": note,
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }


def _unsupported_task_result(
    ctx: ValidationContext,
    *,
    task_name: str,
    note: str,
    applicability: str = "not_applicable",
    route: str = "unsupported_for_selected_model_family",
) -> ValidationTaskResult:
    return _task_result(
        ctx,
        ValidationOutputs(),
        summary=_task_support_payload(
            ctx,
            task_name=task_name,
            support_status=applicability,
            route=route,
            note=note,
        ),
        applicability=applicability,
    )


def _validation_surrogate_penalized_spec(
    ctx: ValidationContext,
) -> tuple[str | None, float | None, float | None, dict[str, Any]]:
    model_key = _primary_model_key(ctx)
    if _is_penalized_primary_model(ctx):
        return (
            model_key,
            float(getattr(ctx.glm, "c", 1.0)) if ctx.glm is not None else 1.0,
            getattr(ctx.glm, "l1_ratio", None) if ctx.glm is not None else None,
            _task_support_payload(
                ctx,
                task_name="penalized_surrogate",
                support_status="applicable",
                route="selected_model_direct",
                note="Using the selected penalized GLM directly for penalized-family diagnostics.",
            ),
        )

    if _is_credibility_primary_model(ctx):
        driver_model = "glm_lasso"
        catalog_payload = planned_credibility_model_catalog().get(model_key, {})
        planned_driver = str(catalog_payload.get("driver_source_model") or "").strip()
        if planned_driver in _PENALIZED_VALIDATION_MODEL_KEYS:
            driver_model = planned_driver
        credibility_tuning: Mapping[str, Any] | None = None
        for key in ("credibility_tuning_by_model", "lasso_credibility_tuning_by_model"):
            payload = ctx.run_payload.get(key)
            if isinstance(payload, Mapping):
                candidate = payload.get(model_key)
                if isinstance(candidate, Mapping):
                    credibility_tuning = candidate
                    break
        surrogate_c = 1.0
        if credibility_tuning is not None:
            surrogate_c = float(credibility_tuning.get("best_c", credibility_tuning.get("c", 1.0)) or 1.0)
        if not np.isfinite(surrogate_c) or surrogate_c <= 0.0:
            surrogate_c = 1.0
        return (
            driver_model,
            surrogate_c,
            None,
            _task_support_payload(
                ctx,
                task_name="penalized_surrogate",
                support_status="partial",
                route="credibility_driver_surrogate",
                note=(
                    "Using the credibility lane's penalized driver model for diagnostics that do not "
                    "support offsets or credibility shrinkage directly."
                ),
            ),
        )

    return None, None, None, _task_support_payload(
        ctx,
        task_name="penalized_surrogate",
        support_status="not_applicable",
        route="no_penalized_surrogate_available",
        note="No valid penalized-GLM surrogate is available for the selected validation primary model.",
    )


def _binary_holdout_profile(y_true: np.ndarray | pd.Series) -> dict[str, Any]:
    y_raw = np.asarray(y_true, dtype=float)
    y = y_raw[np.isfinite(y_raw)].astype(int)
    n_obs = int(len(y))
    if n_obs <= 0:
        return {
            "n_obs": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": float("nan"),
            "has_class_contrast": False,
            "overall_applicability": "not_applicable",
        }

    positive_count = int(np.sum(y))
    negative_count = max(0, n_obs - positive_count)
    has_class_contrast = positive_count > 0 and negative_count > 0
    return {
        "n_obs": n_obs,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": float(positive_count / n_obs),
        "has_class_contrast": bool(has_class_contrast),
        "overall_applicability": "applicable" if has_class_contrast else "partial",
    }


def _probability_model_family_applicability(
    ctx: ValidationContext,
    *,
    task_name: str,
    requires_class_contrast: bool,
    class_contrast_available: bool,
    n_obs: int,
) -> dict[str, Any]:
    if ctx.glm is None:
        resolution_status = str(ctx.primary_model_resolution.get("status") or "")
        if resolution_status in {"partial", "not_supported"}:
            status = "pending"
        else:
            status = "not_applicable"
    elif n_obs <= 0:
        status = "not_applicable"
    elif requires_class_contrast and not class_contrast_available:
        status = "partial"
    else:
        status = "applicable"
    return {
        "task": task_name,
        "status": status,
        "model_family": str(ctx.glm_validation_metadata.model_family or "unknown"),
        "model_lane": str(ctx.glm_validation_metadata.model_lane or "unknown"),
        "model_key": str(ctx.glm_validation_metadata.model_key or ""),
        "requires_class_contrast": bool(requires_class_contrast),
        "class_contrast_available": bool(class_contrast_available),
        "probability_output_method": "predict_proba" if ctx.glm is not None and hasattr(ctx.glm, "predict_proba") else "missing",
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }


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


_MODEL_DEPENDENT_TASKS = {
    "glm_diagnostics",
    "stability",
    "influence",
    "fragility",
    "calibration",
    "market_truth",
    "classification_curves",
}
_HOLDOUT_DEPENDENT_TASKS = {
    "nonlinearity",
    "significance",
    "permutation_importance",
    "calibration",
    "market_truth",
    "classification_curves",
}


def _default_model_family_applicability(
    ctx: ValidationContext,
    *,
    task_name: str,
    contract_applicability: str,
) -> dict[str, Any]:
    if task_name in _MODEL_DEPENDENT_TASKS:
        resolution_status = str(ctx.primary_model_resolution.get("status") or "")
        if ctx.glm is not None:
            status = "applicable" if contract_applicability == "applicable" else contract_applicability
        elif resolution_status in {"partial", "not_supported"}:
            status = "pending"
        else:
            status = "not_applicable"
        reason = (
            str(ctx.primary_model_resolution.get("note") or "")
            if status in {"pending", "partial"}
            else "primary_model_available"
        )
    else:
        status = "not_applicable"
        reason = "task_not_model_family_specific"

    return {
        "task": task_name,
        "status": status,
        "reason": reason,
        "model_family": str(ctx.glm_validation_metadata.model_family or "unknown"),
        "model_lane": str(ctx.glm_validation_metadata.model_lane or "unknown"),
        "model_key": str(ctx.glm_validation_metadata.model_key or ""),
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }


def _record_task_summary(
    ctx: ValidationContext,
    *,
    task_name: str,
    task_result: ValidationTaskResult,
) -> dict[str, Any]:
    summary = dict(task_result.summary or {})
    model_family_applicability = summary.get("model_family_applicability")
    if not isinstance(model_family_applicability, Mapping):
        model_family_applicability = _default_model_family_applicability(
            ctx,
            task_name=task_name,
            contract_applicability=task_result.applicability,
        )
        summary["model_family_applicability"] = model_family_applicability
    summary["contract_applicability"] = str(task_result.applicability)
    summary["applicability_split"] = {
        "contract_level": str(task_result.applicability),
        "model_family_level": str(model_family_applicability.get("status") or "not_applicable"),
    }
    return _task_summary_with_model_metadata(ctx, summary)


def _skipped_task_result(ctx: ValidationContext, task: "ValidationTask") -> ValidationTaskResult:
    if task.name in _MODEL_DEPENDENT_TASKS and ctx.glm is None:
        resolution_status = str(ctx.primary_model_resolution.get("status") or "")
        applicability = "pending" if resolution_status in {"partial", "not_supported"} else "not_applicable"
        reason = "primary_validation_model_unavailable"
    elif task.name in _HOLDOUT_DEPENDENT_TASKS and _holdout_df(ctx).empty:
        applicability = "not_applicable"
        reason = "holdout_split_unavailable"
    else:
        applicability = "pending"
        reason = "task_predicate_not_met"
    summary = {
        "status": "skipped",
        "skip_reason": reason,
        "primary_model_resolution": dict(ctx.primary_model_resolution),
    }
    return ValidationTaskResult(outputs=ValidationOutputs(), applicability=applicability, summary=_task_summary_with_model_metadata(ctx, summary))


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
    out.add_csv(
        section="glm_raw_residual_rows",
        file_name=_validation_path("glm", "residuals", "validation_glm_raw_residual_rows.csv"),
        rows=report["raw_residuals"],
    )
    out.add_csv(
        section="glm_residual_vs_fitted_bins",
        file_name=_validation_path("glm", "residuals", "validation_glm_residual_vs_fitted_bins.csv"),
        rows=report["residual_vs_fitted_bins"],
    )
    out.add_csv(
        section="glm_grouped_residual_summary",
        file_name=_validation_path("glm", "residuals", "validation_glm_grouped_residual_summary.csv"),
        rows=report["grouped_residual_summary"],
    )
    out.add_json(
        section="glm_grouped_residual_metadata",
        file_name=_validation_path("glm", "residuals", "validation_glm_grouped_residual_metadata.json"),
        payload=report["grouped_residual_metadata"],
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


def _task_calibration(ctx: ValidationContext) -> ValidationTaskResult:
    holdout = _holdout_df(ctx)
    p = ctx.glm.predict_proba(holdout)
    y = holdout["home_win"].astype(int).to_numpy()
    holdout_profile = _binary_holdout_profile(y)
    metric_applicability = {
        "ece_mce": "applicable" if holdout_profile["n_obs"] > 0 else "not_applicable",
        "alpha_beta": "applicable" if holdout_profile["has_class_contrast"] else "not_applicable",
        "brier_decomposition": "applicable" if holdout_profile["n_obs"] > 0 else "not_applicable",
    }
    payload = calibration_alpha_beta(y, p) | ece_mce(y, p)
    payload |= brier_decompose(y, p)
    payload["holdout_profile"] = holdout_profile
    payload["metric_applicability"] = metric_applicability
    payload["model_family_applicability"] = _probability_model_family_applicability(
        ctx,
        task_name="calibration",
        requires_class_contrast=False,
        class_contrast_available=bool(holdout_profile["has_class_contrast"]),
        n_obs=int(holdout_profile["n_obs"]),
    )

    out = ValidationOutputs()
    out.add_json(
        section="calibration_robustness",
        file_name=_validation_path("diagnostics", "calibration", "validation_calibration_robustness.json"),
        payload=payload,
    )
    return _task_result(ctx, out, summary=payload, applicability=str(holdout_profile["overall_applicability"]))


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
    holdout_profile = dict(report["applicability_summary"])
    metric_applicability = {
        "quantile_curve": str(holdout_profile["calibration_curve_applicability"]),
        "actual_vs_predicted_curve": str(holdout_profile["calibration_curve_applicability"]),
        "lift_curve": str(holdout_profile["lift_curve_applicability"]),
        "lorenz_curve": str(holdout_profile["lift_curve_applicability"]),
        "roc_curve": str(holdout_profile["roc_curve_applicability"]),
        "operating_points": str(holdout_profile["roc_curve_applicability"]),
        "tossup_sweep": str(holdout_profile["calibration_curve_applicability"]),
    }
    model_family_applicability = _probability_model_family_applicability(
        ctx,
        task_name="classification_curves",
        requires_class_contrast=True,
        class_contrast_available=bool(holdout_profile["has_class_contrast"]),
        n_obs=int(holdout_profile["n_obs"]),
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
    classification_summary = {
        "holdout_profile": holdout_profile,
        "metric_applicability": metric_applicability,
        "model_family_applicability": model_family_applicability,
        "quantile_summary": report["quantile_summary"],
        "actual_vs_predicted_summary": report["actual_vs_predicted_summary"],
        "lift_summary": report["lift_summary"],
        "lorenz_summary": report["lorenz_summary"],
        "roc_summary": report["roc_summary"],
        "tossup_summary": report["tossup_summary"],
    }
    out.add_json(
        section="logit_classification_summary",
        file_name=_validation_path("diagnostics", "classification", "validation_logit_classification_summary.json"),
        payload=classification_summary,
    )
    summary_payload = dict(report["roc_summary"])
    summary_payload["quantile_summary"] = report["quantile_summary"]
    summary_payload["actual_vs_predicted_summary"] = report["actual_vs_predicted_summary"]
    summary_payload["lift_summary"] = report["lift_summary"]
    summary_payload["lorenz_summary"] = report["lorenz_summary"]
    summary_payload["tossup_summary"] = report["tossup_summary"]
    summary_payload["holdout_profile"] = holdout_profile
    summary_payload["metric_applicability"] = metric_applicability
    summary_payload["model_family_applicability"] = model_family_applicability
    return _task_result(
        ctx,
        out,
        summary=summary_payload,
        applicability=str(holdout_profile["overall_applicability"]),
    )


def _validation_market_truth_prediction_frame(ctx: ValidationContext, holdout: pd.DataFrame) -> pd.DataFrame:
    meta_columns = [
        column
        for column in (
            "game_id",
            "season",
            "game_date_utc",
            "start_time_utc",
            "available_as_of_utc",
            "home_team",
            "away_team",
            "home_win",
        )
        if column in holdout.columns
    ]
    predictions = holdout[meta_columns].copy()
    start = pd.to_datetime(predictions["start_time_utc"], errors="coerce", utc=True)
    if "available_as_of_utc" in predictions.columns:
        available = pd.to_datetime(predictions["available_as_of_utc"], errors="coerce", utc=True)
    else:
        available = pd.Series(pd.NaT, index=predictions.index, dtype="datetime64[ns, UTC]")
    prediction_as_of = available.where(available.notna() & start.notna() & available.le(start), start - pd.Timedelta(hours=1))
    predictions["as_of_utc"] = prediction_as_of.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(prediction_as_of.notna(), None)
    model_key = _primary_model_key(ctx) or "primary_model"
    predictions["model_name"] = model_key
    predictions["prob_home_win"] = np.asarray(ctx.glm.predict_proba(holdout), dtype=float)
    return predictions


def _task_market_truth(ctx: ValidationContext) -> ValidationTaskResult:
    holdout = _holdout_df(ctx)
    required = {"game_id", "start_time_utc", "home_win"}
    missing = sorted(required - set(holdout.columns))
    if ctx.glm is None:
        return _unsupported_task_result(
            ctx,
            task_name="market_truth",
            note="Market-truth validation requires a primary probability model.",
        )
    if holdout.empty or missing:
        return _unsupported_task_result(
            ctx,
            task_name="market_truth",
            note=f"Market-truth validation requires holdout rows with {sorted(required)}. Missing={missing}",
        )

    game_ids = [int(value) for value in holdout["game_id"].dropna().unique().tolist()]
    odds_lines = load_market_truth_moneyline_odds(ctx.cfg.paths.db_path, league=ctx.league, game_ids=game_ids)
    if odds_lines.empty:
        return _unsupported_task_result(
            ctx,
            task_name="market_truth",
            note="No moneyline odds snapshots were available for the validation holdout games.",
        )

    predictions = _validation_market_truth_prediction_frame(ctx, holdout)
    model_key = _primary_model_key(ctx) or "primary_model"
    result = build_market_truth(
        predictions,
        odds_lines,
        model_names=[model_key],
        min_edge=0.0,
        calibration_bins=ctx.cfg.modeling.calibration_bins,
    )
    integrity_counts = (
        result.per_game["timestamp_integrity"].value_counts(dropna=False).to_dict() if "timestamp_integrity" in result.per_game.columns else {}
    )
    summary_rows = result.summary.to_dict(orient="records") if not result.summary.empty else []
    current_coverage = float(result.summary["current_market_coverage"].min()) if "current_market_coverage" in result.summary else 0.0
    closing_coverage = float(result.summary["closing_market_coverage"].min()) if "closing_market_coverage" in result.summary else 0.0
    applicability = "applicable" if current_coverage > 0 and closing_coverage > 0 else "partial"
    payload = {
        "status": "ok" if summary_rows else "partial",
        "market_truth_classification": "betting_overlay",
        "note": "Market truth compares validation probabilities against prediction-time and closing no-vig moneyline prices.",
        "n_predictions": int(len(result.per_game)),
        "timestamp_integrity_counts": {str(key): int(value) for key, value in integrity_counts.items()},
        "models": summary_rows,
        "market_data_coverage": {
            "current_market_min": current_coverage,
            "closing_market_min": closing_coverage,
        },
        "model_family_applicability": _probability_model_family_applicability(
            ctx,
            task_name="market_truth",
            requires_class_contrast=False,
            class_contrast_available=True,
            n_obs=int(len(result.per_game)),
        ),
    }

    out = ValidationOutputs()
    out.add_json(
        section="market_truth_summary",
        file_name=_validation_path("market_truth", "validation_market_truth_summary.json"),
        payload=payload,
    )
    out.add_csv(
        section="market_truth_by_model",
        file_name=_validation_path("market_truth", "validation_market_truth_by_model.csv"),
        rows=result.per_game,
    )
    out.add_csv(
        section="market_truth_calibration",
        file_name=_validation_path("market_truth", "validation_market_truth_calibration.csv"),
        rows=result.calibration,
    )
    return _task_result(ctx, out, summary=payload, applicability=applicability)


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
    applicability = "partial" if int(summary.get("n_pending_checks", 0)) > 0 else "applicable"
    return _task_result(ctx, out, summary=summary, applicability=applicability)


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
            name="market_truth",
            runner=_task_market_truth,
            enabled=lambda ctx: ctx.league == "MLB" and _has_glm(ctx) and _has_holdout(ctx),
            family="market_truth",
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
            normalized = _skipped_task_result(ctx, task)
        else:
            task_result = task.runner(ctx)
            if isinstance(task_result, ValidationOutputs):
                if task_result.sections:
                    normalized = ValidationTaskResult(outputs=task_result)
                else:
                    normalized = ValidationTaskResult(
                        outputs=task_result,
                        applicability="not_applicable",
                        summary={"status": "not_applicable", "note": "task_returned_no_artifacts"},
                    )
            else:
                normalized = task_result
        summary = _record_task_summary(
            ctx,
            task_name=task.name,
            task_result=normalized,
        )
        outputs.task_records.append(
            build_validation_artifact_record(
                task_name=task.name,
                family=task.family,
                applicability=normalized.applicability,
                artifacts=[spec.file_name for spec in normalized.outputs.sections],
                summary=summary,
            ).to_dict()
        )
        if normalized.outputs.sections:
            outputs.merge(normalized.outputs)

    outputs.write(
        ctx.out_dir,
        league=ctx.league,
        manifest_metadata=_validation_manifest_metadata(ctx),
    )
    _archive_validation_outputs(ctx, outputs)
    return outputs
