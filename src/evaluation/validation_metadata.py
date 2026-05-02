"""Execution-scope metadata and archive writing for validation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
from typing import Any

from src.common.config import AppConfig
from src.common.time import utc_now_iso
from src.common.utils import ensure_dir
from src.evaluation.validation_artifacts import ValidationOutputs
from src.evaluation.validation_context import ValidationContext
from src.evaluation.validation_governance import split_label_for_context, target_scope_for_context

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
        "split_label": split_label_for_context(ctx),
        "target_scope": target_scope_for_context(ctx),
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
        "split_label": split_label_for_context(ctx),
        "target_scope": target_scope_for_context(ctx),
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


resolve_execution_scope_labels = _resolve_execution_scope_labels
fixture_demo_evidence_labels = _fixture_demo_evidence_labels
resolve_execution_metadata = _resolve_execution_metadata
reset_validation_output_dirs = _reset_validation_output_dirs
validation_manifest_metadata = _validation_manifest_metadata
archive_validation_outputs = _archive_validation_outputs
