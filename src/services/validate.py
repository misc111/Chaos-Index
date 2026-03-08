"""Validation service for regenerating artifacts from an existing trained run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.common.config import AppConfig
from src.common.logging import get_logger
from src.evaluation.validation_pipeline import ValidationOutputs, ValidationTask, run_validation_pipeline
from src.services.train import load_features_dataframe, parse_models_arg
from src.storage.db import Database
from src.training.feature_selection import select_feature_columns

logger = get_logger(__name__)


def _normalize_base_model_run_id(model_run_id: str) -> str:
    token = str(model_run_id or "").strip()
    if not token:
        raise ValueError("model_run_id cannot be empty")
    return token.split("__", 1)[0]


def _resolve_latest_saved_run(
    db: Database,
    *,
    model_run_id: str | None,
) -> tuple[str, Path]:
    if model_run_id:
        base_model_run_id = _normalize_base_model_run_id(model_run_id)
        rows = db.query(
            """
            SELECT model_hash, artifact_path
            FROM model_runs
            WHERE model_hash = ?
               OR model_run_id = ?
               OR model_run_id LIKE ?
            ORDER BY created_at_utc DESC
            LIMIT 1
            """,
            (base_model_run_id, base_model_run_id, f"{base_model_run_id}__%"),
        )
        if not rows:
            raise FileNotFoundError(f"No saved model run found for '{model_run_id}'")
        resolved_id = str(rows[0].get("model_hash") or base_model_run_id)
        artifact_path = str(rows[0].get("artifact_path") or "").strip()
        if not artifact_path:
            raise FileNotFoundError(f"Saved model run '{resolved_id}' does not have an artifact path")
        return resolved_id, Path(artifact_path)

    rows = db.query(
        """
        SELECT model_hash, artifact_path
        FROM model_runs
        WHERE run_type = 'daily_train'
          AND model_name != 'ensemble'
        ORDER BY created_at_utc DESC
        LIMIT 1
        """
    )
    if not rows:
        raise FileNotFoundError("No saved daily_train model run found for validation")

    resolved_id = str(rows[0].get("model_hash") or "").strip()
    artifact_path = str(rows[0].get("artifact_path") or "").strip()
    if not resolved_id or not artifact_path:
        raise FileNotFoundError("Latest saved daily_train run is missing model metadata")
    return resolved_id, Path(artifact_path)


def _resolve_artifact_dir(path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value
    return (Path.cwd() / path_value).resolve()


def _load_run_payload(artifact_dir: Path) -> dict[str, Any]:
    payload_path = artifact_dir / "run_payload.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Saved run payload not found at {payload_path}")
    return json.loads(payload_path.read_text())


def _available_models_for_run(db: Database, *, base_model_run_id: str) -> list[str]:
    rows = db.query(
        """
        SELECT model_name
        FROM model_runs
        WHERE model_hash = ?
           OR model_run_id LIKE ?
        ORDER BY created_at_utc DESC
        """,
        (base_model_run_id, f"{base_model_run_id}__%"),
    )
    seen: set[str] = set()
    available: list[str] = []
    for row in rows:
        model_name = str(row.get("model_name") or "").strip()
        if not model_name or model_name == "ensemble" or model_name in seen:
            continue
        seen.add(model_name)
        available.append(model_name)
    return available


def _resolve_selected_models(
    *,
    models_arg: str | None,
    run_payload: dict[str, Any],
    available_models: Sequence[str],
) -> list[str]:
    requested_models = parse_models_arg(models_arg)
    payload_selected = [str(model) for model in run_payload.get("selected_models", []) if str(model).strip()]
    if not payload_selected:
        payload_selected = list(available_models)
    if not payload_selected:
        raise ValueError("Saved run payload does not declare any selectable models for validation")

    if requested_models is None:
        return payload_selected

    missing = [model for model in requested_models if model not in payload_selected]
    if missing:
        raise ValueError(
            f"Requested models {missing} are not available in the saved run. "
            f"Available={payload_selected}"
        )
    return requested_models


def _resolve_feature_columns(features_df, run_payload: dict[str, Any]) -> list[str]:
    payload_feature_columns = [str(col) for col in run_payload.get("feature_columns", []) if str(col).strip()]
    if payload_feature_columns:
        resolved = [col for col in payload_feature_columns if col in features_df.columns]
        if resolved:
            missing = sorted(set(payload_feature_columns) - set(resolved))
            if missing:
                logger.warning(
                    "Saved run payload referenced %d feature columns that are absent from the current features table",
                    len(missing),
                )
            return resolved

    return select_feature_columns(features_df)


def run_saved_validation(
    cfg: AppConfig,
    *,
    models_arg: str | None = None,
    model_run_id: str | None = None,
    tasks: Sequence[ValidationTask] | None = None,
) -> ValidationOutputs:
    db = Database(cfg.paths.db_path)
    db.init_schema()
    features_df = load_features_dataframe(cfg.paths.processed_dir)

    base_model_run_id, artifact_dir = _resolve_latest_saved_run(db, model_run_id=model_run_id)
    artifact_dir = _resolve_artifact_dir(artifact_dir)
    run_payload = _load_run_payload(artifact_dir)
    available_models = _available_models_for_run(db, base_model_run_id=base_model_run_id)
    selected_models = _resolve_selected_models(
        models_arg=models_arg,
        run_payload=run_payload,
        available_models=available_models,
    )

    validation_payload = dict(run_payload)
    validation_payload["selected_models"] = selected_models
    validation_payload["model_run_id"] = str(validation_payload.get("model_run_id") or base_model_run_id)
    validation_payload["model_dir"] = str(artifact_dir)

    feature_columns = _resolve_feature_columns(features_df, validation_payload)
    if not feature_columns:
        raise ValueError("No usable feature columns were available for validation")

    outputs = run_validation_pipeline(
        {
            "models": {},
            "train_df": features_df,
            "feature_columns": feature_columns,
            "run_payload": validation_payload,
        },
        cfg,
        tasks=tasks,
    )
    logger.info(
        "Validation complete from saved run | source_model_run_id=%s selected_models=%s sections=%d",
        validation_payload["model_run_id"],
        selected_models,
        len(outputs.sections),
    )
    return outputs
