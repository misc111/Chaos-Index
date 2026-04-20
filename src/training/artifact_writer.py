"""Artifact writers for model binaries and forecast payloads."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.training.progress import ProgressCallback, emit_progress


def save_model_artifacts(
    models: dict[str, object],
    model_dir: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, dict[str, str]]:
    artifact_files: dict[str, dict[str, str]] = {}
    for name, model in models.items():
        if not hasattr(model, "save"):
            continue
        emit_progress(
            progress_callback,
            {"kind": "model", "model": name, "stage": "save", "status": "started", "message": f"Saving {name} artifact"},
        )
        ext = "json" if name == "bayes_bt_state_space" else "joblib"
        artifact_path = model_dir / f"{name}.{ext}"
        model.save(artifact_path)
        artifact_files[name] = {"binary": artifact_path.name}
        emit_progress(
            progress_callback,
            {
                "kind": "model",
                "model": name,
                "stage": "save",
                "status": "completed",
                "message": f"Saved {name} artifact",
            },
        )
    return artifact_files


def save_training_outputs(
    model_dir: Path,
    forecasts: pd.DataFrame,
    upcoming_preds: pd.DataFrame,
    oof: pd.DataFrame,
    run_payload: dict,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "save_artifacts", "status": "started", "message": "Saving training artifacts"},
    )
    written: dict[str, str] = {}
    try:
        forecasts_path = model_dir / "upcoming_forecasts.parquet"
        forecasts.to_parquet(forecasts_path, index=False)
    except Exception:
        forecasts_path = model_dir / "upcoming_forecasts.csv"
        forecasts.to_csv(forecasts_path, index=False)
    written["upcoming_forecasts"] = forecasts_path.name
    try:
        upcoming_preds_path = model_dir / "upcoming_model_probs.parquet"
        upcoming_preds.to_parquet(upcoming_preds_path, index=False)
    except Exception:
        upcoming_preds_path = model_dir / "upcoming_model_probs.csv"
        upcoming_preds.to_csv(upcoming_preds_path, index=False)
    written["upcoming_model_probs"] = upcoming_preds_path.name
    if not oof.empty:
        try:
            oof_path = model_dir / "oof_predictions.parquet"
            oof.to_parquet(oof_path, index=False)
        except Exception:
            oof_path = model_dir / "oof_predictions.csv"
            oof.to_csv(oof_path, index=False)
        written["oof_predictions"] = oof_path.name
    payload_path = model_dir / "run_payload.json"
    payload_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True))
    written["run_payload"] = payload_path.name
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "save_artifacts", "status": "completed", "message": "Saved training artifacts"},
    )
    return written
