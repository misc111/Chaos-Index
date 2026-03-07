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
) -> None:
    for name, model in models.items():
        if not hasattr(model, "save"):
            continue
        emit_progress(
            progress_callback,
            {"kind": "model", "model": name, "stage": "save", "status": "started", "message": f"Saving {name} artifact"},
        )
        ext = "json" if name == "bayes_bt_state_space" else "joblib"
        model.save(model_dir / f"{name}.{ext}")
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


def save_training_outputs(
    model_dir: Path,
    forecasts: pd.DataFrame,
    upcoming_preds: pd.DataFrame,
    oof: pd.DataFrame,
    run_payload: dict,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "save_artifacts", "status": "started", "message": "Saving training artifacts"},
    )
    try:
        forecasts.to_parquet(model_dir / "upcoming_forecasts.parquet", index=False)
    except Exception:
        forecasts.to_csv(model_dir / "upcoming_forecasts.csv", index=False)
    try:
        upcoming_preds.to_parquet(model_dir / "upcoming_model_probs.parquet", index=False)
    except Exception:
        upcoming_preds.to_csv(model_dir / "upcoming_model_probs.csv", index=False)
    if not oof.empty:
        try:
            oof.to_parquet(model_dir / "oof_predictions.parquet", index=False)
        except Exception:
            oof.to_csv(model_dir / "oof_predictions.csv", index=False)
    (model_dir / "run_payload.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True))
    emit_progress(
        progress_callback,
        {"kind": "pipeline", "stage": "save_artifacts", "status": "completed", "message": "Saved training artifacts"},
    )
