"""Artifact writers for model binaries and forecast payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.training.progress import ProgressCallback, emit_progress


def _write_json_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True))


def _write_frame_artifact(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def _relativity_frame_from_metadata(metadata: Mapping[str, Any] | None) -> pd.DataFrame:
    payload = dict(metadata or {})
    relativity = payload.get("relativity_summary")
    if not isinstance(relativity, Mapping):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for direction, field_name in (
        ("positive", "top_positive_relativities"),
        ("negative", "top_negative_relativities"),
    ):
        values = relativity.get(field_name)
        if not isinstance(values, list):
            continue
        for rank, row in enumerate(values, start=1):
            if not isinstance(row, Mapping):
                continue
            rows.append(
                {
                    "direction": direction,
                    "rank": int(rank),
                    "feature": str(row.get("feature") or ""),
                    "coef_original": row.get("coef_original"),
                    "odds_ratio": row.get("odds_ratio"),
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["direction", "rank", "feature", "coef_original", "odds_ratio"])


def _save_model_sidecar_artifacts(model_dir: Path, *, model_name: str, model: object) -> dict[str, str]:
    written: dict[str, str] = {}

    coef_frame_fn = getattr(model, "coef_frame", None)
    if callable(coef_frame_fn):
        try:
            coef_frame = coef_frame_fn()
        except Exception:
            coef_frame = None
        if isinstance(coef_frame, pd.DataFrame) and not coef_frame.empty:
            path = model_dir / f"{model_name}_coefficients.csv"
            _write_frame_artifact(path, coef_frame)
            written["coefficients"] = path.name

    fit_metadata_fn = getattr(model, "fit_metadata_dict", None)
    if callable(fit_metadata_fn):
        try:
            fit_metadata = fit_metadata_fn()
        except TypeError:
            fit_metadata = fit_metadata_fn(top_n=8)
        except Exception:
            fit_metadata = None
        if isinstance(fit_metadata, Mapping) and fit_metadata:
            path = model_dir / f"{model_name}_fit_metadata.json"
            _write_json_artifact(path, fit_metadata)
            written["fit_metadata"] = path.name

    credibility_metadata = getattr(model, "credibility_metadata", None)
    if isinstance(credibility_metadata, Mapping) and credibility_metadata:
        path = model_dir / f"{model_name}_credibility_metadata.json"
        _write_json_artifact(path, credibility_metadata)
        written["credibility_metadata"] = path.name

        relativity_frame = _relativity_frame_from_metadata(credibility_metadata)
        relativity_path = model_dir / f"{model_name}_relativity.csv"
        _write_frame_artifact(relativity_path, relativity_frame)
        written["relativity"] = relativity_path.name

    return written


def save_model_artifacts(
    models: dict[str, object],
    model_dir: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, dict[str, str]]:
    artifact_files: dict[str, dict[str, str]] = {}
    for name, model in sorted(models.items()):
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
        artifact_files[name].update(_save_model_sidecar_artifacts(model_dir, model_name=name, model=model))
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
