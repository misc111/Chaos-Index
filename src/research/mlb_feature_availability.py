from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.common.config import AppConfig, load_config
from src.common.utils import ensure_dir
from src.features.leakage_checks import run_leakage_checks
from src.research.artifact_guardrails import require_mlb_tournament_path
from src.research.nested_tournament import DEFAULT_STRUCTURED_SPEC_PATH, _safe_json, _utc_stamp
from src.services.train import load_features_dataframe


@dataclass(frozen=True, slots=True)
class FeatureAvailabilityReportResult:
    run_id: str
    artifact_root: Path
    csv_path: Path
    json_path: Path


def _source_bucket(feature: str) -> str:
    token = feature.lower()
    if any(part in token for part in ("starter", "pitcher", "xfip", "siera")):
        return "starter_pitching"
    if "bullpen" in token:
        return "bullpen"
    if any(part in token for part in ("lineup", "batter", "wrc", "platoon", "chase", "barrel", "hard_hit")):
        return "lineup_offense"
    if any(part in token for part in ("park", "weather", "wind", "humidity", "temperature", "roof", "altitude")):
        return "run_environment"
    if any(part in token for part in ("market", "total", "vig", "bookmaker", "odds")):
        return "market_overlay"
    if any(part in token for part in ("travel", "rest", "timezone", "day_game", "division")):
        return "schedule_context"
    return "general"


def _load_structured_features(path: str | Path | None) -> list[dict[str, Any]]:
    spec_path = Path(path or DEFAULT_STRUCTURED_SPEC_PATH)
    if not spec_path.is_absolute():
        spec_path = Path.cwd() / spec_path
    payload = yaml.safe_load(spec_path.read_text()) or {}
    rows: list[dict[str, Any]] = []
    slates = payload.get("slates") if isinstance(payload.get("slates"), dict) else {}
    for slate_name, slate_payload in slates.items():
        if not isinstance(slate_payload, dict):
            continue
        for ordinal, feature in enumerate(slate_payload.get("feature_order") or [], start=1):
            rows.append(
                {
                    "slate_name": str(slate_name),
                    "feature": str(feature),
                    "feature_order": ordinal,
                    "structured_glm_spec_path": str(spec_path),
                }
            )
    return rows


def _safe_coverage(series: pd.Series | None, row_count: int) -> float:
    if series is None or row_count <= 0:
        return 0.0
    return float(series.notna().mean())


def write_mlb_feature_availability_report(
    cfg: AppConfig,
    *,
    run_id: str | None = None,
    structured_glm_spec_path: str | Path | None = DEFAULT_STRUCTURED_SPEC_PATH,
) -> FeatureAvailabilityReportResult:
    features_df = load_features_dataframe(cfg.paths.processed_dir)
    resolved_run_id = run_id or f"{_utc_stamp()}_feature_availability"
    artifact_root = require_mlb_tournament_path(
        ensure_dir(Path(cfg.paths.artifacts_dir) / "reports" / "mlb" / "tournament" / "feature_availability" / resolved_run_id),
        cfg.paths.artifacts_dir,
        purpose="MLB feature availability report",
    )
    row_count = int(len(features_df))
    rows: list[dict[str, Any]] = []
    for item in _load_structured_features(structured_glm_spec_path):
        feature = str(item["feature"])
        present = feature in features_df.columns
        series = features_df[feature] if present else None
        numeric = bool(present and pd.api.types.is_numeric_dtype(series))
        coverage = _safe_coverage(series, row_count)
        leakage_issues = run_leakage_checks(features_df, feature_columns=[feature]) if present else ["feature_missing"]
        rows.append(
            item
            | {
                "present": bool(present),
                "numeric": bool(numeric),
                "row_count": row_count,
                "non_null_rows": int(series.notna().sum()) if series is not None else 0,
                "coverage": coverage,
                "source_bucket": _source_bucket(feature),
                "pregame_safe": not leakage_issues,
                "leakage_issues": "; ".join(leakage_issues),
                "status": "available" if present and numeric and coverage > 0 else ("missing" if not present else "blocked"),
            }
        )
    report = pd.DataFrame(rows)
    csv_path = artifact_root / "feature_availability.csv"
    json_path = artifact_root / "feature_availability.json"
    report.to_csv(csv_path, index=False)
    summary = {
        "total_features": int(report["feature"].nunique()) if not report.empty else 0,
        "present_features": int(report.loc[report["present"], "feature"].nunique()) if not report.empty else 0,
        "missing_features": int(report.loc[~report["present"], "feature"].nunique()) if not report.empty else 0,
        "row_count": row_count,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    _safe_json(
        json_path,
        {
            "league": str(cfg.data.league).upper(),
            "run_id": resolved_run_id,
            "summary": summary,
            "features": report.replace({np.nan: None}).to_dict(orient="records"),
            "artifacts": {"csv_path": str(csv_path), "json_path": str(json_path)},
        },
    )
    return FeatureAvailabilityReportResult(
        run_id=resolved_run_id,
        artifact_root=artifact_root,
        csv_path=csv_path,
        json_path=json_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an MLB structured-slate feature availability report.")
    parser.add_argument("--config", default="configs/mlb.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--structured-glm-spec", default=DEFAULT_STRUCTURED_SPEC_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    result = write_mlb_feature_availability_report(
        cfg,
        run_id=args.run_id,
        structured_glm_spec_path=args.structured_glm_spec,
    )
    print(f"MLB_FEATURE_AVAILABILITY::{result.run_id}::{result.csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
