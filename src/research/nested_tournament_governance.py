"""Validation gates and family-champion selection for MLB tournaments.

These helpers encode the promotion-facing diagnostics that separate shortlisted
family champions from validation-blocked candidates.  They operate only on
metric rows and keep governance labeling out of model fitting code.
"""

from __future__ import annotations

from typing import Any
import json

import numpy as np
import pandas as pd

from src.research.nested_tournament_contracts import _safe_float

def _gate_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if str(row.get("fit_status") or "") != "ok":
        reasons.append("fit_failed")
    if not bool(row.get("has_class_contrast")):
        reasons.append("no_class_contrast")
    ece = _safe_float(row.get("ece"))
    if ece is None or ece > 0.04:
        reasons.append("ece_gt_0.04")
    auc = _safe_float(row.get("auc"))
    if auc is None or auc < 0.50:
        reasons.append("auc_lt_0.50")
    alpha = _safe_float(row.get("calibration_alpha"))
    beta = _safe_float(row.get("calibration_beta"))
    if alpha is None or beta is None or abs(alpha) > 0.15 or abs(beta - 1.0) > 0.15:
        reasons.append("calibration_alpha_beta_outside_0.15")
    lift_ratio = _safe_float(row.get("top_vs_bottom_actual_lift_ratio"))
    if lift_ratio is None or lift_ratio <= 1.0:
        reasons.append("non_positive_top_bottom_lift")
    drift = _safe_float(row.get("validation_to_cv_log_loss_delta"))
    if drift is None or abs(drift) > 0.05:
        reasons.append("cv_validation_drift_gt_0.05")
    return reasons


GATE_REASON_LABELS = {
    "fit_failed": "fit did not complete",
    "no_class_contrast": "validation split has no class contrast",
    "ece_gt_0.04": "calibration error above 0.04",
    "auc_lt_0.50": "AUROC below 0.50",
    "calibration_alpha_beta_outside_0.15": "calibration intercept/slope outside tolerance",
    "non_positive_top_bottom_lift": "top-vs-bottom lift is not positive",
    "cv_validation_drift_gt_0.05": "validation log loss drifted from CV by more than 0.05",
}


def _gate_reason_labels(reasons: list[str]) -> list[str]:
    return [GATE_REASON_LABELS.get(reason, reason.replace("_", " ")) for reason in reasons]


def _blocked_reason_summary(reasons: list[str]) -> str:
    labels = _gate_reason_labels(reasons)
    return "; ".join(labels) if labels else ""


def _select_family_champions(validation_rows: pd.DataFrame) -> pd.DataFrame:
    if validation_rows.empty:
        return pd.DataFrame()
    valid = validation_rows.copy()
    valid["gate_reasons"] = valid.apply(_gate_reasons, axis=1)
    valid["gate_passed"] = valid["gate_reasons"].map(lambda reasons: len(reasons) == 0)
    champions: list[dict[str, Any]] = []
    for (target_name, model_name), bucket in valid.groupby(["target_name", "model_name"], sort=True):
        passed = bucket[bucket["gate_passed"]].copy()
        fitted = bucket[bucket["fit_status"] == "ok"].copy()
        source = passed if not passed.empty else (fitted if not fitted.empty else bucket)
        ordered = source.sort_values(
            ["log_loss", "brier", "auc", "n_features", "active_parameter_count", "candidate_key"],
            ascending=[True, True, False, True, True, True],
            na_position="last",
        )
        row = ordered.iloc[0].to_dict()
        row["champion_status"] = "shortlist" if bool(row.get("gate_passed")) else "validation-blocked"
        row["family_champion"] = True
        reasons = _reason_list(row.get("gate_reasons"))
        row["gate_reason_labels"] = _gate_reason_labels(reasons)
        row["blocked_reason_summary"] = _blocked_reason_summary(reasons)
        champions.append(row)
    return pd.DataFrame(champions)


def _reason_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    try:
        parsed = json.loads(text.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [token.strip() for token in text.strip("[]").split(",") if token.strip()]
