from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clean_string_list(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    if not values:
        return []
    return [str(value) for value in values if str(value).strip()]


def _clean_dict(values: dict[str, Any] | None) -> dict[str, Any]:
    return dict(values or {})


@dataclass(slots=True)
class PenaltySelection:
    model_name: str
    penalty_family: str
    lambda_value: float | None = None
    c_value: float | None = None
    l1_ratio: float | None = None
    grid_results: list[dict[str, Any]] = field(default_factory=list)
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    active_parameter_count: int | None = None
    active_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_features"] = _clean_string_list(self.active_features)
        payload["grid_results"] = [dict(row) for row in self.grid_results]
        payload["fold_metrics"] = [dict(row) for row in self.fold_metrics]
        return payload


@dataclass(slots=True)
class LassoCredibilityMetadata:
    model_name: str
    complement_kind: str
    complement_label: str
    complement_column: str
    offset_scale: str
    driver_features: list[str] = field(default_factory=list)
    complement_summary: dict[str, Any] = field(default_factory=dict)
    relativity_summary: dict[str, Any] = field(default_factory=dict)
    exposure_summary: dict[str, Any] = field(default_factory=dict)
    lambda_choice_note: str = ""
    complement_choice_note: str = ""
    p_values_reported: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["driver_features"] = _clean_string_list(self.driver_features)
        payload["complement_summary"] = _clean_dict(self.complement_summary)
        payload["relativity_summary"] = _clean_dict(self.relativity_summary)
        payload["exposure_summary"] = _clean_dict(self.exposure_summary)
        return payload


@dataclass(slots=True)
class ModelArtifactRecord:
    model_name: str
    model_family: str
    lane: str
    target_name: str = ""
    distribution: str = ""
    link_function: str = ""
    weight_column: str = ""
    offset_columns: list[str] = field(default_factory=list)
    artifact_files: dict[str, str] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    fit_summary: dict[str, Any] = field(default_factory=dict)
    penalty: PenaltySelection | None = None
    credibility: LassoCredibilityMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "lane": self.lane,
            "target_name": self.target_name,
            "distribution": self.distribution,
            "link_function": self.link_function,
            "weight_column": self.weight_column,
            "offset_columns": _clean_string_list(self.offset_columns),
            "artifact_files": _clean_dict(self.artifact_files),
            "feature_columns": _clean_string_list(self.feature_columns),
            "metrics_summary": _clean_dict(self.metrics_summary),
            "fit_summary": _clean_dict(self.fit_summary),
            "penalty": self.penalty.to_dict() if self.penalty is not None else None,
            "credibility": self.credibility.to_dict() if self.credibility is not None else None,
        }
        return payload


@dataclass(slots=True)
class ValidationArtifactRecord:
    task_name: str
    family: str
    applicability: str
    artifacts: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "family": self.family,
            "applicability": self.applicability,
            "artifacts": _clean_string_list(self.artifacts),
            "summary": _clean_dict(self.summary),
        }


@dataclass(slots=True)
class CandidateScorecardRecord:
    model_name: str
    lane: str
    target_name: str
    distribution: str
    link_function: str
    feature_count: int | None = None
    active_parameter_count: int | None = None
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    stability_metrics: dict[str, Any] = field(default_factory=dict)
    calibration_summary: dict[str, Any] = field(default_factory=dict)
    complement_summary: dict[str, Any] = field(default_factory=dict)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "lane": self.lane,
            "target_name": self.target_name,
            "distribution": self.distribution,
            "link_function": self.link_function,
            "feature_count": int(self.feature_count) if self.feature_count is not None else None,
            "active_parameter_count": self.active_parameter_count,
            "validation_metrics": _clean_dict(self.validation_metrics),
            "stability_metrics": _clean_dict(self.stability_metrics),
            "calibration_summary": _clean_dict(self.calibration_summary),
            "complement_summary": _clean_dict(self.complement_summary),
            "rejection_reasons": _clean_string_list(self.rejection_reasons),
        }


@dataclass(slots=True)
class PromotionDecisionRecord:
    recommended_model: str
    baseline_model: str
    status: str
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    rejected_models: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_model": self.recommended_model,
            "baseline_model": self.baseline_model,
            "status": self.status,
            "rationale": self.rationale,
            "evidence": _clean_dict(self.evidence),
            "rejected_models": {
                str(model_name): _clean_string_list(reasons) for model_name, reasons in (self.rejected_models or {}).items()
            },
        }


@dataclass(slots=True)
class CandidateComparisonContract:
    report_slug: str
    league: str
    target_name: str
    distribution: str
    link_function: str
    candidate_models: list[str] = field(default_factory=list)
    candidate_scorecards: list[CandidateScorecardRecord] = field(default_factory=list)
    promotion_decision: PromotionDecisionRecord | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_slug": self.report_slug,
            "league": self.league,
            "target_name": self.target_name,
            "distribution": self.distribution,
            "link_function": self.link_function,
            "candidate_models": _clean_string_list(self.candidate_models),
            "candidate_scorecards": [record.to_dict() for record in self.candidate_scorecards],
            "promotion_decision": self.promotion_decision.to_dict() if self.promotion_decision is not None else None,
            "artifacts": _clean_dict(self.artifacts),
            "metadata": _clean_dict(self.metadata),
        }


@dataclass(slots=True)
class ModelRunContract:
    model_run_id: str
    league: str
    feature_set_version: str
    selected_models: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    model_feature_columns: dict[str, list[str]] = field(default_factory=dict)
    model_artifacts: list[ModelArtifactRecord] = field(default_factory=list)
    validation_outputs: list[ValidationArtifactRecord] = field(default_factory=list)
    candidate_scorecards: list[CandidateScorecardRecord] = field(default_factory=list)
    promotion_decision: PromotionDecisionRecord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_run_id": self.model_run_id,
            "league": self.league,
            "feature_set_version": self.feature_set_version,
            "selected_models": _clean_string_list(self.selected_models),
            "feature_columns": _clean_string_list(self.feature_columns),
            "model_feature_columns": {
                str(model_name): _clean_string_list(columns)
                for model_name, columns in (self.model_feature_columns or {}).items()
            },
            "model_artifacts": [record.to_dict() for record in self.model_artifacts],
            "validation_outputs": [record.to_dict() for record in self.validation_outputs],
            "candidate_scorecards": [record.to_dict() for record in self.candidate_scorecards],
            "promotion_decision": self.promotion_decision.to_dict() if self.promotion_decision is not None else None,
            "metadata": _clean_dict(self.metadata),
        }
