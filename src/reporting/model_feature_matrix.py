from __future__ import annotations

from dataclasses import dataclass, field
import ast
import json
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import AppConfig
from src.common.time import utc_now_iso
from src.features.leakage_checks import run_leakage_checks
from src.research.candidate_models import (
    DGLMMarginCandidate,
    GAMSplineCandidate,
    GLMMLogitCandidate,
    MARSHingeCandidate,
    PenalizedLogitCandidate,
    VanillaGLMBinomialCandidate,
)
from src.research.model_comparison import _candidate_specs, _select_feature_sets
from src.services.train import load_features_dataframe
from src.training.model_feature_research import load_model_feature_map
from src.training.train import select_feature_columns, train_and_predict

_DEFAULT_REPORT_PATH = Path("docs/mlb/model_feature_beta_matrix.md")
_CURRENT_BEST_MANIFEST = Path("artifacts/reports/mlb/candidate_model_comparison_latest_manifest.json")
_COEFFICIENT_ROUND_DIGITS = 4


@dataclass(slots=True)
class CoefficientRow:
    term: str
    feature: str
    role: str
    beta: float | None
    beta_scale: str
    note: str = ""


@dataclass(slots=True)
class ModelFeatureReport:
    scope: str
    model_name: str
    display_name: str
    lane: str
    selected_features: dict[str, list[str]] = field(default_factory=dict)
    coefficient_rows: list[CoefficientRow] = field(default_factory=list)
    notes: str = ""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _round_beta(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{_COEFFICIENT_ROUND_DIGITS}f}"


def _markdown_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not headers:
        return ""
    escaped_rows = [[_markdown_escape(cell) for cell in row] for row in rows]
    escaped_headers = [_markdown_escape(header) for header in headers]
    widths = [len(header) for header in escaped_headers]
    for row in escaped_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(escaped_headers)) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    body = [
        "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"
        for row in escaped_rows
    ]
    return "\n".join([header_line, divider, *body])


def _parse_param_string(text: str) -> dict[str, Any]:
    token = str(text or "").strip()
    if not token or token == "{}":
        return {}
    out: dict[str, Any] = {}
    for part in token.split(";"):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value_token = raw_value.strip()
        lowered = value_token.lower()
        if lowered in {"none", "null"}:
            value: Any = None
        elif lowered == "true":
            value = True
        elif lowered == "false":
            value = False
        else:
            try:
                if any(ch in value_token for ch in (".", "e", "E")):
                    value = float(value_token)
                else:
                    value = int(value_token)
            except ValueError:
                value = value_token
        out[key] = value
    return out


def _candidate_report_split_fractions(summary_markdown: str) -> tuple[float, float, float]:
    match = re.search(r"Outer split `outer_(\d+)_(\d+)_(\d+)`", summary_markdown)
    if not match:
        return 0.4, 0.3, 0.3
    numerators = [int(match.group(idx)) for idx in range(1, 4)]
    total = sum(numerators)
    if total <= 0:
        return 0.4, 0.3, 0.3
    return tuple(float(value) / float(total) for value in numerators)  # type: ignore[return-value]


def _ordered_feature_union(reports: list[ModelFeatureReport]) -> list[str]:
    ordered: list[str] = []
    for report in reports:
        for feature in report.selected_features:
            if feature not in ordered:
                ordered.append(feature)
    return ordered


def _linear_cell_beta(report: ModelFeatureReport, feature: str) -> float | None:
    matching = [
        row.beta
        for row in report.coefficient_rows
        if row.feature == feature and row.role in {"linear", "fixed_effect", "driver"}
    ]
    finite = [float(value) for value in matching if value is not None and np.isfinite(value)]
    if len(finite) == 1:
        return finite[0]
    return None


def _selection_cell(report: ModelFeatureReport, feature: str) -> str:
    roles = report.selected_features.get(feature, [])
    if not roles:
        return ""
    beta = _linear_cell_beta(report, feature)
    if beta is not None:
        return _round_beta(beta)
    ordered_roles = list(dict.fromkeys(roles))
    if ordered_roles == ["linear"]:
        return "selected"
    short_map = {
        "linear": "linear",
        "driver": "driver",
        "spline": "spline",
        "hinge": "hinge",
        "fixed_effect": "fixed",
        "mean": "mean",
        "dispersion": "disp",
    }
    return ",".join(short_map.get(role, role) for role in ordered_roles)


def _production_model_feature_map(cfg: AppConfig) -> dict[str, list[str]]:
    return load_model_feature_map(cfg.data.league)


def _strip_forbidden_leakage_columns(features_df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    current = list(feature_columns)
    for _ in range(3):
        issues = run_leakage_checks(features_df, feature_columns=current)
        if not issues:
            return current
        removed_any = False
        for issue in issues:
            match = re.search(r"\[(.*)\]", str(issue))
            if not match:
                continue
            try:
                parsed = ast.literal_eval("[" + match.group(1) + "]")
            except Exception:
                continue
            forbidden = {str(value) for value in parsed if str(value).strip()}
            if not forbidden:
                continue
            next_current = [feature for feature in current if feature not in forbidden]
            removed_any = removed_any or len(next_current) != len(current)
            current = next_current
        if not removed_any:
            break
    return current


def _required_complement_column(model_name: str) -> str | None:
    if model_name == "glm_lasso_market_credibility":
        return "market_offset_logit"
    if model_name == "glm_lasso_prior_credibility":
        return "prior_model_offset_logit"
    return None


def _production_reports(
    cfg: AppConfig,
    features_df: pd.DataFrame,
    *,
    selected_models: list[str] | None = None,
) -> list[ModelFeatureReport]:
    feature_map = _production_model_feature_map(cfg)
    if selected_models is None:
        ordered_models = list(feature_map)
    else:
        ordered_models = [model for model in selected_models if model in feature_map]
    if not ordered_models:
        return []

    available_columns = set(features_df.columns)
    filtered_feature_map: dict[str, list[str]] = {}
    missing_by_model: dict[str, list[str]] = {}
    skip_reason_by_model: dict[str, str] = {}
    for model_name in ordered_models:
        requested = list(feature_map.get(model_name, []))
        filtered_feature_map[model_name] = [feature for feature in requested if feature in available_columns]
        missing_by_model[model_name] = [feature for feature in requested if feature not in available_columns]
        complement_column = _required_complement_column(model_name)
        if complement_column and complement_column not in available_columns:
            skip_reason_by_model[model_name] = (
                f"Skipped production fit because required complement column `{complement_column}` is missing from the current processed dataset."
            )
        elif not filtered_feature_map.get(model_name):
            skip_reason_by_model[model_name] = (
                "Skipped production fit because none of the committed mapped features are present in the current processed dataset."
            )
    skipped_models = [model_name for model_name in ordered_models if model_name in skip_reason_by_model]
    trainable_models = [model_name for model_name in ordered_models if model_name not in skip_reason_by_model]

    reports: list[ModelFeatureReport] = []
    for model_name in skipped_models:
        reports.append(
            ModelFeatureReport(
                scope="production",
                model_name=model_name,
                display_name=model_name,
                lane="core",
                selected_features={},
                coefficient_rows=[],
                notes=(
                    skip_reason_by_model.get(model_name, "")
                    + " "
                    + (
                        f"Missing mapped features: {', '.join(missing_by_model.get(model_name, []))}."
                        if missing_by_model.get(model_name)
                        else ""
                    )
                ),
            )
        )
    if not trainable_models:
        return reports
    selected_feature_columns = [
        feature
        for model_name in trainable_models
        for feature in filtered_feature_map.get(model_name, [])
        if str(feature).strip()
    ]
    selected_feature_columns = list(dict.fromkeys(selected_feature_columns))

    with tempfile.TemporaryDirectory(prefix="model_feature_matrix_") as temp_dir:
        result = train_and_predict(
            features_df=features_df,
            feature_set_version="model_feature_matrix_snapshot",
            artifacts_dir=temp_dir,
            bayes_cfg={},
            selected_models=trainable_models,
            selected_feature_columns=selected_feature_columns,
            selected_model_feature_columns=filtered_feature_map,
            league=cfg.data.league,
        )

    model_artifacts = {
        str(row.get("model_name")): row for row in result["run_payload"].get("model_artifacts", []) if isinstance(row, dict)
    }
    for model_name in trainable_models:
        model = result["models"].get(model_name)
        if model is None:
            continue
        artifact = model_artifacts.get(model_name, {})
        display_name = str(artifact.get("display_label") or artifact.get("model_name") or model_name)
        lane = str(artifact.get("lane") or "core")
        selected_feature_columns = list(result["run_payload"].get("model_feature_columns", {}).get(model_name, []))
        coef_frame_fn = getattr(model, "coef_frame", None)
        coef_frame = coef_frame_fn() if callable(coef_frame_fn) else pd.DataFrame()
        coefficient_rows: list[CoefficientRow] = []
        if isinstance(coef_frame, pd.DataFrame) and not coef_frame.empty:
            beta_column = "coef_original" if "coef_original" in coef_frame.columns else ("coef" if "coef" in coef_frame.columns else "")
            for row in coef_frame.itertuples(index=False):
                beta = getattr(row, beta_column) if beta_column else None
                coefficient_rows.append(
                    CoefficientRow(
                        term=str(getattr(row, "feature")),
                        feature=str(getattr(row, "feature")),
                        role="driver",
                        beta=None if beta is None or not np.isfinite(beta) else float(beta),
                        beta_scale="original_feature",
                    )
                )
        reports.append(
            ModelFeatureReport(
                scope="production",
                model_name=model_name,
                display_name=display_name,
                lane=lane,
                selected_features={feature: ["driver"] for feature in selected_feature_columns},
                coefficient_rows=coefficient_rows,
                notes=(
                    f"Production fit on current processed MLB features using the committed model feature map ({len(selected_feature_columns)} available features)."
                    + (
                        f" Missing mapped features in the current dataset: {', '.join(missing_by_model.get(model_name, []))}."
                        if missing_by_model.get(model_name)
                        else ""
                    )
                ),
            )
        )
    return reports


def _comparison_context(cfg: AppConfig) -> tuple[dict[str, Any], pd.DataFrame, tuple[float, float, float], pd.DataFrame]:
    manifest_path = Path(cfg.paths.artifacts_dir) / "reports" / "mlb" / _CURRENT_BEST_MANIFEST.name
    manifest = _load_json(manifest_path)
    summary_path = Path(manifest["material_artifacts"]["summary_path"])
    summary_payload = _load_json(summary_path)
    fit_stats_path = Path(summary_payload["artifacts"]["fit_stats_path"])
    fit_stats = pd.read_csv(fit_stats_path)
    report_path = Path(summary_payload["artifacts"]["report_path"])
    split_fractions = _candidate_report_split_fractions(report_path.read_text())
    features_df = load_features_dataframe(cfg.paths.processed_dir)
    return summary_payload, fit_stats, split_fractions, features_df


def _candidate_basis_beta(coef_scaled: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.divide(
        coef_scaled,
        scale,
        out=np.full_like(coef_scaled, np.nan, dtype=float),
        where=np.isfinite(scale) & (np.abs(scale) > 0),
    )


def _candidate_penalized_rows(model: PenalizedLogitCandidate) -> tuple[dict[str, list[str]], list[CoefficientRow]]:
    coef_scaled = np.asarray(model.model.coef_[0], dtype=float)
    scale = np.asarray(model.scaler.scale_, dtype=float)
    coef_original = _candidate_basis_beta(coef_scaled, scale)
    selection = {feature: ["linear"] for feature in model.features}
    rows = [
        CoefficientRow(
            term=feature,
            feature=feature,
            role="linear",
            beta=None if not np.isfinite(coef_original[idx]) else float(coef_original[idx]),
            beta_scale="original_feature",
        )
        for idx, feature in enumerate(model.features)
    ]
    return selection, rows


def _candidate_vanilla_rows(model: VanillaGLMBinomialCandidate) -> tuple[dict[str, list[str]], list[CoefficientRow]]:
    params = pd.Series(model.result.params, index=model.exog_names, dtype=float)
    scale = pd.Series(np.asarray(model.scaler.scale_, dtype=float), index=model.features)
    coef_original = params.drop(labels=["const"], errors="ignore").reindex(model.features).divide(scale)
    selection = {feature: ["linear"] for feature in model.features}
    rows = [
        CoefficientRow(
            term=feature,
            feature=feature,
            role="linear",
            beta=None if not np.isfinite(coef_original.get(feature, np.nan)) else float(coef_original[feature]),
            beta_scale="original_feature",
        )
        for feature in model.features
    ]
    return selection, rows


def _candidate_gam_rows(model: GAMSplineCandidate) -> tuple[dict[str, list[str]], list[CoefficientRow]]:
    coef_scaled = np.asarray(model.model.coef_[0], dtype=float)
    scale = np.asarray(model.basis_scaler.scale_, dtype=float)
    coef_basis = _candidate_basis_beta(coef_scaled, scale)
    selection: dict[str, list[str]] = {feature: ["linear"] for feature in model.linear_features}
    for feature in model.spline_features:
        selection.setdefault(feature, []).append("spline")
    terms: list[CoefficientRow] = []
    idx = 0
    for feature in model.linear_features:
        terms.append(
            CoefficientRow(
                term=feature,
                feature=feature,
                role="linear",
                beta=None if not np.isfinite(coef_basis[idx]) else float(coef_basis[idx]),
                beta_scale="original_feature",
            )
        )
        idx += 1
    for feature in model.spline_features:
        transformer = model.linear_transformers[feature]
        n_out = int(getattr(transformer, "n_features_out_", transformer.transform(np.zeros((1, 1), dtype=float)).shape[1]))
        for basis_idx in range(n_out):
            beta = coef_basis[idx]
            terms.append(
                CoefficientRow(
                    term=f"{feature}::spline_{basis_idx + 1:02d}",
                    feature=feature,
                    role="spline_basis",
                    beta=None if not np.isfinite(beta) else float(beta),
                    beta_scale="basis_term",
                )
            )
            idx += 1
    return selection, terms


def _candidate_mars_rows(model: MARSHingeCandidate) -> tuple[dict[str, list[str]], list[CoefficientRow]]:
    coef_scaled = np.asarray(model.model.coef_[0], dtype=float)
    scale = np.asarray(model.scaler.scale_, dtype=float)
    coef_basis = _candidate_basis_beta(coef_scaled, scale)
    selection: dict[str, list[str]] = {feature: ["linear"] for feature in model.linear_features}
    for feature in model.hinge_features:
        selection.setdefault(feature, []).append("hinge")
    rows: list[CoefficientRow] = []
    for idx, term in enumerate(model.basis_names):
        feature = term
        role = "hinge_basis"
        if term in model.linear_features:
            feature = term
            role = "linear"
        elif "_hinge_" in term:
            feature = term.split("_hinge_", 1)[0]
        elif "_x_" in term:
            feature = term.split("_x_", 1)[0]
        beta = coef_basis[idx]
        rows.append(
            CoefficientRow(
                term=str(term),
                feature=str(feature),
                role=role,
                beta=None if not np.isfinite(beta) else float(beta),
                beta_scale="basis_term" if role != "linear" else "original_feature",
            )
        )
    return selection, rows


def _candidate_glmm_rows(model: GLMMLogitCandidate) -> tuple[dict[str, list[str]], list[CoefficientRow]]:
    fixed = np.asarray(model.fixed_effect_mean, dtype=float)
    fixed_feature_scaled = fixed[1:]
    scale = np.asarray(model.scaler.scale_, dtype=float)
    coef_original = _candidate_basis_beta(fixed_feature_scaled, scale)
    selection = {feature: ["fixed_effect"] for feature in model.fixed_features}
    rows = [
        CoefficientRow(
            term=feature,
            feature=feature,
            role="fixed_effect",
            beta=None if not np.isfinite(coef_original[idx]) else float(coef_original[idx]),
            beta_scale="original_feature",
        )
        for idx, feature in enumerate(model.fixed_features)
    ]
    return selection, rows


def _candidate_dglm_rows(model: DGLMMarginCandidate) -> tuple[dict[str, list[str]], list[CoefficientRow]]:
    mean_params = pd.Series(np.asarray(model.mean_result.params, dtype=float), index=model.exog_names)
    disp_params = pd.Series(np.asarray(model.dispersion_result.params, dtype=float), index=model.exog_names)
    scale = pd.Series(np.asarray(model.scaler.scale_, dtype=float), index=model.features)
    mean_original = mean_params.drop(labels=["const"], errors="ignore").reindex(model.features).divide(scale)
    disp_original = disp_params.drop(labels=["const"], errors="ignore").reindex(model.features).divide(scale)
    selection = {feature: ["mean", "dispersion"] for feature in model.features}
    rows: list[CoefficientRow] = []
    for feature in model.features:
        mean_beta = mean_original.get(feature, np.nan)
        disp_beta = disp_original.get(feature, np.nan)
        rows.append(
            CoefficientRow(
                term=f"mean::{feature}",
                feature=feature,
                role="mean",
                beta=None if not np.isfinite(mean_beta) else float(mean_beta),
                beta_scale="original_feature",
            )
        )
        rows.append(
            CoefficientRow(
                term=f"dispersion::{feature}",
                feature=feature,
                role="dispersion",
                beta=None if not np.isfinite(disp_beta) else float(disp_beta),
                beta_scale="original_feature",
            )
        )
    return selection, rows


def _candidate_report_rows(model: object) -> tuple[dict[str, list[str]], list[CoefficientRow], str]:
    if isinstance(model, PenalizedLogitCandidate):
        selection, rows = _candidate_penalized_rows(model)
        return selection, rows, "Comparison fit with original-scale linear coefficients."
    if isinstance(model, VanillaGLMBinomialCandidate):
        selection, rows = _candidate_vanilla_rows(model)
        return selection, rows, "Comparison fit with original-scale GLM coefficients."
    if isinstance(model, GAMSplineCandidate):
        selection, rows = _candidate_gam_rows(model)
        return selection, rows, "Spline rows are basis-term coefficients, not a single beta on the raw predictor scale."
    if isinstance(model, MARSHingeCandidate):
        selection, rows = _candidate_mars_rows(model)
        return selection, rows, "Hinge rows are basis-term coefficients on the MARS proxy basis."
    if isinstance(model, GLMMLogitCandidate):
        selection, rows = _candidate_glmm_rows(model)
        return selection, rows, "Fixed-effect coefficients only; team random-effect terms are omitted from the matrix."
    if isinstance(model, DGLMMarginCandidate):
        selection, rows = _candidate_dglm_rows(model)
        return selection, rows, "Mean and dispersion coefficients are both shown for each selected feature."
    raise TypeError(f"Unsupported candidate model type: {type(model)!r}")


def _candidate_reports(
    cfg: AppConfig,
    *,
    selected_models: list[str] | None = None,
) -> list[ModelFeatureReport]:
    summary_payload, fit_stats, split_fractions, features_df = _comparison_context(cfg)
    raw_features = _strip_forbidden_leakage_columns(features_df, select_feature_columns(features_df))
    leakage_issues = run_leakage_checks(features_df, feature_columns=raw_features)
    if leakage_issues:
        raise RuntimeError(f"Leakage checks failed before candidate matrix generation: {leakage_issues}")

    historical_df = features_df[features_df["home_win"].notna()].copy().sort_values("start_time_utc").reset_index(drop=True)
    train_fraction, validation_fraction, _ = split_fractions
    train_end = max(1, int(len(historical_df) * train_fraction))
    validation_end = max(train_end + 1, int(len(historical_df) * (train_fraction + validation_fraction)))
    fit_df = historical_df.iloc[:validation_end].copy()
    feature_sets = _select_feature_sets(fit_df, raw_features)

    test_rows = fit_stats[fit_stats["split"] == "test"].copy()
    test_rows = test_rows[test_rows["model_name"] != "intercept_only"].reset_index(drop=True)
    if selected_models is not None:
        allowed = set(selected_models)
        test_rows = test_rows[test_rows["model_name"].isin(allowed)].reset_index(drop=True)
    if test_rows.empty:
        return []

    specs = {spec.model_name: spec for spec in _candidate_specs(feature_sets, selected_models=set(test_rows["model_name"]))}
    scorecards = {
        str(row.get("model_name")): row
        for row in summary_payload.get("candidate_scorecards", [])
        if isinstance(row, dict)
    }

    reports: list[ModelFeatureReport] = []
    for row in test_rows.itertuples(index=False):
        spec = specs.get(str(row.model_name))
        if spec is None:
            continue
        params = _parse_param_string(str(row.params))
        model = spec.builder(feature_sets, params)
        model.fit(fit_df)
        selected, coefficient_rows, note = _candidate_report_rows(model)
        scorecard = scorecards.get(str(row.model_name), {})
        reports.append(
            ModelFeatureReport(
                scope="candidate_comparison",
                model_name=str(row.model_name),
                display_name=str(row.display_name),
                lane=str(scorecard.get("lane") or "candidate"),
                selected_features=selected,
                coefficient_rows=coefficient_rows,
                notes=(
                    f"{note} Current comparison params: {row.params}. "
                    f"Fit on the latest comparison fit window before the final holdout."
                ),
            )
        )
    return reports


def _section_summary_table(reports: list[ModelFeatureReport]) -> str:
    rows = []
    for report in reports:
        rows.append(
            [
                report.display_name,
                report.model_name,
                report.lane,
                len(report.selected_features),
                len(report.coefficient_rows),
                report.notes,
            ]
        )
    return _markdown_table(
        ["Display Name", "Model", "Lane", "Selected Features", "Coefficient Rows", "Notes"],
        rows,
    )


def _selection_matrix_table(reports: list[ModelFeatureReport]) -> str:
    features = _ordered_feature_union(reports)
    headers = ["Feature", *[report.model_name for report in reports]]
    rows: list[list[object]] = []
    for feature in features:
        rows.append([feature, *[_selection_cell(report, feature) for report in reports]])
    return _markdown_table(headers, rows)


def _coefficient_detail_markdown(reports: list[ModelFeatureReport]) -> str:
    lines: list[str] = []
    for report in reports:
        lines.append(f"### {report.display_name} (`{report.model_name}`)")
        lines.append("")
        if report.notes:
            lines.append(report.notes)
            lines.append("")
        if not report.coefficient_rows:
            lines.append("No coefficient rows are available for this model snapshot.")
            lines.append("")
            continue
        detail_rows = [
            [row.term, row.feature, row.role, _round_beta(row.beta), row.beta_scale, row.note]
            for row in report.coefficient_rows
        ]
        lines.append(
            _markdown_table(
                ["Term", "Base Feature", "Role", "Beta", "Scale", "Note"],
                detail_rows,
            )
        )
        lines.append("")
    return "\n".join(lines).strip()


def build_model_feature_matrix_markdown(
    cfg: AppConfig,
    *,
    production_models: list[str] | None = None,
    comparison_models: list[str] | None = None,
) -> str:
    features_df = load_features_dataframe(cfg.paths.processed_dir)
    production_reports = _production_reports(cfg, features_df, selected_models=production_models)
    comparison_reports = _candidate_reports(cfg, selected_models=comparison_models)

    lines = [
        "# MLB Model Feature/Beta Matrix",
        "",
        f"Generated at: `{utc_now_iso()}`",
        "",
        "Regenerate with:",
        "`python3 scripts/update_model_feature_beta_matrix.py --config configs/mlb.yaml`",
        "",
        "This file separates the production theory-core fit from the latest candidate-comparison fit so the same model key does not silently refer to two different feature sets.",
        "",
    ]

    if production_reports:
        lines.extend(
            [
                "## Production Theory-Core Snapshot",
                "",
                "Source: current processed MLB feature dataset plus the committed production feature map.",
                "",
                _section_summary_table(production_reports),
                "",
                "### Feature Matrix",
                "",
                _selection_matrix_table(production_reports),
                "",
                "### Coefficient Detail",
                "",
                _coefficient_detail_markdown(production_reports),
                "",
            ]
        )

    if comparison_reports:
        lines.extend(
            [
                "## Candidate Comparison Snapshot",
                "",
                "Source: latest candidate comparison manifest plus a reconstruction of the latest comparison fit window and selected hyperparameters.",
                "",
                _section_summary_table(comparison_reports),
                "",
                "### Base-Feature Selection Matrix",
                "",
                _selection_matrix_table(comparison_reports),
                "",
                "### Coefficient Detail",
                "",
                _coefficient_detail_markdown(comparison_reports),
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- Linear production coefficients use original-feature scale when available.",
            "- GAM spline rows and MARS hinge rows are basis-term coefficients, not a single raw-feature beta.",
            "- GLMM rows include fixed effects only in the matrix; team random effects are omitted.",
            "- DGLM rows include both mean and dispersion components for each selected feature.",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines if line is not None).strip() + "\n"


def write_model_feature_matrix_markdown(
    cfg: AppConfig,
    *,
    output_path: str | Path = _DEFAULT_REPORT_PATH,
    production_models: list[str] | None = None,
    comparison_models: list[str] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_model_feature_matrix_markdown(
            cfg,
            production_models=production_models,
            comparison_models=comparison_models,
        )
    )
    return path
