"""Canonical model registry shared across training, reports, and the web app."""

from __future__ import annotations

from typing import Any

from src.registry.types import ModelRegistryEntry


CORE_MODEL_KEYS: tuple[str, ...] = (
    "glm_ridge",
    "glm_elastic_net",
    "glm_lasso",
    "glm_lasso_market_credibility",
    "glm_lasso_prior_credibility",
    "glm_vanilla",
)
THEORY_EXTENSION_MODEL_KEYS: tuple[str, ...] = (
    "gam_spline",
    "glmm_logit",
    "dglm_margin",
    "goals_poisson",
)
PENALIZED_CORE_MODEL_KEYS: tuple[str, ...] = (
    "glm_ridge",
    "glm_elastic_net",
    "glm_lasso",
)
PLANNED_CREDIBILITY_MODEL_KEYS: tuple[str, ...] = (
    "glm_lasso_market_credibility",
    "glm_lasso_prior_credibility",
)
PLANNED_CREDIBILITY_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "glm_lasso_market_credibility": {
        "display_label": "Lasso Credibility (Market Offset)",
        "short_label": "Cred Market",
        "family": "credibility",
        "lane": "core",
        "status": "active_opt_in",
        "driver_source_model": "glm_lasso",
        "governance_note": "Holmes/Casotto lasso-credibility lane for explicit vig-free market complements; core-supported only when the complement is populated and documented pregame.",
        "aliases": ("market_offset_lasso", "market_lasso_credibility"),
        "complement_kind": "market",
        "complement_label": "Vig-free market logit",
        "complement_column": "market_offset_logit",
        "offset_scale": "logit",
    },
    "glm_lasso_prior_credibility": {
        "display_label": "Lasso Credibility (Prior Offset)",
        "short_label": "Cred Prior",
        "family": "credibility",
        "lane": "core",
        "status": "active_opt_in",
        "driver_source_model": "glm_lasso",
        "governance_note": "Holmes/Casotto lasso-credibility lane for explicit prior-model complements on the logit scale; proxy priors remain experimental until an upstream pregame ledger exists.",
        "aliases": ("prior_offset_lasso", "prior_lasso_credibility"),
        "complement_kind": "prior_model",
        "complement_label": "Prior-model offset logit",
        "complement_column": "prior_model_offset_logit",
        "offset_scale": "logit",
    },
}
CAS_CORE_FAMILY_CATALOG: dict[str, dict[str, Any]] = {
    "penalized_glm": {
        "label": "CAS-core penalized GLM family",
        "classification": "core-supported",
        "active_model_keys": PENALIZED_CORE_MODEL_KEYS,
        "planned_model_keys": (),
        "note": "Executable MLB defaults for ridge, elastic net, and lasso stay in the primary theory lane.",
    },
    "lasso_credibility": {
        "label": "CAS-core lasso credibility family",
        "classification": "core-supported",
        "active_model_keys": PLANNED_CREDIBILITY_MODEL_KEYS,
        "planned_model_keys": (),
        "note": "Trainable opt-in credibility variants are available for market-offset and prior-offset workflows, but they stay out of the default lane until the complement columns are populated upstream.",
    },
}
PRIMARY_MODEL_LANE = "core"
DEFAULT_TRAINING_MODEL_KEYS: tuple[str, ...] = (
    "glm_ridge",
    "glm_elastic_net",
    "glm_lasso",
    "glm_vanilla",
)
BASELINE_MODEL_KEYS: tuple[str, ...] = ()
EXPERIMENTAL_MODEL_KEYS: tuple[str, ...] = ()
REPORT_LANE_PRIORITY: dict[str, int] = {
    "core": 0,
    "extension": 1,
    "baseline": 2,
    "experimental": 3,
}
EXPECTED_THEORY_CLASSIFICATION_BY_LANE: dict[str, str] = {
    "core": "core-supported",
    "extension": "theory-compatible extension",
    "baseline": "experimental",
    "experimental": "experimental",
}
EXPECTED_IMPLEMENTATION_NAMESPACE_BY_LANE: dict[str, str | None] = {
    "core": "src.models",
    "extension": "src.models.extensions",
    "baseline": None,
    "experimental": "src.models.experimental",
}
GOVERNANCE_COMPARISON_GROUPS: dict[str, dict[str, Any]] = {
    "theory_core_default": {
        "label": "Theory program default",
        "lane": "core",
        "classification": "core-supported",
        "champion_eligible": True,
        "model_keys": DEFAULT_TRAINING_MODEL_KEYS,
        "note": "Default MLB production/theory-core run list. Extension and experimental challengers require explicit opt-in.",
    },
    "theory_core_opt_in": {
        "label": "Theory-core opt-in",
        "lane": "core",
        "classification": "core-supported",
        "champion_eligible": True,
        "model_keys": PLANNED_CREDIBILITY_MODEL_KEYS,
        "note": "Lasso-credibility complements stay opt-in and should be reported separately from the default run list.",
    },
    "baseline_references": {
        "label": "Baseline references",
        "lane": "baseline",
        "classification": "experimental",
        "champion_eligible": False,
        "model_keys": BASELINE_MODEL_KEYS,
        "note": "Retired for the monograph-only MLB lane. Market-style reference comparisons should be handled through theory-backed complement review instead.",
    },
    "theory_compatible_extensions": {
        "label": "Theory-compatible extensions",
        "lane": "extension",
        "classification": "theory-compatible extension",
        "champion_eligible": False,
        "model_keys": THEORY_EXTENSION_MODEL_KEYS,
        "note": "GLM-adjacent extension challengers are traceable but stay out of the default theory-core namespace and promotion lane.",
    },
    "experimental_challengers": {
        "label": "Experimental challengers",
        "lane": "experimental",
        "classification": "experimental",
        "champion_eligible": False,
        "model_keys": EXPERIMENTAL_MODEL_KEYS,
        "note": "Retired for the monograph-only MLB lane.",
    },
}


MODEL_REGISTRY: tuple[ModelRegistryEntry, ...] = (
    ModelRegistryEntry(
        key="glm_ridge",
        display_label="GLM Ridge",
        short_label="GLM Ridge",
        family="linear",
        lane="core",
        governance_note="Core-supported penalized GLM lane.",
        aliases=("glm", "logit", "glm_logit"),
        legacy_model_keys=("glm_logit",),
        prediction_report_rank=1,
    ),
    ModelRegistryEntry(
        key="glm_elastic_net",
        display_label="GLM ENet",
        short_label="GLM ENet",
        family="linear",
        lane="core",
        governance_note="Core-supported penalized GLM lane; not lasso credibility.",
        aliases=("elastic", "enet", "glm_enet"),
        legacy_model_keys=("glm_ridge", "glm_logit"),
        prediction_report_rank=2,
    ),
    ModelRegistryEntry(
        key="glm_lasso",
        display_label="GLM Lasso",
        short_label="GLM Lasso",
        family="linear",
        lane="core",
        governance_note="Core-supported penalized GLM lane with sparsity-focused review.",
        aliases=("lasso",),
        legacy_model_keys=("glm_ridge", "glm_logit"),
        prediction_report_rank=3,
    ),
    ModelRegistryEntry(
        key="glm_lasso_market_credibility",
        display_label="Lasso Credibility (Market)",
        short_label="Cred Market",
        family="credibility",
        lane="core",
        governance_note="Holmes/Casotto lasso-credibility lane; core-supported only when market complement columns are populated and documented.",
        aliases=("market_offset_lasso", "market_lasso_credibility"),
        default_enabled=False,
        prediction_report_rank=4,
    ),
    ModelRegistryEntry(
        key="glm_lasso_prior_credibility",
        display_label="Lasso Credibility (Prior)",
        short_label="Cred Prior",
        family="credibility",
        lane="core",
        governance_note="Holmes/Casotto lasso-credibility lane; proxy priors remain experimental until a real prior complement ledger is populated.",
        aliases=("prior_offset_lasso", "prior_lasso_credibility"),
        default_enabled=False,
        prediction_report_rank=5,
    ),
    ModelRegistryEntry(
        key="glm_vanilla",
        display_label="Vanilla GLM",
        short_label="Vanilla GLM",
        family="linear",
        lane="core",
        governance_note="Core-supported vanilla binomial/logit GLM lane.",
        aliases=("vanilla_glm",),
        prediction_report_rank=6,
    ),
    ModelRegistryEntry(
        key="gam_spline",
        display_label="GAM Spline",
        short_label="GAM",
        family="nonlinear",
        lane="extension",
        theory_classification="theory-compatible extension",
        implementation_namespace="src.models.extensions",
        governance_note="Theory-compatible GLM extension; spline-basis implementation stays opt-in outside the default theory-core lane.",
        aliases=("gam",),
        default_enabled=False,
        prediction_report_rank=7,
    ),
    ModelRegistryEntry(
        key="glmm_logit",
        display_label="GLMM Logit",
        short_label="GLMM",
        family="nonlinear",
        lane="extension",
        theory_classification="theory-compatible extension",
        implementation_namespace="src.models.extensions",
        governance_note="Theory-compatible GLM extension; opt-in outside the default theory-core lane.",
        aliases=("glmm",),
        default_enabled=False,
        prediction_report_rank=8,
    ),
    ModelRegistryEntry(
        key="dglm_margin",
        display_label="DGLM Margin",
        short_label="DGLM",
        family="nonlinear",
        lane="extension",
        theory_classification="theory-compatible extension",
        implementation_namespace="src.models.extensions",
        governance_note="Theory-compatible GLM extension; margin-to-win bridge requires separate validation before promotion.",
        aliases=("dglm",),
        default_enabled=False,
        prediction_report_rank=9,
    ),
    ModelRegistryEntry(
        key="goals_poisson",
        display_label="Goals Pois",
        short_label="Goals Pois",
        family="goals",
        lane="extension",
        theory_classification="theory-compatible extension",
        implementation_namespace="src.models.extensions",
        governance_note="Theory-compatible count-GLM lane; current MLB score-to-win bridge requires separate validation before promotion.",
        aliases=("goals",),
        default_enabled=False,
        prediction_report_rank=10,
    ),
)

_MODEL_BY_KEY = {entry.key: entry for entry in MODEL_REGISTRY}
_MODEL_ALIAS_MAP = {
    alias.lower(): entry.key
    for entry in MODEL_REGISTRY
    for alias in (entry.key, *entry.aliases)
}


def ordered_model_entries() -> tuple[ModelRegistryEntry, ...]:
    """Return models in canonical registration order."""

    return MODEL_REGISTRY


def trainable_model_names() -> list[str]:
    """Return the canonical trainable-model list."""

    return [entry.key for entry in MODEL_REGISTRY if entry.trainable]


def core_model_names() -> list[str]:
    """Return the MLB core-lane model list."""

    return [key for key in CORE_MODEL_KEYS if key in _MODEL_BY_KEY]


def theory_extension_model_names() -> list[str]:
    """Return theory-compatible extension model keys kept out of the default core lane."""

    return [key for key in THEORY_EXTENSION_MODEL_KEYS if key in _MODEL_BY_KEY]


def penalized_core_model_names() -> list[str]:
    """Return the penalized-GLM portion of the MLB CAS core lane."""

    return [key for key in PENALIZED_CORE_MODEL_KEYS if key in _MODEL_BY_KEY]


def planned_credibility_model_names() -> list[str]:
    """Return reserved lasso-credibility model keys tracked in the catalog."""

    return list(PLANNED_CREDIBILITY_MODEL_KEYS)


def planned_credibility_model_catalog() -> dict[str, dict[str, Any]]:
    """Return reserved metadata for upcoming lasso-credibility models."""

    return {
        model_name: {
            **payload,
            "aliases": list(payload.get("aliases", ())),
        }
        for model_name, payload in PLANNED_CREDIBILITY_MODEL_CATALOG.items()
    }


def planned_model_aliases() -> dict[str, str]:
    """Return aliases for reserved-but-not-trainable cataloged models."""

    out: dict[str, str] = {}
    for model_name, payload in PLANNED_CREDIBILITY_MODEL_CATALOG.items():
        out[str(model_name).lower()] = str(model_name)
        for alias in payload.get("aliases", ()):
            out[str(alias).lower()] = str(model_name)
    return out


def cas_core_family_catalog() -> dict[str, dict[str, Any]]:
    """Return explicit CAS-core family coverage for the MLB theory lane."""

    return {
        family_name: {
            "label": str(payload["label"]),
            "classification": str(payload["classification"]),
            "active_model_keys": [str(value) for value in payload.get("active_model_keys", ())],
            "planned_model_keys": [str(value) for value in payload.get("planned_model_keys", ())],
            "note": str(payload["note"]),
        }
        for family_name, payload in CAS_CORE_FAMILY_CATALOG.items()
    }


def governance_comparison_groups() -> dict[str, dict[str, Any]]:
    """Return governance-first model cohorts for comparison and reporting surfaces."""

    return {
        group_name: {
            "label": str(payload["label"]),
            "lane": str(payload["lane"]),
            "classification": str(payload["classification"]),
            "champion_eligible": bool(payload.get("champion_eligible", False)),
            "model_keys": [str(value) for value in payload.get("model_keys", ())],
            "note": str(payload["note"]),
        }
        for group_name, payload in GOVERNANCE_COMPARISON_GROUPS.items()
    }


def baseline_model_names() -> list[str]:
    """Return the MLB baseline-lane model list."""

    return [key for key in BASELINE_MODEL_KEYS if key in _MODEL_BY_KEY]


def experimental_model_names() -> list[str]:
    """Return the MLB challenger-lane model list."""

    return [key for key in EXPERIMENTAL_MODEL_KEYS if key in _MODEL_BY_KEY]


def default_training_model_names() -> list[str]:
    """Return the default train-time model list for the MLB primary lane."""

    return [key for key in DEFAULT_TRAINING_MODEL_KEYS if key in _MODEL_BY_KEY]


def prediction_report_order() -> list[str]:
    """Return the canonical prediction report ordering."""

    ordered = sorted(
        (entry for entry in MODEL_REGISTRY if entry.trainable),
        key=lambda entry: (REPORT_LANE_PRIORITY.get(entry.lane, 99), entry.prediction_report_rank, entry.key),
    )
    return ["ensemble", *[entry.key for entry in ordered]]


def model_aliases() -> dict[str, str]:
    """Return the canonical model alias mapping."""

    return dict(_MODEL_ALIAS_MAP)


def legacy_model_keys() -> dict[str, tuple[str, ...]]:
    """Return legacy model identifiers keyed by canonical model."""

    return {entry.key: entry.legacy_model_keys for entry in MODEL_REGISTRY if entry.legacy_model_keys}


def model_display_labels() -> dict[str, str]:
    """Return the canonical display label mapping including ensemble."""

    labels = {entry.key: entry.display_label for entry in MODEL_REGISTRY}
    labels["ensemble"] = "Ensemble"
    return labels


def get_model_registry_entry(value: str) -> ModelRegistryEntry:
    """Resolve a canonical model key into registry metadata."""

    return _MODEL_BY_KEY[value]


def validate_model_registry_contract() -> list[str]:
    """Return hard model-governance violations for default MLB training lanes."""

    failures: list[str] = []
    entries = {entry.key: entry for entry in MODEL_REGISTRY}

    if len(entries) != len(MODEL_REGISTRY):
        failures.append("MODEL_REGISTRY contains duplicate model keys.")

    alias_owner: dict[str, str] = {}
    for entry in MODEL_REGISTRY:
        expected_classification = EXPECTED_THEORY_CLASSIFICATION_BY_LANE.get(entry.lane)
        if expected_classification is None:
            failures.append(f"{entry.key}: unknown lane {entry.lane!r}.")
        elif entry.theory_classification != expected_classification:
            failures.append(
                f"{entry.key}: lane {entry.lane!r} must use theory_classification {expected_classification!r}."
            )

        expected_namespace = EXPECTED_IMPLEMENTATION_NAMESPACE_BY_LANE.get(entry.lane)
        if expected_namespace is not None and entry.implementation_namespace != expected_namespace:
            failures.append(f"{entry.key}: lane {entry.lane!r} must live under {expected_namespace}.")

        if entry.lane != PRIMARY_MODEL_LANE and entry.default_enabled:
            failures.append(f"{entry.key}: non-core models must not be default_enabled.")

        for alias in (entry.key, *entry.aliases):
            normalized = alias.lower()
            owner = alias_owner.get(normalized)
            if owner is not None and owner != entry.key:
                failures.append(f"alias {alias!r} is shared by {owner} and {entry.key}.")
            alias_owner[normalized] = entry.key

    for key in DEFAULT_TRAINING_MODEL_KEYS:
        default_entry = entries.get(key)
        if default_entry is None:
            failures.append(f"default training model {key!r} is not registered.")
            continue
        if default_entry.lane != PRIMARY_MODEL_LANE:
            failures.append(f"default training model {key!r} is in non-core lane {default_entry.lane!r}.")
        if default_entry.theory_classification != "core-supported":
            failures.append(f"default training model {key!r} is not core-supported.")
        if not default_entry.default_enabled:
            failures.append(f"default training model {key!r} must be default_enabled.")

    for group_name, payload in GOVERNANCE_COMPARISON_GROUPS.items():
        model_keys = [str(value) for value in payload.get("model_keys", ())]
        unknown = sorted(key for key in model_keys if key not in entries)
        if unknown:
            failures.append(f"governance group {group_name!r} references unknown models: {unknown}.")
        if bool(payload.get("champion_eligible", False)):
            non_core = sorted(key for key in model_keys if key in entries and entries[key].lane != PRIMARY_MODEL_LANE)
            if non_core:
                failures.append(f"champion-eligible group {group_name!r} contains non-core models: {non_core}.")

    default_group_models = tuple(GOVERNANCE_COMPARISON_GROUPS["theory_core_default"]["model_keys"])
    if default_group_models != DEFAULT_TRAINING_MODEL_KEYS:
        failures.append("theory_core_default group must exactly match DEFAULT_TRAINING_MODEL_KEYS.")

    return failures


def model_manifest_payload() -> dict[str, object]:
    """Render the deterministic model manifest payload."""

    contract_failures = validate_model_registry_contract()
    if contract_failures:
        raise RuntimeError("Model registry governance contract failed:\n" + "\n".join(contract_failures))

    return {
        "version": 1,
        "source": "code_registry",
        "primary_lane": PRIMARY_MODEL_LANE,
        "trainable_models": trainable_model_names(),
        "default_training_models": default_training_model_names(),
        "core_models": core_model_names(),
        "theory_extension_models": theory_extension_model_names(),
        "penalized_core_models": penalized_core_model_names(),
        "baseline_models": baseline_model_names(),
        "experimental_models": experimental_model_names(),
        "planned_credibility_model_keys": planned_credibility_model_names(),
        "planned_credibility_models": planned_credibility_model_catalog(),
        "planned_model_aliases": planned_model_aliases(),
        "cas_core_families": cas_core_family_catalog(),
        "lane_labels": {
            "core": "CAS core lane",
            "extension": "Theory-compatible extension lane",
            "baseline": "Baseline comparison lane",
            "experimental": "Experimental challenger lane",
        },
        "lane_notes": {
            "core": "Default MLB actuarial program driven by GLM-family, penalized GLM, and lasso credibility.",
            "extension": "GLM-adjacent extension challengers with monograph traceability, opt-in outside the default core lane.",
            "baseline": "Retired for the monograph-only MLB lane.",
            "experimental": "Retired for the monograph-only MLB lane.",
        },
        "aliases": model_aliases(),
        "legacy_model_keys": {key: list(values) for key, values in legacy_model_keys().items()},
        "prediction_report_order": prediction_report_order(),
        "display_labels": model_display_labels(),
        "models": {
            entry.key: {
                "display_label": entry.display_label,
                "short_label": entry.short_label,
                "family": entry.family,
                "lane": entry.lane,
                "theory_classification": entry.theory_classification,
                "implementation_namespace": entry.implementation_namespace,
                "governance_note": entry.governance_note,
                "aliases": list(entry.aliases),
                "legacy_model_keys": list(entry.legacy_model_keys),
                "trainable": entry.trainable,
                "default_enabled": entry.default_enabled,
                "prediction_report_rank": entry.prediction_report_rank,
            }
            for entry in MODEL_REGISTRY
        },
    }
