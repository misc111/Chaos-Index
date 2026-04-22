"""Canonical model registry shared across training, reports, and the web app."""

from __future__ import annotations

from src.registry.types import ModelRegistryEntry


CORE_MODEL_KEYS: tuple[str, ...] = (
    "glm_ridge",
    "glm_elastic_net",
    "glm_lasso",
    "glm_lasso_market_credibility",
    "glm_lasso_prior_credibility",
    "glm_vanilla",
    "gam_spline",
    "mars_hinge",
    "glmm_logit",
    "dglm_margin",
    "two_stage",
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
PLANNED_CREDIBILITY_MODEL_CATALOG: dict[str, dict[str, object]] = {
    "glm_lasso_market_credibility": {
        "display_label": "Lasso Credibility (Market Offset)",
        "short_label": "Cred Market",
        "family": "credibility",
        "lane": "core",
        "status": "active_opt_in",
        "driver_source_model": "glm_lasso",
        "governance_note": "Direct CAS-theory lasso-credibility lane for vig-free market complements; explicit opt-in until market complements are populated in the MLB feature store.",
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
        "governance_note": "Direct CAS-theory lasso-credibility lane for prior-model complements on the logit scale; explicit opt-in until the prior offset is populated upstream.",
        "aliases": ("prior_offset_lasso", "prior_lasso_credibility"),
        "complement_kind": "prior_model",
        "complement_label": "Prior-model offset logit",
        "complement_column": "prior_model_offset_logit",
        "offset_scale": "logit",
    },
}
CAS_CORE_FAMILY_CATALOG: dict[str, dict[str, object]] = {
    "penalized_glm": {
        "label": "CAS-core penalized GLM family",
        "classification": "Direct theory implementation",
        "active_model_keys": PENALIZED_CORE_MODEL_KEYS,
        "planned_model_keys": (),
        "note": "Executable MLB defaults for ridge, elastic net, and lasso stay in the primary theory lane.",
    },
    "lasso_credibility": {
        "label": "CAS-core lasso credibility family",
        "classification": "Direct theory implementation",
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
    "gam_spline",
    "mars_hinge",
    "glmm_logit",
    "dglm_margin",
    "two_stage",
    "goals_poisson",
)
BASELINE_MODEL_KEYS: tuple[str, ...] = (
    "elo_baseline",
    "dynamic_rating",
    "simulation_first",
)
EXPERIMENTAL_MODEL_KEYS: tuple[str, ...] = (
    "gbdt",
    "rf",
    "bayes_bt_state_space",
    "bayes_goals",
    "nn_mlp",
)
REPORT_LANE_PRIORITY: dict[str, int] = {
    "core": 0,
    "baseline": 1,
    "experimental": 2,
}
GOVERNANCE_COMPARISON_GROUPS: dict[str, dict[str, object]] = {
    "theory_core_default": {
        "label": "Theory-core default",
        "lane": "core",
        "classification": "Direct theory implementation",
        "champion_eligible": True,
        "model_keys": DEFAULT_TRAINING_MODEL_KEYS,
        "note": "Primary MLB run list for core comparison screens and champion decisions.",
    },
    "theory_core_opt_in": {
        "label": "Theory-core opt-in",
        "lane": "core",
        "classification": "Direct theory implementation",
        "champion_eligible": True,
        "model_keys": PLANNED_CREDIBILITY_MODEL_KEYS,
        "note": "Lasso-credibility complements stay opt-in and should be reported separately from the default run list.",
    },
    "baseline_references": {
        "label": "Baseline references",
        "lane": "baseline",
        "classification": "Theory-compatible engineering support",
        "champion_eligible": False,
        "model_keys": BASELINE_MODEL_KEYS,
        "note": "Reference checks for calibration and stability context; not champion candidates by default.",
    },
    "experimental_challengers": {
        "label": "Experimental challengers",
        "lane": "experimental",
        "classification": "Experimental challenger",
        "champion_eligible": False,
        "model_keys": EXPERIMENTAL_MODEL_KEYS,
        "note": "Non-CAS challengers retained behind explicit opt-in and quarantined from default promotions.",
    },
}


MODEL_REGISTRY: tuple[ModelRegistryEntry, ...] = (
    ModelRegistryEntry(
        key="glm_ridge",
        display_label="GLM Ridge",
        short_label="GLM Ridge",
        family="linear",
        lane="core",
        governance_note="Direct CAS-theory core lane.",
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
        governance_note="Direct CAS-theory core lane.",
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
        governance_note="Direct CAS-theory core lane.",
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
        governance_note="Direct CAS-theory lasso-credibility lane; explicit opt-in until market complement columns are populated.",
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
        governance_note="Direct CAS-theory lasso-credibility lane; explicit opt-in until prior complement columns are populated.",
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
        governance_note="Direct CAS-theory core lane.",
        aliases=("vanilla_glm",),
        prediction_report_rank=6,
    ),
    ModelRegistryEntry(
        key="gam_spline",
        display_label="GAM Spline",
        short_label="GAM",
        family="nonlinear",
        lane="core",
        governance_note="Theory-compatible GLM extension in the CAS core lane.",
        aliases=("gam",),
        prediction_report_rank=7,
    ),
    ModelRegistryEntry(
        key="mars_hinge",
        display_label="MARS Hinge",
        short_label="MARS",
        family="nonlinear",
        lane="core",
        governance_note="Theory-compatible GLM extension in the CAS core lane.",
        aliases=("mars",),
        prediction_report_rank=8,
    ),
    ModelRegistryEntry(
        key="glmm_logit",
        display_label="GLMM Logit",
        short_label="GLMM",
        family="nonlinear",
        lane="core",
        governance_note="Theory-compatible GLM extension in the CAS core lane.",
        aliases=("glmm",),
        prediction_report_rank=9,
    ),
    ModelRegistryEntry(
        key="dglm_margin",
        display_label="DGLM Margin",
        short_label="DGLM",
        family="nonlinear",
        lane="core",
        governance_note="Theory-compatible GLM extension in the CAS core lane.",
        aliases=("dglm",),
        prediction_report_rank=10,
    ),
    ModelRegistryEntry(
        key="two_stage",
        display_label="Two Stage",
        short_label="Two Stage",
        family="hybrid",
        lane="core",
        governance_note="Theory-compatible two-stage actuarial lane.",
        prediction_report_rank=11,
    ),
    ModelRegistryEntry(
        key="goals_poisson",
        display_label="Goals Pois",
        short_label="Goals Pois",
        family="goals",
        lane="core",
        governance_note="Direct CAS-theory run-rate lane.",
        aliases=("goals",),
        prediction_report_rank=12,
    ),
    ModelRegistryEntry(
        key="elo_baseline",
        display_label="Elo",
        short_label="Elo",
        family="ratings",
        lane="baseline",
        governance_note="Baseline comparison lane, not champion by default.",
        aliases=("elo",),
        prediction_report_rank=13,
    ),
    ModelRegistryEntry(
        key="dynamic_rating",
        display_label="Dyn Rating",
        short_label="Dyn Rating",
        family="ratings",
        lane="baseline",
        governance_note="Baseline comparison lane, not champion by default.",
        aliases=("dyn", "dynamic"),
        prediction_report_rank=14,
    ),
    ModelRegistryEntry(
        key="simulation_first",
        display_label="Sim",
        short_label="Sim",
        family="simulation",
        lane="baseline",
        governance_note="Baseline comparison lane, not champion by default.",
        aliases=("sim", "simulation"),
        prediction_report_rank=15,
    ),
    ModelRegistryEntry(
        key="gbdt",
        display_label="GBDT Challenger",
        short_label="GBDT",
        family="tree",
        lane="experimental",
        governance_note="Experimental non-CAS challenger; explicit opt-in only.",
        aliases=("gbm",),
        default_enabled=False,
        prediction_report_rank=17,
    ),
    ModelRegistryEntry(
        key="rf",
        display_label="RF Challenger",
        short_label="RF",
        family="tree",
        lane="experimental",
        governance_note="Experimental non-CAS challenger; explicit opt-in only.",
        aliases=("forest",),
        default_enabled=False,
        prediction_report_rank=16,
    ),
    ModelRegistryEntry(
        key="bayes_bt_state_space",
        display_label="Bayes BT Challenger",
        short_label="Bayes BT",
        family="bayes",
        lane="experimental",
        governance_note="Experimental non-CAS challenger; explicit opt-in only.",
        aliases=("bayes_bt",),
        default_enabled=False,
        prediction_report_rank=18,
    ),
    ModelRegistryEntry(
        key="bayes_goals",
        display_label="Bayes Goals Challenger",
        short_label="Bayes Goals",
        family="bayes",
        lane="experimental",
        governance_note="Experimental non-CAS challenger; explicit opt-in only.",
        aliases=("bayes_goals_model",),
        default_enabled=False,
        prediction_report_rank=19,
    ),
    ModelRegistryEntry(
        key="nn_mlp",
        display_label="NN Challenger",
        short_label="NN",
        family="neural",
        lane="experimental",
        governance_note="Experimental non-CAS challenger; explicit opt-in only.",
        aliases=("nn",),
        default_enabled=False,
        prediction_report_rank=20,
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


def penalized_core_model_names() -> list[str]:
    """Return the penalized-GLM portion of the MLB CAS core lane."""

    return [key for key in PENALIZED_CORE_MODEL_KEYS if key in _MODEL_BY_KEY]


def planned_credibility_model_names() -> list[str]:
    """Return reserved lasso-credibility model keys tracked in the catalog."""

    return list(PLANNED_CREDIBILITY_MODEL_KEYS)


def planned_credibility_model_catalog() -> dict[str, dict[str, object]]:
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


def cas_core_family_catalog() -> dict[str, dict[str, object]]:
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


def governance_comparison_groups() -> dict[str, dict[str, object]]:
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


def model_manifest_payload() -> dict[str, object]:
    """Render the deterministic model manifest payload."""

    return {
        "version": 1,
        "source": "code_registry",
        "primary_lane": PRIMARY_MODEL_LANE,
        "trainable_models": trainable_model_names(),
        "default_training_models": default_training_model_names(),
        "core_models": core_model_names(),
        "penalized_core_models": penalized_core_model_names(),
        "baseline_models": baseline_model_names(),
        "experimental_models": experimental_model_names(),
        "planned_credibility_model_keys": planned_credibility_model_names(),
        "planned_credibility_models": planned_credibility_model_catalog(),
        "planned_model_aliases": planned_model_aliases(),
        "cas_core_families": cas_core_family_catalog(),
        "lane_labels": {
            "core": "CAS core lane",
            "baseline": "Baseline comparison lane",
            "experimental": "Experimental challenger lane",
        },
        "lane_notes": {
            "core": "Default MLB actuarial program driven by GLM-family, penalized GLM, lasso credibility, and theory-compatible extensions.",
            "baseline": "Reference models retained for sanity checks and comparison, not for champion promotion by default.",
            "experimental": "Non-CAS challengers retained only behind explicit opt-in and never treated as the default theory lane.",
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
