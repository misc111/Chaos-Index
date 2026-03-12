"""Canonical model registry shared across training, reports, and the web app."""

from __future__ import annotations

from src.registry.types import ModelRegistryEntry


MODEL_REGISTRY: tuple[ModelRegistryEntry, ...] = (
    ModelRegistryEntry(
        key="elo_baseline",
        display_label="Elo",
        short_label="Elo",
        family="ratings",
        aliases=("elo",),
        prediction_report_rank=1,
    ),
    ModelRegistryEntry(
        key="dynamic_rating",
        display_label="Dyn Rating",
        short_label="Dyn Rating",
        family="ratings",
        aliases=("dyn", "dynamic"),
        prediction_report_rank=5,
    ),
    ModelRegistryEntry(
        key="glm_ridge",
        display_label="GLM Ridge",
        short_label="GLM Ridge",
        family="linear",
        aliases=("glm", "logit", "glm_logit"),
        legacy_model_keys=("glm_logit",),
        prediction_report_rank=2,
    ),
    ModelRegistryEntry(
        key="glm_elastic_net",
        display_label="GLM ENet",
        short_label="GLM ENet",
        family="linear",
        aliases=("elastic", "enet", "glm_enet"),
        legacy_model_keys=("glm_ridge", "glm_logit"),
        prediction_report_rank=3,
    ),
    ModelRegistryEntry(
        key="glm_lasso",
        display_label="GLM Lasso",
        short_label="GLM Lasso",
        family="linear",
        aliases=("lasso",),
        legacy_model_keys=("glm_ridge", "glm_logit"),
        prediction_report_rank=4,
    ),
    ModelRegistryEntry(
        key="gbdt",
        display_label="GBDT",
        short_label="GBDT",
        family="tree",
        aliases=("gbm",),
        prediction_report_rank=8,
    ),
    ModelRegistryEntry(
        key="rf",
        display_label="RF",
        short_label="RF",
        family="tree",
        aliases=("forest",),
        prediction_report_rank=6,
    ),
    ModelRegistryEntry(
        key="two_stage",
        display_label="Two Stage",
        short_label="Two Stage",
        family="hybrid",
        prediction_report_rank=9,
    ),
    ModelRegistryEntry(
        key="goals_poisson",
        display_label="Goals Pois",
        short_label="Goals Pois",
        family="goals",
        aliases=("goals",),
        prediction_report_rank=7,
    ),
    ModelRegistryEntry(
        key="simulation_first",
        display_label="Sim",
        short_label="Sim",
        family="simulation",
        aliases=("sim", "simulation"),
        prediction_report_rank=12,
    ),
    ModelRegistryEntry(
        key="bayes_bt_state_space",
        display_label="Bayes BT",
        short_label="Bayes BT",
        family="bayes",
        aliases=("bayes_bt",),
        prediction_report_rank=10,
    ),
    ModelRegistryEntry(
        key="bayes_goals",
        display_label="Bayes Goals",
        short_label="Bayes Goals",
        family="bayes",
        aliases=("bayes_goals_model",),
        prediction_report_rank=11,
    ),
    ModelRegistryEntry(
        key="nn_mlp",
        display_label="NN",
        short_label="NN",
        family="neural",
        aliases=("nn",),
        prediction_report_rank=13,
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


def prediction_report_order() -> list[str]:
    """Return the canonical prediction report ordering."""

    ordered = sorted(MODEL_REGISTRY, key=lambda entry: entry.prediction_report_rank)
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
        "trainable_models": trainable_model_names(),
        "aliases": model_aliases(),
        "legacy_model_keys": {key: list(values) for key, values in legacy_model_keys().items()},
        "prediction_report_order": prediction_report_order(),
        "display_labels": model_display_labels(),
        "models": {
            entry.key: {
                "display_label": entry.display_label,
                "short_label": entry.short_label,
                "family": entry.family,
                "aliases": list(entry.aliases),
                "legacy_model_keys": list(entry.legacy_model_keys),
                "trainable": entry.trainable,
                "prediction_report_rank": entry.prediction_report_rank,
            }
            for entry in MODEL_REGISTRY
        },
    }
