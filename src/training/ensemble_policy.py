"""League-aware policy for which model outputs can drive the ensemble."""

from __future__ import annotations

from collections.abc import Sequence


_DEMOTED_ENSEMBLE_MODELS: dict[str, frozenset[str]] = {
    "NBA": frozenset({"simulation_first"}),
}


def _normalize_league(league: str | None) -> str:
    return str(league or "").strip().upper()


def demoted_ensemble_models(*, league: str | None = None) -> list[str]:
    return sorted(_DEMOTED_ENSEMBLE_MODELS.get(_normalize_league(league), frozenset()))


def ensemble_component_columns(model_columns: Sequence[str], *, league: str | None = None) -> list[str]:
    cols = [str(col) for col in model_columns]
    demoted = set(demoted_ensemble_models(league=league))
    allowed = [col for col in cols if col not in demoted]
    return allowed or cols
