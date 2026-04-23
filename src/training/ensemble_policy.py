"""Policy for which model outputs can drive the MLB ensemble."""

from __future__ import annotations

from collections.abc import Sequence


def demoted_ensemble_models(*, league: str | None = None) -> list[str]:
    if str(league or "MLB").strip().upper() != "MLB":
        raise ValueError(f"Unsupported league '{league}'. Expected only: MLB.")
    return []


def ensemble_component_columns(model_columns: Sequence[str], *, league: str | None = None) -> list[str]:
    cols = [str(col) for col in model_columns]
    demoted = set(demoted_ensemble_models(league=league))
    allowed = [col for col in cols if col not in demoted]
    return allowed or cols
