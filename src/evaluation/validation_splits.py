"""Validation split planning for MLB model diagnostics.

Validation reports need an explicit split contract so operators can distinguish
training, optional validation, and holdout evidence.  This module isolates that
policy from the larger task pipeline: it accepts the already-built training
frame plus application config, then returns sorted dataframes and a compact plan
object that can be serialized into validation artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.common.config import AppConfig


@dataclass(frozen=True, slots=True)
class ValidationSplitPlan:
    """Resolved validation split metadata.

    The ``requested_*`` fields preserve operator intent from configuration.  The
    ``resolved_*`` fields record what the pipeline actually used after applying
    minimum-size safeguards, which is important for bounded MLB fixture slices
    and other non-production evidence.
    """

    requested_mode: str
    requested_method: str
    resolved_mode: str
    resolved_method: str
    train_fraction: float
    validation_fraction: float
    holdout_fraction: float
    random_seed: int | None
    note: str | None = None


def sort_validation_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return validation rows in deterministic chronological order when possible."""

    work = df.copy()
    for col in ("start_time_utc", "game_date_utc"):
        if col in work.columns:
            return work.sort_values(col)
    return work.reset_index(drop=True)


def concat_validation_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate non-empty split frames and preserve validation row ordering."""

    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, axis=0)
    return sort_validation_rows(out)


def _slice_train_test(work: pd.DataFrame, *, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a two-way train/holdout split with an empty validation frame."""

    n_obs = len(work)
    split = int(round(train_fraction * n_obs))
    if split <= 0 or split >= n_obs:
        return work.copy(), work.iloc[0:0].copy(), work.iloc[0:0].copy()
    return work.iloc[:split].copy(), work.iloc[0:0].copy(), work.iloc[split:].copy()


def _slice_train_validation_test(
    work: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a train/validation/holdout split for sufficiently large samples."""

    n_obs = len(work)
    train_end = int(round(train_fraction * n_obs))
    valid_end = int(round((train_fraction + validation_fraction) * n_obs))
    if train_end <= 0 or valid_end <= train_end or valid_end >= n_obs:
        return work.copy(), work.iloc[0:0].copy(), work.iloc[0:0].copy()
    return (
        work.iloc[:train_end].copy(),
        work.iloc[train_end:valid_end].copy(),
        work.iloc[valid_end:].copy(),
    )


def resolve_validation_split(
    train_df: pd.DataFrame,
    cfg: AppConfig,
) -> tuple[ValidationSplitPlan, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve the configured validation split into concrete dataframes.

    Rows without finalized ``home_win`` labels are excluded before splitting.
    Time-ordered splits are the default for pregame betting validation; random
    splits remain available for explicit research diagnostics and record the
    normalized seed in the returned plan.
    """

    work = train_df[train_df["home_win"].notna()].copy()
    split_cfg = cfg.validation_split
    train_fraction, validation_fraction, holdout_fraction = split_cfg.fractions()
    random_seed = None
    note: str | None = None
    if work.empty:
        plan = ValidationSplitPlan(
            requested_mode=split_cfg.mode,
            requested_method=split_cfg.method,
            resolved_mode=split_cfg.mode,
            resolved_method=split_cfg.method,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            holdout_fraction=holdout_fraction,
            random_seed=None,
            note="no_finalized_rows",
        )
        return plan, work, work.copy(), work.copy()

    if split_cfg.method == "random":
        random_seed = split_cfg.normalized_random_seed(fallback_seed=int(cfg.modeling.random_seed))
        work = work.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    else:
        work = sort_validation_rows(work).reset_index(drop=True)

    resolved_mode = split_cfg.mode
    if split_cfg.mode == "train_validation_test":
        tr, va, te = _slice_train_validation_test(
            work,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        # Tiny slices cannot support three evidence partitions.  Fall back to
        # train/test and carry the downgrade reason into report metadata.
        if min(len(tr), len(va), len(te)) < 20:
            resolved_mode = "train_test"
            train_fraction, validation_fraction, holdout_fraction = 0.7, 0.0, 0.3
            tr, va, te = _slice_train_test(work, train_fraction=train_fraction)
            note = "requested_train_validation_test_was_too_thin_so_train_test_was_used"
    else:
        tr, va, te = _slice_train_test(work, train_fraction=train_fraction)

    tr = sort_validation_rows(tr)
    va = sort_validation_rows(va)
    te = sort_validation_rows(te)
    plan = ValidationSplitPlan(
        requested_mode=split_cfg.mode,
        requested_method=split_cfg.method,
        resolved_mode=resolved_mode,
        resolved_method=split_cfg.method,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        holdout_fraction=holdout_fraction,
        random_seed=random_seed,
        note=note,
    )
    return plan, tr, va, te
