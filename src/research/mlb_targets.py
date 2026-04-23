from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from src.storage.schema import EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME


@dataclass(frozen=True, slots=True)
class TargetDefinition:
    target_name: str
    target_col: str
    market: str
    distribution: str = "binomial"
    link_function: str = "logit"
    positive_label: str = ""
    theory_classification: str = "direct theory implementation"


@dataclass(frozen=True, slots=True)
class TargetFrameResult:
    definition: TargetDefinition
    frame: pd.DataFrame
    coverage: dict[str, Any]


TARGET_DEFINITIONS: tuple[TargetDefinition, ...] = (
    TargetDefinition(
        target_name="moneyline_home_win",
        target_col="target_moneyline_home_win",
        market="moneyline",
        positive_label="home_win",
    ),
    TargetDefinition(
        target_name="runline_home_cover",
        target_col="target_runline_home_cover",
        market="runline",
        positive_label="home_cover",
    ),
    TargetDefinition(
        target_name="totals_over",
        target_col="target_totals_over",
        market="totals",
        positive_label="over",
    ),
)


def target_definition(target_name: str) -> TargetDefinition:
    for definition in TARGET_DEFINITIONS:
        if definition.target_name == target_name:
            return definition
    valid = ", ".join(definition.target_name for definition in TARGET_DEFINITIONS)
    raise ValueError(f"Unknown MLB target `{target_name}`. Valid targets: {valid}")


def target_definitions(raw_value: str | None = None) -> list[TargetDefinition]:
    raw = str(raw_value or "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return list(TARGET_DEFINITIONS)
    return [target_definition(token.strip()) for token in raw.split(",") if token.strip()]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _line_source_from_features(features_df: pd.DataFrame, *, market: str) -> pd.DataFrame:
    if market == "runline":
        candidates = ["runline_home_point", "home_runline_point", "spread_home_point", "home_spread_point"]
    elif market == "totals":
        candidates = ["totals_point", "total_point", "game_total_point", "over_total_point"]
    else:
        candidates = []
    for column in candidates:
        if column in features_df.columns:
            return pd.DataFrame({"game_id": features_df["game_id"], "line_point": _safe_numeric(features_df[column])})
    return pd.DataFrame(columns=["game_id", "line_point"])


def _query_odds_lines(db_path: str | Path, *, league: str, market_key: str) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
                (EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME,),
            ).fetchone()
            if table_check is None:
                return pd.DataFrame()
            rows = conn.execute(
                f"""
                SELECT game_id, market_key, outcome_side, outcome_name, outcome_point, effective_odds_as_of_utc
                FROM {EFFECTIVE_ODDS_MARKET_LINES_VIEW_NAME}
                WHERE UPPER(league) = UPPER(?)
                  AND market_key = ?
                  AND game_id IS NOT NULL
                  AND outcome_point IS NOT NULL
                """,
                (league, market_key),
            ).fetchall()
    except sqlite3.Error:
        return pd.DataFrame()
    return pd.DataFrame([dict(row) for row in rows])


def _line_source_from_odds(
    features_df: pd.DataFrame,
    *,
    db_path: str | Path,
    league: str,
    market: str,
) -> pd.DataFrame:
    market_key = "spreads" if market == "runline" else "totals"
    lines = _query_odds_lines(db_path, league=league, market_key=market_key)
    if lines.empty:
        return pd.DataFrame(columns=["game_id", "line_point"])

    meta = features_df[["game_id", "start_time_utc"]].copy()
    meta["start_time_utc"] = pd.to_datetime(meta["start_time_utc"], utc=True, errors="coerce")
    work = lines.merge(meta, on="game_id", how="inner")
    work["effective_odds_as_of_utc"] = pd.to_datetime(work["effective_odds_as_of_utc"], utc=True, errors="coerce")
    work["outcome_point"] = _safe_numeric(work["outcome_point"])
    work = work[work["effective_odds_as_of_utc"].notna() & work["start_time_utc"].notna()]
    work = work[work["effective_odds_as_of_utc"] < work["start_time_utc"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["game_id", "line_point"])

    if market == "runline":
        side = work["outcome_side"].fillna("").str.lower()
        name = work["outcome_name"].fillna("").str.lower()
        work = work[(side == "home") | (name == "home")]
    elif market == "totals":
        side = work["outcome_side"].fillna("").str.lower()
        name = work["outcome_name"].fillna("").str.lower()
        work = work[(side == "over") | (name == "over")]
    if work.empty:
        return pd.DataFrame(columns=["game_id", "line_point"])

    latest = work.groupby("game_id", as_index=False)["effective_odds_as_of_utc"].max()
    work = work.merge(latest, on=["game_id", "effective_odds_as_of_utc"], how="inner")
    return (
        work.groupby("game_id", as_index=False)["outcome_point"]
        .median()
        .rename(columns={"outcome_point": "line_point"})
    )


def _line_source(
    features_df: pd.DataFrame,
    *,
    db_path: str | Path,
    league: str,
    market: str,
) -> pd.DataFrame:
    from_features = _line_source_from_features(features_df, market=market)
    if not from_features.empty and from_features["line_point"].notna().any():
        return from_features
    return _line_source_from_odds(features_df, db_path=db_path, league=league, market=market)


def build_target_frame(
    features_df: pd.DataFrame,
    *,
    definition: TargetDefinition,
    db_path: str | Path,
    league: str = "MLB",
) -> TargetFrameResult:
    frame = features_df.copy()
    original_rows = int(len(frame))
    scored_mask = frame["home_score"].notna() & frame["away_score"].notna() if {"home_score", "away_score"} <= set(frame.columns) else pd.Series(False, index=frame.index)
    coverage: dict[str, Any] = {
        "target_name": definition.target_name,
        "market": definition.market,
        "original_rows": original_rows,
        "scored_rows": int(scored_mask.sum()),
        "line_rows": 0,
        "push_rows": 0,
        "usable_rows": 0,
        "status": "ok",
        "reason": "",
    }

    if definition.target_name == "moneyline_home_win":
        if "home_win" not in frame.columns:
            frame[definition.target_col] = np.nan
            coverage.update({"status": "coverage-blocked", "reason": "missing_home_win_column"})
        else:
            frame[definition.target_col] = _safe_numeric(frame["home_win"])
            coverage["usable_rows"] = int(frame[definition.target_col].notna().sum())
        return TargetFrameResult(definition=definition, frame=frame, coverage=coverage)

    if not {"home_score", "away_score", "game_id", "start_time_utc"} <= set(frame.columns):
        frame[definition.target_col] = np.nan
        coverage.update({"status": "coverage-blocked", "reason": "missing_score_or_game_metadata_columns"})
        return TargetFrameResult(definition=definition, frame=frame, coverage=coverage)

    lines = _line_source(frame, db_path=db_path, league=league, market=definition.market)
    coverage["line_rows"] = int(lines["line_point"].notna().sum()) if not lines.empty else 0
    if lines.empty:
        frame[definition.target_col] = np.nan
        coverage.update({"status": "coverage-blocked", "reason": f"missing_pregame_{definition.market}_lines"})
        return TargetFrameResult(definition=definition, frame=frame, coverage=coverage)

    frame = frame.merge(lines, on="game_id", how="left")
    home_score = _safe_numeric(frame["home_score"])
    away_score = _safe_numeric(frame["away_score"])
    line_point = _safe_numeric(frame["line_point"])
    if definition.target_name == "runline_home_cover":
        margin = home_score - away_score
        settled_value = margin + line_point
    else:
        total_runs = home_score + away_score
        settled_value = total_runs - line_point
    push_mask = settled_value.eq(0.0)
    frame[definition.target_col] = np.where(
        settled_value.notna() & ~push_mask,
        (settled_value > 0.0).astype(int),
        np.nan,
    )
    coverage["push_rows"] = int(push_mask.sum())
    coverage["usable_rows"] = int(pd.Series(frame[definition.target_col]).notna().sum())
    if coverage["usable_rows"] <= 0:
        coverage.update({"status": "coverage-blocked", "reason": f"no_settled_{definition.market}_target_rows"})
    return TargetFrameResult(definition=definition, frame=frame, coverage=coverage)


def build_target_frames(
    features_df: pd.DataFrame,
    *,
    db_path: str | Path,
    league: str = "MLB",
    targets: str | None = None,
) -> list[TargetFrameResult]:
    return [
        build_target_frame(features_df, definition=definition, db_path=db_path, league=league)
        for definition in target_definitions(targets)
    ]

