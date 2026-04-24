from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.leakage_checks import run_leakage_checks
from src.features.pipeline import finalize_feature_frame
from src.training.feature_selection import select_feature_columns


class _DummyStrategy:
    league = "MLB"

    @staticmethod
    def feature_hash_payload(feature_columns: list[str]) -> dict:
        return {"league": "MLB", "feature_columns": list(feature_columns)}


def _base_game_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 2],
            "season": [2026, 2026],
            "game_date_utc": ["2026-04-10", "2026-04-11"],
            "start_time_utc": ["2026-04-10T18:05:00Z", "2026-04-11T18:05:00Z"],
            "venue": ["Fenway Park", "Yankee Stadium"],
            "home_team": ["BOS", "NYY"],
            "away_team": ["NYY", "BOS"],
            "home_score": [5, 3],
            "away_score": [2, 4],
            "status_final": [1, 1],
            "home_win": [1, 0],
            "as_of_utc": ["2026-04-10T16:00:00Z", "2026-04-11T16:00:00Z"],
            "available_as_of_utc": ["2026-04-10T16:00:00Z", "2026-04-11T16:00:00Z"],
        }
    )


def test_finalize_feature_frame_fails_fast_on_post_start_availability(tmp_path: Path) -> None:
    frame = _base_game_frame()
    frame.loc[0, "available_as_of_utc"] = "2026-04-10T19:30:00Z"
    frame.loc[0, "status_final"] = 0

    with pytest.raises(ValueError, match="available_as_of_utc is after start_time_utc for 1 rows"):
        finalize_feature_frame(frame, processed_dir=str(tmp_path / "processed"), strategy=_DummyStrategy())


def test_finalize_feature_frame_normalizes_historical_final_snapshot_availability(tmp_path: Path) -> None:
    frame = _base_game_frame()
    frame["available_as_of_utc"] = ["2026-04-23T19:07:12Z", "2026-04-23T19:07:12Z"]
    frame["as_of_utc"] = frame["available_as_of_utc"]

    out = finalize_feature_frame(frame, processed_dir=str(tmp_path / "processed"), strategy=_DummyStrategy())

    available = pd.to_datetime(out.dataframe["available_as_of_utc"], errors="coerce", utc=True)
    start = pd.to_datetime(out.dataframe["start_time_utc"], errors="coerce", utc=True)
    assert available.lt(start).all()
    assert (start - available).dt.total_seconds().eq(1.0).all()
    assert out.metadata["historical_final_availability_normalized_count"] == 2


def test_finalize_feature_frame_drops_stale_post_start_nonfinal_without_outcome(tmp_path: Path) -> None:
    frame = _base_game_frame()
    frame.loc[0, "status_final"] = 0
    frame.loc[0, "home_win"] = np.nan
    frame.loc[0, "available_as_of_utc"] = "2026-04-23T19:07:12Z"
    frame.loc[0, "as_of_utc"] = "2026-04-23T19:07:12Z"

    out = finalize_feature_frame(frame, processed_dir=str(tmp_path / "processed"), strategy=_DummyStrategy())

    assert out.dataframe["game_id"].tolist() == [2]
    assert out.metadata["stale_post_start_nonfinal_dropped_count"] == 1


def test_finalize_feature_frame_preserves_missing_numeric_values(tmp_path: Path) -> None:
    frame = _base_game_frame()
    frame["diff_starter_era"] = [np.nan, 0.35]
    frame["diff_starter_whip"] = [np.inf, -np.inf]
    frame["travel_diff"] = [0.2, -0.1]

    out = finalize_feature_frame(frame, processed_dir=str(tmp_path / "processed"), strategy=_DummyStrategy())
    saved_path = Path(str(out.metadata["saved_path"]))
    if saved_path.suffix == ".parquet":
        persisted = pd.read_parquet(saved_path)
    else:
        persisted = pd.read_csv(saved_path)

    assert "diff_starter_era" in out.feature_columns
    assert "diff_starter_whip" in out.feature_columns
    assert pd.isna(out.dataframe.loc[0, "diff_starter_era"])
    assert out.dataframe["diff_starter_whip"].isna().all()
    assert pd.isna(persisted.loc[0, "diff_starter_era"])
    assert persisted["diff_starter_whip"].isna().all()


def test_starter_quality_diffs_survive_feature_selection_and_leakage_tokens() -> None:
    df = pd.DataFrame(
        {
            "game_id": [7],
            "start_time_utc": ["2026-04-20T18:05:00Z"],
            "available_as_of_utc": ["2026-04-20T17:00:00Z"],
            "status_final": [1],
            "home_win": [1],
            "diff_starter_era": [0.12],
            "diff_starter_whip": [0.07],
            "diff_starter_era_hinge_000": [0.12],
            "home_runs": [5],
            "winner": [1],
        }
    )

    selected = select_feature_columns(df)
    issues = run_leakage_checks(df, feature_columns=selected)

    assert "diff_starter_era" in selected
    assert "diff_starter_whip" in selected
    assert "diff_starter_era_hinge_000" in selected
    assert "home_runs" not in selected
    assert "winner" not in selected
    assert not any("pattern_forbidden_columns" in issue for issue in issues)
