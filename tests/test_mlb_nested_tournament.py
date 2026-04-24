import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.research.mlb_feature_availability import write_mlb_feature_availability_report
from src.research.mlb_targets import build_target_frames
from src.research.nested_tournament import run_mlb_nested_tournament, run_mlb_parallel_nested_tournament


def _nested_frame(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(44)
    dates = pd.date_range("2025-03-01", periods=n, freq="D")
    signal = rng.normal(0.0, 1.0, n)
    starter = signal + rng.normal(0.0, 0.35, n)
    bullpen = rng.normal(0.0, 1.0, n)
    weather = rng.normal(0.0, 1.0, n)
    market = 1.0 / (1.0 + np.exp(-(0.7 * signal + 0.2 * bullpen)))
    latent = 0.85 * signal + 0.35 * bullpen + rng.normal(0.0, 1.0, n)
    margin = np.where(latent > 0, np.ceil(latent + 1), np.floor(latent - 1)).astype(int)
    total_runs = np.clip(np.round(8.5 + 1.2 * weather + rng.normal(0.0, 1.4, n)), 3, 16).astype(int)
    away_score = np.clip(np.floor((total_runs - margin) / 2), 0, None).astype(int)
    home_score = np.clip(away_score + margin, 0, None).astype(int)
    home_win = (home_score > away_score).astype(int)
    return pd.DataFrame(
        {
            "game_id": np.arange(1, n + 1),
            "season": [2025] * n,
            "game_date_utc": dates.date.astype(str),
            "start_time_utc": dates.astype(str),
            "home_team": [f"H{i % 8}" for i in range(n)],
            "away_team": [f"A{(i + 2) % 8}" for i in range(n)],
            "venue": ["Park"] * n,
            "status_final": [1] * n,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": home_win,
            "as_of_utc": ["2025-03-01T00:00:00Z"] * n,
            "team_strength_edge": signal,
            "starter_quality_edge": starter,
            "bullpen_quality_edge": bullpen,
            "lineup_quality_edge": signal * 0.3 + rng.normal(0.0, 0.8, n),
            "rest_edge": rng.normal(0.0, 1.0, n),
            "travel_miles_edge": rng.normal(0.0, 1.0, n),
            "park_run_factor": weather,
            "market_vig_free_home_prob": market,
            "market_logit_home_prob": np.log(np.clip(market, 0.01, 0.99) / np.clip(1 - market, 0.01, 0.99)),
            "runline_home_point": np.full(n, 1.5),
            "totals_point": np.full(n, 8.5),
        }
    )


def test_mlb_target_builder_covers_moneyline_runline_totals_and_excludes_pushes(tmp_path):
    df = _nested_frame(24)
    df.loc[0, "home_score"] = 4
    df.loc[0, "away_score"] = 3
    df.loc[0, "runline_home_point"] = -1.0
    df.loc[1, "home_score"] = 5
    df.loc[1, "away_score"] = 4
    df.loc[1, "totals_point"] = 9.0

    results = build_target_frames(df, db_path=tmp_path / "missing.sqlite", league="MLB", targets="all")
    by_target = {result.definition.target_name: result for result in results}

    assert by_target["moneyline_home_win"].coverage["usable_rows"] == 24
    assert by_target["runline_home_cover"].coverage["push_rows"] == 1
    assert pd.isna(by_target["runline_home_cover"].frame.loc[0, "target_runline_home_cover"])
    assert by_target["totals_over"].coverage["push_rows"] == 1
    assert pd.isna(by_target["totals_over"].frame.loc[1, "target_totals_over"])


def test_mlb_nested_tournament_writes_target_scoped_champions_and_diagnostics(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.data.league = "MLB"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "mlb_forecast.db")
    cfg.modeling.cv_splits = 2

    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    _nested_frame(150).to_csv(processed_dir / "features.csv", index=False)

    result = run_mlb_nested_tournament(
        cfg,
        run_id="unit_nested",
        targets="moneyline_home_win,runline_home_cover,totals_over",
        candidate_models="glm_vanilla,glm_ridge",
        structured_glm_spec_path=None,
        bootstrap_samples=20,
    )

    assert result.summary_path.exists()
    coverage = pd.read_csv(result.target_coverage_path)
    assert set(coverage["target_name"]) == {"moneyline_home_win", "runline_home_cover", "totals_over"}
    assert set(coverage["status"]) == {"ok"}

    variants = pd.read_csv(result.variant_leaderboard_path)
    vanilla_variants = variants[(variants["model_name"] == "glm_vanilla") & (variants["target_name"] == "moneyline_home_win")]
    assert vanilla_variants["variant_key"].nunique() >= 2
    assert vanilla_variants["candidate_key"].is_unique

    champions = pd.read_csv(result.family_champions_path)
    assert {"candidate_key", "variant_key", "champion_status"} <= set(champions.columns)
    ridge_champions = champions[champions["model_name"] == "glm_ridge"]
    assert not ridge_champions.empty
    assert set(ridge_champions["fit_status"]) == {"ok"}
    inter = pd.read_csv(result.inter_family_leaderboard_path)
    assert set(inter["target_name"]) == {"moneyline_home_win", "runline_home_cover", "totals_over"}
    assert inter.groupby("target_name")["target_rank"].min().eq(1).all()
    assert not ((inter["model_name"] == "glm_ridge") & (inter["fit_status"] == "failed")).any()

    first_artifacts = json.loads(inter.iloc[0]["diagnostic_artifacts"].replace("'", '"')) if isinstance(inter.iloc[0]["diagnostic_artifacts"], str) else {}
    assert first_artifacts or "diagnostic_artifacts" in inter.columns

    current_best = json.loads(result.current_best_models_path.read_text())
    assert current_best["source_kind"] == "nested_mlb_model_tournament"
    assert {row["target_name"] for row in current_best["target_champions"]} == {
        "moneyline_home_win",
        "runline_home_cover",
        "totals_over",
    }


def test_mlb_feature_availability_report_tracks_structured_slates(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.data.league = "MLB"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    _nested_frame(48).to_csv(processed_dir / "features.csv", index=False)

    result = write_mlb_feature_availability_report(
        cfg,
        run_id="unit_feature_availability",
        structured_glm_spec_path="configs/research/mlb_tournament_structured_glm.yaml",
    )

    rows = pd.read_csv(result.csv_path)
    assert {"slate_name", "feature", "present", "coverage", "source_bucket", "pregame_safe"} <= set(rows.columns)
    starter = rows[(rows["slate_name"] == "starter_run_prevention") & (rows["feature"] == "starter_quality_edge")]
    assert not starter.empty
    assert bool(starter.iloc[0]["present"])
    missing_xfip = rows[(rows["slate_name"] == "starter_run_prevention") & (rows["feature"] == "starter_xfip_edge")]
    assert not missing_xfip.empty
    assert not bool(missing_xfip.iloc[0]["present"])
    payload = json.loads(result.json_path.read_text())
    assert payload["run_id"] == "unit_feature_availability"
    assert payload["summary"]["total_features"] >= 1


def test_parallel_nested_tournament_runs_lanes_before_final_holdout(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg.data.league = "MLB"
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.processed_dir = str(tmp_path / "processed")
    cfg.paths.db_path = str(tmp_path / "processed" / "mlb_forecast.db")
    cfg.modeling.cv_splits = 2
    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    _nested_frame(150).to_csv(processed_dir / "features.csv", index=False)

    result = run_mlb_parallel_nested_tournament(
        cfg,
        run_id="unit_parallel_nested",
        targets="moneyline_home_win,totals_over",
        candidate_models="glm_vanilla,glm_ridge",
        structured_glm_spec_path=None,
        bootstrap_samples=20,
        max_workers=2,
    )

    summary = json.loads(result.summary_path.read_text())
    assert summary["orchestration"] == "parallel_intra_family_then_central_final_holdout"
    assert summary["lane_runs"]
    lane_paths = [Path(row["validation_leaderboard_path"]) for row in summary["lane_runs"]]
    assert all(path.exists() for path in lane_paths)
    family_champions = pd.read_csv(result.family_champions_path)
    inter = pd.read_csv(result.inter_family_leaderboard_path)
    assert not family_champions.empty
    promoted = family_champions[family_champions["champion_status"] == "shortlist"]
    if promoted.empty:
        assert inter.empty
        current_best = json.loads(result.current_best_models_path.read_text())
        assert current_best["target_champions"] == []
        assert current_best["blocked_family_champions"]
    else:
        assert set(inter["candidate_key"]).issubset(set(promoted["candidate_key"]))
        assert inter.groupby("target_name")["target_rank"].min().eq(1).all()
