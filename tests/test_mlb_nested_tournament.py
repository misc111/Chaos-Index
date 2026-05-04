import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from src.common.config import load_config
from src.data_sources.base import SourceFetchResult
from src.research.mlb_feature_availability import write_mlb_feature_availability_report
from src.research.mlb_targets import build_target_frames
from src.research.nested_tournament import _gate_reasons, run_mlb_nested_tournament, run_mlb_parallel_nested_tournament
from src.research.nested_tournament_features import _structured_variants
from src.services.ingest import insert_odds_snapshot_and_lines, upsert_games
from src.storage.db import Database


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


def test_mlb_target_builder_adds_pregame_market_features_from_odds_db(tmp_path):
    df = _nested_frame(3).drop(columns=["runline_home_point", "totals_point"])
    db = Database(str(tmp_path / "mlb.db"))
    db.init_schema()
    upsert_games(
        db,
        df[
            [
                "game_id",
                "season",
                "game_date_utc",
                "start_time_utc",
                "home_team",
                "away_team",
                "venue",
                "home_score",
                "away_score",
                "home_win",
                "status_final",
                "as_of_utc",
            ]
        ].assign(
            game_state="Final",
            home_team_id=1,
            away_team_id=2,
            is_neutral_site=0,
            went_ot=0,
            went_so=0,
        ),
        league="MLB",
    )
    odds_rows = []
    for row in df.itertuples(index=False):
        for market_key, sides in {
            "h2h": [("home", row.home_team, -130, None, 0.565217), ("away", row.away_team, 120, None, 0.454545)],
            "spreads": [("home", row.home_team, -110, -1.5, 0.52381), ("away", row.away_team, -110, 1.5, 0.52381)],
            "totals": [("over", "Over", -105, 8.5, 0.512195), ("under", "Under", -115, 8.5, 0.534884)],
        }.items():
            for side, name, price, point, implied in sides:
                odds_rows.append(
                    {
                        "sport_key": "baseball_mlb",
                        "odds_event_id": str(row.game_id),
                        "commence_time_utc": row.start_time_utc,
                        "commence_date_central": row.game_date_utc,
                        "api_home_team": row.home_team,
                        "api_away_team": row.away_team,
                        "home_team": row.home_team,
                        "away_team": row.away_team,
                        "bookmaker_key": "espn",
                        "bookmaker_title": "ESPN",
                        "bookmaker_last_update_utc": "2025-02-28T12:00:00Z",
                        "market_key": market_key,
                        "outcome_name": name,
                        "outcome_side": side,
                        "outcome_team": row.home_team if side == "home" else row.away_team if side == "away" else None,
                        "outcome_price": price,
                        "outcome_point": point,
                        "implied_probability": implied,
                    }
                )
    insert_odds_snapshot_and_lines(
        db,
        league="MLB",
        odds_res=SourceFetchResult(
            source="mlb_historical_odds_backfill",
            snapshot_id="odds_1",
            extracted_at_utc="2025-02-28T12:00:00Z",
            raw_path=str(tmp_path / "odds.csv"),
            metadata={"import_mode": "historical_bundle", "n_events": 3},
            dataframe=pd.DataFrame(odds_rows),
        ),
    )

    by_target = {
        result.definition.target_name: result
        for result in build_target_frames(df, db_path=db.db_path, league="MLB", targets="all")
    }

    moneyline = by_target["moneyline_home_win"].frame
    assert {"market_vig_free_home_prob", "market_logit_home_prob", "market_offset_logit"} <= set(moneyline.columns)
    assert moneyline["market_vig_free_home_prob"].notna().all()
    runline = by_target["runline_home_cover"]
    assert runline.coverage["line_rows"] == 3
    assert {"runline_home_point", "market_runline_vig_free_positive_prob"} <= set(runline.frame.columns)
    totals = by_target["totals_over"]
    assert totals.coverage["line_rows"] == 3
    assert {"totals_point", "market_totals_vig_free_positive_prob"} <= set(totals.frame.columns)


def test_nested_gate_uses_calibration_intercept_near_zero_and_slope_near_one():
    row = pd.Series(
        {
            "fit_status": "ok",
            "has_class_contrast": True,
            "ece": 0.01,
            "auc": 0.61,
            "calibration_alpha": 0.02,
            "calibration_beta": 1.03,
            "top_vs_bottom_actual_lift_ratio": 1.2,
            "validation_to_cv_log_loss_delta": 0.01,
        }
    )

    assert _gate_reasons(row) == []

    row["calibration_alpha"] = 1.0
    assert "calibration_alpha_beta_outside_0.15" in _gate_reasons(row)


def test_structured_weather_context_requires_real_context_features(tmp_path):
    spec_path = tmp_path / "structured.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "slates": {
                    "weather_context": {
                        "minimum_retained_features": 4,
                        "minimum_retained_ratio": 0.5,
                        "required_any_features": ["park_run_factor", "temperature_f", "wind_out_mph"],
                        "feature_order": [
                            "park_run_factor",
                            "temperature_f",
                            "wind_out_mph",
                            "starter_quality_edge",
                            "rest_edge",
                            "travel_miles_edge",
                        ],
                        "width_variants": {"broad": {"feature_count": 6}},
                    },
                    "schedule_context": {
                        "feature_order": ["rest_edge", "travel_miles_edge"],
                        "width_variants": {"narrow": {"feature_count": 2}},
                    },
                }
            }
        )
    )
    feature_sets = SimpleNamespace(
        screened_features=["rest_diff", "travel_diff"],
        ranking_frame=pd.DataFrame({"feature": ["rest_diff", "travel_diff"]}),
        screening_frame=pd.DataFrame(
            [
                {"feature": "rest_edge", "reason": "exact_duplicate", "retained_as": "rest_diff"},
                {"feature": "travel_miles_edge", "reason": "exact_duplicate", "retained_as": "travel_diff"},
                {
                    "feature": "starter_quality_edge",
                    "reason": "constant_or_singleton_on_fit_window",
                    "retained_as": "starter_quality_edge",
                },
            ]
        ),
    )

    variants = _structured_variants(feature_sets, spec_path=spec_path)
    keys = {variant.variant_key for variant in variants}

    assert "weather_context_broad" not in keys
    assert "schedule_context_narrow" in keys


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
        candidate_models="market_baseline,glm_vanilla",
        structured_glm_spec_path=None,
        bootstrap_samples=20,
    )

    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text())
    assert Path(summary["artifacts"]["validation_autopsy_json"]).exists()
    assert Path(summary["artifacts"]["family_champion_gate_autopsy_path"]).exists()
    coverage = pd.read_csv(result.target_coverage_path)
    assert set(coverage["target_name"]) == {"moneyline_home_win", "runline_home_cover", "totals_over"}
    assert set(coverage["status"]) == {"ok"}

    variants = pd.read_csv(result.variant_leaderboard_path)
    vanilla_variants = variants[(variants["model_name"] == "glm_vanilla") & (variants["target_name"] == "moneyline_home_win")]
    assert vanilla_variants["variant_key"].nunique() >= 2
    assert vanilla_variants["variant_key"].str.contains("stability_reduced|market_aware").any()
    assert vanilla_variants["variant_key"].str.contains("platt_calibrated|isotonic_calibrated").any()
    assert vanilla_variants["candidate_key"].is_unique
    assert (
        variants["candidate_key"]
        == variants["target_name"].astype(str) + "::" + variants["model_name"].astype(str) + "::" + variants["variant_key"].astype(str)
    ).all()
    market_rows = variants[(variants["model_name"] == "market_baseline") & (variants["target_name"] == "moneyline_home_win")]
    assert {"market_implied_only", "market_offset_glm", "market_plus_features_glm"} <= set(market_rows["variant_key"])

    feature_coverage_path = Path(summary["artifacts"]["feature_coverage_by_split_path"])
    feature_coverage_json = Path(summary["artifacts"]["feature_coverage_by_split_json"])
    assert feature_coverage_path.exists()
    assert feature_coverage_json.exists()
    feature_coverage = pd.read_csv(feature_coverage_path)
    assert {"target_name", "split", "feature", "present", "coverage"} <= set(feature_coverage.columns)
    assert {"train", "validation", "final_holdout"} <= set(feature_coverage["split"])
    feature_payload = json.loads(feature_coverage_json.read_text())
    assert feature_payload["summary"]

    champions = pd.read_csv(result.family_champions_path)
    assert {"candidate_key", "variant_key", "champion_status"} <= set(champions.columns)
    assert {"blocked_reason_summary", "gate_reason_labels"} <= set(champions.columns)
    assert set(champions["candidate_key"]).issubset(set(variants["candidate_key"]))
    vanilla_champions = champions[champions["model_name"] == "glm_vanilla"]
    assert not vanilla_champions.empty
    assert set(vanilla_champions["fit_status"]) == {"ok"}
    inter = pd.read_csv(result.inter_family_leaderboard_path)
    promoted = champions[champions["champion_status"] == "shortlist"]
    if promoted.empty:
        assert inter.empty
    else:
        assert set(inter["candidate_key"]).issubset(set(promoted["candidate_key"]))
        assert inter.groupby("target_name")["target_rank"].min().eq(1).all()
        assert not ((inter["model_name"] == "glm_ridge") & (inter["fit_status"] == "failed")).any()
        first_artifacts = json.loads(inter.iloc[0]["diagnostic_artifacts"].replace("'", '"')) if isinstance(inter.iloc[0]["diagnostic_artifacts"], str) else {}
        assert first_artifacts or "diagnostic_artifacts" in inter.columns

    current_best = json.loads(result.current_best_models_path.read_text())
    assert current_best["source_kind"] == "nested_mlb_model_tournament"
    assert "blocked_family_champions" in current_best
    assert {row["candidate_key"] for row in current_best["target_champions"]} == set(inter["candidate_key"])


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
        candidate_models="market_baseline,glm_vanilla",
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
    current_best = json.loads(result.current_best_models_path.read_text())
    blocked_keys = set(family_champions.loc[family_champions["champion_status"] != "shortlist", "candidate_key"])
    assert {row["candidate_key"] for row in current_best["blocked_family_champions"]} == blocked_keys
    summary = json.loads(result.summary_path.read_text())
    assert Path(summary["artifacts"]["feature_coverage_by_split_path"]).exists()
