from src.common.config import load_config
from src.common.research import resolve_research_paths


def test_default_config_includes_feature_policy() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.feature_policy.mode in {"production", "research"}
    assert "{league}" in cfg.feature_policy.registry_path
    assert cfg.validation_split.mode == "train_test"
    assert cfg.validation_split.method == "time"
    assert cfg.validation_split.fractions() == (0.7, 0.0, 0.3)
    assert cfg.research.history_seasons == 5
    assert cfg.research.outer_folds == 10
    assert cfg.research.outer_valid_days == 60
    assert cfg.research.inner_folds == 4
    assert cfg.research.inner_valid_days == 30
    assert cfg.research.embargo_days == 1
    assert cfg.research.final_holdout_days == 30
    assert cfg.research.feature_pool == "research_broad"


def test_research_paths_initialize_for_all_supported_leagues() -> None:
    for config_path, league_slug in [
        ("configs/nhl.yaml", "nhl"),
        ("configs/ncaam.yaml", "ncaam"),
        ("configs/nba.yaml", "nba"),
    ]:
        cfg = load_config(config_path)
        research_paths = resolve_research_paths(cfg)
        assert research_paths.league.lower() == league_slug
        assert research_paths.source_manifest.name == cfg.research.source_manifest
        assert research_paths.interim_dir.parts[-2:] == ("research", league_slug)
        assert research_paths.processed_dir.parts[-2:] == ("research", league_slug)
        assert research_paths.artifacts_dir.parts[-2:] == ("research", league_slug)
