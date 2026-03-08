from src.common.config import load_config


def test_default_config_includes_feature_policy() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.feature_policy.mode in {"production", "research"}
    assert "{league}" in cfg.feature_policy.registry_path
    assert cfg.validation_split.mode == "train_test"
    assert cfg.validation_split.method == "time"
    assert cfg.validation_split.fractions() == (0.7, 0.0, 0.3)
