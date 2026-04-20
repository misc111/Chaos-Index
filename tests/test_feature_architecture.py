from importlib.util import find_spec
from pathlib import Path


def test_nhl_only_feature_helpers_are_isolated_from_shared_root() -> None:
    root = Path("src/features")

    assert not (root / "goalie_features.py").exists()
    assert not (root / "rink_adjustments.py").exists()
    assert not (root / "special_teams.py").exists()

    assert find_spec("src.features.nhl.goalies") is not None
    assert find_spec("src.features.nhl.rink_effects") is not None
    assert find_spec("src.features.nhl.special_teams") is not None

    assert find_spec("src.features.goalie_features") is None
    assert find_spec("src.features.rink_adjustments") is None
    assert find_spec("src.features.special_teams") is None
