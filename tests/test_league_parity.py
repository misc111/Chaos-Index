import pytest

from src.league_registry import get_league_adapter, supported_leagues
from src.orchestration.refresh_pipeline import build_data_refresh_steps


def test_mlb_adapter_contract_is_the_only_supported_league() -> None:
    adapter = get_league_adapter("MLB")

    assert adapter.code == "MLB"
    assert adapter.metadata.slug == "mlb"
    assert adapter.metadata.default_config_path == "configs/mlb.yaml"
    assert adapter.metadata.championship_name == "World Series"
    assert adapter.metadata.championship_probability_key == "world_series_prob"

    for attr in (
        "fetch_games",
        "fetch_team_game_stats",
        "fetch_starter_context",
        "fetch_injuries_report",
        "fetch_public_odds_optional",
        "fetch_players",
        "build_results_from_games",
        "fetch_upcoming_schedule",
        "fetch_teams",
        "fetch_context_metrics_optional",
    ):
        assert callable(getattr(adapter, attr))
    assert not hasattr(adapter, "fetch_goalie_game_stats")
    assert not hasattr(adapter, "fetch_xg_optional")


def test_supported_leagues_are_explicit_and_stable() -> None:
    assert supported_leagues() == ("MLB",)


@pytest.mark.parametrize("league", ["MLS"])
def test_non_mlb_league_adapter_is_not_supported(league: str) -> None:
    with pytest.raises(ValueError, match="Unsupported league"):
        get_league_adapter(league)


def test_data_refresh_steps_cover_mlb_first_rebuild_lane() -> None:
    steps = {step.name: step for step in build_data_refresh_steps()}

    assert tuple(steps) == ("mlb:fetch", "mlb:fetch-odds")
    assert steps["mlb:fetch"].command[-1] == "configs/mlb.yaml"
    assert steps["mlb:fetch-odds"].command[-1] == "configs/mlb.yaml"
