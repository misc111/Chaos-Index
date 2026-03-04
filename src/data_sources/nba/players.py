from __future__ import annotations

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult

ESPN_NBA_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ESPN_NBA_ROSTER_URL_TMPL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"



def _team_id_map(client: HttpClient, season: str) -> dict[str, str]:
    payload, _ = client.get_json("nba_teams", ESPN_NBA_TEAMS_URL, key=f"teams_{season}")
    mapping: dict[str, str] = {}
    leagues = (((payload.get("sports") or [{}])[0].get("leagues") or [{}]))
    teams = (leagues[0].get("teams") if leagues else []) or []
    for row in teams:
        team = row.get("team") or {}
        abbr = str(team.get("abbreviation") or "").upper()
        team_id = str(team.get("id") or "").strip()
        if abbr and team_id:
            mapping[abbr] = team_id
    return mapping



def fetch_players(client: HttpClient, team_abbrevs: list[str], season: int | str) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    season_str = str(season)
    rows: list[dict] = []
    raw_paths: list[str] = []

    team_map = _team_id_map(client, season=season_str)

    for team in sorted(set(str(t).upper() for t in team_abbrevs if t)):
        team_id = team_map.get(team)
        if not team_id:
            continue

        url = ESPN_NBA_ROSTER_URL_TMPL.format(team_id=team_id)
        try:
            payload, raw_path = client.get_json("nba_players", url, key=f"{team}_{season_str}")
            raw_paths.append(raw_path)
        except Exception:
            continue

        for player in payload.get("athletes", []):
            rows.append(
                {
                    "season": season_str,
                    "team": team,
                    "player_id": player.get("id"),
                    "position": (player.get("position") or {}).get("abbreviation"),
                    "games_played": None,
                    "goals": None,
                    "assists": None,
                    "points": None,
                    "shots": None,
                    "shooting_pctg": None,
                    "faceoff_win_pctg": None,
                    "penalty_minutes": None,
                    "power_play_goals": None,
                    "shorthanded_goals": None,
                    "game_winning_goals": None,
                    "overtime_goals": None,
                    "avg_shifts_per_game": None,
                    "avg_time_on_ice_per_game": None,
                    "toi_per_game": None,
                    "toi_per_game_minutes": None,
                    "plus_minus": None,
                }
            )

    df = pd.DataFrame(rows)
    metadata = {
        "season": season_str,
        "n_rows": int(len(df)),
        "n_teams": int(len(set(team_abbrevs))),
        "fetched_at_utc": as_of_utc,
        "fallback_used": int(df.empty),
    }
    snapshot_id = client.snapshot_id("nba_players", metadata)
    return SourceFetchResult(
        source="nba_players",
        snapshot_id=snapshot_id,
        extracted_at_utc=as_of_utc,
        raw_path=raw_paths[-1] if raw_paths else "",
        metadata=metadata,
        dataframe=df,
    )
