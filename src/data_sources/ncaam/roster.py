from __future__ import annotations

import pandas as pd

from src.data_sources.base import HttpClient

ESPN_NCAAM_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams"
ESPN_NCAAM_ROSTER_URL_TMPL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/roster"


def _team_id_map(client: HttpClient, season: str) -> dict[str, str]:
    payload, _ = client.get_json("ncaam_teams", ESPN_NCAAM_TEAMS_URL, params={"limit": 500}, key=f"teams_{season}")
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


def fetch_roster_snapshot(
    client: HttpClient,
    team_abbrevs: list[str],
    season: int | str,
) -> tuple[pd.DataFrame, list[str]]:
    season_str = str(season)
    rows: list[dict] = []
    raw_paths: list[str] = []
    team_map = _team_id_map(client, season=season_str)

    for team in sorted(set(str(t).upper() for t in team_abbrevs if t)):
        team_id = team_map.get(team)
        if not team_id:
            continue

        url = ESPN_NCAAM_ROSTER_URL_TMPL.format(team_id=team_id)
        try:
            payload, raw_path = client.get_json("ncaam_rosters", url, key=f"{team}_{season_str}")
            raw_paths.append(raw_path)
        except Exception:
            continue

        athletes = payload.get("athletes") or []
        for player in athletes:
            rows.append(
                {
                    "season": season_str,
                    "team": team,
                    "player_id": str(player.get("id") or "").strip() or None,
                    "player_name": player.get("fullName") or player.get("displayName"),
                    "position": (player.get("position") or {}).get("abbreviation"),
                    "jersey": player.get("jersey"),
                    "status": ((player.get("status") or {}).get("type")) or ((player.get("status") or {}).get("abbreviation")),
                    "injury_status": None,
                    "injury_date": None,
                }
            )

    return pd.DataFrame(rows), raw_paths
