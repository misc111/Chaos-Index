from __future__ import annotations

import pandas as pd

from src.common.time import utc_now_iso
from src.data_sources.base import HttpClient, SourceFetchResult
from src.data_sources.nba.roster import fetch_roster_snapshot

ESPN_NBA_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


def _safe_float(value) -> float | None:
    if value is None or value == "":
        return None
    text = str(value).strip().replace("+", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_minutes(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return _safe_float(minutes) + (_safe_float(seconds) or 0.0) / 60.0
    return _safe_float(text)


def _split_made_attempted(value: str | None) -> tuple[float | None, float | None]:
    text = str(value or "").strip()
    if "-" not in text:
        number = _safe_float(text)
        return number, None
    left, right = text.split("-", 1)
    return _safe_float(left), _safe_float(right)


def _stat_lookup(group: dict) -> dict[str, int]:
    labels = [str(x).strip().upper() for x in group.get("labels") or []]
    return {label: idx for idx, label in enumerate(labels)}


def _stat_value(stats: list[str], mapping: dict[str, int], key: str) -> str | None:
    idx = mapping.get(key)
    if idx is None or idx >= len(stats):
        return None
    return stats[idx]


def _parse_team_players(
    group: dict,
    *,
    game_id: int,
    game_date_utc: str | None,
    start_time_utc: str | None,
    home_team: str | None,
    away_team: str | None,
    current_roster: pd.DataFrame,
) -> list[dict]:
    team = (group.get("team") or {}).get("abbreviation")
    if not team:
        return []

    stats_groups = group.get("statistics") or []
    if not stats_groups:
        return []
    base_group = stats_groups[0] or {}
    mapping = _stat_lookup(base_group)
    athletes = base_group.get("athletes") or []

    roster_lookup = current_roster[current_roster["team"] == team].copy() if not current_roster.empty else pd.DataFrame()
    if not roster_lookup.empty:
        roster_lookup = roster_lookup.drop_duplicates(subset=["player_id"], keep="last").set_index("player_id")

    opponent = away_team if team == home_team else home_team
    is_home = 1 if team == home_team else 0

    rows: list[dict] = []
    for athlete_row in athletes:
        athlete = athlete_row.get("athlete") or {}
        player_id = str(athlete.get("id") or "").strip() or None
        roster_row = roster_lookup.loc[player_id] if player_id and player_id in roster_lookup.index else None
        stat_values = athlete_row.get("stats") or []
        fg_made, fg_att = _split_made_attempted(_stat_value(stat_values, mapping, "FG"))
        three_made, three_att = _split_made_attempted(_stat_value(stat_values, mapping, "3PT"))
        ft_made, ft_att = _split_made_attempted(_stat_value(stat_values, mapping, "FT"))
        minutes = _parse_minutes(_stat_value(stat_values, mapping, "MIN"))
        did_not_play = bool(athlete_row.get("didNotPlay"))
        played = 0 if did_not_play else 1

        rows.append(
            {
                "season": None,
                "team": team,
                "current_team": roster_row["team"] if roster_row is not None else team,
                "player_id": player_id,
                "player_name": athlete.get("displayName") or athlete.get("shortName"),
                "position": (athlete.get("position") or {}).get("abbreviation")
                or (roster_row["position"] if roster_row is not None else None),
                "current_status": roster_row["status"] if roster_row is not None else None,
                "current_injury_status": roster_row["injury_status"] if roster_row is not None else None,
                "current_injury_date": roster_row["injury_date"] if roster_row is not None else None,
                "game_id": int(game_id),
                "game_date_utc": game_date_utc,
                "start_time_utc": start_time_utc,
                "home_team": home_team,
                "away_team": away_team,
                "opponent": opponent,
                "is_home": is_home,
                "played": played,
                "starter": 1 if bool(athlete_row.get("starter")) else 0,
                "minutes": minutes or 0.0,
                "points": _safe_float(_stat_value(stat_values, mapping, "PTS")) or 0.0,
                "assists": _safe_float(_stat_value(stat_values, mapping, "AST")) or 0.0,
                "rebounds_offensive": _safe_float(_stat_value(stat_values, mapping, "OREB")) or 0.0,
                "rebounds_defensive": _safe_float(_stat_value(stat_values, mapping, "DREB")) or 0.0,
                "rebounds_total": _safe_float(_stat_value(stat_values, mapping, "REB")) or 0.0,
                "steals": _safe_float(_stat_value(stat_values, mapping, "STL")) or 0.0,
                "blocks": _safe_float(_stat_value(stat_values, mapping, "BLK")) or 0.0,
                "turnovers": _safe_float(_stat_value(stat_values, mapping, "TO")) or 0.0,
                "fouls_personal": _safe_float(_stat_value(stat_values, mapping, "PF")) or 0.0,
                "field_goals_made": fg_made or 0.0,
                "field_goals_attempted": fg_att or 0.0,
                "free_throws_made": ft_made or 0.0,
                "free_throws_attempted": ft_att or 0.0,
                "three_pointers_made": three_made or 0.0,
                "three_pointers_attempted": three_att or 0.0,
                "plus_minus_points": _safe_float(_stat_value(stat_values, mapping, "+/-")) or 0.0,
            }
        )

    return rows


def fetch_players(
    client: HttpClient,
    team_abbrevs: list[str],
    season: int | str,
    games_df: pd.DataFrame | None = None,
) -> SourceFetchResult:
    as_of_utc = utc_now_iso()
    season_str = str(season)
    rows: list[dict] = []
    raw_paths: list[str] = []
    roster_df, roster_raw_paths = fetch_roster_snapshot(client, team_abbrevs=team_abbrevs, season=season_str)
    raw_paths.extend(roster_raw_paths)

    final_games = pd.DataFrame()
    if games_df is not None and not games_df.empty:
        final_games = games_df[games_df["status_final"] == 1].copy()
        if not final_games.empty:
            final_games["game_id"] = pd.to_numeric(final_games["game_id"], errors="coerce")
            final_games = final_games[final_games["game_id"].notna()]
            final_games = final_games.sort_values("start_time_utc").drop_duplicates(subset=["game_id"])

    for game in final_games.itertuples(index=False):
        game_id = int(game.game_id)
        try:
            payload, raw_path = client.get_json("nba_players", ESPN_NBA_SUMMARY_URL, params={"event": game_id}, key=str(game_id))
            raw_paths.append(raw_path)
        except Exception:
            continue

        boxscore_players = (payload.get("boxscore") or {}).get("players") or []
        for group in boxscore_players:
            rows.extend(
                _parse_team_players(
                    group,
                    game_id=game_id,
                    game_date_utc=getattr(game, "game_date_utc", None),
                    start_time_utc=getattr(game, "start_time_utc", None),
                    home_team=getattr(game, "home_team", None),
                    away_team=getattr(game, "away_team", None),
                    current_roster=roster_df,
                )
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["season"] = season_str
        df = df.sort_values(["start_time_utc", "team", "player_name"]).reset_index(drop=True)

    metadata = {
        "season": season_str,
        "n_rows": int(len(df)),
        "n_teams": int(len(set(team_abbrevs))),
        "n_final_games": int(len(final_games)),
        "n_roster_rows": int(len(roster_df)),
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
