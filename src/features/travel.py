from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

from src.common.time import parse_iso


@dataclass(frozen=True)
class TeamCity:
    lat: float
    lon: float
    utc_offset_hours: int


NHL_TEAM_CITY = {
    "ANA": TeamCity(33.8078, -117.8767, -8),
    "ARI": TeamCity(33.5312, -112.2617, -7),
    "BOS": TeamCity(42.3662, -71.0621, -5),
    "BUF": TeamCity(42.8740, -78.8768, -5),
    "CAR": TeamCity(35.8033, -78.7218, -5),
    "CBJ": TeamCity(39.9690, -83.0063, -5),
    "CGY": TeamCity(51.0374, -114.0519, -7),
    "CHI": TeamCity(41.8807, -87.6742, -6),
    "COL": TeamCity(39.7487, -105.0077, -7),
    "DAL": TeamCity(32.7905, -96.8103, -6),
    "DET": TeamCity(42.3410, -83.0551, -5),
    "EDM": TeamCity(53.5461, -113.4970, -7),
    "FLA": TeamCity(26.1585, -80.3256, -5),
    "LAK": TeamCity(34.0430, -118.2673, -8),
    "MIN": TeamCity(44.9448, -93.1010, -6),
    "MTL": TeamCity(45.4960, -73.5693, -5),
    "NJD": TeamCity(40.7335, -74.1711, -5),
    "NSH": TeamCity(36.1591, -86.7785, -6),
    "NYI": TeamCity(40.7229, -73.5909, -5),
    "NYR": TeamCity(40.7505, -73.9934, -5),
    "OTT": TeamCity(45.2969, -75.9272, -5),
    "PHI": TeamCity(39.9012, -75.1720, -5),
    "PIT": TeamCity(40.4392, -79.9899, -5),
    "SEA": TeamCity(47.6221, -122.3540, -8),
    "SJS": TeamCity(37.3328, -121.9010, -8),
    "STL": TeamCity(38.6268, -90.2026, -6),
    "TBL": TeamCity(27.9427, -82.4518, -5),
    "TOR": TeamCity(43.6435, -79.3791, -5),
    "UTA": TeamCity(40.7683, -111.9012, -7),
    "VAN": TeamCity(49.2777, -123.1089, -8),
    "VGK": TeamCity(36.1020, -115.1783, -8),
    "WPG": TeamCity(49.8927, -97.1436, -6),
    "WSH": TeamCity(38.8981, -77.0209, -5),
}

NBA_TEAM_CITY = {
    "ATL": TeamCity(33.7573, -84.3963, -5),
    "BOS": TeamCity(42.3663, -71.0622, -5),
    "BKN": TeamCity(40.6827, -73.9751, -5),
    "BRK": TeamCity(40.6827, -73.9751, -5),
    "CHA": TeamCity(35.2251, -80.8392, -5),
    "CHI": TeamCity(41.8807, -87.6742, -6),
    "CLE": TeamCity(41.4965, -81.6882, -5),
    "DAL": TeamCity(32.7905, -96.8103, -6),
    "DEN": TeamCity(39.7487, -105.0077, -7),
    "DET": TeamCity(42.3410, -83.0551, -5),
    "GSW": TeamCity(37.7680, -122.3877, -8),
    "GS": TeamCity(37.7680, -122.3877, -8),
    "HOU": TeamCity(29.7508, -95.3621, -6),
    "IND": TeamCity(39.7639, -86.1555, -5),
    "LAC": TeamCity(34.0430, -118.2673, -8),
    "LAL": TeamCity(34.0430, -118.2673, -8),
    "MEM": TeamCity(35.1382, -90.0505, -6),
    "MIA": TeamCity(25.7814, -80.1870, -5),
    "MIL": TeamCity(43.0451, -87.9172, -6),
    "MIN": TeamCity(44.9795, -93.2760, -6),
    "NOP": TeamCity(29.9490, -90.0821, -6),
    "NO": TeamCity(29.9490, -90.0821, -6),
    "NYK": TeamCity(40.7505, -73.9934, -5),
    "NY": TeamCity(40.7505, -73.9934, -5),
    "OKC": TeamCity(35.4634, -97.5151, -6),
    "ORL": TeamCity(28.5392, -81.3839, -5),
    "PHI": TeamCity(39.9012, -75.1720, -5),
    "PHX": TeamCity(33.4457, -112.0712, -7),
    "PHO": TeamCity(33.4457, -112.0712, -7),
    "POR": TeamCity(45.5316, -122.6668, -8),
    "SAC": TeamCity(38.5806, -121.4996, -8),
    "SAS": TeamCity(29.4270, -98.4375, -6),
    "SA": TeamCity(29.4270, -98.4375, -6),
    "TOR": TeamCity(43.6435, -79.3791, -5),
    "UTA": TeamCity(40.7683, -111.9012, -7),
    "UTAH": TeamCity(40.7683, -111.9012, -7),
    "WAS": TeamCity(38.8981, -77.0209, -5),
    "WSH": TeamCity(38.8981, -77.0209, -5),
}

MLB_TEAM_CITY = {
    "ARI": TeamCity(33.4484, -112.0740, -7),
    "AZ": TeamCity(33.4484, -112.0740, -7),
    "ATL": TeamCity(33.7490, -84.3880, -5),
    "BAL": TeamCity(39.2904, -76.6122, -5),
    "BOS": TeamCity(42.3601, -71.0589, -5),
    "CHC": TeamCity(41.8781, -87.6298, -6),
    "CHW": TeamCity(41.8781, -87.6298, -6),
    "CWS": TeamCity(41.8781, -87.6298, -6),
    "CIN": TeamCity(39.1031, -84.5120, -5),
    "CLE": TeamCity(41.4993, -81.6944, -5),
    "COL": TeamCity(39.7392, -104.9903, -7),
    "DET": TeamCity(42.3314, -83.0458, -5),
    "HOU": TeamCity(29.7604, -95.3698, -6),
    "KCR": TeamCity(39.0997, -94.5786, -6),
    "KC": TeamCity(39.0997, -94.5786, -6),
    "LAA": TeamCity(33.8366, -117.9143, -8),
    "ANA": TeamCity(33.8366, -117.9143, -8),
    "LAD": TeamCity(34.0635, -118.3580, -8),
    "MIA": TeamCity(25.7617, -80.1918, -5),
    "FLA": TeamCity(25.7617, -80.1918, -5),
    "MIL": TeamCity(43.0389, -87.9065, -6),
    "MIN": TeamCity(44.9778, -93.2650, -6),
    "NYM": TeamCity(40.7128, -74.0060, -5),
    "NYY": TeamCity(40.7128, -74.0060, -5),
    "OAK": TeamCity(38.5816, -121.4944, -8),
    "ATH": TeamCity(38.5816, -121.4944, -8),
    "PHI": TeamCity(39.9526, -75.1652, -5),
    "PIT": TeamCity(40.4406, -79.9959, -5),
    "SD": TeamCity(32.7157, -117.1611, -8),
    "SDP": TeamCity(32.7157, -117.1611, -8),
    "SEA": TeamCity(47.6062, -122.3321, -8),
    "SF": TeamCity(37.7749, -122.4194, -8),
    "SFG": TeamCity(37.7749, -122.4194, -8),
    "STL": TeamCity(38.6270, -90.1994, -6),
    "TB": TeamCity(27.9506, -82.4572, -5),
    "TBR": TeamCity(27.9506, -82.4572, -5),
    "TEX": TeamCity(32.7357, -97.1081, -6),
    "TOR": TeamCity(43.6532, -79.3832, -5),
    "WSH": TeamCity(38.9072, -77.0369, -5),
    "WSN": TeamCity(38.9072, -77.0369, -5),
}

TEAM_CITY_BY_LEAGUE = {
    "MLB": MLB_TEAM_CITY,
    "NHL": NHL_TEAM_CITY,
    "NBA": NBA_TEAM_CITY,
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.7613
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def _rolling_load(flags: list[int], window: int) -> list[int]:
    arr = np.array(flags, dtype=int)
    out = []
    for i in range(len(arr)):
        lo = max(0, i - window)
        out.append(int(arr[lo:i].sum()))
    return out


def build_travel_features(games_df: pd.DataFrame, league: str = "NHL") -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    league_code = str(league or "NHL").strip().upper()
    team_city = TEAM_CITY_BY_LEAGUE.get(league_code, NHL_TEAM_CITY)
    games = games_df.sort_values("start_time_utc").copy()
    team_rows = []
    for _, r in games.iterrows():
        venue_city = team_city.get(r["home_team"])
        for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
            team = r[team_col]
            opp = r["away_team"] if prefix == "home" else r["home_team"]
            team_rows.append(
                {
                    "game_id": r["game_id"],
                    "team": team,
                    "opp": opp,
                    "prefix": prefix,
                    "start_time_utc": r["start_time_utc"],
                    "lat": venue_city.lat if venue_city else np.nan,
                    "lon": venue_city.lon if venue_city else np.nan,
                    "tz": venue_city.utc_offset_hours if venue_city else np.nan,
                }
            )

    tdf = pd.DataFrame(team_rows).sort_values(["team", "start_time_utc", "game_id"]).reset_index(drop=True)
    by_team_frames = []
    for _, grp in tdf.groupby("team", sort=False):
        g = grp.copy().sort_values(["start_time_utc", "game_id"]).reset_index(drop=True)
        g["start_dt"] = g["start_time_utc"].map(parse_iso)
        g["prev_start_dt"] = g["start_dt"].shift(1)
        g["rest_days"] = (g["start_dt"] - g["prev_start_dt"]).dt.total_seconds() / 86400
        g["rest_days"] = g["rest_days"].fillna(7).clip(lower=0)
        g["b2b"] = (g["rest_days"] <= 1.1).astype(int)
        b2b_list = g["b2b"].tolist()
        g["gms_3in4"] = [int(x >= 2) for x in _rolling_load(b2b_list, 3)]
        g["gms_4in6"] = [int(x >= 3) for x in _rolling_load(b2b_list, 5)]
        g["prev_lat"] = g["lat"].shift(1)
        g["prev_lon"] = g["lon"].shift(1)
        g["travel_miles"] = [
            0.0
            if np.isnan(pl) or np.isnan(po) or np.isnan(cl) or np.isnan(co)
            else haversine_miles(pl, po, cl, co)
            for pl, po, cl, co in zip(g["prev_lat"], g["prev_lon"], g["lat"], g["lon"])
        ]
        g["travel_miles"] = g["travel_miles"].fillna(0.0)
        g["prev_tz"] = g["tz"].shift(1)
        g["tz_change"] = (g["tz"] - g["prev_tz"]).fillna(0)
        g["utc_hour"] = g["start_dt"].dt.hour
        g["local_hour"] = g["utc_hour"] + g["tz"]
        g["local_start_mismatch"] = ((g["local_hour"] < 15) | (g["local_hour"] > 22)).astype(int)
        by_team_frames.append(g)

    feat = pd.concat(by_team_frames, ignore_index=True)
    rows = []
    for prefix in ("home", "away"):
        prefix_frame = feat[feat["prefix"] == prefix][
            [
                "game_id",
                "rest_days",
                "b2b",
                "gms_3in4",
                "gms_4in6",
                "travel_miles",
                "tz_change",
                "local_start_mismatch",
            ]
        ].rename(
            columns={
                "rest_days": f"{prefix}_rest_days",
                "b2b": f"{prefix}_b2b",
                "gms_3in4": f"{prefix}_3in4",
                "gms_4in6": f"{prefix}_4in6",
                "travel_miles": f"{prefix}_travel_miles",
                "tz_change": f"{prefix}_tz_change",
                "local_start_mismatch": f"{prefix}_local_start_mismatch",
            }
        )
        rows.append(prefix_frame)

    merged = rows[0].merge(rows[1], on="game_id", how="outer")
    merged["travel_diff"] = merged["home_travel_miles"] - merged["away_travel_miles"]
    merged["rest_diff"] = merged["home_rest_days"] - merged["away_rest_days"]
    return merged
