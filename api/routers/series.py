"""
/series and /series/{series_id} endpoints.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Path

from api import data as data_module

router = APIRouter()


def _parse_series_id(series_id: str) -> tuple[list[str], int]:
    """
    Parse series_id like "('BOS', 'MIA')_2011" into (["BOS","MIA"], 2012).
    Kaggle season is start-year; display as season+1.
    """
    m = re.match(r"\('([A-Z]+)',\s*'([A-Z]+)'\)_(\d+)", series_id)
    if not m:
        return ([], 0)
    team1, team2, season_str = m.group(1), m.group(2), m.group(3)
    return ([team1, team2], int(season_str) + 1)


def _derive_round(event_name: str) -> str:
    """Return ECF / WCF / FINALS from event_name string."""
    upper = event_name.upper()
    if "FINALS" in upper and ("ECF" not in upper and "WCF" not in upper):
        return "FINALS"
    if "ECF" in upper:
        return "ECF"
    if "WCF" in upper:
        return "WCF"
    return "PLAYOFFS"


@router.get("")
def list_series():
    """List all series with readable labels, sorted by season desc."""
    game_agg = data_module.app_data.game_agg

    # One row per series: take the first game's event_name for round derivation
    series_meta = (
        game_agg.groupby("series_id")
        .agg(
            n_games=("game_num", "count"),
            event_name=("event_name", "first"),
        )
        .reset_index()
    )

    result = []
    for _, row in series_meta.iterrows():
        sid = row["series_id"]
        teams, season = _parse_series_id(sid)
        if not teams:
            continue
        label = f"{teams[0]} vs {teams[1]} ({season})"
        rnd = _derive_round(row["event_name"])
        result.append(
            {
                "series_id": sid,
                "label": label,
                "teams": teams,
                "season": season,
                "n_games": int(row["n_games"]),
                "round": rnd,
            }
        )

    # Sort by season desc, then by series_id for stable ordering
    result.sort(key=lambda x: (-x["season"], x["series_id"]))
    return result


@router.get("/{series_id:path}")
def get_series(series_id: str = Path(...)):
    """Game-by-game sentiment arc for a single series."""
    game_agg = data_module.app_data.game_agg

    games = game_agg[game_agg["series_id"] == series_id].copy()
    if games.empty:
        raise HTTPException(status_code=404, detail=f"Series '{series_id}' not found")

    games = games.sort_values("game_num")
    teams, season = _parse_series_id(series_id)
    label = f"{teams[0]} vs {teams[1]} ({season})" if teams else series_id
    rnd = _derive_round(games.iloc[0]["event_name"])

    game_list = []
    for _, row in games.iterrows():
        game_list.append(
            {
                "game_num": int(row["game_num"]),
                "date": str(row["date"].date()),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win": int(row["home_win"]),
                "point_diff": float(row["point_diff"]) if row["point_diff"] == row["point_diff"] else None,
                "home_series_wins": int(row["home_series_wins"]),
                "away_series_wins": int(row["away_series_wins"]),
                "mean_sentiment": round(float(row["mean_sentiment"]), 4),
                "pct_positive": round(float(row["pct_positive"]), 4),
                "pct_negative": round(float(row["pct_negative"]), 4),
                "n_turns": int(row["n_turns"]),
            }
        )

    return {
        "series_id": series_id,
        "label": label,
        "round": rnd,
        "games": game_list,
    }
