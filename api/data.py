"""
Loads pre-computed CSVs once at startup and exposes them as a module-level singleton.
Routers import `app_data` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


@dataclass
class AppData:
    game_agg: pd.DataFrame   # 141 rows - one per game
    joined: pd.DataFrame     # 10,881 rows - one per speaker turn matched to a game


app_data: AppData | None = None


def load() -> AppData:
    """Load both CSVs and return an AppData instance. Called once during lifespan."""
    game_agg = pd.read_csv(_DATA_DIR / "game_sentiment_agg.csv", parse_dates=["date"])
    joined = pd.read_csv(_DATA_DIR / "sentiment_game_joined.csv", parse_dates=["date"])

    n_games = len(game_agg)
    n_turns = len(joined)
    print(f"Data loaded: {n_games} games, {n_turns} turns")

    return AppData(game_agg=game_agg, joined=joined)
