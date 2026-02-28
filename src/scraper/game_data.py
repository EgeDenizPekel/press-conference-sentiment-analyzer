"""
Loads Kaggle NBA game logs and returns a clean DataFrame of playoff games
for the seasons covered by our ASAP Sports transcript corpus (2012-2021).

Kaggle SEASON encoding: the year the season started (e.g., SEASON='2013' = 2013-14 season).
Playoff game filter: GAME_ID[0] == '4' (NBA encoding: 4=playoffs, 2=regular season).
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "kaggle"

# Kaggle SEASONS corresponding to ASAP Sports years 2013-2022.
# ASAP year = Kaggle season + 1 (Finals happen in June of the following calendar year).
# Exception: COVID seasons (2019 bubble = Oct 2020, 2020 bubble = July 2021).
COVERED_SEASONS = list(range(2012, 2022))  # 2012-2021 inclusive


def load_playoff_games() -> pd.DataFrame:
    """
    Returns a DataFrame of NBA playoff games for covered seasons, with columns:
        game_id, date, season, home_team, away_team,
        home_pts, away_pts, home_win, point_diff
    """
    games = pd.read_csv(DATA_DIR / "games.csv", parse_dates=["GAME_DATE_EST"])
    teams = pd.read_csv(DATA_DIR / "teams.csv")

    # Build team_id -> abbreviation map
    team_abbr = teams.set_index("TEAM_ID")["ABBREVIATION"].to_dict()

    # Filter to playoff games
    playoffs = games[games["GAME_ID"].astype(str).str[0] == "4"].copy()

    # Filter to covered seasons
    playoffs = playoffs[playoffs["SEASON"].isin(COVERED_SEASONS)].copy()

    # Compute point differential (from home team perspective)
    playoffs["point_diff"] = playoffs["PTS_home"] - playoffs["PTS_away"]

    # Map team IDs to abbreviations
    playoffs["home_team"] = playoffs["HOME_TEAM_ID"].map(team_abbr)
    playoffs["away_team"] = playoffs["VISITOR_TEAM_ID"].map(team_abbr)

    result = playoffs.rename(columns={
        "GAME_ID": "game_id",
        "GAME_DATE_EST": "date",
        "SEASON": "season",
        "PTS_home": "home_pts",
        "PTS_away": "away_pts",
        "HOME_TEAM_WINS": "home_win",
    })[[
        "game_id", "date", "season",
        "home_team", "away_team",
        "home_pts", "away_pts", "home_win", "point_diff",
    ]]

    return result.reset_index(drop=True)


if __name__ == "__main__":
    df = load_playoff_games()
    print(f"Playoff games loaded: {len(df)}")
    print(f"Seasons covered: {sorted(df['season'].unique())}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.head())
