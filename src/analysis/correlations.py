"""
Join sentiment scores with game outcomes and run correlation analyses.

Four analyses:
  1. Pearson correlation: game-day aggregate sentiment vs point differential
  2. Pearson correlation: game-day sentiment vs NEXT game point differential
  3. Series trajectory: average sentiment by game number (1-7)
  4. Elimination game sentiment: is aggregate sentiment more negative in series-ending games?

Usage:
    python -m src.analysis.correlations
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from src.scraper.game_data import load_playoff_games

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Team nickname -> abbreviation map
# Covers all tokens that appear in ASAP Sports event_name strings
# ---------------------------------------------------------------------------
NICKNAME_TO_ABBR: dict[str, str] = {
    # Direct nickname matches (from teams.csv NICKNAME column)
    "HAWKS":        "ATL",
    "CELTICS":      "BOS",
    "BULLS":        "CHI",
    "CAVALIERS":    "CLE",
    "MAVERICKS":    "DAL",
    "NUGGETS":      "DEN",
    "PISTONS":      "DET",
    "WARRIORS":     "GSW",
    "ROCKETS":      "HOU",
    "PACERS":       "IND",
    "CLIPPERS":     "LAC",
    "LAKERS":       "LAL",
    "HEAT":         "MIA",
    "BUCKS":        "MIL",
    "GRIZZLIES":    "MEM",
    "NETS":         "BKN",
    "KNICKS":       "NYK",
    "MAGIC":        "ORL",
    "76ERS":        "PHI",
    "SUNS":         "PHX",
    "TRAIL BLAZERS": "POR",
    "KINGS":        "SAC",
    "SPURS":        "SAS",
    "THUNDER":      "OKC",
    "RAPTORS":      "TOR",
    "JAZZ":         "UTA",
    "WIZARDS":      "WAS",
    "HORNETS":      "CHA",
    # City / full-name variants used in ASAP Sports event strings
    "GOLDEN STATE": "GSW",
    "CLEVELAND":    "CLE",
    "TORONTO":      "TOR",
    "OKC":          "OKC",
}


# ---------------------------------------------------------------------------
# Event name parsing
# ---------------------------------------------------------------------------

def _parse_teams_from_event(event_name: str) -> tuple[str, str] | None:
    """
    Parse the two team abbreviations from an event_name string.

    Examples:
      "NBA FINALS: CELTICS VS WARRIORS"     -> ("BOS", "GSW")
      "NBA WCF: GOLDEN STATE vs THUNDER"    -> ("GSW", "OKC")
      "NBA FINALS: CAVALIERS v WARRIORS (a)"-> ("CLE", "GSW")

    Returns None if either team token cannot be mapped.
    """
    after_colon = event_name.split(":")[-1].strip()
    # Strip trailing annotations like "(a)", "(b)"
    after_colon = re.sub(r"\s*\([a-zA-Z]\)\s*$", "", after_colon).strip()
    # Split on VS / vs / V / v (with surrounding spaces)
    parts = re.split(r"\s+[Vv][Ss]?\s+", after_colon, maxsplit=1)
    if len(parts) != 2:
        return None

    t1_raw = parts[0].strip().upper()
    t2_raw = parts[1].strip().upper()

    a1 = NICKNAME_TO_ABBR.get(t1_raw)
    a2 = NICKNAME_TO_ABBR.get(t2_raw)

    if a1 is None or a2 is None:
        return None
    return a1, a2


# ---------------------------------------------------------------------------
# Game enrichment: series IDs and game numbers
# ---------------------------------------------------------------------------

def _enrich_games(games: pd.DataFrame) -> pd.DataFrame:
    """Add series_id and game_num columns to the games DataFrame."""
    games = games.sort_values("date").copy()
    games["team_pair"] = games.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )
    games["series_id"] = (
        games["team_pair"].astype(str) + "_" + games["season"].astype(str)
    )
    games["game_num"] = games.groupby("series_id").cumcount() + 1

    # Series length (determines which game is the elimination game)
    series_len = games.groupby("series_id")["game_num"].transform("max")
    games["is_elimination_game"] = (games["game_num"] == series_len).astype(int)

    # Cumulative wins for home and away teams within series
    games["home_series_wins"] = games.groupby("series_id")["home_win"].cumsum()
    games["away_series_wins"] = games.groupby("series_id").apply(
        lambda g: (1 - g["home_win"]).cumsum(), include_groups=False
    ).reset_index(level=0, drop=True)

    # Next-game point differential within series (from home team perspective)
    games["next_point_diff"] = games.groupby("series_id")["point_diff"].shift(-1)

    return games


# ---------------------------------------------------------------------------
# Join sentiment scores to games
# ---------------------------------------------------------------------------

def join_with_games(
    scores: pd.DataFrame | None = None,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join game-day transcript sentiment to game outcomes.

    For each unique (date, event_name) in `scores`, find the matching game
    on that date where the home/away team pair matches the event teams.

    Returns a DataFrame with all sentiment_scores columns plus game metadata.
    Only game-day transcripts are returned (off-day press conferences are dropped).
    """
    if scores is None:
        scores = pd.read_csv(PROCESSED_DIR / "sentiment_scores.csv", parse_dates=["date"])
    if games is None:
        games = _enrich_games(load_playoff_games())

    # Build a lookup: (date, frozenset{team1, team2}) -> game row
    game_lookup: dict[tuple, dict] = {}
    for _, row in games.iterrows():
        key = (row["date"].date(), frozenset([row["home_team"], row["away_team"]]))
        game_lookup[key] = row.to_dict()

    # For each score row, attempt to join
    game_cols = [
        "game_id", "season", "home_team", "away_team",
        "home_pts", "away_pts", "home_win", "point_diff",
        "series_id", "game_num", "is_elimination_game",
        "home_series_wins", "away_series_wins", "next_point_diff",
    ]

    joined_rows = []
    unmatched = 0

    for _, row in scores.iterrows():
        if pd.isna(row["date"]):
            unmatched += 1
            continue
        teams = _parse_teams_from_event(row["event_name"])
        if teams is None:
            unmatched += 1
            continue
        key = (pd.Timestamp(row["date"]).date(), frozenset(teams))
        game = game_lookup.get(key)
        if game is None:
            # Off-day press conference - no game on this date
            unmatched += 1
            continue
        new_row = row.to_dict()
        for col in game_cols:
            new_row[col] = game.get(col)
        joined_rows.append(new_row)

    joined = pd.DataFrame(joined_rows)
    print(f"Joined: {len(joined):,} turns matched to games")
    print(f"Unmatched (off-day or parse failure): {unmatched:,}")
    return joined


# ---------------------------------------------------------------------------
# Game-level aggregates
# ---------------------------------------------------------------------------

def build_game_aggregates(joined: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment to one row per game (all speakers, both teams combined).

    Columns added:
        mean_sentiment   - mean sentiment_numeric across all turns for that game
        pct_positive     - fraction of turns labeled POSITIVE
        pct_negative     - fraction of turns labeled NEGATIVE
        n_turns          - number of turns
    """
    agg = (
        joined.groupby(["game_id", "date", "event_name", "series_id", "game_num",
                        "home_team", "away_team", "home_win", "point_diff",
                        "is_elimination_game", "next_point_diff",
                        "home_series_wins", "away_series_wins"])
        .agg(
            mean_sentiment=("sentiment_numeric", "mean"),
            pct_positive=("sentiment_label", lambda x: (x == "POSITIVE").mean()),
            pct_negative=("sentiment_label", lambda x: (x == "NEGATIVE").mean()),
            n_turns=("sentiment_numeric", "count"),
        )
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Analysis 1: Pearson correlations
# ---------------------------------------------------------------------------

def pearson_analysis(game_agg: pd.DataFrame) -> dict:
    """
    Pearson correlations between game-day aggregate sentiment and game metrics.

    Returns a dict with correlation results.
    """
    results = {}

    pairs = [
        ("mean_sentiment", "point_diff",       "sentiment vs point_diff (same game)"),
        ("mean_sentiment", "next_point_diff",   "sentiment vs next-game point_diff"),
        ("pct_positive",   "point_diff",        "pct_positive vs point_diff"),
        ("pct_negative",   "point_diff",        "pct_negative vs point_diff"),
    ]

    for x_col, y_col, label in pairs:
        mask = game_agg[[x_col, y_col]].notna().all(axis=1)
        x = game_agg.loc[mask, x_col]
        y = game_agg.loc[mask, y_col]
        r, p = stats.pearsonr(x, y)
        results[label] = {"r": round(r, 3), "p": round(p, 4), "n": int(mask.sum())}

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Trajectory by game number
# ---------------------------------------------------------------------------

def trajectory_analysis(joined: pd.DataFrame) -> pd.DataFrame:
    """
    Average sentiment_numeric by game number within series (1-7).

    Returns a DataFrame with game_num, mean_sentiment, std_sentiment, n_turns.
    """
    traj = (
        joined.groupby("game_num")["sentiment_numeric"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_sentiment", "std": "std_sentiment", "count": "n_turns"})
        .reset_index()
    )
    return traj


# ---------------------------------------------------------------------------
# Analysis 3: Elimination game sentiment
# ---------------------------------------------------------------------------

def elimination_analysis(game_agg: pd.DataFrame) -> dict:
    """
    Compare aggregate sentiment on elimination games vs non-elimination games.

    Also runs logistic regression: mean_sentiment -> is_elimination_game.
    """
    elim   = game_agg[game_agg["is_elimination_game"] == 1]["mean_sentiment"]
    normal = game_agg[game_agg["is_elimination_game"] == 0]["mean_sentiment"]

    # Elimination games are missing from matched transcript data (data coverage gap -
    # last game of each series was not captured by the scraper for these series).
    if len(elim) == 0:
        print("  WARNING: No elimination games in matched data - skipping t-test and logistic regression.")
        return {
            "elim_mean":             float("nan"),
            "normal_mean":           round(float(normal.mean()), 3),
            "t_stat":                float("nan"),
            "p_value":               float("nan"),
            "n_elim":                0,
            "n_normal":              int(len(normal)),
            "logit_coef":            float("nan"),
            "classification_report": "No elimination games in matched transcript data.",
        }

    t_stat, p_val = stats.ttest_ind(elim, normal)

    # Logistic regression - only possible when both classes are present
    X = game_agg[["mean_sentiment"]].dropna()
    y = game_agg.loc[X.index, "is_elimination_game"]

    if y.nunique() < 2:
        return {
            "elim_mean":             round(float(elim.mean()), 3),
            "normal_mean":           round(float(normal.mean()), 3),
            "t_stat":                round(float(t_stat), 3),
            "p_value":               round(float(p_val), 4),
            "n_elim":                int(len(elim)),
            "n_normal":              int(len(normal)),
            "logit_coef":            float("nan"),
            "classification_report": "Only one class present - logistic regression not applicable.",
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)

    return {
        "elim_mean":             round(float(elim.mean()), 3),
        "normal_mean":           round(float(normal.mean()), 3),
        "t_stat":                round(float(t_stat), 3),
        "p_value":               round(float(p_val), 4),
        "n_elim":                int(len(elim)),
        "n_normal":              int(len(normal)),
        "logit_coef":            round(float(clf.coef_[0][0]), 3),
        "classification_report": classification_report(y, y_pred),
    }


# ---------------------------------------------------------------------------
# Analysis 4: Sentiment by series position (leading vs trailing)
# ---------------------------------------------------------------------------

def series_position_analysis(joined: pd.DataFrame) -> pd.DataFrame:
    """
    For each speaker turn, determine if the speaker's team is in the series lead.

    Since we don't have speaker-team attribution, we use a proxy:
    aggregate sentiment for games where the home team leads the series
    vs games where the away team leads.

    Returns a DataFrame comparing sentiment by lead status.
    """
    df = joined.copy()

    # Home team series lead status before this game
    df["home_leads"] = df["home_series_wins"] > df["away_series_wins"]
    df["tied"]       = df["home_series_wins"] == df["away_series_wins"]

    traj = (
        df.groupby(["game_num", "home_leads"])["sentiment_numeric"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_sentiment", "count": "n_turns"})
        .reset_index()
    )
    return traj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(save: bool = True) -> dict:
    """Run all four analyses and return results dict."""
    print("Loading sentiment scores...")
    scores = pd.read_csv(PROCESSED_DIR / "sentiment_scores.csv", parse_dates=["date"])
    print(f"  {len(scores):,} scored turns")

    print("\nJoining with game data...")
    joined = join_with_games(scores)

    print("\nBuilding game-level aggregates...")
    game_agg = build_game_aggregates(joined)
    print(f"  {len(game_agg)} games with press conference data")

    if save:
        joined.to_csv(PROCESSED_DIR / "sentiment_game_joined.csv", index=False)
        game_agg.to_csv(PROCESSED_DIR / "game_sentiment_agg.csv", index=False)
        print(f"  Saved sentiment_game_joined.csv and game_sentiment_agg.csv")

    print("\n--- Analysis 1: Pearson correlations ---")
    pearson = pearson_analysis(game_agg)
    for label, res in pearson.items():
        print(f"  {label}: r={res['r']}, p={res['p']}, n={res['n']}")

    print("\n--- Analysis 2: Series trajectory ---")
    traj = trajectory_analysis(joined)
    print(traj.to_string(index=False))

    print("\n--- Analysis 3: Elimination game sentiment ---")
    elim = elimination_analysis(game_agg)
    print(f"  Elimination games:     mean sentiment = {elim['elim_mean']}")
    print(f"  Non-elimination games: mean sentiment = {elim['normal_mean']}")
    print(f"  t-test: t={elim['t_stat']}, p={elim['p_value']}")

    print("\n--- Analysis 4: Series position ---")
    pos = series_position_analysis(joined)
    print(pos.to_string(index=False))

    return {
        "joined":       joined,
        "game_agg":     game_agg,
        "pearson":      pearson,
        "trajectory":   traj,
        "elimination":  elim,
        "series_pos":   pos,
    }


if __name__ == "__main__":
    run_all(save=True)
