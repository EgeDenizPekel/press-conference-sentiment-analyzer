"""
/analysis/* endpoints.
All data is pre-computed from game_sentiment_agg.csv and sentiment_game_joined.csv.
"""
from __future__ import annotations

import math

import numpy as np
from fastapi import APIRouter
from scipy import stats

from api import data as data_module

router = APIRouter()


@router.get("/summary")
def get_summary():
    """Overview page: corpus stats, sentiment distribution, Pearson results, model comparison."""
    d = data_module.app_data
    game_agg = d.game_agg
    joined = d.joined

    # Corpus stats
    n_series = game_agg["series_id"].nunique()

    # Sentiment distribution from joined (all matched turns)
    pct_positive = round(float((joined["sentiment_label"] == "POSITIVE").mean()), 4)
    pct_negative = round(float((joined["sentiment_label"] == "NEGATIVE").mean()), 4)
    pct_neutral = round(1 - pct_positive - pct_negative, 4)
    mean_numeric = round(float(joined["sentiment_numeric"].mean()), 4)

    # Pearson: same-game point diff
    valid_same = game_agg[game_agg["point_diff"].notna()]
    r_same, p_same = stats.pearsonr(
        valid_same["mean_sentiment"], valid_same["point_diff"]
    )

    # Pearson: next-game point diff
    valid_next = game_agg[game_agg["next_point_diff"].notna()]
    r_next, p_next = stats.pearsonr(
        valid_next["mean_sentiment"], valid_next["next_point_diff"]
    )

    return {
        "corpus": {
            "n_transcripts": 2790,
            "n_turns": 23166,
            "n_games": len(game_agg),
            "n_series": int(n_series),
        },
        "sentiment_dist": {
            "positive": pct_positive,
            "neutral": pct_neutral,
            "negative": pct_negative,
            "mean_numeric": mean_numeric,
        },
        "pearson": [
            {
                "label": "Sentiment vs same-game point diff",
                "r": round(float(r_same), 3),
                "p": round(float(p_same), 3),
                "n": len(valid_same),
            },
            {
                "label": "Sentiment vs next-game point diff",
                "r": round(float(r_next), 3),
                "p": round(float(p_next), 3),
                "n": len(valid_next),
            },
        ],
        "model_comparison": [
            {"model": "Fine-tuned RoBERTa (ours)", "accuracy": 0.92, "macro_f1": 0.932, "ours": True},
            {"model": "Twitter RoBERTa", "accuracy": 0.54, "macro_f1": 0.467, "ours": False},
            {"model": "DistilBERT SST-2", "accuracy": 0.52, "macro_f1": 0.380, "ours": False},
            {"model": "FinBERT", "accuracy": 0.34, "macro_f1": 0.288, "ours": False},
        ],
    }


@router.get("/trajectory")
def get_trajectory():
    """Sentiment arc across game positions 1-7 (aggregated over all series)."""
    joined = data_module.app_data.joined

    grp = (
        joined.groupby("game_num")["sentiment_numeric"]
        .agg(mean_sentiment="mean", std_sentiment="std", n_turns="count")
        .reset_index()
    )

    result = []
    for _, row in grp.iterrows():
        result.append(
            {
                "game_num": int(row["game_num"]),
                "mean_sentiment": round(float(row["mean_sentiment"]), 4),
                "std_sentiment": round(float(row["std_sentiment"]) if not math.isnan(row["std_sentiment"]) else 0.0, 4),
                "n_turns": int(row["n_turns"]),
            }
        )
    return sorted(result, key=lambda x: x["game_num"])


@router.get("/series-position")
def get_series_position():
    """Sentiment by game_num split by whether the home team leads the series."""
    joined = data_module.app_data.joined

    # home leads: home_series_wins > away_series_wins at start of game
    # We use post-game values: home_series_wins includes current game result
    # Approximate pre-game lead using prior wins:
    # home leads before this game if (home_series_wins - home_win) > (away_series_wins - (1-home_win))
    df = joined.copy()
    df["home_wins_before"] = df["home_series_wins"] - df["home_win"].astype(int)
    df["away_wins_before"] = df["away_series_wins"] - (1 - df["home_win"].astype(int))
    df["home_leads"] = df["home_wins_before"] > df["away_wins_before"]

    grp = (
        df.groupby(["game_num", "home_leads"])["sentiment_numeric"]
        .agg(mean_sentiment="mean", n_turns="count")
        .reset_index()
    )

    result = []
    for _, row in grp.iterrows():
        result.append(
            {
                "game_num": int(row["game_num"]),
                "home_leads": bool(row["home_leads"]),
                "mean_sentiment": round(float(row["mean_sentiment"]), 4),
                "n_turns": int(row["n_turns"]),
            }
        )
    return result
