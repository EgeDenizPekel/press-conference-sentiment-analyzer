"""
/speakers endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from api import data as data_module

router = APIRouter()


@router.get("")
def get_speakers(min_turns: int = Query(default=50, ge=1)):
    """
    Speakers ranked by mean sentiment, filtered to those with at least min_turns.
    Returns top 20 to keep charts readable.
    """
    joined = data_module.app_data.joined

    grp = (
        joined.groupby("speaker")
        .agg(
            mean_sentiment=("sentiment_numeric", "mean"),
            pct_positive=("sentiment_label", lambda s: (s == "POSITIVE").mean()),
            n_turns=("sentiment_numeric", "count"),
        )
        .reset_index()
    )

    filtered = grp[grp["n_turns"] >= min_turns].copy()
    filtered = filtered.sort_values("mean_sentiment", ascending=False).head(20)

    result = []
    for _, row in filtered.iterrows():
        result.append(
            {
                "speaker": row["speaker"],
                "mean_sentiment": round(float(row["mean_sentiment"]), 4),
                "pct_positive": round(float(row["pct_positive"]), 4),
                "n_turns": int(row["n_turns"]),
            }
        )
    return result
