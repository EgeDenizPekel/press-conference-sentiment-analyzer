"""
Run the fine-tuned sentiment model on all speaker turns.

Outputs data/processed/sentiment_scores.csv with three new columns:
  sentiment_label   - POSITIVE / NEUTRAL / NEGATIVE
  sentiment_score   - confidence for the predicted class (0-1)
  sentiment_numeric - P(POSITIVE) - P(NEGATIVE), continuous score in [-1, 1]

Usage:
    python -m src.analysis.score
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent.parent / "models" / "fine-tuned-sports-sentiment"

# Label order matches twitter-roberta: LABEL_0=NEG, LABEL_1=NEU, LABEL_2=POS
# Fine-tuned model saves with id2label names, so also support direct label names
_LABEL_MAP = {
    "LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE",
    "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL", "POSITIVE": "POSITIVE",
}

BATCH_SIZE = 64


def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def score_turns(turns: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Run the fine-tuned model on every row of `turns` (or speaker_turns.csv).

    Returns the input DataFrame with three additional columns:
        sentiment_label, sentiment_score, sentiment_numeric
    """
    if turns is None:
        turns = pd.read_csv(PROCESSED_DIR / "speaker_turns.csv")

    device = _auto_device()
    print(f"Loading model from {MODELS_DIR} on {device}...")

    pipe = pipeline(
        "text-classification",
        model=str(MODELS_DIR),
        tokenizer=str(MODELS_DIR),
        device=device,
        truncation=True,
        max_length=256,
        top_k=3,          # return all 3 class probabilities
    )

    texts = turns["turn_text"].tolist()
    print(f"Scoring {len(texts):,} turns in batches of {BATCH_SIZE}...")

    labels, scores, numerics = [], [], []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        results = pipe(batch)

        for turn_result in results:
            # turn_result is a list of 3 dicts: [{label, score}, ...]
            prob = {_LABEL_MAP[r["label"]]: r["score"] for r in turn_result}
            top = max(turn_result, key=lambda r: r["score"])
            labels.append(_LABEL_MAP[top["label"]])
            scores.append(round(top["score"], 4))
            numerics.append(round(prob.get("POSITIVE", 0) - prob.get("NEGATIVE", 0), 4))

        if (start // BATCH_SIZE + 1) % 10 == 0:
            done = min(start + BATCH_SIZE, len(texts))
            print(f"  {done:,} / {len(texts):,} ({100 * done / len(texts):.0f}%)")

    out = turns.copy()
    out["sentiment_label"]   = labels
    out["sentiment_score"]   = scores
    out["sentiment_numeric"] = numerics
    return out


if __name__ == "__main__":
    result = score_turns()
    out_path = PROCESSED_DIR / "sentiment_scores.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result):,} rows to {out_path}")
    print(f"Label distribution:\n{result['sentiment_label'].value_counts().to_string()}")
    print(f"Mean sentiment_numeric: {result['sentiment_numeric'].mean():.3f}")
