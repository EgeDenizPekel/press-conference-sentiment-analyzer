"""
Build a HuggingFace DatasetDict from combined hand-labeled seed + GPT weak labels.

Outputs:
  - data/processed/training_labels.csv  (unified label table with source column)
  - DatasetDict with train / validation splits (80/20 stratified)

Usage:
    python -m src.training.dataset
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MAX_LENGTH = 256

# Label ordering matches twitter-roberta: LABEL_0=NEG, LABEL_1=NEU, LABEL_2=POS
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Build unified training_labels.csv
# ---------------------------------------------------------------------------

def build_training_labels() -> pd.DataFrame:
    """
    Merge hand-labeled seed turns with GPT weak labels into a single DataFrame.

    Seed turns take priority on deduplication: if a (interview_id, turn_idx) pair
    appears in both files, the hand label is kept.

    Saves to data/processed/training_labels.csv and returns the DataFrame.
    """
    seed = pd.read_csv(PROCESSED_DIR / "labels_seed.csv")
    seed = seed[["interview_id", "turn_idx", "turn_text", "label"]].copy()
    seed["source"] = "hand"

    weak = pd.read_csv(PROCESSED_DIR / "weak_labels.csv")
    # Attach turn text from speaker_turns (weak_labels only has IDs + label)
    turns = pd.read_csv(PROCESSED_DIR / "speaker_turns.csv")
    weak = weak.merge(
        turns[["interview_id", "turn_idx", "turn_text"]],
        on=["interview_id", "turn_idx"],
        how="left",
    )
    weak = weak.rename(columns={"gpt_label": "label"})[
        ["interview_id", "turn_idx", "turn_text", "label"]
    ].copy()
    weak["source"] = "gpt"

    # Concatenate with seed first so drop_duplicates keeps seed rows
    combined = pd.concat([seed, weak], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["interview_id", "turn_idx"], keep="first")
    dropped = before - len(combined)
    if dropped:
        print(f"Deduplicated {dropped} overlapping rows (seed takes priority).")

    combined["label_id"] = combined["label"].map(LABEL2ID)
    combined = combined.dropna(subset=["turn_text", "label", "label_id"])
    combined["label_id"] = combined["label_id"].astype(int)

    out = PROCESSED_DIR / "training_labels.csv"
    combined.to_csv(out, index=False)
    print(f"Saved {len(combined)} rows to {out}")
    print(f"Label distribution:\n{combined['label'].value_counts().to_string()}")
    print(f"Source distribution:\n{combined['source'].value_counts().to_string()}")

    return combined


# ---------------------------------------------------------------------------
# Build DatasetDict
# ---------------------------------------------------------------------------

def build_dataset(df: pd.DataFrame | None = None) -> DatasetDict:
    """
    Tokenize the training label table and split into train/validation.

    Parameters
    ----------
    df : pre-loaded DataFrame from build_training_labels(), or None to load from disk.

    Returns
    -------
    DatasetDict with keys "train" and "validation".
    Each example has tokenizer fields + "labels" (int).
    """
    if df is None:
        df = pd.read_csv(PROCESSED_DIR / "training_labels.csv")
        df["label_id"] = df["label"].map(LABEL2ID)
        df = df.dropna(subset=["turn_text", "label", "label_id"])
        df["label_id"] = df["label_id"].astype(int)

    # Stratified 80/20 split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Split -> train: {len(train_df)}, val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch["turn_text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    def _make_hf_dataset(split_df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({
            "turn_text": split_df["turn_text"].tolist(),
            "labels":    split_df["label_id"].tolist(),
        })
        return ds.map(_tokenize_batch, batched=True, remove_columns=["turn_text"])

    return DatasetDict({
        "train":      _make_hf_dataset(train_df),
        "validation": _make_hf_dataset(val_df),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building training_labels.csv...")
    combined = build_training_labels()

    print("\nBuilding HuggingFace DatasetDict...")
    dataset = build_dataset(combined)
    print(f"\nDataset:\n{dataset}")
