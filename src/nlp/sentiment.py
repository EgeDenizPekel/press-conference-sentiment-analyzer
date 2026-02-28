"""
Baseline sentiment inference for NBA press conference speaker turns.

Wraps three off-the-shelf models and normalizes their output to
POSITIVE / NEGATIVE / NEUTRAL so they can be compared against
each other and against our hand-labeled seed set.

Models:
  - cardiffnlp/twitter-roberta-base-sentiment  (3-class)
  - ProsusAI/finbert                            (3-class, financial domain)
  - distilbert-base-uncased-finetuned-sst-2-english (2-class)
"""

import torch
import pandas as pd
from pathlib import Path
from transformers import pipeline

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

# Maximum tokens to send to any model (hard limit is 512 for these models;
# leaving headroom for special tokens)
MAX_TOKENS = 500

# ---------------------------------------------------------------------------
# Label normalization
# Each model uses different internal label strings -> map to canonical 3-class
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, dict[str, str]] = {
    "cardiffnlp/twitter-roberta-base-sentiment": {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "POSITIVE",
    },
    "ProsusAI/finbert": {
        "positive": "POSITIVE",
        "negative": "NEGATIVE",
        "neutral":  "NEUTRAL",
    },
    "distilbert-base-uncased-finetuned-sst-2-english": {
        "POSITIVE": "POSITIVE",
        "NEGATIVE": "NEGATIVE",
        # 2-class model: no neutral; caller should treat low-confidence as neutral if needed
    },
}


class BaselinePredictor:
    """
    Thin wrapper around a HuggingFace text-classification pipeline.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Number of texts to send to the model at once.
    device : str | None
        'cpu', 'mps', 'cuda', or None to auto-select.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        self._norm = _LABEL_MAP.get(model_name, {})

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Run inference on a list of texts.

        Returns a list of dicts with keys:
          raw_label   : model's original label string
          label       : normalized POSITIVE / NEGATIVE / NEUTRAL
          score       : confidence for the predicted label (0-1)
        """
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            raw = self._pipe(batch)
            for item in raw:
                raw_label = item["label"]
                norm_label = self._norm.get(raw_label, raw_label.upper())
                results.append({
                    "raw_label": raw_label,
                    "label":     norm_label,
                    "score":     round(float(item["score"]), 4),
                })
        return results

    def predict_df(self, df: pd.DataFrame, text_col: str = "turn_text") -> pd.DataFrame:
        """
        Convenience: add prediction columns to a copy of `df`.

        Adds columns:
          {model_short}_label   : e.g. twitter_label
          {model_short}_score
        """
        short = _model_short(self.model_name)
        preds = self.predict(df[text_col].tolist())
        out = df.copy()
        out[f"{short}_label"] = [p["label"] for p in preds]
        out[f"{short}_score"] = [p["score"]  for p in preds]
        return out


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def _model_short(model_name: str) -> str:
    """Return a short snake_case identifier for a model name."""
    mapping = {
        "cardiffnlp/twitter-roberta-base-sentiment": "twitter",
        "ProsusAI/finbert": "finbert",
        "distilbert-base-uncased-finetuned-sst-2-english": "distilbert",
    }
    return mapping.get(model_name, model_name.split("/")[-1].replace("-", "_"))


def run_all_baselines(
    df: pd.DataFrame | None = None,
    text_col: str = "turn_text",
    sample_n: int | None = None,
) -> pd.DataFrame:
    """
    Run all three baseline models on `df` (or load speaker_turns.csv).

    Parameters
    ----------
    df       : DataFrame with a `text_col` column (loads speaker_turns.csv if None)
    sample_n : Optionally limit to N rows for quick testing
    """
    if df is None:
        df = pd.read_csv(PROCESSED_DIR / "speaker_turns.csv")

    if sample_n is not None:
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)

    models = [
        "cardiffnlp/twitter-roberta-base-sentiment",
        "ProsusAI/finbert",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ]
    result = df.copy()
    for model_name in models:
        print(f"Running {model_name}...")
        predictor = BaselinePredictor(model_name)
        preds = predictor.predict(result[text_col].tolist())
        short = _model_short(model_name)
        result[f"{short}_label"] = [p["label"] for p in preds]
        result[f"{short}_score"] = [p["score"]  for p in preds]
        print(f"  Done. Label distribution:\n{result[f'{short}_label'].value_counts().to_string()}")

    return result


if __name__ == "__main__":
    # Quick smoke-test on 50 seed turns (fastest meaningful check)
    print("Loading seed labels...")
    seed = pd.read_csv(PROCESSED_DIR / "labels_seed.csv")
    print(f"  Seed turns: {len(seed)}")

    result = run_all_baselines(df=seed[["interview_id","turn_idx","speaker","role","round","turn_text","label"]])

    out_path = PROCESSED_DIR / "baseline_predictions.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Simple accuracy per model against hand labels
    for short in ["twitter", "finbert", "distilbert"]:
        col = f"{short}_label"
        if col in result.columns:
            acc = (result[col] == result["label"]).mean()
            print(f"  {short} accuracy on seed set: {acc:.1%}")
