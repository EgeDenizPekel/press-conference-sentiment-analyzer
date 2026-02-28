"""
Weak-label 2,000 speaker turns using GPT-4o-mini.

Steps:
  1. Validate on 50 seed turns and log accuracy vs hand labels.
  2. Stratified sample 2,000 turns from the remaining corpus by round.
  3. Batch 20 turns per API call with incremental checkpointing.
  4. Final output: data/processed/weak_labels.csv

Usage:
    export OPENAI_API_KEY=sk-...
    python -m src.training.label
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
CHECKPOINT_PATH = PROCESSED_DIR / "weak_labels_partial.csv"
OUTPUT_PATH = PROCESSED_DIR / "weak_labels.csv"

BATCH_SIZE = 20
TARGET_N = 2_000
CHECKPOINT_EVERY = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a sports sentiment classifier for NBA playoff press conferences.

Classify each speaker turn as exactly one of: POSITIVE, NEGATIVE, or NEUTRAL.

DEFINITIONS:
- POSITIVE: confidence, satisfaction, praise of teammates/performance, optimism about upcoming games, resilience after a win
- NEGATIVE: disappointment, frustration, self-criticism, concern after a loss, expressions of defeat
- NEUTRAL: tactical/analytical commentary, factual game description, injury updates, media questions deflected - no clear emotional valence

EXAMPLES:
Turn: "I have full confidence that tomorrow will be much better. He's one of the best players in the league and I trust he'll bounce back from a tough night."
Label: POSITIVE
Confidence: 1

Turn: "He has got some different plays he draws up and writes the numbers down. It's great. It's a lot of fun working with this group."
Label: POSITIVE
Confidence: 1

Turn: "Our guys were disappointed about the loss, but now we've got to bounce back and be ready for Game 2. It's behind us."
Label: NEGATIVE
Confidence: 1

Turn: "I don't think I can explain it honestly. I'm just devastated. All that work and heart they showed, and we still couldn't get it done."
Label: NEGATIVE
Confidence: 1

Turn: "Well, I need to look at the tape. One of them was possibly a block/charge - those are tough calls that go each way. I'll know more after I review it."
Label: NEUTRAL
Confidence: 1

Turn: "Yes. We've had conversations throughout the season. Both teams are very close to the finish. We're aware of what's going on around us."
Label: NEUTRAL
Confidence: 0

RESPONSE FORMAT:
Return a JSON object with a single key "results" containing an array. Each element must have:
  - "id": integer matching the input id
  - "label": one of "POSITIVE", "NEGATIVE", "NEUTRAL"
  - "confidence": 1 if clear-cut, 0 if ambiguous or mixed valence

Example: {"results": [{"id": 0, "label": "POSITIVE", "confidence": 1}, {"id": 1, "label": "NEUTRAL", "confidence": 0}]}

Return only the JSON object - no other text."""

VALID_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_gpt(client: OpenAI, turns: list[dict]) -> list[dict]:
    """
    Call GPT-4o-mini with a batch of turns.

    turns: list of {"id": int, "text": str}
    Returns list of {"id": int, "label": str, "confidence": int}
    """
    user_content = json.dumps(
        [{"id": t["id"], "text": t["text"]} for t in turns],
        ensure_ascii=False,
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    # Extract the list from {"results": [...]} or any wrapper
    if isinstance(parsed, dict):
        for key in ("results", "classifications", "labels", "data", "output"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        # Fallback: grab the first list value
        for v in parsed.values():
            if isinstance(v, list):
                return v
        raise ValueError(f"No list found in GPT response: {list(parsed.keys())}")

    if isinstance(parsed, list):
        return parsed

    raise ValueError(f"Unexpected GPT response type: {type(parsed)}")


def _extract_predictions(raw_results: list[dict], batch_size: int) -> list[tuple[str, int]]:
    """
    Convert raw GPT result list to an ordered list of (label, confidence) tuples.

    Uses local batch id (0..batch_size-1) to reconstruct ordering.
    Falls back to ("NEUTRAL", 0) for missing or invalid entries.
    """
    id_to_result: dict[int, dict] = {}
    for r in raw_results:
        if isinstance(r, dict) and "id" in r:
            id_to_result[int(r["id"])] = r

    preds = []
    for i in range(batch_size):
        r = id_to_result.get(i)
        if r is None:
            log.warning("Missing GPT result for batch index %d, defaulting to NEUTRAL/0", i)
            preds.append(("NEUTRAL", 0))
        else:
            label = str(r.get("label", "NEUTRAL")).upper()
            if label not in VALID_LABELS:
                log.warning("Invalid label '%s' at index %d, defaulting to NEUTRAL/0", label, i)
                label = "NEUTRAL"
                confidence = 0
            else:
                confidence = int(bool(r.get("confidence", 1)))
            preds.append((label, confidence))

    return preds


# ---------------------------------------------------------------------------
# Seed validation
# ---------------------------------------------------------------------------

def validate_on_seed(client: OpenAI, seed: pd.DataFrame) -> float:
    """
    Run the labeler on the 50 seed turns and compare to hand labels.
    Logs per-class accuracy breakdown and returns overall accuracy.
    """
    log.info("Validating on %d seed turns...", len(seed))
    all_preds: list[str] = []

    for start in range(0, len(seed), BATCH_SIZE):
        batch = seed.iloc[start : start + BATCH_SIZE].reset_index(drop=True)
        turns_input = [
            {"id": i, "text": str(row["turn_text"])}
            for i, (_, row) in enumerate(batch.iterrows())
        ]

        try:
            raw = _call_gpt(client, turns_input)
            preds = _extract_predictions(raw, len(batch))
            all_preds.extend(label for label, _ in preds)
        except Exception as e:
            log.error("Seed batch starting at %d failed: %s", start, e)
            all_preds.extend(["NEUTRAL"] * len(batch))

    gold = seed["label"].tolist()
    correct = sum(p == g for p, g in zip(all_preds, gold))
    accuracy = correct / len(gold)

    log.info(
        "Seed validation: %d/%d correct  accuracy=%.1f%%",
        correct, len(gold), accuracy * 100,
    )

    for cls in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        indices = [i for i, g in enumerate(gold) if g == cls]
        if not indices:
            continue
        cls_correct = sum(all_preds[i] == cls for i in indices)
        log.info(
            "  %s: %d turns  %.1f%% accuracy",
            cls, len(indices), 100 * cls_correct / len(indices),
        )

    if accuracy < 0.65:
        log.warning(
            "Seed accuracy %.1f%% is below 65%% threshold - consider revising the prompt.",
            accuracy * 100,
        )

    return accuracy


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def sample_turns(turns: pd.DataFrame, seed: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Stratified sample of n turns from the corpus, excluding seed turns.
    Sampling is proportional to round distribution (Finals / ECF / WCF).
    """
    seed_keys = set(zip(seed["interview_id"], seed["turn_idx"]))
    pool_mask = ~pd.Series(
        list(zip(turns["interview_id"], turns["turn_idx"]))
    ).isin(seed_keys)
    pool = turns[pool_mask.values].copy()

    log.info("Sampling pool after excluding %d seed turns: %d available", len(seed), len(pool))

    round_counts = pool["round"].value_counts()
    total_pool = len(pool)
    rounds = round_counts.index.tolist()

    frames = []
    remaining_n = n
    for i, rnd in enumerate(rounds):
        rnd_pool = pool[pool["round"] == rnd]
        if i == len(rounds) - 1:
            # Last round absorbs any rounding difference
            n_rnd = remaining_n
        else:
            n_rnd = round(n * len(rnd_pool) / total_pool)
        n_rnd = min(n_rnd, len(rnd_pool))
        sampled = rnd_pool.sample(n=n_rnd, random_state=42)
        frames.append(sampled)
        remaining_n -= n_rnd
        log.info("  %s: %d sampled / %d available", rnd, n_rnd, len(rnd_pool))

    result = pd.concat(frames).reset_index(drop=True)
    log.info("Total sampled: %d", len(result))
    return result


# ---------------------------------------------------------------------------
# Incremental labeling loop
# ---------------------------------------------------------------------------

def label_turns(
    client: OpenAI,
    to_label: pd.DataFrame,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """
    Label all turns in `to_label` using GPT-4o-mini, with checkpointing.

    Resumes from checkpoint_path if it already exists, skipping completed turns.
    Checkpoints every CHECKPOINT_EVERY turns.
    """
    # Load checkpoint
    existing_rows: list[dict] = []
    labeled_keys: set[tuple] = set()

    if checkpoint_path.exists():
        ckpt = pd.read_csv(checkpoint_path)
        labeled_keys = set(zip(ckpt["interview_id"], ckpt["turn_idx"]))
        existing_rows = ckpt.to_dict("records")
        log.info("Resuming from checkpoint: %d turns already done", len(labeled_keys))

    # Filter out already-labeled rows
    remaining_mask = ~pd.Series(
        list(zip(to_label["interview_id"], to_label["turn_idx"]))
    ).isin(labeled_keys)
    remaining = to_label[remaining_mask.values].reset_index(drop=True)
    log.info("%d turns remaining to label", len(remaining))

    all_rows = existing_rows.copy()

    for start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining.iloc[start : start + BATCH_SIZE].reset_index(drop=True)

        turns_input = [
            {"id": i, "text": str(row["turn_text"])}
            for i, (_, row) in enumerate(batch.iterrows())
        ]

        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                raw = _call_gpt(client, turns_input)
                preds = _extract_predictions(raw, len(batch))
                for i, (label, confidence) in enumerate(preds):
                    row = batch.iloc[i]
                    all_rows.append({
                        "interview_id":   int(row["interview_id"]),
                        "turn_idx":       int(row["turn_idx"]),
                        "gpt_label":      label,
                        "gpt_confidence": confidence,
                    })
                success = True
                break
            except Exception as e:
                wait = 2 ** attempt
                if attempt < max_retries - 1:
                    log.warning(
                        "Batch at index %d failed (attempt %d/%d): %s - retrying in %ds",
                        start, attempt + 1, max_retries, e, wait,
                    )
                    time.sleep(wait)
                else:
                    log.error(
                        "Batch at index %d failed after %d attempts: %s - defaulting to NEUTRAL/0",
                        start, max_retries, e,
                    )

        if not success:
            for i, (_, row) in enumerate(batch.iterrows()):
                all_rows.append({
                    "interview_id":   int(row["interview_id"]),
                    "turn_idx":       int(row["turn_idx"]),
                    "gpt_label":      "NEUTRAL",
                    "gpt_confidence": 0,
                })

        total_done = start + len(batch)
        if total_done % CHECKPOINT_EVERY < BATCH_SIZE or total_done >= len(remaining):
            ckpt_df = pd.DataFrame(all_rows)
            ckpt_df.to_csv(checkpoint_path, index=False)
            pct = 100 * total_done / len(remaining)
            log.info("Checkpoint saved: %d / %d turns (%.0f%%)", total_done, len(remaining), pct)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    log.info("Loading speaker turns and seed labels...")
    turns = pd.read_csv(PROCESSED_DIR / "speaker_turns.csv")
    seed = pd.read_csv(PROCESSED_DIR / "labels_seed.csv")
    log.info("  Speaker turns: %d", len(turns))
    log.info("  Seed turns:    %d", len(seed))

    # Step 1: validate prompt quality on seed set
    validate_on_seed(client, seed)

    # Step 2: stratified sample of TARGET_N turns (excluding seed)
    to_label = sample_turns(turns, seed, TARGET_N)

    # Step 3: label with incremental checkpointing
    result = label_turns(client, to_label, CHECKPOINT_PATH)

    # Step 4: save final output
    result.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved %d labeled turns to %s", len(result), OUTPUT_PATH)

    log.info("Label distribution:\n%s", result["gpt_label"].value_counts().to_string())
    uncertain = (result["gpt_confidence"] == 0).sum()
    log.info(
        "Uncertain turns (gpt_confidence=0): %d (%.1f%%)",
        uncertain, 100 * uncertain / len(result),
    )

    # Clean up checkpoint now that we have the final file
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        log.info("Removed checkpoint file.")


if __name__ == "__main__":
    main()
