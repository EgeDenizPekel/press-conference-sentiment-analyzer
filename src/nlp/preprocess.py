"""
Preprocessing pipeline for ASAP Sports transcripts.

Main outputs:
  - Cleaned transcript-level DataFrame (one row per transcript)
  - Speaker turns DataFrame (one row per speaker answer)

Fixes two Phase 1 data quality issues:
  1. Date parse failures for joint press conferences (two speakers on one page)
     -> falls back to third-to-last segment when second-to-last isn't a month
  2. COACH LASTNAME -> FULL NAME normalization for coaches
"""

import re
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Coach name normalization
# Maps "COACH KERR" style entries (last name only after stripping prefix)
# to the coach's full name. All keys uppercase.
# ---------------------------------------------------------------------------
COACH_NAME_MAP = {
    "COACH KERR":        "STEVE KERR",
    "COACH SPOELSTRA":   "ERIK SPOELSTRA",
    "COACH POPOVICH":    "GREGG POPOVICH",
    "COACH LUE":         "TYRONN LUE",
    "COACH VOGEL":       "FRANK VOGEL",
    "COACH BUDENHOLZER": "MIKE BUDENHOLZER",
    "COACH BLATT":       "DAVID BLATT",
    "COACH BROWN":       "MIKE BROWN",
    "COACH SAUNDERS":    "FLIP SAUNDERS",
    "COMMISSIONER SILVER": "ADAM SILVER",
    "COMMISSIONER STERN":  "DAVID STERN",
}

# Months for date parsing validation
MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

# Filler phrases to remove from speaker turns
FILLERS = re.compile(
    r"\b(you know|i mean|um+|uh+|like i said|you know what i mean|"
    r"obviously|basically|literally|at the end of the day)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_date(row: pd.Series) -> pd.Timestamp:
    """
    Parse date from event_date_from_page title string.
    Format: '...  - YEAR - EVENT - Month Day - Speaker [- Speaker2]'
    Falls back to third-to-last segment when second-to-last isn't a month
    (joint press conferences have two speaker names at the end).
    """
    parts = str(row["event_date_from_page"]).split(" - ")
    year = int(row["asap_year"])

    for offset in (2, 3):
        if len(parts) < offset + 1:
            continue
        candidate = parts[-offset].strip()
        first_word = candidate.split()[0].lower() if candidate.split() else ""
        if first_word in MONTHS:
            try:
                return pd.to_datetime(f"{year} {candidate}", format="%Y %B %d")
            except ValueError:
                continue
    return pd.NaT


# ---------------------------------------------------------------------------
# Speaker turn extraction
# ---------------------------------------------------------------------------

# Matches "ALL CAPS NAME:" attribution at the start of a speaker turn.
# Allows hyphens (e.g. "DRAYMOND GREEN:"), apostrophes (e.g. "D'ANGELO:"),
# and periods (e.g. "J.R. SMITH:").
_SPEAKER_TAG = re.compile(r"^([A-Z][A-Z\s'\.\-]{1,40}):\s*", re.MULTILINE)

# Matches "Q." question markers (with optional trailing whitespace/newline)
_QUESTION = re.compile(r"Q\.\s*", re.MULTILINE)


def extract_speaker_turns(transcript: str, speaker: str) -> list[str]:
    """
    Extract individual answer turns for the given speaker from a transcript.

    Strategy:
      1. Strip the header (everything before the first 'Q.')
      2. Split the Q&A body on question markers
      3. For each Q/A block, find the answer text after the speaker attribution
      4. In group press conferences, only keep turns attributed to `speaker`

    Returns list of cleaned answer strings (empty strings excluded).
    """
    # 1. Strip header
    qa_start = transcript.find("Q.")
    if qa_start == -1:
        return []
    body = transcript[qa_start:]

    # 2. Split on question markers to get blocks
    # Each block is: "Q. [question text] SPEAKER: [answer text]"
    blocks = _QUESTION.split(body)

    turns = []
    speaker_upper = speaker.upper().strip()

    for block in blocks:
        if not block.strip():
            continue

        # 3. Find speaker attribution in this block
        match = _SPEAKER_TAG.search(block)
        if match:
            attributed = match.group(1).strip().upper()
            answer_text = block[match.end():].strip()

            # 4. For group press conferences, skip other speakers' turns
            # Allow partial match: "LEBRON" matches "LEBRON JAMES"
            if attributed not in speaker_upper and speaker_upper not in attributed:
                continue
        else:
            # No speaker tag found - single-speaker interview, entire block is the answer
            answer_text = block.strip()

        answer_text = _clean_turn(answer_text)
        if answer_text:
            turns.append(answer_text)

    return turns


def _clean_turn(text: str) -> str:
    """Clean a single speaker turn for NLP input."""
    # Remove any remaining speaker tags within the turn
    text = _SPEAKER_TAG.sub("", text)
    # Remove filler phrases
    text = FILLERS.sub("", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main preprocessing functions
# ---------------------------------------------------------------------------

def load_and_clean_transcripts() -> pd.DataFrame:
    """
    Load raw transcripts, fix speaker names, parse dates.
    Returns cleaned transcript-level DataFrame.
    """
    df = pd.read_csv(PROCESSED_DIR / "transcripts.csv")

    # Fix COACH LASTNAME -> FULL NAME
    df["speaker"] = df["speaker"].str.upper().str.strip()
    df["speaker"] = df["speaker"].replace(COACH_NAME_MAP)

    # Parse dates with fallback
    df["date"] = df.apply(_parse_date, axis=1)

    # Round classification
    df["round"] = df["event_name"].apply(
        lambda x: "Finals" if "FINALS" in x and "CONFERENCE" not in x
        else ("WCF" if "WCF" in x else "ECF")
    )

    # Word count
    df["word_count"] = df["transcript"].str.split().str.len()

    # Game day flag (requires games data)
    return df


def build_speaker_turns(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Extract speaker turns from all transcripts.
    Returns one row per turn with columns:
        interview_id, speaker, role, round, asap_year, date,
        is_game_day (NaN until joined), turn_idx, turn_text, word_count
    """
    if df is None:
        df = load_and_clean_transcripts()

    records = []
    for _, row in df.iterrows():
        turns = extract_speaker_turns(row["transcript"], row["speaker"])
        for idx, turn in enumerate(turns):
            wc = len(turn.split())
            if wc < 10:  # drop very short turns - insufficient for sentiment
                continue
            records.append({
                "interview_id": row["interview_id"],
                "speaker":      row["speaker"],
                "asap_year":    row["asap_year"],
                "event_name":   row["event_name"],
                "round":        row["round"],
                "date":         row["date"],
                "turn_idx":     idx,
                "turn_text":    turn,
                "word_count":   wc,
            })

    turns_df = pd.DataFrame(records)
    return turns_df


if __name__ == "__main__":
    print("Loading and cleaning transcripts...")
    df = load_and_clean_transcripts()
    date_failures = df["date"].isna().sum()
    print(f"  Transcripts: {len(df)}")
    print(f"  Date parse failures: {date_failures} (was 150 before fix)")

    print("\nExtracting speaker turns...")
    turns = build_speaker_turns(df)
    print(f"  Speaker turns: {len(turns)}")
    print(f"  Avg words per turn: {turns['word_count'].mean():.0f}")
    print(f"  Turns < 10 words dropped automatically")

    out = PROCESSED_DIR / "speaker_turns.csv"
    turns.to_csv(out, index=False)
    print(f"\nSaved to {out}")
