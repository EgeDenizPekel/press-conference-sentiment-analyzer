"""
ASAP Sports playoff transcript scraper.

URL hierarchy (4 levels):
  1. show_events.php?event_id=X  -- series index (e.g., "NBA WCF: Warriors vs Spurs")
  2. show_event.php?id=X         -- single game/day index
  3. show_interview.php?id=X     -- individual speaker transcript

Hardcoded event IDs were validated manually from ASAP Sports year pages.
ASAP year 2013 = Kaggle SEASON 2012, ASAP year 2014 = Kaggle SEASON 2013, etc.

Output: data/processed/transcripts.csv
Columns: asap_year, event_name, event_date, game_event_id, interview_id,
         speaker, transcript, url
"""

import csv
import time
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "http://www.asapsports.com"
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "transcripts"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY = 2.0  # seconds between requests

# ---------------------------------------------------------------------------
# Hardcoded event IDs validated from ASAP Sports year pages (2013-2022).
# Format: { asap_year: { event_label: event_id } }
# Duplicate event IDs for the same series are intentional -- some games are
# only listed under one of the two event IDs. Deduplication happens by
# (event_date, speaker) at write time.
# ---------------------------------------------------------------------------
EVENTS: dict[int, dict[str, int]] = {
    2013: {
        "NBA WCF: GRIZZLIES v SPURS": 89430,
        "NBA ECF: PACERS v HEAT": 89620,
        "NBA FINALS: SPURS v HEAT": 90211,
    },
    2014: {
        "NBA ECF: HEAT v PACERS": 99304,
        "NBA WCF: THUNDER v SPURS": 99343,
        "NBA FINALS: HEAT v SPURS": 100046,
    },
    2015: {
        "NBA ECF: CAVALIERS v HAWKS": 109450,
        "NBA WCF: ROCKETS v WARRIORS": 109488,
        "NBA FINALS: CAVALIERS v WARRIORS": 110335,
    },
    2016: {
        "NBA ECF: RAPTORS v CAVALIERS (a)": 119659,
        "NBA ECF: CLEVELAND VS TORONTO (b)": 119672,
        "NBA WCF: THUNDER v WARRIORS (a)": 119751,
        "NBA WCF: GOLDEN STATE VS OKC (b)": 119758,
        "NBA FINALS: CAVALIERS v WARRIORS (a)": 120652,
        "NBA FINALS: CLEVELAND VS GOLDEN STATE (b)": 120675,
    },
    2017: {
        "NBA WCF: WARRIORS VS SPURS": 130048,
        "NBA ECF: CELTICS VS CAVALIERS": 130144,
        "NBA FINALS: CLEVELAND VS GOLDEN STATE": 131021,
    },
    2018: {
        "NBA ECF: CAVALIERS vs CELTICS": 140311,
        "NBA WCF: WARRIORS vs ROCKETS": 140336,
        "NBA FINALS: CAVALIERS vs WARRIORS (a)": 140722,
        "NBA FINALS: CAVALIERS VS WARRIORS (b)": 140731,
    },
    2019: {
        "NBA WCF: TRAIL BLAZERS vs WARRIORS": 149913,
        "NBA ECF: RAPTORS vs BUCKS": 150077,
        "NBA FINALS: WARRIORS vs RAPTORS": 151002,
    },
    2020: {
        "NBA WCF: LAKERS VS NUGGETS": 159915,
        "NBA ECF: CELTICS VS HEAT": 159923,
        "NBA FINALS: LAKERS VS HEAT": 160336,
    },
    2021: {
        "NBA WCF: CLIPPERS VS SUNS": 166928,
        "NBA ECF: HAWKS VS BUCKS": 167015,
        "NBA FINALS: BUCKS VS SUNS": 167917,
    },
    2022: {
        "NBA WCF: MAVERICKS VS WARRIORS": 176543,
        "NBA ECF: CELTICS VS HEAT": 176755,
        "NBA FINALS: CELTICS VS WARRIORS": 177451,
    },
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "http://www.asapsports.com/",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def _get(session: requests.Session, url: str, cache_path: Path | None = None) -> BeautifulSoup | None:
    """Fetch URL with caching. Returns BeautifulSoup or None on error."""
    if cache_path and cache_path.exists():
        return BeautifulSoup(cache_path.read_text(encoding="utf-8"), "lxml")
    try:
        time.sleep(REQUEST_DELAY)
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        if cache_path:
            cache_path.write_text(resp.text, encoding="utf-8")
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None


def _resolve_event_url(session: requests.Session, year: int, event_id: int) -> str | None:
    """
    Fetch the year page to find the full URL for a given event_id.
    The full URL includes the required `year` and `title` query params.
    Year pages are cached so this only hits the network once per year.
    """
    url = f"{BASE_URL}/show_year.php?category=11&year={year}"
    cache = RAW_DIR / f"year_{year}.html"
    soup = _get(session, url, cache)
    if soup is None:
        return None

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if f"event_id={event_id}" in href and "show_events.php" in href:
            return href if href.startswith("http") else f"{BASE_URL}/{href}"
    return None


def _get_game_urls(session: requests.Session, year: int, event_id: int) -> list[tuple[str, str]]:
    """
    Resolve the series event URL then return list of (game_page_url, date_text) tuples.
    Game page links use show_event.php?event_id=X&category=11&date=...&title=...
    """
    event_url = _resolve_event_url(session, year, event_id)
    if event_url is None:
        log.warning(f"Could not resolve URL for event_id={event_id} year={year}")
        return []

    cache = RAW_DIR / f"series_{event_id}.html"
    soup = _get(session, event_url, cache)
    if soup is None:
        return []

    results = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "show_event.php?event_id=" in href:
            full_url = href if href.startswith("http") else f"{BASE_URL}/{href}"
            date_text = a.get_text(strip=True)
            results.append((full_url, date_text))
    return results


def _get_interview_ids(session: requests.Session, game_url: str, game_event_id: int) -> list[tuple[int, str]]:
    """
    Fetch a game/day page and return list of (interview_id, speaker_text) tuples.
    Links on this page point to show_interview.php?id=X.
    """
    cache = RAW_DIR / f"game_{game_event_id}.html"
    soup = _get(session, game_url, cache)
    if soup is None:
        return []

    results = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "show_interview.php?id=" in href:
            try:
                iid = int(href.split("id=")[1].split("&")[0])
                speaker = a.get_text(strip=True)
                results.append((iid, speaker))
            except (ValueError, IndexError):
                continue
    return results


def _get_transcript(session: requests.Session, interview_id: int) -> dict | None:
    """
    Fetch an interview page and return a dict with keys:
        speaker, event_date, transcript
    Returns None on failure or if no transcript content found.
    """
    url = f"{BASE_URL}/show_interview.php?id={interview_id}"
    cache = RAW_DIR / f"interview_{interview_id}.html"
    soup = _get(session, url, cache)
    if soup is None:
        return None

    # Extract date from page -- typically in a <td> or <title> near the top
    date_text = ""
    title = soup.find("title")
    if title:
        date_text = title.get_text(strip=True)

    # Find the transcript TD block.
    # ASAP Sports pages have two large TD blocks: one with nav boilerplate prepended,
    # one without. We want the largest TD that:
    #   - Contains "Q." (question-answer format)
    #   - Does not start with "Browse by Sport" (navigation boilerplate)
    transcript = ""
    for td in sorted(soup.find_all("td"), key=lambda t: len(t.get_text()), reverse=True):
        text = td.get_text(separator="\n", strip=True)
        if len(text) > 200 and "Q." in text and not text.startswith("Browse by Sport"):
            transcript = text
            break

    if not transcript:
        return None

    # Extract speaker name: appears as ALL-CAPS before a colon in the Q&A body.
    # Only search after the first "Q." to skip event name header lines
    # (e.g., "NBA FINALS: CELTICS VS. WARRIORS") which match the same pattern.
    speaker = ""
    qa_start = transcript.find("Q.")
    qa_section = transcript[qa_start:] if qa_start != -1 else transcript
    for line in qa_section.splitlines():
        line = line.strip()
        if ":" in line:
            candidate = line.split(":")[0].strip()
            words = candidate.split()
            if (
                candidate.isupper()
                and 1 < len(words) <= 4
                and candidate.replace(" ", "").isalpha()
            ):
                speaker = candidate
                break

    return {
        "speaker": speaker,
        "event_date": date_text,
        "transcript": transcript,
    }


def scrape(asap_years: list[int] | None = None) -> None:
    """
    Main scrape entrypoint. Crawls all events for the specified ASAP years
    (defaults to all years in EVENTS) and writes transcripts.csv.

    Deduplicates rows by (interview_id) -- each interview has a unique ID,
    so running the scraper multiple times is safe (cached HTML is reused).
    """
    if asap_years is None:
        asap_years = sorted(EVENTS.keys())

    output_path = PROCESSED_DIR / "transcripts.csv"
    seen_interview_ids: set[int] = set()

    # Load existing output to avoid re-processing
    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen_interview_ids.add(int(row["interview_id"]))
        log.info(f"Resuming -- {len(seen_interview_ids)} transcripts already saved")

    session = _session()

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "asap_year", "event_name", "event_id",
            "game_event_id", "game_date_text",
            "interview_id", "speaker", "event_date_from_page",
            "transcript", "url",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if output_path.stat().st_size == 0:
            writer.writeheader()

        for year in asap_years:
            year_events = EVENTS.get(year, {})
            log.info(f"Year {year}: {len(year_events)} events")

            for event_name, event_id in year_events.items():
                log.info(f"  Event: {event_name} (id={event_id})")
                game_urls = _get_game_urls(session, year, event_id)
                log.info(f"    Found {len(game_urls)} game/day pages")

                for game_url, game_date_text in game_urls:
                    # Extract game_event_id from URL for cache key
                    try:
                        game_event_id = int(game_url.split("event_id=")[1].split("&")[0])
                    except (IndexError, ValueError):
                        game_event_id = hash(game_url)

                    interview_ids = _get_interview_ids(session, game_url, game_event_id)

                    for interview_id, speaker_hint in interview_ids:
                        if interview_id in seen_interview_ids:
                            continue

                        data = _get_transcript(session, interview_id)
                        if data is None:
                            log.warning(f"      No transcript for interview {interview_id}")
                            continue

                        writer.writerow({
                            "asap_year": year,
                            "event_name": event_name,
                            "event_id": event_id,
                            "game_event_id": game_event_id,
                            "game_date_text": game_date_text,
                            "interview_id": interview_id,
                            "speaker": data["speaker"] or speaker_hint,
                            "event_date_from_page": data["event_date"],
                            "transcript": data["transcript"],
                            "url": f"{BASE_URL}/show_interview.php?id={interview_id}",
                        })
                        f.flush()
                        seen_interview_ids.add(interview_id)
                        log.info(f"      Saved interview {interview_id}: {data['speaker'] or speaker_hint}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape ASAP Sports playoff transcripts")
    parser.add_argument(
        "--years", nargs="+", type=int,
        help="ASAP years to scrape (default: all). E.g. --years 2021 2022"
    )
    args = parser.parse_args()
    scrape(asap_years=args.years)
