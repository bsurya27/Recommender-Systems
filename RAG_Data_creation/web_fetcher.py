import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

# Optional: use the high-level `wikipedia` library if it is available.
try:
    import wikipedia  # type: ignore
    from wikipedia.exceptions import DisambiguationError, PageError  # type: ignore
except ImportError:  # pragma: no cover – library is optional
    wikipedia = None  # type: ignore
    DisambiguationError = PageError = Exception  # type: ignore

###############################################################################
# Configuration constants – tweak as needed                                 #
###############################################################################

################################################################################
# Resolve paths relative to this script so we do not depend on the caller's    #
# working directory.                                                           #
################################################################################

# Directory containing this script (…/RAG_Data_creation)
BASE_DIR: Path = Path(__file__).resolve().parent

# Output and log locations (inside same folder as this script)
OUTPUT_DIR: Path = BASE_DIR / "wiki_pages"
LOG_FILE: Path = BASE_DIR / "wiki_fetch.log"

RATE_LIMIT_SECONDS: float = 0.5  # Be nice to Wikipedia servers
MIN_VALID_BYTES: int = 200  # Consider a page valid only if it has at least this many bytes

###############################################################################
# Utility / helper functions                                                 #
###############################################################################

def setup_logging() -> None:
    """Configure the root logger to log to file *and* stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )


def ensure_output_dir() -> None:
    """Create the directory where we will store wiki pages and the log file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure the directory for the log file also exists (it is BASE_DIR)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _find_anime_csv() -> Path:
    """Return the first existing path among common filename variants."""
    candidates = [
        BASE_DIR / "anime.csv",
        BASE_DIR.parent / "anime.csv",
        BASE_DIR / "Anime.csv",
        BASE_DIR.parent / "Anime.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate anime.csv. Please place it in the project root or in RAG_Data_creation."\
    )


# CSV path determined at runtime
ANIME_CSV: Path = _find_anime_csv()

def load_anime_catalogue(csv_path: str | Path = ANIME_CSV) -> pd.DataFrame:
    """Load the anime catalogue and keep only columns we need."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run AnimeCollection.py first to generate it.")

    cols_to_keep = [
        "anime_id",
        "name",
        "title_english",
        "title_japanese",
    ]
    df = pd.read_csv(csv_path, usecols=[c for c in cols_to_keep if c in pd.read_csv(csv_path, nrows=0).columns])
    return df

###############################################################################
# Wikipedia fetching (two strategies)                                        #
###############################################################################

def _search_with_requests(query: str) -> Optional[str]:
    """Use MediaWiki search API to find the most relevant page title."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
    }
    try:
        response = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get("query", {}).get("search", [])
        if data:
            return data[0]["title"]  # type: ignore[index]
    except Exception as exc:  # noqa: BLE001 – broad but acceptable in scraper
        logging.debug("MediaWiki search failed for '%s': %s", query, exc)
    return None


def _download_with_requests(title: str) -> Optional[str]:
    """Download the plain-text extract of a Wikipedia page via API."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exlimit": 1,
        "titles": title,
        "format": "json",
    }
    try:
        response = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        if not pages:
            return None
        # The pages dict key is the pageid – get the first / only value.
        page_data = next(iter(pages.values()))
        content = page_data.get("extract")
        return content  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logging.debug("Could not download page '%s' via API: %s", title, exc)
    return None


def fetch_wikipedia_content(candidates: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Attempt to fetch Wikipedia content given several possible titles.

    Parameters
    ----------
    candidates : list[str]
        Alternative names for the same anime (original, English, Japanese).

    Returns
    -------
    (title, content) where both may be ``None`` if retrieval failed.
    """
    # 1. Try via the optional `wikipedia` library (makes redirects easier).
    if wikipedia is not None:
        for title in candidates:
            if not isinstance(title, str) or title.strip() == "" or title == "N/A":
                continue
            try:
                page = wikipedia.page(title, auto_suggest=True, redirect=True, preload=False)
                return page.title, page.content
            except (DisambiguationError, PageError):
                # Fall back to search method for ambiguous or missing pages
                try:
                    search_results = wikipedia.search(title, results=1)
                    if search_results:
                        page = wikipedia.page(search_results[0], auto_suggest=False)
                        return page.title, page.content
                except Exception:
                    pass  # We will try requests-based approach below.
            except Exception:
                # Network or other error – fall through.
                pass

    # 2. Requests-based fallback (no external dependency).
    for title in candidates:
        if not isinstance(title, str) or title.strip() == "" or title == "N/A":
            continue
        search_title = _search_with_requests(title)
        if search_title is None:
            continue
        content = _download_with_requests(search_title)
        if content:
            return search_title, content

    # Failed to retrieve.
    return None, None

###############################################################################
# Persistence                                                                #
###############################################################################

def save_page(anime_id: int, page_title: str, content: str) -> None:
    """Save raw wiki content to <OUTPUT_DIR>/<anime_id>.txt atomically.

    We first write to <anime_id>.tmp and then move to final path so that a crash
    mid-write does not leave a truncated file that would be mistaken for a
    completed download on the next run.
    """
    final_path = (OUTPUT_DIR / f"{int(anime_id)}.txt").as_posix()
    tmp_path = final_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as fp:
        fp.write(f"# {page_title}\n\n{content}")

    # Atomically replace (or create) the final file.
    os.replace(tmp_path, final_path)

###############################################################################
# Orchestration                                                              #
###############################################################################

def process_anime_row(row: pd.Series) -> None:
    """Fetch and persist Wikipedia content for a single anime DataFrame row."""
    anime_id = int(row["anime_id"])
    output_path = (OUTPUT_DIR / f"{anime_id}.txt").as_posix()

    if os.path.exists(output_path) and os.path.getsize(output_path) >= MIN_VALID_BYTES:
        # Already downloaded – skip silently
        return

    # Gather possible titles in priority order
    title_candidates: List[str] = [
        str(row.get("name", "")),
        str(row.get("title_english", "")),
        str(row.get("title_japanese", "")),
    ]

    page_title, content = fetch_wikipedia_content(title_candidates)
    if content is None:
        logging.warning("Could not find Wikipedia page for anime %s (titles=%s)", anime_id, title_candidates)
        return

    save_page(anime_id, page_title or title_candidates[0], content)
    logging.info("Saved Wikipedia page for anime %s as '%s'", anime_id, page_title)
    time.sleep(RATE_LIMIT_SECONDS)


def fetch_all_wiki_pages() -> None:
    """Main driver: iterate over the dataset and fetch pages one by one."""
    setup_logging()
    ensure_output_dir()

    logging.info("Starting Wikipedia scraping job")
    df = load_anime_catalogue()
    for _, row in df.iterrows():
        try:
            process_anime_row(row)
        except KeyboardInterrupt:
            logging.warning("Interrupted by user – exiting gracefully.")
            break
        except Exception as exc:  # noqa: BLE001 – keep the scraper running
            anime_id = row.get("anime_id", "?")
            logging.exception("Unhandled error while processing anime %s: %s", anime_id, exc)

    logging.info("Finished Wikipedia scraping job")

###############################################################################
# Entry point                                                                #
###############################################################################

if __name__ == "__main__":
    fetch_all_wiki_pages()
