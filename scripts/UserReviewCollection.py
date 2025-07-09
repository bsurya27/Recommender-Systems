import os
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# === Config ===
ANIME_CSV = "Anime.csv"
OUTPUT_CSV = "Rating.csv"
ERROR_CSV = "review_errors.csv"
LOG_FILE = "review_scrape.log"
RATE_LIMIT_DELAY = 1  # seconds
USERUPDATES_PER_ANIME = 750
USERUPDATES_PER_PAGE = 75  # based on Jikan default

# === Ensure output CSV exists ===
columns = ["user_id", "anime_id", "score", "status", "episodes_seen"]
if not os.path.exists(OUTPUT_CSV):
    pd.DataFrame(columns=columns).to_csv(OUTPUT_CSV, index=False)

# Load processed anime IDs
processed_ids = set(pd.read_csv(OUTPUT_CSV)["anime_id"].unique())

# Load anime list
anime_list = pd.read_csv(ANIME_CSV)["anime_id"].unique().tolist()

# Logging
def log(msg):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts} {msg}\n")
        f.flush()
    print(msg)

# Scrape up to N userupdates for a given anime
def scrape_userupdates(anime_id):
    collected = []
    page = 1
    max_pages = int(np.ceil(USERUPDATES_PER_ANIME / USERUPDATES_PER_PAGE))

    while page <= max_pages:
        try:
            url = f"https://api.jikan.moe/v4/anime/{anime_id}/userupdates?page={page}"
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Status {response.status_code}: {response.text}")

            users = response.json().get("data", [])
            if not users:
                break

            for user in users:
                uid = user.get("user", {}).get("username")
                if uid is None:
                    continue

                row = {
                    "user_id": uid,
                    "anime_id": anime_id,
                    "score": user.get("score", -1),
                    "status": user.get("status", "N/A"),
                    "episodes_seen": user.get("episodes_seen", -1)
                }
                collected.append(row)

                if len(collected) >= USERUPDATES_PER_ANIME:
                    break

            if len(collected) >= USERUPDATES_PER_ANIME:
                break

            page += 1
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            log(f"❌ Error fetching anime {anime_id} page {page}: {e}")
            time.sleep(5)
            break

    return collected

# Main loop
def scrape_all_userupdates():
    for anime_id in tqdm(anime_list, desc="Scraping userupdates", ncols=100):
        if anime_id in processed_ids:
            continue

        userupdates = scrape_userupdates(anime_id)
        if userupdates:
            pd.DataFrame(userupdates).to_csv(OUTPUT_CSV, mode="a", index=False, header=False)
            log(f"✅ Saved {len(userupdates)} userupdates for anime {anime_id}")
        else:
            pd.DataFrame([{"anime_id": anime_id}]).to_csv(
                ERROR_CSV,
                mode="a",
                index=False,
                header=not os.path.exists(ERROR_CSV) or os.path.getsize(ERROR_CSV) == 0
            )
            log(f"❌ Logged anime ID {anime_id} to review_errors.csv")

        time.sleep(RATE_LIMIT_DELAY)

if __name__ == "__main__":
    log("🚀 Starting userupdates scraping job")
    scrape_all_userupdates()
    log("✅ Finished userupdates scraping job")
