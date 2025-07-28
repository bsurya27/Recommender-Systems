import time
import os
import pandas as pd
from jikanpy import Jikan
from tqdm import tqdm
from datetime import datetime

# === Setup ===
jikan = Jikan()
CSV_FILE = "Anime.csv"
LOG_FILE = "anime_scrape.log"
PAGE_LIMIT = 1500  # Adjust to go beyond top 50 pages (25 anime/page)

columns = [
    "anime_id", "name", "title_english", "title_japanese", "type", "episodes", "rating",
    "rank", "popularity", "members", "status", "source", "duration",
    "aired_from", "aired_to", "genre", "studio", "synopsis"
]

# === Ensure CSV File Exists ===
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=columns).to_csv(CSV_FILE, index=False)

# === Load Already Fetched IDs ===
existing_df = pd.read_csv(CSV_FILE)
existing_ids = set(existing_df["anime_id"])

# === Logging ===
def log(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")
    print(message)

# === Fetch Anime Data ===
def fetch_anime_data(anime_id):
    try:
        response = jikan.anime(anime_id)
        data = response.get("data", {})

        genres = ", ".join([genre["name"] for genre in data.get("genres", [])])
        studios = ", ".join([studio["name"] for studio in data.get("studios", [])])

        return {
            "anime_id": anime_id,
            "name": data.get("title", "N/A"),
            "title_english": data.get("title_english", "N/A"),
            "title_japanese": data.get("title_japanese", "N/A"),
            "type": data.get("type", "N/A"),
            "episodes": data.get("episodes", -1),
            "rating": data.get("score", -1),
            "rank": data.get("rank", -1),
            "popularity": data.get("popularity", -1),
            "members": data.get("members", -1),
            "status": data.get("status", "N/A"),
            "source": data.get("source", "N/A"),
            "duration": data.get("duration", "N/A"),
            "aired_from": data.get("aired", {}).get("from", "N/A"),
            "aired_to": data.get("aired", {}).get("to", "N/A"),
            "genre": genres,
            "studio": studios,
            "synopsis": (data.get("synopsis", "N/A")[:500] + "...") if data.get("synopsis") else "N/A"
        }

    except Exception as e:
        log(f"❌ Error fetching anime ID {anime_id}: {e}")
        return None

# === Main Scraping Loop ===
def scrape_top_anime():
    for page in range(1, PAGE_LIMIT + 1):
        log(f"\n📄 Fetching page {page} of top anime...")
        try:
            top_anime = jikan.top(type='anime', page=page)
        except Exception as e:
            log(f"❌ Error fetching top anime list (page {page}): {e}")
            time.sleep(5)
            continue

        for entry in tqdm(top_anime['data'], desc=f"Page {page}", ncols=100):
            anime_id = entry['mal_id']
            if anime_id in existing_ids:
                log(f"⚠️ Skipping anime ID {anime_id} (already saved)")
                continue

            data = fetch_anime_data(anime_id)
            if data:
                pd.DataFrame([data]).to_csv(CSV_FILE, mode='a', index=False, header=False)
                log(f"✅ Saved {data['name']} (ID: {anime_id})")
                existing_ids.add(anime_id)

            time.sleep(1)  # Respect rate limit

# === Run ===
if __name__ == "__main__":
    log("🔁 Starting scrape session")
    scrape_top_anime()
    log("✅ Scraping complete")
