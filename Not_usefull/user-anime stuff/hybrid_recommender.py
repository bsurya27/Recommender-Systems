import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data and models
anime_df = pd.read_csv("anime_metadata_cleaned.csv")
combined_vectors = np.load("anime_combined_vectors.npy")

with open("genre_encoder.pkl", "rb") as f:
    mlb = pickle.load(f)

with open("svd_model_surprise.pkl", "rb") as f:
    svd = pickle.load(f)

# Recreate ID-to-index map
anime_id_to_index = {aid: idx for idx, aid in enumerate(anime_df["anime_id"])}

# --- Helper: Find anime by name (English or original) ---
def find_anime_id_by_name(name):
    name = name.lower()
    for idx, row in anime_df.iterrows():
        if name == str(row["name"]).lower() or name == str(row.get("english_name", "")).lower():
            return row["anime_id"]
    # fallback: partial match
    for idx, row in anime_df.iterrows():
        if name in str(row["name"]).lower() or name in str(row.get("english_name", "")).lower():
            return row["anime_id"]
    return None

# --- Similarity-based recommender ---
def recommend_similar(anime_id, top_n=10, filter_genres=None):
    idx = anime_id_to_index.get(anime_id)
    if idx is None:
        return []
    query_vec = combined_vectors[idx].reshape(1, -1)
    similarities = cosine_similarity(query_vec, combined_vectors)[0]
    if filter_genres:
        mask = anime_df["genre_list"].apply(lambda genres: any(g in genres for g in filter_genres))
        similarities = similarities * mask.to_numpy()
    top_idx = similarities.argsort()[::-1]
    top_idx = [i for i in top_idx if i != idx][:top_n]
    results = anime_df.iloc[top_idx][["anime_id", "name", "genre", "synopsis"]].copy()
    results["similarity"] = similarities[top_idx]
    return results

# --- Collaborative filtering recommender ---
def recommend_cf(user_id, top_n=10):
    # Recommend anime the user hasn't rated, sorted by predicted rating
    watched = set()  # You may want to load user's watched/rated anime here
    try:
        watched = set(svd.trainset.ur[svd.trainset.to_inner_uid(user_id)])
        watched = set(svd.trainset.to_raw_iid(iid) for iid, _ in watched)
    except Exception:
        pass
    candidates = [aid for aid in anime_df["anime_id"] if aid not in watched]
    preds = []
    for aid in candidates:
        try:
            pred = svd.predict(user_id, aid).est
            preds.append((aid, pred))
        except Exception:
            continue
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:top_n]
    results = anime_df[anime_df["anime_id"].isin([aid for aid, _ in top])][["anime_id", "name", "genre", "synopsis"]].copy()
    results["predicted_rating"] = [score for _, score in top]
    return results

# --- Simple CLI ---
def main():
    print("Choose recommendation type:")
    print("1. Find me something new (based on my past watches)")
    print("2. Find me something similar to a given anime")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        user_id = input("Enter your user ID: ").strip()
        recs = recommend_cf(user_id)
        print("\nTop recommendations for you:")
        print(recs[["name", "genre", "synopsis", "predicted_rating"]].head(10))
    elif choice == "2":
        name = input("Enter anime name (English or original): ").strip()
        anime_id = find_anime_id_by_name(name)
        if anime_id is None:
            print("Anime not found.")
            return
        recs = recommend_similar(anime_id)
        print(f"\nTop similar anime to '{name}':")
        print(recs[["name", "genre", "synopsis", "similarity"]].head(10))
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 