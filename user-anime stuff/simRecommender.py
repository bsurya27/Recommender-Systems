import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load everything
anime_df = pd.read_csv("anime_metadata_cleaned.csv")
combined_vectors = np.load("anime_combined_vectors.npy")

with open("genre_encoder.pkl", "rb") as f:
    mlb = pickle.load(f)

# Recreate ID-to-index map
anime_id_to_index = {aid: idx for idx, aid in enumerate(anime_df["anime_id"])}

def recommend_similar(anime_id, top_n=10, filter_genres=None):
    idx = anime_id_to_index.get(anime_id)
    if idx is None:
        return []

    query_vec = combined_vectors[idx].reshape(1, -1)
    similarities = cosine_similarity(query_vec, combined_vectors)[0]

    # Optional genre filter
    if filter_genres:
        mask = anime_df["genre_list"].apply(lambda genres: any(g in genres for g in filter_genres))
        similarities = similarities * mask.to_numpy()

    # Get top-N results excluding the original anime
    top_idx = similarities.argsort()[::-1]
    top_idx = [i for i in top_idx if i != idx][:top_n]

    results = anime_df.iloc[top_idx][["anime_id", "name", "genre", "synopsis"]].copy()
    results["similarity"] = similarities[top_idx]
    return results

def __main__():
    anime_id = 1535  # Example anime ID
    top_n = 5
    # filter_genres = ["Action", "Adventure"]  # Example genre filter

    recommendations = recommend_similar(anime_id, top_n)
    print(recommendations)

if __name__ == "__main__":
    __main__()