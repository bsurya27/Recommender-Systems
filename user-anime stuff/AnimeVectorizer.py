import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import ast
import re
import pickle
# Load data
anime_df = pd.read_csv("..\somewhatcleanedAnime.csv")  # should include synopsis + genres

# Clean synopsis
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

anime_df["cleaned_synopsis"] = anime_df["synopsis"].apply(clean_text)

# Convert genre strings into list
def parse_genre(genre_str):
    try:
        return [g.strip() for g in genre_str.split(',')]
    except:
        return []

print("Parsing genres...")
anime_df["genre_list"] = anime_df["genre"].apply(parse_genre)

# Load sentence-transformer
print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate sentence embeddings for synopses
print("Generating synopsis embeddings...")
anime_df["synopsis_embedding"] = model.encode(
    anime_df["cleaned_synopsis"].tolist(),
    show_progress_bar=True
).tolist()

# Multi-hot encode genres
mlb = MultiLabelBinarizer()
genre_onehot = mlb.fit_transform(anime_df["genre_list"])

# Save the genre column names for reference
genre_columns = mlb.classes_

# Convert list of embeddings into ndarray
synopsis_embeddings = np.array(anime_df["synopsis_embedding"].tolist())
print(f"Synopsis embeddings shape: {synopsis_embeddings.shape}")
print(f"Genre one-hot shape: {genre_onehot.shape}")


# Combine synopsis embeddings + genre one-hot vectors
combined_vectors = np.concatenate([synopsis_embeddings, genre_onehot], axis=1)
print('Saving combined vectors to "anime_combined_vectors.npy"...')
np.save("anime_combined_vectors.npy", combined_vectors)

anime_df.drop(columns=["synopsis_embedding"], errors="ignore", inplace=True)
anime_df.to_csv("anime_metadata_cleaned.csv", index=False)

print('Dumping')
# Save genre one-hot encoder (MultiLabelBinarizer)
with open("genre_encoder.pkl", "wb") as f:
    pickle.dump(mlb, f)

print("Done processing anime data.")