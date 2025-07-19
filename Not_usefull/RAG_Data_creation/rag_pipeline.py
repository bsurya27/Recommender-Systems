"""RAG (Retrieval-Augmented Generation) data preparation utilities.

This script reads Wikipedia text files that were downloaded by ``web_fetcher.py``
( one file per anime, named <anime_id>.txt ),
merges them with the anime catalogue, splits the text into manageable chunks,
embeds them with a SentenceTransformer model, and finally builds an in-memory
FAISS index that can be used to retrieve relevant text given a natural-language
query.

All steps are broken into reusable functions so you can import this module from
notebooks or other scripts, or run it directly to build the index and test a
sample query.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore
import json, gzip

################################################################################
# Paths & configuration                                                         #
################################################################################

BASE_DIR: Path = Path(__file__).resolve().parent
WIKI_FOLDER: Path = BASE_DIR / "wiki_pages"  # Matches OUTPUT_DIR in web_fetcher
# CATALOGUE_CSV: Path = BASE_DIR.parent / "anime.csv"  # lower-case preferred

# if not CATALOGUE_CSV.exists():
CATALOGUE_CSV: Path = BASE_DIR / "Anime.csv"  # Path to anime metadata CSV

CHUNK_SIZE: int = 512  # words per chunk
MODEL_NAME: str = "all-MiniLM-L6-v2"

################################################################################
# Loading utilities                                                             #
################################################################################

def load_wiki_texts_by_id(folder_path: str | Path = WIKI_FOLDER) -> Dict[int, str]:
    """Return a mapping {anime_id: full_wiki_text}."""
    folder = Path(folder_path)
    id_to_text: Dict[int, str] = {}
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} does not exist – run web_fetcher first.")

    for file in folder.iterdir():
        if file.suffix.lower() == ".txt":
            try:
                anime_id = int(file.stem)
            except ValueError:
                # Skip files that are not pure integers
                continue
            id_to_text[anime_id] = file.read_text(encoding="utf-8")
    return id_to_text


def load_anime_catalogue(csv_path: str | Path = CATALOGUE_CSV) -> pd.DataFrame:
    """Load the anime metadata CSV (expects at least 'anime_id' & 'name' columns)."""
    df = pd.read_csv(csv_path)
    if not {"anime_id", "name"}.issubset(df.columns):
        raise ValueError("anime_id and name columns are required in the catalogue CSV")
    return df

################################################################################
# Text processing & embedding                                                   #
################################################################################

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split *text* into chunks of up to *chunk_size* words (simple whitespace split)."""
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def embed_chunks(chunks: List[str], model: SentenceTransformer) -> List[np.ndarray]:
    """Compute embeddings for each chunk and return a list of numpy arrays."""
    if not chunks:
        return []
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=False)
    # Ensure each embedding is a float32 row-vector
    return [np.asarray(vec, dtype="float32") for vec in embeddings]

################################################################################
# Index construction                                                            #
################################################################################

def build_faiss_index(df: pd.DataFrame, model: SentenceTransformer) -> tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    """Create a FAISS L2 index and accompanying metadata list.

    Returns
    -------
    index : faiss.IndexFlatL2
        Dense vector index containing one entry per text chunk.
    metadata : list[dict]
        Parallel array where metadata[i] describes the chunk at index *i* in the FAISS index.
    """
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)  # L2 distance on raw vectors
    metadata: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        for chunk, emb in zip(row["chunks"], row["embeddings"]):
            index.add(np.expand_dims(emb, axis=0))
            metadata.append({
                "anime_id": int(row["anime_id"]),
                "name": row["name"],
                "chunk": chunk,
            })
    return index, metadata

################################################################################
# High-level orchestration                                                      #
################################################################################

def build_index_pipeline() -> tuple[pd.DataFrame, SentenceTransformer, faiss.IndexFlatL2, List[Dict[str, Any]]]:
    """Full pipeline: load data, embed, build index."""
    print("Loading metadata & wiki texts…")
    df = load_anime_catalogue()
    wiki_texts = load_wiki_texts_by_id(WIKI_FOLDER)
    df["wiki_text"] = df["anime_id"].map(wiki_texts)

    print("Chunking texts…")
    df["chunks"] = df["wiki_text"].apply(lambda x: chunk_text(x) if isinstance(x, str) else [])

    print(f"Loading model {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding chunks… (this might take a while)")
    df["embeddings"] = df["chunks"].apply(lambda lst: embed_chunks(lst, model))

    print("Building FAISS index…")
    index, meta = build_faiss_index(df, model)
    print(f"Indexed {len(meta)} chunks.")

    # After index, meta are created
    faiss.write_index(index, str(BASE_DIR / "wiki_index.faiss"))

    with gzip.open(BASE_DIR / "wiki_meta.json.gz", "wt", encoding="utf-8") as fp:
        json.dump(meta, fp)

    return df, model, index, meta

################################################################################
# Retrieval helper                                                              #
################################################################################

def retrieve_context(query: str, model: SentenceTransformer, index: faiss.IndexFlatL2, metadata: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve *top_k* most similar chunks for the *query*."""
    query_vec = model.encode(query).astype("float32")
    D, I = index.search(np.expand_dims(query_vec, axis=0), top_k)
    return [metadata[i] for i in I[0]]

################################################################################
# CLI entry-point                                                               #
################################################################################

if __name__ == "__main__":
    df, model, index, meta = build_index_pipeline()
    # Simple interactive demo
    print("\nType a query to retrieve context (empty line to exit).\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        results = retrieve_context(query, model, index, meta, top_k=5)
        for res in results:
            snippet = res["chunk"][:200].replace("\n", " ")
            print(f"- {res['name']} (ID {res['anime_id']}) :: {snippet}…\n")

    index = faiss.read_index(str(BASE_DIR / "wiki_index.faiss"))
    with gzip.open(BASE_DIR / "wiki_meta.json.gz", "rt", encoding="utf-8") as fp:
        metadata = json.load(fp) 