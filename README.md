# Anime Recommender-Systems

End-to-end playground for building modern recommendation pipelines for anime:

1. Collect real-world data from MyAnimeList (via the public Jikan REST API)
2. Impute missing metadata with LLMs (OpenAI)
3. Fetch Wikipedia pages to provide context for LLM-based synopsis & genre imputation
4. Train & evaluate collaborative-filtering / hybrid models
5. Analyse results and iterate in notebooks

---

## Quick start

```bash
# clone & enter the repo
python -m venv .venv && . .venv/bin/activate   # or your preferred way
pip install -r requirements.txt
# optional extras for the RAG pipeline
pip install -r RAG_Data_creation/requirements.txt
```

---

## Repository layout (high-level)

```
scripts/                  one-off data-collection utilities
├── AnimeCollection.py      ─ scrape anime catalogue → Anime.csv
├── UserReviewCollection.py ─ download user ratings → Rating.csv
├── gemini_synopsis_imputer.py  ─ generate missing synopses (LLM)
└── genre_imputer.py            ─ assign genres from synopsis (LLM)

RAG_Data_creation/        Wikipedia scraping utilities (+ optional FAISS index for RAG experiments)
├── web_fetcher.py        ─ download Wikipedia pages per anime → wiki_pages/*.txt
├── rag_pipeline.py       ─ chunk, embed & index wiki pages with FAISS
└── requirements.txt      ─ extra deps (faiss, sentence-transformers)

analysis.ipynb            interactive exploration, modelling & visualisation
Collaborative Filtering/  theoretical notes & experiments

dataset/                  raw & intermediate CSVs (Anime.csv, Rating.csv …)
generated_synopses/       LLM-generated synopsis files (generated)
generated_genres/         LLM-generated genre files (generated)
```

---

## Typical data pipeline

1. **Scrape the master catalogue**
   ```bash
   python scripts/AnimeCollection.py
   ```
   Produces `Anime.csv` with ~1000s of entries.

2. **Scrape user ratings** (optional, time-consuming)
   ```bash
   python scripts/UserReviewCollection.py
   ```
   Produces `Rating.csv`.

3. **Fetch Wikipedia corpus (for synopsis & genre imputation)**
   ```bash
   python RAG_Data_creation/web_fetcher.py   # downloads wiki_pages/*.txt per anime
   ```
   *(Optional)* If you want to experiment with retrieval-augmented-generation later, you can also run:
   ```bash
   python RAG_Data_creation/rag_pipeline.py  # builds wiki_index.faiss & wiki_meta.json.gz
   ```

4. **Fill missing synopsis & genre** (requires `OPENAI_API_KEY`)
   ```bash
   export OPENAI_API_KEY=sk-...
   python scripts/gemini_synopsis_imputer.py
   python scripts/genre_imputer.py
   ```

---
