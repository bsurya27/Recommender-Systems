# Anime Recommender-Systems

End-to-end playground for building modern recommendation pipelines for anime:

1. **Data Collection**: Scrape real-world data from MyAnimeList (via the public Jikan REST API)
2. **Data Cleaning & EDA**: Clean and analyze anime metadata and user-anime ratings
3. **Feature Engineering**: Impute missing metadata with LLMs (OpenAI) and create hybrid feature vectors
4. **ML Model Development**: Train collaborative filtering and content-based recommendation models
5. **Agent Tools**: Build specialized tools for the AI agent to interact with the recommendation system
6. **Integration**: Combine all components into an intelligent anime recommendation agent

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

## Complete Workflow

### 1. Data Scraping & Collection
```bash
# Scrape anime metadata from MyAnimeList
python ML_dev/scripts/AnimeCollection.py
# Produces: Anime.csv with ~1000s of anime entries

# Scrape user-anime ratings (time-consuming)
python ML_dev/scripts/UserReviewCollection.py  
# Produces: Rating.csv with ~16.5M+ user ratings
```

### 2. Data Cleaning & EDA on Anime Metadata
- **Dataset Size**: 28,467 unique anime entries with 19 attributes
- **Genre Analysis**: 927 unique genre combinations, top genres include Comedy (7,688), Fantasy (5,881), Action (5,521)
- **Missing Data Handling**: Processed 714 missing studio entries, 63 missing synopses, 46 missing genres
- **Feature Engineering**: Extracted 9 anime types, 17 source types, 3 status categories
- **Output**: Cleaned anime metadata with standardized formats for ML models

### 3. Data Cleaning & EDA on User-Anime Ratings
- **Initial Dataset**: 16,573,880 user-anime rating records
- **Bot Detection**: Removed 354,038 troll users (users who only watched 1 episode)
- **Suspicious Pattern Removal**: Filtered out 6,471 suspicious entries with unrealistic episode counts
- **User Quality Filtering**: Removed 530,472 low-activity users (<10 reviews) and 164,304 users with low rating variance
- **Final Dataset**: 1,900,982 clean user-anime ratings from 27,318 unique anime
- **Output**: Quality-filtered dataset with minimum 76 reviews per anime for reliable recommendations

### 4. Machine Learning Model Development

#### Content-Based Filtering
```bash
# Create anime feature vectors
python ML_dev/user-anime stuff/AnimeVectorizer.py
# Produces: anime_combined_vectors.npy, genre_encoder.pkl
```

#### Collaborative Filtering
```bash
# Train SVD-based collaborative filtering model
python ML_dev/user-anime stuff/cf_train.py
# Produces: svd_model_surprise.pkl
```

#### Hybrid Recommender
```bash
# Combine both approaches
python ML_dev/user-anime stuff/hybrid_recommender.py
```

### 5. Agent Tools Development
- **Content-Based Recommendation Tool**: `recommend_similar_anime()` - Uses cosine similarity with pre-computed embeddings to find similar anime based on synopsis and genre vectors
- **Collaborative Filtering Tool**: `collaborative_filtering_recommend()` - Uses pre-trained SVD model to predict user ratings and recommend anime based on similar user preferences
- **Search & Discovery Tools**: 
  - `get_anime_id_by_name()` - Find anime by name (English or Japanese)
  - `get_anime_ids_by_genre()` - Filter anime by genre keywords
  - `search_anime_ids_by_synopsis()` - Search anime by synopsis content
  - `get_anime_ids_before_year()` / `get_anime_ids_after_year()` - Filter by release year
- **Data Analysis Tools**: 
  - `get_anime_details()` - Retrieve full metadata for specific anime IDs
  - `recommend_anime()` - Genre-based recommendations using rating-based ranking
- **Integration Framework**: 9 specialized tools enabling the agent to perform complex recommendation tasks with both ML models and metadata analysis

### 6. Agent Integration
- **Intelligent Agent Architecture**: Combines OpenAI LLM with 9 specialized tools for natural language interaction
- **Multi-Model Recommendations**: Seamlessly switches between content-based (cosine similarity) and collaborative filtering (SVD) approaches
- **Interactive User Experience**: Handles natural language queries like "Find me anime similar to Death Note" or "Recommend action anime from the 2010s"
- **Advanced Search Capabilities**: Supports complex queries combining genre, year, synopsis keywords, and similarity searches
- **Fallback Mechanisms**: Graceful handling when ML models are unavailable, falling back to metadata-based recommendations
```bash
# Run the intelligent anime recommendation agent
python app.py
```

---

## Repository Layout

```
ML_dev/                         # Pre-agent development work
├── scripts/                     # Data collection utilities
│   ├── AnimeCollection.py       # Scrape anime catalogue → Anime.csv
│   ├── UserReviewCollection.py  # Download user ratings → Rating.csv
│   ├── gemini_synopsis_imputer.py  # Generate missing synopses (LLM)
│   ├── genre_imputer.py        # Assign genres from synopsis (LLM)
│   └── synopsis_imputer.py     # Synopsis processing utilities
├── user-anime stuff/           # ML model development
│   ├── clean-preprocess.ipynb  # EDA and data cleaning
│   ├── AnimeVectorizer.py      # Create feature vectors
│   ├── cf_train.py            # Train collaborative filtering
│   ├── hybrid_recommender.py  # Combined recommendation system
│   └── simRecommender.py      # Similarity-based recommendations
├── RAG_Data_creation/         # Wikipedia integration
│   ├── web_fetcher.py         # Wikipedia page scraping
│   ├── rag_pipeline.py        # RAG processing pipeline
│   ├── rag_creation.py        # RAG creation utilities
│   ├── requirements.txt       # Dependencies
│   └── wiki_meta.json.gz     # Processed Wikipedia data
└── AnimeDataCleaning.ipynb    # Main data cleaning notebook
```

# Agent-related files

```
├── app.py                      # Main agent application
├── Agent.py                    # Agent implementation
├── llm_agent.py               # LLM integration
├── tools.py                    # Agent tools
├── prompt.py                   # Agent prompts
└── interactive_recommender.py  # Rudimentary recommendation interface

# Data and generated files
├── dataset/                    # Raw & intermediate CSVs
├── generated_synopses/         # LLM-generated synopsis files
├── generated_genres/           # LLM-generated genre files
└── RAG_Data_creation/         # Wikipedia scraping utilities
---

## Data Pipeline Details

### Data Collection Phase
1. **Anime Metadata Scraping**: Collect anime information from MyAnimeList API
2. **User Ratings Collection**: Gather user-anime rating data with rate limiting
3. **Wikipedia Integration**: Fetch Wikipedia pages for enhanced context

### Data Cleaning & EDA Phase
1. **Anime Data Analysis**: Explore genres, ratings, popularity patterns
2. **User Data Cleaning**: Remove bots, handle missing values, analyze rating distributions
3. **Feature Engineering**: Create hybrid vectors combining text embeddings and categorical features

### ML Model Development Phase
1. **Content-Based Filtering**: Use TF-IDF and sentence embeddings for similarity
2. **Collaborative Filtering**: Implement SVD matrix factorization
3. **Hybrid Approach**: Combine both methods for improved recommendations

### Agent Integration Phase
1. **Tool Development**: Create specialized tools for recommendation tasks
2. **Agent Architecture**: Build intelligent agent with access to ML models
3. **User Interface**: Provide interactive recommendation experience

---

## Technologies Used

- **Data Collection**: Python, Jikan API, requests, pandas
- **Data Processing**: pandas, numpy, scikit-learn, sentence-transformers
- **Machine Learning**: Surprise library, SVD, cosine similarity, TF-IDF
- **Natural Language Processing**: SentenceTransformers, MultiLabelBinarizer
- **Agent Development**: OpenAI API, custom tools, interactive interfaces
- **Visualization**: matplotlib, seaborn
- **Data Storage**: CSV, pickle, numpy arrays
