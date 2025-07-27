import pandas as pd
from collections import Counter
from typing import List
from langchain_core.tools import tool

# -------------------- Advanced recommender (similarity based) --------------------
# We lazily import and initialize the heavy `InteractiveAnimeRecommender` once so that
# subsequent calls are fast.  The recommender lives in
# `Not_usefull/user-anime stuff/interactive_recommender.py` and loads several large
# assets (SVD model, combined vectors, etc.).

_adv_recommender = None  # type: ignore


def _load_adv_recommender():
    """Return a cached `InteractiveAnimeRecommender` instance.

    Strategy:
    1. Try regular import `import interactive_recommender` assuming the file is in
       the PYTHONPATH / project root.
    2. Fallback: manually load from the historical location under
       `Not_usefull/user-anime stuff/interactive_recommender.py`.
    """
    global _adv_recommender
    if _adv_recommender is not None:
        return _adv_recommender

    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    # Attempt 1: plain import
    try:
        module = importlib.import_module("interactive_recommender")
    except ModuleNotFoundError:
        # Attempt 2: legacy path
        proj_root = Path(__file__).resolve().parent
        legacy_path = proj_root / "Not_usefull" / "user-anime stuff" / "interactive_recommender.py"
        if not legacy_path.exists():
            raise ImportError(
                "interactive_recommender module not found in current directory or legacy path"
            )
        spec = importlib.util.spec_from_file_location("interactive_recommender", legacy_path)
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load interactive_recommender module from legacy path")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[attr-defined]

    InteractiveAnimeRecommender = getattr(module, "InteractiveAnimeRecommender")
    _adv_recommender = InteractiveAnimeRecommender()
    _adv_recommender.load_data()
    return _adv_recommender


@tool
def recommend_similar_anime(anime_name: str, top_n: int = 10) -> List[dict]:
    """Return up to `top_n` anime that are most similar to the given *anime_name*.

    This tool uses a pre-computed embedding + metadata index (see
    `interactive_recommender.py`) to find titles with the highest cosine similarity
    to the query anime. It searches both original and English names, handling null values properly.
    If the anime cannot be found, an empty list is returned.

    Args:
        anime_name: Title of an anime (either English or original) to base the
            similarity search on. Works with names like "Death Note", "Attack on Titan", etc.
        top_n: Maximum number of similar anime to return (default 10).

    Returns:
        A list of dictionaries.  Each dictionary contains:
            - "anime_id": ID of the recommended anime (str)
            - "name": Title of the recommended anime (str)
            - "genre": Genre string (str)
            - "synopsis": Short synopsis (str)
            - "similarity": Cosine similarity score (float between 0-1)
    """
    print("🔧 Using tool: recommend_similar_anime")
    recommender = _load_adv_recommender()
    anime_id = recommender.find_anime_id_by_name(anime_name)
    if anime_id is None:
        print(f"⚠️  Anime '{anime_name}' not found in database")
        return []

    print(f"✅ Found anime ID {anime_id} for '{anime_name}'")
    df_recs = recommender.recommend_similar(anime_id, top_n=top_n)
    # Convert DataFrame rows to list[dict]
    return df_recs.to_dict(orient="records")

_csv_path = "Data/anime_clean.csv"
_df_cache = None

def _df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(_csv_path)
    return _df_cache

@tool
def get_anime_ids_before_year(year: int) -> List[int]:
    """Return anime IDs released before a given year.

    Args:
        year (int): Only anime with `aired_from_year` strictly less than this value are selected.

    Returns:
        List[int]: List of matching `anime_id` integers.
    """
    print("🔧 Using tool: get_anime_ids_before_year")
    df = _df()
    return df[(df['aired_from_year'].notna()) & (df['aired_from_year'] < year)]['anime_id'].tolist()

@tool
def get_anime_ids_after_year(year: int) -> List[int]:
    """Return anime IDs released after a given year.

    Args:
        year (int): Only anime with `aired_from_year` strictly greater than this value are selected.

    Returns:
        List[int]: List of matching `anime_id` integers.
    """
    print("🔧 Using tool: get_anime_ids_after_year")
    df = _df()
    return df[(df['aired_from_year'].notna()) & (df['aired_from_year'] > year)]['anime_id'].tolist()

@tool
def get_anime_ids_by_genre(genre: str) -> List[int]:
    """Return anime IDs whose genre text contains a keyword.

    Args:
        genre (str): Case-insensitive keyword to search within the `genre` column.

    Returns:
        List[int]: List of matching `anime_id` integers.
    """
    print("🔧 Using tool: get_anime_ids_by_genre")
    df = _df()
    mask = df['genre'].fillna('').str.contains(genre, case=False, na=False)
    return df[mask]['anime_id'].tolist()

@tool
def search_anime_ids_by_synopsis(keyword: str) -> List[int]:
    """Return anime IDs whose synopsis contains a keyword.

    Args:
        keyword (str): Case-insensitive search term applied to the `synopsis` column.

    Returns:
        List[int]: List of matching `anime_id` integers.
    """
    print("🔧 Using tool: search_anime_ids_by_synopsis")
    df = _df()
    mask = df['synopsis'].fillna('').str.contains(keyword, case=False, na=False)
    return df[mask]['anime_id'].tolist()

@tool
def get_anime_details(anime_ids: List[int]) -> pd.DataFrame:
    """Return full rows from the dataset for a set of IDs.

    Args:
        anime_ids (List[int]): One or more `anime_id` values.

    Returns:
        pandas.DataFrame: DataFrame containing all columns for the requested IDs.
    """
    print("🔧 Using tool: get_anime_details")
    df = _df()
    return df[df['anime_id'].isin(anime_ids)].reset_index(drop=True)


@tool
def recommend_anime(anime_ids: List[int]) -> List[int]:
    """Recommend anime based on a set of IDs.

    Args:
        anime_ids (List[int]): One or more `anime_id` values.

    Returns:
        List[int]: List of recommended `anime_id` integers.
    """
    print("🔧 Using tool: recommend_anime")
    # Get the base dataframe
    df = _df()

    # Get all anime details for the input IDs
    input_anime = df[df['anime_id'].isin(anime_ids)]

    # Extract genres from input anime
    input_genres = []
    for genres in input_anime['genre'].fillna('').str.split(','):
        input_genres.extend([g.strip() for g in genres if g.strip()])

    # Count genre frequencies
    genre_counts = Counter(input_genres)
    top_genres = [g for g,_ in genre_counts.most_common(3)]

    # Find similar anime based on genres
    similar_mask = df['genre'].fillna('').apply(
        lambda x: any(g in x for g in top_genres)
    )

    # Filter out input anime and get top rated matches
    recommendations = df[similar_mask & ~df['anime_id'].isin(anime_ids)]
    recommendations = recommendations.nlargest(10, 'rating')

    # Return recommended anime IDs
    return recommendations['anime_id'].tolist()




@tool
def get_anime_id_by_name(anime_name: str) -> int:
    """Get anime ID by name (searches both original and English names).
    
    Args:
        anime_name: Name of the anime to look up
        
    Returns:
        Anime ID (int) or None if not found
    """
    print("🔧 Using tool: get_anime_id_by_name")
    recommender = _load_adv_recommender()
    anime_id = recommender.find_anime_id_by_name(anime_name)
    return anime_id


@tool
def collaborative_filtering_recommend(user_ratings: dict, top_n: int = 10) -> pd.DataFrame:
    """Recommend anime using collaborative filtering based on user ratings.

    This tool uses a pre-trained SVD (Singular Value Decomposition) model to predict
    ratings for anime the user hasn't seen, based on their ratings of other anime.

    Args:
        user_ratings: Dictionary mapping anime_id (str) to rating (int 1-10)
        top_n: Maximum number of recommendations to return (default 10)

    Returns:
        pandas.DataFrame: DataFrame with columns ["anime_id", "name", "genre", "predicted_rating"]
    """
    print("🔧 Using tool: collaborative_filtering_recommend")
    
    # Get the advanced recommender with SVD model
    recommender = _load_adv_recommender()
    
    if recommender.svd is None:
        # Fallback to content-based if SVD not available
        print("⚠️  SVD model not available, falling back to content-based recommendations")
        anime_ids = [int(aid) for aid in user_ratings.keys()]
        return recommend_anime(anime_ids)
    
    # Extract liked genres from high-rated anime (rating >= 7)
    user_liked_genres = set()
    for anime_id, rating in user_ratings.items():
        if rating >= 7:
            try:
                genres = recommender.anime_info.loc[int(anime_id), 'genre']
                if pd.notna(genres):
                    for g in str(genres).split(','):
                        user_liked_genres.add(g.strip())
            except KeyError:
                continue
    
    # Get recommendations using the existing collaborative filtering method
    recommendations = recommender.recommend(user_ratings, user_liked_genres, n_recommendations=top_n)
    
    # Convert to DataFrame format
    if recommendations:
        df_recs = pd.DataFrame(recommendations)
        df_recs = df_recs.rename(columns={'score': 'predicted_rating'})
        return df_recs[["anime_id", "name", "genre", "predicted_rating"]]
    else:
        return pd.DataFrame(columns=["anime_id", "name", "genre", "predicted_rating"])




tools_list = [
    recommend_similar_anime,
    get_anime_ids_before_year,
    get_anime_ids_after_year,
    get_anime_ids_by_genre,
    search_anime_ids_by_synopsis,
    get_anime_details,
    recommend_anime,
    get_anime_id_by_name,
    collaborative_filtering_recommend
]