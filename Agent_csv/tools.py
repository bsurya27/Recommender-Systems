import pandas as pd
from langchain_core.tools import tool
from typing import List, Dict, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import re

csv_path = "../Suprise_reccomend/anime_clean.csv"

# Cache for expensive operations
_df_cache = None
_similarity_cache = None

def _get_dataframe():
    """Get cached dataframe to avoid repeated CSV reads"""
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(csv_path)
    return _df_cache

def _get_similarity_matrix():
    """Get cached similarity matrix for content-based recommendations"""
    global _similarity_cache
    if _similarity_cache is None:
        df = _get_dataframe()
        # Create content features for similarity
        df_clean = df.copy()
        for col in ['genre', 'studio', 'synopsis', 'source', 'type']:
            df_clean[col] = df_clean[col].fillna('')
        
        # Combine features
        df_clean['content'] = (
            df_clean['genre'] + ' ' + 
            df_clean['studio'] + ' ' + 
            df_clean['source'] + ' ' + 
            df_clean['type'] + ' ' + 
            df_clean['synopsis']
        )
        
        # TF-IDF and cosine similarity
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df_clean['content'])
        _similarity_cache = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return _similarity_cache

@tool
def get_anime_ids_after_year(year: int) -> List[int]:
    """
    Get all anime IDs for anime that aired after the specified year.
    
    Args:
        year (int): The year to filter by. Returns anime that aired after this year.
        
    Returns:
        List[int]: A list of anime IDs for anime that aired after the given year.
    """
    try:
        df = _get_dataframe()
        filtered_df = df[(df['aired_from_year'].notna()) & (df['aired_from_year'] > year)]
        anime_ids = filtered_df['anime_id'].tolist()
        return anime_ids
    except Exception as e:
        print(f"Error reading CSV or filtering data: {e}")
        return []

@tool
def get_anime_details_by_ids(anime_ids: List[int]) -> pd.DataFrame:
    """
    Get complete anime information for a list of anime IDs.
    
    Args:
        anime_ids (List[int]): List of anime IDs to get information for.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing all information about the specified anime.
                     Returns empty DataFrame if no anime found or error occurs.
    """
    try:
        df = _get_dataframe()
        filtered_df = df[df['anime_id'].isin(anime_ids)]
        filtered_df = filtered_df.sort_values('anime_id').reset_index(drop=True)
        return filtered_df
    except Exception as e:
        print(f"Error reading CSV or filtering data: {e}")
        return pd.DataFrame()

@tool
def search_anime_ids_by_name(search_query: str) -> List[int]:
    """
    Search for anime IDs by name using a search query.
    Searches through both the original name and English title columns.
    
    Args:
        search_query (str): The search term to look for in anime names.
                           Case-insensitive partial matching is used.
        
    Returns:
        List[int]: A list of anime IDs that match the search query.
                  Returns empty list if no matches found or error occurs.
    """
    try:
        df = _get_dataframe()
        search_lower = search_query.lower()
        
        name_match = df['name'].fillna('').str.lower().str.contains(search_lower, na=False)
        title_english_match = df['title_english'].fillna('').str.lower().str.contains(search_lower, na=False)
        
        matches = df[name_match | title_english_match]
        anime_ids = matches['anime_id'].tolist()
        return anime_ids
    except Exception as e:
        print(f"Error reading CSV or searching data: {e}")
        return []

@tool
def search_anime_by_genre(genres: List[str], match_all: bool = False) -> List[int]:
    """
    Find anime by genre(s). Can match any genre or all genres.
    
    Args:
        genres (List[str]): List of genres to search for (e.g., ["Action", "Drama"])
        match_all (bool): If True, anime must have ALL genres; if False, ANY genre
        
    Returns:
        List[int]: A list of anime IDs that match the genre criteria.
    """
    try:
        df = _get_dataframe()
        df_clean = df.copy()
        df_clean['genre'] = df_clean['genre'].fillna('')
        
        if match_all:
            # Must contain all genres
            mask = df_clean['genre'].str.len() > 0
            for genre in genres:
                mask = mask & df_clean['genre'].str.contains(genre, case=False, na=False)
        else:
            # Must contain any genre
            mask = pd.Series([False] * len(df_clean))
            for genre in genres:
                mask = mask | df_clean['genre'].str.contains(genre, case=False, na=False)
        
        matches = df_clean[mask]
        return matches['anime_id'].tolist()
    except Exception as e:
        print(f"Error searching by genre: {e}")
        return []

@tool
def get_anime_by_rating_range(min_rating: float, max_rating: float = 10.0) -> List[int]:
    """
    Find anime within a specific rating range.
    
    Args:
        min_rating (float): Minimum rating (e.g., 8.0)
        max_rating (float): Maximum rating (default: 10.0)
        
    Returns:
        List[int]: A list of anime IDs within the rating range.
    """
    try:
        df = _get_dataframe()
        filtered_df = df[
            (df['rating'].notna()) & 
            (df['rating'] >= min_rating) & 
            (df['rating'] <= max_rating)
        ]
        return filtered_df['anime_id'].tolist()
    except Exception as e:
        print(f"Error filtering by rating: {e}")
        return []

@tool
def get_popular_anime(min_members: int = 100000, limit: int = 50) -> List[int]:
    """
    Get popular anime based on member count.
    
    Args:
        min_members (int): Minimum number of members (default: 100000)
        limit (int): Maximum number of results to return (default: 50)
        
    Returns:
        List[int]: A list of popular anime IDs.
    """
    try:
        df = _get_dataframe()
        filtered_df = df[
            (df['members'].notna()) & 
            (df['members'] >= min_members)
        ].sort_values('members', ascending=False).head(limit)
        
        return filtered_df['anime_id'].tolist()
    except Exception as e:
        print(f"Error getting popular anime: {e}")
        return []

@tool
def search_anime_by_studio(studio_name: str) -> List[int]:
    """
    Find anime by animation studio.
    
    Args:
        studio_name (str): Name of the animation studio to search for
        
    Returns:
        List[int]: A list of anime IDs from the specified studio.
    """
    try:
        df = _get_dataframe()
        matches = df[df['studio'].fillna('').str.contains(studio_name, case=False, na=False)]
        return matches['anime_id'].tolist()
    except Exception as e:
        print(f"Error searching by studio: {e}")
        return []

@tool
def get_anime_by_source(source_type: str) -> List[int]:
    """
    Find anime by source material.
    
    Args:
        source_type (str): Source material type (e.g., "Manga", "Light novel", "Original")
        
    Returns:
        List[int]: A list of anime IDs from the specified source.
    """
    try:
        df = _get_dataframe()
        matches = df[df['source'].fillna('').str.contains(source_type, case=False, na=False)]
        return matches['anime_id'].tolist()
    except Exception as e:
        print(f"Error searching by source: {e}")
        return []

@tool
def find_similar_anime(anime_id: int, top_n: int = 10) -> List[int]:
    """
    Find similar anime based on genre, studio, rating, and other features.
    Uses content-based similarity.
    
    Args:
        anime_id (int): ID of the anime to find similar ones for
        top_n (int): Number of similar anime to return (default: 10)
        
    Returns:
        List[int]: A list of similar anime IDs.
    """
    try:
        df = _get_dataframe()
        similarity_matrix = _get_similarity_matrix()
        
        # Find the index of the anime
        anime_index = df[df['anime_id'] == anime_id].index
        if len(anime_index) == 0:
            return []
        
        idx = anime_index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar anime (excluding the anime itself)
        sim_scores = sim_scores[1:top_n+1]
        anime_indices = [i[0] for i in sim_scores]
        
        return df.iloc[anime_indices]['anime_id'].tolist()
    except Exception as e:
        print(f"Error finding similar anime: {e}")
        return []

@tool
def search_anime_by_synopsis(keywords: str) -> List[int]:
    """
    Search anime by keywords in synopsis using text matching.
    
    Args:
        keywords (str): Keywords to search for in anime synopsis
        
    Returns:
        List[int]: A list of anime IDs with matching synopsis content.
    """
    try:
        df = _get_dataframe()
        keywords_lower = keywords.lower()
        
        matches = df[df['synopsis'].fillna('').str.lower().str.contains(keywords_lower, na=False)]
        return matches['anime_id'].tolist()
    except Exception as e:
        print(f"Error searching by synopsis: {e}")
        return []

@tool
def get_trending_anime(time_period: str = "recent") -> List[int]:
    """
    Get trending anime based on member count, rating, and recency.
    
    Args:
        time_period (str): Time period filter ("recent", "this_year", "all_time")
        
    Returns:
        List[int]: A list of trending anime IDs.
    """
    try:
        df = _get_dataframe()
        current_year = 2024
        
        if time_period == "recent":
            # Last 5 years
            df_filtered = df[df['aired_from_year'] >= current_year - 5]
        elif time_period == "this_year":
            # This year
            df_filtered = df[df['aired_from_year'] >= current_year]
        else:
            # All time
            df_filtered = df
        
        # Score based on rating and member count
        df_filtered = df_filtered[
            (df_filtered['rating'].notna()) & 
            (df_filtered['members'].notna())
        ].copy()
        
        # Normalize and combine scores
        df_filtered['rating_norm'] = df_filtered['rating'] / 10.0
        df_filtered['members_norm'] = df_filtered['members'] / df_filtered['members'].max()
        df_filtered['trend_score'] = df_filtered['rating_norm'] * 0.6 + df_filtered['members_norm'] * 0.4
        
        # Sort by trend score
        trending = df_filtered.sort_values('trend_score', ascending=False).head(50)
        return trending['anime_id'].tolist()
    except Exception as e:
        print(f"Error getting trending anime: {e}")
        return []

@tool
def analyze_user_preferences(liked_anime_ids: List[int]) -> Dict[str, Union[List[str], float, int]]:
    """
    Analyze user preferences based on their liked anime.
    
    Args:
        liked_anime_ids (List[int]): List of anime IDs the user liked
        
    Returns:
        Dict: Analysis of user preferences including genres, studios, ratings, etc.
    """
    try:
        df = _get_dataframe()
        liked_anime = df[df['anime_id'].isin(liked_anime_ids)]
        
        if len(liked_anime) == 0:
            return {}
        
        # Analyze genres
        all_genres = []
        for genres_str in liked_anime['genre'].dropna():
            genres = [g.strip() for g in str(genres_str).split(',')]
            all_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        top_genres = [genre for genre, count in genre_counts.most_common(5)]
        
        # Analyze studios
        studio_counts = Counter(liked_anime['studio'].dropna())
        top_studios = [studio for studio, count in studio_counts.most_common(3)]
        
        # Analyze other preferences
        avg_rating = liked_anime['rating'].mean()
        avg_episodes = liked_anime['episodes'].mean()
        
        # Source preferences
        source_counts = Counter(liked_anime['source'].dropna())
        top_sources = [source for source, count in source_counts.most_common(3)]
        
        # Year preferences
        year_data = liked_anime['aired_from_year'].dropna()
        avg_year = year_data.mean() if len(year_data) > 0 else None
        
        return {
            "preferred_genres": top_genres,
            "preferred_studios": top_studios,
            "preferred_sources": top_sources,
            "average_rating": float(avg_rating) if pd.notna(avg_rating) else None,
            "average_episodes": float(avg_episodes) if pd.notna(avg_episodes) else None,
            "average_year": float(avg_year) if avg_year else None,
            "total_analyzed": len(liked_anime)
        }
    except Exception as e:
        print(f"Error analyzing user preferences: {e}")
        return {}

@tool
def advanced_anime_search(
    genres: Optional[List[str]] = None,
    min_rating: Optional[float] = None,
    max_episodes: Optional[int] = None,
    studios: Optional[List[str]] = None,
    year_range: Optional[List[int]] = None,
    anime_type: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50
) -> List[int]:
    """
    Advanced search with multiple filters combined.
    
    Args:
        genres (Optional[List[str]]): Genres to filter by
        min_rating (Optional[float]): Minimum rating
        max_episodes (Optional[int]): Maximum episode count
        studios (Optional[List[str]]): Studios to filter by
        year_range (Optional[List[int]]): Year range as [start_year, end_year]
        anime_type (Optional[str]): Type of anime (TV, Movie, OVA, etc.)
        source (Optional[str]): Source material
        limit (int): Maximum results to return
        
    Returns:
        List[int]: A list of anime IDs matching all criteria.
    """
    try:
        df = _get_dataframe()
        filtered_df = df.copy()
        
        # Apply genre filter
        if genres:
            genre_mask = pd.Series([False] * len(filtered_df))
            for genre in genres:
                genre_mask = genre_mask | filtered_df['genre'].fillna('').str.contains(genre, case=False, na=False)
            filtered_df = filtered_df[genre_mask]
        
        # Apply rating filter
        if min_rating is not None:
            filtered_df = filtered_df[
                (filtered_df['rating'].notna()) & 
                (filtered_df['rating'] >= min_rating)
            ]
        
        # Apply episode filter
        if max_episodes is not None:
            filtered_df = filtered_df[
                (filtered_df['episodes'].notna()) & 
                (filtered_df['episodes'] <= max_episodes)
            ]
        
        # Apply studio filter
        if studios:
            studio_mask = pd.Series([False] * len(filtered_df))
            for studio in studios:
                studio_mask = studio_mask | filtered_df['studio'].fillna('').str.contains(studio, case=False, na=False)
            filtered_df = filtered_df[studio_mask]
        
        # Apply year range filter
        if year_range and len(year_range) == 2:
            start_year, end_year = year_range[0], year_range[1]
            filtered_df = filtered_df[
                (filtered_df['aired_from_year'].notna()) & 
                (filtered_df['aired_from_year'] >= start_year) & 
                (filtered_df['aired_from_year'] <= end_year)
            ]
        
        # Apply type filter
        if anime_type:
            filtered_df = filtered_df[
                filtered_df['type'].fillna('').str.contains(anime_type, case=False, na=False)
            ]
        
        # Apply source filter
        if source:
            filtered_df = filtered_df[
                filtered_df['source'].fillna('').str.contains(source, case=False, na=False)
            ]
        
        # Sort by rating and limit results
        filtered_df = filtered_df.sort_values('rating', ascending=False, na_position='last').head(limit)
        
        return filtered_df['anime_id'].tolist()
    except Exception as e:
        print(f"Error in advanced search: {e}")
        return []

@tool
def get_anime_quick_filters() -> Dict[str, List[str]]:
    """
    Get available filter options for quick filtering.
    
    Returns:
        Dict[str, List[str]]: Dictionary containing available filter options.
    """
    try:
        df = _get_dataframe()
        
        # Get unique genres
        all_genres = set()
        for genres_str in df['genre'].dropna():
            genres = [g.strip() for g in str(genres_str).split(',')]
            all_genres.update(genres)
        
        # Get unique studios (top 20)
        top_studios = df['studio'].value_counts().head(20).index.tolist()
        
        # Get unique sources
        sources = df['source'].dropna().unique().tolist()
        
        # Get unique types
        types = df['type'].dropna().unique().tolist()
        
        # Get year range
        years = df['aired_from_year'].dropna()
        year_range = [int(years.min()), int(years.max())] if len(years) > 0 else []
        
        return {
            "genres": sorted(list(all_genres)),
            "studios": top_studios,
            "sources": sources,
            "types": types,
            "year_range": year_range,
            "rating_range": [0.0, 10.0]
        }
    except Exception as e:
        print(f"Error getting quick filters: {e}")
        return {}

@tool
def get_anime_by_type(anime_type: str) -> List[int]:
    """
    Find anime by type (TV, Movie, OVA, etc.).
    
    Args:
        anime_type (str): Type of anime to search for
        
    Returns:
        List[int]: A list of anime IDs of the specified type.
    """
    try:
        df = _get_dataframe()
        matches = df[df['type'].fillna('').str.contains(anime_type, case=False, na=False)]
        return matches['anime_id'].tolist()
    except Exception as e:
        print(f"Error searching by type: {e}")
        return []

@tool
def get_anime_statistics(anime_ids: List[int]) -> Dict[str, Union[float, int, List[str]]]:
    """
    Get aggregated statistics for a list of anime.
    
    Args:
        anime_ids (List[int]): List of anime IDs to analyze
        
    Returns:
        Dict: Statistics including average rating, episode count, common genres, etc.
    """
    try:
        df = _get_dataframe()
        anime_data = df[df['anime_id'].isin(anime_ids)]
        
        if len(anime_data) == 0:
            return {}
        
        # Basic statistics
        avg_rating = anime_data['rating'].mean()
        avg_episodes = anime_data['episodes'].mean()
        avg_duration = anime_data['duration_mins'].mean()
        
        # Genre analysis
        all_genres = []
        for genres_str in anime_data['genre'].dropna():
            genres = [g.strip() for g in str(genres_str).split(',')]
            all_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        top_genres = [genre for genre, count in genre_counts.most_common(5)]
        
        # Studio analysis
        studio_counts = Counter(anime_data['studio'].dropna())
        top_studios = [studio for studio, count in studio_counts.most_common(3)]
        
        # Year distribution
        years = anime_data['aired_from_year'].dropna()
        year_stats = {
            "earliest": int(years.min()) if len(years) > 0 else None,
            "latest": int(years.max()) if len(years) > 0 else None,
            "average": float(years.mean()) if len(years) > 0 else None
        }
        
        return {
            "count": len(anime_data),
            "average_rating": float(avg_rating) if pd.notna(avg_rating) else None,
            "average_episodes": float(avg_episodes) if pd.notna(avg_episodes) else None,
            "average_duration_mins": float(avg_duration) if pd.notna(avg_duration) else None,
            "top_genres": top_genres,
            "top_studios": top_studios,
            "year_statistics": year_stats
        }
    except Exception as e:
        print(f"Error getting anime statistics: {e}")
        return {}

# List of available tools for the agent
tools = [
    get_anime_ids_after_year, 
    get_anime_details_by_ids, 
    search_anime_ids_by_name,
    search_anime_by_genre,
    get_anime_by_rating_range,
    get_popular_anime,
    search_anime_by_studio,
    get_anime_by_source,
    find_similar_anime,
    search_anime_by_synopsis,
    get_trending_anime,
    analyze_user_preferences,
    advanced_anime_search,
    get_anime_quick_filters,
    get_anime_by_type,
    get_anime_statistics
]
