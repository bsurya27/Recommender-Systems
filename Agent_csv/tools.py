import pandas as pd
from langchain_core.tools import tool
from typing import List

csv_path = "../Suprise_reccomend/anime_clean.csv"

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
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Filter anime that aired after the given year
        # Handle potential NaN values in aired_from_year
        filtered_df = df[(df['aired_from_year'].notna()) & (df['aired_from_year'] > year)]
        
        # Get the anime IDs and convert to list
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
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Filter the dataframe to only include the specified anime IDs
        filtered_df = df[df['anime_id'].isin(anime_ids)]
        
        # Sort by anime_id to maintain consistent ordering
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
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert search query to lowercase for case-insensitive search
        search_lower = search_query.lower()
        
        # Create boolean masks for searching in both name columns
        # Handle potential NaN values by filling with empty strings
        name_match = df['name'].fillna('').str.lower().str.contains(search_lower, na=False)
        title_english_match = df['title_english'].fillna('').str.lower().str.contains(search_lower, na=False)
        
        # Combine both searches with OR logic
        matches = df[name_match | title_english_match]
        
        # Get the anime IDs and convert to list
        anime_ids = matches['anime_id'].tolist()
        
        return anime_ids
    
    except Exception as e:
        print(f"Error reading CSV or searching data: {e}")
        return []

# List of available tools for the agent
tools = [get_anime_ids_after_year, get_anime_details_by_ids, search_anime_ids_by_name]
