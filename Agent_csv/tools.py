import pandas as pd
from langchain_core.tools import tool
from typing import List

csv_path = "Suprise_reccomend/anime_clean.csv"

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

# List of available tools for the agent
tools = [get_anime_ids_after_year]
