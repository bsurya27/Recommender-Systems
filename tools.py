import pandas as pd
from typing import List
from langchain_core.tools import tool

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
    df = _df()
    return df[df['anime_id'].isin(anime_ids)].reset_index(drop=True) 