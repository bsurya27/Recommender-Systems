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
    df = _df()
    return df[(df['aired_from_year'].notna()) & (df['aired_from_year'] < year)]['anime_id'].tolist()

@tool
def get_anime_ids_after_year(year: int) -> List[int]:
    df = _df()
    return df[(df['aired_from_year'].notna()) & (df['aired_from_year'] > year)]['anime_id'].tolist()

@tool
def get_anime_ids_by_genre(genre: str) -> List[int]:
    df = _df()
    mask = df['genre'].fillna('').str.contains(genre, case=False, na=False)
    return df[mask]['anime_id'].tolist()

@tool
def get_anime_details(anime_ids: List[int]) -> pd.DataFrame:
    df = _df()
    return df[df['anime_id'].isin(anime_ids)].reset_index(drop=True) 