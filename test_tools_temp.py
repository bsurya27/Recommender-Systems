"""Quick sanity-check script for the anime tools.
Run: python test_tools_temp.py
"""

from tools import (
    get_anime_ids_before_year,
    get_anime_ids_after_year,
    get_anime_ids_by_genre,
    search_anime_ids_by_synopsis,
    get_anime_details,
)


def preview(label, ids):
    print(f"{label}: {len(ids)} IDs")
    print("Sample:", ids[:5])
    if ids:
        details = get_anime_details.invoke({"anime_ids": ids[:5]})
        print(details[["anime_id", "name", "genre", "rating"].copy()])
    print("-" * 40)


action_ids = get_anime_ids_by_genre.invoke({"genre": "Action"})
preview("Action genre", action_ids)

drama_ids = get_anime_ids_by_genre.invoke({"genre": "Drama"})
preview("Drama genre", drama_ids)

after_2015 = get_anime_ids_after_year.invoke({"year": 2015})
preview("After 2015", after_2015)

before_2000 = get_anime_ids_before_year.invoke({"year": 2000})
preview("Before 2000", before_2000)

magic_syn = search_anime_ids_by_synopsis.invoke({"keyword": "magic"})
preview("Synopsis contains 'magic'", magic_syn) 