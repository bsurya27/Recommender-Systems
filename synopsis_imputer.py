import pandas as pd
import os
from pathlib import Path
import time

print("Starting synopsis imputer script...")

# Load the anime data
try:
    animeData = pd.read_csv('dataset/Anime.csv')
    print(f"Successfully loaded anime data with {len(animeData)} records")
except Exception as e:
    print(f"Error loading anime data: {e}")
    exit(1)

# Find animes with missing synopsis
missing_synopsis = animeData[animeData['synopsis'].isna()]
print(f"Total animes with missing synopsis: {len(missing_synopsis)}")

# Check which of these have wiki pages
wiki_pages_dir = Path('dataset/archive/wiki_pages/wiki_pages')
print(f"Looking for wiki pages in: {wiki_pages_dir}")
print(f"Directory exists: {wiki_pages_dir.exists()}")

available_wiki_pages = set()

if wiki_pages_dir.exists():
    try:
        for file_path in wiki_pages_dir.glob('*.txt'):
            anime_id = file_path.stem  # Get filename without extension
            available_wiki_pages.add(int(anime_id))
        
        print(f"Available wiki pages: {len(available_wiki_pages)}")
        
        # Find animes with missing synopsis that have wiki pages
        missing_with_wiki = missing_synopsis[missing_synopsis['anime_id'].isin(available_wiki_pages)]
        print(f"Animes with missing synopsis that have wiki pages: {len(missing_with_wiki)}")
        
        # Show first few examples
        print("\nFirst 5 animes with missing synopsis that have wiki pages:")
        for _, anime in missing_with_wiki.head().iterrows():
            print(f"ID: {anime['anime_id']}, Name: {anime['name']}")
            
        # Save the list for processing
        missing_with_wiki.to_csv('missing_synopsis_with_wiki.csv', index=False)
        print(f"\nSaved list to 'missing_synopsis_with_wiki.csv'")
        
    except Exception as e:
        print(f"Error processing wiki pages: {e}")
else:
    print("Wiki pages directory not found!") 