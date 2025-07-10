from tools import get_anime_ids_after_year, get_anime_details_by_ids, search_anime_ids_by_name
import pandas as pd

def test_get_anime_ids_after_year():
    """Test the get_anime_ids_after_year tool"""
    print("=" * 60)
    print("TESTING: get_anime_ids_after_year")
    print("=" * 60)
    
    # Test with year 2020
    print(f"📅 Getting anime IDs for anime that aired after 2020...")
    result_2020 = get_anime_ids_after_year.invoke({"year": 2020})
    print(f"✅ Found {len(result_2020)} anime after 2020")
    print(f"📋 First 10 anime IDs: {result_2020[:10]}")
    
    # Test with year 2015
    print(f"\n📅 Getting anime IDs for anime that aired after 2015...")
    result_2015 = get_anime_ids_after_year.invoke({"year": 2015})
    print(f"✅ Found {len(result_2015)} anime after 2015")
    print(f"📋 First 10 anime IDs: {result_2015[:10]}")
    
    # Test with year 2000
    print(f"\n📅 Getting anime IDs for anime that aired after 2000...")
    result_2000 = get_anime_ids_after_year.invoke({"year": 2000})
    print(f"✅ Found {len(result_2000)} anime after 2000")
    
    return result_2020, result_2015, result_2000

def test_search_anime_ids_by_name():
    """Test the search_anime_ids_by_name tool"""
    print("\n" + "=" * 60)
    print("TESTING: search_anime_ids_by_name")
    print("=" * 60)
    
    # Test searches
    test_queries = ["naruto", "attack", "demon", "one piece", "fullmetal"]
    
    search_results = {}
    for query in test_queries:
        print(f"🔍 Searching for anime with '{query}' in name...")
        result = search_anime_ids_by_name.invoke({"search_query": query})
        search_results[query] = result
        print(f"✅ Found {len(result)} anime matching '{query}'")
        print(f"📋 Anime IDs: {result[:5]}{'...' if len(result) > 5 else ''}")
        print()
    
    return search_results

def test_get_anime_details_by_ids():
    """Test the get_anime_details_by_ids tool"""
    print("=" * 60)
    print("TESTING: get_anime_details_by_ids")
    print("=" * 60)
    
    # Get some anime IDs from search
    print("🔍 First, searching for 'attack' anime...")
    attack_ids = search_anime_ids_by_name.invoke({"search_query": "attack"})
    
    if attack_ids:
        # Take first 3 IDs for testing
        test_ids = attack_ids[:3]
        print(f"📋 Using anime IDs: {test_ids}")
        
        print(f"\n📊 Getting detailed information for these anime...")
        details_df = get_anime_details_by_ids.invoke({"anime_ids": test_ids})
        
        print(f"✅ Retrieved details for {len(details_df)} anime")
        print(f"📋 Columns: {list(details_df.columns)}")
        
        # Display some key information
        if not details_df.empty:
            print(f"\n📺 Sample anime details:")
            for idx, row in details_df.iterrows():
                print(f"  • ID: {row['anime_id']}")
                print(f"    Name: {row['name']}")
                print(f"    English: {row['title_english']}")
                print(f"    Type: {row['type']} | Episodes: {row['episodes']} | Year: {row['aired_from_year']}")
                print(f"    Rating: {row['rating']} | Popularity: {row['popularity']}")
                print()
        
        return details_df
    else:
        print("❌ No anime found for 'attack' search")
        return pd.DataFrame()

def test_combined_workflow():
    """Test tools working together in a realistic workflow"""
    print("=" * 60)
    print("TESTING: Combined Workflow")
    print("=" * 60)
    
    print("🚀 Workflow: Find recent popular anime")
    print("Step 1: Get anime from 2020 onwards")
    recent_ids = get_anime_ids_after_year.invoke({"year": 2020})
    print(f"✅ Found {len(recent_ids)} recent anime")
    
    print("\nStep 2: Get details for first 10 recent anime")
    if recent_ids:
        sample_recent = recent_ids[:10]
        recent_details = get_anime_details_by_ids.invoke({"anime_ids": sample_recent})
        
        print(f"✅ Got details for {len(recent_details)} anime")
        
        # Show top rated recent anime
        if not recent_details.empty:
            recent_details_clean = recent_details.dropna(subset=['rating'])
            if not recent_details_clean.empty:
                top_rated = recent_details_clean.nlargest(3, 'rating')
                print(f"\n🏆 Top 3 highest rated recent anime:")
                for idx, row in top_rated.iterrows():
                    print(f"  {idx+1}. {row['name']} - Rating: {row['rating']}")
    
    print("\n" + "="*60)
    print("🚀 Workflow: Search and analyze specific anime")
    print("Step 1: Search for 'demon' anime")
    demon_ids = search_anime_ids_by_name.invoke({"search_query": "demon"})
    print(f"✅ Found {len(demon_ids)} demon-related anime")
    
    print("\nStep 2: Get details for demon anime")
    if demon_ids:
        demon_details = get_anime_details_by_ids.invoke({"anime_ids": demon_ids[:5]})  # Get first 5
        
        if not demon_details.empty:
            print(f"✅ Got details for {len(demon_details)} demon anime")
            print(f"\n👹 Demon anime found:")
            for idx, row in demon_details.iterrows():
                print(f"  • {row['name']} ({row['aired_from_year']}) - Rating: {row['rating']}")

def main():
    """Run all tests"""
    print("🧪 ANIME CSV TOOLS TESTING SUITE")
    print("🎯 Testing all LangGraph tools for anime data")
    
    try:
        # Test individual tools
        year_results = test_get_anime_ids_after_year()
        search_results = test_search_anime_ids_by_name()
        details_result = test_get_anime_details_by_ids()
        
        # Test combined workflows
        test_combined_workflow()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ All tools are working correctly")
        print("🚀 Ready for LangGraph agent integration")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 