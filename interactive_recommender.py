import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class InteractiveAnimeRecommender:
    def __init__(self):
        self.svd = None
        self.anime_info = None
        self.all_anime_ids = None
        self.combined_vectors = None
        self.anime_df = None
        self.anime_id_to_index = None

    def load_data(self):
        # ------------------------------------------------------------
        # Attempt to load the Surprise SVD model. If `scikit-surprise`
        # is not installed, unpickling will fail.  We handle this
        # gracefully so that similarity-based recommendations can still
        # function without the collaborative-filtering component.
        # ------------------------------------------------------------
        print("Loading SVD model (optional)...")
        try:
            with open('Data/svd_model_surprise.pkl', 'rb') as f:
                self.svd = pickle.load(f)
        except (ModuleNotFoundError, ImportError):
            print("⚠️  scikit-surprise not available – skipping SVD model.\n"
                  "    Only similarity-based recommendations will work.")
            self.svd = None
        except FileNotFoundError:
            print("⚠️  SVD model file not found – skipping collaborative filtering.")
            self.svd = None
        except Exception as e:
            print(f"⚠️  Failed to load SVD model ({e}) – proceeding without it.")
            self.svd = None

        print("Loading anime info...")
        anime_df = pd.read_csv('Data/somewhatcleanedAnime.csv')
        self.anime_info = anime_df.set_index('anime_id')
        self.all_anime_ids = set(self.anime_info.index.astype(str))
        # For similarity-based
        print("Loading similarity-based model data...")
        self.anime_df = pd.read_csv('Data/anime_metadata_cleaned.csv')
        self.combined_vectors = np.load('Data/anime_combined_vectors.npy')
        self.anime_id_to_index = {aid: idx for idx, aid in enumerate(self.anime_df["anime_id"])}

    def get_default_anime_suggestions(self, n_suggestions=10):
        popular_anime = self.anime_info.sort_values('members', ascending=False).head(n_suggestions)
        suggestions = []
        for anime_id, row in popular_anime.iterrows():
            suggestions.append({'anime_id': str(anime_id), 'name': row['name'], 'members': row['members']})
        return suggestions

    def ask_anime_preferences(self):
        print("\n=== Anime Preferences ===")
        print("Are you new to anime or have you watched some before?")
        print("1 = New to anime, 2 = I've watched some anime")
        while True:
            try:
                response = input("Your choice (1 or 2): ")
                choice = int(response)
                if choice in [1, 2]:
                    break
                else:
                    print("Please enter 1 or 2")
            except ValueError:
                print("Please enter a valid number")
        user_ratings = {}
        user_liked_genres = set()
        if choice == 1:
            print("\nNo problem! Let me suggest some popular anime to get started.")
            default_anime = self.get_default_anime_suggestions(8)
            print("Please rate these popular anime on a scale of 1-10:")
            print("(1 = Didn't like it, 5 = It was okay, 10 = Loved it)")
            print("If you haven't watched it, just press Enter to skip")
            for anime in default_anime:
                while True:
                    try:
                        response = input(f"{anime['name']}: ")
                        if response.strip() == "":
                            break
                        score = int(response)
                        if 1 <= score <= 10:
                            user_ratings[anime['anime_id']] = score
                            # Track liked genres
                            if score >= 7:
                                genres = self.anime_info.loc[int(anime['anime_id']), 'genre']
                                if pd.notna(genres):
                                    for g in str(genres).split(','):
                                        user_liked_genres.add(g.strip())
                            break
                        else:
                            print("Please enter a number between 1 and 10, or press Enter to skip")
                    except ValueError:
                        print("Please enter a valid number, or press Enter to skip")
        else:
            print("\nGreat! Please list some anime you've watched, separated by commas.")
            print("For example: Death Note, Attack on Titan, One Piece")
            while True:
                anime_input = input("Anime you've watched: ").strip()
                if anime_input:
                    break
                print("Please enter at least one anime name")
            anime_names = [name.strip() for name in anime_input.split(',')]
            matched_anime = []
            for name in anime_names:
                matches = self.anime_info[self.anime_info['name'].str.contains(name, case=False, na=False)]
                if len(matches) == 1:
                    matched_anime.append(matches.iloc[0])
                elif len(matches) > 1:
                    print(f"\nMultiple matches found for '{name}':")
                    for i, (anime_id, row) in enumerate(matches.head(5).iterrows(), 1):
                        print(f"{i}. {row['name']}")
                    try:
                        choice = int(input("Which one did you mean? (enter number): ")) - 1
                        if 0 <= choice < len(matches):
                            matched_anime.append(matches.iloc[choice])
                    except (ValueError, IndexError):
                        print("Invalid choice, skipping this anime")
                else:
                    print(f"Could not find '{name}' in our database, skipping")
            if matched_anime:
                print(f"\nPlease rate these anime on a scale of 1-10:")
                for anime in matched_anime:
                    while True:
                        try:
                            response = input(f"{anime['name']}: ")
                            score = int(response)
                            if 1 <= score <= 10:
                                user_ratings[str(anime.name)] = score
                                # Track liked genres
                                if score >= 7:
                                    genres = anime['genre']
                                    if pd.notna(genres):
                                        for g in str(genres).split(','):
                                            user_liked_genres.add(g.strip())
                                break
                            else:
                                print("Please enter a number between 1 and 10")
                        except ValueError:
                            print("Please enter a valid number")
            else:
                print("No anime found. Let me suggest some popular ones instead.")
                default_anime = self.get_default_anime_suggestions(5)
                for anime in default_anime:
                    while True:
                        try:
                            response = input(f"{anime['name']}: ")
                            if response.strip() == "":
                                break
                            score = int(response)
                            if 1 <= score <= 10:
                                user_ratings[anime['anime_id']] = score
                                # Track liked genres
                                if score >= 7:
                                    genres = self.anime_info.loc[int(anime['anime_id']), 'genre']
                                    if pd.notna(genres):
                                        for g in str(genres).split(','):
                                            user_liked_genres.add(g.strip())
                                break
                            else:
                                print("Please enter a number between 1 and 10, or press Enter to skip")
                        except ValueError:
                            print("Please enter a valid number, or press Enter to skip")
        return user_ratings, user_liked_genres

    def recommend(self, user_ratings, user_liked_genres, n_recommendations=10):
        # Predict ratings for all anime the user hasn't rated
        predictions = []
        for anime_id in self.all_anime_ids:
            if anime_id not in user_ratings:
                pred = self.svd.predict('new_user', anime_id)
                score = pred.est
                # Boost score if anime is in a liked genre
                genres = self.anime_info.loc[int(anime_id), 'genre'] if int(anime_id) in self.anime_info.index else None
                if genres and user_liked_genres:
                    for g in str(genres).split(','):
                        if g.strip() in user_liked_genres:
                            score *= 1.2
                            break
                predictions.append((anime_id, score, genres))
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        recs = []
        for anime_id, score, genres in predictions[:n_recommendations]:
            if int(anime_id) in self.anime_info.index:
                recs.append({'anime_id': anime_id, 'name': self.anime_info.loc[int(anime_id), 'name'], 'score': score, 'genre': genres})
        return recs

    # --- Similarity-based recommender ---
    def find_anime_id_by_name(self, name):
        name = name.lower().strip()
        
        # First try exact matches
        for idx, row in self.anime_df.iterrows():
            # Check name column (handle nulls)
            anime_name = str(row["name"]).lower() if pd.notna(row["name"]) else ""
            # Check title_english column (handle nulls)
            english_name = str(row.get("title_english", "")).lower() if pd.notna(row.get("title_english")) else ""
            
            if name == anime_name or name == english_name:
                return row["anime_id"]
        
        # Fallback: partial matches (more flexible)
        for idx, row in self.anime_df.iterrows():
            # Check name column (handle nulls)
            anime_name = str(row["name"]).lower() if pd.notna(row["name"]) else ""
            # Check title_english column (handle nulls)
            english_name = str(row.get("title_english", "")).lower() if pd.notna(row.get("title_english")) else ""
            
            if name in anime_name or name in english_name:
                return row["anime_id"]
        
        return None

    def recommend_similar(self, anime_id, top_n=10, filter_genres=None):
        idx = self.anime_id_to_index.get(anime_id)
        if idx is None:
            return []
        query_vec = self.combined_vectors[idx].reshape(1, -1)
        similarities = cosine_similarity(query_vec, self.combined_vectors)[0]
        if filter_genres:
            mask = self.anime_df["genre_list"].apply(lambda genres: any(g in genres for g in filter_genres))
            similarities = similarities * mask.to_numpy()
        top_idx = similarities.argsort()[::-1]
        top_idx = [i for i in top_idx if i != idx][:top_n]
        results = self.anime_df.iloc[top_idx][["anime_id", "name", "genre", "synopsis"]].copy()
        results["similarity"] = similarities[top_idx]
        return results

    def show_recommendations(self, recommendations):
        print(f"\n=== Your Anime Recommendations ===")
        print("Based on your preferences, here are some anime you might enjoy:")
        print()
        for i, rec in enumerate(recommendations, 1):
            genre_str = f" | Genres: {rec['genre']}" if rec['genre'] else ""
            print(f"{i}. {rec['name']} (Predicted Score: {rec['score']:.2f}){genre_str}")

    def show_similar_recommendations(self, recs, base_name):
        print(f"\n=== Anime Similar to '{base_name}' ===")
        for i, row in recs.iterrows():
            print(f"{i+1}. {row['name']} | Genres: {row['genre']} | Similarity: {row['similarity']:.3f}")

    def run_interactive_session(self):
        print("=== Welcome to the Interactive Anime Recommender (Surprise SVD + Similarity)! ===")
        self.load_data()
        print("Choose recommendation type:")
        print("1. Personalized recommendations (collaborative filtering)")
        print("2. Find anime similar to a given anime")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            user_ratings, user_liked_genres = self.ask_anime_preferences()
            recommendations = self.recommend(user_ratings, user_liked_genres)
            self.show_recommendations(recommendations)
        elif choice == "2":
            name = input("Enter anime name (English or original): ").strip()
            anime_id = self.find_anime_id_by_name(name)
            if anime_id is None:
                print("Anime not found.")
                return
            recs = self.recommend_similar(anime_id)
            self.show_similar_recommendations(recs, name)
        else:
            print("Invalid choice.")
        print("\n=== Session Complete ===")

def main():
    recommender = InteractiveAnimeRecommender()
    recommender.run_interactive_session()

if __name__ == "__main__":
    main() 