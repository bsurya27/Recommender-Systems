import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class InteractiveAnimeRecommender:
    def __init__(self):
        self.user_anime_matrix = None
        self.anime_info = None
        self.anime_ids = None

    def load_data(self):
        print("Loading user-anime matrix and anime info...")
        self.user_anime_matrix = pd.read_csv('user_anime_matrix.csv', index_col=0)
        valid_anime = pd.read_csv('../somewhatcleanedAnime.csv')
        self.anime_info = valid_anime.set_index('anime_id')
        self.anime_ids = self.user_anime_matrix.columns.astype(str)

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
                                break
                            else:
                                print("Please enter a number between 1 and 10, or press Enter to skip")
                        except ValueError:
                            print("Please enter a valid number, or press Enter to skip")
        return user_ratings

    def build_user_vector(self, user_ratings):
        # Build a vector matching the columns of the user-anime matrix
        vec = np.zeros(len(self.anime_ids))
        for idx, anime_id in enumerate(self.anime_ids):
            if anime_id in user_ratings:
                vec[idx] = user_ratings[anime_id]
        return vec.reshape(1, -1)

    def get_knn_recommendations(self, user_vector, k=10, n_recommendations=10):
        # Compute cosine similarity to all users
        matrix = self.user_anime_matrix.values
        similarities = cosine_similarity(user_vector, matrix)[0]
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        similar_users = self.user_anime_matrix.iloc[top_k_idx]
        # Weighted sum of ratings from similar users
        weighted_scores = (similar_users.T * similarities[top_k_idx]).T.sum(axis=0)
        # Recommend anime not rated by the user, sorted by score
        user_rated = set(np.where(user_vector.flatten() > 0)[0])
        recs = []
        for idx in np.argsort(weighted_scores)[::-1]:
            if idx not in user_rated and weighted_scores[idx] > 0:
                anime_id = self.anime_ids[idx]
                if anime_id in self.anime_info.index.astype(str):
                    recs.append({
                        'anime_id': anime_id,
                        'name': self.anime_info.loc[int(anime_id), 'name'],
                        'score': weighted_scores[idx]
                    })
            if len(recs) >= n_recommendations:
                break
        return recs

    def show_recommendations(self, recommendations):
        print(f"\n=== Your Anime Recommendations ===")
        print("Based on your preferences, here are some anime you might enjoy:")
        print()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} (Score: {rec['score']:.2f})")

    def run_interactive_session(self):
        print("=== Welcome to the Interactive Anime Recommender (KNN)! ===")
        self.load_data()
        user_ratings = self.ask_anime_preferences()
        user_vector = self.build_user_vector(user_ratings)
        recommendations = self.get_knn_recommendations(user_vector)
        self.show_recommendations(recommendations)
        print("\n=== Session Complete ===")

def main():
    recommender = InteractiveAnimeRecommender()
    recommender.run_interactive_session()

if __name__ == "__main__":
    main() 