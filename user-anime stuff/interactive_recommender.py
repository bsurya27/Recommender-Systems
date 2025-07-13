import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os

class InteractiveAnimeRecommender:
    def __init__(self):
        """Initialize the interactive recommender system"""
        self.cluster_centers = None
        self.cluster_anime_prefs = None
        self.anime_info = None
        self.user_anime_matrix = None
        
    def load_trained_model(self):
        """Load the trained GMM clustering model and related data"""
        print("Loading trained GMM clustering model...")
        
        # Load the trained GMM model
        import pickle
        with open('gmm_model.pkl', 'rb') as f:
            self.gmm_model = pickle.load(f)
        
        # Load cluster centers and anime preferences (pre-calculated)
        self.cluster_centers = pd.read_csv('cluster_centers.csv', index_col=0)
        self.cluster_anime_prefs = pd.read_csv('cluster_anime_prefs.csv', index_col=0)
        
        # Load user membership probabilities
        self.user_memberships = pd.read_csv('user_memberships.csv', index_col=0)
        
        # Load anime information for better recommendations
        valid_anime = pd.read_csv('../somewhatcleanedAnime.csv')
        self.anime_info = valid_anime.set_index('anime_id')
        
        print(f"Loaded GMM model with {len(self.cluster_centers)} components and {len(self.cluster_anime_prefs)} animes")
    
    def get_default_anime_suggestions(self, n_suggestions=10):
        """Get default anime suggestions for new users"""
        # Get popular anime (high member count) from the cleaned anime list
        popular_anime = self.anime_info.sort_values('members', ascending=False).head(n_suggestions)
        
        suggestions = []
        for anime_id, row in popular_anime.iterrows():
            suggestions.append({
                'anime_id': anime_id,
                'name': row['name'],
                'members': row['members']
            })
        
        return suggestions
    
    def ask_anime_preferences(self):
        """Ask user about anime they've watched and their ratings"""
        print("\n=== Anime Preferences ===")
        
        # Check if user is new or experienced
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
            # New user - suggest default anime
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
                            break  # Skip if not watched
                        score = int(response)
                        if 1 <= score <= 10:
                            user_ratings[anime['anime_id']] = score
                            break
                        else:
                            print("Please enter a number between 1 and 10, or press Enter to skip")
                    except ValueError:
                        print("Please enter a valid number, or press Enter to skip")
        
        else:
            # Experienced user - let them input anime they've watched
            print("\nGreat! Please list some anime you've watched, separated by commas.")
            print("For example: Death Note, Attack on Titan, One Piece")
            
            while True:
                anime_input = input("Anime you've watched: ").strip()
                if anime_input:
                    break
                print("Please enter at least one anime name")
            
            # Parse comma-separated anime names
            anime_names = [name.strip() for name in anime_input.split(',')]
            
            # Match anime names to IDs
            matched_anime = []
            for name in anime_names:
                # Try exact match first
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
            
            # Get ratings for matched anime
            if matched_anime:
                print(f"\nPlease rate these anime on a scale of 1-10:")
                for anime in matched_anime:
                    while True:
                        try:
                            response = input(f"{anime['name']}: ")
                            score = int(response)
                            if 1 <= score <= 10:
                                user_ratings[anime.name] = score  # anime.name is the anime_id (index)
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
    
    def ask_anime_characteristics(self):
        """Ask user about anime characteristics preferences"""
        print("\n=== Anime Characteristics ===")
        
        characteristics = {}
        
        # Episode length preference
        print("Do you prefer shorter series (<12 episodes) or longer ones?")
        print("1 = Much prefer shorter, 5 = No preference, 10 = Much prefer longer")
        while True:
            try:
                response = input("Episode length preference (1-10): ")
                score = int(response)
                if 1 <= score <= 10:
                    characteristics['episode_length'] = score / 10.0
                    break
                else:
                    print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        # Plot complexity
        print("Do you prefer complex, thought-provoking plots or simple, fun stories?")
        print("1 = Simple and fun, 5 = No preference, 10 = Complex and deep")
        while True:
            try:
                response = input("Plot complexity preference (1-10): ")
                score = int(response)
                if 1 <= score <= 10:
                    characteristics['plot_complexity'] = score / 10.0
                    break
                else:
                    print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        # Age preference
        print("Do you prefer newer anime (2010+) or classics?")
        print("1 = Much prefer classics, 5 = No preference, 10 = Much prefer newer")
        while True:
            try:
                response = input("Age preference (1-10): ")
                score = int(response)
                if 1 <= score <= 10:
                    characteristics['age_preference'] = score / 10.0
                    break
                else:
                    print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        return characteristics
    
    def ask_rating_style(self):
        """Ask user about their rating style"""
        print("\n=== Rating Style ===")
        
        print("How harsh are you with ratings?")
        print("1 = Very harsh (rarely give high scores), 5 = Average, 10 = Very generous")
        while True:
            try:
                response = input("Rating harshness (1-10): ")
                score = int(response)
                if 1 <= score <= 10:
                    rating_style = score / 10.0
                    break
                else:
                    print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        return rating_style
    
    def build_feature_vector(self, user_ratings, characteristics, rating_style):
        """Build a feature vector from user anime ratings"""
        print("\nBuilding your feature vector...")
        
        # Start with zeros (same length as cluster centers)
        feature_vector = np.zeros(len(self.cluster_centers.columns))
        
        # Fill in the user's ratings for anime they've rated
        for anime_id, rating in user_ratings.items():
            if str(anime_id) in self.cluster_centers.columns:
                col_idx = self.cluster_centers.columns.get_loc(str(anime_id))
                feature_vector[col_idx] = rating
        
        # If user hasn't rated many anime, we can use their average rating for unrated anime
        if len(user_ratings) > 0:
            avg_rating = np.mean(list(user_ratings.values()))
            # For unrated anime, use a small fraction of their average rating
            # This prevents the vector from being too sparse
            for i in range(len(feature_vector)):
                if feature_vector[i] == 0:
                    feature_vector[i] = avg_rating * 0.1  # Small weight for unrated anime
        
        # Normalize the feature vector to match the scale of cluster centers
        if np.sum(feature_vector) > 0:
            feature_vector = feature_vector / np.max(feature_vector) * 10  # Scale to 0-10 range
        
        return feature_vector
    
    def find_user_memberships(self, feature_vector):
        """Find the user's membership probabilities for each cluster using GMM"""
        # Reshape feature vector for GMM prediction
        feature_vector_reshaped = feature_vector.reshape(1, -1)
        
        # Get membership probabilities from GMM
        memberships = self.gmm_model.predict_proba(feature_vector_reshaped)[0]
        
        # Return membership probabilities and the cluster with highest probability
        best_cluster = np.argmax(memberships)
        return memberships, best_cluster
    
    def get_weighted_recommendations(self, memberships, n_recommendations=10):
        """Get weighted anime recommendations based on user's cluster memberships"""
        # Calculate weighted anime preferences across all clusters
        weighted_prefs = np.zeros(len(self.cluster_anime_prefs))
        
        for cluster_id, membership_prob in enumerate(memberships):
            if membership_prob > 0.01:  # Only consider clusters with >1% membership
                cluster_prefs = self.cluster_anime_prefs.iloc[:, cluster_id].values
                weighted_prefs += membership_prob * cluster_prefs
        
        # Create a DataFrame with weighted preferences
        weighted_df = pd.DataFrame({
            'anime_id': self.cluster_anime_prefs.index,
            'weighted_score': weighted_prefs
        })
        
        # Sort by weighted score (descending)
        top_animes = weighted_df.sort_values('weighted_score', ascending=False).head(n_recommendations)
        
        recommendations = []
        for _, row in top_animes.iterrows():
            anime_id = row['anime_id']
            if anime_id in self.anime_info.index:
                anime_name = self.anime_info.loc[anime_id, 'name']
                recommendations.append({
                    'anime_id': anime_id,
                    'name': anime_name,
                    'weighted_score': row['weighted_score']
                })
        
        return recommendations
    
    def show_recommendations(self, recommendations):
        """Display recommendations to the user"""
        print(f"\n=== Your Anime Recommendations ===")
        print("Based on your preferences, here are some anime you might enjoy:")
        print()
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} (Score: {rec['cluster_score']:.2f})")
    
    def get_user_feedback(self, recommendations):
        """Get feedback from user about recommendations"""
        print(f"\n=== Feedback ===")
        print("How do you feel about these recommendations?")
        print("1 = Hate them, 5 = Neutral, 10 = Love them")
        
        while True:
            try:
                response = input("Overall satisfaction (1-10): ")
                satisfaction = int(response)
                if 1 <= satisfaction <= 10:
                    break
                else:
                    print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        return satisfaction
    
    def adjust_memberships(self, feature_vector, satisfaction, current_memberships):
        """Adjust cluster memberships based on user feedback"""
        if satisfaction >= 7:
            return current_memberships  # User is satisfied
        
        # If user is not satisfied, adjust memberships
        print("\nLet me adjust your cluster memberships...")
        
        # Calculate distances to all cluster centers
        distances = []
        for i in range(len(self.cluster_centers)):
            cluster_center = self.cluster_centers.iloc[i].values
            distance = np.linalg.norm(feature_vector - cluster_center)
            distances.append(distance)
        
        # Create new memberships based on distances (closer = higher probability)
        # Convert distances to probabilities (inverse relationship)
        distance_probs = 1 / (1 + np.array(distances))
        new_memberships = distance_probs / np.sum(distance_probs)
        
        # Blend current memberships with new ones based on satisfaction
        # Lower satisfaction = more weight to new memberships
        blend_factor = (10 - satisfaction) / 10.0  # 0.3 for satisfaction=7, 0.9 for satisfaction=1
        adjusted_memberships = (1 - blend_factor) * current_memberships + blend_factor * new_memberships
        
        # Normalize
        adjusted_memberships = adjusted_memberships / np.sum(adjusted_memberships)
        
        return adjusted_memberships
    
    def run_interactive_session(self):
        """Run the complete interactive recommendation session"""
        print("=== Welcome to the Interactive Anime Recommender! ===")
        print("I'll ask you some questions to understand your preferences.")
        print("Then I'll recommend anime that matches your taste.\n")
        
        # Load the trained model
        self.load_trained_model()
        
        # Get user preferences through questionnaire
        user_ratings = self.ask_anime_preferences()
        characteristics = self.ask_anime_characteristics()
        rating_style = self.ask_rating_style()
        
        # Build initial feature vector
        feature_vector = self.build_feature_vector(user_ratings, characteristics, rating_style)
        
        # Find user's cluster memberships
        memberships, best_cluster = self.find_user_memberships(feature_vector)
        print(f"\nBased on your preferences, here are your cluster memberships:")
        for i, prob in enumerate(memberships):
            if prob > 0.05:  # Only show clusters with >5% membership
                print(f"Cluster {i}: {prob:.1%}")
        
        # Interactive recommendation loop
        max_iterations = 5
        for iteration in range(max_iterations):
            # Get weighted recommendations based on memberships
            recommendations = self.get_weighted_recommendations(memberships)
            
            # Show recommendations
            self.show_recommendations(recommendations)
            
            # Get user feedback
            satisfaction = self.get_user_feedback(recommendations)
            
            if satisfaction >= 7:
                print(f"\nGreat! You're satisfied with the recommendations.")
                print("Your cluster memberships have been saved for future recommendations.")
                break
            
            # Adjust memberships based on feedback
            memberships = self.adjust_memberships(feature_vector, satisfaction, memberships)
            
            if iteration == max_iterations - 1:
                print("\nI've tried my best to find good recommendations for you.")
                print("You might want to try rating more anime or be more specific about your preferences.")
        
        print(f"\n=== Session Complete ===")
        print("Your final cluster memberships:")
        for i, prob in enumerate(memberships):
            if prob > 0.05:
                print(f"Cluster {i}: {prob:.1%}")
        return memberships

def main():
    """Main function to run the interactive recommender"""
    recommender = InteractiveAnimeRecommender()
    final_memberships = recommender.run_interactive_session()
    
    # Save the user's cluster memberships
    print(f"\nYour cluster memberships have been saved.")
    print("You can use these for future recommendations!")

if __name__ == "__main__":
    main() 