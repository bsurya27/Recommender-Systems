import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle

# 1. Load the user-anime ratings data
print("Loading user-anime ratings data...")
df = pd.read_csv('final_user_anime.csv')

# 2. Prepare the data for Surprise
# Surprise expects columns: user, item, rating
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df[['user_id', 'anime_id', 'score']], reader)

# 3. Train SVD model
print("Training SVD model with Surprise...")
trainset = data.build_full_trainset()
svd = SVD(n_factors=100, random_state=42)
svd.fit(trainset)

# 4. Save the trained model
print("Saving trained SVD model to 'svd_model_surprise.pkl'...")
with open('svd_model_surprise.pkl', 'wb') as f:
    pickle.dump(svd, f)

print("Done.") 