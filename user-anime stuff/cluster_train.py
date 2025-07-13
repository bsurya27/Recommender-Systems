import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load the cleaned user-anime data
# Assumes 'cleaned_userdata.csv' is in the same directory as this script
# The file should have columns: user_id, anime_id, score, status, episodes_seen
print("Loading cleaned user-anime data...")
df = pd.read_csv('final_user_anime.csv')

# 2. Pivot to a user-anime matrix
# Rows: user_id, Columns: anime_id, Values: score (0 if not rated)
print("Creating user-anime rating matrix...")
user_anime_matrix = df.pivot_table(index='user_id', columns='anime_id', values='score', fill_value=0)

# 3. Cluster users using Gaussian Mixture Model (soft clustering)
# Choose a reasonable number of components (can tune this)
n_components = 10
print(f"Clustering users into {n_components} components with Gaussian Mixture Model...")
gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type='diag')
user_clusters = gmm.fit_predict(user_anime_matrix)

# Get membership probabilities for each user
user_memberships = gmm.predict_proba(user_anime_matrix)

# Add cluster assignments to the matrix for later use
user_anime_matrix['cluster'] = user_clusters

# 4. Compute cluster preferences for each anime
# For each cluster, calculate the mean score for each anime
print("Calculating cluster preferences for each anime...")
cluster_anime_pref = user_anime_matrix.groupby('cluster').mean().drop(columns=['cluster']).T  # shape: (anime, cluster)

# 5. Visualize anime positions in 2D using PCA
# Each anime is a point, colored by the cluster that likes it most
print("Reducing cluster-anime preference matrix to 2D with PCA for visualization...")
pca = PCA(n_components=2)
anime_coords = pca.fit_transform(cluster_anime_pref.values)

# Assign each anime to the cluster with the highest mean score
anime_best_cluster = cluster_anime_pref.idxmax(axis=1)

print("Plotting anime positions by cluster preference...")
plt.figure(figsize=(12, 8))
for i in range(n_components):
    idx = anime_best_cluster == i
    plt.scatter(anime_coords[idx, 0], anime_coords[idx, 1], label=f'Cluster {i}', alpha=0.6)
plt.legend()
plt.title("Anime positions by cluster preference (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

# 6. Save the cluster assignments for each user to a CSV
print("Saving user cluster assignments to 'user_clusters.csv'...")
user_cluster_df = pd.DataFrame({
    'user_id': user_anime_matrix.index,
    'cluster': user_anime_matrix['cluster']
})
user_cluster_df.to_csv('user_clusters.csv', index=False)

# 7. Save the trained GMM model for later use
print("Saving trained GMM model to 'gmm_model.pkl'...")
import pickle
with open('gmm_model.pkl', 'wb') as f:
    pickle.dump(gmm, f)

# 8. Save cluster centers, anime preferences, and user memberships for the interactive recommender
print("Saving cluster centers, anime preferences, and user memberships...")
cluster_centers = user_anime_matrix.groupby('cluster').mean().drop(columns=['cluster'])
cluster_centers.to_csv('cluster_centers.csv')

cluster_anime_prefs = user_anime_matrix.groupby('cluster').mean().drop(columns=['cluster']).T
cluster_anime_prefs.to_csv('cluster_anime_prefs.csv')

# Save user membership probabilities
user_membership_df = pd.DataFrame(
    user_memberships, 
    index=user_anime_matrix.index,
    columns=[f'cluster_{i}' for i in range(n_components)]
)
user_membership_df.to_csv('user_memberships.csv')

print("Done.") 