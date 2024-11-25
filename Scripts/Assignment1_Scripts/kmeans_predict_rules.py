import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import umap
from scipy.spatial.distance import cdist
from matplotlib.colors import ListedColormap

# This script took me the longest to put together, especially with all the moving parts in the plots and visualizations, which needed LOTS of tweaking to get right.
# There were also MANY errors that had to be worked through to ensure taht the data used in the plots was correct and consistent with the identified clusters.

# Set random state and number of initializations
# Note that I set a relatively high number of initializations to ensure that the KMeans algorithm converges to a stable solution that i can have faith represented the data well
# I knew i was reaching a relatively stable solution when any random seed value resulted in nearly identical clusters at a given number of initializations, which in this case was 1000
init = 1000
rand = 7

# Paths
embeddings_folder = 'Data/embeddings/bills'
committees_embedding_folder = 'Data/embeddings/committees'
bills_csv_path = 'Data/Main_Data/Bills.csv'
output_csv_path = f'Data/Main_Data/K-Means-Results/KMeans_Bills_Seed{rand}_N{init}.csv'

# Load bill embeddings and bill names
# In the process of improving this script, i realized that I should sort the embedding files to ensure consistency with other versions of the script i tried to run
embedding_files = [f for f in os.listdir(embeddings_folder) if f.endswith('.npy')]
embedding_files.sort()

embeddings = []
bill_names = []

# pull each embedding file and its corresponding bill name into the embeddings and bill_names lists
for file in embedding_files:
    embedding_path = os.path.join(embeddings_folder, file)
    embeddings.append(np.load(embedding_path))
    bill_name = file.replace('.npy', '')
    bill_names.append(bill_name)

embeddings = np.array(embeddings)

# Load the bills CSV for committee info without setting index
bills_df = pd.read_csv(bills_csv_path)

# We are using 21 clusters for the analysis since there are 21 committees in the House of Representatives, to which a bill can be referred
# the theory of the case was that by setting the number of clusters to the number of committees, we could potentially see if the KMeans algorithm would cluster the bills by the committee to which they were referred in reality
# Note that OpenAI's text-embedding-3-large model was used to embed the text, which already pre-normalizes embeddings to have a unit norm.
# that means that we don't need to normalize the embeddings before running the KMeans algorithm, as magnitudes of the embeddings are already consistent
n_clusters = 21
kmeans = KMeans(n_clusters=n_clusters, n_init=init, random_state=rand)
kmeans.fit(embeddings)

# Get the cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Calculate the inertia value (sum of squared distances)
# this helps me understand how well the clusters are separated and how well the data points are assigned to their respective clusters
inertia_value = kmeans.inertia_

# I order to get a more fine-grained understanding of the clusters, I calculated the distance from each bill to its assigned cluster centroid
# as well as the silhouette scores for each bill in respect to its cluster
# this helps me understand how well each bill fits within its cluster and how well-separated the clusters are
distances_to_centroid = np.linalg.norm(embeddings - centroids[cluster_labels], axis=1)
silhouette_scores = silhouette_samples(embeddings, cluster_labels)

# Create a dataframe with the results
results_df = pd.DataFrame({
    'Bill Name': bill_names,
    'Cluster': cluster_labels,
    'Distance to Centroid': distances_to_centroid,
    'Silhouette Score': silhouette_scores
})

# Merge the results with the bills_df on 'Bill Name'
# These lines of code came after a lot of trial and error to ensure that the data was merged correctly and that the columns were aligned properly
merged_df = pd.merge(bills_df, results_df, on='Bill Name', how='left')
merged_df.to_csv(output_csv_path, index=False)
print(f"CSV with cluster labels, distances to centroid, and silhouette scores saved at {output_csv_path}")

# Now, update bills_df to use the merged data
# Again, a simple line fo code that took a lot of trial and error to get right, otherwise the plots would not have worked as intended
bills_df = merged_df.copy()

# To allow for later analysis and visualization, without running the KMeans algorithm again, I saved the cluster labels and centroids to a numpy file.
# Having these numpy files on hand coudl allow for other analyses and visualizations to be performed without repeating this entire process
centroid_output_path = f'Data/Main_Data/K-Means-Results/centroids_seed{rand}_n{init}_inertia{inertia_value:.2f}.npy'
np.save(centroid_output_path, centroids)
print(f"Centroids saved to {centroid_output_path}")

# --- KMeans Heatmap of Bills per Committee per Cluster ---
# Now for a crucial step in the analysis, I loaded each committee's embedding and predicted its cluster
# As mentioned earlier, the theory of the case was that the KMeans algorithm would cluster the bills by the committee to which they were referred in reality
committee_cluster = {}
committees = bills_df['Committee Name'].unique()
for committee in committees:
    committee_embedding_path = os.path.join(committees_embedding_folder, f"{committee}.npy")
    if os.path.exists(committee_embedding_path):
        committee_embedding = np.load(committee_embedding_path)
        committee_cluster[committee] = kmeans.predict([committee_embedding])[0]

# And now, for the purpose of the heatmaps and other visualizations, we calculate the number of bills per cluster per committee it was assigned to in reality, based on the data in the bills_df
cluster_committee_counts = pd.DataFrame(0, index=range(n_clusters), columns=committees)
for index, row in bills_df.iterrows():
    cluster = row['Cluster']
    committee = row['Committee Name']
    cluster_committee_counts.loc[cluster, committee] += 1

# This creates a heatmap of the number of bills per committee per cluster, a major part of the analysis that I wanted to visualize
plt.figure(figsize=(14, 10))
ax = sns.heatmap(cluster_committee_counts.transpose(), annot=True, fmt="d", cmap="YlOrRd", cbar=True, linewidths=0.5)
plt.ylabel("Committee")
plt.xlabel("Cluster")
plt.title(f"K-Means Heatmap: Bills per Committee per Cluster (Seed={rand}, N={init}, Inertia={inertia_value:.2f})")

# We want to understand where the embedding representing the actual committee and it's areas of focus fall within the clusters
# so we visualize this by placed a blue rectangle around the cell that represents the committee's embeddigns predicted cluster
for committee, cluster in committee_cluster.items():
    if committee in committees:
        committee_index = list(committees).index(committee)  # Get row (committee) index
        ax.add_patch(plt.Rectangle((cluster, committee_index), 1, 1, fill=False, edgecolor='blue', lw=3))

# Final adjustments to get the plot looking right consistently -- based on values i take from manual adustment after running previous versions of the script
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.subplots_adjust(left=0.232, bottom=0.076, right=1.0, top=0.94, wspace=0.2, hspace=0.2)

# and then save the heatmap image
heatmap_output_path = f'Data/Main_Data/K-Means-Results/KMeans_Heatmap_Seed{rand}_N{init}_Inertia{inertia_value:.2f}.png'
plt.savefig(heatmap_output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Heatmap with committee embedding highlights saved to {heatmap_output_path}")


# --- Cluster Distance Heatmap ---
# The purpsoe of this heatmap is to visualize the distances between the centroids of the clusters and how they relate to each other
# first we calculate the euclidean distances between the centroids
# note that openai's text-embedding-3-large model was used to embed the text, which already pre-normalize embeddings
centroid_distances = cdist(centroids, centroids, metric='euclidean')

# Plot the cluster distance heatmap with custom layout parameters
plt.figure(figsize=(10, 8))
sns.heatmap(centroid_distances, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title(f"Cluster Distance Heatmap (Seed={rand}, N={init}, Inertia={inertia_value:.2f})")
plt.xlabel("Cluster")
plt.ylabel("Cluster")
plt.subplots_adjust(left=0.104, bottom=0.068, right=1.0, top=0.957, wspace=0.2, hspace=0.2)

# Save the cluster distance heatmap
cluster_distance_heatmap_path = f'Data/Main_Data/K-Means-Results/Cluster_Distance_Heatmap_Seed{rand}_N{init}_Inertia{inertia_value:.2f}.png'
plt.savefig(cluster_distance_heatmap_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Cluster Distance Heatmap saved to {cluster_distance_heatmap_path}")

# --- UMAP Projection for 2D Visualization ---
# The final step in the analysis is to visualize the bill embeddings in a 2D space using UMAP
# for this, i decided to use umap rather than t-sne, as we have a relatively large number of bills and umap is known to be faster and more efficient
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=rand)
embeddings_2d = umap_model.fit_transform(embeddings)

# Define a color map with 21 discrete colors representing each of the 21 clusters. I found that tab20 was a good color palette for this purpose
discrete_cmap = ListedColormap(sns.color_palette("tab20", n_clusters))

# Plot UMAP 2D projection with cluster labels as colors
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap=discrete_cmap, s=10)
cbar = plt.colorbar(scatter, ticks=range(n_clusters))
cbar.set_label("Cluster")
cbar.set_ticks(range(n_clusters))
cbar.set_ticklabels(range(n_clusters))  # Ensuring discrete integer values 0-20, since each cluster was assigned an integer label from 0 to 20
plt.title(f"UMAP 2D Projection of Bill Embeddings by Cluster (Seed={rand}, N={init}, Inertia={inertia_value:.2f})")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.subplots_adjust(left=0.073, bottom=0.074, right=0.84, top=0.93, wspace=0.206, hspace=0.2)

# Save the UMAP projection plot by cluster
umap_output_path = f'Data/Main_Data/K-Means-Results/UMAP_Projection_Cluster_Seed{rand}_N{init}_Inertia{inertia_value:.2f}.png'
plt.savefig(umap_output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"UMAP Projection by Cluster saved to {umap_output_path}")

# Plot UMAP 2D projection with committee labels as colors
# This plot is similar to the previous one, but with committee labels as colors, to help us understand how this same data looks when we color it by the committee to which the bill was referred in reality
committee_labels = bills_df['Committee Name'].tolist()
unique_committees = bills_df['Committee Name'].unique()
committee_color_map = ListedColormap(sns.color_palette("tab20", len(unique_committees)))

# same as before, but this time we color the plot by committee rather than cluster
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=[list(unique_committees).index(c) for c in committee_labels], cmap=committee_color_map, s=10)
cbar = plt.colorbar(scatter, ticks=range(len(unique_committees)))
cbar.set_label("Committee")
cbar.set_ticks(range(len(unique_committees)))
cbar.set_ticklabels(unique_committees)  # Display committee names
plt.title(f"UMAP 2D Projection of Bill Embeddings by Committee (Seed={rand}, N={init}, Inertia={inertia_value:.2f})")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.subplots_adjust(left=0.073, bottom=0.074, right=0.84, top=0.93, wspace=0.206, hspace=0.2)

# Save the UMAP projection plot by committee. 
umap_committee_output_path = f'Data/Main_Data/K-Means-Results/UMAP_Projection_Committee_Seed{rand}_N{init}_Inertia{inertia_value:.2f}.png'
plt.savefig(umap_committee_output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"UMAP Projection by Committee saved to {umap_committee_output_path}")