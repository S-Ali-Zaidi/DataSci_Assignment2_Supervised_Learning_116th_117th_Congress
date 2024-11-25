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

'''
This script is almost identical to the kmeans_predict_rules.py script.
Since we already ran it and have already identified clusters, analyzed representative samples of bills within each cluster, and identified the themes of each cluster, 
we can now just run the k_means algorithm again with the exact same random seed and initialization settings to get the same, reproducible results.

The only difference in this script compared to the kmeans_predict_rules.py script is in the visualization part.
In the umap projection, we assign descriptive labels to each cluster based on the most common themes in the bills that we identified after viewing the markdown file generated in the explore_clusters.py script.
'''

# Repeat all the steps from the kmeans_predict_rules.py script, this time I'm not going to explain the code as it's already been covered in the previous script.
init = 1000
rand = 7
embeddings_folder = 'Data/embeddings/bills'
committees_embedding_folder = 'Data/embeddings/committees'
bills_csv_path = 'Data/Main_Data/Bills.csv'
output_csv_path = f'Data/Main_Data/Results/KMeans_Bills_Seed{rand}_N{init}.csv'
embedding_files = [f for f in os.listdir(embeddings_folder) if f.endswith('.npy')]
embedding_files.sort()  

embeddings = []
bill_names = []

for file in embedding_files:
    embedding_path = os.path.join(embeddings_folder, file)
    embeddings.append(np.load(embedding_path))
    bill_name = file.replace('.npy', '')
    bill_names.append(bill_name)

embeddings = np.array(embeddings)

bills_df = pd.read_csv(bills_csv_path)

n_clusters = 21
kmeans = KMeans(n_clusters=n_clusters, n_init=init, random_state=rand)
kmeans.fit(embeddings)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia_value = kmeans.inertia_
distances_to_centroid = np.linalg.norm(embeddings - centroids[cluster_labels], axis=1)
silhouette_scores = silhouette_samples(embeddings, cluster_labels)


results_df = pd.DataFrame({
    'Bill Name': bill_names,
    'Cluster': cluster_labels,
    'Distance to Centroid': distances_to_centroid,
    'Silhouette Score': silhouette_scores
})

merged_df = pd.merge(bills_df, results_df, on='Bill Name', how='left')
merged_df.to_csv(output_csv_path, index=False)
print(f"CSV with cluster labels, distances to centroid, and silhouette scores saved at {output_csv_path}")
bills_df = merged_df.copy()

centroid_output_path = f'Data/Main_Data/Results/centroids_seed{rand}_n{init}_inertia{inertia_value:.2f}.npy'
np.save(centroid_output_path, centroids)
print(f"Centroids saved to {centroid_output_path}")

committee_cluster = {}
committees = bills_df['Committee Name'].unique()
for committee in committees:
    committee_embedding_path = os.path.join(committees_embedding_folder, f"{committee}.npy")
    if os.path.exists(committee_embedding_path):
        committee_embedding = np.load(committee_embedding_path)
        committee_cluster[committee] = kmeans.predict([committee_embedding])[0]


cluster_committee_counts = pd.DataFrame(0, index=range(n_clusters), columns=committees)
for index, row in bills_df.iterrows():
    cluster = row['Cluster']
    committee = row['Committee Name']
    cluster_committee_counts.loc[cluster, committee] += 1

plt.figure(figsize=(14, 10))
ax = sns.heatmap(cluster_committee_counts.transpose(), annot=True, fmt="d", cmap="YlOrRd", cbar=True, linewidths=0.5)
plt.ylabel("Committee")
plt.xlabel("Cluster")
plt.title(f"K-Means Heatmap: Bills per Committee per Cluster (Seed={rand}, N={init}, Inertia={inertia_value:.2f})")

for committee, cluster in committee_cluster.items():
    if committee in committees:
        committee_index = list(committees).index(committee)  # Get row (committee) index
        ax.add_patch(plt.Rectangle((cluster, committee_index), 1, 1, fill=False, edgecolor='blue', lw=3))

plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.subplots_adjust(left=0.232, bottom=0.076, right=1.0, top=0.94, wspace=0.2, hspace=0.2)

heatmap_output_path = f'Data/Main_Data/Results/KMeans_Heatmap_Seed{rand}_N{init}_Inertia{inertia_value:.2f}.png'
plt.savefig(heatmap_output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Heatmap with committee embedding highlights saved to {heatmap_output_path}")

# --- Cluster Distance Heatmap ---
centroid_distances = cdist(centroids, centroids, metric='euclidean')
plt.figure(figsize=(10, 8))
sns.heatmap(centroid_distances, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title(f"Cluster Distance Heatmap (Seed={rand}, N={init}, Inertia={inertia_value:.2f})")
plt.xlabel("Cluster")
plt.ylabel("Cluster")
plt.subplots_adjust(left=0.104, bottom=0.068, right=1.0, top=0.957, wspace=0.2, hspace=0.2)

cluster_distance_heatmap_path = f'Data/Main_Data/Results/Cluster_Distance_Heatmap_Seed{rand}_N{init}_Inertia{inertia_value:.2f}.png'
plt.savefig(cluster_distance_heatmap_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Cluster Distance Heatmap saved to {cluster_distance_heatmap_path}")

# --- UMAP Projection for 2D Visualization ---
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=rand)
embeddings_2d = umap_model.fit_transform(embeddings)
discrete_cmap = ListedColormap(sns.color_palette("tab20", n_clusters))

############################
'''
# This part is the major difference between this script and the kmeans_predict_rules.py script.
# since we already ran this before and have the cluster labels, we can assign descriptive labels to each cluster based on the most common themes in the bills.
'''
cluster_labels = [
    "Prohibition of Federal Funding for COVID-19 Mandates and Limitation of Government Actions",
    "Education and Workforce Development for Underserved Communities",
    "Congressional Oversight and Requests for Executive Documentation",
    "Enhancement of Veterans’ Services and Benefits",
    "Environmental Conservation and Renewable Energy Initiatives",
    "Healthcare Access Improvement and Medicare Policy Reforms",
    "Recognition and Commemoration of Military and Service Personnel",
    "Government Ethics, Accountability, and Regulatory Reforms",
    "Healthcare Access and Public Health Reforms",
    "Designation of National Awareness Weeks and Observances",
    "Tax Policy Amendments and Economic Incentives",
    "Infrastructure Development and Environmental Management",
    "Foreign Policy, Sanctions, and National Security",
    "Law Enforcement, Immigration, and Border Security Policies",
    "Tax Policy Amendments, Economic Incentives, and Infrastructure Investment",
    "Comprehensive Infrastructure, Climate Action, Health, Housing, and Democracy Protection",
    "Gun Control and Firearms Regulation",
    "Education, Small Business Support, Social Programs, and Regulatory Reforms",
    "Resolutions and Recognitions for Social Issues, Human Rights, and Commendations",
    "Law Enforcement Recognition, National Security, and Social Resolutions",
    "Designation of Federal and Postal Facilities and Memorials"
]
#############################

plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=kmeans.labels_, cmap=discrete_cmap, s=10)
cbar = plt.colorbar(scatter, ticks=range(n_clusters))
cbar.set_label("Cluster Label")
cbar.set_ticks(range(n_clusters))
cbar.set_ticklabels(cluster_labels)
plt.title(f"UMAP 2D Projection of Bill Embeddings by Cluster with Labels (Seed={rand}, N={init}, Inertia={inertia_value:.2f})")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.subplots_adjust(left=0.073, bottom=0.074, right=0.84, top=0.93, wspace=0.206, hspace=0.2)

umap_output_path = f'Data/Main_Data/Results/UMAP_Projection_Cluster_Labeled_Seed{rand}_N{init}_Inertia{inertia_value:.2f}.png'
plt.savefig(umap_output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"UMAP Projection by Cluster with Labels saved to {umap_output_path}")

# --- UMAP Projection by Committee ---
committee_labels = bills_df['Committee Name'].tolist()
unique_committees = bills_df['Committee Name'].unique()
committee_color_map = ListedColormap(sns.color_palette("tab20", len(unique_committees)))

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

umap_committee_output_path = f'Data/Main_Data/Results/UMAP_Projection_Committee_Seed{rand}_N{init}_Inertia{inertia_value:.2f}.png'
plt.savefig(umap_committee_output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"UMAP Projection by Committee saved to {umap_committee_output_path}")