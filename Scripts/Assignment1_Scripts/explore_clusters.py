import os
import pandas as pd

# In this script, we will explore the actual text of bills in each cluster to identify the most representative bills that helps us understand the themes of each cluster.
# we already have the necessary data for each bill from the previous steps of the analysis, including the cluster assignments, distance to centroid, and silhouette scores.

# Define paths
# note that in the previous script, i ran multiple iterations of the clustering algorithm with different random seeds and k-means initialization settings.
# so it's important to choose the correct CSV file that corresponds to the desired clustering results.
input_path = 'Data/Main_Data/116-K-Means-Results/KMeans_Bills_Seed7_N1000.csv'  
output_folder = 'Data/Main_Data/116-K-Means-Results/Highlights'  
markdown_output_path = os.path.join(output_folder, 'Cluster_Highlights.md')  

# Ensure the output folder exists and load the data
os.makedirs(output_folder, exist_ok=True)
df = pd.read_csv(input_path)

# Initialize a list to store the Markdown content
# I'm using markdown rather than plan text as i can easily format this in Obsidian or GitHub for better readability.
markdown_content = []

# And now for the main loop that will go through each cluster and identify the most representative bills on the basis of distance to centroid.
# we want to ensure we are getting a wide enough sample of texts to understand the themes of each cluster
# so we focus on getting the best bill for each committee in the cluster, and then the top 10 bills overall in the cluster.
# this way, we ensure we understand how bills assigned to all committes in reality relate to the cluster's theme, including committtes that may have only one bill in the cluster.
for cluster_id in range(21):
    
    # we start by getting the overall closest bill to the centroid for each committee in the cluster
    cluster_df = df[df['Cluster'] == cluster_id]
    lowest_distance_per_committee = (
        cluster_df.loc[cluster_df.groupby('Committee Name')['Distance to Centroid'].idxmin()]
        .sort_values(by='Distance to Centroid')
    )
    
    # then we get the top 10 overall bills in the cluster based on distance to centroid, while ensuring that none of the bills are duplicates from the previous step
    top_10_overall = cluster_df.nsmallest(10, 'Distance to Centroid')
    combined_df = pd.concat([lowest_distance_per_committee, top_10_overall]).drop_duplicates()
    
    # In these steps, we take the bills we jsut identified and save them to a CSV file for each cluster
    selected_columns_df = combined_df[['Bill Name', 'Long Title', 'Distance to Centroid', 'Silhouette Score']]
    sorted_df = selected_columns_df.sort_values(by='Distance to Centroid')
    cluster_output_path = os.path.join(output_folder, f"Cluster_{cluster_id}.csv")
    sorted_df.to_csv(cluster_output_path, index=False)
    
    # we take these same bills, format them in markdown according to my desired format, and save them to a markdown file for easy reading
    markdown_content.append(f"# Cluster {cluster_id}\n")
    for _, row in sorted_df.iterrows():
        bill_info = (
            f"**Bill Name**: {row['Bill Name']}\n"
            f"**Long Title**: {row['Long Title']}\n"
            f"**Distance to Centroid**: {row['Distance to Centroid']}\n"
            f"**Silhouette Score**: {row['Silhouette Score']}\n"
            "\n---\n"  # let's make sure each bill is separated by a horizontal line and some space
        )
        markdown_content.append(bill_info)
    markdown_content.append("\n")  # Extra space between clusters

# And finally, save the markdown content to a single file for all clusters
with open(markdown_output_path, 'w') as md_file:
    md_file.write('\n'.join(markdown_content))

print(f"Data saved successfully in the '{output_folder}' directory with CSVs for each cluster and a Markdown summary.")