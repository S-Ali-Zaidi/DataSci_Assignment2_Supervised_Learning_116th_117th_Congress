import pandas as pd

file_116 = "Data/Main_Data/116-Bills.csv"
file_117 = "Data/Main_Data/117-Bills.csv"
output_merged = "Data/Main_Data/Merged_Bills.csv"
output_committee_counts = "Data/Main_Data/Committee_Counts.csv"

columns_to_keep = ["Bill Name", "Congress", "Committee Name"]

df_116 = pd.read_csv(file_116, usecols=columns_to_keep)
df_117 = pd.read_csv(file_117, usecols=columns_to_keep)

merged_df = pd.concat([df_116, df_117], ignore_index=True)

merged_df.to_csv(output_merged, index=False)
print(f"Merged dataset saved to: {output_merged}")

committee_counts = merged_df["Committee Name"].value_counts().reset_index()
committee_counts.columns = ["Committee Name", "Count"]

committee_counts.to_csv(output_committee_counts, index=False)
print(f"Committee counts saved to: {output_committee_counts}")

print("\n=== Summary ===")
print(f"Total Bills: {len(merged_df)}")
print(f"Total Unique Committees: {len(committee_counts)}")
print(committee_counts.head(10))