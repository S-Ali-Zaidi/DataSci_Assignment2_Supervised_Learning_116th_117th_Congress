#SplitDataset.py

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/Main_Data/Merged_Bills.csv")

# Using stratified splitting to ensure that the test, validation, and training sets
# have roughly the same ditribution of bills from each committee
train, temp = train_test_split(df, test_size=0.3, stratify=df["Committee Name"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["Committee Name"], random_state=42)

train.to_csv("Data/Model/Train.csv", index=False)
val.to_csv("Data/Model/Validation.csv", index=False)
test.to_csv("Data/Model/Test.csv", index=False)

print(f"Training samples: {len(train)}")
print(f"Validation samples: {len(val)}")
print(f"Test samples: {len(test)}")

# double checking to ensure the distirbution looks healthy in each split
def check_distribution(df, name):
    print(f"\n{name} Distribution:")
    print(df["Committee Name"].value_counts(normalize=True))

# Check distribution in each split
check_distribution(train, "Training Set")
check_distribution(val, "Validation Set")
check_distribution(test, "Test Set")