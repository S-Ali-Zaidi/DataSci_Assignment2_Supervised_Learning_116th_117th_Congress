#RandomForest.py

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# input and output paths
embeddings_116_dir = "Data/embeddings/116-bills"
embeddings_117_dir = "Data/embeddings/117-bills"

train_csv = "Data/Model/Train.csv"
val_csv = "Data/Model/Validation.csv"
test_csv = "Data/Model/Test.csv"

model_path = "Data/Model/RandomForest/RandomForestModel.joblib"
val_report_path = "Data/Model/RandomForest/ValidationReport.txt"
test_report_path = "Data/Model/RandomForest/TestReport.txt"
val_results_csv = "Data/Model/RandomForest/ValidationResults.csv"
test_results_csv = "Data/Model/RandomForest/TestResults.csv"


def load_embedding(row):
    congress = row["Congress"]
    bill_name = row["Bill Name"]
    embeddings_dir = embeddings_116_dir if congress == 116 else embeddings_117_dir
    embedding_path = os.path.join(embeddings_dir, f"{bill_name}.npy")

    if os.path.exists(embedding_path):
        return np.load(embedding_path)
    else:
        raise FileNotFoundError(f"Embedding not found for: {bill_name}")

# defining the function to ientify the split in the dataset
def process_split(split_csv):
    df = pd.read_csv(split_csv)
    print(f"Loading embeddings for {split_csv}...")
    df["Embedding"] = df.apply(load_embedding, axis=1)
    X = np.vstack(df["Embedding"].values)  
    y = df["Committee Name"]
    return X, y

train_X, train_y = process_split(train_csv)
val_X, val_y = process_split(val_csv)
test_X, test_y = process_split(test_csv)

# Need to ensure tht the committee names are encoded properly
print("Encoding committee names...")
label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)
val_y_encoded = label_encoder.transform(val_y)
test_y_encoded = label_encoder.transform(test_y)

#Train! using default hyperparameters
print("Training the Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
rf_model.fit(train_X, train_y_encoded)

# And save!
joblib.dump(rf_model, model_path)
print(f"Random Forest model saved to {model_path}")

# func for eval on val and test sets
def evaluate_model(model, X, y, split_name, label_encoder, report_path, results_csv):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions, target_names=label_encoder.classes_)
    conf_matrix = confusion_matrix(y, predictions)

    with open(report_path, "w") as report_file:
        report_file.write(f"=== {split_name} Report ===\n")
        report_file.write(f"Accuracy: {accuracy:.4f}\n\n")
        report_file.write(report)
        report_file.write("\nConfusion Matrix:\n")
        report_file.write(str(conf_matrix))
    print(f"{split_name} Report saved to {report_path}")

    results = pd.DataFrame({
        "True Label": label_encoder.inverse_transform(y),
        "Predicted Label": label_encoder.inverse_transform(predictions)
    })
    results.to_csv(results_csv, index=False)
    print(f"{split_name} Predictions saved to {results_csv}")

    print(f"\n=== {split_name} Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

# eval on val set
evaluate_model(rf_model, val_X, val_y_encoded, "Validation", label_encoder, val_report_path, val_results_csv)
# eval on test set
evaluate_model(rf_model, test_X, test_y_encoded, "Test", label_encoder, test_report_path, test_results_csv)


print(f"Training set size: {len(train_X)} samples")
print(f"Validation set size: {len(val_X)} samples")
print(f"Test set size: {len(test_X)} samples")