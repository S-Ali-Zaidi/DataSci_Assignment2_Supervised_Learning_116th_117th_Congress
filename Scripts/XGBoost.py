#XGBoost.py

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#input and output paths
embeddings_116_dir = "Data/embeddings/116-bills"
embeddings_117_dir = "Data/embeddings/117-bills"
train_csv = "Data/Model/Train.csv"
val_csv = "Data/Model/Validation.csv"
test_csv = "Data/Model/Test.csv"

model_path = "Data/Model/XGBoost/XGBoostModel.json"
val_report_path = "Data/Model/XGBoost/ValidationReport.txt"
test_report_path = "Data/Model/XGBoost/TestReport.txt"
val_results_csv = "Data/Model/XGBoost/ValidationResults.csv"
test_results_csv = "Data/Model/XGBoost/TestResults.csv"

# load embedding for each bill
def load_embedding(row):
    congress = row["Congress"]
    bill_name = row["Bill Name"]
    embeddings_dir = embeddings_116_dir if congress == 116 else embeddings_117_dir
    embedding_path = os.path.join(embeddings_dir, f"{bill_name}.npy")
    if os.path.exists(embedding_path):
        return np.load(embedding_path)
    else:
        raise FileNotFoundError(f"Embedding not found for: {bill_name}")

# loading splits
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

# encode committees
print("Encoding committee names...")
label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)
val_y_encoded = label_encoder.transform(val_y)
test_y_encoded = label_encoder.transform(test_y)

# Ttrain! using default hyperparameters
print("Training the XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    eval_metric="mlogloss",  # Evaluation metric for multi-class classification
    n_jobs=-1,  # Use all available CPU cores
    random_state=117
)
xgb_model.fit(train_X, train_y_encoded)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
xgb_model.save_model(model_path)
print(f"XGBoost model saved to {model_path}")

# eval and save results
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

# eval on val  set
evaluate_model(xgb_model, val_X, val_y_encoded, "Validation", label_encoder, val_report_path, val_results_csv)

# eval on test set
evaluate_model(xgb_model, test_X, test_y_encoded, "Test", label_encoder, test_report_path, test_results_csv)


print(f"Training set size: {len(train_X)} samples")
print(f"Validation set size: {len(val_X)} samples")
print(f"Test set size: {len(test_X)} samples")