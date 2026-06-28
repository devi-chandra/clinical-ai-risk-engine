import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = "data/Dengue_diseases_dataset_modified (1).csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("Clinical AI Risk Stratification Engine")
print("=" * 50)

print(f"\nDataset Shape: {df.shape}")

# -----------------------------
# Feature Engineering
# -----------------------------

# Create new feature
df["is_child"] = (df["gender"] == "Child").astype(int)

# Convert gender into binary
gender_map = {"Male": 1, "Female": 0, "Child": 0}
df["gender"] = df["gender"].map(gender_map).astype(int)

# -----------------------------
# Missing Value Handling
# -----------------------------

numeric_columns = [
    "hemoglobin_g_dl",
    "wbc_count",
    "differential_count",
    "rbc_count",
    "platelet_count",
    "platelet_distribution_width"
]

imputer = SimpleImputer(strategy="median")

df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# -----------------------------
# Features and Target
# -----------------------------

X = df.drop("dengue_label", axis=1)
y = df["dengue_label"]

feature_names = X.columns.tolist()

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Create models directory
# -----------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")
joblib.dump(imputer, "models/imputer.pkl")

print("\nData preprocessing completed successfully!")
print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")
print("\nSaved:")
print("- scaler.pkl")
print("- feature_names.pkl")
print("- imputer.pkl")

# -----------------------------
# Model Training
# -----------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric="logloss"
    )
}

results = {}

best_model = None
best_name = ""
best_auc = 0

print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

for name, model in models.items():

    # Logistic Regression uses scaled data
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:,1]

    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    }

    print(f"\n{name}")
    print("-"*40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)

print(f"Model: {best_name}")
print(f"ROC-AUC: {best_auc:.4f}")

joblib.dump(best_model, "models/risk_model.pkl")

results_df = pd.DataFrame(results).T
results_df.to_csv("models/model_results.csv")

print("\nSaved:")
print("- risk_model.pkl")
print("- model_results.csv")