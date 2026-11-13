
import joblib
import pandas as pd
from pathlib import Path

print("=" * 60)
print("SAVED MODEL INSPECTION")
print("=" * 60)

# Load saved model
model = joblib.load("models/best_fraud_model.pkl")

print("\n1. MODEL INFORMATION")
print("Model type:", type(model))
print("Model:", model)

# Load saved feature names
features = joblib.load("models/feature_columns.pkl")

print("\n2. SAVED FEATURE COLUMNS")
print("Number of features:", len(features))
print("Feature names:")
for feature in features:
    print("-", feature)

# Load cleaned provider dataset
df = pd.read_csv("data/processed/provider_features_clean.csv")

print("\n3. DATASET INFORMATION")
print("Dataset shape:", df.shape)
print("Dataset columns:")
print(df.columns.tolist())

print("\n4. FEATURE MATCHING")
print("Saved features found in dataset:",
      sum(feature in df.columns for feature in features))

missing = [feature for feature in features if feature not in df.columns]
print("Missing features:", missing)

print("\nINSPECTION COMPLETED")
