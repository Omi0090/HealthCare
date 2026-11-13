import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_provider_features.csv"
)

# Separate features and target
X = df.drop(columns=["Provider", "PotentialFraud"])
y = df["PotentialFraud"].map({
    "No": 0,
    "Yes": 1
})

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display results
print("\n========== DATA SPLIT RESULTS ==========")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())

print("\nFeature columns:", len(X.columns))

print("\nData preparation completed successfully!")