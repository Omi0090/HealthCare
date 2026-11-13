import pandas as pd

# Load master dataset
df = pd.read_csv("reports/master_provider_features.csv")

print("Before cleaning:")
print("Missing values:", df.isnull().sum().sum())

# Fill missing AverageDeductible with median
median_value = df["AverageDeductible"].median()

df["AverageDeductible"] = df["AverageDeductible"].fillna(
    median_value
)

# Save cleaned dataset
df.to_csv(
    "data/processed/cleaned_provider_features.csv",
    index=False
)

print("\nAfter cleaning:")
print("Missing values:", df.isnull().sum().sum())

print("\nDataset shape:", df.shape)
print("\nCleaned dataset saved successfully!")