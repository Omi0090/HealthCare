import pandas as pd
from pathlib import Path

# Load the provider feature dataset
input_file = Path("data/processed/provider_features.csv")
output_file = Path("data/processed/provider_features_clean.csv")

df = pd.read_csv(input_file)

print("ORIGINAL DATASET SHAPE:", df.shape)

# Remove unwanted index columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Check required columns
required_columns = ["Provider", "PotentialFraud"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

# Check duplicate provider IDs
if df["Provider"].duplicated().any():
    raise ValueError("Duplicate provider IDs found!")

# Check target values
if not set(df["PotentialFraud"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected target labels found!")

# Check missing values
if df.isnull().sum().sum() > 0:
    raise ValueError("Missing values found!")

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nCLEANING COMPLETED SUCCESSFULLY!")
print("Final dataset shape:", df.shape)
print("Total providers:", len(df))
print("\nTarget distribution:")
print(df["PotentialFraud"].value_counts())
print("\nSaved to:", output_file)