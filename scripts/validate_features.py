
import pandas as pd
import numpy as np
from pathlib import Path

file_path = Path("data/processed/provider_features.csv")

df = pd.read_csv(file_path)

print("=" * 55)
print("PROVIDER FEATURE DATASET VALIDATION")
print("=" * 55)

# 1. Dataset dimensions
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Duplicate provider check
print("\n2. DUPLICATE PROVIDERS")
print("Duplicate providers:", df["Provider"].duplicated().sum())

# 3. Missing values
print("\n3. MISSING VALUES")
print("Total missing values:", df.isnull().sum().sum())

# 4. Infinite values
numeric_df = df.select_dtypes(include=np.number)
print("\n4. INFINITE VALUES")
print("Total infinite values:",
      np.isinf(numeric_df).sum().sum())

# 5. Fraud label validation
print("\n5. FRAUD LABELS")
print(df["PotentialFraud"].value_counts(dropna=False))

# 6. Check claim count consistency
expected_total = (
    df["inpatient_claim_count"]
    + df["outpatient_claim_count"]
)

print("\n6. CLAIM COUNT VALIDATION")
print("Incorrect total claim counts:",
      (df["total_claims"] != expected_total).sum())

# 7. Feature summary
print("\n7. IMPORTANT FEATURE SUMMARY")
columns = [
    "inpatient_claim_count",
    "outpatient_claim_count",
    "total_claims",
    "total_reimbursement"
]

print(df[columns].describe().round(2))

print("\n" + "=" * 55)
print("VALIDATION COMPLETED")
print("=" * 55)
