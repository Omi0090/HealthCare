import pandas as pd

file_path = "data/processed/provider_features.csv"

df = pd.read_csv(file_path)

print("\nDATASET SHAPE:")
print(df.shape)

print("\nALL COLUMN NAMES:")
for i, col in enumerate(df.columns, start=1):
    print(i, col)

print("\nDATA TYPES:")
print(df.dtypes)

print("\nFIRST 5 ROWS:")
print(df.head().to_string())

print("\nPOSSIBLE EXTRA INDEX COLUMNS:")
print([col for col in df.columns if "unnamed" in col.lower()])

print("\nTARGET DISTRIBUTION:")
print(df["PotentialFraud"].value_counts())