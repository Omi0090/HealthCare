import pandas as pd

df = pd.read_csv("reports/master_provider_features.csv")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
for col in df.columns:
    print(col)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n========== TARGET DISTRIBUTION ==========")
print(df["PotentialFraud"].value_counts())

print("\n========== DUPLICATE PROVIDERS ==========")
print(df["Provider"].duplicated().sum())