import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

print("=" * 70)
print("HEALTHCARE FRAUD DETECTION - DATASET STRUCTURE")
print("=" * 70)

for file in sorted(DATA_DIR.glob("*.csv")):

    print("\n" + "=" * 70)
    print("FILE:", file.name)

    # Read only column names and first 3 records
    df = pd.read_csv(file, nrows=3)

    print("\nColumns:")
    for col in df.columns:
        print("-", col)

    print("\nSample records:")
    print(df.to_string(index=False))

print("\nINSPECTION COMPLETED")