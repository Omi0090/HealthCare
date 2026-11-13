
from pathlib import Path
import pandas as pd
import gc

DATA_DIR = Path("data/raw")

# Find all 8 CSV files
files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 60)
print("HEALTHCARE FRAUD DETECTION - DATASET INSPECTION")
print("=" * 60)

if not files:
    print("ERROR: No CSV files found. Check the data/raw folder.")
else:
    print(f"\nTotal CSV files found: {len(files)}")

    for file in files:
        print("\n" + "=" * 60)
        print(f"FILE: {file.name}")

        try:
            df = pd.read_csv(file, low_memory=False)

            print(f"Rows: {df.shape[0]:,}")
            print(f"Columns: {df.shape[1]}")

            print("\nColumn names:")
            print(df.columns.tolist())

            print("\nData types:")
            print(df.dtypes.to_string())

            print("\nMissing values:")
            print(df.isnull().sum().to_string())

            print(f"\nDuplicate rows: {df.duplicated().sum():,}")

            del df
            gc.collect()

        except Exception as error:
            print(f"ERROR reading file: {error}")

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 60)
