import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

files = [
    "Train_Inpatientdata-1542865627584.csv",
    "Train_Outpatientdata-1542865627584.csv",
]

print("=" * 65)
print("HEALTHCARE FRAUD DETECTION - CLAIM DATA INSPECTION")
print("=" * 65)

for filename in files:
    path = RAW / filename

    print(f"\n\nFILE: {filename}")

    df = pd.read_csv(path, nrows=5)

    print("\nColumns:")
    for col in df.columns:
        print("-", col)

    print("\nSample records:")
    print(df.head(3).to_string(index=False))

    print("\nData types:")
    print(df.dtypes)

print("\n\nINSPECTION COMPLETED")