
import pandas as pd
from pathlib import Path

data_path = Path("data/raw")

files = [
    "Train_Inpatientdata-1542865627584.csv",
    "Train_Outpatientdata-1542865627584.csv"
]

for filename in files:

    print("\n" + "=" * 60)
    print("FILE:", filename)
    print("=" * 60)

    df = pd.read_csv(data_path / filename)

    # 1. Claim amount analysis
    print("\nCLAIM AMOUNT SUMMARY:")
    print(
        df["InscClaimAmtReimbursed"]
        .describe()
        .round(2)
    )

    # 2. Claim date analysis
    dates = pd.to_datetime(
        df["ClaimStartDt"],
        errors="coerce"
    )

    print("\nCLAIM DATE RANGE:")
    print("Earliest claim:", dates.min())
    print("Latest claim:", dates.max())

    # 3. Claims per provider
    claims_per_provider = (
        df.groupby("Provider")["ClaimID"].count()
    )

    print("\nCLAIMS PER PROVIDER SUMMARY:")
    print(claims_per_provider.describe().round(2))

print("\nDATA ANALYSIS COMPLETED SUCCESSFULLY!")
