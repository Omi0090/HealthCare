
from pathlib import Path
import pandas as pd

DATA = Path("data/raw")

# Actual filenames from your dataset
train_labels = pd.read_csv(
    DATA / "Train-1542865627584.csv"
)

inpatient = pd.read_csv(
    DATA / "Train_Inpatientdata-1542865627584.csv",
    usecols=["Provider", "ClaimID", "BeneID",
             "InscClaimAmtReimbursed"]
)

outpatient = pd.read_csv(
    DATA / "Train_Outpatientdata-1542865627584.csv",
    usecols=["Provider", "ClaimID", "BeneID",
             "InscClaimAmtReimbursed"]
)

# 1. Fraud distribution
print("\n========== FRAUD DISTRIBUTION ==========")
print(train_labels["PotentialFraud"].value_counts())
print("\nPercentage:")
print(
    train_labels["PotentialFraud"]
    .value_counts(normalize=True).mul(100).round(2)
)

# 2. Combine claim summaries
claims = pd.concat(
    [inpatient, outpatient],
    ignore_index=True
)

print("\n========== CLAIM SUMMARY ==========")
print("Total claims:", len(claims))
print("Unique providers:", claims["Provider"].nunique())
print("Unique beneficiaries:", claims["BeneID"].nunique())
print("Unique Claim IDs:", claims["ClaimID"].nunique())

# 3. Provider-level aggregation
provider_features = claims.groupby("Provider").agg(
    TotalClaims=("ClaimID", "count"),
    UniqueBeneficiaries=("BeneID", "nunique"),
    TotalReimbursement=("InscClaimAmtReimbursed", "sum"),
    AverageReimbursement=("InscClaimAmtReimbursed", "mean")
).reset_index()

print("\n========== PROVIDER FEATURES ==========")
print(provider_features.head(10).to_string(index=False))

# 4. Add provider labels
analysis = provider_features.merge(
    train_labels,
    on="Provider",
    how="inner"
)

print("\n========== LABELLED PROVIDER ANALYSIS ==========")
print("Providers with claims and labels:", len(analysis))

print("\nAverage features by fraud category:")
print(
    analysis.groupby("PotentialFraud")[
        ["TotalClaims", "UniqueBeneficiaries",
         "TotalReimbursement", "AverageReimbursement"]
    ].mean().round(2)
)

# 5. Save the results
Path("reports").mkdir(exist_ok=True)

provider_features.to_csv(
    "reports/provider_features_initial.csv",
    index=False
)

analysis.to_csv(
    "reports/provider_analysis_initial.csv",
    index=False
)

print("\nAnalysis completed successfully!")
