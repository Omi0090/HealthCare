
from pathlib import Path
import pandas as pd
import gc

DATA = Path("data/raw")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

print("Loading training claims...")

# Load inpatient claims
inpatient = pd.read_csv(
    DATA / "Train_Inpatientdata-1542865627584.csv"
)

inpatient["ClaimType"] = "Inpatient"

# Load outpatient claims
outpatient = pd.read_csv(
    DATA / "Train_Outpatientdata-1542865627584.csv"
)

outpatient["ClaimType"] = "Outpatient"

# Combine claims
claims = pd.concat(
    [inpatient, outpatient],
    ignore_index=True
)

print("Total claims loaded:", len(claims))

# Convert dates
claims["ClaimStartDt"] = pd.to_datetime(
    claims["ClaimStartDt"], errors="coerce"
)

claims["ClaimEndDt"] = pd.to_datetime(
    claims["ClaimEndDt"], errors="coerce"
)

# Calculate claim duration
claims["ClaimDuration"] = (
    claims["ClaimEndDt"] - claims["ClaimStartDt"]
).dt.days

# Aggregate claims by provider
features = claims.groupby("Provider").agg(
    TotalClaims=("ClaimID", "count"),
    InpatientClaims=("ClaimType", lambda x: (x == "Inpatient").sum()),
    OutpatientClaims=("ClaimType", lambda x: (x == "Outpatient").sum()),
    UniqueBeneficiaries=("BeneID", "nunique"),
    TotalReimbursement=("InscClaimAmtReimbursed", "sum"),
    AverageReimbursement=("InscClaimAmtReimbursed", "mean"),
    MedianReimbursement=("InscClaimAmtReimbursed", "median"),
    MaximumReimbursement=("InscClaimAmtReimbursed", "max"),
    AverageDeductible=("DeductibleAmtPaid", "mean"),
    AverageClaimDuration=("ClaimDuration", "mean")
).reset_index()

# Load provider labels
labels = pd.read_csv(
    DATA / "Train-1542865627584.csv"
)

# Attach target labels
final_data = features.merge(
    labels,
    on="Provider",
    how="inner",
    validate="one_to_one"
)

# Save the provider-level dataset
output = REPORTS / "provider_features.csv"

final_data.to_csv(output, index=False)

print("\n========== FEATURE ENGINEERING SUMMARY ==========")
print("Total providers:", len(final_data))
print("Total features:", len(final_data.columns) - 2)
print("\nFeature columns:")
print(final_data.columns.tolist())

print("\nFraud distribution:")
print(final_data["PotentialFraud"].value_counts())

print("\nSample provider features:")
print(final_data.head(5).to_string(index=False))

print("\nSaved successfully:", output)

del claims, inpatient, outpatient
gc.collect()
