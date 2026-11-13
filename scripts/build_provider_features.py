
import pandas as pd
from pathlib import Path

# Project paths
RAW_PATH = Path("data/raw")
OUTPUT_PATH = Path("data/processed")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Dataset filenames
inpatient_file = "Train_Inpatientdata-1542865627584.csv"
outpatient_file = "Train_Outpatientdata-1542865627584.csv"
labels_file = "Train-1542865627584.csv"


def create_provider_features(filename, claim_type):

    print(f"\nProcessing {claim_type} claims...")

    df = pd.read_csv(RAW_PATH / filename)

    # Convert reimbursement amount to numeric
    df["InscClaimAmtReimbursed"] = pd.to_numeric(
        df["InscClaimAmtReimbursed"],
        errors="coerce"
    )

    # Convert dates
    df["ClaimStartDt"] = pd.to_datetime(
        df["ClaimStartDt"], errors="coerce"
    )

    df["ClaimEndDt"] = pd.to_datetime(
        df["ClaimEndDt"], errors="coerce"
    )

    # Calculate claim duration
    df["ClaimDuration"] = (
        df["ClaimEndDt"] - df["ClaimStartDt"]
    ).dt.days

    # Aggregate claims by provider
    features = df.groupby("Provider").agg(
        claim_count=("ClaimID", "count"),
        total_reimbursement=("InscClaimAmtReimbursed", "sum"),
        average_reimbursement=("InscClaimAmtReimbursed", "mean"),
        median_reimbursement=("InscClaimAmtReimbursed", "median"),
        maximum_reimbursement=("InscClaimAmtReimbursed", "max"),
        unique_beneficiaries=("BeneID", "nunique"),
        average_claim_duration=("ClaimDuration", "mean"),
        maximum_claim_duration=("ClaimDuration", "max"),
        unique_attending_physicians=("AttendingPhysician", "nunique")
    )

    # Rename columns to distinguish inpatient/outpatient
    features.columns = [
        f"{claim_type}_{column}"
        for column in features.columns
    ]

    print(f"{claim_type} providers:", len(features))

    return features


# Build inpatient features
inpatient_features = create_provider_features(
    inpatient_file, "inpatient"
)

# Build outpatient features
outpatient_features = create_provider_features(
    outpatient_file, "outpatient"
)

# Combine both datasets
provider_features = inpatient_features.join(
    outpatient_features,
    how="outer"
)

# Providers without a particular claim type receive zero counts.
# These zero values represent no records in that claim category.
provider_features = provider_features.fillna(0)

# Calculate total claims
provider_features["total_claims"] = (
    provider_features["inpatient_claim_count"]
    + provider_features["outpatient_claim_count"]
)

# Calculate total reimbursement
provider_features["total_reimbursement"] = (
    provider_features["inpatient_total_reimbursement"]
    + provider_features["outpatient_total_reimbursement"]
)

# Load provider fraud labels
labels = pd.read_csv(RAW_PATH / labels_file)

labels["PotentialFraud"] = (
    labels["PotentialFraud"]
    .map({"Yes": 1, "No": 0})
)

# Join labels with provider features
final_data = provider_features.merge(
    labels,
    on="Provider",
    how="left",
    validate="one_to_one"
)

# Check missing labels
print("\nMISSING FRAUD LABELS:", final_data["PotentialFraud"].isna().sum())

# Save the feature dataset
output_file = OUTPUT_PATH / "provider_features.csv"

final_data.to_csv(output_file, index=True)

print("\nFEATURE ENGINEERING COMPLETED!")
print("Final dataset shape:", final_data.shape)
print("Total providers:", len(final_data))
print("\nFraud label distribution:")
print(final_data["PotentialFraud"].value_counts(dropna=False))
print("\nSaved file:", output_file)
