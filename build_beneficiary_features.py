from pathlib import Path
import pandas as pd

DATA = Path("data/raw")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

print("Loading data...")

beneficiary = pd.read_csv(
    DATA / "Train_Beneficiarydata-1542865627584.csv"
)

inpatient = pd.read_csv(
    DATA / "Train_Inpatientdata-1542865627584.csv",
    usecols=["Provider", "BeneID", "ClaimStartDt"]
)

outpatient = pd.read_csv(
    DATA / "Train_Outpatientdata-1542865627584.csv",
    usecols=["Provider", "BeneID", "ClaimStartDt"]
)

claims = pd.concat([inpatient, outpatient], ignore_index=True)

claims["ClaimStartDt"] = pd.to_datetime(
    claims["ClaimStartDt"], errors="coerce"
)
beneficiary["DOB"] = pd.to_datetime(
    beneficiary["DOB"], errors="coerce"
)

claims = claims.merge(
    beneficiary,
    on="BeneID",
    how="left",
    validate="many_to_one"
)

claims["Age"] = (
    claims["ClaimStartDt"] - claims["DOB"]
).dt.days / 365.25

claims.loc[
    (claims["Age"] < 0) | (claims["Age"] > 120),
    "Age"
] = float("nan")

# One record per provider-beneficiary pair
patient_data = claims.drop_duplicates(
    subset=["Provider", "BeneID"]
).copy()

# Identify chronic condition columns
chronic_columns = [
    col for col in beneficiary.columns
    if col.startswith("ChronicCond_")
]

# Convert 1/2 encoding into 0/1
for col in chronic_columns:
    patient_data[col] = pd.to_numeric(
        patient_data[col], errors="coerce"
    ).map({1: 0, 2: 1})

# Aggregate features by provider
aggregation = {
    "AveragePatientAge": ("Age", "mean"),
    "MaximumPatientAge": ("Age", "max"),
    "MinimumPatientAge": ("Age", "min"),
    "UniquePatients": ("BeneID", "nunique")
}

for col in chronic_columns:
    aggregation[f"Avg_{col}"] = (col, "mean")

features = patient_data.groupby("Provider").agg(
    **aggregation
).reset_index()

output = REPORTS / "beneficiary_provider_features.csv"
features.to_csv(output, index=False)

print("\n========== CORRECTED BENEFICIARY FEATURES ==========")
print("Providers:", len(features))
print("Features:", len(features.columns) - 1)

print("\nSample results:")
print(features.head(10).to_string(index=False))

print("\nSaved:", output)
print("\nChronic condition encoding corrected successfully!")