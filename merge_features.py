from pathlib import Path
import pandas as pd

REPORTS = Path("reports")

print("Loading feature datasets...")

provider = pd.read_csv(
    REPORTS / "provider_features.csv"
)

beneficiary = pd.read_csv(
    REPORTS / "beneficiary_provider_features.csv"
)

print("Provider features:", provider.shape)
print("Beneficiary features:", beneficiary.shape)

# Merge using Provider ID
master = provider.merge(
    beneficiary,
    on="Provider",
    how="inner",
    validate="one_to_one"
)

# Save master dataset
output = REPORTS / "master_provider_features.csv"
master.to_csv(output, index=False)

print("\n========== MASTER DATASET ==========")
print("Total providers:", len(master))
print("Total columns:", len(master.columns))
print("Missing values:", master.isnull().sum().sum())

print("\nSample data:")
print(master.head().to_string(index=False))

print("\nSaved:", output)
print("\nFeature merging completed successfully!")