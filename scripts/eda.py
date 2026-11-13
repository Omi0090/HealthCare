"""
EDA Script - Generates all required visualisations for Phase 3
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Load master
master_path = Path("reports/master_provider_features.csv")
if not master_path.exists():
    master_path = Path("data/processed/cleaned_provider_features.csv")
if not master_path.exists():
    raise FileNotFoundError("No master dataset found")

df = pd.read_csv(master_path)
print(f"EDA on {df.shape}")

# Ensure PotentialFraud is string for hue
if "PotentialFraud" in df.columns and df["PotentialFraud"].dtype != object:
    # Try to map 0/1 to No/Yes for display if needed
    df["PotentialFraud"] = df["PotentialFraud"].map({1:"Yes",0:"No"}).fillna(df["PotentialFraud"])

# 1. Class distribution
plt.figure(figsize=(7,5))
sns.countplot(data=df, x="PotentialFraud", palette={"Yes":"#ef4444","No":"#10b981"})
plt.title("Class Distribution: Fraud vs Non-Fraud Providers")
plt.xlabel("PotentialFraud (Provider Label)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_class_distribution.png", dpi=200)
plt.close()

# 2. Claim distribution
fig, axes = plt.subplots(2,2, figsize=(16,12))
sns.histplot(data=df, x="TotalClaims", hue="PotentialFraud", bins=50, ax=axes[0,0], palette={"Yes":"#ef4444","No":"#10b981"}, alpha=0.6)
axes[0,0].set_title("Total Claims Distribution")
axes[0,0].set_xscale("log")
sns.boxplot(data=df, x="PotentialFraud", y="TotalClaims", ax=axes[0,1])
axes[0,1].set_title("Total Claims by Fraud")
sns.histplot(data=df, x="UniqueBeneficiaries", hue="PotentialFraud", bins=50, ax=axes[1,0], palette={"Yes":"#ef4444","No":"#10b981"})
axes[1,0].set_title("Unique Beneficiaries")
sns.boxplot(data=df, x="PotentialFraud", y="UniqueBeneficiaries", ax=axes[1,1])
axes[1,1].set_title("Beneficiaries by Fraud")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_claim_distribution.png", dpi=200)
plt.close()

# 3. Reimbursement distribution
fig, axes = plt.subplots(2,2, figsize=(16,12))
sns.histplot(data=df, x="TotalReimbursement", hue="PotentialFraud", bins=50, ax=axes[0,0], palette={"Yes":"#ef4444","No":"#10b981"})
axes[0,0].set_title("Total Reimbursement")
axes[0,0].set_xscale("log")
sns.boxplot(data=df, x="PotentialFraud", y="AverageReimbursement", ax=axes[0,1])
axes[0,1].set_title("Average Reimbursement by Fraud")
sns.boxplot(data=df, x="PotentialFraud", y="MaximumReimbursement", ax=axes[1,0])
axes[1,0].set_title("Maximum Reimbursement by Fraud")
sns.histplot(data=df, x="AverageReimbursement", hue="PotentialFraud", bins=50, ax=axes[1,1], palette={"Yes":"#ef4444","No":"#10b981"})
axes[1,1].set_title("Average Reimbursement Distribution")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_reimbursement.png", dpi=200)
plt.close()

# 4. Inpatient vs outpatient
if "InpatientClaims" in df.columns and "OutpatientClaims" in df.columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="InpatientClaims", y="OutpatientClaims", hue="PotentialFraud", alpha=0.6, palette={"Yes":"#ef4444","No":"#10b981"})
    plt.title("Inpatient vs Outpatient Claims")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_inpatient_outpatient.png", dpi=200)
    plt.close()

# 5. Provider activity (TotalClaims vs TotalReimbursement)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="TotalClaims", y="TotalReimbursement", hue="PotentialFraud", alpha=0.5, palette={"Yes":"#ef4444","No":"#10b981"})
plt.xscale("log")
plt.yscale("log")
plt.title("Provider Activity: Claims vs Reimbursement")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_provider_activity.png", dpi=200)
plt.close()

# 6. Beneficiary demographics
if "AveragePatientAge" in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x="AveragePatientAge", hue="PotentialFraud", bins=40, palette={"Yes":"#ef4444","No":"#10b981"}, alpha=0.6)
    plt.title("Average Patient Age by Fraud")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_beneficiary_age.png", dpi=200)
    plt.close()

# 7. Chronic conditions - avg prevalence
chronic_cols = [c for c in df.columns if c.startswith("Avg_ChronicCond_")]
if chronic_cols:
    means = df.groupby("PotentialFraud")[chronic_cols].mean().T
    means.columns = [str(c) for c in means.columns]
    plt.figure(figsize=(10,8))
    sns.heatmap(means, cmap="Blues", annot=True, fmt=".2f")
    plt.title("Chronic Condition Prevalence: Fraud vs Non-Fraud")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_chronic_conditions.png", dpi=200)
    plt.close()

# 8. Correlation heatmap
numeric = df.select_dtypes(include=[np.number])
# Select top 15 numeric to keep heatmap readable
corr_cols = numeric.columns[:20] if len(numeric.columns) > 20 else numeric.columns
corr = numeric[corr_cols].corr()
plt.figure(figsize=(14,12))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap - Provider Features")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "eda_correlation.png", dpi=200)
plt.close()

# 9. Monthly claim trends (if temporal data available)
temporal_path = REPORTS_DIR / "temporal_provider_features.csv"
if temporal_path.exists():
    temp = pd.read_csv(temporal_path)
    plt.figure(figsize=(10,5))
    sns.histplot(data=temp, x="AvgMonthlyClaims", bins=50, kde=True)
    plt.title("Distribution of Average Monthly Claims per Provider")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_monthly_trends.png", dpi=200)
    plt.close()

# 10. Duplicate claims
dup_summary = REPORTS_DIR / "duplicate_claims_summary.csv"
if dup_summary.exists():
    dup = pd.read_csv(dup_summary)
    print("Duplicate summary:")
    print(dup)

print("EDA complete. Reports saved to reports/eda_*.png")
