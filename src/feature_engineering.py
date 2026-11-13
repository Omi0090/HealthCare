"""
Feature Engineering Module - Phase 2
Creates provider-level features from raw claim tables.
Reproducible, leakage-safe, includes peer-group comparisons and robust deviation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gc
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
PROCESSED_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

def find_raw_file(pattern: str) -> Path:
    matches = sorted(RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {RAW_DIR}")
    return matches[0]

def load_and_preprocess_claims():
    """Load inpatient and outpatient claims with date parsing and amount cleaning."""
    inpatient_path = find_raw_file("Train_Inpatientdata-*.csv")
    outpatient_path = find_raw_file("Train_Outpatientdata-*.csv")
    beneficiary_path = find_raw_file("Train_Beneficiarydata-*.csv")
    labels_path = find_raw_file("Train-*.csv")
    
    print(f"Loading inpatient: {inpatient_path.name}")
    inpatient = pd.read_csv(inpatient_path, low_memory=False)
    print(f"  -> {len(inpatient):,} rows, {len(inpatient.columns)} cols")
    
    print(f"Loading outpatient: {outpatient_path.name}")
    outpatient = pd.read_csv(outpatient_path, low_memory=False)
    print(f"  -> {len(outpatient):,} rows, {len(outpatient.columns)} cols")
    
    print(f"Loading beneficiary: {beneficiary_path.name}")
    beneficiary = pd.read_csv(beneficiary_path, low_memory=False)
    print(f"  -> {len(beneficiary):,} rows")
    
    print(f"Loading labels: {labels_path.name}")
    labels = pd.read_csv(labels_path, low_memory=False)
    print(f"  -> {len(labels):,} providers, Fraud dist: {labels['PotentialFraud'].value_counts().to_dict()}")
    
    # Clean amounts
    for df in [inpatient, outpatient]:
        for col in ["InscClaimAmtReimbursed", "DeductibleAmtPaid"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Dates
        for dcol in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
            if dcol in df.columns:
                df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
        if "ClaimStartDt" in df.columns and "ClaimEndDt" in df.columns:
            df["ClaimDuration"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days
            df["ClaimDuration"] = df["ClaimDuration"].fillna(0).clip(lower=0)
            df["ClaimMonth"] = df["ClaimStartDt"].dt.to_period("M").astype(str)
            df["ClaimYearMonth"] = df["ClaimStartDt"].dt.strftime("%Y-%m")
    
    # Beneficiary: handle chronic encoding 1/2 -> 0/1, DOB parsing
    beneficiary["DOB"] = pd.to_datetime(beneficiary["DOB"], errors='coerce')
    beneficiary["DOD"] = pd.to_datetime(beneficiary["DOD"], errors='coerce')
    chronic_cols = [c for c in beneficiary.columns if c.startswith("ChronicCond_")]
    for col in chronic_cols:
        vals = set(beneficiary[col].dropna().unique())
        if vals.issubset({1,2,1.0,2.0}):
            beneficiary[col] = beneficiary[col].map({1:0,2:1,1.0:0,2.0:1})
    # Age features will be computed via claims merge
    
    return inpatient, outpatient, beneficiary, labels

def compute_claim_level_features(inpatient: pd.DataFrame, outpatient: pd.DataFrame) -> pd.DataFrame:
    """Aggregate claim-level features to provider."""
    # Tag claim type
    inpatient["ClaimType"] = "Inpatient"
    outpatient["ClaimType"] = "Outpatient"
    # Ensure both have needed cols
    # Combine for unified processing but keep separate aggregates too
    claims = pd.concat([inpatient, outpatient], ignore_index=True)
    print(f"Combined claims: {len(claims):,}")
    
    # 1. Basic provider aggregates
    agg_dict = {
        "TotalClaims": ("ClaimID", "count"),
        "InpatientClaims": ("ClaimType", lambda x: (x == "Inpatient").sum()),
        "OutpatientClaims": ("ClaimType", lambda x: (x == "Outpatient").sum()),
        "UniqueBeneficiaries": ("BeneID", "nunique"),
        "TotalReimbursement": ("InscClaimAmtReimbursed", "sum"),
        "AverageReimbursement": ("InscClaimAmtReimbursed", "mean"),
        "MedianReimbursement": ("InscClaimAmtReimbursed", "median"),
        "MaximumReimbursement": ("InscClaimAmtReimbursed", "max"),
        "MinimumReimbursement": ("InscClaimAmtReimbursed", "min"),
        "StdReimbursement": ("InscClaimAmtReimbursed", "std"),
        "AverageDeductible": ("DeductibleAmtPaid", "mean"),
        "TotalDeductible": ("DeductibleAmtPaid", "sum"),
        "AverageClaimDuration": ("ClaimDuration", "mean"),
        "MaxClaimDuration": ("ClaimDuration", "max"),
        "UniqueAttendingPhysicians": ("AttendingPhysician", "nunique"),
        "UniqueOperatingPhysicians": ("OperatingPhysician", "nunique"),
        "UniqueOtherPhysicians": ("OtherPhysician", "nunique"),
    }
    # Filter agg_dict to existing cols
    valid_agg = {}
    for k, (col, func) in agg_dict.items():
        if col in claims.columns or col == "ClaimType":
            valid_agg[k] = (col, func)
    
    provider_claims = claims.groupby("Provider").agg(**valid_agg).reset_index()
    
    # Derived ratios
    provider_claims["InpatientClaimRatio"] = provider_claims["InpatientClaims"] / provider_claims["TotalClaims"].replace(0, np.nan)
    provider_claims["OutpatientClaimRatio"] = provider_claims["OutpatientClaims"] / provider_claims["TotalClaims"].replace(0, np.nan)
    provider_claims["ClaimsPerBeneficiary"] = provider_claims["TotalClaims"] / provider_claims["UniqueBeneficiaries"].replace(0, np.nan)
    provider_claims["AvgReimbursementPerBeneficiary"] = provider_claims["TotalReimbursement"] / provider_claims["UniqueBeneficiaries"].replace(0, np.nan)
    provider_claims["ReimbursementVariation"] = provider_claims["StdReimbursement"] / provider_claims["AverageReimbursement"].replace(0, np.nan)  # CV
    provider_claims["ReimbursementVariation"] = provider_claims["ReimbursementVariation"].fillna(0)
    
    # Inpatient-only aggregates
    if not inpatient.empty:
        inp_agg = inpatient.groupby("Provider").agg(
            Inp_TotalReimbursement=("InscClaimAmtReimbursed", "sum"),
            Inp_AvgReimbursement=("InscClaimAmtReimbursed", "mean"),
            Inp_MaxReimbursement=("InscClaimAmtReimbursed", "max"),
            Inp_AvgDuration=("ClaimDuration", "mean"),
            Inp_TotalClaims=("ClaimID", "count"),
        ).reset_index()
        provider_claims = provider_claims.merge(inp_agg, on="Provider", how="left")
    # Outpatient-only
    if not outpatient.empty:
        out_agg = outpatient.groupby("Provider").agg(
            Out_TotalReimbursement=("InscClaimAmtReimbursed", "sum"),
            Out_AvgReimbursement=("InscClaimAmtReimbursed", "mean"),
            Out_MaxReimbursement=("InscClaimAmtReimbursed", "max"),
            Out_AvgDuration=("ClaimDuration", "mean"),
            Out_TotalClaims=("ClaimID", "count"),
        ).reset_index()
        provider_claims = provider_claims.merge(out_agg, on="Provider", how="left")
    
    # Fill missing (providers with only one type)
    provider_claims = provider_claims.fillna(0)
    
    # Temporal features: monthly frequency
    monthly = claims.groupby(["Provider", "ClaimYearMonth"]).size().reset_index(name="MonthlyClaimCount")
    # Compute provider monthly stats
    monthly_stats = monthly.groupby("Provider").agg(
        AvgMonthlyClaims=("MonthlyClaimCount", "mean"),
        MaxMonthlyClaims=("MonthlyClaimCount", "max"),
        StdMonthlyClaims=("MonthlyClaimCount", "std"),
        NumActiveMonths=("ClaimYearMonth", "nunique")
    ).reset_index()
    monthly_stats["StdMonthlyClaims"] = monthly_stats["StdMonthlyClaims"].fillna(0)
    # Claim frequency variation
    monthly_stats["MonthlyClaimVariation"] = monthly_stats["StdMonthlyClaims"] / monthly_stats["AvgMonthlyClaims"].replace(0, np.nan)
    monthly_stats["MonthlyClaimVariation"] = monthly_stats["MonthlyClaimVariation"].fillna(0)
    
    provider_claims = provider_claims.merge(monthly_stats, on="Provider", how="left")
    provider_claims = provider_claims.fillna(0)
    
    # Duplicate claim count (provider-level)
    # Exact duplicate: same BeneID, Provider, ClaimStartDt, ClaimEndDt, Amount
    dup_keys = ["BeneID", "Provider", "ClaimStartDt", "ClaimEndDt", "InscClaimAmtReimbursed"]
    # Use vectorized duplicated
    dup_mask = claims.duplicated(subset=dup_keys, keep=False)
    dup_counts = claims[dup_mask].groupby("Provider").size().reset_index(name="DuplicateClaimCount")
    provider_claims = provider_claims.merge(dup_counts, on="Provider", how="left")
    provider_claims["DuplicateClaimCount"] = provider_claims["DuplicateClaimCount"].fillna(0)
    provider_claims["DuplicateClaimRatio"] = provider_claims["DuplicateClaimCount"] / provider_claims["TotalClaims"].replace(0, np.nan)
    provider_claims["DuplicateClaimRatio"] = provider_claims["DuplicateClaimRatio"].fillna(0)
    
    # Provider billing frequency: avg days between claims?
    # For each provider, compute days since last claim
    claims_sorted = claims.sort_values(["Provider", "ClaimStartDt"])
    claims_sorted["DaysSinceLastClaim"] = claims_sorted.groupby("Provider")["ClaimStartDt"].diff().dt.days
    billing_freq = claims_sorted.groupby("Provider").agg(
        AvgDaysBetweenClaims=("DaysSinceLastClaim", "mean"),
        MinDaysBetweenClaims=("DaysSinceLastClaim", "min"),
    ).reset_index()
    billing_freq["AvgDaysBetweenClaims"] = billing_freq["AvgDaysBetweenClaims"].fillna(0)
    billing_freq["MinDaysBetweenClaims"] = billing_freq["MinDaysBetweenClaims"].fillna(0)
    provider_claims = provider_claims.merge(billing_freq, on="Provider", how="left")
    provider_claims = provider_claims.fillna(0)
    
    return provider_claims, claims

def compute_beneficiary_features(beneficiary: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Compute beneficiary aggregates at provider level."""
    # Merge claims with beneficiary to get age and chronic conditions
    # Need BeneID mapping: beneficiary has chronic + demographics, claims has BeneID+Provider+ClaimStartDt
    # DOB already parsed
    # Compute age at claim time
    claims_for_age = claims[["Provider", "BeneID", "ClaimStartDt"]].copy()
    claims_for_age["ClaimStartDt"] = pd.to_datetime(claims_for_age["ClaimStartDt"], errors='coerce')
    beneficiary_age = beneficiary[["BeneID", "DOB"] + [c for c in beneficiary.columns if c.startswith("ChronicCond_")] ].copy()
    # Add age-related cols if available: Gender, Race etc?
    extra_cols = [c for c in ["Gender","Race","State","County","RenalDiseaseIndicator","NoOfMonths_PartACov","NoOfMonths_PartBCov"] if c in beneficiary.columns]
    for c in extra_cols:
        beneficiary_age[c] = beneficiary[c]
    
    merged = claims_for_age.merge(beneficiary_age, on="BeneID", how="left")
    merged["ClaimStartDt"] = pd.to_datetime(merged["ClaimStartDt"], errors='coerce')
    merged["Age"] = (merged["ClaimStartDt"] - merged["DOB"]).dt.days / 365.25
    merged.loc[(merged["Age"] < 0) | (merged["Age"] > 120), "Age"] = np.nan
    
    # One record per provider-beneficiary (avoid overcounting multiple claims per same patient)
    patient_level = merged.drop_duplicates(subset=["Provider","BeneID"])
    
    # Aggregations
    chronic_cols = [c for c in beneficiary.columns if c.startswith("ChronicCond_")]
    agg_dict = {
        "AveragePatientAge": ("Age", "mean"),
        "MaximumPatientAge": ("Age", "max"),
        "MinimumPatientAge": ("Age", "min"),
        "StdPatientAge": ("Age", "std"),
        "UniquePatients": ("BeneID", "nunique"),
    }
    for col in chronic_cols:
        agg_dict[f"Avg_{col}"] = (col, "mean")
    # Add demographic aggregates if available
    if "Gender" in patient_level.columns:
        agg_dict["Avg_Gender"] = ("Gender", "mean")
    if "NoOfMonths_PartACov" in patient_level.columns:
        agg_dict["Avg_MonthsPartA"] = ("NoOfMonths_PartACov", "mean")
    if "NoOfMonths_PartBCov" in patient_level.columns:
        agg_dict["Avg_MonthsPartB"] = ("NoOfMonths_PartBCov", "mean")
    
    provider_beneficiary = patient_level.groupby("Provider").agg(**agg_dict).reset_index()
    provider_beneficiary["StdPatientAge"] = provider_beneficiary["StdPatientAge"].fillna(0)
    
    # Also compute beneficiary concentration: UniquePatients vs UniqueBeneficiaries from claims?
    # We'll keep both and later create ratio
    
    return provider_beneficiary

def engineer_features(save: bool = True) -> pd.DataFrame:
    """Main pipeline: orchestrates feature engineering and saves master dataset."""
    inpatient, outpatient, beneficiary, labels = load_and_preprocess_claims()
    
    print("\n[Feature Engineering] Computing claim-level features...")
    provider_claims, claims_combined = compute_claim_level_features(inpatient, outpatient)
    print(f"  Claim features shape: {provider_claims.shape}")
    
    print("[Feature Engineering] Computing beneficiary features...")
    provider_bene = compute_beneficiary_features(beneficiary, claims_combined)
    print(f"  Beneficiary features shape: {provider_bene.shape}")
    
    # Merge all
    master = provider_claims.merge(provider_bene, on="Provider", how="left")
    # Merge labels
    labels["PotentialFraud"] = labels["PotentialFraud"].map({"Yes": "Yes", "No": "No", "Yes ": "Yes", "No ": "No"})
    # Keep string for reporting, also create binary
    master = master.merge(labels, on="Provider", how="inner")
    print(f"Merged master shape: {master.shape}")
    print(f"  Fraud distribution:\n{master['PotentialFraud'].value_counts()}")
    
    # Create peer groups: by TotalClaims quantile
    master["PeerGroup"] = pd.qcut(master["TotalClaims"], q=5, labels=["Very Low","Low","Medium","High","Very High"], duplicates='drop').astype(str)
    # Compute peer deviation for key numeric features
    peer_features = ["TotalReimbursement","AverageReimbursement","ClaimsPerBeneficiary","AverageClaimDuration","DuplicateClaimRatio"]
    for feat in peer_features:
        if feat in master.columns:
            peer_median = master.groupby("PeerGroup")[feat].transform("median")
            peer_mad = master.groupby("PeerGroup")[feat].transform(lambda x: np.median(np.abs(x - np.median(x))) if len(x)>0 else 1.0)
            peer_mad = peer_mad.replace(0, 1.0).fillna(1.0)
            master[f"{feat}_PeerMedian"] = peer_median
            master[f"{feat}_PeerMAD"] = peer_mad
            master[f"{feat}_PeerZ"] = (master[feat] - peer_median) / (1.4826 * peer_mad)
    
    # Handle missing bene features (providers not in bene merge? should be inner but some may miss)
    master = master.fillna(0)
    
    # Additional derived: log transform for skewed reimbursements
    master["LogTotalReimbursement"] = np.log1p(master["TotalReimbursement"])
    master["LogAverageReimbursement"] = np.log1p(master["AverageReimbursement"])
    
    # Ensure PotentialFraud strings remain, create binary for modeling (not saved as target? save both)
    # For modeling we need numeric, but keep string for display
    master["PotentialFraudBinary"] = master["PotentialFraud"].map({"Yes":1,"No":0})
    
    # Data quality checks
    print(f"\n[Quality] Missing: {master.isnull().sum().sum()}, Duplicates: {master.duplicated().sum()}")
    print(f"  Columns: {len(master.columns)}")
    
    if save:
        REPORTS_DIR.mkdir(exist_ok=True)
        PROCESSED_DIR.mkdir(exist_ok=True)
        master.to_csv(REPORTS_DIR / "master_provider_features.csv", index=False)
        master.to_csv(PROCESSED_DIR / "cleaned_provider_features.csv", index=False)
        # Also save provider_features.csv for backward compat (old pipeline)
        # Create legacy-compatible file with subset?
        print(f"Saved master to {REPORTS_DIR / 'master_provider_features.csv'}")
        print(f"Saved cleaned to {PROCESSED_DIR / 'cleaned_provider_features.csv'}")
    
    # Free memory
    del inpatient, outpatient, beneficiary, claims_combined, provider_claims, provider_bene, labels
    gc.collect()
    
    return master

if __name__ == "__main__":
    df = engineer_features(save=True)
    print(df.head())
    print(df.columns.tolist())
