"""
Data Loader Module - AI Healthcare Fraud Detection
Handles loading, validation, and basic inspection of raw Medicare datasets.
Leakage-safe: never exposes test labels.

Design: Follows PPT methodology - provider, beneficiary, inpatient, outpatient tables.
Respects data governance: uses only public, de-identified or synthetic data.
"""
import gc
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

# Supported raw file patterns (avoid hardcoded timestamps where possible)
TRAIN_PATTERNS = {
    "labels": "Train-*.csv",
    "beneficiary": "Train_Beneficiarydata-*.csv",
    "inpatient": "Train_Inpatientdata-*.csv",
    "outpatient": "Train_Outpatientdata-*.csv",
}

TEST_PATTERNS = {
    "providers": "Test-*.csv",
    "beneficiary": "Test_Beneficiarydata-*.csv",
    "inpatient": "Test_Inpatientdata-*.csv",
    "outpatient": "Test_Outpatientdata-*.csv",
}

def find_file(directory: Path, pattern: str) -> Optional[Path]:
    """Find first matching file for a pattern."""
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None

def get_raw_file_map(train: bool = True) -> Dict[str, Path]:
    """Return dict of available raw files."""
    patterns = TRAIN_PATTERNS if train else TEST_PATTERNS
    result = {}
    for key, pat in patterns.items():
        f = find_file(RAW_DIR, pat)
        if f and f.exists():
            result[key] = f
    return result

def load_raw_datasets(train: bool = True) -> Dict[str, pd.DataFrame]:
    """Load raw CSVs into dataframes with low_memory=False for mixed types."""
    file_map = get_raw_file_map(train=train)
    datasets = {}
    for key, path in file_map.items():
        try:
            df = pd.read_csv(path, low_memory=False)
            datasets[key] = df
            print(f"[LOAD] {key}: {path.name} -> {df.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
    return datasets

def inventory_datasets(save_report: bool = True) -> Dict:
    """Create dataset inventory with verification checks."""
    inventory = {}
    all_files = sorted(RAW_DIR.glob("*.csv"))
    inventory["total_files"] = len(all_files)
    inventory["files"] = []
    
    for f in all_files:
        try:
            df = pd.read_csv(f, low_memory=False, nrows=5)
            # Full read for row count (chunked if large)
            row_count = sum(1 for _ in open(f, 'r', encoding='utf-8', errors='ignore')) - 1
            cols = list(pd.read_csv(f, nrows=0).columns)
            dups = pd.read_csv(f, low_memory=False).duplicated().sum() if row_count < 200000 else "skipped (>200k rows)"
            missing = pd.read_csv(f, low_memory=False).isnull().sum().sum() if row_count < 200000 else "skipped"
            inventory["files"].append({
                "filename": f.name,
                "rows": row_count,
                "columns": len(cols),
                "column_names": cols,
                "duplicates": dups,
                "total_missing": missing
            })
        except Exception as e:
            inventory["files"].append({"filename": f.name, "error": str(e)})
    if save_report:
        REPORTS_DIR.mkdir(exist_ok=True)
        import json
        with open(REPORTS_DIR / "dataset_inventory.json", "w") as jf:
            json.dump(inventory, jf, indent=2, default=str)
    return inventory

def verify_relationships(train: bool = True) -> Dict:
    """Verify primary/foreign keys and dataset relationships."""
    datasets = load_raw_datasets(train=train)
    report = {}
    if "labels" in datasets and "inpatient" in datasets and "outpatient" in datasets:
        labels = datasets["labels"]
        inpatient = datasets["inpatient"]
        outpatient = datasets["outpatient"]
        beneficiary = datasets.get("beneficiary", pd.DataFrame())
        
        report["train_providers"] = labels["Provider"].nunique()
        report["train_labels_dist"] = labels["PotentialFraud"].value_counts().to_dict() if "PotentialFraud" in labels.columns else {}
        report["inpatient_providers"] = inpatient["Provider"].nunique() if "Provider" in inpatient.columns else 0
        report["outpatient_providers"] = outpatient["Provider"].nunique() if "Provider" in outpatient.columns else 0
        report["inpatient_claims"] = len(inpatient)
        report["outpatient_claims"] = len(outpatient)
        report["beneficiaries"] = len(beneficiary) if not beneficiary.empty else 0
        report["unique_bene_inpatient"] = inpatient["BeneID"].nunique() if "BeneID" in inpatient.columns else 0
        report["unique_bene_outpatient"] = outpatient["BeneID"].nunique() if "BeneID" in outpatient.columns else 0
        report["providers_with_labels_only"] = labels["Provider"].nunique()
        # Check FK coverage
        all_providers_claims = set(inpatient["Provider"].unique()).union(set(outpatient["Provider"].unique()))
        labeled_providers = set(labels["Provider"].unique())
        report["providers_in_claims_not_in_labels"] = len(all_providers_claims - labeled_providers)
        report["providers_in_labels_not_in_claims"] = len(labeled_providers - all_providers_claims)
        report["class_imbalance_ratio"] = labels["PotentialFraud"].value_counts(normalize=True).to_dict() if "PotentialFraud" in labels.columns else {}
        # Data leakage checks
        report["leakage_checks"] = {
            "target_in_claims": "PotentialFraud" in inpatient.columns or "PotentialFraud" in outpatient.columns,
            "claimid_duplicates": inpatient["ClaimID"].duplicated().sum() if "ClaimID" in inpatient.columns else 0,
            "outpatient_claimid_duplicates": outpatient["ClaimID"].duplicated().sum() if "ClaimID" in outpatient.columns else 0,
        }
    return report

def create_data_dictionary(save: bool = True) -> pd.DataFrame:
    """Create data dictionary from actual datasets."""
    datasets = load_raw_datasets(train=True)
    rows = []
    for table_name, df in datasets.items():
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = int(df[col].isna().sum())
            unique = int(df[col].nunique())
            sample_vals = str(df[col].dropna().unique()[:3].tolist())[:100]
            rows.append({
                "Table": table_name,
                "Column": col,
                "Dtype": dtype,
                "Missing": missing,
                "Unique": unique,
                "SampleValues": sample_vals,
                "PrimaryKeyCandidate": col in ["Provider", "BeneID", "ClaimID"],
                "Description": get_column_description(col)
            })
    dd = pd.DataFrame(rows)
    if save:
        REPORTS_DIR.mkdir(exist_ok=True)
        dd.to_csv(REPORTS_DIR / "data_dictionary.csv", index=False)
    return dd

def get_column_description(col: str) -> str:
    """Human readable description for known Medicare columns."""
    descriptions = {
        "Provider": "Unique healthcare provider identifier (primary key for provider-level analysis)",
        "PotentialFraud": "Fraud label: Yes/No at provider level (training only). Investigation signal, not proof.",
        "BeneID": "Beneficiary identifier (de-identified)",
        "ClaimID": "Unique claim identifier",
        "ClaimStartDt": "Claim start date",
        "ClaimEndDt": "Claim end date",
        "InscClaimAmtReimbursed": "Insurance claim amount reimbursed (USD)",
        "DeductibleAmtPaid": "Deductible amount paid",
        "AttendingPhysician": "Attending physician identifier",
        "OperatingPhysician": "Operating physician identifier",
        "OtherPhysician": "Other physician identifier",
        "AdmissionDt": "Inpatient admission date",
        "DischargeDt": "Discharge date",
        "ClmAdmitDiagnosisCode": "Admission diagnosis code (ICD-9)",
        "DiagnosisGroupCode": "Diagnosis group code (DRG)",
        "DOB": "Date of birth (beneficiary)",
        "DOD": "Date of death (if applicable)",
        "Gender": "1=Male, 2=Female",
        "Race": "Beneficiary race code",
        "RenalDiseaseIndicator": "Renal disease indicator Y/N",
        "State": "Beneficiary state",
        "County": "Beneficiary county",
        "NoOfMonths_PartACov": "Months of Part A coverage",
        "NoOfMonths_PartBCov": "Months of Part B coverage",
        "ChronicCond_Alzheimer": "Chronic condition: Alzheimer (1=No, 2=Yes)",
        "ChronicCond_Heartfailure": "Chronic condition: Heart Failure",
        "ChronicCond_KidneyDisease": "Chronic condition: Kidney Disease",
        "ChronicCond_Cancer": "Chronic condition: Cancer",
        "ChronicCond_ObstrPulmonary": "Chronic condition: COPD",
        "ChronicCond_Depression": "Chronic condition: Depression",
        "ChronicCond_Diabetes": "Chronic condition: Diabetes",
        "ChronicCond_IschemicHeart": "Chronic condition: Ischemic Heart",
        "ChronicCond_Osteoporasis": "Chronic condition: Osteoporosis",
        "ChronicCond_rheumatoidarthritis": "Chronic condition: Rheumatoid Arthritis",
        "ChronicCond_stroke": "Chronic condition: Stroke/Transient Ischemic Attack",
        "IPAnnualReimbursementAmt": "Annual inpatient reimbursement",
        "OPAnnualReimbursementAmt": "Annual outpatient reimbursement",
    }
    # Generic for diagnosis/procedure codes
    if "ClmDiagnosisCode" in col:
        return "Claim diagnosis code (ICD-9)"
    if "ClmProcedureCode" in col:
        return "Claim procedure code"
    return descriptions.get(col, "Medicare claim attribute")

def data_quality_report(save: bool = True) -> Dict:
    """Generate data quality report with missing, duplicate, invalid checks."""
    datasets = load_raw_datasets(train=True)
    report = {}
    for name, df in datasets.items():
        q = {}
        q["rows"] = len(df)
        q["columns"] = len(df.columns)
        q["total_missing"] = int(df.isnull().sum().sum())
        q["missing_by_col"] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
        q["duplicate_rows"] = int(df.duplicated().sum())
        # Invalid values checks
        invalid = {}
        if "InscClaimAmtReimbursed" in df.columns:
            neg = (pd.to_numeric(df["InscClaimAmtReimbursed"], errors='coerce') < 0).sum()
            invalid["negative_reimbursement"] = int(neg)
        if "DeductibleAmtPaid" in df.columns:
            neg2 = (pd.to_numeric(df["DeductibleAmtPaid"], errors='coerce') < 0).sum()
            invalid["negative_deductible"] = int(neg2)
        if "ClaimStartDt" in df.columns and "ClaimEndDt" in df.columns:
            try:
                s = pd.to_datetime(df["ClaimStartDt"], errors='coerce')
                e = pd.to_datetime(df["ClaimEndDt"], errors='coerce')
                invalid["end_before_start"] = int((e < s).sum())
                invalid["invalid_dates"] = int(s.isna().sum() + e.isna().sum())
            except:
                pass
        q["invalid_values"] = invalid
        # Class imbalance
        if name == "labels" and "PotentialFraud" in df.columns:
            q["class_distribution"] = df["PotentialFraud"].value_counts().to_dict()
            q["class_imbalance"] = df["PotentialFraud"].value_counts(normalize=True).to_dict()
        report[name] = q
    if save:
        REPORTS_DIR.mkdir(exist_ok=True)
        import json
        with open(REPORTS_DIR / "data_quality_report.json", "w") as jf:
            json.dump(report, jf, indent=2, default=str)
        # Also save human readable txt
        with open(REPORTS_DIR / "data_quality_report.txt", "w") as tf:
            tf.write("HEALTHCARE FRAUD DETECTION - DATA QUALITY REPORT\n")
            tf.write("="*60 + "\n")
            for k, v in report.items():
                tf.write(f"\nTable: {k}\n")
                for kk, vv in v.items():
                    tf.write(f"  {kk}: {vv}\n")
    return report

if __name__ == "__main__":
    print(inventory_datasets())
    print(verify_relationships())
    print(create_data_dictionary().head())
    print(data_quality_report())
