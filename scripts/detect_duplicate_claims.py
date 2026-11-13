"""
AI-Based Healthcare Insurance Fraud Detection
Module: Claim-Level Duplicate Detection

Screening Tool Disclaimer:
This script is an automated screening tool designed to flag potential duplicate billing
patterns for administrative and clinical review. A duplicate match is NOT proof of
insurance fraud; legitimate duplicate claims may occur due to rebilling after initial
denials, void/replace workflows, system corrections, or multi-phase legitimate treatments.

Requirements:
- Read training inpatient and outpatient claim CSV files from data/raw/
- Automatically identify files using filename patterns
- Detect exact duplicates and potential duplicate billing (BeneID, Provider, dates, amounts, codes)
- Keep inpatient and outpatient results separate
- Preserve all original records (no records deleted)
- Never treat repeated BeneID, Provider, or ClaimID values alone as proof of fraud
- Assign a risk flag, group ID, and human-readable reason to every flagged record
- Save results in reports/ as CSV files
- Output a concise summary table
- Memory and CPU friendly (< 2GB RAM footprint, vectorized operations)
"""

import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


# =====================================================================
# PATH CONFIGURATION
# =====================================================================
RAW_DIR = Path("data/raw")
REPORTS_DIR = Path("reports")


def ensure_reports_dir() -> Path:
    """Ensure that the reports/ output directory exists."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def find_raw_claims_file(directory: Path, pattern: str) -> Path:
    """
    Automatically identify training claim files matching a given pattern.
    Avoids hardcoded timestamps in filenames.
    """
    matches = sorted(list(directory.glob(pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No file matching pattern '{pattern}' found in {directory.resolve()}"
        )
    return matches[0]


# =====================================================================
# DUPLICATE DETECTION LOGIC
# =====================================================================
def screen_duplicate_claims(
    csv_path: Path,
    claim_type: str,
    output_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Screen inpatient or outpatient claims for exact and potential duplicate billing.

    Parameters:
    -----------
    csv_path : Path
        Path to raw CSV file.
    claim_type : str
        'Inpatient' or 'Outpatient'.
    output_dir : Path
        Directory to save flagged CSV report.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict]: Flagged DataFrame and summary statistics dictionary.
    """
    print(f"\n{'=' * 75}")
    print(f"SCREENING [{claim_type.upper()}] CLAIMS FOR DUPLICATE BILLING")
    print(f"Source file: {csv_path.name}")
    print(f"{'=' * 75}")

    # 1. Load data with optimized dtypes
    # Read original DataFrame intact without modifying raw source
    df_raw = pd.read_csv(csv_path, low_memory=False)
    total_claims = len(df_raw)
    print(f"Loaded {total_claims:,} raw {claim_type} claims.")

    # 2. Identify clinical and diagnosis code columns safely
    diag_cols = [c for c in df_raw.columns if "ClmDiagnosisCode" in c]
    proc_cols = [c for c in df_raw.columns if "ClmProcedureCode" in c]
    admit_diag_col = "ClmAdmitDiagnosisCode" if "ClmAdmitDiagnosisCode" in df_raw.columns else None

    # 3. Create cleaned screening keys (handling missing values safely)
    # Using normalized series prevents NaN inequality issues or whitespace mismatch
    clean_bene = df_raw["BeneID"].fillna("").astype(str).str.strip()
    clean_provider = df_raw["Provider"].fillna("").astype(str).str.strip()
    clean_start_dt = df_raw["ClaimStartDt"].fillna("").astype(str).str.strip()
    clean_end_dt = df_raw["ClaimEndDt"].fillna("").astype(str).str.strip()
    clean_reimbursed = (
        pd.to_numeric(df_raw["InscClaimAmtReimbursed"], errors="coerce")
        .fillna(0.0)
        .round(2)
    )
    clean_deductible = (
        pd.to_numeric(df_raw["DeductibleAmtPaid"], errors="coerce")
        .fillna(0.0)
        .round(2)
    )
    clean_primary_diag = (
        df_raw["ClmDiagnosisCode_1"].fillna("").astype(str).str.strip()
        if "ClmDiagnosisCode_1" in df_raw.columns
        else pd.Series("", index=df_raw.index)
    )
    clean_primary_proc = (
        df_raw["ClmProcedureCode_1"].fillna("").astype(str).str.strip()
        if "ClmProcedureCode_1" in df_raw.columns
        else pd.Series("", index=df_raw.index)
    )

    # Temporary DataFrame for vectorized matching
    df_keys = pd.DataFrame({
        "BeneID": clean_bene,
        "Provider": clean_provider,
        "ClaimStartDt": clean_start_dt,
        "ClaimEndDt": clean_end_dt,
        "InscClaimAmtReimbursed": clean_reimbursed,
        "DeductibleAmtPaid": clean_deductible,
        "PrimaryDiag": clean_primary_diag,
        "PrimaryProc": clean_primary_proc
    }, index=df_raw.index)

    # -----------------------------------------------------------------
    # DUPLICATE DETECTION TIERS
    # -----------------------------------------------------------------
    # Tier 1: Exact full duplicates (all columns including ClaimID)
    mask_exact_full = df_raw.duplicated(keep=False)
    exact_full_count = int(mask_exact_full.sum())

    # Tier 2: Exact content duplicates (all columns excluding ClaimID)
    cols_excluding_id = [c for c in df_raw.columns if c != "ClaimID"]
    mask_exact_content = df_raw.duplicated(subset=cols_excluding_id, keep=False) & (~mask_exact_full)
    exact_content_count = int(mask_exact_content.sum())

    # Tier 3: Same Provider Duplicate Billing
    # Criteria: Identical BeneID + Provider + ClaimStartDt + ClaimEndDt + Reimbursed Amount
    # Note: A patient visiting a provider on DIFFERENT dates is NOT duplicate billing.
    # Repeated BeneID or Provider alone is NEVER treated as fraud.
    same_provider_cols = ["BeneID", "Provider", "ClaimStartDt", "ClaimEndDt", "InscClaimAmtReimbursed"]
    mask_same_provider = df_keys.duplicated(subset=same_provider_cols, keep=False)

    # Sub-tier 3A: Same Provider with Matching Primary Diagnosis or Procedure
    mask_same_prov_diag = (
        mask_same_provider
        & (df_keys["PrimaryDiag"] != "")
        & df_keys.duplicated(subset=same_provider_cols + ["PrimaryDiag"], keep=False)
    )

    # Sub-tier 3B: Same Provider but Different/Missing Primary Diagnosis (potential split billing)
    mask_same_prov_other = mask_same_provider & (~mask_same_prov_diag)

    # Tier 4: Cross-Provider Duplicate Billing
    # Criteria: Identical BeneID + ClaimStartDt + ClaimEndDt + Reimbursed Amount across DIFFERENT providers
    cross_cols = ["BeneID", "ClaimStartDt", "ClaimEndDt", "InscClaimAmtReimbursed"]
    mask_cross_provider_all = df_keys.duplicated(subset=cross_cols, keep=False)
    mask_cross_provider = mask_cross_provider_all & (~mask_same_provider)

    # Master duplicate mask (combining all tiers, mutual exclusivity handled hierarchically)
    mask_any_duplicate = (
        mask_exact_full
        | mask_exact_content
        | mask_same_provider
        | mask_cross_provider
    )
    total_flagged = int(mask_any_duplicate.sum())

    print(f"Duplicate screening analysis completed:")
    print(f"  - Full exact duplicates: {exact_full_count}")
    print(f"  - Content exact duplicates (excl. ClaimID): {exact_content_count}")
    print(f"  - Same-provider duplicates with matching primary diagnosis: {int(mask_same_prov_diag.sum())}")
    print(f"  - Same-provider duplicates with differing/blank diagnosis: {int(mask_same_prov_other.sum())}")
    print(f"  - Cross-provider potential duplicate claims: {int(mask_cross_provider.sum())}")
    print(f"  - Total claims flagged for review: {total_flagged:,} ({total_flagged / total_claims:.2%})")

    # -----------------------------------------------------------------
    # BUILD AUDIT REPORTS (WITHOUT DELETING ANY RECORDS)
    # -----------------------------------------------------------------
    if total_flagged > 0:
        # Create a deep copy of flagged claims to avoid altering original dataframe
        flagged_df = df_raw[mask_any_duplicate].copy()
        keys_flagged = df_keys[mask_any_duplicate].copy()

        # Generate cluster / group identifiers so auditors can see matched claim pairs
        # Hierarchical grouping key
        # If same provider duplicate: group by BeneID, Provider, ClaimStartDt, ClaimEndDt, Amount
        # If cross provider duplicate: group by BeneID, ClaimStartDt, ClaimEndDt, Amount
        keys_flagged["GroupKey"] = np.where(
            mask_same_provider[mask_any_duplicate],
            keys_flagged["BeneID"] + "_" + keys_flagged["Provider"] + "_" + keys_flagged["ClaimStartDt"] + "_" + keys_flagged["InscClaimAmtReimbursed"].astype(str),
            keys_flagged["BeneID"] + "_CROSS_" + keys_flagged["ClaimStartDt"] + "_" + keys_flagged["InscClaimAmtReimbursed"].astype(str)
        )

        # Assign unique group numbers
        prefix = "INP" if claim_type.lower() == "inpatient" else "OUT"
        dense_group_ids = keys_flagged.groupby("GroupKey", sort=False).ngroup() + 1
        flagged_df["DuplicateGroupID"] = dense_group_ids.apply(lambda x: f"{prefix}_DUP_{x:05d}")

        # Compute duplicate group sizes
        group_sizes = flagged_df.groupby("DuplicateGroupID")["ClaimID"].transform("count")
        flagged_df["DuplicateGroupSize"] = group_sizes

        # Assign Risk Flags and Detailed Human-Readable Reasons
        risk_flags = pd.Series("NO_DUPLICATE", index=flagged_df.index)
        risk_reasons = pd.Series("", index=flagged_df.index)
        matched_fields = pd.Series("", index=flagged_df.index)

        # Apply flags hierarchically
        # 1. Exact Full
        sub_full = mask_exact_full[mask_any_duplicate]
        risk_flags[sub_full] = "HIGH_RISK_EXACT_FULL_DUPLICATE"
        risk_reasons[sub_full] = "Exact identical record submitted across all fields including ClaimID."
        matched_fields[sub_full] = "All Columns"

        # 2. Exact Content
        sub_content = mask_exact_content[mask_any_duplicate]
        risk_flags[sub_content] = "HIGH_RISK_EXACT_CONTENT_DUPLICATE"
        risk_reasons[sub_content] = "Exact identical clinical, financial, and physician details resubmitted under a distinct ClaimID."
        matched_fields[sub_content] = "All Columns except ClaimID"

        # 3. Same Provider with matching primary diagnosis
        sub_same_diag = mask_same_prov_diag[mask_any_duplicate]
        risk_flags[sub_same_diag] = "HIGH_RISK_SAME_PROVIDER_DUPLICATE"
        risk_reasons[sub_same_diag] = (
            "Same provider billed identical reimbursement amount for same beneficiary on identical dates with matching primary diagnosis code."
        )
        matched_fields[sub_same_diag] = "BeneID, Provider, ClaimStartDt, ClaimEndDt, InscClaimAmtReimbursed, ClmDiagnosisCode_1"

        # 4. Same Provider with differing diagnosis
        sub_same_other = mask_same_prov_other[mask_any_duplicate]
        risk_flags[sub_same_other] = "MEDIUM_RISK_SAME_PROVIDER_DUPLICATE"
        risk_reasons[sub_same_other] = (
            "Same provider billed identical reimbursement amount for same beneficiary on identical dates (potential split billing or altered diagnosis codes)."
        )
        matched_fields[sub_same_other] = "BeneID, Provider, ClaimStartDt, ClaimEndDt, InscClaimAmtReimbursed"

        # 5. Cross Provider
        sub_cross = mask_cross_provider[mask_any_duplicate]
        risk_flags[sub_cross] = "MEDIUM_RISK_CROSS_PROVIDER_DUPLICATE"
        risk_reasons[sub_cross] = (
            "Identical beneficiary, service dates, and reimbursement amount billed across different healthcare providers (cross-facility duplicate screening)."
        )
        matched_fields[sub_cross] = "BeneID, ClaimStartDt, ClaimEndDt, InscClaimAmtReimbursed"

        flagged_df["DuplicateRiskFlag"] = risk_flags
        flagged_df["DuplicateReason"] = risk_reasons
        flagged_df["MatchedKeyFields"] = matched_fields

        # Extract all non-empty diagnosis codes per claim for clinical review
        def collect_diagnoses(row):
            codes = []
            if admit_diag_col and pd.notna(row.get(admit_diag_col)):
                codes.append(str(row[admit_diag_col]).strip())
            for c in diag_cols:
                val = row.get(c)
                if pd.notna(val) and str(val).strip() != "" and str(val).strip() != "nan":
                    codes.append(str(val).strip())
            return "; ".join(sorted(list(set(codes))))

        flagged_df["AllDiagnosisCodes"] = flagged_df.apply(collect_diagnoses, axis=1)

        # Reorder columns to put duplicate audit indicators at the front
        audit_cols = [
            "DuplicateGroupID",
            "DuplicateGroupSize",
            "DuplicateRiskFlag",
            "DuplicateReason",
            "MatchedKeyFields",
            "AllDiagnosisCodes"
        ]
        remaining_cols = [c for c in flagged_df.columns if c not in audit_cols]
        flagged_df = flagged_df[audit_cols + remaining_cols]

        # Sort by DuplicateGroupID, BeneID, ClaimStartDt for intuitive reviewing
        flagged_df = flagged_df.sort_values(
            by=["DuplicateGroupID", "BeneID", "ClaimStartDt", "ClaimID"]
        ).reset_index(drop=True)

        total_groups = flagged_df["DuplicateGroupID"].nunique()
    else:
        flagged_df = pd.DataFrame()
        total_groups = 0

    # 4. Save results to reports/ CSV
    report_filename = f"{claim_type.lower()}_duplicate_claims_report.csv"
    report_path = output_dir / report_filename
    flagged_df.to_csv(report_path, index=False)
    print(f"Saved {claim_type} duplicate report to: {report_path.resolve()} ({len(flagged_df):,} records)")

    # 5. Clean up memory for 12GB laptop compatibility
    del df_keys
    del df_raw
    gc.collect()

    # Summary statistics
    summary_stats = {
        "ClaimType": claim_type,
        "TotalClaims": total_claims,
        "ExactFullDuplicates": exact_full_count,
        "ExactContentDuplicates": exact_content_count,
        "SameProviderDiagMatchClaims": int(mask_same_prov_diag.sum()),
        "SameProviderOtherClaims": int(mask_same_prov_other.sum()),
        "CrossProviderClaims": int(mask_cross_provider.sum()),
        "TotalDuplicateGroups": total_groups,
        "TotalFlaggedClaims": total_flagged,
        "FlaggedPercentage": f"{(total_flagged / total_claims):.3%}",
        "ReportPath": str(report_path)
    }

    return flagged_df, summary_stats


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    print("#" * 75)
    print("AI-BASED HEALTHCARE INSURANCE FRAUD DETECTION")
    print("MODULE: CLAIM-LEVEL DUPLICATE DETECTION")
    print("#" * 75)
    print("NOTICE: This is an automated screening tool for fraud risk triage.")
    print("        A duplicate match is NOT proof of insurance fraud.")
    print("#" * 75)

    reports_dir = ensure_reports_dir()

    # 1. Automatically find raw files using glob patterns
    try:
        inpatient_file = find_raw_claims_file(RAW_DIR, "*Train*Inpatient*.csv")
        outpatient_file = find_raw_claims_file(RAW_DIR, "*Train*Outpatient*.csv")
    except FileNotFoundError as err:
        print(f"Error locating raw files: {err}", file=sys.stderr)
        sys.exit(1)

    # 2. Execute screening on Inpatient claims
    inp_flagged, inp_summary = screen_duplicate_claims(
        csv_path=inpatient_file,
        claim_type="Inpatient",
        output_dir=reports_dir
    )

    # 3. Execute screening on Outpatient claims
    out_flagged, out_summary = screen_duplicate_claims(
        csv_path=outpatient_file,
        claim_type="Outpatient",
        output_dir=reports_dir
    )

    # 4. Generate consolidated summary table
    summary_df = pd.DataFrame([inp_summary, out_summary])
    summary_csv_path = reports_dir / "duplicate_claims_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # 5. Print Concise ASCII Summary Table
    print("\n" + "=" * 75)
    print("DUPLICATE CLAIMS SCREENING SUMMARY")
    print("=" * 75)

    table_data = [
        ["Claim Type", inp_summary["ClaimType"], out_summary["ClaimType"]],
        ["Total Claims Evaluated", f"{inp_summary['TotalClaims']:,}", f"{out_summary['TotalClaims']:,}"],
        ["Exact Full Duplicates", f"{inp_summary['ExactFullDuplicates']:,}", f"{out_summary['ExactFullDuplicates']:,}"],
        ["Content Duplicates (excl. ClaimID)", f"{inp_summary['ExactContentDuplicates']:,}", f"{out_summary['ExactContentDuplicates']:,}"],
        ["Same-Provider Duplicates (Diag Match)", f"{inp_summary['SameProviderDiagMatchClaims']:,}", f"{out_summary['SameProviderDiagMatchClaims']:,}"],
        ["Same-Provider Duplicates (Other)", f"{inp_summary['SameProviderOtherClaims']:,}", f"{out_summary['SameProviderOtherClaims']:,}"],
        ["Cross-Provider Potential Duplicates", f"{inp_summary['CrossProviderClaims']:,}", f"{out_summary['CrossProviderClaims']:,}"],
        ["Total Potential Duplicate Groups", f"{inp_summary['TotalDuplicateGroups']:,}", f"{out_summary['TotalDuplicateGroups']:,}"],
        ["Total Flagged Claims", f"{inp_summary['TotalFlaggedClaims']:,}", f"{out_summary['TotalFlaggedClaims']:,}"],
        ["Flagged % of Dataset", inp_summary["FlaggedPercentage"], out_summary["FlaggedPercentage"]],
    ]

    col_w0 = 40
    col_w1 = 16
    col_w2 = 16

    print(f"{'Metric':<{col_w0}} | {'Inpatient':>{col_w1}} | {'Outpatient':>{col_w2}}")
    print("-" * (col_w0 + col_w1 + col_w2 + 6))
    for row in table_data[1:]:
        print(f"{row[0]:<{col_w0}} | {row[1]:>{col_w1}} | {row[2]:>{col_w2}}")
    print("=" * 75)

    print("\nGenerated Reports in 'reports/':")
    print(f"  1. {inp_summary['ReportPath']}")
    print(f"  2. {out_summary['ReportPath']}")
    print(f"  3. {summary_csv_path.resolve()}")
    print("\nSCREENING EXECUTION FINISHED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
