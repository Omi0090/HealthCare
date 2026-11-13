"""
Anomaly & Temporal Detection Modules - Phase 8A, 8B, 8D
- Duplicate / near-duplicate detection (already has script but expose as importable)
- Temporal anomaly detection (monthly claim volume, reimbursement spikes)
- Peer-group deviation (robust MAD)
- Isolation Forest wrapper
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Tuple

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")

def load_raw_claims_for_anomaly():
    """Load claims for temporal and duplicate analysis."""
    def find(pattern): 
        m = sorted(RAW_DIR.glob(pattern))
        return m[0] if m else None
    inpatient = find("Train_Inpatientdata-*.csv")
    outpatient = find("Train_Outpatientdata-*.csv")
    import pandas as pd
    dfs = []
    if inpatient:
        df = pd.read_csv(inpatient, low_memory=False, usecols=lambda c: c in ["Provider","BeneID","ClaimID","ClaimStartDt","ClaimEndDt","InscClaimAmtReimbursed","DeductibleAmtPaid"])
        df["ClaimType"] = "Inpatient"
        dfs.append(df)
    if outpatient:
        df = pd.read_csv(outpatient, low_memory=False, usecols=lambda c: c in ["Provider","BeneID","ClaimID","ClaimStartDt","ClaimEndDt","InscClaimAmtReimbursed","DeductibleAmtPaid"])
        df["ClaimType"] = "Outpatient"
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    claims = pd.concat(dfs, ignore_index=True)
    claims["ClaimStartDt"] = pd.to_datetime(claims["ClaimStartDt"], errors='coerce')
    claims["ClaimEndDt"] = pd.to_datetime(claims["ClaimEndDt"], errors='coerce')
    return claims

def temporal_anomaly_detection(claims: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze monthly claim volume/reimbursement per provider.
    Flags sudden increases using rolling z-scores and peer comparison.
    Returns: provider_temporal_features, anomaly_alerts
    """
    if claims is None:
        claims = load_raw_claims_for_anomaly()
    if claims.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    claims["YearMonth"] = claims["ClaimStartDt"].dt.to_period("M").astype(str)
    # Monthly per provider
    monthly = claims.groupby(["Provider","YearMonth"]).agg(
        MonthlyClaims=("ClaimID","count"),
        MonthlyReimbursement=("InscClaimAmtReimbursed","sum"),
        AvgClaimAmount=("InscClaimAmtReimbursed","mean")
    ).reset_index()
    
    # Sort for rolling
    monthly = monthly.sort_values(["Provider","YearMonth"])
    # Provider-level temporal features
    # Compute rolling mean/std (3-month window) per provider
    alerts = []
    provider_features = []
    
    for provider, group in monthly.groupby("Provider"):
        group = group.sort_values("YearMonth")
        # Basic stats
        avg_monthly = group["MonthlyClaims"].mean()
        std_monthly = group["MonthlyClaims"].std()
        max_monthly = group["MonthlyClaims"].max()
        avg_reimb = group["MonthlyReimbursement"].mean()
        max_reimb = group["MonthlyReimbursement"].max()
        # Detect spikes: monthly count > mean + 2*std
        if std_monthly and std_monthly > 0:
            group["Claim_z"] = (group["MonthlyClaims"] - avg_monthly) / std_monthly
            group["Reimb_z"] = (group["MonthlyReimbursement"] - avg_reimb) / (group["MonthlyReimbursement"].std() if group["MonthlyReimbursement"].std()>0 else 1)
            spikes = group[group["Claim_z"] > 2.0]
            for _, row in spikes.iterrows():
                alerts.append({
                    "Provider": provider,
                    "YearMonth": row["YearMonth"],
                    "Metric": "ClaimVolumeSpike",
                    "Value": row["MonthlyClaims"],
                    "ZScore": row["Claim_z"],
                    "ExpectedMean": avg_monthly,
                    "Context": f"Claims {row['MonthlyClaims']} vs avg {avg_monthly:.1f} (z={row['Claim_z']:.2f})",
                    "Severity": "High" if row["Claim_z"] > 3 else "Medium"
                })
        # Trend: increasing over time? Compare first half vs second half mean
        if len(group) >= 4:
            mid = len(group)//2
            first_half = group.iloc[:mid]["MonthlyClaims"].mean()
            second_half = group.iloc[mid:]["MonthlyClaims"].mean()
            trend_ratio = second_half / max(first_half,1)
            # flag sharp increase (>2x)
            if trend_ratio > 2.0:
                alerts.append({
                    "Provider": provider,
                    "YearMonth": group.iloc[-1]["YearMonth"],
                    "Metric": "IncreasingTrend",
                    "Value": second_half,
                    "ZScore": trend_ratio,
                    "ExpectedMean": first_half,
                    "Context": f"Recent avg {second_half:.1f} is {trend_ratio:.1f}x earlier avg {first_half:.1f}",
                    "Severity": "High" if trend_ratio>3 else "Medium"
                })
        provider_features.append({
            "Provider": provider,
            "AvgMonthlyClaims": avg_monthly,
            "MaxMonthlyClaims": max_monthly,
            "StdMonthlyClaims": std_monthly if not np.isnan(std_monthly) else 0,
            "AvgMonthlyReimbursement": avg_reimb,
            "MaxMonthlyReimbursement": max_reimb,
            "NumActiveMonths": len(group),
            "ClaimVolatility": std_monthly/avg_monthly if avg_monthly>0 else 0,
            "TrendRatio": trend_ratio if len(group)>=4 else 1.0
        })
    
    prov_temporal = pd.DataFrame(provider_features)
    alerts_df = pd.DataFrame(alerts)
    
    # Save
    REPORT_DIR.mkdir(exist_ok=True)
    if not prov_temporal.empty:
        prov_temporal.to_csv(REPORT_DIR / "temporal_provider_features.csv", index=False)
    if not alerts_df.empty:
        alerts_df.to_csv(REPORT_DIR / "temporal_anomaly_alerts.csv", index=False)
    else:
        # create empty with header
        pd.DataFrame(columns=["Provider","YearMonth","Metric","Value","ZScore","ExpectedMean","Context","Severity"]).to_csv(REPORT_DIR / "temporal_anomaly_alerts.csv", index=False)
    
    print(f"Temporal analysis: {len(prov_temporal)} providers, {len(alerts_df)} alerts")
    return prov_temporal, alerts_df

def peer_group_analysis(master_path: Path = Path("reports/master_provider_features.csv")) -> pd.DataFrame:
    """
    Provider peer-group analysis using already engineered PeerGroup.
    Identifies unusually high reimbursement, frequency, etc.
    """
    if not master_path.exists():
        # Try processed
        master_path = Path("data/processed/cleaned_provider_features.csv")
    if not master_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(master_path)
    # Peer deviations already computed if engineer_features ran; recompute if needed
    if "PeerGroup" not in df.columns:
        df["PeerGroup"] = pd.qcut(df["TotalClaims"], q=5, labels=["Very Low","Low","Medium","High","Very High"], duplicates='drop').astype(str)
    
    # Determine suspicious deviations: |peer_z| > 2.5
    peer_z_cols = [c for c in df.columns if c.endswith("_PeerZ")]
    alerts = []
    for _, row in df.iterrows():
        provider = row["Provider"]
        peer = row["PeerGroup"]
        for col in peer_z_cols:
            z = row[col]
            if abs(z) > 2.5:
                feat = col.replace("_PeerZ","")
                alerts.append({
                    "Provider": provider,
                    "PeerGroup": peer,
                    "Feature": feat,
                    "Value": row[feat],
                    "PeerMedian": row.get(feat+"_PeerMedian", np.nan),
                    "ZScore": z,
                    "Deviation": "High" if z>0 else "Low",
                    "Severity": "Critical" if abs(z)>3.5 else "High" if abs(z)>3 else "Medium",
                    "Context": f"{feat}={row[feat]:.2f} vs peer median {row.get(feat+'_PeerMedian',0):.2f} (z={z:.2f})"
                })
    alerts_df = pd.DataFrame(alerts)
    if not alerts_df.empty:
        REPORT_DIR.mkdir(exist_ok=True)
        alerts_df.to_csv(REPORT_DIR / "peer_group_alerts.csv", index=False)
        # Summary per provider
        summary = alerts_df.groupby("Provider").agg(
            NumPeerDeviations=("Feature","count"),
            MaxAbsZ=("ZScore", lambda x: max(abs(x))),
            Features=("Feature", lambda x: ", ".join(x.unique()))
        ).reset_index()
        summary.to_csv(REPORT_DIR / "peer_group_deviation_summary.csv", index=False)
        print(f"Peer-group analysis: {len(alerts_df)} deviations across {summary['Provider'].nunique()} providers")
    else:
        print("Peer-group analysis: no significant deviations found (|z|<=2.5)")
        pd.DataFrame(columns=["Provider","PeerGroup","Feature","Value","PeerMedian","ZScore","Deviation","Severity","Context"]).to_csv(REPORT_DIR / "peer_group_alerts.csv", index=False)
    return alerts_df

def isolation_forest_scoring(X: pd.DataFrame = None, contamination: float = 0.1) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Train Isolation Forest and compute anomaly scores.
    Returns model and dataframe with scores.
    """
    if X is None:
        # Load master
        path = Path("reports/master_provider_features.csv")
        if not path.exists():
            path = Path("data/processed/cleaned_provider_features.csv")
        df = pd.read_csv(path)
        feature_cols = [c for c in df.columns if c not in ["Provider","PotentialFraud","PotentialFraudBinary","PeerGroup"] and pd.api.types.is_numeric_dtype(df[c])]
        X = df[feature_cols]
        providers = df["Provider"] if "Provider" in df.columns else pd.Series([f"P{i}" for i in range(len(X))])
    else:
        # X is already provided, need providers
        providers = pd.Series([f"P{i}" for i in range(len(X))])
        feature_cols = X.columns.tolist()
    
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", IsolationForest(contamination=contamination, random_state=42, n_estimators=200))
    ])
    pipe.fit(X)
    # Scores
    imputed = pipe.named_steps["imputer"].transform(X)
    scaled = pipe.named_steps["scaler"].transform(imputed)
    raw_scores = pipe.named_steps["clf"].decision_function(scaled)
    # Convert to anomaly score 0-100 where 100 = most anomalous
    # decision_function higher = more normal, so anomaly = -score
    anomaly_raw = -raw_scores
    # Normalize to 0-1 then 0-100
    mm = MinMaxScaler()
    anomaly_score = mm.fit_transform(anomaly_raw.reshape(-1,1)).ravel() * 100
    # Binary prediction: -1 anomaly, 1 normal
    preds = pipe.named_steps["clf"].predict(scaled)  # -1 anomaly
    is_anomaly = (preds == -1).astype(int)
    
    result = pd.DataFrame({
        "Provider": providers.values if hasattr(providers, "values") else providers,
        "AnomalyScore": anomaly_score,
        "IsAnomaly": is_anomaly,
        "RawAnomaly": anomaly_raw
    })
    result = result.sort_values("AnomalyScore", ascending=False)
    REPORT_DIR.mkdir(exist_ok=True)
    result.to_csv(REPORT_DIR / "isolation_forest_scores.csv", index=False)
    print(f"Isolation Forest: {is_anomaly.sum()} anomalies flagged out of {len(result)}")
    return pipe, result

def duplicate_detection_summary() -> pd.DataFrame:
    """Wrap existing duplicate detection script to produce provider-level summary."""
    # Check if reports already exist from scripts/detect_duplicate_claims.py
    inpatient_report = REPORT_DIR / "inpatient_duplicate_claims_report.csv"
    outpatient_report = REPORT_DIR / "outpatient_duplicate_claims_report.csv"
    summary = REPORT_DIR / "duplicate_claims_summary.csv"
    provider_level = []
    
    # If reports exist, aggregate to provider
    for path in [inpatient_report, outpatient_report]:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            if "Provider" in df.columns:
                counts = df.groupby("Provider").size().reset_index(name="DuplicateFlaggedClaims")
                for _, row in counts.iterrows():
                    provider_level.append({
                        "Provider": row["Provider"],
                        "Source": path.stem,
                        "DuplicateFlaggedClaims": row["DuplicateFlaggedClaims"]
                    })
    if provider_level:
        prov_df = pd.DataFrame(provider_level)
        # Pivot to have inpatient/outpatient separately
        pivot = prov_df.pivot_table(index="Provider", columns="Source", values="DuplicateFlaggedClaims", aggfunc="sum", fill_value=0).reset_index()
        pivot.to_csv(REPORT_DIR / "duplicate_provider_summary.csv", index=False)
        print(f"Duplicate provider summary: {len(pivot)} providers with flagged claims")
        return pivot
    else:
        print("No duplicate reports found; run scripts/detect_duplicate_claims.py to generate them")
        return pd.DataFrame()

if __name__ == "__main__":
    temporal_anomaly_detection()
    peer_group_analysis()
    isolation_forest_scoring()
    duplicate_detection_summary()
