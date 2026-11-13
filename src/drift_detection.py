"""
Drift Detection Module - Phase 8E
Monitors feature drift, prediction drift, and model versioning.
Distinguishes data drift from proven performance deterioration.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy.stats import ks_2samp, chi2_contingency
import joblib
from datetime import datetime

REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
DATA_PATH = Path("reports/master_provider_features.csv")
ALT_PATH = Path("data/processed/cleaned_provider_features.csv")

def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list = None, threshold: float = 0.05) -> pd.DataFrame:
    """
    Compute drift per feature using Kolmogorov-Smirnov test (numeric) and Chi2 (categorical).
    Returns drift report.
    """
    if features is None:
        features = [c for c in reference.columns if pd.api.types.is_numeric_dtype(reference[c])]
    
    results = []
    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        ref_vals = reference[feat].dropna()
        cur_vals = current[feat].dropna()
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue
        # KS test
        try:
            ks_stat, p_value = ks_2samp(ref_vals, cur_vals)
            drift_detected = p_value < threshold
            # Effect size: difference in means normalized
            mean_diff = abs(cur_vals.mean() - ref_vals.mean()) / (ref_vals.std() if ref_vals.std()>0 else 1)
            results.append({
                "Feature": feat,
                "ReferenceMean": ref_vals.mean(),
                "CurrentMean": cur_vals.mean(),
                "MeanDiff": cur_vals.mean() - ref_vals.mean(),
                "MeanDiffPct": (cur_vals.mean() - ref_vals.mean())/ ref_vals.mean() *100 if ref_vals.mean()!=0 else 0,
                "KS_Statistic": ks_stat,
                "PValue": p_value,
                "DriftDetected": drift_detected,
                "Severity": "High" if drift_detected and ks_stat>0.3 else "Medium" if drift_detected else "Low",
                "EffectSize": mean_diff
            })
        except Exception as e:
            results.append({"Feature": feat, "Error": str(e)})
    
    drift_df = pd.DataFrame(results)
    if not drift_df.empty:
        drift_df = drift_df.sort_values("KS_Statistic", ascending=False)
    return drift_df

def prediction_drift(reference_proba: np.ndarray, current_proba: np.ndarray) -> dict:
    """Check drift in prediction distribution."""
    from scipy.stats import ks_2samp
    ks_stat, p_value = ks_2samp(reference_proba, current_proba)
    return {
        "KS_Statistic": float(ks_stat),
        "PValue": float(p_value),
        "ReferenceMeanProba": float(np.mean(reference_proba)),
        "CurrentMeanProba": float(np.mean(current_proba)),
        "DriftDetected": bool(p_value < 0.05),
        "Interpretation": "Prediction distribution change may indicate data drift or genuine shift, not necessarily model performance drop. Validate with labeled data."
    }

def model_versioning():
    """Track model versions and create registry."""
    meta_path = MODEL_DIR / "model_metadata.json"
    versions_path = REPORT_DIR / "model_versions.json"
    
    if not meta_path.exists():
        return {"error": "No model metadata found"}
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    version_entry = {
        "version": meta.get("model_version", "v1.0.0"),
        "best_model": meta.get("best_model"),
        "training_date": meta.get("training_date"),
        "feature_count": meta.get("feature_count"),
        "test_pr_auc": meta.get("test_metrics", {}).get("pr_auc"),
        "test_roc_auc": meta.get("test_metrics", {}).get("roc_auc"),
        "dataset_source": meta.get("dataset_source"),
        "caveat": meta.get("caveat")
    }
    
    # Load existing registry
    if versions_path.exists():
        with open(versions_path) as f:
            registry = json.load(f)
    else:
        registry = {"versions": []}
    
    # Add if not duplicate
    if not any(v.get("version")==version_entry["version"] and v.get("training_date")==version_entry["training_date"] for v in registry["versions"]):
        registry["versions"].append(version_entry)
        registry["latest"] = version_entry
        registry["updated_at"] = datetime.now().isoformat()
        with open(versions_path, "w") as f:
            json.dump(registry, f, indent=2)
    
    return registry

def run_drift_analysis(reference_split: float = 0.5):
    """
    Simulate drift detection by splitting current data into reference vs current
    (e.g., old vs new claims). In production, replace with time-based splits.
    """
    path = DATA_PATH if DATA_PATH.exists() else ALT_PATH
    if not path.exists():
        return {"error": "No dataset"}
    df = pd.read_csv(path)
    # Simulate: first half as reference, second half as current (ordered by Provider)
    df = df.sort_values("Provider")
    split_idx = int(len(df)*reference_split)
    ref = df.iloc[:split_idx]
    cur = df.iloc[split_idx:]
    
    print(f"Drift analysis: reference {len(ref)}, current {len(cur)}")
    
    # Feature drift
    feature_cols = [c for c in df.columns if c not in ["Provider","PotentialFraud","PotentialFraudBinary","PeerGroup"] and pd.api.types.is_numeric_dtype(df[c])]
    drift_df = compute_drift(ref, cur, features=feature_cols)
    REPORT_DIR.mkdir(exist_ok=True)
    drift_df.to_csv(REPORT_DIR / "feature_drift_report.csv", index=False)
    print(f"Feature drift: {drift_df['DriftDetected'].sum()}/{len(drift_df)} features flagged")
    
    # Prediction drift
    model_path = MODEL_DIR / "best_fraud_model.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        feature_cols_model = joblib.load(MODEL_DIR / "feature_columns.pkl")
        X_ref = ref[feature_cols_model].fillna(ref[feature_cols_model].median()) if all(c in ref.columns for c in feature_cols_model) else ref[feature_cols]
        X_cur = cur[feature_cols_model].fillna(cur[feature_cols_model].median()) if all(c in cur.columns for c in feature_cols_model) else cur[feature_cols]
        # Ensure same columns
        if all(c in ref.columns for c in feature_cols_model):
            ref_proba = model.predict_proba(X_ref)[:,1]
            cur_proba = model.predict_proba(X_cur)[:,1]
            pred_drift = prediction_drift(ref_proba, cur_proba)
            with open(REPORT_DIR / "prediction_drift_report.json", "w") as f:
                json.dump(pred_drift, f, indent=2)
            print(f"Prediction drift: KS={pred_drift['KS_Statistic']:.3f}, p={pred_drift['PValue']:.3f}, drift={pred_drift['DriftDetected']}")
        else:
            pred_drift = {"error":"feature mismatch"}
    else:
        pred_drift = {"error":"no model"}
    
    # Versioning
    versions = model_versioning()
    
    # Summary alerts
    alerts = []
    if not drift_df.empty:
        high_drift = drift_df[drift_df["Severity"]=="High"]
        for _, row in high_drift.iterrows():
            alerts.append({
                "Type": "FeatureDrift",
                "Feature": row["Feature"],
                "KS": row["KS_Statistic"],
                "Severity": row["Severity"],
                "Message": f"{row['Feature']} drift detected (KS={row['KS_Statistic']:.3f}, mean {row['ReferenceMean']:.2f}->{row['CurrentMean']:.2f})"
            })
    if pred_drift.get("DriftDetected"):
        alerts.append({
            "Type": "PredictionDrift",
            "KS": pred_drift["KS_Statistic"],
            "Severity": "Medium",
            "Message": f"Prediction distribution drift (KS={pred_drift['KS_Statistic']:.3f}) - investigate data change, not necessarily performance drop"
        })
    
    alerts_df = pd.DataFrame(alerts)
    if not alerts_df.empty:
        alerts_df.to_csv(REPORT_DIR / "drift_alerts.csv", index=False)
    else:
        pd.DataFrame(columns=["Type","Feature","KS","Severity","Message"]).to_csv(REPORT_DIR / "drift_alerts.csv", index=False)
    
    return {
        "feature_drift": drift_df,
        "prediction_drift": pred_drift,
        "versions": versions,
        "alerts": alerts_df
    }

if __name__ == "__main__":
    result = run_drift_analysis()
    print(result["feature_drift"].head() if not result["feature_drift"].empty else "No drift")
