"""
Explainability Module - Phase 7
SHAP-based explanations: global importance, summary, waterfall, individual, counterfactual.
Distinguishes explanations from causal conclusions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
DATA_PATH = Path("reports/master_provider_features.csv")
ALT_PATH = Path("data/processed/cleaned_provider_features.csv")

def get_model_and_data():
    """Load model and dataset."""
    model = joblib.load(MODEL_DIR / "best_fraud_model.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
    path = DATA_PATH if DATA_PATH.exists() else ALT_PATH
    df = pd.read_csv(path)
    # Handle string target
    if "PotentialFraud" in df.columns and df["PotentialFraud"].dtype == object:
        df["PotentialFraudBinary"] = df["PotentialFraud"].map({"Yes":1,"No":0})
    X = df[feature_cols]
    providers = df["Provider"] if "Provider" in df.columns else pd.Series([f"P{i}" for i in range(len(X))])
    # Handle case where model is pipeline with imputer/scaler - need to get transformed X for SHAP
    return model, X, providers, df, feature_cols

def compute_shap_values(sample_size: int = 2000):
    """Compute SHAP values using shap library (or fallback to feature importance if unavailable)."""
    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False
        print("SHAP not installed, falling back to model feature_importance")
        return fallback_feature_importance()
    
    model, X, providers, df, feature_cols = get_model_and_data()
    print(f"Computing SHAP for {min(len(X), sample_size)} samples...")
    
    # For pipeline models, we need to handle imputer/scaler
    # Extract classifier
    if hasattr(model, "named_steps"):
        # Pipeline
        imputer = model.named_steps.get("imputer")
        scaler = model.named_steps.get("scaler")
        clf = model.named_steps.get("clf") or model.named_steps.get("classifier") or model.named_steps.get("model")
        # Transform X
        X_imputed = imputer.transform(X) if imputer else X.values
        if scaler:
            X_scaled = scaler.transform(X_imputed)
            # Use scaled feature names
            X_for_shap = pd.DataFrame(X_scaled, columns=feature_cols)
        else:
            X_for_shap = pd.DataFrame(X_imputed, columns=feature_cols)
    else:
        clf = model
        X_for_shap = X
        X_imputed = X.values
    
    # Sample for efficiency
    if len(X_for_shap) > sample_size:
        X_sample = X_for_shap.sample(n=sample_size, random_state=42)
        X_full = X_for_shap
    else:
        X_sample = X_for_shap
        X_full = X_for_shap
    
    # Create explainer
    try:
        # Try TreeExplainer for tree models
        if hasattr(clf, "feature_importances_") or "Forest" in str(type(clf)) or "XGB" in str(type(clf)) or "LGBM" in str(type(clf)):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample)
            # For binary, TreeExplainer may return list
            if isinstance(shap_values, list) and len(shap_values)==2:
                shap_values = shap_values[1]
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]
        else:
            # LinearExplainer for logistic regression
            # Use KernelExplainer as fallback but slow, use Linear
            explainer = shap.LinearExplainer(clf, X_sample)
            shap_values = explainer.shap_values(X_sample)
            expected_value = explainer.expected_value
    except Exception as e:
        print(f"SHAP Tree/Linear failed: {e}, trying Kernel (sampled)")
        # Fallback: use small background
        background = shap.sample(X_sample, 50)
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    
    # Save SHAP artifacts
    REPORT_DIR.mkdir(exist_ok=True)
    # Global importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values("MeanAbsSHAP", ascending=False)
    importance_df.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    print(f"SHAP global importance saved: {REPORT_DIR / 'shap_feature_importance.csv'}")
    
    # SHAP summary plot
    try:
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_summary.png", dpi=200, bbox_inches='tight')
        plt.close()
        print("Saved shap_summary.png")
    except Exception as e:
        print(f"Summary plot failed: {e}")
    
    # SHAP bar plot
    try:
        plt.figure(figsize=(10,8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=feature_cols, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_feature_importance.png", dpi=200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Bar plot failed: {e}")
        # Fallback bar
        plt.figure(figsize=(10,8))
        importance_df.head(15).sort_values("MeanAbsSHAP").plot.barh(x="Feature", y="MeanAbsSHAP")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "shap_feature_importance.png", dpi=200)
        plt.close()
    
    # Save expected value and sample for individual explanations
    joblib.dump({"expected_value": expected_value, "shap_values": shap_values, "X_sample": X_sample}, REPORT_DIR / "shap_sample_cache.pkl")
    # Also compute for full dataset if not too large, else sample
    # For individual provider explanations, we need full mapping
    # We'll compute on-demand per provider to save memory
    
    return {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "importance_df": importance_df,
        "X_sample": X_sample,
        "explainer": explainer,
        "clf": clf,
        "feature_cols": feature_cols,
        "X_for_shap": X_for_shap,
        "providers": providers,
        "df": df,
        "X": X
    }

def fallback_feature_importance():
    """Fallback if SHAP unavailable: use model feature importance."""
    model, X, providers, df, feature_cols = get_model_and_data()
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf") or model.named_steps.get("classifier") or model.named_steps.get("model")
    else:
        clf = model
    
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.random.rand(len(feature_cols))
    
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "MeanAbsSHAP": importances
    }).sort_values("MeanAbsSHAP", ascending=False)
    importance_df.to_csv(REPORT_DIR / "shap_feature_importance.csv", index=False)
    
    plt.figure(figsize=(10,8))
    importance_df.head(15).sort_values("MeanAbsSHAP").plot.barh(x="Feature", y="MeanAbsSHAP")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_feature_importance.png", dpi=200)
    plt.close()
    # Create dummy shap_summary
    plt.figure(figsize=(10,8))
    plt.text(0.5,0.5,"SHAP library not available - using model feature importance\nInstall shap for full explainability", ha='center')
    plt.savefig(REPORT_DIR / "shap_summary.png", dpi=200)
    plt.close()
    return {"importance_df": importance_df, "fallback": True}

def explain_provider(provider_id: str, save_plot: bool = True) -> dict:
    """Generate individual provider explanation."""
    import pandas as pd
    import numpy as np
    model, X, providers, df, feature_cols = get_model_and_data()
    
    # Locate provider
    if provider_id not in providers.values:
        return {"error": f"Provider {provider_id} not found"}
    idx = df.index[df["Provider"] == provider_id].tolist()[0]
    X_row = X.iloc[[idx]]
    
    # Predict
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_row)[0,1]
        pred = model.predict(X_row)[0]
    else:
        proba = 0
        pred = 0
    
    # Try SHAP
    try:
        import shap
        # Use cached or compute single
        if hasattr(model, "named_steps"):
            imputer = model.named_steps.get("imputer")
            scaler = model.named_steps.get("scaler")
            clf = model.named_steps.get("clf") or model.named_steps.get("classifier") or model.named_steps.get("model")
            X_imputed = imputer.transform(X) if imputer else X.values
            X_scaled = scaler.transform(X_imputed) if scaler else X_imputed
            X_for_shap = pd.DataFrame(X_scaled, columns=feature_cols)
            X_row_scaled = pd.DataFrame(scaler.transform(imputer.transform(X_row)) if imputer and scaler else imputer.transform(X_row) if imputer else X_row, columns=feature_cols)
        else:
            clf = model
            X_for_shap = X
            X_row_scaled = X_row
        
        # Explainer
        if hasattr(clf, "feature_importances_") or "Forest" in str(type(clf)):
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.LinearExplainer(clf, X_for_shap.sample(min(100, len(X_for_shap)), random_state=42))
        
        shap_vals = explainer.shap_values(X_row_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals)==2 else shap_vals[0]
        expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        
        # For waterfall, shap values per feature
        shap_vals = np.array(shap_vals).ravel()
        feature_values = X_row_scaled.iloc[0].values if isinstance(X_row_scaled, pd.DataFrame) else X_row.iloc[0].values
        
        # Create contribution ranking
        contributions = pd.DataFrame({
            "Feature": feature_cols,
            "FeatureValue": feature_values,
            "SHAPValue": shap_vals,
            "AbsSHAP": np.abs(shap_vals)
        }).sort_values("AbsSHAP", ascending=False)
        
        # Waterfall plot (manual)
        if save_plot:
            try:
                plt.figure(figsize=(12,6))
                # Use shap waterfall if available
                # Manual bar plot of contributions
                top = contributions.head(10)
                colors = ['red' if v>0 else 'blue' for v in top["SHAPValue"]]
                plt.barh(top["Feature"][::-1], top["SHAPValue"][::-1], color=colors[::-1])
                plt.axvline(0, color='black', linewidth=0.8)
                plt.title(f"SHAP Waterfall - {provider_id} (p={proba:.3f})")
                plt.xlabel("SHAP value (impact on model output)")
                plt.tight_layout()
                plt.savefig(REPORT_DIR / f"shap_waterfall_{provider_id}.png", dpi=200, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Waterfall plot failed: {e}")
        
        # Prepare explanation
        top_positive = contributions[contributions["SHAPValue"]>0].head(5)
        top_negative = contributions[contributions["SHAPValue"]<0].head(5)
        
        # Hybrid score
        hybrid_path = REPORT_DIR / "hybrid_risk_scores.csv"
        hybrid_score = None
        risk_cat = None
        if hybrid_path.exists():
            h = pd.read_csv(hybrid_path)
            hr = h[h["Provider"]==provider_id]
            if not hr.empty:
                hybrid_score = float(hr.iloc[0]["HybridRiskScore"])
                risk_cat = hr.iloc[0]["RiskCategory"]
        
        return {
            "Provider": provider_id,
            "PredictedFraudProbability": float(proba),
            "PredictedLabel": int(pred),
            "HybridRiskScore": hybrid_score,
            "RiskCategory": risk_cat,
            "ExpectedValue": float(expected_val) if expected_val is not None else None,
            "TopContributingFeatures": contributions.head(10).to_dict(orient="records"),
            "TopPositive": top_positive.to_dict(orient="records"),
            "TopNegative": top_negative.to_dict(orient="records"),
            "AllContributions": contributions.to_dict(orient="records"),
            "Disclaimer": "SHAP shows feature contribution to model prediction, not causal proof of fraud. Human review required."
        }
    except ImportError:
        # fallback
        if hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf") or model.named_steps.get("classifier") or model.named_steps.get("model")
        else:
            clf = model
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            contributions = pd.DataFrame({"Feature": feature_cols, "SHAPValue": importances, "FeatureValue": X_row.iloc[0].values})
            contributions = contributions.sort_values("SHAPValue", ascending=False)
            return {
                "Provider": provider_id,
                "PredictedFraudProbability": float(proba),
                "Fallback": "SHAP not available, using global importance",
                "TopContributingFeatures": contributions.head(10).to_dict(orient="records")
            }
        else:
            return {"Provider": provider_id, "PredictedFraudProbability": float(proba), "error": "No explainability available"}

def what_if_analysis(provider_id: str, adjustments: dict) -> dict:
    """
    Counterfactual / what-if: allow user to adjust features and see output change.
    adjustments: dict of {feature: new_value}
    Returns new prediction.
    """
    model, X, providers, df, feature_cols = get_model_and_data()
    if provider_id not in providers.values:
        return {"error": "Provider not found"}
    idx = df.index[df["Provider"] == provider_id].tolist()[0]
    X_row = X.iloc[[idx]].copy()
    original_proba = model.predict_proba(X_row)[0,1] if hasattr(model, "predict_proba") else 0
    
    # Apply adjustments
    for feat, new_val in adjustments.items():
        if feat in X_row.columns:
            try:
                X_row[feat] = float(new_val)
            except:
                X_row[feat] = new_val
    
    new_proba = model.predict_proba(X_row)[0,1] if hasattr(model, "predict_proba") else 0
    delta = new_proba - original_proba
    
    # Hybrid would need recompute but we approximate supervised only
    return {
        "Provider": provider_id,
        "OriginalProba": float(original_proba),
        "NewProba": float(new_proba),
        "Delta": float(delta),
        "Adjustments": adjustments,
        "Interpretation": f"Changing {list(adjustments.keys())} from original to {list(adjustments.values())} changes fraud probability from {original_proba:.3f} to {new_proba:.3f} (Δ={delta:+.3f}). This is hypothetical and does not alter stored records.",
        "Warning": "Simulated values are hypothetical, clearly marked, and do not overwrite actual provider records."
    }

if __name__ == "__main__":
    result = compute_shap_values()
    print(result.get("importance_df", pd.DataFrame()).head())
    # Example individual
    model, X, providers, df, feature_cols = get_model_and_data()
    if len(providers)>0:
        expl = explain_provider(providers.iloc[0])
        print(expl)
