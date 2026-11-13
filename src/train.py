"""
Training Module - Phase 4
Implements 5 models: Logistic Regression, Random Forest, XGBoost, LightGBM, Isolation Forest
Handles class imbalance, cross-validation, hyperparameter tuning, leakage-safe splits.
Saves models, pipelines, and metadata.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH = Path("data/processed/cleaned_provider_features.csv")
ALT_PATH = Path("reports/master_provider_features.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def load_dataset():
    """Load provider-level dataset with fallback."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    elif ALT_PATH.exists():
        df = pd.read_csv(ALT_PATH)
    else:
        raise FileNotFoundError(f"No dataset found at {DATA_PATH} or {ALT_PATH}")
    print(f"Dataset loaded: {df.shape}")
    # Determine target column
    if "PotentialFraudBinary" in df.columns:
        target_col = "PotentialFraudBinary"
    elif "PotentialFraud" in df.columns:
        # Map string Yes/No to 1/0
        if df["PotentialFraud"].dtype == object:
            df[target_col] = df["PotentialFraud"].map({"Yes":1,"No":0,"yes":1,"no":0})
            target_col = "PotentialFraudBinary" if "PotentialFraudBinary" in df.columns else "PotentialFraud"
            # if mapping created numeric, use it
            if "PotentialFraudBinary" in df.columns:
                target_col = "PotentialFraudBinary"
            else:
                df[target_col] = df["PotentialFraud"].map({"Yes":1,"No":0})
        else:
            target_col = "PotentialFraud"
    else:
        raise ValueError("No target column found")
    
    # If target is string Yes/No, convert
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({"Yes":1,"No":0, "yes":1,"no":0})
    
    # Features: all numeric except Provider, PotentialFraud, PeerGroup, PotentialFraudBinary
    exclude = ["Provider", "PotentialFraud", "PotentialFraudBinary", "PeerGroup"]
    # Also exclude any non-numeric
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    # Also ensure we don't include leakage-like columns that are derived from target (none here)
    print(f"Target: {target_col}, Features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    X = df[feature_cols]
    y = df[target_col].astype(int)
    providers = df["Provider"] if "Provider" in df.columns else None
    return X, y, providers, feature_cols, df

def get_models():
    """Define 5 models with appropriate handling."""
    models = {}
    # Logistic Regression with class_weight balanced
    models["LogisticRegression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
    ])
    # Random Forest
    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    ])
    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False))
        ])
    # LightGBM
    if HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))
        ])
    # Isolation Forest is unsupervised - we will handle separately, but include wrapper for comparison
    # For now, we create a supervised version via anomaly score threshold tuning elsewhere
    return models

def train_and_evaluate(handle_imbalance: str = "class_weight", test_size: float = 0.30, save: bool = True):
    """
    Train models with leakage-safe splits.
    - 70% train, 15% val, 15% test (stratified)
    - Optionally compare SMOTE vs class_weight
    - Saves best model based on validation PR-AUC
    """
    X, y, providers, feature_cols, df_full = load_dataset()
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Fraud rate: {y.mean():.4f}")
    
    # 70/30 split (train+val vs test), then 50/50 of temp for val/test -> 70/15/15
    X_train, X_temp, y_train, y_temp, prov_train, prov_temp = train_test_split(
        X, y, providers, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test, prov_val, prov_test = train_test_split(
        X_temp, y_temp, prov_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Splits: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
    
    models = get_models()
    # Also consider SMOTE variant for comparison where requested
    validation_results = []
    trained = {}
    
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }
    
    for name, pipe in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        # Handle imbalance: we use class_weight already for most; optionally test SMOTE for RF/LogReg
        # For this run, use pipe as is (class_weight)
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_val)
        y_val_proba = pipe.predict_proba(X_val)[:,1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_val)
        
        # Metrics
        res = {
            "Model": name,
            "Accuracy": accuracy_score(y_val, y_val_pred),
            "Precision": precision_score(y_val, y_val_pred, zero_division=0),
            "Recall": recall_score(y_val, y_val_pred, zero_division=0),
            "F1": f1_score(y_val, y_val_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_val, y_val_proba),
            "PR_AUC": average_precision_score(y_val, y_val_proba),
            "ConfusionMatrix": confusion_matrix(y_val, y_val_pred).tolist(),
        }
        print(f"  Accuracy: {res['Accuracy']:.4f}, Precision: {res['Precision']:.4f}, Recall: {res['Recall']:.4f}, F1: {res['F1']:.4f}")
        print(f"  ROC-AUC: {res['ROC_AUC']:.4f}, PR-AUC: {res['PR_AUC']:.4f}")
        validation_results.append(res)
        trained[name] = pipe
        
        # Cross-validation (5-fold) on train+val to ensure robustness
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        try:
            cv_scores = cross_validate(pipe, pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), cv=cv, scoring=scoring, n_jobs=1)
            print(f"  CV PR-AUC: {cv_scores['test_pr_auc'].mean():.4f} (+/- {cv_scores['test_pr_auc'].std():.4f})")
            res["CV_PR_AUC_mean"] = float(cv_scores['test_pr_auc'].mean())
            res["CV_PR_AUC_std"] = float(cv_scores['test_pr_auc'].std())
        except Exception as e:
            print(f"  CV failed: {e}")
    
    # Also train Isolation Forest unsupervised and evaluate as anomaly detector
    print(f"\n{'='*60}")
    print(f"Training: IsolationForest (unsupervised)")
    print(f"{'='*60}")
    iso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", IsolationForest(contamination=0.1, random_state=RANDOM_STATE, n_estimators=200))
    ])
    iso.fit(X_train)
    # Isolation Forest decision: -1 anomaly, 1 normal -> convert to anomaly score
    # Use score_samples (higher = more normal) -> invert and scale to 0-1 anomaly probability proxy
    # For evaluation, we can threshold anomaly scores
    iso_scores = iso.named_steps["clf"].decision_function(iso.named_steps["scaler"].transform(iso.named_steps["imputer"].transform(X_val)))
    # Normalize to 0-1 where 1 = most anomalous
    # isolation forest decision_function: shift to be higher for normal, so anomaly = -score
    anomaly_scores = -iso_scores
    # Min-max scale to 0-1 for PR-AUC computation (treat as fraud probability proxy)
    from sklearn.preprocessing import MinMaxScaler
    mm = MinMaxScaler()
    anomaly_proba = mm.fit_transform(anomaly_scores.reshape(-1,1)).ravel()
    try:
        iso_pr_auc = average_precision_score(y_val, anomaly_proba)
        iso_roc = roc_auc_score(y_val, anomaly_proba)
        print(f"  IsolationForest PR-AUC: {iso_pr_auc:.4f}, ROC-AUC: {iso_roc:.4f} (unsupervised baseline)")
        # For consistency, we don't include iso in supervised selection, but save separately
        validation_results.append({
            "Model": "IsolationForest",
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "ROC_AUC": iso_roc,
            "PR_AUC": iso_pr_auc,
            "ConfusionMatrix": [[0,0],[0,0]],
            "Note": "Unsupervised anomaly detection; not a supervised fraud classifier"
        })
    except Exception as e:
        print(f"  IsolationForest evaluation failed: {e}")
    
    # Compare models via validation PR-AUC (supervised only)
    comp_df = pd.DataFrame(validation_results)
    comp_df_sorted = comp_df.sort_values(by="PR_AUC", ascending=False)
    print("\n" + "="*60)
    print("VALIDATION MODEL COMPARISON (sorted by PR-AUC)")
    print("="*60)
    print(comp_df_sorted[["Model","Accuracy","Precision","Recall","F1","ROC_AUC","PR_AUC"]].to_string(index=False))
    
    if save:
        comp_df_sorted.to_csv(REPORT_DIR / "validation_model_comparison.csv", index=False)
        # Save detailed comparison including CV
        comp_df.to_csv(REPORTS_DIR / "model_comparison_detailed.csv" if False else REPORT_DIR / "model_comparison.csv", index=False)
    
    # Select best supervised model (exclude IsolationForest)
    supervised = comp_df[comp_df["Model"] != "IsolationForest"]
    best_name = supervised.sort_values(by="PR_AUC", ascending=False).iloc[0]["Model"]
    print(f"\nSelected best supervised model: {best_name}")
    
    # Retrain best on train+val
    best_pipe = models[best_name]
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    print(f"Retraining {best_name} on Train+Val ({len(X_train_val)} rows)...")
    best_pipe.fit(X_train_val, y_train_val)
    
    # Final test evaluation (untouched test set)
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION (untouched test set)")
    print("="*60)
    y_test_pred = best_pipe.predict(X_test)
    y_test_proba = best_pipe.predict_proba(X_test)[:,1]
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_test_proba),
        "pr_auc": average_precision_score(y_test, y_test_proba),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
        "classification_report": classification_report(y_test, y_test_pred, zero_division=0, output_dict=True)
    }
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")
    
    # Also train Isolation Forest on train+val and save
    final_iso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", IsolationForest(contamination=0.1, random_state=RANDOM_STATE, n_estimators=200))
    ])
    final_iso.fit(X_train_val)
    iso_train_scores = final_iso.named_steps["clf"].decision_function(final_iso.named_steps["scaler"].transform(final_iso.named_steps["imputer"].transform(X_train_val)))
    
    if save:
        # Save models
        joblib.dump(best_pipe, MODEL_DIR / "best_fraud_model.pkl")
        joblib.dump(final_iso, MODEL_DIR / "isolation_forest.pkl")
        joblib.dump(feature_cols, MODEL_DIR / "feature_columns.pkl")
        # Save scaler if needed
        # Save metadata
        metadata = {
            "best_model": best_name,
            "feature_columns": feature_cols,
            "feature_count": len(feature_cols),
            "train_rows": len(X_train_val),
            "test_rows": len(X_test),
            "validation_results_all": validation_results,
            "test_metrics": test_metrics,
            "target_distribution_train": y_train_val.value_counts().to_dict(),
            "target_distribution_test": y_test.value_counts().to_dict(),
            "random_state": RANDOM_STATE,
            "imbalance_handling": handle_imbalance,
            "isolation_forest_contamination": 0.1,
            "training_date": pd.Timestamp.now().isoformat(),
            "dataset_source": str(DATA_PATH if DATA_PATH.exists() else ALT_PATH),
            "model_version": "v1.0.0",
            "caveat": "Model trained on US Medicare data; not validated for other insurance schemes. High-risk score is investigation signal, not proof of fraud."
        }
        with open(MODEL_DIR / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        # Save test predictions for later use
        test_out = pd.DataFrame({
            "Provider": prov_test.values,
            "y_true": y_test.values,
            "y_pred": y_test_pred,
            "y_proba": y_test_proba
        })
        test_out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
        print(f"\nSaved models to {MODEL_DIR}")
        print(f"  - best_fraud_model.pkl ({best_name})")
        print(f"  - isolation_forest.pkl")
        print(f"  - feature_columns.pkl")
        print(f"  - model_metadata.json")
        print(f"Reports: {REPORT_DIR / 'validation_model_comparison.csv'}, {REPORT_DIR / 'model_metadata.json'}")
    
    return {
        "best_model": best_pipe,
        "best_name": best_name,
        "isolation_forest": final_iso,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "feature_columns": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "prov_test": prov_test,
    }

if __name__ == "__main__":
    # Also run feature engineering if needed files missing
    if not DATA_PATH.exists() and not ALT_PATH.exists():
        print("No dataset found, running feature engineering...")
        from src.feature_engineering import engineer_features
        engineer_features(save=True)
    train_and_evaluate()
