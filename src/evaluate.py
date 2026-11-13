"""
Evaluation Module - Phase 6
Comprehensive metrics, ROC/PR curves, threshold optimization, confusion matrices.
Generates reproducible evaluation reports.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)
import joblib

REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
REPORT_DIR.mkdir(exist_ok=True)

def threshold_optimization(y_true, y_proba, thresholds=np.arange(0.1, 0.9, 0.05)):
    """Find optimal threshold maximizing F1 and examine precision-recall tradeoff."""
    results = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        results.append({
            "Threshold": round(float(thr),2),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_true, y_pred),
        })
    df = pd.DataFrame(results)
    # Best F1
    best_f1_row = df.loc[df["F1"].idxmax()]
    # Best balanced (closest to 0.5 precision-recall balance) - just report
    return df, best_f1_row

def plot_roc_curve(y_true, y_proba, model_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return auc

def plot_pr_curve(y_true, y_proba, model_name, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(7,6))
    plt.plot(recall, precision, label=f"{model_name} (PR-AUC={pr_auc:.3f})", linewidth=2)
    # Baseline (random)
    baseline = y_true.mean()
    plt.axhline(baseline, linestyle='--', color='gray', label=f"Baseline (fraud rate={baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return pr_auc

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xticks([0.5,1.5], ["Not Fraud","Fraud"])
    plt.yticks([0.5,1.5], ["Not Fraud","Fraud"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return cm

def evaluate_all_models():
    """Evaluate best model on test set and generate reports."""
    # Load metadata and test predictions if available
    meta_path = MODEL_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Loaded metadata: best model {meta.get('best_model')}")
    else:
        meta = {}
    
    # Try to load test predictions
    test_pred_path = REPORT_DIR / "test_predictions.csv"
    if test_pred_path.exists():
        df = pd.read_csv(test_pred_path)
        y_true = df["y_true"]
        y_proba = df["y_proba"]
        y_pred = df["y_pred"]
        model_name = meta.get("best_model", "BestModel")
        print(f"Evaluating {model_name} on saved test predictions: {len(df)} rows")
    else:
        # Re-run inference by loading model and dataset
        from src.train import load_dataset
        X, y, providers, feature_cols, df_full = load_dataset()
        model = joblib.load(MODEL_DIR / "best_fraud_model.pkl")
        # Need to recreate splits to get test set - use train.py logic to regenerate
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        y_true = y_test
        y_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)
        model_name = meta.get("best_model", "BestModel")
    
    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
    }
    cm = confusion_matrix(y_true, y_pred)
    print("\n=== Evaluation Metrics ===")
    for k,v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Plots
    plot_roc_curve(y_true, y_proba, model_name, REPORT_DIR / "roc_curve.png")
    plot_pr_curve(y_true, y_proba, model_name, REPORT_DIR / "pr_curve.png")
    plot_confusion_matrix(y_true, y_pred, model_name, REPORT_DIR / "confusion_matrix.png")
    print(f"Saved ROC, PR, Confusion Matrix to reports/")
    
    # Threshold analysis
    thr_df, best_thr = threshold_optimization(y_true, y_proba)
    thr_df.to_csv(REPORT_DIR / "threshold_comparison.csv", index=False)
    print(f"\nThreshold analysis saved. Best F1 threshold: {best_thr['Threshold']} with F1={best_thr['F1']:.4f}")
    # Plot threshold vs metrics
    plt.figure(figsize=(9,6))
    plt.plot(thr_df["Threshold"], thr_df["Precision"], marker='o', label="Precision")
    plt.plot(thr_df["Threshold"], thr_df["Recall"], marker='s', label="Recall")
    plt.plot(thr_df["Threshold"], thr_df["F1"], marker='^', label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Metrics (validation/test)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "threshold_analysis.png", dpi=200)
    plt.close()
    
    # Save evaluation report json
    report = {
        "model": model_name,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [["TN","FP"],["FN","TP"]],
        "best_threshold_F1": float(best_thr["Threshold"]),
        "best_threshold_F1_score": float(best_thr["F1"]),
        "threshold_table": thr_df.to_dict(orient="records"),
        "caveat": "Metrics on US Medicare provider-level data. High-risk ≠ proof of fraud. Report training performance vs test performance separately.",
        "minority_class_focus": {
            "recall": metrics["Recall"],
            "precision": metrics["Precision"],
            "pr_auc": metrics["PR_AUC"],
            "note": "With ~9.35% fraud prevalence, PR-AUC and Recall are primary; accuracy is misleading"
        }
    }
    with open(REPORT_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Also compare all models if model_comparison exists
    comp_path = REPORT_DIR / "validation_model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        # Plot model comparison
        plt.figure(figsize=(10,6))
        models = comp["Model"]
        x = np.arange(len(models))
        width = 0.2
        # If comp has those columns
        for i, metric in enumerate(["Precision","Recall","F1","PR_AUC" if "PR_AUC" in comp.columns else "ROC_AUC"]):
            if metric in comp.columns:
                plt.bar(x + i*width, comp[metric], width, label=metric)
        plt.xticks(x + width, models, rotation=15)
        plt.ylabel("Score")
        plt.title("Model Comparison (Validation)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "model_comparison.png", dpi=200)
        plt.close()
    
    print("Evaluation report saved to reports/evaluation_report.json")
    return metrics, thr_df

if __name__ == "__main__":
    evaluate_all_models()
