
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

DATA_PATH = Path("data/processed/provider_features_clean.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("HEALTHCARE FRAUD DETECTION - IMPROVED MODEL TRAINING")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

target = "PotentialFraud"

X = df.drop(columns=[target, "Provider"], errors="ignore")
y = df[target].astype(int)

X = X.select_dtypes(include=["number"])

print("Input features:", X.shape[1])
print("\nTarget distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. SPLIT DATA: 70% TRAIN, 15% VALIDATION, 15% TEST
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDATA SPLIT")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 4. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# --------------------------------------------------
# 5. TRAIN AND COMPARE USING VALIDATION DATA
# --------------------------------------------------

validation_results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 65)
    print("TRAINING:", name)

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_probabilities = model.predict_proba(X_val)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, val_predictions),
        "Precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_val, val_predictions, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_val, val_probabilities
        )
    }

    validation_results.append(results)
    trained_models[name] = model

    print("\nValidation results:")
    for metric, value in results.items():
        if metric != "Model":
            print(metric + ":", round(value, 4))

# --------------------------------------------------
# 6. SELECT MODEL USING VALIDATION PR-AUC
# --------------------------------------------------

comparison = pd.DataFrame(validation_results)
comparison = comparison.sort_values(
    by="PR-AUC", ascending=False
)

print("\n" + "=" * 65)
print("VALIDATION MODEL COMPARISON")
print("=" * 65)
print(comparison.to_string(index=False))

comparison.to_csv(
    REPORT_DIR / "validation_model_comparison.csv",
    index=False
)

best_name = comparison.iloc[0]["Model"]

print("\nSelected model:", best_name)

# --------------------------------------------------
# 7. RETRAIN SELECTED MODEL ON TRAIN + VALIDATION
# --------------------------------------------------

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

final_model = models[best_name]

print("\nRetraining selected model on training + validation data...")

final_model.fit(X_train_val, y_train_val)

# --------------------------------------------------
# 8. FINAL EVALUATION ON UNTOUCHED TEST DATA
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

test_predictions = final_model.predict(X_test)
test_probabilities = final_model.predict_proba(X_test)[:, 1]

print("\nAccuracy:",
      round(accuracy_score(y_test, test_predictions), 4))

print("Precision:",
      round(precision_score(y_test, test_predictions, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, test_predictions, zero_division=0), 4))

print("F1 Score:",
      round(f1_score(y_test, test_predictions, zero_division=0), 4))

print("PR-AUC:",
      round(average_precision_score(y_test, test_probabilities), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

# --------------------------------------------------
# 9. SAVE FINAL MODEL AND FEATURE NAMES
# --------------------------------------------------

joblib.dump(
    final_model,
    MODEL_DIR / "best_fraud_model.pkl"
)

joblib.dump(
    list(X.columns),
    MODEL_DIR / "feature_columns.pkl"
)

print("\nFINAL MODEL:", best_name)
print("Saved model: models/best_fraud_model.pkl")
print("Saved features: models/feature_columns.pkl")
print("Saved comparison: reports/validation_model_comparison.csv")

print("\nIMPROVED MODEL TRAINING COMPLETED!")
