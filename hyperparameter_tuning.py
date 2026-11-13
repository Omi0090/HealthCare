import os
import joblib
import json
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Load dataset
df = pd.read_csv(
    "reports/master_provider_features.csv"
)

# Separate features and target
X = df.drop(columns=["Provider", "PotentialFraud"])
y = df["PotentialFraud"].map({"No": 0, "Yes": 1})

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Parameters to test
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=15,
    scoring="average_precision",
    cv=5,
    random_state=42,
    n_jobs=1,
    verbose=1
)

print("Starting Random Forest hyperparameter tuning...")

random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_
# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the complete trained pipeline
joblib.dump(best_model, "models/tuned_random_forest.joblib")

print("\nTrained model saved successfully!")

print("\n========== BEST PARAMETERS ==========")
print(random_search.best_params_)

print("\nBest Cross-Validation PR-AUC:")
print(round(random_search.best_score_, 4))

# Validation predictions
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

print("\n========== TUNED RANDOM FOREST RESULTS ==========")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=[
        "Not Potentially Fraudulent",
        "Potentially Fraudulent"
    ]
))

print("ROC-AUC:", round(roc_auc_score(y_val, y_prob), 4))
print("PR-AUC:", round(average_precision_score(y_val, y_prob), 4))

print("PR-AUC:", round(average_precision_score(y_val, y_prob), 4))

from sklearn.metrics import precision_score, recall_score, f1_score

print("\n========== TUNED RF THRESHOLD COMPARISON ==========")

for threshold in [0.3, 0.4, 0.5, 0.6]:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(y_val, y_pred_threshold)
    recall = recall_score(y_val, y_pred_threshold)
    f1 = f1_score(y_val, y_pred_threshold)

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")