import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load cleaned dataset
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

# Create Random Forest pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Train model
print("Training Random Forest model...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Evaluation
print("\n========== RANDOM FOREST RESULTS ==========")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=["Not Potentially Fraudulent", "Potentially Fraudulent"]
))

print("ROC-AUC Score:", round(roc_auc_score(y_val, y_prob), 4))

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

print(
    "PR-AUC Score:",
    round(average_precision_score(y_val, y_prob), 4)
)

from sklearn.metrics import precision_score, recall_score, f1_score

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in [0.3, 0.4, 0.5, 0.6]:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(y_val, y_pred_threshold)
    recall = recall_score(y_val, y_pred_threshold)
    f1 = f1_score(y_val, y_pred_threshold)

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")