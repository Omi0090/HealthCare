
import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

print("=" * 65)
print("HEALTHCARE FRAUD DETECTION - SHAP EXPLAINABILITY")
print("=" * 65)

# 1. Load the cleaned dataset
data_path = "data/processed/provider_features_clean.csv"
df = pd.read_csv(data_path)

# 2. Load the saved model and feature names
model = joblib.load("models/best_fraud_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("\nDataset shape:", df.shape)
print("Number of model features:", len(feature_columns))

# 3. Prepare input features
X = df[feature_columns].copy()

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

# 4. Extract the scaler and Logistic Regression model
scaler = model.named_steps["scaler"]
logistic_model = model.named_steps["model"]

# Apply the same scaling used during model training
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(
    X_scaled,
    columns=feature_columns,
    index=X.index
)

# 5. Select a sample for SHAP analysis
sample_size = min(500, len(X_scaled))
X_sample = X_scaled.sample(
    n=sample_size,
    random_state=42
)

print("\nGenerating SHAP explanations...")
print("Number of providers analyzed:", sample_size)

# 6. Create SHAP explainer
explainer = shap.LinearExplainer(
    logistic_model,
    X_scaled.sample(min(200, len(X_scaled)), random_state=42)
)

shap_values = explainer(X_sample)

# Handle binary classification output
if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

# 7. Create reports folder if required
os.makedirs("reports", exist_ok=True)

# 8. Global feature importance plot
plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("SHAP - Top 15 Important Features")
plt.tight_layout()
plt.savefig(
    "reports/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 9. SHAP summary plot
plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("SHAP - Feature Impact Summary")
plt.tight_layout()
plt.savefig(
    "reports/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 10. Explain one individual provider
plt.figure()
shap.plots.waterfall(shap_values[0], max_display=15, show=False)
plt.title("SHAP - Individual Provider Explanation")
plt.tight_layout()
plt.savefig(
    "reports/shap_individual_explanation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 11. Display top features
importance = pd.DataFrame({
    "Feature": feature_columns,
    "Mean_Absolute_SHAP": abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

importance.to_csv(
    "reports/shap_feature_importance.csv",
    index=False
)

print("\nTOP 10 IMPORTANT FEATURES:")
print(importance.head(10).to_string(index=False))

print("\nSHAP EXPLANATIONS GENERATED SUCCESSFULLY!")
print("Saved files:")
print("- reports/shap_feature_importance.png")
print("- reports/shap_summary.png")
print("- reports/shap_individual_explanation.png")
print("- reports/shap_feature_importance.csv")
