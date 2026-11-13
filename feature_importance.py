
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/tuned_random_forest.joblib")

# Load the master dataset to get feature names
df = pd.read_csv("reports/master_provider_features.csv")

# Remove Provider ID and target column
X = df.drop(columns=["Provider", "PotentialFraud"])

# Extract feature importance from Random Forest
classifier = model.named_steps["classifier"]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": classifier.feature_importances_
})

# Sort features from highest to lowest importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 features
print("\n========== TOP 15 IMPORTANT FEATURES ==========\n")
print(importance_df.head(15).to_string(index=False))

# Save complete feature importance results
importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

print("\nFeature importance analysis completed!")
print("Results saved to reports/feature_importance.csv")
