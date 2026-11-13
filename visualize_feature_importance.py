
import pandas as pd
import matplotlib.pyplot as plt

# Load feature importance results
df = pd.read_csv("reports/feature_importance.csv")

# Select top 15 features
top_features = df.head(15).sort_values(
    by="Importance",
    ascending=True
)

# Create horizontal bar chart
plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 15 Important Features - Random Forest")

plt.tight_layout()

# Save graph
plt.savefig(
    "reports/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Feature importance graph saved successfully!")
