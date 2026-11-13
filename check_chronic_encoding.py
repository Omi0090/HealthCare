import pandas as pd

df = pd.read_csv(
    "data/raw/Train_Beneficiarydata-1542865627584.csv"
)

columns = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Diabetes",
    "ChronicCond_Heartfailure"
]

for col in columns:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False))