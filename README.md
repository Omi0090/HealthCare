# Healthcare Provider Fraud Detection

An end-to-end Machine Learning pipeline designed to detect potentially fraudulent healthcare providers using Medicare claims data.

---

## 📌 Problem Statement
Healthcare fraud leads to tens of billions of dollars lost annually. In Medicare and health insurance systems, fraud often manifests at the **provider level** rather than isolated claim level. Fraudulent behaviors include:
- Billing for services or procedures never performed ("phantom billing")
- Upcoding (billing for more expensive treatments than provided)
- Unbundling procedures into multiple claim entries
- Duplicate claims and suspicious physician/beneficiary overlap

The objective of this project is to analyze inpatient claims, outpatient claims, and beneficiary demographics to predict whether a healthcare provider exhibits potential fraud (`PotentialFraud: Yes/No`).

---

## 📂 Repository Structure
```
Healthcare_Fraud_Detection/
├── dashboard/               # Streamlit web application for interactive fraud analysis
├── data/
│   ├── raw/                # Original Medicare CSV datasets
│   └── processed/          # Merged and engineered provider-level features
├── models/                 # Serialized model artifacts (.joblib / .pkl)
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── reports/                # Evaluation plots, confusion matrices, feature importance
├── src/                    # Modular Python source code
│   ├── data_loader.py      # Data loading and initial cleaning
│   ├── feature_engineering.py # Aggregating claims & beneficiaries to provider level
│   ├── train.py            # Model training, cross-validation & hyperparameter tuning
│   └── evaluate.py         # Evaluation metrics (PR-AUC, ROC-AUC, Recall)
├── README.md               # Project documentation
└── requirements.txt        # Python package dependencies
```

---

## 📊 Dataset Description
The data originates from Medicare claims datasets:
1. **Train / Test Provider Labels**: Unique `Provider` IDs mapped to `PotentialFraud` (Yes / No).
2. **Beneficiary Data**: Beneficiary demographics (DOB, Gender, Race, State, County, RenalDisease, chronic conditions like Alzheimer's, Heart Failure, Cancer, Diabetes).
3. **Inpatient Claims**: Hospital admission & discharge dates, diagnosis codes (ICD-9), procedure codes, deductible amounts, and attending/operating physician IDs.
4. **Outpatient Claims**: Outpatient visits, claim start/end dates, physician details, diagnosis codes, deductible and reimbursed amounts.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (Python 3.11/3.12/3.14 recommended)

### 2. Environment Setup
Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Pipeline Workflow
1. **Data Preprocessing & Feature Engineering**:
   - Merge inpatient and outpatient claims.
   - Aggregate claim and beneficiary attributes up to Provider level.
2. **Model Training & Evaluation**:
   - Address severe class imbalance using SMOTE / class weighting.
   - Train ensemble models (Random Forest, LightGBM, XGBoost).
   - Evaluate primarily using PR-AUC, ROC-AUC, and Recall.
3. **Interactive Dashboard**:
   - Launch Streamlit to search providers, view risk scores, and inspect key contributing risk factors.

---

## 📈 Evaluation Metrics
Because fraudulent providers make up a small minority of all providers (~9% positive class), accuracy is misleading. The pipeline evaluates models using:
- **PR-AUC (Precision-Recall Area Under Curve)**: Primary optimization metric for rare positive fraud cases.
- **ROC-AUC**: Overall discriminatory power.
- **Recall / Sensitivity**: Prioritizing identification of true fraud cases to prevent financial leakage.
- **F1-Score**: Balance between precision and recall.
