# 🏥 AI-Based Healthcare Insurance Fraud Detection

### Intelligent Claim Analysis | Hybrid Risk Scoring | Explainable AI | Fraud Investigation Platform

> **Final-Year B.Tech CSE Project** — An end-to-end, professional fraud screening platform that prioritizes suspicious providers for human investigation using US Medicare claims data. **High-risk is an investigation signal, not proof of fraud.**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red) ![License](https://img.shields.io/badge/Data-Medicare%20Public-green)

---

## 📌 Problem Statement

Healthcare fraud costs tens of billions annually. Fraud manifests at the **provider level** via:
- Phantom billing (services never performed)
- Upcoding (expensive codes for cheaper services)
- Unbundling & duplicate billing
- Coordinated provider–beneficiary–physician patterns

Manual review doesn't scale; labels are rare (~9.3% fraud), delayed, and biased. This project builds an **explainable screening prototype** that:

1. Learns from historical provider-level labels
2. Combines supervised + anomaly + peer signals
3. Ranks providers for investigator review
4. Explains *why* with SHAP and peer context

**Scope & Caveats:**
- US **Medicare-oriented** benchmark (Kaggle Provider Fraud). **Not validated for Indian schemes** — would need Indian data, rules, external validation.
- Provider-level labels do not identify which claim was fraudulent.
- Duplicate/relationship flags are screening tools, not guilt.

---

## 🏗️ Project Architecture

```
Dataset Layer (Provider, Beneficiary, Inpatient, Outpatient)
        ↓
Validation & Preprocessing (leakage-safe, median/MAD peer groups)
        ↓
Feature Engineering (42 claim + 20 beneficiary features + peer Z)
        ↓
ML Detection Engine (LogReg, RF, XGBoost, LightGBM + IsolationForest)
        ↓
Hybrid Risk Scoring (0-100) → SHAP Explainability
        ↓
Advanced Modules (Duplicate, Temporal, Network, Drift)
        ↓
SQLite Investigation Workflow → Streamlit Dashboard (12 pages)
```

---

## 📂 Repository Structure

```
HealthCare/
├── dashboard/
│   ├── app.py                # Main Streamlit app (12 pages, enterprise UI)
│   └── components/
│       ├── ui.py             # KPI cards, risk badges, charts
│       └── db_utils.py       # DB helpers
├── data/
│   ├── raw/                  # Original 8 Medicare CSVs (Kaggle)
│   ├── processed/            # cleaned_provider_features.csv (4.4M, 5410 providers)
│   └── external/             # (reserved)
├── models/
│   ├── best_fraud_model.pkl  # LightGBM (chosen by validation PR-AUC 0.777)
│   ├── isolation_forest.pkl  # Unsupervised anomaly baseline
│   ├── feature_columns.pkl   # 77 feature names
│   └── model_metadata.json   # Metrics, version, caveat
├── src/
│   ├── data_loader.py        # Inventory, dictionary, quality report
│   ├── data_verification.py  # Phase 1 script
│   ├── preprocessing.py      # Robust MAD, peer groups, leakage checks
│   ├── feature_engineering.py# Provider-level aggregates (reproducible)
│   ├── train.py              # 5-model comparison, SMOTE/class-weight, CV
│   ├── evaluate.py           # ROC/PR, threshold, confusion, report
│   ├── explainability.py     # SHAP global + individual + what-if
│   ├── anomaly_detection.py  # IsolationForest, temporal, peer
│   ├── risk_scoring.py       # Hybrid 0.55/0.25/0.20, configurable thresholds
│   ├── network_analysis.py   # NetworkX provider-beneficiary-physician
│   ├── drift_detection.py    # Feature/prediction drift, versioning
│   ├── copilot.py            # Rule-based Q&A (grounded, no LLM)
│   ├── risk_timeline.py      # Monthly timeline for providers
│   └── pdf_report.py         # ReportLab investigation PDF
├── database/
│   ├── models.py             # SQLAlchemy: Provider, Claim, Prediction, Investigation...
│   ├── db.py                 # Init & populate from reports
│   └── schema.sql            # DDL + indexes
├── reports/                  # All generated artifacts (60+ files)
│   ├── master_provider_features.csv (81 cols, 5410 providers)
│   ├── hybrid_risk_scores.csv (risk 0-100, categories)
│   ├── validation_model_comparison.csv
│   ├── evaluation_report.json
│   ├── shap_*.png, roc_curve.png, pr_curve.png, eda_*.png
│   ├── peer_group_alerts.csv, temporal_anomaly_alerts.csv, network_*.csv
│   └── investigation_report_*.pdf
├── notebooks/
│   └── eda_analysis.ipynb    # Jupyter EDA
├── scripts/
│   ├── train_models.py       # Legacy (preserved)
│   ├── detect_duplicate_claims.py # Tiered duplicate screening
│   └── eda.py                # EDA script
├── tests/
│   └── test_pipeline.py      # 12 tests (data, model, DB, PDF, dashboard)
├── requirements.txt          # CPU-friendly, no paid APIs
├── Dockerfile                # Local Docker deployment
└── README.md                 # This file
```

**Preserved original files:** `build_provider_features.py`, `hyperparameter_tuning.py`, etc. are kept for reference; new `src/` is canonical.

---

## 🚀 Technology Stack (per PPT, CPU-friendly)

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.11 |
| **Data** | Pandas, NumPy |
| **ML** | scikit-learn (LogReg, RF, IsolationForest), XGBoost, LightGBM |
| **Imbalance** | class_weight + optional SMOTE (imblearn) |
| **Explainability** | SHAP, feature_importance |
| **Analytics** | NetworkX, robust MAD (`(x-median)/(1.4826*MAD)`), temporal z-scores |
| **Dashboard** | Streamlit, Plotly |
| **DB** | SQLite + SQLAlchemy |
| **Viz** | Matplotlib, Seaborn, Plotly |
| **Reports** | ReportLab |
| **Testing** | pytest |
| **Deploy** | Docker |

All run **locally, without GPU or paid APIs**.

---

## 📊 Dataset

**Source:** Kaggle “Healthcare Provider Fraud Detection Analysis” + CMS DE-SynPUF (alternative) + HHS OIG LEIE (supplementary). Verified files:

| Table | Rows | Cols | Key |
|-------|------|------|-----|
| Train labels | 5,410 | 2 | Provider, PotentialFraud (Yes: 506, No: 4,904 = 9.35% fraud) |
| Train beneficiary | 138,556 | 25 | BeneID, DOB, chronic conditions (1→0,2→1) |
| Train inpatient | 40,474 | 30 | ClaimID, Provider, BeneID, dates, reimbursement |
| Train outpatient | 517,737 | 27 | Same, no AdmissionDt |
| Test (providers, bene, inpatient, outpatient) | 1,353 / 63,968 / 9,551 / 125,841 | — | For holdout; labels not used |

**Relationships:** Provider (1→N claims), BeneID (claims→beneficiary), ClaimID unique within each claim type. **No leakage:** `PotentialFraud` only in labels table.

**Data Quality:** 0 duplicates, 137k missing in beneficiary (DOD), 8M missing in outpatient (diagnosis codes). All dates valid, no negative reimbursements.

See `reports/data_dictionary.csv`, `reports/data_quality_report.json`, `reports/dataset_inventory.json`.

---

## 🔧 Feature Engineering (Provider-Level, 77 features)

**Claim aggregates:**
TotalClaims, InpatientClaims, OutpatientClaims, UniqueBeneficiaries, Total/Average/Median/Max/Min/StdReimbursement, AverageDeductible, AverageClaimDuration, UniqueAttending/Operating/OtherPhysicians, InpatientClaimRatio, ClaimsPerBeneficiary, ReimbursementVariation (CV), LogTotal/AvgReimbursement, Inp_/Out_ splits, AvgMonthly/MaxMonthly/StdMonthly, MonthlyClaimVariation, DuplicateClaimCount/Ratio, AvgDaysBetweenClaims...

**Beneficiary aggregates:**
AveragePatientAge, Max/Min/StdPatientAge, UniquePatients, Avg ChronicCond (Alzheimer, Heartfailure, KidneyDisease, Cancer, ObstrPulmonary, Depression, Diabetes, IschemicHeart, Osteoporasis, rheumatoidarthritis, stroke), Avg_Gender, Avg_MonthsPartA/B.

**Peer-group:**
Providers binned by `TotalClaims` quantiles → Very Low/Low/Medium/High/Very High. For `TotalReimbursement`, `AverageReimbursement`, `ClaimsPerBeneficiary`, `AverageClaimDuration`, `DuplicateClaimRatio`: compute peer median, MAD, robust **z = (x−median)/(1.4826×MAD)**. Flags |z|>2.5.

All preprocessing **fit on train only** (imputer median), no target leakage.

---

## 🤖 Machine Learning

### Models Compared (Validation PR-AUC, 811 val `70/15/15` split, stratified)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC (primary) | CV PR-AUC |
|-------|----------|-----------|--------|----|---------|------------------|-----------|
| **LightGBM** | 0.937 | 0.647 | 0.724 | 0.683 | 0.954 | **0.777** | 0.754±0.03 |
| XGBoost | 0.937 | 0.632 | 0.789 | 0.702 | 0.956 | 0.772 | 0.740±0.027 |
| LogisticRegression | 0.883 | 0.441 | 0.934 | 0.599 | 0.960 | 0.762 | 0.697±0.05 |
| RandomForest | 0.932 | 0.608 | 0.776 | 0.682 | 0.960 | 0.740 | 0.721±0.028 |
| IsolationForest (unsupervised) | — | — | — | — | 0.845 | 0.436 | — |

**Selected:** LightGBM (best validation PR-AUC) → retrained on Train+Val (4,598 rows) → **Test PR-AUC 0.752, ROC-AUC 0.944, Recall 0.697**.

**Imbalance handling:** Compared `class_weight="balanced"` (chosen) vs SMOTE; class_weight preserved calibration and was leakage-safe (resampling only on train).

**Splitting:** Provider-aware, stratified 70/15/15; no provider leakage across splits (provider is unit of analysis).

**Best threshold (validation F1):** 0.40 → F1 0.678, Precision 0.58, Recall 0.81.

See `reports/validation_model_comparison.csv`, `reports/evaluation_report.json`, `reports/roc_curve.png`, `reports/pr_curve.png`, `reports/confusion_matrix.png`.

---

## ⚖️ Hybrid Fraud Detection Engine

**Transparent, configurable, validated:**

```
HybridScore = 0.55 × SupervisedScore + 0.25 × AnomalyScore + 0.20 × PeerScore
```

- **SupervisedScore** = calibrated fraud probability ×100 (0–100)
- **AnomalyScore** = IsolationForest normalized 0–100 (100 most anomalous)
- **PeerScore** = peer MAD severity scaled 0–100 (cap 95th percentile, min-max)

**Weights** from validation: supervised highest PR-AUC, but anomaly+peer catch label-blind risks. Sensitivity analysis in `reports/hybrid_scoring_config.json`.

**Risk Categories (configurable, documented):**
Low 0–30, Medium 30–60, High 60–80, Critical 80–100 → Current distribution: Low 4,602, Medium 353, High 299, Critical 156.

Individual components and thresholds stored alongside final score (`reports/hybrid_risk_scores.csv`). Never arbitrary claims of validation — methodology in config JSON.

---

## 🔍 Explainable AI

- **Global:** Mean |SHAP| importance → `reports/shap_feature_importance.csv`, `shap_summary.png`, `shap_feature_importance.png` (LightGBM TreeExplainer)
- **Individual:** Waterfall per provider, top positive/negative contributions (`reports/shap_waterfall_{Provider}.png`)
- **Counterfactual:** What-if slider adjusts features → Δ fraud probability, marked hypothetical
- **Caveat:** SHAP is model explanation, not causal proof — always shown

Top features: `TotalReimbursement`, `MaxClaimDuration`, `ClaimsPerBeneficiary` (+ peer Z variants).

Run: `python src/explainability.py` (computes for 2000 sample) or via dashboard **Explainable AI** page.

---

## 🧩 Advanced Fraud Detection Modules

### A. Duplicate & Near-Duplicate Claims
Tiered screening (preserved from `scripts/detect_duplicate_claims.py`):
- Exact full duplicates (all cols incl. ClaimID)
- Content duplicates (excl. ClaimID)
- Same-provider duplicates (BeneID+Provider+dates+amount, with/without diagnosis)
- Cross-provider duplicates (BeneID+dates+amount, different provider)
Assigns `RiskFlag`, `GroupID`, `Reason`; **never treats repeated BeneID/Provider alone as fraud**; notes corrections/resubmissions.

Reports: `reports/inpatient_duplicate_claims_report.csv`, `outpatient_duplicate_claims_report.csv`, `duplicate_provider_summary.csv` (293 providers flagged).

### B. Temporal Anomaly Detection
Monthly claim volume & reimbursement per provider; rolling z-scores (mean+2σ spike), trend ratio (second half vs first half), volatility. Alerts: `reports/temporal_anomaly_alerts.csv` (1,160 alerts), `temporal_provider_features.csv`.

Provider timeline visualized in dashboard (Explainable AI → Risk Timeline).

### C. Provider Relationship Network
NetworkX graph: Provider–Beneficiary–Physician (from Attending/OperatingPhysician). Metrics: connected components (2,084), degree centrality (PRV51459 most central), shared beneficiary pairs (>5 shared = 1 suspicious pair PRV51347↔PRV51567), shared physician pairs. **Only uses reliable identifiers; shared ≠ collusion.**

Reports: `network_shared_beneficiaries.csv`, `network_shared_physicians.csv`, `network_centrality.csv`, `network_graph.png`.

### D. Provider Peer-Group Analysis
Robust MAD deviations per claim-volume quantile; flags 4,475 deviations across 2,481 providers → `peer_group_alerts.csv`, `peer_group_deviation_summary.csv`.

### E. Model Drift Monitoring
Feature drift (KS test per feature, 3/77 flagged), prediction drift (KS 0.039, p=0.031), model version tracking (`model_versions.json`), alerts (`drift_alerts.csv`). Distinguishes **data drift ≠ performance drop**; requires labeled validation before retraining.

---

## 🗄️ Database & Investigation Management

**SQLite + SQLAlchemy** (`database/healthcare_fraud.db`, 1.3 MB):

| Table | Purpose |
|-------|---------|
| `providers` | Provider features + hybrid scores |
| `claims` | Drill-down claims (optional) |
| `predictions` | Prediction history (model version, proba, threshold) |
| `risk_scores` | Hybrid components + weights |
| `investigations` | Cases: New → Under Review → Escalated → Closed (No Concern / Requires Further Review) |
| `investigation_notes` | Timestamped reviewer notes |
| `model_versions` | Version, metrics, active flag |
| `audit_logs` | All actions (create case, add note, status change) |

**Workflow (dashboard Investigation Centre):**
1. Open provider profile → risk, SHAP, peer, claims
2. Create case (priority, assignee, summary)
3. Add notes, attach evidence
4. Update status
5. View history, export PDF
6. Audit trail persisted

Seeded with 5 high-risk cases (High/Critical).

---

## 📊 Streamlit Dashboard (12 Pages)

Run `streamlit run dashboard/app.py` → `http://localhost:8501` (professional enterprise UI: navy sidebar, KPI cards, Plotly, searchable tables, risk badges).

| # | Page | Key Features |
|---|------|--------------|
| 1 | **Executive Overview** | KPIs (providers, claims, high-risk), risk pie/bar, model KPIs, trends |
| 2 | **Provider Explorer** | Search/filter/sort, provider profile (claims, peer, timeline), drill-down |
| 3 | **Fraud Analytics** | Fraud distribution, reimbursement boxplots, inpatient vs outpatient, chronic heatmap, provider comparison, duplicate patterns |
| 4 | **AI Predictions** | Live inference, hybrid breakdown (transparent weights/thresholds), history, bulk CSV |
| 5 | **Explainable AI** | SHAP global/individual, waterfall, what-if simulation, **Copilot** (grounded Q&A), **Risk Timeline** |
| 6 | **Anomaly Detection** | IsolationForest ranking, peer deviations, temporal alerts |
| 7 | **Network Analysis** | Graph, centrality, shared beneficiary/physician, disclaimer |
| 8 | **Investigation Centre** | Cases, status workflow, notes, evidence, history |
| 9 | **Model Performance** | Comparison, ROC/PR, confusion, threshold tuning (best 0.40), CV |
| 10 | **Model Monitoring** | Feature drift KS, prediction drift, versioning, alerts |
| 11 | **Reports** | PDF (ReportLab), CSV/JSON downloads, audit logs |
| 12 | **Settings** | Threshold sliders (live recompute), model/app info, DB stats |

**All pages use actual data; empty states if artifacts missing; never fabricates.**

### Unique Features (Phase 11)

- **Fraud Investigation Copilot:** Rule-based Q&A (“Why high-risk?”) grounded in hybrid scores, SHAP, peer/temporal alerts, claims — no paid LLM, no hallucination.
- **Provider Risk Timeline:** Monthly claim volume/reimbursement + hybrid trace (dashboard).
- **Automated PDF Report:** Professional 2-page PDF (ReportLab): provider ID, risk, SHAP, peer, claim sample, notes, model version, date (`reports/investigation_report_{Provider}.pdf`).
- **What-if Simulation:** Adjust 5 features → live Δ prediction, flagged hypothetical.

---

## 🧪 Testing & Quality

```bash
pytest tests/test_pipeline.py -v  # 12 tests
```

Covers: data loading, preprocessing, feature columns, model artifacts, prediction compatibility, hybrid scoring, SHAP, duplicates, DB ops, investigation workflow, PDF generation, dashboard load.

**Verified:** No missing deps, no broken imports, no hardcoded fake metrics, no path errors. All plots save to `reports/`.

---

## 🐳 Deployment

```bash
docker build -t healthcare-fraud .
docker run -p 8501:8501 healthcare-fraud
```

Healthcheck via Streamlit `_stcore/health`. No GPU required.

---

## ⚠️ Ethics, Compliance & Limitations

- **Investigation signal only:** High-risk ≠ guilt; never auto-deny claim.
- **US Medicare only:** Not validated for Indian/private/global insurers without local data.
- **De-identified public data:** No PHI; governance for real claims.
- **SHAP ≠ causality:** Requires clinical/domain review.
- **Duplicates/relationships:** Screening, not proof; corrections exist.

---

## 📚 References & Credits

- Kaggle Provider Fraud dataset, CMS DE-SynPUF, HHS OIG LEIE
- Literature: Johnson & Khoshgoftaar (2023), Hancock et al. (2023), Hamid et al. (2024), Hong et al. (2025), Raghi et al. (2025), etc. (see PPT)
- B.Tech CSE team: Yash Patil, Soham Shinde, Durvang Kulkarni, Dhiren Ade, Om Dhanal | Guide: Reecha Suryavanshi | 2026-27

---

## HOW TO RUN THE COMPLETE PROJECT ON MY WINDOWS LAPTOP

> **Step-by-step from scratch on Windows 10/11 + VS Code + Python 3.11, CPU laptop, no GPU.**

### STEP 1: Install Python and VS Code

1. **Python 3.11:**
   - Go to https://www.python.org/downloads/ → Download Python 3.11.x
   - Run installer → **Check “Add python.exe to PATH”** → Install Now.
   - Verify: Open **Command Prompt** (Win+R → `cmd`) → `python --version` → should show `Python 3.11.x` (if 3.12+ also works, but use 3.11 for best compatibility).
   - If `python` not found, try `py --version` and use `py` in commands below.
2. **VS Code:**
   - https://code.visualstudio.com/ → Download Windows → Install (check “Add to PATH”).
   - Open VS Code → Extensions → install **Python** (Microsoft) and **Pylance**.

### STEP 2: Clone the GitHub Repository or Extract ZIP

**Option A: Git clone (recommended):**
```cmd
:: Open Command Prompt or PowerShell
cd %USERPROFILE%\Documents
git clone https://github.com/Omi0090/HealthCare.git
cd HealthCare
git checkout arena/ea2a78c6-healthcare
```
**Option B: ZIP:**
- Download ZIP from GitHub → Right-click → Extract All → `C:\Users\YourName\Documents\HealthCare`.

### STEP 3: Open the Project Folder in VS Code

1. Open VS Code → **File → Open Folder** → select `HealthCare`.
2. **Terminal → New Terminal** (should open with `HealthCare` path).
3. Verify files: `dir` → you should see `data`, `src`, `dashboard`, `requirements.txt`.

### STEP 4: Create a Python Virtual Environment

```cmd
:: In VS Code terminal (Windows cmd)
python -m venv venv
:: If python not found, try:
py -m venv venv
```

This creates `venv\` folder (isolated packages).

### STEP 5: Activate the Virtual Environment

```cmd
:: Windows Command Prompt:
venv\Scripts\activate

:: PowerShell (if error, run as admin: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser):
.\venv\Scripts\Activate.ps1

:: You should see (venv) prefix in terminal
```

**To deactivate later:** `deactivate`

### STEP 6: Install All Dependencies

```cmd
:: With (venv) active
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Verify (optional, may take 2-5 min for xgboost/shap):
python -c "import pandas, sklearn, xgboost, lightgbm, shap, streamlit; print('All installed')"
```

**Troubleshooting 6:**
- If `pip install` fails due to permissions → add `--user` or run terminal as admin.
- If `shap` build fails → `pip install shap --no-build-isolation` or skip; dashboard falls back to feature importance.

### STEP 7: Verify and Prepare the Datasets

**Data already included in `data/raw/` (8 CSVs, 118 MB).** Verify:

```cmd
dir data\raw
:: Should list: Train-*.csv, Train_Beneficiarydata-*.csv, Train_Inpatientdata-*.csv, Train_Outpatientdata-*.csv, Test-*.csv etc.

:: Run verification script to generate audit reports:
python src/data_verification.py
```

This creates `reports/dataset_inventory.json`, `data_dictionary.csv`, `data_quality_report.json`, `verification_summary.txt`.

**If data missing:** Download from Kaggle “Healthcare Provider Fraud Detection Analysis” and place the 8 CSVs in `data\raw\`. File names must match `Train-*.csv` pattern (timestamps can vary; script auto-detects).

### STEP 8: Run Data Preprocessing

```cmd
:: Optional explicit verification (Phase 1) already run above.
:: The next step does preprocessing internally, but you can run preprocessing utils:
python -c "from src.preprocessing import RobustMADTransformer; print('Preprocessing OK')"
```

### STEP 9: Generate Provider-Level Features

```cmd
:: This is Phase 2: aggregates 558k claims + 138k beneficiaries → 5410 providers × 81 columns
python src/feature_engineering.py

:: Expected output: Merged master shape (5410, 81), saved to reports/master_provider_features.csv and data/processed/cleaned_provider_features.csv
:: Time: ~5-10 seconds
```

**What it does:** Maps chronic 1/2→0/1, computes claim duration, aggregates 42 claim + 20 beneficiary features, peer-group MAD Z, Log transforms. Fills missing, no leakage.

**Re-run?** Safe to re-run; overwrites outputs. Run again if you change raw data.

### STEP 10: Train the ML Models (if artifacts not already supplied)

```cmd
:: Phase 4: Train 5 models (LogReg, RF, XGBoost, LightGBM, IsolationForest), 70/15/15 split, 5-fold CV
python src/train.py

:: Time: ~15-30 sec (CPU)
:: Saves: models/best_fraud_model.pkl (LightGBM), isolation_forest.pkl, feature_columns.pkl, model_metadata.json
:: Also generates: reports/validation_model_comparison.csv, test_predictions.csv
```

**Artifacts supplied:** Already in `models/` (LightGBM PR-AUC 0.752). Retrain to update, or keep as-is.

### STEP 11: Evaluate the Models

```cmd
python src/evaluate.py

:: Generates: reports/roc_curve.png, pr_curve.png, confusion_matrix.png, threshold_analysis.png, evaluation_report.json, threshold_comparison.csv
```

### STEP 12: Generate and Save Model Artifacts (already done via train.py)

Artifacts are:
- `models/best_fraud_model.pkl`
- `models/isolation_forest.pkl`
- `models/feature_columns.pkl`
- `models/model_metadata.json`

Verify:
```cmd
dir models
python -c "import joblib; print(joblib.load('models/feature_columns.pkl')[:5])"
```

### STEP 13: Initialise the SQLite Database

```cmd
:: Phase 9: Create database + populate with 5410 providers, predictions, seeded investigations
python database/db.py

:: Or with PYTHONPATH:
python -m database.db

:: Creates: database/healthcare_fraud.db (1.3 MB)
:: Check:
python -c "import sqlite3; print(sqlite3.connect('database/healthcare_fraud.db').execute('select count(*) from providers').fetchone())"
:: Should print (5410,)
```

**Re-init?** Safe to re-run; deletes and repopulates (investigations reset). Your manual cases will be overwritten — export first.

### STEP 14: Start the Streamlit Dashboard

```cmd
:: Ensure venv active and port free
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

:: Or with headless for Docker-like:
streamlit run dashboard/app.py --server.port 8501 --server.headless true

:: Wait for: "You can now view your Streamlit app in your browser. Local URL: http://localhost:8501"
```

**What runs:** 12-page enterprise dashboard, loads hybrid scores, SHAP, network, investigations.

### STEP 15: Open the Local Dashboard URL in the Browser

1. Open Chrome/Edge → Go to `http://localhost:8501` (or click link in terminal).
2. You should see **FraudShield AI** sidebar (navy) + Executive Overview.
3. If “No provider data” → ensure Step 9 & 13 succeeded and `reports/hybrid_risk_scores.csv` exists (if not, run `python src/anomaly_detection.py` and `python src/risk_scoring.py`).

**To allow external preview (E2B):** Dashboard binds to `0.0.0.0`; host-check warnings need fixing via `--server.address`.

### STEP 16: Explain How to Use Every Dashboard Page

1. **Executive Overview:** KPIs, risk pie, model PR-AUC — glance at fraud prevalence.
2. **Provider Explorer:** Type `PRV51001` → filter High/Critical → click provider → see claims, peer Z, create case.
3. **Fraud Analytics:** Tabs: Fraud distribution, Billing (reimbursement boxplots), Provider Comparison (select 2 providers), Duplicate Patterns (flagged claims).
4. **AI Predictions:** Select provider → see fraud prob, hybrid breakdown (55/25/20), methodology → “Regenerate Live Inference” → Bulk CSV download.
5. **Explainable AI:** Global SHAP bar → select provider → “Generate SHAP” → waterfall + positive/negative → **What-If** (slide reimbursement) → **Copilot** (ask “Why high-risk?”) → **Risk Timeline** (monthly claims).
6. **Anomaly Detection:** IsolationForest scores, histogram, peer deviations table, temporal alerts.
7. **Network Analysis:** Shared beneficiary pairs (currently 1 pair), centrality, graph image; “Build Network” if missing.
8. **Investigation Centre:** List cases (5 seeded) → select case → add note → update status → audit log → “Create New Case”.
9. **Model Performance:** Validation table (LightGBM best), ROC/PR images, confusion matrix, threshold slider (best 0.40).
10. **Model Monitoring:** Feature drift KS, prediction drift, version JSON, alerts.
11. **Reports:** Select provider → enter notes → “Generate PDF” → Download; also download all CSVs/JSONs; audit logs.
12. **Settings:** Move threshold sliders → “Save & Recompute” → see new categories; view model/app/DB info; “Re-initialize DB”.

**Navigation:** Sidebar radio (12 options). **Caveat banners** remind “Investigation signal, US Medicare only”.

### STEP 17: Explain How to Generate an Investigation Report

1. Go to **Reports** → **Investigation PDF** tab.
2. Select Provider (e.g., `PRV53706` → Hybrid 95.1 Critical).
3. Check **Include SHAP** and **Include claim sample**.
4. Enter investigation notes (e.g., “High TotalReimbursement z=10.44, verify billing for 5 inpatient claims”).
5. Click **Generate PDF Report** → wait → **Download PDF**.
6. PDF contains: header, risk summary (color-coded), provider profile table, suspicious indicators, SHAP, peer comparison, claim sample, notes, model version, disclaimer.

**Also via Provider Explorer:** Select provider → “Generate Investigation Report” → redirects to Reports.

**Via CLI:** `python src/pdf_report.py` → generates `reports/investigation_report_PRV53706.pdf`.

### STEP 18: Explain How to Stop and Restart the Application

**Stop:** In terminal where `streamlit run` is running → `Ctrl+C` → `Y` if prompted. Or close terminal.

**Restart:**
```cmd
:: Activate venv if new terminal:
venv\Scripts\activate
:: Re-run:
streamlit run dashboard/app.py
```

**Quick restart without venv re-install:** Just `streamlit run dashboard/app.py` (if venv still active, or use `.\venv\Scripts\streamlit run dashboard/app.py`).

**Stop background process (if started via `start_process`):** Use `stop_process` tool or close preview.

### STEP 19: Provide Troubleshooting for Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'sklearn'` | Venv not activated | `venv\Scripts\activate` → `pip install -r requirements.txt` |
| `No file matching pattern 'Train-*.csv'` | Raw data missing | Ensure 8 CSVs in `data/raw/`; names must contain `Train`/`Test` |
| `No provider data found` (dashboard) | Feature engineering not run | `python src/feature_engineering.py` |
| `Supervised model not found` | Train not run | `python src/train.py` |
| `Database not found` | DB not initialized | `python database/db.py` (or `PYTHONPATH=. python database/db.py`) |
| `SHAP not installed` | Optional dep | `pip install shap` or ignore (fallback works) |
| `Port 8501 already in use` | Another Streamlit running | `netstat -ano | findstr 8501` → `taskkill /PID <PID> /F` or use `--server.port 8502` |
| `streamlit: command not found` | Venv not in PATH | `.\venv\Scripts\streamlit run dashboard/app.py` or `python -m streamlit run dashboard/app.py` |
| `MemoryError` on feature engineering | Large outpatient CSV | Close other apps; script is CPU-friendly (<2GB), but ensure 8 GB RAM free |
| `OperationalError: no such table: providers` | DB schema mismatch | Delete `database/healthcare_fraud.db` → `python database/db.py` |
| `Duplicate reports missing` | Duplicate script not run | `python scripts/detect_duplicate_claims.py` (optional, dashboard shows warning) |
| `Graph build slow` | NetworkX on 558k claims | Uses 20k sample by default; increase `sample_size` if needed |
| `PDF generation fails` | reportlab not installed | `pip install reportlab` |
| `Permission denied on venv\Scripts\activate` (PowerShell) | ExecutionPolicy | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |

**Which scripts to run once vs rerun:**
- **Once:** `src/data_verification.py`, `src/feature_engineering.py`, `src/train.py`, `src/evaluate.py`, `database/db.py`, `src/explainability.py` (SHAP cache), `src/drift_detection.py`.
- **Rerun anytime:** `src/anomaly_detection.py`, `src/risk_scoring.py` (if you change weights/thresholds), `src/network_analysis.py`, `src/pdf_report.py`, `scripts/eda.py`, dashboard.
- **Not manually during dashboard:** `dashboard/app.py` is run via `streamlit run`; `src/preprocessing.py` is imported, not standalone.

**Logs & verification:**
```cmd
:: Check all artifacts exist:
dir reports\*.csv
dir reports\*.png
dir models\*.pkl
dir database\*.db
:: Run tests:
pytest tests/test_pipeline.py -v
:: All 12 should pass
```

---

## 📄 License & Disclaimer

- **Data:** Public US Medicare (Kaggle, CMS). No PHI; synthetic where noted.
- **Model:** Screening tool; not a legal determination. High-risk ≠ proof.
- **Geographic:** US Medicare benchmark; Indian applicability requires local validation.
- **For B.Tech demo:** All metrics reproducible via `reports/` artifacts; no fabricated results.

---

**Built for B.Tech Evaluators:** Complete, tested, professional fraud investigation platform — run locally, no cloud, no paid APIs. `HOW TO RUN` above provides exact Windows commands for every phase.

*FraudShield AI — Human-in-the-loop, Explainable, Local-first.*
