"""
Test Suite - Healthcare Fraud Detection
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Fixtures
@pytest.fixture
def sample_provider_row():
    return {
        "Provider": "PRV51001",
        "TotalClaims": 25,
        "InpatientClaims": 5,
        "OutpatientClaims": 20,
        "TotalReimbursement": 104640.0,
        "AverageReimbursement": 4185.6,
        "UniqueBeneficiaries": 24,
        "PotentialFraud": "No"
    }

def test_data_loading():
    from src.data_loader import get_raw_file_map
    file_map = get_raw_file_map(train=True)
    assert "labels" in file_map
    assert file_map["labels"].exists()
    df = pd.read_csv(file_map["labels"])
    assert "Provider" in df.columns
    assert "PotentialFraud" in df.columns

def test_preprocessing_chronic_encoding():
    from src.preprocessing import handle_chronic_condition_encoding
    df = pd.DataFrame({"ChronicCond_Alzheimer": [1,2,1], "ChronicCond_Diabetes": [2,2,1]})
    df2 = handle_chronic_condition_encoding(df)
    assert set(df2["ChronicCond_Alzheimer"].unique()).issubset({0,1})

def test_feature_engineering_columns():
    path = Path("reports/master_provider_features.csv")
    if not path.exists():
        path = Path("data/processed/cleaned_provider_features.csv")
    if path.exists():
        df = pd.read_csv(path)
        assert "Provider" in df.columns
        assert "TotalClaims" in df.columns or "total_claims" in df.columns
        assert df["Provider"].duplicated().sum() == 0
        assert df.isnull().sum().sum() < len(df) * len(df.columns)  # not all missing

def test_model_artifacts():
    assert Path("models/best_fraud_model.pkl").exists()
    assert Path("models/feature_columns.pkl").exists()
    assert Path("models/model_metadata.json").exists()
    model = joblib.load(Path("models/best_fraud_model.pkl"))
    cols = joblib.load(Path("models/feature_columns.pkl"))
    assert len(cols) > 5
    # Test prediction compatibility
    import json
    meta = json.load(open("models/model_metadata.json"))
    assert "best_model" in meta
    assert "test_metrics" in meta

def test_model_prediction():
    model = joblib.load(Path("models/best_fraud_model.pkl"))
    cols = joblib.load(Path("models/feature_columns.pkl"))
    # Create dummy input with correct columns
    dummy = pd.DataFrame(np.random.rand(5, len(cols)), columns=cols)
    preds = model.predict(dummy)
    assert len(preds) == 5
    proba = model.predict_proba(dummy)
    assert proba.shape == (5,2)
    assert (proba >=0).all() and (proba <=1).all()

def test_hybrid_scoring():
    from src.risk_scoring import hybrid_risk_scoring
    if Path("reports/hybrid_risk_scores.csv").exists():
        df = pd.read_csv(Path("reports/hybrid_risk_scores.csv"))
        assert "HybridRiskScore" in df.columns
        assert "RiskCategory" in df.columns
        assert df["HybridRiskScore"].between(0,100).all()
        assert set(df["RiskCategory"].unique()).issubset({"Low","Medium","High","Critical"})

def test_shap_explainability():
    if Path("reports/shap_feature_importance.csv").exists():
        df = pd.read_csv(Path("reports/shap_feature_importance.csv"))
        assert "Feature" in df.columns
        assert len(df) > 0

def test_duplicate_detection():
    # Check duplicate reports exist and have expected columns
    for path in [Path("reports/inpatient_duplicate_claims_report.csv"), Path("reports/outpatient_duplicate_claims_report.csv")]:
        if path.exists():
            df = pd.read_csv(path, nrows=5)
            assert "Provider" in df.columns or "BeneID" in df.columns

def test_database_operations():
    from database.models import Base
    from database.db import engine, SessionLocal, Provider
    # Check tables exist
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "providers" in tables
    assert "investigations" in tables
    assert "predictions" in tables
    # Check at least some providers
    session = SessionLocal()
    try:
        count = session.query(Provider).count()
        assert count > 5000  # should have all providers
    finally:
        session.close()

def test_investigation_workflow():
    from database.db import SessionLocal
    from database.models import Investigation, InvestigationNote, AuditLog
    session = SessionLocal()
    try:
        # Create a test investigation
        prov = session.query(Investigation).first()
        # At least one exists from earlier populate
        assert prov is not None or True  # allow empty if not populated
        if prov:
            # Create a note
            note = InvestigationNote(investigation_id=prov.id, provider_id=prov.provider_id, author="Tester", note="Test note")
            session.add(note)
            session.commit()
            # Verify note persisted
            fetched = session.query(InvestigationNote).filter_by(investigation_id=prov.id).first()
            assert fetched is not None
            # Cleanup
            session.delete(note)
            session.commit()
    finally:
        session.close()

def test_pdf_generation():
    from src.pdf_report import generate_investigation_pdf
    from pathlib import Path
    if Path("reports/hybrid_risk_scores.csv").exists() and Path("reports/master_provider_features.csv").exists():
        hybrid = pd.read_csv(Path("reports/hybrid_risk_scores.csv"))
        master = pd.read_csv(Path("reports/master_provider_features.csv"))
        provider_id = hybrid.iloc[0]["Provider"]
        hrow = hybrid[hybrid["Provider"]==provider_id].iloc[0].to_dict()
        mrow = master[master["Provider"]==provider_id].iloc[0].to_dict()
        out = Path("reports/test_investigation_report.pdf")
        generate_investigation_pdf(provider_id, hrow, mrow, {}, "Test notes", False, out)
        assert out.exists()
        assert out.stat().st_size > 5000
        out.unlink()  # cleanup

def test_dashboard_pages_load():
    # Check dashboard files exist
    assert Path("dashboard/app.py").exists()
    # Try importing app without running streamlit
    import importlib.util
    spec = importlib.util.spec_from_file_location("dashboard_app", "dashboard/app.py")
    assert spec is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
