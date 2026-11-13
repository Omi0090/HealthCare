"""
Database initialization and operations
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

from database.models import Base, Provider, Claim, Prediction, RiskScore, Investigation, InvestigationNote, ModelVersion, AuditLog

DB_PATH = Path("database/healthcare_fraud.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at {DB_PATH}")

def get_session():
    return SessionLocal()

def populate_from_reports():
    """Populate DB from processed datasets and risk scores."""
    session = get_session()
    try:
        # Clear existing (for re-init)
        # But preserve investigations? For now, wipe and repopulate providers/predictions
        session.query(AuditLog).delete()
        session.query(InvestigationNote).delete()
        session.query(Investigation).delete()
        session.query(RiskScore).delete()
        session.query(Prediction).delete()
        session.query(Provider).delete()
        session.commit()
        
        # Load master data
        master_path = Path("reports/master_provider_features.csv")
        if not master_path.exists():
            master_path = Path("data/processed/cleaned_provider_features.csv")
        if not master_path.exists():
            print(f"No master data at {master_path}")
            return
        
        df = pd.read_csv(master_path)
        # Load hybrid scores
        hybrid_path = Path("reports/hybrid_risk_scores.csv")
        hybrid = pd.read_csv(hybrid_path) if hybrid_path.exists() else None
        
        # Load model metadata
        meta_path = Path("models/model_metadata.json")
        model_version = "v1.0.0"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                model_version = meta.get("model_version", "v1.0.0")
        
        for _, row in df.iterrows():
            provider_id = row.get("Provider") or row.get("provider_id")
            # Get hybrid data if available
            hybrid_score = None
            risk_cat = None
            sup = None
            anom = None
            peer = None
            if hybrid is not None and provider_id in hybrid["Provider"].values:
                hr = hybrid[hybrid["Provider"]==provider_id].iloc[0]
                hybrid_score = float(hr.get("HybridRiskScore", 0))
                risk_cat = hr.get("RiskCategory", "Low")
                sup = float(hr.get("SupervisedScore", 0))
                anom = float(hr.get("AnomalyScore", 0))
                peer = float(hr.get("PeerScore", 0))
            
            provider = Provider(
                provider_id=str(provider_id),
                potential_fraud=str(row.get("PotentialFraud", "")),
                total_claims=int(row.get("TotalClaims", row.get("total_claims", 0)) or 0),
                total_reimbursement=float(row.get("TotalReimbursement", row.get("total_reimbursement", 0)) or 0),
                average_reimbursement=float(row.get("AverageReimbursement", 0) or 0),
                inpatient_claims=int(row.get("InpatientClaims", 0) or 0),
                outpatient_claims=int(row.get("OutpatientClaims", 0) or 0),
                unique_beneficiaries=int(row.get("UniqueBeneficiaries", 0) or 0),
                hybrid_risk_score=hybrid_score,
                risk_category=risk_cat,
                supervised_proba=sup/100 if sup is not None else None,
                anomaly_score=anom,
                peer_score=peer
            )
            session.add(provider)
            
            # Create prediction entry
            if hybrid_score is not None:
                pred = Prediction(
                    provider_id=str(provider_id),
                    model_version=model_version,
                    fraud_probability=sup/100 if sup is not None else 0,
                    hybrid_risk_score=hybrid_score,
                    risk_category=risk_cat,
                    supervised_score=sup,
                    anomaly_score=anom,
                    peer_score=peer,
                    threshold_used=0.5
                )
                session.add(pred)
        
        # Create sample investigations for high-risk providers
        if hybrid is not None:
            high_risk = hybrid[hybrid["RiskCategory"].isin(["High","Critical"])].sort_values("HybridRiskScore", ascending=False).head(5)
            for _, row in high_risk.iterrows():
                inv = Investigation(
                    provider_id=row["Provider"],
                    status="New",
                    priority=row["RiskCategory"],
                    summary=f"Auto-flagged: Hybrid risk {row['HybridRiskScore']:.1f} ({row['RiskCategory']}). Requires review of billing patterns and peer comparison."
                )
                session.add(inv)
        
        # Model version entry
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            mv = ModelVersion(
                version=meta.get("model_version","v1.0.0"),
                model_name=meta.get("best_model","Unknown"),
                training_date=datetime.fromisoformat(meta.get("training_date")) if meta.get("training_date") else datetime.utcnow(),
                metrics=json.dumps(meta.get("test_metrics", {})),
                feature_count=meta.get("feature_count", 0),
                is_active=True
            )
            session.add(mv)
        
        session.commit()
        print(f"Populated DB: {df.shape[0]} providers, {len(session.query(Investigation).all())} investigations")
        
        # Audit log
        audit = AuditLog(action="DB_INIT", entity_type="database", entity_id="healthcare_fraud.db", details=f"Populated {len(df)} providers from {master_path.name}")
        session.add(audit)
        session.commit()
        
    except Exception as e:
        session.rollback()
        print(f"Populate failed: {e}")
        raise
    finally:
        session.close()

def get_provider_count():
    session = get_session()
    try:
        return session.query(Provider).count()
    finally:
        session.close()

if __name__ == "__main__":
    init_db()
    populate_from_reports()
