"""
SQLAlchemy Models for Investigation Workflow
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Provider(Base):
    __tablename__ = "providers"
    provider_id = Column(String, primary_key=True)  # e.g., PRV51001
    potential_fraud = Column(String)  # Yes/No (original label)
    total_claims = Column(Integer)
    total_reimbursement = Column(Float)
    average_reimbursement = Column(Float)
    inpatient_claims = Column(Integer)
    outpatient_claims = Column(Integer)
    unique_beneficiaries = Column(Integer)
    hybrid_risk_score = Column(Float)
    risk_category = Column(String)  # Low/Medium/High/Critical
    supervised_proba = Column(Float)
    anomaly_score = Column(Float)
    peer_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    predictions = relationship("Prediction", back_populates="provider", cascade="all, delete-orphan")
    investigations = relationship("Investigation", back_populates="provider", cascade="all, delete-orphan")

class Claim(Base):
    __tablename__ = "claims"
    claim_id = Column(String, primary_key=True)
    bene_id = Column(String)
    provider_id = Column(String, ForeignKey("providers.provider_id"))
    claim_start_dt = Column(DateTime)
    claim_end_dt = Column(DateTime)
    claim_type = Column(String)  # Inpatient/Outpatient
    reimbursed = Column(Float)
    deductible = Column(Float)
    diagnosis_code = Column(String)
    procedure_code = Column(String)
    attending_physician = Column(String)
    
    provider = relationship("Provider", backref="claims")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String, ForeignKey("providers.provider_id"))
    model_version = Column(String)
    fraud_probability = Column(Float)
    hybrid_risk_score = Column(Float)
    risk_category = Column(String)
    supervised_score = Column(Float)
    anomaly_score = Column(Float)
    peer_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    threshold_used = Column(Float)
    
    provider = relationship("Provider", back_populates="predictions")

class RiskScore(Base):
    __tablename__ = "risk_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String, ForeignKey("providers.provider_id"))
    hybrid_score = Column(Float)
    supervised_component = Column(Float)
    anomaly_component = Column(Float)
    peer_component = Column(Float)
    weights = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

class Investigation(Base):
    __tablename__ = "investigations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String, ForeignKey("providers.provider_id"))
    status = Column(String, default="New")  # New, Under Review, Escalated, Closed – No Concern, Closed – Requires Further Review
    assigned_to = Column(String, default="Unassigned")
    priority = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    summary = Column(Text)
    
    provider = relationship("Provider", back_populates="investigations")
    notes = relationship("InvestigationNote", back_populates="investigation", cascade="all, delete-orphan")

class InvestigationNote(Base):
    __tablename__ = "investigation_notes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    investigation_id = Column(Integer, ForeignKey("investigations.id"))
    provider_id = Column(String)
    author = Column(String, default="Investigator")
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    investigation = relationship("Investigation", back_populates="notes")

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String)
    model_name = Column(String)
    training_date = Column(DateTime)
    metrics = Column(Text)  # JSON
    feature_count = Column(Integer)
    is_active = Column(Boolean, default=True)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String)
    entity_type = Column(String)
    entity_id = Column(String)
    user = Column(String, default="system")
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

