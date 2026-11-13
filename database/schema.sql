-- SQLite schema for Healthcare Fraud Detection
-- This file documents the schema; SQLAlchemy creates tables via models.py

-- Providers: aggregated provider-level features + labels
CREATE TABLE IF NOT EXISTS providers (
    provider_id TEXT PRIMARY KEY,
    potential_fraud TEXT,
    total_claims INTEGER,
    total_reimbursement REAL,
    average_reimbursement REAL,
    inpatient_claims INTEGER,
    outpatient_claims INTEGER,
    unique_beneficiaries INTEGER,
    hybrid_risk_score REAL,
    risk_category TEXT,
    supervised_proba REAL,
    anomaly_score REAL,
    peer_score REAL,
    created_at TIMESTAMP
);

-- Claims: raw claim-level data (optional, for drill-down)
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    bene_id TEXT,
    provider_id TEXT REFERENCES providers(provider_id),
    claim_start_dt TIMESTAMP,
    claim_end_dt TIMESTAMP,
    claim_type TEXT,
    reimbursed REAL,
    deductible REAL,
    diagnosis_code TEXT,
    procedure_code TEXT,
    attending_physician TEXT
);

-- Predictions: history of model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT REFERENCES providers(provider_id),
    model_version TEXT,
    fraud_probability REAL,
    hybrid_risk_score REAL,
    risk_category TEXT,
    supervised_score REAL,
    anomaly_score REAL,
    peer_score REAL,
    created_at TIMESTAMP,
    threshold_used REAL
);

-- Risk scores: components for transparency
CREATE TABLE IF NOT EXISTS risk_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT REFERENCES providers(provider_id),
    hybrid_score REAL,
    supervised_component REAL,
    anomaly_component REAL,
    peer_component REAL,
    weights TEXT,
    created_at TIMESTAMP
);

-- Investigations: workflow
CREATE TABLE IF NOT EXISTS investigations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT REFERENCES providers(provider_id),
    status TEXT CHECK(status IN ('New','Under Review','Escalated','Closed – No Concern','Closed – Requires Further Review')),
    assigned_to TEXT,
    priority TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    summary TEXT
);

-- Investigation notes
CREATE TABLE IF NOT EXISTS investigation_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER REFERENCES investigations(id),
    provider_id TEXT,
    author TEXT,
    note TEXT,
    created_at TIMESTAMP
);

-- Model versions
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT,
    model_name TEXT,
    training_date TIMESTAMP,
    metrics TEXT,
    feature_count INTEGER,
    is_active BOOLEAN
);

-- Audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT,
    entity_type TEXT,
    entity_id TEXT,
    user TEXT,
    details TEXT,
    timestamp TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_provider_risk ON providers(risk_category);
CREATE INDEX IF NOT EXISTS idx_provider_score ON providers(hybrid_risk_score);
CREATE INDEX IF NOT EXISTS idx_investigation_provider ON investigations(provider_id);
CREATE INDEX IF NOT EXISTS idx_investigation_status ON investigations(status);
