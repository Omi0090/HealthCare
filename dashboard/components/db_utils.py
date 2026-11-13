"""
DB utilities for dashboard
"""
import pandas as pd
from pathlib import Path
import sqlite3

DB_PATH = Path("database/healthcare_fraud.db")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

def get_db_connection():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def load_providers():
    conn = get_db_connection()
    if not conn:
        # fallback to CSV
        for p in [Path("reports/master_provider_features.csv"), Path("data/processed/cleaned_provider_features.csv")]:
            if p.exists():
                return pd.read_csv(p)
        return pd.DataFrame()
    try:
        df = pd.read_sql_query("SELECT * FROM providers", conn)
        if df.empty:
            # fallback
            for p in [Path("reports/master_provider_features.csv"), Path("data/processed/cleaned_provider_features.csv")]:
                if p.exists():
                    df = pd.read_csv(p)
                    break
        return df
    except:
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def load_hybrid_scores():
    path = REPORTS_DIR / "hybrid_risk_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def load_investigations():
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query("SELECT * FROM investigations", conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

def get_provider_detail(provider_id):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM providers WHERE provider_id=?", (provider_id,))
        row = cur.fetchone()
        if row:
            return dict(row)
        return None
    except:
        return None
    finally:
        conn.close()

def get_claims_for_provider(provider_id, limit=100):
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM claims WHERE provider_id=? LIMIT ?", conn, params=(provider_id, limit))
            if not df.empty:
                return df
        except:
            pass
        finally:
            conn.close()
    # Fallback: load raw claims and filter
    raw_files = list(Path("data/raw").glob("Train_Inpatientdata-*.csv")) + list(Path("data/raw").glob("Train_Outpatientdata-*.csv"))
    frames = []
    for f in raw_files[:2]:
        try:
            df = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ["Provider","ClaimID","BeneID","ClaimStartDt","ClaimEndDt","InscClaimAmtReimbursed","DeductibleAmtPaid","ClmDiagnosisCode_1","ClmProcedureCode_1"])
            df = df[df["Provider"]==provider_id].head(limit)
            frames.append(df)
        except:
            pass
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()
