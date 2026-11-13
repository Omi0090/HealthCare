"""
Preprocessing Module - Leakage-safe transformations
Fits transformers on training data only.
Includes robust statistical deviation (median/MAD) and peer-group comparisons.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import joblib

class RobustMADTransformer(BaseEstimator, TransformerMixin):
    """
    Robust scaling using median and MAD (Median Absolute Deviation).
    z = (x - median) / (1.4826 * MAD)
    Handles MAD=0 case gracefully.
    """
    def __init__(self):
        self.medians_ = {}
        self.mads_ = {}
    
    def fit(self, X, y=None):
        for col in X.columns:
            median = X[col].median()
            mad = np.median(np.abs(X[col] - median))
            self.medians_[col] = median
            self.mads_[col] = mad if mad != 0 else 1.0
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in X.columns:
            median = self.medians_[col]
            mad = self.mads_[col]
            scale = 1.4826 * mad
            X_copy[col] = (X[col] - median) / scale
        return X_copy

def handle_chronic_condition_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ChronicCond 1/2 encoding to 0/1. Handles already-converted data."""
    chronic_cols = [c for c in df.columns if c.startswith("ChronicCond_")]
    df = df.copy()
    for col in chronic_cols:
        # If values are {1,2} -> map to {0,1}, if {0,1} leave, otherwise handle
        vals = set(df[col].dropna().unique())
        if vals == {1,2} or vals == {1.0,2.0}:
            df[col] = df[col].map({1:0, 2:1, 1.0:0, 2.0:1})
        elif vals.issubset({0,1,0.0,1.0}):
            pass # already correct
        else:
            # Try numeric conversion then map
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').map({1:0,2:1}).fillna(df[col])
            except:
                pass
    return df

def clean_claim_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and compute claim duration, handling invalids."""
    df = df.copy()
    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if "ClaimStartDt" in df.columns and "ClaimEndDt" in df.columns:
        df["ClaimDuration"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days
        df["ClaimDuration"] = df["ClaimDuration"].clip(lower=0)  # negative durations set to 0, flagged elsewhere
        # Admission duration for inpatient
        if "AdmissionDt" in df.columns and "DischargeDt" in df.columns:
            df["AdmissionDuration"] = (df["DischargeDt"] - df["AdmissionDt"]).dt.days.clip(lower=0)
    return df

def validate_no_leakage(X: pd.DataFrame, y=None) -> None:
    """Ensure no target or post-investigation leakage columns are present."""
    leakage_cols = ["PotentialFraud", "Fraud", "IsFraud", "Investigation", "Excluded"]
    found = [c for c in leakage_cols if c in X.columns]
    if found:
        raise ValueError(f"Potential leakage columns detected in features: {found}")

def create_peer_groups(df: pd.DataFrame, method: str = "quantile", n_groups: int = 5) -> pd.Series:
    """
    Create peer groups based on claim volume to enable fair comparison.
    Groups providers with similar activity levels.
    """
    if "TotalClaims" not in df.columns and "total_claims" in df.columns:
        claim_col = "total_claims"
    elif "TotalClaims" in df.columns:
        claim_col = "TotalClaims"
    else:
        # fallback: use total_claims if exists else create groups from provider count
        return pd.Series(["All"] * len(df), index=df.index)
    
    if method == "quantile":
        try:
            groups = pd.qcut(df[claim_col], q=n_groups, labels=[f"Q{i+1}" for i in range(n_groups)], duplicates='drop')
            return groups.astype(str)
        except:
            return pd.Series(["All"] * len(df), index=df.index)
    elif method == "kmeans":
        from sklearn.cluster import KMeans
        try:
            km = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
            labels = km.fit_predict(df[[claim_col]])
            return pd.Series([f"Cluster_{l}" for l in labels], index=df.index)
        except:
            return pd.Series(["All"] * len(df), index=df.index)
    else:
        return pd.Series(["All"] * len(df), index=df.index)

def compute_peer_deviation(df: pd.DataFrame, feature_cols: List[str], peer_col: str = "PeerGroup") -> pd.DataFrame:
    """
    Compute robust peer-group deviation using median and MAD.
    Adds columns: {feature}_peer_median, {feature}_peer_mad, {feature}_peer_z
    """
    df = df.copy()
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        # Compute peer group stats
        peer_median = df.groupby(peer_col)[feat].transform('median')
        # MAD per group
        def mad(x): return np.median(np.abs(x - np.median(x)))
        peer_mad = df.groupby(peer_col)[feat].transform(lambda x: np.median(np.abs(x - np.median(x))) if len(x) > 1 else 1.0)
        peer_mad = peer_mad.replace(0, 1.0).fillna(1.0)
        df[f"{feat}_peer_median"] = peer_median
        df[f"{feat}_peer_mad"] = peer_mad
        df[f"{feat}_peer_z"] = (df[feat] - peer_median) / (1.4826 * peer_mad)
    return df

def get_feature_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """Get numeric feature columns excluding identifiers and target."""
    if exclude is None:
        exclude = ["Provider", "PotentialFraud", "ProviderID"]
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols

class PreprocessingPipeline:
    """
    Encapsulates fitting and transformation steps.
    Ensures all transformations are fitted on training data only.
    """
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = None  # Optional, created on fit
        self.feature_columns = None
        self.medians = {}
        
    def fit(self, X: pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = numeric_cols
        self.imputer.fit(X[numeric_cols])
        # Store medians for manual imputation tracking
        for col in numeric_cols:
            self.medians[col] = X[col].median()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.feature_columns is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        # Only transform columns seen during fit
        for col in self.feature_columns:
            if col in X.columns:
                # Use stored imputer
                pass
        X_numeric = X[self.feature_columns]
        X_imputed = pd.DataFrame(self.imputer.transform(X_numeric), columns=self.feature_columns, index=X.index)
        return X_imputed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
    
    def save(self, path: Path):
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: Path):
        return joblib.load(path)

if __name__ == "__main__":
    # Quick test
    df = pd.read_csv("reports/master_provider_features.csv")
    print("Original:", df.shape)
    X = df.drop(columns=["Provider", "PotentialFraud"], errors="ignore")
    pipe = PreprocessingPipeline()
    Xt = pipe.fit_transform(X)
    print("Transformed:", Xt.shape)
