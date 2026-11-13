"""
AI-Based Healthcare Insurance Fraud Detection - Main Dashboard
Executive, enterprise-style Streamlit application.
Covers 12 pages as per project specification.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
import sqlite3

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dashboard.components.ui import kpi_card, risk_badge, disclaimer_box, caveat_box, plotly_chart, styled_dataframe
from dashboard.components.db_utils import load_providers, load_hybrid_scores, load_investigations, get_provider_detail

st.set_page_config(
    page_title="Healthcare Fraud Detection Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enterprise look
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio > div { background: rgba(255,255,255,0.05); border-radius: 8px; padding: 8px; }
    h1, h2, h3 { color: #0f172a; font-weight: 700; }
    .stMetric { background: white; padding: 16px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 6px 0;">
        <div style="font-size: 22px; font-weight: 800; color: white; letter-spacing: -0.02em;">🏥 FraudShield AI</div>
        <div style="font-size: 11px; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin-top:4px;">Healthcare Fraud Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "1. Executive Overview",
            "2. Provider Explorer",
            "3. Fraud Analytics",
            "4. AI Predictions",
            "5. Explainable AI",
            "6. Anomaly Detection",
            "7. Network Analysis",
            "8. Investigation Centre",
            "9. Model Performance",
            "10. Model Monitoring",
            "11. Reports",
            "12. Settings"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("ℹ️ High-risk = investigation signal, not proof of fraud")
    st.caption("US Medicare benchmark • Not validated for other schemes")
    if st.button("🔄 Refresh Data"):
        st.rerun()

# Load core data (cached)
@st.cache_data(show_spinner=False, ttl=60)
def load_core():
    providers = load_providers()
    hybrid = load_hybrid_scores()
    investigations = load_investigations()
    # master for fallback
    master = None
    for p in [Path("reports/master_provider_features.csv"), Path("data/processed/cleaned_provider_features.csv")]:
        if p.exists():
            master = pd.read_csv(p)
            break
    return providers, hybrid, investigations, master

providers, hybrid, investigations, master = load_core()

# Try loading model metadata
meta = {}
meta_path = Path("models/model_metadata.json")
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)

# Helper: get provider display df
def get_providers_enriched():
    # Prefer hybrid enriched
    if hybrid is not None and not hybrid.empty:
        # Merge with master for extra fields if needed
        if master is not None:
            # hybrid already has provider, scores; merge some master cols for display
            extra_cols = ["TotalClaims","TotalReimbursement","AverageReimbursement","UniqueBeneficiaries","InpatientClaims","OutpatientClaims","PotentialFraud"] if master is not None else []
            cols_to_merge = [c for c in extra_cols if c in master.columns]
            if cols_to_merge:
                return hybrid.merge(master[["Provider"]+cols_to_merge], on="Provider", how="left")
        return hybrid
    elif master is not None:
        return master
    elif providers is not None and not providers.empty:
        return providers
    else:
        return pd.DataFrame()

enriched = get_providers_enriched()

# ================= PAGE 1: Executive Overview =================
if page == "1. Executive Overview":
    st.title("Executive Overview")
    st.caption("AI-powered fraud screening • Human-in-the-loop investigation • Transparent risk scoring")
    caveat_box()
    disclaimer_box()
    
    if enriched.empty:
        st.error("No provider data found. Run src/feature_engineering.py and src/train.py to generate datasets.")
        st.stop()
    
    # KPIs
    total_providers = len(enriched)
    # Try columns
    total_claims = 0
    if "TotalClaims" in enriched.columns:
        total_claims = enriched["TotalClaims"].sum()
    elif "total_claims" in enriched.columns:
        total_claims = enriched["total_claims"].sum()
    
    high_risk = len(enriched[enriched["RiskCategory"].isin(["High","Critical"])]) if "RiskCategory" in enriched.columns else 0
    critical = len(enriched[enriched["RiskCategory"]=="Critical"]) if "RiskCategory" in enriched.columns else 0
    fraud_labelled = 0
    if "PotentialFraud" in enriched.columns:
        fraud_labelled = (enriched["PotentialFraud"]=="Yes").sum() if enriched["PotentialFraud"].dtype==object else (enriched["PotentialFraud"]==1).sum()
    avg_risk = enriched["HybridRiskScore"].mean() if "HybridRiskScore" in enriched.columns else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        kpi_card("Total Providers", f"{total_providers:,}", help_text="Unique healthcare providers in training set")
    with col2:
        kpi_card("Total Claims", f"{int(total_claims):,}" if total_claims else "N/A", help_text="Inpatient + outpatient claims analyzed")
    with col3:
        kpi_card("High-Risk Providers", f"{high_risk:,}", delta=f"Critical: {critical}", help_text="Risk ≥60 (High/Critical)")
    with col4:
        kpi_card("Fraud-Labelled", f"{fraud_labelled:,}", delta=f"{fraud_labelled/total_providers*100:.1f}% prevalence", help_text="Provider-level labels (train)")
    with col5:
        kpi_card("Avg Risk Score", f"{avg_risk:.1f}/100", help_text="Hybrid score: supervised+anomaly+peer")
    
    st.markdown("### Risk Category Distribution")
    if "RiskCategory" in enriched.columns:
        dist = enriched["RiskCategory"].value_counts().reindex(["Low","Medium","High","Critical"], fill_value=0)
        cA, cB = st.columns([2,3])
        with cA:
            fig = px.pie(values=dist.values, names=dist.index, hole=0.5, color=dist.index,
                         color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444","Critical":"#991b1b"},
                         title="Risk Categories")
            plotly_chart(fig, height=350)
        with cB:
            fig2 = px.bar(x=dist.index, y=dist.values, color=dist.index, 
                         color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444","Critical":"#991b1b"},
                         labels={"x":"Risk Category","y":"Provider Count"}, title="Providers by Risk Level")
            fig2.update_layout(showlegend=False)
            plotly_chart(fig2, height=350)
    else:
        st.warning("Hybrid risk scores not yet generated. Run src/risk_scoring.py")
    
    # Trends: monthly claims? Show reimbursement vs claims
    st.markdown("### Suspicious Activity Trends")
    if master is not None and "TotalClaims" in master.columns:
        col1, col2 = st.columns(2)
        with col1:
            # Reimbursement distribution fraud vs non-fraud
            if "PotentialFraud" in master.columns:
                fig = px.box(master, x="PotentialFraud", y="TotalReimbursement", color="PotentialFraud",
                             title="Total Reimbursement by Fraud Label", points="outliers")
                fig.update_layout(yaxis_type="log")
                plotly_chart(fig)
            else:
                st.info("No fraud label for trend")
        with col2:
            # Claims vs reimbursement scatter colored by risk
            if "HybridRiskScore" in enriched.columns and "TotalClaims" in enriched.columns:
                fig = px.scatter(enriched, x="TotalClaims", y="TotalReimbursement", color="RiskCategory",
                                 color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444","Critical":"#991b1b"},
                                 hover_name="Provider", title="Claims vs Reimbursement (Risk-colored)", opacity=0.7)
                fig.update_layout(yaxis_type="log")
                plotly_chart(fig)
    
    # Model KPIs
    st.markdown("### Model Performance KPIs")
    if meta:
        c1,c2,c3,c4 = st.columns(4)
        tm = meta.get("test_metrics", {})
        with c1:
            kpi_card("PR-AUC (Primary)", f"{tm.get('pr_auc',0):.3f}", help_text="Precision-Recall AUC (rare positives)")
        with c2:
            kpi_card("ROC-AUC", f"{tm.get('roc_auc',0):.3f}")
        with c3:
            kpi_card("Recall", f"{tm.get('recall',0):.3f}", help_text="Fraud detection sensitivity")
        with c4:
            kpi_card("F1-Score", f"{tm.get('f1',0):.3f}")
        st.caption(f"Model: {meta.get('best_model')} • Version: {meta.get('model_version')} • Trained: {meta.get('training_date')}")
    else:
        st.info("Model metadata not found. Train models to see performance.")

# ================= PAGE 2: Provider Explorer =================
elif page == "2. Provider Explorer":
    st.title("Provider Explorer")
    st.caption("Search, filter, and inspect provider profiles • Evidence-based review")
    if enriched.empty:
        st.error("No provider data available")
        st.stop()
    
    # Filters
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            search = st.text_input("🔍 Search Provider ID", placeholder="e.g., PRV51001")
        with col2:
            risk_filter = st.multiselect("Risk Category", ["Low","Medium","High","Critical"], default=[])
        with col3:
            fraud_filter = st.selectbox("Fraud Label", ["All","Yes","No"])
        with col4:
            sort_by = st.selectbox("Sort by", ["HybridRiskScore","TotalReimbursement","TotalClaims","AverageReimbursement"])
    
    df = enriched.copy()
    if search:
        df = df[df["Provider"].astype(str).str.contains(search, case=False, na=False)]
    if risk_filter:
        df = df[df["RiskCategory"].isin(risk_filter)]
    if fraud_filter != "All" and "PotentialFraud" in df.columns:
        df = df[df["PotentialFraud"].astype(str)==fraud_filter]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    st.markdown(f"**Showing {len(df):,} providers** (total {len(enriched):,})")
    # Table
    display_cols = [c for c in ["Provider","RiskCategory","HybridRiskScore","SupervisedScore","AnomalyScore","PeerScore","TotalClaims","TotalReimbursement","AverageReimbursement","PotentialFraud"] if c in df.columns]
    if display_cols:
        # Format scores
        st.dataframe(df[display_cols].head(500), use_container_width=True, height=400)
    
    # Provider profile
    st.markdown("---")
    st.subheader("Provider Profile")
    provider_ids = df["Provider"].astype(str).tolist() if "Provider" in df.columns else []
    if provider_ids:
        selected = st.selectbox("Select Provider for detailed view", provider_ids[:200], key="provider_select")
        if selected:
            detail = get_provider_detail(selected)
            # Also from enriched
            row = enriched[enriched["Provider"]==selected].iloc[0] if selected in enriched["Provider"].values else None
            if row is not None:
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("Hybrid Risk Score", f"{row.get('HybridRiskScore',0):.1f}")
                    if "RiskCategory" in row:
                        risk_badge(row["RiskCategory"])
                with c2:
                    st.metric("Fraud Probability", f"{row.get('SupervisedScore',0)/100:.1%}" if "SupervisedScore" in row else "N/A")
                    st.metric("PotentialFraud Label", str(row.get("PotentialFraud","Unknown")))
                with c3:
                    st.metric("Total Claims", f"{int(row.get('TotalClaims',0)):,}")
                    st.metric("Unique Beneficiaries", f"{int(row.get('UniqueBeneficiaries',0)):,}" if "UniqueBeneficiaries" in row else "N/A")
                with c4:
                    st.metric("Total Reimbursement", f"${row.get('TotalReimbursement',0):,.0f}" if "TotalReimbursement" in row else "N/A")
                    st.metric("Avg Reimbursement", f"${row.get('AverageReimbursement',0):,.0f}" if "AverageReimbursement" in row else "N/A")
                
                # Historical info: show provider vs peer
                if "PeerGroup" in row or "PeerGroup" in enriched.columns:
                    st.markdown("**Peer Group Context**")
                    # Show peer median comparison
                    peer_z_cols = [c for c in enriched.columns if c.endswith("_PeerZ")]
                    if peer_z_cols:
                        peer_info = {}
                        for c in peer_z_cols[:5]:
                            feat = c.replace("_PeerZ","")
                            peer_info[feat] = {"Value": row.get(feat), "Z": row.get(c), "Median": row.get(feat+"_PeerMedian")}
                        st.json(peer_info)
                
                # Claims table
                st.markdown("**Recent Claims (sample)**")
                claims_df = get_claims_for_provider(selected, limit=20) if 'get_claims_for_provider' in globals() else pd.DataFrame()
                # fallback to db_utils
                try:
                    from dashboard.components.db_utils import get_claims_for_provider as gcp
                    claims_df = gcp(selected, limit=20)
                except:
                    pass
                if not claims_df.empty:
                    st.dataframe(claims_df, use_container_width=True)
                else:
                    st.info("No claim drill-down available. Check data/raw claims for this provider exists.")
                
                # Action buttons
                colA, colB = st.columns(2)
                with colA:
                    if st.button("🔍 Explain this Provider (SHAP)"):
                        st.session_state['explain_provider'] = selected
                        st.info("Go to Explainable AI page and select this provider")
                with colB:
                    if st.button("📄 Generate Investigation Report (PDF)"):
                        st.session_state['report_provider'] = selected
                        st.info("Go to Reports page to download PDF")
            else:
                st.warning("Provider not found in enriched view")

# ================= PAGE 3: Fraud Analytics =================
elif page == "3. Fraud Analytics":
    st.title("Fraud Analytics")
    st.caption("Distribution, billing patterns, reimbursement analysis, provider comparisons")
    if master is None or master.empty:
        st.error("Master provider features not found. Run feature engineering.")
        st.stop()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Fraud Distribution","Billing & Reimbursement","Provider Comparison","Duplicate Patterns"])
    
    with tab1:
        # Fraud distribution
        if "PotentialFraud" in master.columns:
            dist = master["PotentialFraud"].value_counts()
            dist_norm = master["PotentialFraud"].value_counts(normalize=True)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=dist.values, names=dist.index, hole=0.4, title="Fraud Label Distribution")
                plotly_chart(fig)
            with col2:
                st.metric("Fraud Prevalence", f"{dist_norm.get('Yes',0)*100:.2f}%" if "Yes" in dist_norm else f"{dist_norm.get(1,0)*100:.2f}%" if 1 in dist_norm else "N/A")
                st.dataframe(pd.DataFrame({"Count":dist, "Percent":dist_norm*100}), use_container_width=True)
        
        # Claim distribution
        colA, colB = st.columns(2)
        with colA:
            fig = px.histogram(master, x="TotalClaims", color="PotentialFraud" if "PotentialFraud" in master.columns else None, nbins=50, title="Claim Count Distribution (log scale)", barmode="overlay", opacity=0.7)
            fig.update_layout(xaxis_type="log")
            plotly_chart(fig)
        with colB:
            fig = px.histogram(master, x="UniqueBeneficiaries", color="PotentialFraud" if "PotentialFraud" in master.columns else None, nbins=50, title="Unique Beneficiaries Distribution", barmode="overlay")
            plotly_chart(fig)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(master, x="PotentialFraud" if "PotentialFraud" in master.columns else None, y="AverageReimbursement", color="PotentialFraud" if "PotentialFraud" in master.columns else None, title="Avg Reimbursement by Fraud")
            plotly_chart(fig)
        with col2:
            fig = px.box(master, x="PotentialFraud" if "PotentialFraud" in master.columns else None, y="TotalReimbursement", color="PotentialFraud" if "PotentialFraud" in master.columns else None, title="Total Reimbursement by Fraud")
            fig.update_layout(yaxis_type="log")
            plotly_chart(fig)
        # Inpatient vs outpatient
        if "InpatientClaims" in master.columns and "OutpatientClaims" in master.columns:
            fig = px.scatter(master, x="InpatientClaims", y="OutpatientClaims", color="PotentialFraud" if "PotentialFraud" in master.columns else None, hover_name="Provider", title="Inpatient vs Outpatient Claims")
            plotly_chart(fig)
        # Beneficiary demographics
        if "AveragePatientAge" in master.columns:
            fig = px.histogram(master, x="AveragePatientAge", color="PotentialFraud" if "PotentialFraud" in master.columns else None, nbins=40, title="Average Patient Age by Fraud", barmode="overlay")
            plotly_chart(fig)
        # Chronic conditions heatmap? Show avg
        chronic_cols = [c for c in master.columns if c.startswith("Avg_ChronicCond_")]
        if chronic_cols:
            st.markdown("**Chronic Condition Prevalence (Fraud vs Non-Fraud)**")
            fraud_means = master.groupby("PotentialFraud" if "PotentialFraud" in master.columns else master.columns[0])[chronic_cols].mean().T
            fig = px.imshow(fraud_means, color_continuous_scale="Blues", title="Avg Chronic Condition Rate")
            plotly_chart(fig)
    
    with tab3:
        # Provider comparison: select two providers
        st.markdown("**Provider Comparison Tool**")
        p_ids = master["Provider"].tolist()
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.selectbox("Provider A", p_ids[:100], key="comp_a")
        with c2:
            p2 = st.selectbox("Provider B", p_ids[100:200] if len(p_ids)>200 else p_ids[:100], key="comp_b")
        if p1 and p2:
            row1 = master[master["Provider"]==p1].iloc[0]
            row2 = master[master["Provider"]==p2].iloc[0]
            comp_df = pd.DataFrame({"Metric": ["TotalClaims","TotalReimbursement","AverageReimbursement","UniqueBeneficiaries","AverageClaimDuration"], 
                                    p1: [row1.get("TotalClaims"), row1.get("TotalReimbursement"), row1.get("AverageReimbursement"), row1.get("UniqueBeneficiaries"), row1.get("AverageClaimDuration")],
                                    p2: [row2.get("TotalClaims"), row2.get("TotalReimbursement"), row2.get("AverageReimbursement"), row2.get("UniqueBeneficiaries"), row2.get("AverageClaimDuration")]})
            st.dataframe(comp_df, use_container_width=True)
            # Radar? simple bar
            fig = go.Figure()
            fig.add_trace(go.Bar(name=p1, x=comp_df["Metric"], y=comp_df[p1]))
            fig.add_trace(go.Bar(name=p2, x=comp_df["Metric"], y=comp_df[p2]))
            fig.update_layout(title=f"Comparison: {p1} vs {p2}", barmode="group")
            plotly_chart(fig)
    
    with tab4:
        st.markdown("**Duplicate Claim Patterns**")
        dup_summary = Path("reports/duplicate_claims_summary.csv")
        dup_inp = Path("reports/inpatient_duplicate_claims_report.csv")
        dup_out = Path("reports/outpatient_duplicate_claims_report.csv")
        if dup_summary.exists():
            st.dataframe(pd.read_csv(dup_summary), use_container_width=True)
        else:
            st.info("Duplicate summary not found. Run python scripts/detect_duplicate_claims.py")
        col1, col2 = st.columns(2)
        with col1:
            if dup_inp.exists():
                df = pd.read_csv(dup_inp)
                st.markdown("**Inpatient Duplicates (sample)**")
                st.dataframe(df.head(100), use_container_width=True)
                st.caption(f"Total flagged inpatient: {len(df):,}")
        with col2:
            if dup_out.exists():
                df = pd.read_csv(dup_out)
                st.markdown("**Outpatient Duplicates (sample)**")
                st.dataframe(df.head(100), use_container_width=True)
                st.caption(f"Total flagged outpatient: {len(df):,}")
        st.warning("Duplicate flags are screening signals for review; corrections/resubmissions can appear as duplicates.")

# ================= PAGE 4: AI Predictions =================
elif page == "4. AI Predictions":
    st.title("AI Predictions")
    st.caption("Generate fraud probability + hybrid risk score on demand • Persisted history")
    disclaimer_box()
    
    if enriched.empty:
        st.error("No data")
        st.stop()
    
    provider_list = enriched["Provider"].astype(str).tolist()
    selected = st.selectbox("Select Provider to Predict", provider_list[:500])
    
    if selected:
        # Show current prediction if available
        row = enriched[enriched["Provider"]==selected].iloc[0] if selected in enriched["Provider"].values else None
        if row is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                proba = row.get("SupervisedScore", 0) /100 if "SupervisedScore" in row else row.get("SupervisedProba", 0) if "SupervisedProba" in row else 0
                st.metric("Fraud Probability", f"{proba:.3f}")
                st.progress(float(proba))
            with col2:
                st.metric("Hybrid Risk Score", f"{row.get('HybridRiskScore',0):.1f}/100")
                if "RiskCategory" in row:
                    risk_badge(row["RiskCategory"])
            with col3:
                if "PotentialFraud" in row:
                    st.metric("Ground Truth Label", str(row["PotentialFraud"]))
                st.metric("Model Version", meta.get("model_version","v1.0.0") if meta else "v1.0.0")
            
            # Components
            st.markdown("**Scoring Components (Transparent)**")
            c1,c2,c3 = st.columns(3)
            with c1:
                st.metric("Supervised (55%)", f"{row.get('SupervisedScore',0):.1f}", help_text="Calibrated fraud probability from Random Forest/XGBoost")
            with c2:
                st.metric("Anomaly (25%)", f"{row.get('AnomalyScore',0):.1f}", help_text="Isolation Forest normalized 0-100")
            with c3:
                st.metric("Peer Deviation (20%)", f"{row.get('PeerScore',0):.1f}", help_text="Robust MAD peer z-score severity")
            
            with st.expander("Scoring Methodology (Transparent)"):
                st.markdown("""
                **Hybrid = 0.55×Supervised + 0.25×Anomaly + 0.20×Peer**
                - **Supervised**: Best validated model (currently {}), trained with class-imbalance handling, leakage-safe splits.
                - **Anomaly**: Isolation Forest (contamination 0.1), unsupervised, catches label-blind unusual behavior.
                - **Peer**: Median/MAD robust deviation within claim-volume peer groups; severity-scaled.
                Weights chosen via validation PR-AUC trade-off; configurable in Settings.
                Thresholds: Low 0-30, Medium 30-60, High 60-80, Critical 80-100.
                """.format(meta.get("best_model","RandomForest") if meta else "RandomForest"))
                st.json({"weights": {"supervised":0.55,"anomaly":0.25,"peer":0.20}, "thresholds": {"Low":(0,30),"Medium":(30,60),"High":(60,80),"Critical":(80,100)}})
            
            if st.button("🔁 Regenerate Prediction (Live Inference)"):
                try:
                    import joblib
                    from pathlib import Path
                    model = joblib.load(Path("models/best_fraud_model.pkl"))
                    feature_cols = joblib.load(Path("models/feature_columns.pkl"))
                    # Load current provider features from master
                    master_path = Path("reports/master_provider_features.csv")
                    if master_path.exists():
                        df = pd.read_csv(master_path)
                        crow = df[df["Provider"]==selected]
                        if not crow.empty:
                            X = crow[feature_cols].fillna(crow[feature_cols].median())
                            proba_live = model.predict_proba(X)[0,1]
                            st.success(f"Live inference fraud probability: {proba_live:.4f} (matches stored {proba:.4f})")
                        else:
                            st.error("Provider not in master")
                    else:
                        st.error("Master not found")
                except Exception as e:
                    st.error(f"Inference failed: {e}")
    
    # Prediction history
    st.markdown("### Prediction History")
    conn = None
    try:
        conn = sqlite3.connect("database/healthcare_fraud.db")
        hist = pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 100", conn)
        if not hist.empty:
            st.dataframe(hist, use_container_width=True)
        else:
            st.info("No prediction history in DB yet. Predictions from CSV are shown above.")
    except:
        st.info("No DB history available")
    finally:
        if conn:
            conn.close()
    
    # Bulk scoring
    st.markdown("### Bulk Scoring")
    if st.button("📊 View All Providers Risk Ranking"):
        if "HybridRiskScore" in enriched.columns:
            ranked = enriched.sort_values("HybridRiskScore", ascending=False)[["Provider","HybridRiskScore","RiskCategory","SupervisedScore","AnomalyScore","PeerScore","PotentialFraud"] if "PotentialFraud" in enriched.columns else ["Provider","HybridRiskScore","RiskCategory"]]
            st.dataframe(ranked.head(1000), use_container_width=True, height=500)
            csv = ranked.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "risk_ranking.csv", "text/csv")

# ================= PAGE 5: Explainable AI =================
elif page == "5. Explainable AI":
    st.title("Explainable AI")
    st.caption("SHAP-based global and individual explanations • Not causal proof")
    disclaimer_box("SHAP shows feature contribution to model output, not causality or guilt.")
    
    # Global importance
    shap_imp = Path("reports/shap_feature_importance.csv")
    if shap_imp.exists():
        imp = pd.read_csv(shap_imp)
        st.markdown("### Global Feature Importance (Mean |SHAP|)")
        c1, c2 = st.columns([2,3])
        with c1:
            st.dataframe(imp.head(15), use_container_width=True)
        with c2:
            fig = px.bar(imp.head(15).sort_values("MeanAbsSHAP"), x="MeanAbsSHAP", y="Feature", orientation="h", title="Top 15 Features by SHAP Importance")
            plotly_chart(fig)
        # Images if exist
        for img in ["reports/shap_summary.png","reports/shap_feature_importance.png","reports/shap_individual_explanation.png"]:
            p = Path(img)
            if p.exists():
                st.image(str(p), caption=p.name, use_column_width=True)
    else:
        st.warning("SHAP artifacts not found. Run: python src/explainability.py")
        # Fallback to feature_importance
        fi = Path("reports/feature_importance.csv")
        if fi.exists():
            st.dataframe(pd.read_csv(fi), use_container_width=True)
    
    # Individual explanation
    st.markdown("---")
    st.subheader("Individual Provider Explanation")
    provider_list = enriched["Provider"].astype(str).tolist() if not enriched.empty else []
    sel = st.selectbox("Select Provider for SHAP Waterfall", provider_list[:300], key="shap_provider")
    if sel and st.button("Generate SHAP Explanation"):
        with st.spinner("Computing SHAP (may take 10-20s)..."):
            try:
                import sys
                sys.path.append(str(Path.cwd()))
                from src.explainability import explain_provider
                result = explain_provider(sel, save_plot=True)
                if "error" in result:
                    st.error(result["error"])
                else:
                    col1,col2,col3 = st.columns(3)
                    with col1:
                        st.metric("Fraud Probability", f"{result['PredictedFraudProbability']:.3f}")
                    with col2:
                        st.metric("Hybrid Risk", f"{result.get('HybridRiskScore',0):.1f}" if result.get('HybridRiskScore') else "N/A")
                        if result.get('RiskCategory'):
                            risk_badge(result['RiskCategory'])
                    with col3:
                        st.metric("Expected Value", f"{result.get('ExpectedValue',0):.3f}" if result.get('ExpectedValue') else "N/A")
                    
                    # Contributions
                    st.markdown("**Top Contributing Features**")
                    contrib = pd.DataFrame(result["TopContributingFeatures"])
                    st.dataframe(contrib, use_container_width=True)
                    
                    # Show positive vs negative
                    cA, cB = st.columns(2)
                    with cA:
                        st.markdown("🔴 **Increases Fraud Probability**")
                        pos = pd.DataFrame(result["TopPositive"])
                        if not pos.empty:
                            fig = px.bar(pos, x="SHAPValue", y="Feature", orientation="h", title="Positive Contributions")
                            plotly_chart(fig)
                            st.dataframe(pos, use_container_width=True)
                        else:
                            st.info("No positive contributions")
                    with cB:
                        st.markdown("🔵 **Decreases Fraud Probability**")
                        neg = pd.DataFrame(result["TopNegative"])
                        if not neg.empty:
                            fig = px.bar(neg, x="SHAPValue", y="Feature", orientation="h", title="Negative Contributions")
                            plotly_chart(fig)
                            st.dataframe(neg, use_container_width=True)
                        else:
                            st.info("No negative contributions")
                    
                    # Waterfall image
                    wf = Path(f"reports/shap_waterfall_{sel}.png")
                    if wf.exists():
                        st.image(str(wf), caption=f"SHAP Waterfall for {sel}")
                    st.success(result["Disclaimer"])
            except Exception as e:
                st.error(f"SHAP failed: {e}. Ensure shap is installed (pip install shap)")
    
    # Counterfactual
    st.markdown("---")
    st.subheader("What-If / Counterfactual Analysis")
    st.caption("Adjust hypothetical feature values to see prediction change • Clearly marked as simulated")
    if sel:
        st.markdown(f"**Simulating changes for {sel}**")
        # Get current values
        if sel in enriched["Provider"].values:
            crow = enriched[enriched["Provider"]==sel].iloc[0]
            # Pick adjustable features
            adjustable = [c for c in ["TotalReimbursement","AverageReimbursement","TotalClaims","DuplicateClaimRatio","AverageClaimDuration"] if c in enriched.columns]
            cols = st.columns(len(adjustable))
            adjustments = {}
            for i, feat in enumerate(adjustable):
                with cols[i]:
                    curr = float(crow[feat])
                    new_val = st.number_input(f"{feat}", value=curr, key=f"whatif_{feat}")
                    if new_val != curr:
                        adjustments[feat] = new_val
            if adjustments and st.button("Run What-If Simulation"):
                try:
                    from src.explainability import what_if_analysis
                    res = what_if_analysis(sel, adjustments)
                    st.json(res)
                    st.warning(res["Warning"])
                    # Show delta visually
                    fig = go.Figure(go.Bar(x=["Original","Simulated"], y=[res["OriginalProba"], res["NewProba"]], marker_color=["#6b7280","#ef4444"]))
                    fig.update_layout(title=f"What-If: Fraud Prob {res['OriginalProba']:.3f} → {res['NewProba']:.3f} (Δ={res['Delta']:+.3f})")
                    plotly_chart(fig)
                except Exception as e:
                    st.error(f"What-if failed: {e}")
        else:
            st.info("Select provider above to run what-if")

    # Fraud Investigation Copilot (Rule-based, grounded)
    st.markdown("---")
    st.subheader("🤖 Fraud Investigation Copilot (Local, Grounded)")
    st.caption("Ask about the selected provider; answers use actual DB, predictions, SHAP - no fabrication")
    if sel:
        example_q = st.selectbox("Example questions", [""] + ["Why was this provider assigned a high-risk score?","What features contributed most to the fraud prediction?","Show peer-group comparison for this provider.","Summarize billing patterns and duplicate claims.","Are there temporal spikes in claim volume?","Does this provider share beneficiaries with others?","What is the fraud probability vs hybrid score?"], key="copilot_example")
        user_q = st.text_input("Your question about "+sel, value=example_q, placeholder="e.g., Why was this provider assigned a high-risk score?")
        if user_q and st.button("Ask Copilot", key="ask_copilot"):
            try:
                from src.copilot import answer_question
                result = answer_question(sel, user_q)
                st.markdown(f"**Answer:**\n{result['answer']}")
                with st.expander("Sources (grounded)"):
                    for s in result['sources']:
                        st.markdown(f"- {s}")
                st.caption(f"Grounded: {result['grounded']} • Context used: {', '.join(result['context_used'])}")
            except Exception as e:
                st.error(f"Copilot failed: {e}")
    else:
        st.info("Select a provider above to use the Copilot")

    # Provider Risk Timeline
    st.markdown("---")
    st.subheader("📈 Provider Risk Timeline")
    st.caption("Monthly claim volume, reimbursement, and risk scores over time")
    if sel:
        try:
            from src.risk_timeline import build_risk_timeline
            timeline = build_risk_timeline(sel)
            if not timeline.empty:
                st.dataframe(timeline, use_container_width=True)
                fig = px.line(timeline, x="YearMonth", y=["Claims","Reimbursement"], title=f"Monthly Activity for {sel}")
                plotly_chart(fig)
                if "HybridRiskScore" in timeline.columns:
                    st.line_chart(timeline.set_index("YearMonth")[["HybridRiskScore"]])
                st.caption("Timeline shows actual monthly claims; hybrid score is provider-level (stable), while temporal alerts flag spikes. For true risk-over-time, recompute model per time window.")
            else:
                st.info("No timeline data for this provider (check raw claims)")
        except Exception as e:
            st.error(f"Timeline failed: {e}")
    else:
        st.info("Select provider to view timeline")

# ================= PAGE 6: Anomaly Detection =================
elif page == "6. Anomaly Detection":
    st.title("Anomaly Detection")
    st.caption("Isolation Forest + peer deviations + temporal anomalies • Unsupervised signals")
    disclaimer_box("Anomaly ≠ fraud. Unusual does not mean fraudulent; human review needed.")
    
    tab1, tab2, tab3 = st.tabs(["Isolation Forest","Peer-Group Deviations","Temporal Anomalies"])
    
    with tab1:
        iso_path = Path("reports/isolation_forest_scores.csv")
        if iso_path.exists():
            iso = pd.read_csv(iso_path)
            st.metric("Anomalies Flagged", f"{(iso['IsAnomaly']==1).sum()} / {len(iso)}", delta=f"Contamination 10%")
            fig = px.histogram(iso, x="AnomalyScore", nbins=40, title="Anomaly Score Distribution", color="IsAnomaly", color_discrete_map={0:"#10b981",1:"#ef4444"})
            plotly_chart(fig)
            st.dataframe(iso.sort_values("AnomalyScore", ascending=False).head(100), use_container_width=True)
            # Scatter hybrid vs anomaly
            if not enriched.empty and "HybridRiskScore" in enriched.columns:
                merged = enriched.merge(iso, on="Provider", how="left")
                fig = px.scatter(merged, x="AnomalyScore", y="HybridRiskScore", color="RiskCategory" if "RiskCategory" in merged.columns else None, hover_name="Provider", title="Anomaly vs Hybrid Risk")
                plotly_chart(fig)
        else:
            st.warning("Isolation Forest scores not found. Run src/anomaly_detection.py")
            if st.button("Run Anomaly Scoring Now"):
                try:
                    from src.anomaly_detection import isolation_forest_scoring
                    _, df = isolation_forest_scoring()
                    st.success("Computed!")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(str(e))
    
    with tab2:
        peer_path = Path("reports/peer_group_alerts.csv")
        if peer_path.exists():
            peer = pd.read_csv(peer_path)
            if not peer.empty:
                st.metric("Peer Deviations", f"{len(peer):,}", delta=f"{peer['Provider'].nunique()} providers")
                st.dataframe(peer.head(200), use_container_width=True)
                # Summary
                summary_path = Path("reports/peer_group_deviation_summary.csv")
                if summary_path.exists():
                    st.markdown("**Provider Deviation Summary**")
                    st.dataframe(pd.read_csv(summary_path).sort_values("MaxAbsZ", ascending=False).head(50), use_container_width=True)
                # Z distribution
                fig = px.histogram(peer, x="ZScore", color="Severity", title="Peer Z-Score Distribution", nbins=40)
                plotly_chart(fig)
            else:
                st.info("No significant peer deviations (|z|>2.5) found.")
        else:
            st.warning("Peer alerts not found. Run src/anomaly_detection.py")
    
    with tab3:
        temp_path = Path("reports/temporal_anomaly_alerts.csv")
        temp_feat = Path("reports/temporal_provider_features.csv")
        if temp_path.exists():
            alerts = pd.read_csv(temp_path)
            if not alerts.empty:
                st.metric("Temporal Alerts", f"{len(alerts):,}")
                st.dataframe(alerts.head(200), use_container_width=True)
                fig = px.scatter(alerts, x="YearMonth", y="Value", color="Severity", size="ZScore", hover_name="Provider", title="Temporal Anomalies Over Time")
                plotly_chart(fig)
            else:
                st.info("No temporal anomalies detected (volume spikes or trend changes).")
            if temp_feat.exists():
                st.markdown("**Provider Temporal Features**")
                st.dataframe(pd.read_csv(temp_feat).head(100), use_container_width=True)
        else:
            st.warning("Temporal analysis not found. Run src/anomaly_detection.py")
    
    st.markdown("### Anomaly Ranking (Combined)")
    if not enriched.empty:
        # Show top anomalies by combining scores
        if "AnomalyScore" in enriched.columns or Path("reports/isolation_forest_scores.csv").exists():
            try:
                iso = pd.read_csv(Path("reports/isolation_forest_scores.csv")) if Path("reports/isolation_forest_scores.csv").exists() else pd.DataFrame()
                if not iso.empty:
                    ranking = enriched.merge(iso, on="Provider", how="left") if "Provider" in enriched.columns else enriched
                    # sort by anomaly then hybrid
                    ranking = ranking.sort_values("AnomalyScore", ascending=False).head(50)
                    st.dataframe(ranking[["Provider","AnomalyScore","HybridRiskScore","RiskCategory","TotalClaims","TotalReimbursement"] if "Provider" in ranking.columns else ranking.head()], use_container_width=True)
            except Exception as e:
                st.error(str(e))

# ================= PAGE 7: Network Analysis =================
elif page == "7. Network Analysis":
    st.title("Network Analysis")
    st.caption("Provider–Beneficiary–Physician relationships • Shared connections are not collusion")
    disclaimer_box("Shared beneficiary/physician links indicate co-occurrence. Do not treat as proof of collusion.")
    
    # Load reports
    shared_bene = Path("reports/network_shared_beneficiaries.csv")
    shared_phys = Path("reports/network_shared_physicians.csv")
    centrality = Path("reports/network_centrality.csv")
    network_img = Path("reports/network_graph.png")
    
    if shared_bene.exists():
        df = pd.read_csv(shared_bene)
        st.markdown("### Providers Sharing Beneficiaries (Potential Co-location)")
        if not df.empty:
            st.dataframe(df.sort_values("SharedBeneficiaries", ascending=False).head(100), use_container_width=True)
            fig = px.bar(df.head(20).sort_values("SharedBeneficiaries"), x="SharedBeneficiaries", y=df.head(20).apply(lambda r: f"{r['Provider1']} ↔ {r['Provider2']}", axis=1), orientation="h", title="Top Shared Beneficiary Pairs")
            plotly_chart(fig)
        else:
            st.info("No provider pairs sharing >5 beneficiaries found (threshold). Lower threshold would show more, but increases noise.")
    else:
        st.warning("Network analysis not yet run. Click below to build graph (may take 30-60s).")
        if st.button("Build Provider Network (20k claims sample)"):
            with st.spinner("Building NetworkX graph..."):
                try:
                    from src.network_analysis import build_provider_network, analyze_network, plot_network
                    G = build_provider_network(sample_size=20000)
                    analysis = analyze_network(G)
                    st.json(analysis)
                    plot_network(G)
                    st.success("Network built and saved to reports/")
                    st.rerun()
                except Exception as e:
                    st.error(f"Network build failed: {e}")
    
    if shared_phys.exists():
        df2 = pd.read_csv(shared_phys)
        st.markdown("### Providers Sharing Physicians")
        if not df2.empty:
            st.dataframe(df2.head(100), use_container_width=True)
        else:
            st.info("No provider pairs sharing >2 physicians.")
    
    if centrality.exists():
        st.markdown("### Centrality (Influence in Network)")
        cent = pd.read_csv(centrality)
        st.dataframe(cent.head(20), use_container_width=True)
        fig = px.bar(cent.head(20), x="Centrality", y="Node", color="Type", orientation="h", title="Top Central Nodes")
        plotly_chart(fig)
    
    if network_img.exists():
        st.image(str(network_img), caption="Provider Network Graph (sampled)")

# ================= PAGE 8: Investigation Centre =================
elif page == "8. Investigation Centre":
    st.title("Investigation Centre")
    st.caption("Case management • Status tracking • Reviewer notes • Evidence")
    
    inv_df = load_investigations()
    if inv_df.empty:
        st.info("No investigations yet. High-risk providers can be auto-flagged, or create a case manually.")
        # Show high-risk candidates
        if not enriched.empty and "RiskCategory" in enriched.columns:
            candidates = enriched[enriched["RiskCategory"].isin(["High","Critical"])].sort_values("HybridRiskScore", ascending=False).head(10)
            st.markdown("**High-Risk Candidates for Investigation**")
            st.dataframe(candidates[["Provider","HybridRiskScore","RiskCategory","PotentialFraud"] if "PotentialFraud" in candidates.columns else ["Provider","HybridRiskScore","RiskCategory"]], use_container_width=True)
    else:
        # KPIs
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            kpi_card("Total Cases", f"{len(inv_df):,}")
        with col2:
            kpi_card("Open (New/Review)", f"{len(inv_df[inv_df['status'].isin(['New','Under Review'])]) if 'status' in inv_df.columns else 0:,}")
        with col3:
            kpi_card("Escalated", f"{len(inv_df[inv_df['status']=='Escalated']) if 'status' in inv_df.columns else 0:,}")
        with col4:
            kpi_card("Closed", f"{len(inv_df[inv_df['status'].str.contains('Closed', na=False)]) if 'status' in inv_df.columns else 0:,}")
        
        # Status distribution
        if "status" in inv_df.columns:
            dist = inv_df["status"].value_counts()
            fig = px.pie(values=dist.values, names=dist.index, hole=0.5, title="Investigation Status")
            plotly_chart(fig)
        
        st.markdown("### All Investigations")
        st.dataframe(inv_df, use_container_width=True)
        
        # Case detail
        st.markdown("---")
        st.subheader("Case Detail & Workflow")
        case_ids = inv_df["id"].tolist() if "id" in inv_df.columns else []
        if case_ids:
            case_id = st.selectbox("Select Case ID", case_ids, key="case_select")
            if case_id:
                conn = sqlite3.connect("database/healthcare_fraud.db")
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM investigations WHERE id=?", (case_id,))
                    case = cur.fetchone()
                    if case:
                        # Convert row to dict
                        cols = [d[0] for d in cur.description]
                        case_dict = dict(zip(cols, case))
                        st.json(case_dict)
                        
                        # Provider profile link
                        provider_id = case_dict.get("provider_id")
                        if provider_id:
                            st.markdown(f"**Provider: {provider_id}**")
                            prow = enriched[enriched["Provider"]==provider_id].iloc[0] if provider_id in enriched["Provider"].values else None
                            if prow is not None:
                                st.metric("Risk Score", f"{prow.get('HybridRiskScore',0):.1f}")
                                if "RiskCategory" in prow:
                                    risk_badge(prow["RiskCategory"])
                            
                            # Notes
                            cur.execute("SELECT * FROM investigation_notes WHERE investigation_id=? ORDER BY created_at DESC", (case_id,))
                            notes = cur.fetchall()
                            if notes:
                                st.markdown("**Investigation Notes**")
                                for n in notes:
                                    ncols = [d[0] for d in cur.description]
                                    nd = dict(zip(ncols, n))
                                    st.markdown(f"**{nd.get('author','')}** @ {nd.get('created_at')}:\n> {nd.get('note')}")
                            else:
                                st.info("No notes yet")
                            
                            # Add note
                            with st.form(f"add_note_{case_id}"):
                                author = st.text_input("Author", value="Investigator")
                                note_text = st.text_area("Add Note / Evidence")
                                if st.form_submit_button("Add Note"):
                                    if note_text.strip():
                                        cur.execute("INSERT INTO investigation_notes (investigation_id, provider_id, author, note) VALUES (?,?,?,?)", (case_id, provider_id, author, note_text))
                                        cur.execute("INSERT INTO audit_logs (action, entity_type, entity_id, user, details) VALUES (?,?,?,?,?)", ("ADD_NOTE","investigation", str(case_id), author, note_text[:200]))
                                        conn.commit()
                                        st.success("Note added!")
                                        st.rerun()
                                    else:
                                        st.error("Note cannot be empty")
                            
                            # Status update
                            st.markdown("**Update Status**")
                            new_status = st.selectbox("New Status", ["New","Under Review","Escalated","Closed – No Concern","Closed – Requires Further Review"], index=0, key=f"status_{case_id}")
                            if st.button("Update Status", key=f"update_{case_id}"):
                                cur.execute("UPDATE investigations SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (new_status, case_id))
                                cur.execute("INSERT INTO audit_logs (action, entity_type, entity_id, details) VALUES (?,?,?,?)", ("STATUS_UPDATE","investigation", str(case_id), f"Status -> {new_status}"))
                                conn.commit()
                                st.success(f"Status updated to {new_status}")
                                st.rerun()
                finally:
                    conn.close()
    
    # Create new case
    st.markdown("---")
    st.subheader("Create New Investigation Case")
    with st.form("create_case"):
        prov_input = st.text_input("Provider ID", placeholder="PRV51001")
        priority = st.selectbox("Priority", ["Low","Medium","High","Critical"])
        summary = st.text_area("Summary / Reason for Investigation", placeholder="e.g., High hybrid risk score due to peer deviation in TotalReimbursement...")
        assign = st.text_input("Assign To", value="Investigator A")
        if st.form_submit_button("Create Case"):
            if not prov_input.strip():
                st.error("Provider ID required")
            else:
                conn = sqlite3.connect("database/healthcare_fraud.db")
                try:
                    cur = conn.cursor()
                    # Check provider exists
                    cur.execute("SELECT provider_id FROM providers WHERE provider_id=?", (prov_input.strip(),))
                    if not cur.fetchone():
                        st.warning(f"Provider {prov_input} not in DB, but creating case anyway (may be from test set)")
                    cur.execute("INSERT INTO investigations (provider_id, status, assigned_to, priority, summary) VALUES (?,?,?,?,?)", (prov_input.strip(), "New", assign, priority, summary))
                    conn.commit()
                    st.success(f"Investigation created for {prov_input}")
                    # Audit
                    cur.execute("INSERT INTO audit_logs (action, entity_type, entity_id, details) VALUES (?,?,?,?)", ("CREATE_CASE","investigation", prov_input, f"Priority {priority}"))
                    conn.commit()
                except Exception as e:
                    st.error(str(e))
                finally:
                    conn.close()

# ================= PAGE 9: Model Performance =================
elif page == "9. Model Performance":
    st.title("Model Performance")
    st.caption("Validation vs Test • ROC/PR curves • Confusion matrices • Threshold analysis")
    disclaimer_box("Report test performance, not training. Pay particular attention to minority-class PR-AUC.")
    
    # Load comparison
    comp_path = Path("reports/validation_model_comparison.csv")
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        st.markdown("### Model Comparison (Validation PR-AUC)")
        st.dataframe(comp, use_container_width=True)
        # Bar chart
        if "PR_AUC" in comp.columns or "PR-AUC" in comp.columns:
            pr_col = "PR_AUC" if "PR_AUC" in comp.columns else "PR-AUC"
            fig = px.bar(comp, x="Model", y=pr_col, color="Model", title="Validation PR-AUC by Model")
            plotly_chart(fig)
        if "ROC_AUC" in comp.columns:
            fig = px.bar(comp, x="Model", y="ROC_AUC", color="Model", title="Validation ROC-AUC by Model")
            plotly_chart(fig)
    else:
        st.warning("Validation comparison not found. Run src/train.py")
    
    # Detailed
    det_path = Path("reports/model_comparison.csv")
    if det_path.exists():
        st.markdown("### Detailed Validation Metrics")
        st.dataframe(pd.read_csv(det_path), use_container_width=True)
    
    # Test metrics
    if meta and meta.get("test_metrics"):
        tm = meta["test_metrics"]
        st.markdown(f"### Final Test Evaluation (Untouched Test Set) • Model: {meta.get('best_model')}")
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1:
            kpi_card("Accuracy", f"{tm.get('accuracy',0):.3f}")
        with c2:
            kpi_card("Precision", f"{tm.get('precision',0):.3f}")
        with c3:
            kpi_card("Recall", f"{tm.get('recall',0):.3f}", help_text="Primary for fraud detection")
        with c4:
            kpi_card("F1-Score", f"{tm.get('f1',0):.3f}")
        with c5:
            kpi_card("PR-AUC", f"{tm.get('pr_auc',0):.3f}", help_text="Primary optimization metric")
        st.markdown(f"**Confusion Matrix:** `{tm.get('confusion_matrix')}`")
        # Classification report
        if "classification_report" in tm:
            st.json(tm["classification_report"])
    
    # Plots
    col1, col2 = st.columns(2)
    with col1:
        roc_path = Path("reports/roc_curve.png")
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curve")
        else:
            st.info("ROC curve not generated. Run src/evaluate.py")
    with col2:
        pr_path = Path("reports/pr_curve.png")
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curve")
    
    cm_path = Path("reports/confusion_matrix.png")
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix")
    
    # Threshold analysis
    thr_path = Path("reports/threshold_comparison.csv")
    if thr_path.exists():
        st.markdown("### Threshold Optimization (Validation)")
        thr = pd.read_csv(thr_path)
        st.dataframe(thr, use_container_width=True)
        fig = go.Figure()
        for col in ["Precision","Recall","F1"]:
            if col in thr.columns:
                fig.add_trace(go.Scatter(x=thr["Threshold"], y=thr[col], name=col, mode='lines+markers'))
        fig.update_layout(title="Threshold vs Precision/Recall/F1", xaxis_title="Threshold", yaxis_title="Score")
        plotly_chart(fig)
        # Show best
        if "F1" in thr.columns:
            best = thr.loc[thr["F1"].idxmax()]
            st.success(f"Best F1 threshold: {best['Threshold']} → F1={best['F1']:.3f}, Precision={best['Precision']:.3f}, Recall={best['Recall']:.3f}")
    else:
        st.info("Threshold analysis not found. Run src/evaluate.py")
    
    # Cross-validation results (if available)
    cv_path = Path("reports/model_comparison.csv")
    if cv_path.exists():
        st.markdown("### Cross-Validation (5-Fold)")
        cv = pd.read_csv(cv_path)
        # If has CV columns
        if any("CV" in c for c in cv.columns):
            st.dataframe(cv, use_container_width=True)

# ================= PAGE 10: Model Monitoring =================
elif page == "10. Model Monitoring":
    st.title("Model Monitoring")
    st.caption("Feature drift • Prediction drift • Model versioning • Data vs performance drift")
    
    drift_feat = Path("reports/feature_drift_report.csv")
    if drift_feat.exists():
        drift = pd.read_csv(drift_feat)
        st.markdown("### Feature Drift (KS Test)")
        st.dataframe(drift, use_container_width=True)
        # Highlight high drift
        high = drift[drift["DriftDetected"]==True] if "DriftDetected" in drift.columns else pd.DataFrame()
        if not high.empty:
            st.warning(f"⚠️ {len(high)} features show significant drift (p<0.05). Investigate data pipeline changes, not necessarily model failure.")
            fig = px.bar(high.sort_values("KS_Statistic", ascending=False).head(15), x="KS_Statistic", y="Feature", orientation="h", title="Top Drifting Features (KS)")
            plotly_chart(fig)
        else:
            st.success("No significant feature drift detected.")
    else:
        st.info("Drift report not found. Run src/drift_detection.py")
        if st.button("Run Drift Analysis Now"):
            try:
                from src.drift_detection import run_drift_analysis
                run_drift_analysis()
                st.success("Drift analysis completed")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    
    # Prediction drift
    pred_drift = Path("reports/prediction_drift_report.json")
    if pred_drift.exists():
        with open(pred_drift) as f:
            pd_drift = json.load(f)
        st.markdown("### Prediction Distribution Drift")
        st.json(pd_drift)
        if pd_drift.get("DriftDetected"):
            st.warning("Prediction drift detected. This is data drift, not proven performance deterioration. Validate with labeled holdout before retraining.")
    
    # Alerts
    alerts_path = Path("reports/drift_alerts.csv")
    if alerts_path.exists():
        alerts = pd.read_csv(alerts_path)
        if not alerts.empty:
            st.markdown("### Drift Alerts")
            st.dataframe(alerts, use_container_width=True)
    
    # Model versions
    st.markdown("### Model Version Tracking")
    versions_path = Path("reports/model_versions.json")
    if versions_path.exists():
        with open(versions_path) as f:
            vers = json.load(f)
        st.json(vers)
    elif meta:
        st.json({"latest": {"version": meta.get("model_version"), "best_model": meta.get("best_model"), "training_date": meta.get("training_date"), "metrics": meta.get("test_metrics")}})
    else:
        st.info("No version history")
    
    # Feature distribution monitoring (simple: show current vs reference mean)
    if not enriched.empty:
        st.markdown("### Current Data Distribution (vs Reference)")
        # Show enriched means vs? Use master
        if master is not None:
            # Compare enriched means to master means for key features
            key_feats = [c for c in ["TotalClaims","TotalReimbursement","AverageReimbursement"] if c in master.columns]
            if key_feats:
                comp = pd.DataFrame({
                    "Feature": key_feats,
                    "CurrentMean": [enriched[feat].mean() if feat in enriched.columns else master[feat].mean() for feat in key_feats],
                    "ReferenceMean": [master[feat].mean() for feat in key_feats]
                })
                st.dataframe(comp, use_container_width=True)
                fig = px.bar(comp, x="Feature", y=["CurrentMean","ReferenceMean"], barmode="group", title="Current vs Reference Means")
                plotly_chart(fig)

# ================= PAGE 11: Reports =================
elif page == "11. Reports":
    st.title("Reports")
    st.caption("Investigation PDFs • CSV exports • Model evaluation • Audit logs")
    
    tab1, tab2, tab3 = st.tabs(["Investigation PDF","CSV & Data Exports","Audit Logs"])
    
    with tab1:
        st.markdown("### Automated PDF Investigation Report")
        st.caption("Professional PDF with provider risk, SHAP, peer comparison, claim summaries, notes")
        provider_list = enriched["Provider"].astype(str).tolist() if not enriched.empty else []
        sel = st.selectbox("Select Provider for Report", provider_list[:200], key="pdf_provider")
        include_shap = st.checkbox("Include SHAP explanation (if available)", value=True)
        include_claims = st.checkbox("Include claim sample", value=True)
        notes_input = st.text_area("Investigation Notes (to include in PDF)", placeholder="e.g., Reviewed billing patterns, checked duplicate claims...")
        
        if sel and st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
                    from src.pdf_report import generate_investigation_pdf
                    # Get hybrid row
                    row = enriched[enriched["Provider"]==sel].iloc[0] if sel in enriched["Provider"].values else None
                    if row is not None:
                        # Prepare output path
                        out_path = Path(f"reports/investigation_report_{sel}.pdf")
                        generate_investigation_pdf(
                            provider_id=sel,
                            hybrid_row=row.to_dict(),
                            master_row=master[master["Provider"]==sel].iloc[0].to_dict() if master is not None and sel in master["Provider"].values else row.to_dict(),
                            provider_info=get_provider_detail(sel) if 'get_provider_detail' in globals() else {},
                            investigation_notes=notes_input,
                            include_shap=include_shap,
                            output_path=out_path
                        )
                        st.success(f"PDF generated: {out_path}")
                        with open(out_path, "rb") as f:
                            st.download_button("Download PDF", f, file_name=out_path.name, mime="application/pdf")
                        # Show preview info
                        st.info("PDF contains: Provider ID, Risk Assessment, Fraud Prob, Hybrid Score, Suspicious Indicators, SHAP, Peer Comparison, Claim Summaries, Notes, Model Version, Date")
                    else:
                        st.error("Provider not found")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                    st.exception(e)
        
        # List existing reports
        existing = list(Path("reports").glob("investigation_report_*.pdf"))
        if existing:
            st.markdown("**Existing Reports**")
            for f in existing[:10]:
                st.markdown(f"- {f.name} ({f.stat().st_size/1024:.1f} KB)")
                with open(f, "rb") as file:
                    st.download_button(f"Download {f.name}", file, file_name=f.name, mime="application/pdf", key=f"dl_{f.name}")
    
    with tab2:
        st.markdown("### Download Datasets & Reports")
        files = {
            "Provider Risk Ranking (hybrid_risk_scores.csv)": Path("reports/hybrid_risk_scores.csv"),
            "Master Provider Features (master_provider_features.csv)": Path("reports/master_provider_features.csv"),
            "Model Evaluation (evaluation_report.json)": Path("reports/evaluation_report.json"),
            "Threshold Comparison (threshold_comparison.csv)": Path("reports/threshold_comparison.csv"),
            "Validation Comparison (validation_model_comparison.csv)": Path("reports/validation_model_comparison.csv"),
            "Feature Drift (feature_drift_report.csv)": Path("reports/feature_drift_report.csv"),
            "Duplicate Summary (duplicate_claims_summary.csv)": Path("reports/duplicate_claims_summary.csv"),
            "SHAP Importance (shap_feature_importance.csv)": Path("reports/shap_feature_importance.csv"),
        }
        for label, path in files.items():
            if path.exists():
                with open(path, "rb") as f:
                    st.download_button(f"📥 {label}", f, file_name=path.name, key=f"dl2_{path.name}")
            else:
                st.caption(f"Missing: {label} ({path.name})")
        
        if not enriched.empty:
            st.markdown("### Export Current View")
            csv = enriched.to_csv(index=False).encode('utf-8')
            st.download_button("Download All Providers (CSV)", csv, "providers_export.csv", "text/csv")
    
    with tab3:
        st.markdown("### Audit Logs")
        conn = sqlite3.connect("database/healthcare_fraud.db") if Path("database/healthcare_fraud.db").exists() else None
        if conn:
            try:
                logs = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 100", conn)
                if not logs.empty:
                    st.dataframe(logs, use_container_width=True)
                else:
                    st.info("No audit logs yet. Actions like creating cases will appear here.")
            except:
                st.info("No audit logs table or empty")
            finally:
                conn.close()
        else:
            st.info("DB not initialized. Audit logs will appear after investigation actions.")

# ================= PAGE 12: Settings =================
elif page == "12. Settings":
    st.title("Settings")
    st.caption("Risk thresholds • Model selection • App info • Database")
    
    tab1, tab2, tab3 = st.tabs(["Risk Configuration","Model & App Info","Database"])
    
    with tab1:
        st.markdown("### Risk Threshold Configuration")
        st.caption("Categories are configurable and documented. High-risk ≠ fraud.")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            low = st.slider("Low: 0 -", 10, 50, 30)
        with c2:
            medium = st.slider("Medium: Low -", 40, 80, 60)
        with c3:
            high = st.slider("High: Medium -", 60, 90, 80)
        with c4:
            crit = st.slider("Critical: High - 100", 80, 100, 80)
        
        st.json({"Low": (0, low), "Medium": (low, medium), "High": (medium, high), "Critical": (high, 100)})
        
        if st.button("Save & Recompute Hybrid Scores"):
            try:
                from src.risk_scoring import hybrid_risk_scoring
                new_thr = {"Low":(0, low), "Medium":(low, medium), "High":(medium, high), "Critical":(high, 100)}
                hybrid_risk_scoring(thresholds=new_thr, save=True)
                st.success("Hybrid scores recomputed with new thresholds! Refresh to see changes.")
            except Exception as e:
                st.error(str(e))
        
        st.markdown("**Current Config**")
        cfg_path = Path("reports/hybrid_scoring_config.json")
        if cfg_path.exists():
            st.json(json.load(open(cfg_path)))
    
    with tab2:
        st.markdown("### Application Information")
        st.markdown("""
        - **Project**: AI-Based Healthcare Insurance Fraud Detection
        - **Version**: 1.0.0
        - **Tech Stack**: Python, Pandas, Scikit-learn, XGBoost, LightGBM, Isolation Forest, SHAP, Streamlit, Plotly, SQLite, ReportLab
        - **Dataset**: US Medicare (Kaggle Healthcare Provider Fraud) – provider-level labels, inpatient/outpatient claims, beneficiary demographics
        - **Primary Metrics**: PR-AUC, Recall, F1 (fraud prevalence ~9.35%)
        - **Caveat**: Model trained on US Medicare; not validated for Indian insurance schemes. High-risk is investigation signal, not proof.
        """)
        if meta:
            st.json(meta)
        st.markdown("### Model Selection")
        st.caption("Best model chosen via validation PR-AUC. See Model Performance page for comparison.")
        if comp_path.exists():
            st.dataframe(pd.read_csv(comp_path), use_container_width=True)
    
    with tab3:
        st.markdown("### Database Information")
        db_path = Path("database/healthcare_fraud.db")
        if db_path.exists():
            st.success(f"Database found: {db_path} ({db_path.stat().st_size/1024:.1f} KB)")
            conn = sqlite3.connect(str(db_path))
            try:
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                st.dataframe(tables, use_container_width=True)
                for table in tables["name"]:
                    cnt = pd.read_sql_query(f"SELECT COUNT(*) as cnt FROM {table}", conn)
                    st.caption(f"{table}: {cnt.iloc[0]['cnt']} rows")
            except Exception as e:
                st.error(str(e))
            finally:
                conn.close()
            if st.button("Re-initialize Database"):
                try:
                    from database.db import init_db, populate_from_reports
                    init_db()
                    populate_from_reports()
                    st.success("Database re-initialized")
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("Database not found. Run: python -m database.db or python database/db.py")
            if st.button("Initialize Database Now"):
                try:
                    from database.db import init_db, populate_from_reports
                    init_db()
                    populate_from_reports()
                    st.success("Database initialized")
                except Exception as e:
                    st.error(str(e))
    
    st.markdown("---")
    st.caption("For troubleshooting, see README.md section 'HOW TO RUN THE COMPLETE PROJECT ON MY WINDOWS LAPTOP'")

# Footer
st.markdown("---")
st.caption("© 2026 FraudShield AI | B.Tech CSE Final Year Project | Human-in-the-loop • Explainable • Local-first | High-risk ≠ proof of fraud")
