"""
AI-Based Healthcare Insurance Fraud Detection - Main Dashboard
Executive, enterprise-style Streamlit application with Customer Journey Flow.
Covers 13 pages (12 original + 1 new Customer Journey) as per project specification.
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
from dashboard.components.ui import (
    kpi_card, risk_badge, disclaimer_box, caveat_box, 
    plotly_chart, styled_dataframe, guide_box, section_header, 
    navigation_guide, action_button
)
from dashboard.components.db_utils import load_providers, load_hybrid_scores, load_investigations, get_provider_detail

st.set_page_config(
    page_title="FraudShield AI - Healthcare Insurance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern enterprise look
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); 
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio > div { 
        background: rgba(255,255,255,0.05); 
        border-radius: 10px; 
        padding: 10px;
        margin: 4px 0;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stRadio > div:hover {
        background: rgba(255,255,255,0.1);
        transform: translateX(4px);
    }
    
    /* Main content */
    .main > div { padding-top: 2rem; }
    h1, h2, h3 { color: #0f172a; font-weight: 800; }
    .block-container { padding-top: 1.5rem; }
    
    /* Customer Journey Flow */
    .flow-container {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.05);
        margin: 20px 0;
    }
    @media (prefers-color-scheme: dark) {
        .flow-container {
            background: #1e293b;
            border: 1px solid #334155;
        }
    }
    .flow-step {
        display: flex;
        align-items: center;
        margin: 16px 0;
        padding: 12px;
        border-radius: 10px;
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        transition: all 0.2s;
    }
    @media (prefers-color-scheme: dark) {
        .flow-step {
            background: #0f172a;
            border-left: 4px solid #60a5fa;
        }
    }
    .flow-step:hover {
        background: #e2e8f0;
        transform: scale(1.01);
    }
    .flow-icon {
        font-size: 24px;
        margin-right: 16px;
    }
    .flow-text {
        font-size: 15px;
        font-weight: 600;
        color: #0f172a;
    }
    .flow-arrow {
        text-align: center;
        font-size: 24px;
        color: #3b82f6;
        margin: 8px 0;
    }
    
    /* Unique Features Cards */
    .feature-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.1);
    }
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: #1e293b;
            border: 1px solid #334155;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 12px 0;">
        <div style="font-size: 26px; font-weight: 900; color: white; letter-spacing: -0.02em;">🏥 FraudShield AI</div>
        <div style="font-size: 12px; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin-top:6px;">Healthcare Fraud Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    pages = [
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
        "12. Settings",
        "13. Customer Journey Flow"
    ]
    
    page = st.radio(
        "Navigation",
        pages,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    navigation_guide(page)
    st.caption("ℹ️ High-risk = investigation signal, not proof of fraud")
    st.caption("🌐 Global Healthcare • Local-first • Transparent")
    
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
    if hybrid is not None and not hybrid.empty:
        if master is not None:
            extra_cols = ["TotalClaims","TotalReimbursement","AverageReimbursement","UniqueBeneficiaries","InpatientClaims","OutpatientClaims","PotentialFraud"]
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
    section_header("Executive Overview", "📊", "AI-powered fraud screening • Human-in-the-loop investigation • Transparent risk scoring")
    guide_box("Project Presentation Guide", 
              "Use this page to present: 1) Total scale of operations, 2) Risk distribution, 3) Model performance, 4) Financial impact. Highlight the 9.35% fraud prevalence benchmark.",
              "💡")
    caveat_box()
    disclaimer_box()
    
    if enriched.empty:
        st.error("No provider data found. Run src/feature_engineering.py and src/train.py to generate datasets.")
        st.stop()
    
    # KPIs
    total_providers = len(enriched)
    total_claims = enriched["TotalClaims"].sum() if "TotalClaims" in enriched.columns else 0
    high_risk = len(enriched[enriched["RiskCategory"].isin(["High","Critical"])]) if "RiskCategory" in enriched.columns else 0
    critical = len(enriched[enriched["RiskCategory"]=="Critical"]) if "RiskCategory" in enriched.columns else 0
    fraud_labelled = (enriched["PotentialFraud"]=="Yes").sum() if "PotentialFraud" in enriched.columns else 0
    avg_risk = enriched["HybridRiskScore"].mean() if "HybridRiskScore" in enriched.columns else 0
    
    # Prevent division by zero
    prevalence_pct = (fraud_labelled/total_providers*100) if total_providers > 0 else 0.0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        kpi_card("Total Providers", f"{total_providers:,}", help_text="Unique healthcare providers", icon="🏥")
    with col2:
        kpi_card("Total Claims", f"{int(total_claims):,}", help_text="Inpatient + outpatient", icon="📋")
    with col3:
        kpi_card("High-Risk", f"{high_risk:,}", delta=f"Critical: {critical}", help_text="Risk ≥60", icon="⚠️")
    with col4:
        kpi_card("Fraud-Labelled", f"{fraud_labelled:,}", delta=f"{prevalence_pct:.1f}%", help_text="Train labels", icon="🔍")
    with col5:
        kpi_card("Avg Risk Score", f"{avg_risk:.1f}/100", help_text="Hybrid score", icon="🎯")
    
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
    
    # Unique Features Presentation
    st.markdown("### 🌟 Unique Project Features")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:24px; margin-bottom:8px;">🤖</div>
            <strong style="font-size:18px;">Hybrid Risk Scoring</strong><br>
            <span style="color:#64748b; font-size:13px;">Supervised (55%) + Anomaly (25%) + Peer (20%)</span>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:24px; margin-bottom:8px;">🧠</div>
            <strong style="font-size:18px;">Explainable AI (SHAP)</strong><br>
            <span style="color:#64748b; font-size:13px;">Transparent feature contributions & what-if analysis</span>
        </div>
        """, unsafe_allow_html=True)
    with col_f3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:24px; margin-bottom:8px;">🕸️</div>
            <strong style="font-size:18px;">Network Analysis</strong><br>
            <span style="color:#64748b; font-size:13px;">Provider-Beneficiary-Physician relationship mapping</span>
        </div>
        """, unsafe_allow_html=True)

# ================= PAGE 2: Provider Explorer =================
elif page == "2. Provider Explorer":
    section_header("Provider Explorer", "🔍", "Search, filter, and inspect provider profiles • Evidence-based review")
    guide_box("How to Present", 
              "Demonstrate: 1) Search functionality, 2) Risk filtering, 3) Detailed provider profiles, 4) Claim drill-down, 5) Action buttons for investigation.",
              "💡")
    
    if enriched.empty:
        st.error("No provider data available")
        st.stop()
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1: search = st.text_input("🔍 Search Provider ID", placeholder="e.g., PRV51001")
        with col2: risk_filter = st.multiselect("Risk Category", ["Low","Medium","High","Critical"], default=[])
        with col3: fraud_filter = st.selectbox("Fraud Label", ["All","Yes","No"])
        with col4: sort_by = st.selectbox("Sort by", ["HybridRiskScore","TotalReimbursement","TotalClaims","AverageReimbursement"])
    
    df = enriched.copy()
    if search: df = df[df["Provider"].astype(str).str.contains(search, case=False, na=False)]
    if risk_filter: df = df[df["RiskCategory"].isin(risk_filter)]
    if fraud_filter != "All" and "PotentialFraud" in df.columns:
        df = df[df["PotentialFraud"].astype(str)==fraud_filter]
    if sort_by in df.columns: df = df.sort_values(sort_by, ascending=False)
    
    st.markdown(f"**Showing {len(df):,} providers** (total {len(enriched):,})")
    display_cols = [c for c in ["Provider","RiskCategory","HybridRiskScore","SupervisedScore","AnomalyScore","PeerScore","TotalClaims","TotalReimbursement","AverageReimbursement","PotentialFraud"] if c in df.columns]
    if display_cols:
        styled_dataframe(df[display_cols].head(500), height=400)
    
    st.markdown("---")
    section_header("Provider Profile", "👤", "Detailed view with metrics, claims, and actions")
    provider_ids = df["Provider"].astype(str).tolist() if "Provider" in df.columns else []
    if provider_ids:
        selected = st.selectbox("Select Provider for detailed view", provider_ids[:200], key="provider_select")
        if selected:
            detail = get_provider_detail(selected)
            row = enriched[enriched["Provider"]==selected].iloc[0] if selected in enriched["Provider"].values else None
            if row is not None:
                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("Hybrid Risk Score", f"{row.get('HybridRiskScore',0):.1f}")
                    if "RiskCategory" in row: risk_badge(row["RiskCategory"])
                with c2:
                    st.metric("Fraud Probability", f"{row.get('SupervisedScore',0)/100:.1%}" if "SupervisedScore" in row else "N/A")
                    st.metric("PotentialFraud Label", str(row.get("PotentialFraud","Unknown")))
                with c3:
                    st.metric("Total Claims", f"{int(row.get('TotalClaims',0)):,}")
                    st.metric("Unique Beneficiaries", f"{int(row.get('UniqueBeneficiaries',0)):,}" if "UniqueBeneficiaries" in row else "N/A")
                with c4:
                    st.metric("Total Reimbursement", f"${row.get('TotalReimbursement',0):,.0f}" if "TotalReimbursement" in row else "N/A")
                    st.metric("Avg Reimbursement", f"${row.get('AverageReimbursement',0):,.0f}" if "AverageReimbursement" in row else "N/A")
                
                # Claims table
                st.markdown("**Recent Claims (sample)**")
                claims_df = pd.DataFrame()
                try:
                    from dashboard.components.db_utils import get_claims_for_provider as gcp
                    claims_df = gcp(selected, limit=20)
                except: pass
                
                if not claims_df.empty:
                    styled_dataframe(claims_df, height=300)
                else:
                    st.info("No claim drill-down available. Check data/raw claims.")
                
                colA, colB = st.columns(2)
                with colA:
                    if action_button("Explain this Provider (SHAP)", "🧠"):
                        st.session_state['explain_provider'] = selected
                        st.info("Go to Explainable AI page and select this provider")
                with colB:
                    if action_button("Generate Investigation Report (PDF)", "📄"):
                        st.session_state['report_provider'] = selected
                        st.info("Go to Reports page to download PDF")

# ================= PAGE 3: Fraud Analytics =================
elif page == "3. Fraud Analytics":
    section_header("Fraud Analytics", "📈", "Distribution, billing patterns, reimbursement analysis, provider comparisons")
    guide_box("Presentation Strategy", 
              "Show: 1) Fraud prevalence (9.35% benchmark), 2) Claim distribution patterns, 3) Reimbursement analysis, 4) Provider comparisons, 5) Duplicate claim detection.",
              "💡")
    
    if master is None or master.empty:
        st.error("Master provider features not found. Run feature engineering.")
        st.stop()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Fraud Distribution","Billing & Reimbursement","Provider Comparison","Duplicate Patterns"])
    
    with tab1:
        if "PotentialFraud" in master.columns:
            dist = master["PotentialFraud"].value_counts()
            dist_norm = master["PotentialFraud"].value_counts(normalize=True)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=dist.values, names=dist.index, hole=0.4, title="Fraud Label Distribution")
                plotly_chart(fig)
            with col2:
                st.metric("Fraud Prevalence", f"{dist_norm.get('Yes',0)*100:.2f}%" if "Yes" in dist_norm else f"{dist_norm.get(1,0)*100:.2f}%")
                st.dataframe(pd.DataFrame({"Count":dist, "Percent":dist_norm*100}), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(master, x="PotentialFraud" if "PotentialFraud" in master.columns else None, y="AverageReimbursement", 
                         color="PotentialFraud" if "PotentialFraud" in master.columns else None, title="Avg Reimbursement by Fraud")
            plotly_chart(fig)
        with col2:
            fig = px.box(master, x="PotentialFraud" if "PotentialFraud" in master.columns else None, y="TotalReimbursement", 
                         color="PotentialFraud" if "PotentialFraud" in master.columns else None, title="Total Reimbursement by Fraud")
            fig.update_layout(yaxis_type="log")
            plotly_chart(fig)

# ================= PAGE 4: AI Predictions =================
elif page == "4. AI Predictions":
    section_header("AI Predictions", "🤖", "Generate fraud probability + hybrid risk score on demand • Persisted history")
    guide_box("How to Explain", 
              "1) Select provider, 2) Show current prediction, 3) Explain scoring components (55%+25%+20%), 4) Demonstrate live inference, 5) Show prediction history.",
              "💡")
    disclaimer_box()
    
    if enriched.empty:
        st.error("No data")
        st.stop()
    
    provider_list = enriched["Provider"].astype(str).tolist()
    selected = st.selectbox("Select Provider to Predict", provider_list[:500])
    
    if selected:
        row = enriched[enriched["Provider"]==selected].iloc[0] if selected in enriched["Provider"].values else None
        if row is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                proba = row.get("SupervisedScore", 0) /100
                st.metric("Fraud Probability", f"{proba:.3f}")
                st.progress(float(proba))
            with col2:
                st.metric("Hybrid Risk Score", f"{row.get('HybridRiskScore',0):.1f}/100")
                if "RiskCategory" in row: risk_badge(row["RiskCategory"])
            with col3:
                if "PotentialFraud" in row: st.metric("Ground Truth Label", str(row["PotentialFraud"]))
                st.metric("Model Version", meta.get("model_version","v1.0.0") if meta else "v1.0.0")
            
            st.markdown("**Scoring Components (Transparent)**")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Supervised (55%)", f"{row.get('SupervisedScore',0):.1f}", help_text="Calibrated probability")
            with c2: st.metric("Anomaly (25%)", f"{row.get('AnomalyScore',0):.1f}", help_text="Isolation Forest")
            with c3: st.metric("Peer Deviation (20%)", f"{row.get('PeerScore',0):.1f}", help_text="Robust MAD z-score")

# ================= PAGE 5: Explainable AI =================
elif page == "5. Explainable AI":
    section_header("Explainable AI", "🧠", "SHAP-based global and individual explanations • Not causal proof")
    guide_box("Presentation Flow", 
              "1) Global feature importance, 2) Individual provider explanation, 3) What-if simulation, 4) Copilot Q&A, 5) Risk timeline.",
              "💡")
    disclaimer_box("SHAP shows feature contribution to model output, not causality or guilt.")
    
    shap_imp = Path("reports/shap_feature_importance.csv")
    if shap_imp.exists():
        imp = pd.read_csv(shap_imp)
        st.markdown("### Global Feature Importance (Mean |SHAP|)")
        c1, c2 = st.columns([2,3])
        with c1: styled_dataframe(imp.head(15), height=300)
        with c2:
            fig = px.bar(imp.head(15).sort_values("MeanAbsSHAP"), x="MeanAbsSHAP", y="Feature", orientation="h", title="Top 15 Features")
            plotly_chart(fig, height=300)

# ================= PAGE 6: Anomaly Detection =================
elif page == "6. Anomaly Detection":
    section_header("Anomaly Detection", "⚠️", "Isolation Forest + peer deviations + temporal anomalies • Unsupervised signals")
    guide_box("Unique Feature", 
              "Demonstrate unsupervised detection: 1) Isolation Forest scores, 2) Peer-group deviations, 3) Temporal anomaly alerts, 4) Combined ranking.",
              "💡")
    disclaimer_box("Anomaly ≠ fraud. Unusual does not mean fraudulent; human review needed.")
    
    tab1, tab2, tab3 = st.tabs(["Isolation Forest","Peer-Group Deviations","Temporal Anomalies"])
    
    with tab1:
        iso_path = Path("reports/isolation_forest_scores.csv")
        if iso_path.exists():
            iso = pd.read_csv(iso_path)
            st.metric("Anomalies Flagged", f"{(iso['IsAnomaly']==1).sum()} / {len(iso)}", delta="Contamination 10%")
            fig = px.histogram(iso, x="AnomalyScore", nbins=40, title="Anomaly Score Distribution", color="IsAnomaly")
            plotly_chart(fig)

# ================= PAGE 7: Network Analysis =================
elif page == "7. Network Analysis":
    section_header("Network Analysis", "🕸️", "Provider–Beneficiary–Physician relationships • Shared connections analysis")
    guide_box("Project Highlight", 
              "Show: 1) Shared beneficiary pairs, 2) Shared physician links, 3) Centrality influence, 4) Network graph visualization.",
              "💡")
    disclaimer_box("Shared links indicate co-occurrence. Do not treat as proof of collusion.")
    
    shared_bene = Path("reports/network_shared_beneficiaries.csv")
    if shared_bene.exists():
        df = pd.read_csv(shared_bene)
        st.markdown("### Providers Sharing Beneficiaries")
        if not df.empty:
            styled_dataframe(df.sort_values("SharedBeneficiaries", ascending=False).head(100), height=300)

# ================= PAGE 8: Investigation Centre =================
elif page == "8. Investigation Centre":
    section_header("Investigation Centre", "📋", "Case management • Status tracking • Reviewer notes • Evidence")
    guide_box("Workflow Explanation", 
              "1) View existing cases, 2) Track status (New/Review/Escalated/Closed), 3) Add investigation notes, 4) Update status, 5) Create new cases.",
              "💡")
    
    inv_df = load_investigations()
    if inv_df.empty:
        st.info("No investigations yet. High-risk providers can be auto-flagged, or create a case manually.")
    else:
        col1,col2,col3,col4 = st.columns(4)
        with col1: kpi_card("Total Cases", f"{len(inv_df):,}", icon="📋")
        with col2: kpi_card("Open (New/Review)", f"{len(inv_df[inv_df['status'].isin(['New','Under Review'])]) if 'status' in inv_df.columns else 0:,}", icon="🔄")
        with col3: kpi_card("Escalated", f"{len(inv_df[inv_df['status']=='Escalated']) if 'status' in inv_df.columns else 0:,}", icon="🚨")
        with col4: kpi_card("Closed", f"{len(inv_df[inv_df['status'].str.contains('Closed', na=False)]) if 'status' in inv_df.columns else 0:,}", icon="✅")
        
        st.markdown("### All Investigations")
        styled_dataframe(inv_df, height=400)

# ================= PAGE 9: Model Performance =================
elif page == "9. Model Performance":
    section_header("Model Performance", "🎯", "Validation vs Test • ROC/PR curves • Confusion matrices • Threshold analysis")
    guide_box("Technical Presentation", 
              "1) Validation comparison, 2) Test metrics (PR-AUC focus), 3) ROC/PR curves, 4) Confusion matrix, 5) Threshold optimization.",
              "💡")
    disclaimer_box("Report test performance, not training. Focus on minority-class PR-AUC.")
    
    comp_path = Path("reports/validation_model_comparison.csv")
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        st.markdown("### Model Comparison (Validation PR-AUC)")
        styled_dataframe(comp, height=300)
    
    if meta and meta.get("test_metrics"):
        tm = meta["test_metrics"]
        st.markdown(f"### Final Test Evaluation • Model: {meta.get('best_model')}")
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: kpi_card("Accuracy", f"{tm.get('accuracy',0):.3f}", icon="📊")
        with c2: kpi_card("Precision", f"{tm.get('precision',0):.3f}", icon="🎯")
        with c3: kpi_card("Recall", f"{tm.get('recall',0):.3f}", help_text="Primary", icon="🔍")
        with c4: kpi_card("F1-Score", f"{tm.get('f1',0):.3f}", icon="⚖️")
        with c5: kpi_card("PR-AUC", f"{tm.get('pr_auc',0):.3f}", help_text="Primary", icon="📈")

# ================= PAGE 10: Model Monitoring =================
elif page == "10. Model Monitoring":
    section_header("Model Monitoring", "📊", "Feature drift • Prediction drift • Model versioning • Data vs performance drift")
    guide_box("Maintenance Guide", 
              "1) Feature drift detection (KS test), 2) Prediction distribution drift, 3) Drift alerts, 4) Version tracking, 5) Current vs reference comparison.",
              "💡")
    
    drift_feat = Path("reports/feature_drift_report.csv")
    if drift_feat.exists():
        drift = pd.read_csv(drift_feat)
        st.markdown("### Feature Drift (KS Test)")
        styled_dataframe(drift, height=300)

# ================= PAGE 11: Reports =================
elif page == "11. Reports":
    section_header("Reports", "📄", "Investigation PDFs • CSV exports • Model evaluation • Audit logs")
    guide_box("New Report Creation", 
              "1) Select provider, 2) Choose components (SHAP, claims, notes), 3) Generate PDF, 4) Download, 5) View existing reports.",
              "💡")
    
    tab1, tab2, tab3 = st.tabs(["Investigation PDF","CSV & Data Exports","Audit Logs"])
    
    with tab1:
        st.markdown("### Automated PDF Investigation Report")
        provider_list = enriched["Provider"].astype(str).tolist() if not enriched.empty else []
        sel = st.selectbox("Select Provider for Report", provider_list[:200], key="pdf_provider")
        include_shap = st.checkbox("Include SHAP explanation", value=True)
        include_claims = st.checkbox("Include claim sample", value=True)
        notes_input = st.text_area("Investigation Notes", placeholder="e.g., Reviewed billing patterns...")
        
        if sel and action_button("Generate PDF Report", "📄"):
            with st.spinner("Generating PDF..."):
                try:
                    from src.pdf_report import generate_investigation_pdf
                    row = enriched[enriched["Provider"]==sel].iloc[0] if sel in enriched["Provider"].values else None
                    if row is not None:
                        out_path = Path(f"reports/investigation_report_{sel}.pdf")
                        generate_investigation_pdf(
                            provider_id=sel, hybrid_row=row.to_dict(),
                            master_row=master[master["Provider"]==sel].iloc[0].to_dict() if master is not None and sel in master["Provider"].values else row.to_dict(),
                            provider_info=get_provider_detail(sel) if 'get_provider_detail' in globals() else {},
                            investigation_notes=notes_input, include_shap=include_shap, output_path=out_path
                        )
                        st.success(f"PDF generated: {out_path}")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

# ================= PAGE 12: Settings =================
elif page == "12. Settings":
    section_header("Settings", "⚙️", "Risk thresholds • Model selection • App info • Database")
    guide_box("Configuration Guide", 
              "1) Adjust risk thresholds, 2) View application info, 3) Database management, 4) Re-initialize if needed.",
              "💡")
    
    tab1, tab2, tab3 = st.tabs(["Risk Configuration","Model & App Info","Database"])
    
    with tab1:
        st.markdown("### Risk Threshold Configuration")
        c1,c2,c3,c4 = st.columns(4)
        with c1: low = st.slider("Low: 0 -", 10, 50, 30)
        with c2: medium = st.slider("Medium: Low -", 40, 80, 60)
        with c3: high = st.slider("High: Medium -", 60, 90, 80)
        with c4: crit = st.slider("Critical: High - 100", 80, 100, 80)
        st.json({"Low": (0, low), "Medium": (low, medium), "High": (medium, high), "Critical": (high, 100)})

# ================= PAGE 13: Customer Journey Flow =================
elif page == "13. Customer Journey Flow":
    section_header("Customer Journey Flow", "👤➡️🏥", "End-to-end insurance claim process with AI fraud detection integration")
    guide_box("Unique Project Feature", 
              "This is the complete customer journey from registration to notification, showing where AI fraud detection fits in the pipeline.",
              "💡")
    
    st.markdown("""
    <div class="flow-container">
        <!-- Customer Registration -->
        <div class="flow-step">
            <div class="flow-icon">👤</div>
            <div class="flow-text"><strong>1. Customer Register / Login</strong><br><span style="font-size:12px; color:#64748b;">Create account or authenticate</span></div>
        </div>
        <div class="flow-arrow">⬇️</div>
        
        <!-- Purchase Insurance -->
        <div class="flow-step">
            <div class="flow-icon">💳</div>
            <div class="flow-text"><strong>2. Purchase Insurance</strong><br><span style="font-size:12px; color:#64748b;">Select plan and complete payment</span></div>
        </div>
        <div class="flow-arrow">⬇️</div>
        
        <!-- Submit Claim -->
        <div class="flow-step">
            <div class="flow-icon">📋</div>
            <div class="flow-text"><strong>3. Submit Insurance Claim</strong><br><span style="font-size:12px; color:#64748b;">File claim with details and amounts</span></div>
        </div>
        <div class="flow-arrow">⬇️</div>
        
        <!-- Upload Documents -->
        <div class="flow-step">
            <div class="flow-icon">📎</div>
            <div class="flow-text"><strong>4. Upload Documents</strong><br><span style="font-size:12px; color:#64748b;">Medical reports, bills, receipts, evidence</span></div>
        </div>
        <div class="flow-arrow">⬇️</div>
        
        <!-- Data Validation -->
        <div class="flow-step">
            <div class="flow-icon">✅</div>
            <div class="flow-text"><strong>5. Claim Data Validation</strong><br><span style="font-size:12px; color:#64748b;">Verify completeness, format, and policy compliance</span></div>
        </div>
        <div class="flow-arrow">⬇️</div>
        
        <!-- AI Fraud Detection -->
        <div class="flow-step" style="border-left-color: #ef4444; background: #fee2e2;">
            <div class="flow-icon">🤖</div>
            <div class="flow-text"><strong>6. AI Fraud Detection</strong><br><span style="font-size:12px; color:#991b1b;">Hybrid scoring: Supervised + Anomaly + Peer</span></div>
        </div>
        <div class="flow-arrow">⬇️</div>
        
        <!-- Fraud Risk Scoring -->
        <div class="flow-step" style="border-left-color: #ef4444; background: #fee2e2;">
            <div class="flow-icon">🎯</div>
            <div class="flow-text"><strong>7. Fraud Risk Scoring</strong><br><span style="font-size:12px; color:#991b1b;">Calculate risk score (0-100)</span></div>
        </div>
        <div class="flow-arrow">⬇️</div>
        
        <!-- Risk Categorization -->
        <div style="display:flex; justify-content:space-around; margin:20px 0;">
            <!-- LOW -->
            <div style="text-align:center; width:30%;">
                <div style="background:#d1fae5; color:#065f46; padding:12px; border-radius:10px; border:1px solid #10b981; margin-bottom:8px;">
                    <strong>LOW (0-30)</strong><br>
                    <span style="font-size:12px;">Normal Review</span>
                </div>
                <div class="flow-arrow">⬇️</div>
                <div style="background:#d1fae5; color:#065f46; padding:8px; border-radius:8px; font-size:13px;">✅ Approved → Payment → Notification</div>
            </div>
            
            <!-- MEDIUM -->
            <div style="text-align:center; width:30%;">
                <div style="background:#fef3c7; color:#92400e; padding:12px; border-radius:10px; border:1px solid #f59e0b; margin-bottom:8px;">
                    <strong>MEDIUM (30-60)</strong><br>
                    <span style="font-size:12px;">Manual Review</span>
                </div>
                <div class="flow-arrow">⬇️</div>
                <div style="background:#fef3c7; color:#92400e; padding:8px; border-radius:8px; font-size:13px;">⚖️ Decision → Approved/Rejected</div>
            </div>
            
            <!-- HIGH -->
            <div style="text-align:center; width:30%;">
                <div style="background:#fee2e2; color:#991b1b; padding:12px; border-radius:10px; border:1px solid #ef4444; margin-bottom:8px;">
                    <strong>HIGH (60-100)</strong><br>
                    <span style="font-size:12px;">Priority Investigation</span>
                </div>
                <div class="flow-arrow">⬇️</div>
                <div style="background:#fee2e2; color:#991b1b; padding:8px; border-radius:8px; font-size:13px;">🚨 Investigation → Decision</div>
            </div>
        </div>
        
        <!-- Officer Decision -->
        <div class="flow-step">
            <div class="flow-icon">👮‍♂️</div>
            <div class="flow-text"><strong>8. Officer Decision</strong><br><span style="font-size:12px; color:#64748b;">Final determination based on evidence</span></div>
        </div>
        <div class="flow-arrow">⬇️</div>
        
        <!-- Outcomes -->
        <div style="display:flex; justify-content:space-around; margin:20px 0;">
            <!-- Approved -->
            <div style="text-align:center; width:45%;">
                <div style="background:#d1fae5; color:#065f46; padding:12px; border-radius:10px; border:1px solid #10b981; margin-bottom:8px;">
                    <strong>✅ APPROVED</strong>
                </div>
                <div class="flow-arrow">⬇️</div>
                <div style="background:#d1fae5; color:#065f46; padding:8px; border-radius:8px; font-size:13px;">💰 Claim Payment</div>
                <div class="flow-arrow">⬇️</div>
                <div style="background:#d1fae5; color:#065f46; padding:8px; border-radius:8px; font-size:13px;">📧 Customer Notification</div>
            </div>
            
            <!-- Rejected -->
            <div style="text-align:center; width:45%;">
                <div style="background:#fee2e2; color:#991b1b; padding:12px; border-radius:10px; border:1px solid #ef4444; margin-bottom:8px;">
                    <strong>❌ REJECTED</strong>
                </div>
                <div class="flow-arrow">⬇️</div>
                <div style="background:#fee2e2; color:#991b1b; padding:8px; border-radius:8px; font-size:13px;">📝 Reason Recorded</div>
                <div class="flow-arrow">⬇️</div>
                <div style="background:#fee2e2; color:#991b1b; padding:8px; border-radius:8px; font-size:13px;">📧 Customer Notification</div>
            </div>
        </div>
        
        <!-- Final Notification -->
        <div class="flow-step">
            <div class="flow-icon">📧</div>
            <div class="flow-text"><strong>9. Customer Notification</strong><br><span style="font-size:12px; color:#64748b;">Final status communicated to customer</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    section_header("Project Presentation Tips", "💡", "How to explain this flow to stakeholders")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        <div class="feature-card">
            <strong style="font-size:18px;">1. Customer Experience</strong><br>
            <span style="color:#64748b; font-size:13px;">Seamless journey from registration to notification with transparent AI integration.</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <strong style="font-size:18px;">2. Risk-Based Triage</strong><br>
            <span style="color:#64748b; font-size:13px;">Low risk → Automatic, Medium → Manual, High → Priority investigation.</span>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown("""
        <div class="feature-card">
            <strong style="font-size:18px;">3. AI Transparency</strong><br>
            <span style="color:#64748b; font-size:13px;">Explainable scoring components visible to officers and customers (where applicable).</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <strong style="font-size:18px;">4. Audit & Compliance</strong><br>
            <span style="color:#64748b; font-size:13px;">Full traceability from claim submission to final notification with investigation logs.</span>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2026 FraudShield AI | B.Tech CSE Final Year Project | Human-in-the-loop • Explainable • Local-first | High-risk ≠ proof of fraud")
