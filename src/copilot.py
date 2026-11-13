"""
Fraud Investigation Copilot - Phase 11A
Rule-based assistant that answers provider questions using actual DB/records.
No paid APIs; grounded in data.
"""
import pandas as pd
from pathlib import Path
import json

REPORTS_DIR = Path("reports")
MODEL_DIR = Path("models")

def load_context(provider_id: str) -> dict:
    """Load all available context for a provider."""
    ctx = {"provider_id": provider_id}
    # Hybrid scores
    hybrid_path = REPORTS_DIR / "hybrid_risk_scores.csv"
    if hybrid_path.exists():
        hybrid = pd.read_csv(hybrid_path)
        row = hybrid[hybrid["Provider"]==provider_id]
        if not row.empty:
            ctx["hybrid"] = row.iloc[0].to_dict()
    # Master features
    master_path = REPORTS_DIR / "master_provider_features.csv"
    if not master_path.exists():
        master_path = Path("data/processed/cleaned_provider_features.csv")
    if master_path.exists():
        master = pd.read_csv(master_path)
        row = master[master["Provider"]==provider_id]
        if not row.empty:
            ctx["master"] = row.iloc[0].to_dict()
    # SHAP if available
    shap_path = REPORTS_DIR / "shap_feature_importance.csv"
    if shap_path.exists():
        ctx["shap_global"] = pd.read_csv(shap_path).head(5).to_dict(orient="records")
    # Peer alerts
    peer_path = REPORTS_DIR / "peer_group_alerts.csv"
    if peer_path.exists():
        peer = pd.read_csv(peer_path)
        ctx["peer_alerts"] = peer[peer["Provider"]==provider_id].to_dict(orient="records")
    # Temporal
    temp_path = REPORTS_DIR / "temporal_anomaly_alerts.csv"
    if temp_path.exists():
        temp = pd.read_csv(temp_path)
        ctx["temporal_alerts"] = temp[temp["Provider"]==provider_id].to_dict(orient="records")
    # Isolation forest
    iso_path = REPORTS_DIR / "isolation_forest_scores.csv"
    if iso_path.exists():
        iso = pd.read_csv(iso_path)
        row = iso[iso["Provider"]==provider_id]
        if not row.empty:
            ctx["anomaly"] = row.iloc[0].to_dict()
    return ctx

def answer_question(provider_id: str, question: str) -> dict:
    """
    Rule-based answering grounded in data.
    Matches question intent to context.
    """
    question_lower = question.lower()
    ctx = load_context(provider_id)
    
    if not ctx.get("hybrid") and not ctx.get("master"):
        return {
            "answer": f"No data found for provider {provider_id}. Verify provider ID exists in training set.",
            "sources": [],
            "grounded": False
        }
    
    hybrid = ctx.get("hybrid", {})
    master = ctx.get("master", {})
    peer_alerts = ctx.get("peer_alerts", [])
    temporal = ctx.get("temporal_alerts", [])
    
    answer = ""
    sources = []
    grounded = True
    
    # Intent: why high risk?
    if any(k in question_lower for k in ["why", "high-risk", "high risk", "risk", "flagged", "suspicious"]):
        risk_score = hybrid.get("HybridRiskScore", "N/A")
        risk_cat = hybrid.get("RiskCategory", "Unknown")
        sup = hybrid.get("SupervisedScore", 0)
        anom = hybrid.get("AnomalyScore", 0)
        peer = hybrid.get("PeerScore", 0)
        answer += f"Provider {provider_id} has a hybrid risk score of {risk_score:.1f} ({risk_cat}). "
        sources.append(f"hybrid_risk_scores.csv: HybridRiskScore={risk_score}, RiskCategory={risk_cat}")
        answer += f"This combines supervised fraud probability {sup:.1f}/100 (weight 55%), anomaly score {anom:.1f}/100 (25%), and peer deviation {peer:.1f}/100 (20%). "
        
        if peer_alerts:
            top_peer = peer_alerts[0]
            answer += f"Key suspicious indicator: {top_peer.get('Feature')} is {top_peer.get('Value'):.2f} vs peer median {top_peer.get('PeerMedian'):.2f} (z={top_peer.get('ZScore'):.2f}, {top_peer.get('Severity')}). "
            sources.append(f"peer_group_alerts.csv: {top_peer.get('Feature')} z={top_peer.get('ZScore'):.2f}")
        if temporal:
            answer += f"Temporal alert: {temporal[0].get('Metric')} on {temporal[0].get('YearMonth')} ({temporal[0].get('Context')}). "
            sources.append(f"temporal_anomaly_alerts.csv: {temporal[0].get('Metric')}")
        
        # Fraud label context
        if master.get("PotentialFraud"):
            answer += f"Historical label: PotentialFraud={master.get('PotentialFraud')} (training set). "
            sources.append("master_provider_features.csv: PotentialFraud")
        
        answer += "High-risk is an investigation signal, not proof of fraud. Review SHAP explanation and claim samples for evidence."
    
    elif any(k in question_lower for k in ["peer", "comparison", "median", "deviation"]):
        if peer_alerts:
            answer = f"Peer-group comparison (group: {master.get('PeerGroup','Unknown')} by claim volume):\n"
            for p in peer_alerts[:3]:
                answer += f"- {p.get('Feature')}: {p.get('Value'):.2f} vs peer median {p.get('PeerMedian'):.2f} (z={p.get('ZScore'):.2f}, {p.get('Severity')})\n"
                sources.append(f"peer_group_alerts.csv: {p.get('Feature')}")
        else:
            answer = f"No extreme peer deviations found for {provider_id} (all z within ±2.5). Provider's metrics are typical for its peer group {master.get('PeerGroup','')}."
    
    elif any(k in question_lower for k in ["shap", "feature", "contribute", "explanation"]):
        # Try individual SHAP
        try:
            from src.explainability import explain_provider
            expl = explain_provider(provider_id, save_plot=False)
            if "error" not in expl:
                top = expl.get("TopContributingFeatures", [])[:3]
                answer = f"Top features driving the prediction for {provider_id} (fraud prob {expl.get('PredictedFraudProbability',0):.3f}):\n"
                for t in top:
                    direction = "increases" if t["SHAPValue"]>0 else "decreases"
                    answer += f"- {t['Feature']} = {t['FeatureValue']:.2f} {direction} risk (SHAP {t['SHAPValue']:+.4f})\n"
                    sources.append(f"SHAP: {t['Feature']} SHAP={t['SHAPValue']:.4f}")
                answer += "SHAP shows contribution to model output, not causality. Positive = pushes toward fraud prediction."
            else:
                answer = expl.get("error")
        except Exception as e:
            answer = f"SHAP explanation not available for {provider_id}: {e}. Ensure SHAP is installed and model computed."
    
    elif any(k in question_lower for k in ["claim", "reimbursement", "billing", "duplicate"]):
        total_claims = master.get("TotalClaims",0)
        total_reimb = master.get("TotalReimbursement",0)
        avg_reimb = master.get("AverageReimbursement",0)
        answer = f"Billing summary for {provider_id}: {total_claims} total claims, ${total_reimb:,.2f} total reimbursement, ${avg_reimb:,.2f} avg per claim. "
        sources.append(f"master_provider_features.csv: TotalClaims={total_claims}, TotalReimbursement={total_reimb}")
        # Duplicate info
        dup_path = REPORTS_DIR / "duplicate_provider_summary.csv"
        if dup_path.exists():
            dup = pd.read_csv(dup_path)
            row = dup[dup["Provider"]==provider_id]
            if not row.empty:
                answer += f"Duplicate screening flagged {row.iloc[0].to_dict()} for review (not proof of fraud). "
                sources.append("duplicate_provider_summary.csv")
        # Inpatient/outpatient
        if master.get("InpatientClaims") is not None:
            answer += f"Inpatient {master.get('InpatientClaims')} ({master.get('InpatientClaims',0)/max(total_claims,1)*100:.1f}%), Outpatient {master.get('OutpatientClaims')}."
    
    elif any(k in question_lower for k in ["temporal", "monthly", "trend", "spike"]):
        if temporal:
            answer = f"Temporal analysis for {provider_id}: "
            for t in temporal[:2]:
                answer += f"{t.get('YearMonth')} {t.get('Metric')} value {t.get('Value')} (z={t.get('ZScore'):.2f}, expected {t.get('ExpectedMean'):.1f}) – {t.get('Context')}. "
                sources.append(f"temporal_anomaly_alerts.csv: {t.get('YearMonth')}")
        else:
            answer = f"No temporal spikes detected for {provider_id}. Monthly claim volume is stable within peer expectations. Check temporal_provider_features.csv for average monthly claims {master.get('AvgMonthlyClaims','N/A')}."
    
    elif any(k in question_lower for k in ["network", "physician", "beneficiary", "shared", "collusion"]):
        # Check network shared
        shared_bene = REPORTS_DIR / "network_shared_beneficiaries.csv"
        if shared_bene.exists():
            shared = pd.read_csv(shared_bene)
            related = shared[(shared["Provider1"]==provider_id) | (shared["Provider2"]==provider_id)]
            if not related.empty:
                answer = f"Network: {provider_id} shares beneficiaries with other providers:\n"
                for _, r in related.head(3).iterrows():
                    other = r["Provider2"] if r["Provider1"]==provider_id else r["Provider1"]
                    answer += f"- Shares {r['SharedBeneficiaries']} beneficiaries with {other}\n"
                    sources.append(f"network_shared_beneficiaries.csv: {provider_id} ↔ {other} = {r['SharedBeneficiaries']}")
                answer += "Shared connections are co-occurrence, not proof of collusion. Requires identifier validation and clinical review."
            else:
                answer = f"No unusually high shared beneficiary/physician links found for {provider_id} in network analysis."
        else:
            answer = "Network analysis not yet run. Generate via src/network_analysis.py."
    
    elif any(k in question_lower for k in ["probability", "predict", "score"]):
        sup = hybrid.get("SupervisedScore",0)
        hybrid_score_val = hybrid.get("HybridRiskScore",0)
        answer = f"Supervised fraud probability: {sup/100:.3f} ({sup:.1f}/100). Hybrid risk score: {hybrid_score_val:.1f}/100 ({hybrid.get('RiskCategory','Unknown')}). Methodology: hybrid = 0.55×supervised + 0.25×anomaly + 0.20×peer, normalized 0-100. Thresholds: Low 0-30, Medium 30-60, High 60-80, Critical 80-100."
        sources.append("hybrid_scoring_config.json: weights supervised 0.55, anomaly 0.25, peer 0.20")
    
    else:
        # Default overview
        answer = f"Provider {provider_id} overview: Hybrid {hybrid.get('HybridRiskScore',0):.1f} ({hybrid.get('RiskCategory','')}), Supervised {hybrid.get('SupervisedScore',0):.1f}, Anomaly {hybrid.get('AnomalyScore',0):.1f}, Peer {hybrid.get('PeerScore',0):.1f}. "
        if peer_alerts:
            answer += f"Top peer deviation: {peer_alerts[0].get('Feature')} z={peer_alerts[0].get('ZScore'):.2f}. "
        answer += "Ask about SHAP explanations, peer comparison, claims, temporal, network, or why-risk."
        sources.append("hybrid_risk_scores.csv + master_provider_features.csv")
    
    # Always add disclaimer
    answer += "\n\n⚖️ Note: High-risk is an investigation signal, not proof of fraud. Validate with human review and policy rules."
    
    return {
        "provider_id": provider_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "grounded": grounded,
        "context_used": list(ctx.keys())
    }

def list_example_questions():
    return [
        "Why was this provider assigned a high-risk score?",
        "What features contributed most to the fraud prediction?",
        "Show peer-group comparison for this provider.",
        "Summarize billing patterns and duplicate claims.",
        "Are there temporal spikes in claim volume?",
        "Does this provider share beneficiaries with others?",
        "What is the fraud probability vs hybrid score?"
    ]
