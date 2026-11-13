"""
PDF Report Generation - Phase 11C
Professional investigation PDF with ReportLab.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime
import pandas as pd
import json

REPORT_DIR = Path("reports")

def generate_investigation_pdf(provider_id: str, hybrid_row: dict, master_row: dict, provider_info: dict, investigation_notes: str, include_shap: bool, output_path: Path):
    """Generate professional PDF report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=0.6*inch, bottomMargin=0.6*inch, leftMargin=0.7*inch, rightMargin=0.7*inch,
                            title=f"Investigation Report - {provider_id}", author="FraudShield AI")
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleCustom', parent=styles['Title'], fontSize=22, textColor=colors.HexColor("#0f172a"), alignment=TA_CENTER, spaceAfter=6)
    heading_style = ParagraphStyle('HeadingCustom', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor("#1e293b"), spaceBefore=12, spaceAfter=6, borderPadding=(3,0,3), backColor=colors.HexColor("#f1f5f9"))
    subheading_style = ParagraphStyle('SubHeading', parent=styles['Heading3'], fontSize=11, textColor=colors.HexColor("#334155"))
    body_style = ParagraphStyle('BodyCustom', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor("#334155"))
    small_style = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor("#64748b"))
    badge_style = ParagraphStyle('Badge', parent=styles['Normal'], fontSize=9, textColor=colors.white, backColor=colors.HexColor("#ef4444"), borderPadding=(3,6,3), alignment=TA_CENTER)
    
    story = []
    
    # Header
    story.append(Paragraph("🏥 FraudShield AI", ParagraphStyle('Brand', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#0f172a"), alignment=TA_CENTER, fontName="Helvetica-Bold")))
    story.append(Paragraph("Healthcare Insurance Fraud Detection & Investigation Platform", ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor("#64748b"), alignment=TA_CENTER, spaceAfter=8)))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=12))
    
    story.append(Paragraph("PROVIDER INVESTIGATION REPORT", title_style))
    story.append(Paragraph(f"Confidential • Human Review Required • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", small_style))
    story.append(Spacer(1, 12))
    
    # Risk Assessment Summary Box
    risk_cat = hybrid_row.get("RiskCategory", "Unknown")
    hybrid_score = hybrid_row.get("HybridRiskScore", 0)
    # Color based on risk
    risk_colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444", "Critical": "#991b1b"}
    risk_color = risk_colors.get(risk_cat, "#6b7280")
    
    # Summary table
    summary_data = [
        [Paragraph("<b>Provider ID</b>", body_style), Paragraph(f"<b>{provider_id}</b>", heading_style)],
        [Paragraph("Hybrid Risk Score", body_style), Paragraph(f"<b>{hybrid_score:.1f} / 100</b>  <font color='{risk_color}'><b>{risk_cat}</b></font>", body_style)],
        [Paragraph("Fraud Probability (Supervised)", body_style), Paragraph(f"{hybrid_row.get('SupervisedScore',0):.1f} / 100  ({hybrid_row.get('SupervisedScore',0)/100:.1%})", body_style)],
        [Paragraph("Anomaly Score (Isolation Forest)", body_style), Paragraph(f"{hybrid_row.get('AnomalyScore',0):.1f} / 100", body_style)],
        [Paragraph("Peer Deviation Score", body_style), Paragraph(f"{hybrid_row.get('PeerScore',0):.1f} / 100", body_style)],
        [Paragraph("Assessment Date", body_style), Paragraph(datetime.now().strftime("%Y-%m-%d"), body_style)],
        [Paragraph("Model Version", body_style), Paragraph(str(Path("models/model_metadata.json").exists() and json.load(open("models/model_metadata.json")).get("model_version","v1.0.0") or "v1.0.0"), small_style)],
    ]
    t = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph("⚖️ <b>Legal Disclaimer:</b> This risk score is an <i>investigation signal</i>, not proof of fraud. All findings require human validation against clinical records and policy rules before any determination. No automatic claim denial.", small_style))
    story.append(Spacer(1, 12))
    
    # Provider Profile
    story.append(Paragraph("1. Provider Profile & Claim Summary", heading_style))
    profile_data = [
        [Paragraph("<b>Metric</b>", subheading_style), Paragraph("<b>Value</b>", subheading_style), Paragraph("<b>Context</b>", subheading_style)],
        [Paragraph("Total Claims", body_style), Paragraph(f"{int(master_row.get('TotalClaims',0)):,}", body_style), Paragraph("All inpatient + outpatient", small_style)],
        [Paragraph("Total Reimbursement", body_style), Paragraph(f"${master_row.get('TotalReimbursement',0):,.2f}", body_style), Paragraph("Sum of InscClaimAmtReimbursed", small_style)],
        [Paragraph("Average Reimbursement", body_style), Paragraph(f"${master_row.get('AverageReimbursement',0):,.2f}", body_style), Paragraph("Per-claim average", small_style)],
        [Paragraph("Unique Beneficiaries", body_style), Paragraph(f"{int(master_row.get('UniqueBeneficiaries',0)):,}", body_style), Paragraph("Distinct patients", small_style)],
        [Paragraph("Inpatient Claims", body_style), Paragraph(f"{int(master_row.get('InpatientClaims',0)):,}", body_style), Paragraph(f"Ratio: {master_row.get('InpatientClaims',0)/max(master_row.get('TotalClaims',1),1)*100:.1f}%", small_style)],
        [Paragraph("Outpatient Claims", body_style), Paragraph(f"{int(master_row.get('OutpatientClaims',0)):,}", body_style), Paragraph(f"Ratio: {master_row.get('OutpatientClaims',0)/max(master_row.get('TotalClaims',1),1)*100:.1f}%", small_style)],
        [Paragraph("Average Claim Duration", body_style), Paragraph(f"{master_row.get('AverageClaimDuration',0):.2f} days", body_style), Paragraph("ClaimEnd - ClaimStart", small_style)],
        [Paragraph("Ground Truth Label", body_style), Paragraph(str(master_row.get('PotentialFraud','Unknown')), body_style), Paragraph("Train set only; Yes = provider flagged in historical review", small_style)],
    ]
    t2 = Table(profile_data, colWidths=[2.0*inch, 1.5*inch, 2.5*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))
    if master_row.get("PeerGroup"):
        story.append(Paragraph(f"<b>Peer Group:</b> {master_row.get('PeerGroup')} (by claim volume) • Peer comparison uses robust median/MAD deviation: z = (x − median)/(1.4826×MAD)", small_style))
    story.append(Spacer(1, 12))
    
    # Suspicious Indicators
    story.append(Paragraph("2. Suspicious Indicators", heading_style))
    # Collect peer alerts for this provider
    peer_path = Path("reports/peer_group_alerts.csv")
    indicators = []
    if peer_path.exists():
        try:
            peer = pd.read_csv(peer_path)
            prov_peer = peer[peer["Provider"]==provider_id]
            for _, r in prov_peer.iterrows():
                indicators.append([Paragraph(r["Feature"], body_style), Paragraph(f"{r['Value']:.2f}", body_style), Paragraph(f"Peer median {r['PeerMedian']:.2f}", small_style), Paragraph(f"z={r['ZScore']:.2f} ({r['Severity']})", body_style)])
        except:
            pass
    # Temporal
    temp_path = Path("reports/temporal_anomaly_alerts.csv")
    if temp_path.exists():
        try:
            temp = pd.read_csv(temp_path)
            prov_temp = temp[temp["Provider"]==provider_id]
            for _, r in prov_temp.iterrows():
                indicators.append([Paragraph(r["Metric"], body_style), Paragraph(str(r["Value"]), body_style), Paragraph(r["YearMonth"], small_style), Paragraph(r["Context"][:60], small_style)])
        except:
            pass
    if not indicators:
        indicators = [[Paragraph("No extreme peer deviations", body_style), Paragraph("—", body_style), Paragraph("Peer z within ±2.5", small_style), Paragraph("Low", body_style)]]
    
    ind_data = [[Paragraph("<b>Indicator</b>", subheading_style), Paragraph("<b>Provider Value</b>", subheading_style), Paragraph("<b>Peer/Context</b>", subheading_style), Paragraph("<b>Severity</b>", subheading_style)]] + indicators[:10]
    t3 = Table(ind_data, colWidths=[1.6*inch, 1.2*inch, 1.8*inch, 1.4*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t3)
    story.append(Spacer(1, 6))
    story.append(Paragraph("Each indicator warrants administrative/clinical review; high value alone is not proof. See duplicate, temporal, and network modules for additional evidence.", small_style))
    story.append(Spacer(1, 12))
    
    # SHAP Explanation
    if include_shap:
        story.append(Paragraph("3. Model Explainability (SHAP)", heading_style))
        shap_path = Path(f"reports/shap_waterfall_{provider_id}.png")
        # Try to generate if not exists
        shap_imp_path = Path("reports/shap_feature_importance.csv")
        if shap_imp_path.exists():
            try:
                imp = pd.read_csv(shap_imp_path)
                shap_data = [[Paragraph("<b>Feature</b>", subheading_style), Paragraph("<b>Mean |SHAP|</b>", subheading_style), Paragraph("<b>Interpretation</b>", subheading_style)]]
                for _, r in imp.head(8).iterrows():
                    interp = "Increases risk" if r["Feature"] in ["TotalReimbursement","AverageReimbursement","DuplicateClaimRatio"] else "Model-weighted"
                    shap_data.append([Paragraph(r["Feature"], body_style), Paragraph(f"{r['MeanAbsSHAP']:.4f}", body_style), Paragraph(interp, small_style)])
                t4 = Table(shap_data, colWidths=[2.0*inch, 1.2*inch, 2.8*inch])
                t4.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
                ]))
                story.append(t4)
                story.append(Spacer(1, 6))
                if shap_path.exists():
                    story.append(Image(str(shap_path), width=6*inch, height=3*inch))
                else:
                    story.append(Paragraph("SHAP waterfall plot for this provider can be generated in the Explainable AI dashboard page (select provider → Generate SHAP). Global SHAP summary shown below.", body_style))
                    # Try global SHAP image
                    global_shap = Path("reports/shap_summary.png")
                    if global_shap.exists():
                        story.append(Image(str(global_shap), width=6*inch, height=4*inch))
            except Exception as e:
                story.append(Paragraph(f"SHAP details unavailable: {e}", small_style))
        story.append(Paragraph("SHAP values indicate contribution to model prediction, not causal fraud proof. Positive SHAP pushes score up; negative pushes down. Validate with domain expertise.", small_style))
        story.append(Spacer(1, 12))
    
    # Peer Comparison
    story.append(Paragraph("4. Peer-Group Comparison", heading_style))
    peer_comp_data = [
        [Paragraph("<b>Metric</b>", subheading_style), Paragraph("<b>Provider</b>", subheading_style), Paragraph("<b>Peer Median</b>", subheading_style), Paragraph("<b>Peer Z</b>", subheading_style)]
    ]
    for feat in ["TotalReimbursement","AverageReimbursement","ClaimsPerBeneficiary","AverageClaimDuration","DuplicateClaimRatio"]:
        if feat in master_row and f"{feat}_PeerMedian" in master_row:
            peer_comp_data.append([
                Paragraph(feat, body_style),
                Paragraph(f"{master_row[feat]:,.2f}", body_style),
                Paragraph(f"{master_row.get(feat+'_PeerMedian',0):,.2f}", body_style),
                Paragraph(f"{master_row.get(feat+'_PeerZ',0):.2f}", body_style)
            ])
    if len(peer_comp_data) > 1:
        t5 = Table(peer_comp_data, colWidths=[1.8*inch, 1.4*inch, 1.4*inch, 1.4*inch])
        t5.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#334155")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(t5)
    else:
        story.append(Paragraph("Peer comparison data not available; run feature engineering with peer groups.", small_style))
    story.append(Spacer(1, 12))
    
    # Claim Summaries (sample)
    story.append(Paragraph("5. Claim Sample & Evidence", heading_style))
    # Load raw claims sample for this provider
    from pathlib import Path as P
    raw_inp = sorted(P("data/raw").glob("Train_Inpatientdata-*.csv"))
    raw_out = sorted(P("data/raw").glob("Train_Outpatientdata-*.csv"))
    claims_sample = []
    for path in (raw_inp + raw_out)[:2]:
        if path:
            try:
                df = pd.read_csv(path, low_memory=False, nrows=20000)
                subset = df[df["Provider"]==provider_id].head(5)
                for _, r in subset.iterrows():
                    claims_sample.append([Paragraph(str(r.get("ClaimID","")), small_style), Paragraph(str(r.get("BeneID",""))[:8], small_style), Paragraph(str(r.get("ClaimStartDt","")), small_style), Paragraph(f"${r.get('InscClaimAmtReimbursed',0):,.0f}", body_style)])
            except:
                pass
    if claims_sample:
        claim_data = [[Paragraph("<b>ClaimID</b>", subheading_style), Paragraph("<b>BeneID</b>", subheading_style), Paragraph("<b>Claim Date</b>", subheading_style), Paragraph("<b>Amount</b>", subheading_style)]] + claims_sample
        t6 = Table(claim_data, colWidths=[1.8*inch, 1.4*inch, 1.5*inch, 1.3*inch])
        t6.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(t6)
    else:
        story.append(Paragraph("No claim sample available (large raw files not scanned for brevity). Review inpatient/outpatient CSVs for full evidence.", small_style))
    story.append(Spacer(1, 12))
    
    # Investigation Notes
    story.append(Paragraph("6. Investigation Notes", heading_style))
    if investigation_notes and investigation_notes.strip():
        story.append(Paragraph(investigation_notes, body_style))
    else:
        story.append(Paragraph("No investigator notes provided. Use the Investigation Centre to add timestamped notes, status updates, and audit trail.", small_style))
    story.append(Spacer(1, 12))
    
    # Model & Caveat
    story.append(Paragraph("7. Model & Compliance Information", heading_style))
    story.append(Paragraph(f"<b>Model:</b> {json.load(open('models/model_metadata.json')).get('best_model','Unknown') if Path('models/model_metadata.json').exists() else 'Unknown'} • <b>Trained:</b> {json.load(open('models/model_metadata.json')).get('training_date','N/A') if Path('models/model_metadata.json').exists() else 'N/A'} • <b>PR-AUC:</b> {json.load(open('models/model_metadata.json')).get('test_metrics',{}).get('pr_auc','N/A') if Path('models/model_metadata.json').exists() else 'N/A'}", body_style))
    story.append(Paragraph("<b>Dataset:</b> US Medicare Provider Fraud Detection Analysis (Kaggle) • <b>Geographic Caveat:</b> US Medicare-oriented; not validated for Indian insurance schemes without local data/rules/validation.", small_style))
    story.append(Paragraph("<b>Methodology:</b> Hybrid 0.55×Supervised + 0.25×Anomaly + 0.20×Peer • Thresholds Low 0-30, Medium 30-60, High 60-80, Critical 80-100 • Review validation_model_comparison.csv for model choice.", small_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(f"Report generated by FraudShield AI • {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} • Confidential • For authorized investigative use only. High-risk score is an investigation signal, not proof of fraud.", small_style))
    
    doc.build(story)
    print(f"PDF saved to {output_path}")

if __name__ == "__main__":
    # Test
    import pandas as pd
    hybrid = pd.read_csv("reports/hybrid_risk_scores.csv")
    master = pd.read_csv("reports/master_provider_features.csv")
    provider_id = hybrid.iloc[0]["Provider"]
    generate_investigation_pdf(
        provider_id=provider_id,
        hybrid_row=hybrid[hybrid["Provider"]==provider_id].iloc[0].to_dict(),
        master_row=master[master["Provider"]==provider_id].iloc[0].to_dict(),
        provider_info={},
        investigation_notes="Test note: reviewed peer deviation in TotalReimbursement (z=3.2). See claims sample.",
        include_shap=True,
        output_path=Path(f"reports/investigation_report_{provider_id}.pdf")
    )
