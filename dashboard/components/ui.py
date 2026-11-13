"""
UI Components - modern, user-friendly, responsive cards, charts, and navigation guides
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Global Modern CSS ---
st.markdown("""
<style>
    /* Modern dark/light theme adaptive */
    @media (prefers-color-scheme: dark) {
        .main > div { background-color: #0f172a; color: #e2e8f0; }
        .stMetric { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
    } else {
        .main > div { background-color: #f8fafc; color: #1e293b; }
        .stMetric { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%); }
    }
    .guide-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 16px 0;
        font-size: 14px;
    }
    .guide-box-dark {
        background: #1e3a8a;
        border-left: 4px solid #60a5fa;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 16px 0;
        font-size: 14px;
        color: #e2e8f0;
    }
    .card-container {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    @media (prefers-color-scheme: dark) {
        .card-container {
            background: #1e293b;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
    }
</style>
""", unsafe_allow_html=True)

def guide_box(title, text, icon="💡"):
    """User-friendly guidance box"""
    st.markdown(f"""
    <div class="guide-box">
        <strong>{icon} {title}</strong><br>
        {text}
    </div>
    """, unsafe_allow_html=True)

def kpi_card(title, value, delta=None, help_text=None, icon="📊"):
    """Modern KPI card with icon and smooth styling"""
    with st.container():
        st.markdown(f"""
        <div class="card-container">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="font-size: 14px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    {icon} {title}
                </div>
                <div style="font-size: 18px; opacity: 0.8;">{help_text or ''}</div>
            </div>
            <div style="font-size: 32px; font-weight: 800; color: #0f172a; margin-top: 8px; line-height: 1.1;">
                {value}
            </div>
            {f'<div style="font-size: 14px; color: #10b981; margin-top: 6px; font-weight: 600;">▲ {delta}</div>' if delta and '+' in str(delta) else ''}
            {f'<div style="font-size: 14px; color: #ef4444; margin-top: 6px; font-weight: 600;">▼ {delta}</div>' if delta and '-' in str(delta) else ''}
        </div>
        """, unsafe_allow_html=True)

def risk_badge(category):
    """Beautiful risk badge with gradient and animation"""
    colors = {
        "Low": ("#10b981", "#d1fae5", "#065f46"),
        "Medium": ("#f59e0b", "#fef3c7", "#92400e"),
        "High": ("#ef4444", "#fee2e2", "#991b1b"),
        "Critical": ("#991b1b", "#fee2e2", "#7f1d1d")
    }
    fg, bg, border = colors.get(category, ("#6b7280", "#f9fafb", "#9ca3af"))
    st.markdown(f"""
    <div style="display:inline-block; background:{bg}; color:{fg}; padding:6px 16px; border-radius:9999px; font-weight:700; font-size:13px; border:1px solid {border}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        {category}
    </div>
    """, unsafe_allow_html=True)

def styled_dataframe(df, height=400):
    """Clean dataframe with modern styling"""
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)

def plotly_chart(fig, height=400):
    """Clean Plotly chart with modern theme"""
    fig.update_layout(
        margin=dict(l=10,r=10,t=30,b=10),
        height=height,
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def disclaimer_box(text="High-risk score is an investigation signal, not proof of fraud. Human review required."):
    """Professional disclaimer with icon"""
    st.markdown(f"""
    <div style="background: #fff7ed; border: 1px solid #fcd34d; padding: 16px; border-radius: 10px; margin: 16px 0;">
        <div style="font-size: 15px; font-weight: 600; color: #92400e; margin-bottom: 8px;">⚖️ Investigation Notice</div>
        <div style="font-size: 14px; color: #78350f;">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def caveat_box(text="US Medicare-oriented benchmark; not validated for other insurance schemes."):
    """Geographic caveat with icon"""
    st.markdown(f"""
    <div style="background: #eff6ff; border: 1px solid #3b82f6; padding: 16px; border-radius: 10px; margin: 16px 0;">
        <div style="font-size: 15px; font-weight: 600; color: #1e40af; margin-bottom: 8px;">🌍 Geographic Caveat</div>
        <div style="font-size: 14px; color: #1e40af;">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def section_header(title, icon="🏥", subtitle=""):
    """Modern section header with icon and subtitle"""
    st.markdown(f"""
    <div style="margin: 24px 0 16px 0;">
        <h2 style="margin:0; font-size: 24px; font-weight: 800; color: #0f172a; display:flex; align-items:center; gap: 10px;">
            {icon} {title}
        </h2>
        <div style="color: #64748b; font-size: 14px; margin-top: 4px;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def action_button(label, icon="🚀", **kwargs):
    """Modern action button"""
    return st.button(f"{icon} {label}", **kwargs)

def navigation_guide(current_page):
    """Shows user where they are and what's next"""
    pages = [
        ("1. Executive Overview", "📊 High-level metrics & trends"),
        ("2. Provider Explorer", "🔍 Search & filter providers"),
        ("3. Fraud Analytics", "📈 Distribution & billing patterns"),
        ("4. AI Predictions", "🤖 Live inference & scoring"),
        ("5. Explainable AI", "🧠 SHAP & what-if analysis"),
        ("6. Anomaly Detection", "⚠️ Unusual behavior signals"),
        ("7. Network Analysis", "🕸️ Provider relationships"),
        ("8. Investigation Centre", "📋 Case management & workflow"),
        ("9. Model Performance", "🎯 Validation & test metrics"),
        ("10. Model Monitoring", "📊 Drift & version tracking"),
        ("11. Reports", "📄 PDF & CSV exports"),
        ("12. Settings", "⚙️ Configuration & database")
    ]
    
    st.markdown("---")
    st.caption("📍 **Navigation Guide**")
    cols = st.columns(min(len(pages), 4))
    for i, (name, desc) in enumerate(pages):
        with cols[i % len(cols)]:
            active = "🟢" if current_page == name else "⚪"
            st.markdown(f"<small>{active} <strong>{name.split('.')[0]}</strong><br><span style='color:#64748b; font-size:11px;'>{desc}</span></small>", unsafe_allow_html=True)
    st.markdown("---")
