"""
UI Components - reusable cards, charts, helpers
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def kpi_card(title, value, delta=None, help_text=None):
    with st.container():
        st.markdown(f"""
        <div style="background: white; padding: 16px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
            <div style="font-size: 13px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
            <div style="font-size: 28px; font-weight: 700; color: #111827; margin-top: 6px;">{value}</div>
            {f'<div style="font-size: 13px; color: #059669; margin-top: 4px;">{delta}</div>' if delta else ''}
            {f'<div style="font-size: 11px; color: #9ca3af; margin-top: 6px;">{help_text}</div>' if help_text else ''}
        </div>
        """, unsafe_allow_html=True)

def risk_badge(category):
    colors = {
        "Low": ("#10b981", "#ecfdf5"),
        "Medium": ("#f59e0b", "#fffbeb"),
        "High": ("#ef4444", "#fef2f2"),
        "Critical": ("#991b1b", "#fef2f2")
    }
    fg, bg = colors.get(category, ("#6b7280", "#f9fafb"))
    st.markdown(f"<span style='background:{bg}; color:{fg}; padding:4px 10px; border-radius:999px; font-weight:600; font-size:12px; border:1px solid {fg}20;'>{category}</span>", unsafe_allow_html=True)

def styled_dataframe(df, height=400):
    st.dataframe(df, use_container_width=True, height=height)

def plotly_chart(fig, height=400):
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=height, paper_bgcolor='white', plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

def disclaimer_box(text="High-risk score is an investigation signal, not proof of fraud. Human review required."):
    st.info(f"⚖️ **Investigation Notice:** {text}")

def caveat_box(text="US Medicare-oriented benchmark; not validated for other insurance schemes."):
    st.warning(f"🌍 **Geographic Caveat:** {text}")
