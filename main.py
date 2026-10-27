"""
SGP-II: AI-Augmented Decision-Support Sandbox
===============================================

Streamlit Multi-Page Application Entry Point.

This is the central navigation hub for the SGP-II dashboard, using Streamlit's
native multi-page architecture via ``st.navigation()``. It registers all pages,
initializes shared session state, and configures global theming.

Pages
-----
1. **📊 Dashboard** — Daily recommendation matrix with signal highlights
2. **🏥 Fundamentals** — Fundamental universe inspector with health toggles
3. **📰 Sentiment** — LLM-powered sentiment feed and history
4. **🤖 ML Engine** — Model training, evaluation, and feature importance
5. **📈 Backtester** — Strategy backtesting with interactive charts
6. **💼 Portfolio** — Paper trading position management & P&L tracking
7. **📉 Performance** — Closed-loop recommendation & model performance analytics

Usage
-----
Run the dashboard locally::

    streamlit run main.py

Author: Shreyansh Patel
Project: SGP-II — AI-Augmented Decision-Support Sandbox
Phase: Production v2.0 — Multi-Page Architecture
"""

import streamlit as st

# =============================================================================
# PAGE CONFIGURATION (must be first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="SGP-II | AI Trading Sandbox",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Shreyansh1812/SGP-II",
        "Report a bug": "https://github.com/Shreyansh1812/SGP-II/issues",
        "About": """
        ## SGP-II: AI-Augmented Decision-Support Sandbox
        **Version:** 2.0.0
        
        A production-grade quantitative decision-support engine combining
        ML technical analysis, LLM sentiment scoring, and fundamental
        screening for US equities.
        
        Built with Python, Streamlit, XGBoost, and Google Gemini.
        """,
    },
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize shared session state variables across all pages."""
    defaults = {
        "db_initialized": False,
        "screener_results": None,
        "sentiment_result": None,
        "model_trained": False,
        "signals_generated": None,
        "portfolio_data": None,
        "benchmark_reports": None,
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


init_session_state()


# =============================================================================
# PAGE NAVIGATION
# =============================================================================

# Define pages using Streamlit's native multi-page system
dashboard_page = st.Page("pages/1_📊_Dashboard.py", title="Dashboard", icon="📊", default=True)
fundamentals_page = st.Page("pages/2_🏥_Fundamentals.py", title="Fundamentals", icon="🏥")
sentiment_page = st.Page("pages/3_📰_Sentiment.py", title="Sentiment", icon="📰")
ml_engine_page = st.Page("pages/4_🤖_ML_Engine.py", title="ML Engine", icon="🤖")
backtester_page = st.Page("pages/5_📈_Backtester.py", title="Backtester", icon="📈")
portfolio_page = st.Page("pages/6_💼_Portfolio.py", title="Portfolio", icon="💼")
performance_page = st.Page("pages/7_📉_Performance.py", title="Performance", icon="📉")

# Build navigation
nav = st.navigation(
    {
        "Decision Support": [dashboard_page, fundamentals_page, sentiment_page],
        "Engine Room": [ml_engine_page, backtester_page],
        "Portfolio & Performance": [portfolio_page, performance_page],
    }
)

# =============================================================================
# GLOBAL SIDEBAR BRANDING
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center; padding: 10px 0;'>
            <h2 style='margin: 0;'>🧠 SGP-II</h2>
            <p style='margin: 0; font-size: 12px; color: #888;'>
                AI-Augmented Decision Support
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

# =============================================================================
# RUN ACTIVE PAGE
# =============================================================================

nav.run()