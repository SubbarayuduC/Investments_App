"""
Investment AI + Data Science Dashboard
--------------------------------------
Single-file Streamlit app implementing:
- Data collection (yfinance + macro placeholder)
- EDA & visualizations
- Predictive models (ARIMA + RandomForest on lag features)
- Portfolio optimization (mean-variance with scipy.optimize)
- Risk metrics (Sharpe, max drawdown, VaR)
- NLP sentiment (VADER) on pasted news text
- Auto-generated plain-language report + risk-profile strategies

Run locally:
  pip install streamlit yfinance pandas numpy matplotlib plotly scikit-learn statsmodels scipy nltk
  streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

# --- NLP (VADER) ---
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER is available
try:
    _ = nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# ---------- UI CONFIG ----------
st.set_page_config(
    page_title="Investment AI + DS Dashboard",
    layout="wide",
)

st.title("📈 Investment AI + Data Science Dashboard")
st.caption("Stocks • ETFs • Crypto • EDA • Forecasts • Optimization • NLP Insights")

# ---------- SIDEBAR INPUTS ----------
with st.sidebar:
    st.header("Project Setup")
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    tickers = st.text_input(
        "Tickers (comma-separated)",
        value=", ".join(default_tickers),
        help="Enter Yahoo Finance symbols (e.g., AAPL, MSFT, TSLA, ^NSEI, BTC-USD)",
    )
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    benchmark = st.text_input(
        "Benchmark index (optional)", value="^GSPC", help="e.g., ^GSPC, ^NSEI"
    )

    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - timedelta(days=365 * 3))

    interval = st.selectbox("Data interval", ["1d", "1wk", "1mo"], index=0)

    st.header("Modeling")
    horizon = st.number_input(
        "Forecast horizon (days)", min_value=5, max_value=252, value=30, step=5
    )
    use_arima = st.checkbox("Use ARIMA", value=True)
    use_rf = st.checkbox("Use RandomForest (lag features)", value=True)

    st.header("Portfolio Constraints")
    allow_short = st.checkbox("Allow shorting (weights can be negative)", value=False)
    max_weight = st.slider("Max weight per asset", 0.1, 1.0, 0.4, 0.05)

    st.header("Risk Profile")
    risk_profile = st.selectbox(
        "Choose risk profile", ["Conservative", "Moderate", "Aggressive"], index=1
    )

    st.header("NLP Sentiment")
    nlp_enabled = st.checkbox("Enable NLP sentiment on pasted news text", value=True)


# ---------- DATA COLLECTION ----------
@st.cache_data(show_spinner=True)
def load_prices(
    symbols: List[str], start: date, end: date, interval: str
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance."""
    data = yf.download(
        symbols,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        group_by="ticker",
        auto_adjust=False,
    )

    if isinstance(symbols, str) or len(symbols) == 1:
        if isinstance(symbols, list):
            symbols = symbols[0]
        prices = data["Adj Close"].to_frame(name=symbols)
    else:
        prices = pd.DataFrame({tic: data[tic]["Adj Close"] for tic in symbols})

    return prices.dropna()


if len(tickers) == 0:
    st.warning("Please enter at least one ticker symbol.")
    st.stop()

prices = load_prices(tickers, start_date, end_date, interval)
if prices.empty:
    st.error("No price data found for the given symbols/date range.")
    st.stop()

# Benchmark
bench = None
if benchmark:
    try:
        bench = load_prices([benchmark], start_date, end_date, interval)
    except Exception:
        bench = None

# Returns
returns = prices.pct_change().dropna()
ann_factor = {"1d": 252, "1wk": 52, "1mo": 12}[interval]

# ---------- TABS ----------
tab_data, tab_eda, tab_models, tab_portfolio, tab_nlp, tab_report = st.tabs(
    [
        "📦 Data",
        "🔎 EDA",
        "🤖 Models",
        "🧮 Portfolio",
        "📰 NLP Insights",
        "📝 Report",
    ]
)

# ---------- DATA TAB ----------
with tab_data:
    st.subheader("Collected Data")
    st.dataframe(prices.tail())

    fig = go.Figure()
    for col in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode="lines", name=col))
    if bench is not None:
        fig.add_trace(
            go.Scatter(
                x=bench.index,
                y=bench[benchmark],
                mode="lines",
                name=benchmark,
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(
        title="Adjusted Close Prices", xaxis_title="Date", yaxis_title="Price"
    )
    st.plotly_chart(fig, use_container_width=True)

# (The rest of the EDA, Models, Portfolio, NLP, and Report sections from your code remain unchanged.)
# Just keep your original code after this block, since the key fix was in load_prices.
