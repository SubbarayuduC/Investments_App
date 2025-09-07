"""
Investment AI + Data Science Dashboard (single-file)

Features:
- Data collection (yfinance)
- EDA & visualizations (Plotly)
- Predictive models (ARIMA + RandomForest with lag features)
- Portfolio optimization (mean-variance / max Sharpe)
- Risk metrics (Sharpe, Max Drawdown, Historical VaR)
- NLP sentiment (VADER)
- Auto-generated plain-language report + download

Run:
  pip install streamlit yfinance pandas numpy matplotlib plotly scikit-learn statsmodels scipy nltk
  streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

# NLP (VADER)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER is available at runtime
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# ---------- UI CONFIG ----------
st.set_page_config(page_title="Investment AI + DS Dashboard", layout="wide")
st.title("📈 Investment AI + Data Science Dashboard")
st.caption("Stocks • ETFs • Crypto • EDA • Forecasts • Optimization • NLP Insights")

# ---------- SIDEBAR INPUTS ----------
with st.sidebar:
    st.header("Project Setup")
    default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value=", ".join(default_tickers),
        help="Enter Yahoo Finance symbols (e.g., AAPL, MSFT, TSLA, ^NSEI, BTC-USD)",
    )
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    benchmark = st.text_input(
        "Benchmark index (optional)", value="^GSPC", help="e.g., ^GSPC, ^NSEI"
    )

    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - timedelta(days=365 * 3))

    interval = st.selectbox("Data interval", ["1d", "1wk", "1mo"], index=0)

    st.header("Modeling")
    horizon = st.number_input(
        "Forecast horizon (periods)", min_value=5, max_value=252, value=30, step=5
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
    """
    Download adjusted close prices from yfinance.
    Returns a DataFrame indexed by Date with columns for each ticker.
    """
    if not symbols:
        return pd.DataFrame()

    # yfinance accepts both list and space separated str; use list
    try:
        data = yf.download(
            symbols,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            group_by="ticker",
            auto_adjust=False,  # we'll use 'Adj Close'
        )
    except Exception as e:
        st.error(f"yfinance download error: {e}")
        return pd.DataFrame()

    # Normalize output to DataFrame with columns per ticker
    if isinstance(symbols, (list, tuple)) and len(symbols) == 1:
        symbols = symbols[0]

    try:
        if isinstance(symbols, str):
            # single symbol
            if "Adj Close" in data:
                prices = data["Adj Close"].to_frame(name=symbols)
            else:
                # sometimes returned without multiindex
                prices = data["Adj Close"].to_frame(name=symbols)
        else:
            # multiple symbols -> data has structure data[ticker]['Adj Close']
            frames = []
            for t in symbols:
                try:
                    frames.append(data[t]["Adj Close"].rename(t))
                except Exception:
                    # Some tickers may be missing — skip them
                    st.warning(f"No data for {t}; skipping.")
            if not frames:
                return pd.DataFrame()
            prices = pd.concat(frames, axis=1)
    except Exception:
        # fallback: try simpler path (yfinance sometimes returns different formats)
        try:
            prices = data["Adj Close"]
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
        except Exception as e:
            st.error(f"Failed to parse download result: {e}")
            return pd.DataFrame()

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices.dropna(how="all")


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

# Returns & annualization
returns = prices.pct_change().dropna()
ann_factor = {"1d": 252, "1wk": 52, "1mo": 12}[interval]

# ---------- TABS ----------
tab_data, tab_eda, tab_models, tab_portfolio, tab_nlp, tab_report = st.tabs(
    ["📦 Data", "🔎 EDA", "🤖 Models", "🧮 Portfolio", "📰 NLP Insights", "📝 Report"]
)

# ---------- DATA TAB ----------
with tab_data:
    st.subheader("Collected Data")
    st.dataframe(prices.tail(10))

    # Price time series plot
    fig = go.Figure()
    for col in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[col], mode="lines", name=col))
    if bench is not None and not bench.empty:
        # bench is a DataFrame with one column (benchmark ticker)
        bcol = bench.columns[0]
        fig.add_trace(
            go.Scatter(
                x=bench.index,
                y=bench[bcol],
                mode="lines",
                name=benchmark,
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(
        title="Adjusted Close Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(x=0.98, y=0.02),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- EDA TAB ----------
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### Summary statistics (prices)")
        st.dataframe(prices.describe().T)

        st.write("### Summary statistics (returns)")
        st.dataframe(returns.describe().T)

    with col2:
        st.write("### Correlation (returns)")
        corr = returns.corr()
        fig_corr = px.imshow(
            corr, text_auto=True, title="Returns Correlation", aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.write("### Cumulative Returns (growth of $1)")
    cumulative_returns = (1 + returns).cumprod()
    fig_cr = go.Figure()
    for col in cumulative_returns.columns:
        fig_cr.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[col],
                mode="lines",
                name=col,
            )
        )
    fig_cr.update_layout(
        title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of $1"
    )
    st.plotly_chart(fig_cr, use_container_width=True)

    st.write("### Rolling Volatility (annualized)")
    rolling_vol = returns.rolling(window=21).std() * np.sqrt(ann_factor)
    fig_vol = go.Figure()
    for col in rolling_vol.columns:
        fig_vol.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol[col], mode="lines", name=col)
        )
    fig_vol.update_layout(
        title="21-period Rolling Volatility (annualized)",
        xaxis_title="Date",
        yaxis_title="Volatility",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ---------- MODELS TAB ----------
with tab_models:
    st.subheader("Price Forecasting")

    model_panels = st.expander("Model controls / diagnostics", expanded=False)
    with model_panels:
        st.write("Forecast horizon:", horizon)
        st.write("Use ARIMA:", use_arima, "; Use RandomForest:", use_rf)
        st.write(
            "Note: ARIMA may be slow for many tickers. RandomForest uses lag features."
        )

    forecasts: Dict[str, pd.Series] = {}

    # Helper to infer freq for forecast index
    def infer_freq_for_index(idx: pd.DatetimeIndex, interval_choice: str) -> str:
        inferred = pd.infer_freq(idx)
        if inferred is not None:
            return inferred
        # fallback based on user interval
        return {"1d": "D", "1wk": "W", "1mo": "MS"}[interval_choice]

    freq = infer_freq_for_index(prices.index, interval)

    if use_arima:
        st.write("### ARIMA forecasts")
        arima_col1, arima_col2 = st.columns([1, 2])
        with arima_col1:
            p = st.number_input("AR order (p)", min_value=0, max_value=10, value=5)
            d = st.number_input("I order (d)", min_value=0, max_value=2, value=1)
            q = st.number_input("MA order (q)", min_value=0, max_value=10, value=0)
        with arima_col2:
            st.write("ARIMA will run for each ticker sequentially (may take time).")

        for symbol in prices.columns:
            try:
                series = prices[symbol].dropna()
                # statsmodels ARIMA needs 1-D series
                arima_model = ARIMA(series, order=(int(p), int(d), int(q)))
                arima_fit = arima_model.fit()
                pred = arima_fit.forecast(steps=int(horizon))
                # build index
                last_date = series.index[-1]
                idx = pd.date_range(
                    start=last_date, periods=int(horizon) + 1, freq=freq
                )[1:]
                pred.index = idx
                forecasts.setdefault(symbol, pd.Series(dtype=float))
                forecasts[symbol] = forecasts[symbol].append(pred)
                # plot combined
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(x=series.index, y=series, name=symbol))
                fig_s.add_trace(
                    go.Scatter(
                        x=pred.index,
                        y=pred,
                        name=f"{symbol} ARIMA forecast",
                        line=dict(dash="dash"),
                    )
                )
                fig_s.update_layout(
                    title=f"ARIMA Forecast - {symbol}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                )
                st.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                st.warning(f"ARIMA failed for {symbol}: {e}")

    if use_rf:
        st.write("### RandomForest (lag-features) forecasts")
        rf_col1, rf_col2 = st.columns([1, 2])
        with rf_col1:
            n_lags = st.number_input(
                "Number of lag features", min_value=1, max_value=12, value=3
            )
            n_estimators = st.number_input(
                "RF n_estimators", min_value=10, max_value=1000, value=100, step=10
            )
        with rf_col2:
            st.write(
                "RandomForest uses previous lag values to predict next step (iterative forecasting)."
            )

        for symbol in prices.columns:
            try:
                df = prices[[symbol]].copy().dropna()
                for lag in range(1, n_lags + 1):
                    df[f"lag_{lag}"] = df[symbol].shift(lag)
                df = df.dropna()
                X = df[[f"lag_{i}" for i in range(1, n_lags + 1)]]
                y = df[symbol]

                tscv = TimeSeriesSplit(n_splits=3)
                rf = RandomForestRegressor(
                    n_estimators=int(n_estimators), random_state=42
                )
                # Fit on full data; optionally could CV
                rf.fit(X, y)

                # iterative forecast
                last_window = (
                    df[[f"lag_{i}" for i in range(1, n_lags + 1)]]
                    .iloc[-1]
                    .values.tolist()
                )
                preds = []
                window = last_window.copy()
                for i in range(int(horizon)):
                    pred_val = rf.predict([window])[0]
                    preds.append(pred_val)
                    # update window
                    window = [pred_val] + window[:-1]

                last_date = df.index[-1]
                idx = pd.date_range(
                    start=last_date, periods=int(horizon) + 1, freq=freq
                )[1:]
                pred_series = pd.Series(preds, index=idx)
                forecasts.setdefault(symbol, pd.Series(dtype=float))
                forecasts[symbol] = forecasts[symbol].append(pred_series)

                # Plot
                fig_rf = go.Figure()
                fig_rf.add_trace(
                    go.Scatter(x=prices[symbol].index, y=prices[symbol], name=symbol)
                )
                fig_rf.add_trace(
                    go.Scatter(
                        x=pred_series.index,
                        y=pred_series,
                        name=f"{symbol} RF forecast",
                        line=dict(dash="dash"),
                    )
                )
                fig_rf.update_layout(
                    title=f"RandomForest Forecast - {symbol}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                )
                st.plotly_chart(fig_rf, use_container_width=True)
            except Exception as e:
                st.warning(f"RandomForest failed for {symbol}: {e}")

    if not use_arima and not use_rf:
        st.info(
            "No forecasting model selected. Enable ARIMA and/or RandomForest in the sidebar."
        )

# ---------- PORTFOLIO TAB ----------
with tab_portfolio:
    st.subheader("Portfolio Optimization & Risk Metrics")

    st.write("### Inputs")
    st.write(f"Number of assets: {len(prices.columns)}")
    st.write(f"Allow shorting: {allow_short}")
    st.write(f"Max weight per asset: {max_weight}")

    # Compute mean returns and covariance (annualized)
    mean_returns = returns.mean() * ann_factor
    cov_matrix = returns.cov() * ann_factor

    # Risk metric helpers
    def portfolio_performance(weights: np.ndarray) -> Tuple[float, float]:
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return ret, vol

    def neg_sharpe(weights: np.ndarray) -> float:
        ret, vol = portfolio_performance(weights)
        # avoid division by zero
        if vol == 0:
            return 1e6
        return -(ret / vol)  # assuming risk-free ~ 0

    num_assets = len(prices.columns)
    init_guess = np.repeat(1 / num_assets, num_assets)
    if allow_short:
        bounds = [(-max_weight, max_weight) for _ in range(num_assets)]
    else:
        bounds = [(0.0, max_weight) for _ in range(num_assets)]
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    try:
        opt = minimize(
            neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        opt_weights = opt.x
    except Exception as e:
        st.warning(f"Optimization failed: {e}. Using equal weights fallback.")
        opt_weights = init_guess

    weights_series = pd.Series(opt_weights, index=prices.columns)
    st.write("### Optimal Weights (Max Sharpe)")
    st.dataframe(weights_series.round(4))

    port_ret, port_vol = portfolio_performance(opt_weights)
    st.metric("Expected Annual Return", f"{port_ret:.2%}")
    st.metric("Expected Annual Volatility", f"{port_vol:.2%}")

    # Risk metrics for historical portfolio
    # For a simple historical portfolio series, assume daily returns weighted
    historical_portfolio_returns = (returns * opt_weights).sum(axis=1)
    cumulative_portfolio = (1 + historical_portfolio_returns).cumprod()

    # Max drawdown
    def max_drawdown(serie: pd.Series) -> float:
        roll_max = serie.cummax()
        drawdown = (serie / roll_max) - 1.0
        return drawdown.min()

    mdd = max_drawdown(cumulative_portfolio)
    st.write(f"Maximum Drawdown (historical): {mdd:.2%}")

    # Historical VaR (95%)
    var95 = np.percentile(historical_portfolio_returns.dropna(), 5)
    st.write(f"Historical VaR (95%): {var95:.2%} (loss level over single period)")

    # Show cumulative portfolio chart
    fig_port = go.Figure()
    fig_port.add_trace(
        go.Scatter(
            x=cumulative_portfolio.index,
            y=cumulative_portfolio,
            name="Portfolio Growth",
        )
    )
    fig_port.update_layout(
        title="Historical Portfolio Growth (based on optimized weights)",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
    )
    st.plotly_chart(fig_port, use_container_width=True)

# ---------- NLP TAB ----------
with tab_nlp:
    st.subheader("NLP Sentiment (VADER)")
    if not nlp_enabled:
        st.info("NLP Sentiment is disabled. Enable it from the sidebar.")
    else:
        news_text = st.text_area(
            "Paste news / analyst notes / social text here", height=250
        )
        if st.button("Analyze Sentiment"):
            if not news_text.strip():
                st.warning("Please paste some text to analyze.")
            else:
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(news_text)
                st.write("### Sentiment Scores")
                st.json(scores)
                # Simple interpretation
                comp = scores.get("compound", 0)
                if comp >= 0.05:
                    st.success("Overall sentiment: Positive")
                elif comp <= -0.05:
                    st.error("Overall sentiment: Negative")
                else:
                    st.info("Overall sentiment: Neutral")

# ---------- REPORT TAB ----------
with tab_report:
    st.subheader("Auto-generated Report")
    st.write(
        "This report summarizes the key results and suggested strategy for the selected risk profile."
    )

    # Build textual report
    def build_report() -> str:
        lines = []
        lines.append("# Investment AI + Data Science Dashboard Report")
        lines.append(f"Generated: {pd.Timestamp.now()}")
        lines.append("")
        lines.append("## Portfolio Summary")
        lines.append(f"- Tickers analyzed: {', '.join(prices.columns)}")
        lines.append(
            f"- Date range: {prices.index.min().date()} to {prices.index.max().date()}"
        )
        lines.append(f"- Interval: {interval}")
        lines.append("")
        lines.append("### Optimized allocation (Max Sharpe)")
        for t, w in weights_series.items():
            lines.append(f"- {t}: {w:.2%}")
        lines.append(f"Expected annual return: {port_ret:.2%}")
        lines.append(f"Expected annual volatility: {port_vol:.2%}")
        lines.append(f"Historical maximum drawdown (portfolio): {mdd:.2%}")
        lines.append(f"Historical VaR (95%): {var95:.2%}")
        lines.append("")
        lines.append("## Forecast Summary")
        if forecasts:
            lines.append("- Forecasts were generated for the following tickers:")
            for sym in forecasts:
                lines.append(
                    f"  - {sym}: forecast horizon {horizon} periods (see charts in Models tab)"
                )
        else:
            lines.append(
                "- No forecasts generated (ARIMA and RandomForest both disabled or failed)."
            )

        lines.append("")
        lines.append("## Suggested Strategy (based on risk profile)")
        if risk_profile == "Conservative":
            lines.append(
                "- Conservative: prioritize stable, low-volatility assets. Consider larger cash or bond allocation; reduce weights on highly volatile names."
            )
        elif risk_profile == "Moderate":
            lines.append(
                "- Moderate: balanced allocation between growth and defensive assets; keep diversification and monitor drawdown risk."
            )
        else:
            lines.append(
                "- Aggressive: higher allocation to growth-oriented and volatile names; accept higher drawdown risk for potential higher returns."
            )

        lines.append("")
        lines.append("## Sentiment Notes")
        if nlp_enabled:
            lines.append(
                "- NLP sentiment is available in the NLP tab for manual text analysis."
            )
        else:
            lines.append("- NLP sentiment was disabled.")

        lines.append("")
        lines.append("## Limitations & Notes")
        lines.append("- Past performance is not indicative of future results.")
        lines.append(
            "- Forecasts are simple models for illustrative purposes; use caution and further validation in production."
        )
        lines.append(
            "- This dashboard is educational and should not be used as financial advice."
        )
        return "\n".join(lines)

    report_text = build_report()
    st.markdown("### Report Preview")
    st.text_area("Report (editable)", value=report_text, height=400)

    # Download button
    b = io.BytesIO(report_text.encode("utf-8"))
    st.download_button(
        "Download report (.txt)",
        data=b,
        file_name="investment_report.txt",
        mime="text/plain",
    )

    st.success("Report generated. You can edit and download it above.")

# ---------- Footer ----------
st.write("---")
st.caption(
    "Built with Streamlit • Data from Yahoo Finance (yfinance) • NLP via NLTK VADER"
)
