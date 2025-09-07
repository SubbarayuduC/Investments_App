
````markdown
# 📈 Investment AI + Data Science Dashboard

A **Streamlit-based interactive dashboard** for investment research and portfolio analytics.  
Combines **data collection, EDA, predictive modeling, portfolio optimization, risk metrics, and NLP sentiment analysis** into a single app.

---

## 🚀 Features

- **Data Collection**
  - Fetch historical prices for Stocks, ETFs, Crypto via `yfinance`
  - Optional benchmark comparison (e.g., S&P 500: `^GSPC`, Nifty 50: `^NSEI`)

- **Exploratory Data Analysis (EDA)**
  - Interactive charts with Plotly
  - Returns calculation and visualization

- **Predictive Models**
  - ARIMA for time series forecasting
  - Random Forest Regressor using lag features
  - Configurable forecast horizon

- **Portfolio Optimization**
  - Mean-variance optimization (`scipy.optimize.minimize`)
  - Adjustable constraints: shorting, max weights per asset
  - Risk metrics: Sharpe ratio, Maximum Drawdown, Value at Risk (VaR)

- **NLP Sentiment Analysis**
  - Analyze pasted news text with **VADER sentiment**

- **Automated Reporting**
  - Plain-language report summarizing:
    - Key data trends
    - Forecasts
    - Portfolio allocation
    - Risk profile insights

---

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/investment-ai-dashboard.git
   cd investment-ai-dashboard
````

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:

   ```bash
   pip install streamlit yfinance pandas numpy matplotlib plotly scikit-learn statsmodels scipy nltk
   ```

3. **Download NLTK VADER Lexicon (first run only):**

   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

---

## ▶️ Usage

Run the dashboard locally:

```bash
streamlit run app.py
```

Open the app in your browser at **`http://localhost:8501`**.

---

## ⚙️ How It Works

1. **Input your tickers** (comma-separated, e.g., `AAPL, MSFT, GOOGL`)
2. **Set your date range and interval** (`1d`, `1wk`, or `1mo`)
3. **Choose forecasting models** (ARIMA, Random Forest, or both)
4. **Adjust portfolio constraints and risk profile**
5. **Paste news text** (optional) for NLP sentiment insights
6. **Generate automated report** in the **Report tab**

---

## 📦 Project Structure

```
investment-ai-dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📊 Example Tickers

* **US Stocks:** `AAPL, MSFT, NVDA, AMZN`
* **Indices:** `^GSPC` (S\&P 500), `^NSEI` (Nifty 50)
* **Crypto:** `BTC-USD, ETH-USD`

---

## 🧠 Future Enhancements

* Integration with real-time news APIs
* Deep learning models for forecasting
* Multi-language NLP sentiment
* Exportable PDF/Excel reports
* Portfolio backtesting engine

---

## 📄 License

This project is licensed under the **MIT License** – feel free to modify and use it.

---

## 🙌 Acknowledgments

* [Streamlit](https://streamlit.io)
* [Yahoo Finance API](https://pypi.org/project/yfinance/)
* [NLTK VADER](https://github.com/cjhutto/vaderSentiment)

---

**Author:** Subbarayudu C
**Contact:** [raayudu.civil.1997@gmail.com](subbublockchaindoveloper@gmail.com)

```
