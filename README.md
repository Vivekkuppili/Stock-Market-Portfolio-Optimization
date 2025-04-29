# AI Portfolio Optimizer: Data-Driven Investment Strategies

A user-friendly platform that blends machine learning with modern portfolio theory to help users build diversified investment portfolios. The app uses a Random Forest model and technical indicators (RSI, MACD, moving averages) to generate AI-powered stock return predictions. Asset allocation is optimized for the Sharpe Ratio, with options for Conservative, Balanced, and Growth profiles. Interactive pie and bar charts, along with key risk metrics, provide clear insights into portfolio composition and performance.

## Features

- Portfolio Optimization using Mean-Variance Optimization (MPT)
- AI-Powered Return Predictions (Random Forest)
- Technical Indicators (RSI, MACD, Moving Averages)
- Three Portfolio Profiles: Conservative, Balanced, Growth-Oriented
- Risk Metrics: Volatility, Beta, Max Drawdown, Concentration
- Interactive Visualizations (Pie Chart for Allocation, Bar Chart for Predicted Returns)
- Performance Metrics (Sharpe, Sortino, Drawdown)
- Robust error handling and user feedback

## How It Works

1. **Select up to 10 stocks** from the top 50 S&P 500.
2. **Enter your investment amount.**
3. **Generate portfolio analysis** to see:
    - AI-predicted returns for each stock (Random Forest)
    - Portfolio allocation pie chart
    - Predicted returns bar chart (color-coded)
    - Optimal asset allocation and investment breakdown
    - Overall portfolio risk and performance metrics
    - Suggested Conservative, Balanced, and Growth portfolios

## Technology

- **Frontend:** Streamlit (Python)
- **Backend:** yfinance, pandas, numpy, scikit-learn, plotly
- **AI/ML:** Random Forest for return prediction
- **Portfolio Theory:** Mean-Variance Optimization (MPT), Sharpe Ratio
- **Visualization:** Plotly (pie chart, bar chart)

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Activate the virtual environment (Windows):
   ```powershell
   .\venv\Scripts\activate
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open the URL provided by Streamlit in your browser (typically http://localhost:8501).

---

This app empowers users with data-driven, AI-enhanced portfolio optimization and transparent risk diagnostics.

## Conclusion and Future Directions

The AI Portfolio Optimizer effectively integrates machine learning with Modern Portfolio Theory to guide users in constructing diversified investment portfolios. By predicting stock returns with a Random Forest model and optimizing allocations based on the Sharpe Ratio and key risk metrics, the platform empowers investors to make informed decisions. Its interactive interface, live data integration, and transparent risk analysis—including volatility, beta, and maximum drawdown—make portfolio management both accessible and insightful. Users can easily compare Conservative, Balanced, and Growth strategies tailored to different risk appetites. Looking ahead, incorporating features like sentiment analysis, deep learning models, and broader asset diversification could further enhance the system’s capabilities. Overall, the platform provides a robust, data-driven foundation for responsible and confident investing, adaptable to evolving market conditions.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Data Sources

- S&P 500 stocks (top 50 companies)
- Live market data using yfinance
- Industry sector information
- Historical price data

## Project Structure

- `app.py`: Main Streamlit application
- `portfolio_optimizer.py`: Core logic for optimization, risk, and prediction

## License

MIT License 