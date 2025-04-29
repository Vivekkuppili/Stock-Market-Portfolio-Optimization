import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from portfolio_optimizer import (
    get_sp500_stocks,
    get_live_data,
    predict_returns,
    optimize_portfolio,
    analyze_risk_profile,
    portfolio_performance,
    sortino_ratio,
    max_drawdown,
    calculate_beta
)
import time

# Set page config
st.set_page_config(
    page_title="AI Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .risk-warning {
        color: #ff4b4b;
    }
    .risk-good {
        color: #00cc00;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4e8df5;
    }
    .stock-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        border-top: 4px solid #4e8df5;
    }
    .stock-ticker {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1e3a8a;
    }
    .stock-metrics {
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache expensive operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_sp500_stocks():
    return get_sp500_stocks()

@st.cache_data(ttl=3600)
def get_cached_live_data(tickers):
    return get_live_data(tickers)

def main():
    st.title("📈 AI Portfolio Optimizer")
    st.markdown("""
    This application uses AI and modern portfolio theory to help you optimize your investment portfolio.
    Select up to 10 stocks from the top 50 S&P 500 companies and get AI-powered predictions and optimal allocations.
    """)
    
    # Portfolio Optimization
    portfolio_optimization_tab()

def portfolio_optimization_tab():
    # Sidebar for inputs
    with st.sidebar:
        st.header("Portfolio Settings")
        
        try:
            # Get top 50 S&P 500 stocks (cached)
            sp500_tickers = get_cached_sp500_stocks()
            if not sp500_tickers:
                st.error("Failed to fetch S&P 500 stocks. Please refresh the page.")
                return
            
            # Create a DataFrame for stock selection
            stocks_df = pd.DataFrame({
                'Number': range(1, len(sp500_tickers) + 1),
                'Stock': sp500_tickers
            })
            
            st.markdown("### Select Stocks")
            st.markdown("Choose up to 10 stocks from the top 50 S&P 500 companies:")
            
            # Multi-select for stocks
            selected_stocks = st.multiselect(
                "Select stocks (up to 10):",
                options=stocks_df['Stock'].tolist(),
                format_func=lambda x: f"{x}",
                max_selections=10
            )
            
            # Investment amount
            investment = st.number_input(
                "Enter investment amount ($):",
                min_value=1000,
                value=100000,
                step=1000
            )
            
            # Generate button
            generate_button = st.button("Generate Portfolio Analysis")
            
        except Exception as e:
            st.error(f"Error in portfolio settings: {str(e)}")
            return

    # Main content area
    if generate_button and selected_stocks:
        with st.spinner("Fetching market data and generating analysis..."):
            try:
                # Get market data with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Fetching market data...")
                market_data = get_cached_live_data(selected_stocks)
                progress_bar.progress(30)
                
                if market_data is None:
                    st.error("Failed to fetch market data. Please try again.")
                else:
                    status_text.text("Calculating AI predictions...")
                    predictions = {}
                    for ticker in selected_stocks:
                        try:
                            pred_result = predict_returns(market_data, ticker)
                            if pred_result:
                                model, scaler, feature_importance, train_score, test_score, prediction, pred_date = pred_result
                                # Calculate a more meaningful confidence score
                                # Use exponential scaling to emphasize good scores
                                base_confidence = max(0, min(100, (1 + train_score) * 50))
                                confidence = min(100, base_confidence * (1.2 if base_confidence > 40 else 1.0))
                                
                                # Adjust confidence based on prediction magnitude
                                pred_magnitude = abs(prediction)
                                if pred_magnitude > 0.02:  # If prediction > 2%
                                    confidence = max(20, confidence * 0.8)  # Reduce confidence for large predictions
                                
                                predictions[ticker] = {
                                    'prediction': prediction,
                                    'confidence': confidence / 100,  # Store as decimal
                                    'feature_importance': feature_importance,
                                    'prediction_date': pred_date
                                }
                        except Exception as e:
                            st.warning(f"Could not generate prediction for {ticker}: {str(e)}")
                            predictions[ticker] = {
                                'prediction': 0,
                                'confidence': 0,
                                'feature_importance': pd.DataFrame({'feature': ['Error'], 'importance': [0]}),
                                'prediction_date': None
                            }
                    progress_bar.progress(60)
                    
                    # Display AI Predictions with improved formatting
                    if predictions:
                        st.subheader("AI-Powered Return Predictions")
                        pred_df = pd.DataFrame([
                            {
                                'Stock': ticker,
                                'Predicted Return': f"{pred['prediction']*100:.2f}%",
                                'Top Feature': pred['feature_importance'].iloc[0]['feature'],
                                'Prediction Date': pred['prediction_date'].strftime('%Y-%m-%d') if pred['prediction_date'] else 'N/A'
                            }
                            for ticker, pred in predictions.items()
                        ])
                        
                        # Sort by absolute predicted return
                        pred_df['Sort Value'] = [abs(float(x.strip('%'))) for x in pred_df['Predicted Return']]
                        pred_df = pred_df.sort_values('Sort Value', ascending=False).drop('Sort Value', axis=1)
                        
                        st.dataframe(pred_df)

                        # --- Predicted Returns Bar Chart ---
                        if not pred_df.empty:
                            pred_returns = [float(x.strip('%')) for x in pred_df['Predicted Return']]
                            pred_colors = ['green' if val >= 0 else 'red' for val in pred_returns]
                            fig_bar = px.bar(
                                pred_df,
                                x='Stock',
                                y=pred_returns,
                                title='Predicted Returns by Stock',
                                labels={'y': 'Predicted Return (%)', 'x': 'Stock'},
                                text=[f"{x:.2f}%" for x in pred_returns],
                            )
                            fig_bar.update_traces(marker_color=pred_colors, textposition='outside')
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # --- Actual vs Predicted Returns Chart ---
                        actual_vs_pred = []
                        for ticker in selected_stocks:
                            # Get actual return (last week's realized return)
                            try:
                                if isinstance(market_data['Close'], pd.DataFrame):
                                    close_prices = market_data['Close'][ticker].dropna()
                                else:
                                    close_prices = market_data['Close'].dropna()
                                if len(close_prices) > 6:
                                    actual_return = (close_prices.iloc[-1] - close_prices.iloc[-6]) / close_prices.iloc[-6] * 100
                                else:
                                    actual_return = np.nan
                            except Exception:
                                actual_return = np.nan
                            pred_return = predictions[ticker]['prediction'] * 100 if ticker in predictions else np.nan
                            actual_vs_pred.append({'Stock': ticker, 'Actual Return': actual_return, 'Predicted Return': pred_return})
                        actual_vs_pred_df = pd.DataFrame(actual_vs_pred)
                    
                    status_text.text("Optimizing portfolio...")
                    # Get S&P 500 data for comparison
                    sp500_data = get_cached_live_data(["^GSPC"])
                    if sp500_data is not None:
                        # Calculate returns
                        returns = market_data['Close'].pct_change().dropna()
                        sp500_returns = sp500_data['Close'].pct_change().dropna()
                        
                        weights, mean_returns, cov_matrix = optimize_portfolio(returns, sp500_returns)
                        if weights is not None:
                            progress_bar.progress(90)
                            
                            status_text.text("Generating final analysis...")
                            # Calculate portfolio metrics
                            portfolio_returns, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
                            risk_profile = analyze_risk_profile(returns, weights, sp500_returns)
                            
                            # Display results
                            st.header("Portfolio Analysis Results")
                            
                            # Portfolio Allocation
                            st.subheader("Optimal Portfolio Allocation")
                            allocation_df = pd.DataFrame({
                                'Stock': selected_stocks,
                                'Weight': [f"{w*100:.2f}%" for w in weights]
                            })
                            st.dataframe(allocation_df)

                            # --- Portfolio Allocation Pie Chart ---
                            if weights is not None and len(weights) == len(selected_stocks):
                                allocation_pie_df = pd.DataFrame({
                                    'Stock': selected_stocks,
                                    'Allocation': [w * 100 for w in weights]
                                })
                                if allocation_pie_df['Allocation'].sum() > 0 and allocation_pie_df['Allocation'].notna().any():
                                    fig_pie = px.pie(
                                        allocation_pie_df,
                                        names='Stock',
                                        values='Allocation',
                                        title='Portfolio Allocation by Stock',
                                        hole=0.4
                                    )
                                    fig_pie.update_traces(textinfo='percent+label')
                                    st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Investment Allocation
                            st.subheader("Investment Allocation")
                            investment_df = pd.DataFrame({
                                'Stock': selected_stocks,
                                'Amount ($)': [f"${w*investment:,.2f}" for w in weights]
                            })
                            st.dataframe(investment_df)
                            
                            # Risk Analysis
                            if risk_profile:
                                st.subheader("Risk Analysis")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Risk Score", f"{risk_profile['risk_score']:.1f}/100")
                                    st.metric("Risk Category", risk_profile['risk_category'])
                                    st.metric("Annual Volatility", f"{risk_profile['volatility']*100:.1f}%")
                                    st.metric("Beta", f"{risk_profile['beta']:.2f}")
                                with col2:
                                    st.metric("Maximum Drawdown", f"{risk_profile['max_drawdown']*100:.1f}%")
                                    st.metric("Downside Volatility", f"{risk_profile['downside_volatility']*100:.1f}%")
                                    st.metric("Concentration Index", f"{risk_profile['concentration']:.2f}")
                                
                                if risk_profile['risk_warnings']:
                                    st.warning("Risk Warnings:")
                                    for warning in risk_profile['risk_warnings']:
                                        st.write(f"- {warning}")
                            
                            # Performance Metrics
                            st.subheader("Performance Metrics")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Expected Annual Return", f"{portfolio_returns*252*100:.1f}%")
                                st.metric("Annual Volatility", f"{portfolio_volatility*np.sqrt(252)*100:.1f}%")
                                st.metric("Sharpe Ratio", f"{(portfolio_returns - 0.05/252)/portfolio_volatility:.2f}")
                            with col2:
                                st.metric("Sortino Ratio", f"{sortino_ratio(weights, returns):.2f}")
                                st.metric("Beta", f"{calculate_beta(returns.dot(weights), sp500_returns):.2f}")
                            
                            # Portfolio Combinations Suggestions
                            st.subheader("Suggested Portfolio Combinations")
                            st.markdown("""
                            Based on our analysis, here are some suggested portfolio combinations:
                            """)
                            
                            # Create three portfolio suggestions
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("""
                                #### Conservative Portfolio
                                **Risk Level: Low**
                                - AAPL (25%)
                                - MSFT (20%)
                                - JNJ (20%)
                                - PG (20%)
                                - KO (15%)
                                
                                *Focus: Stable blue-chip companies with strong dividends*
                                """)
                            
                            with col2:
                                st.markdown("""
                                #### Balanced Portfolio
                                **Risk Level: Moderate**
                                - AAPL (20%)
                                - MSFT (20%)
                                - GOOGL (20%)
                                - AMZN (20%)
                                - NVDA (20%)
                                
                                *Focus: Mix of tech leaders with growth potential*
                                """)
                            
                            with col3:
                                st.markdown("""
                                #### Growth Portfolio
                                **Risk Level: High**
                                - NVDA (30%)
                                - TSLA (25%)
                                - AMD (20%)
                                - META (15%)
                                - CRM (10%)
                                
                                *Focus: High-growth tech companies*
                                """)
                            
                            # Add a note about the suggestions
                            st.info("""
                            **Note:** These are pre-defined portfolio combinations based on different risk preferences. 
                            You can click on any of these combinations in the sidebar to analyze them further.
                            The percentages shown are suggested allocation weights.
                            """)
                            
                            progress_bar.progress(100)
                            status_text.text("Analysis complete!")
                        else:
                            st.error("Failed to optimize portfolio. Please try different stocks.")
                    else:
                        st.error("Failed to fetch S&P 500 data. Please try again.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again with different stocks or a different investment amount.")

if __name__ == "__main__":
    main() 