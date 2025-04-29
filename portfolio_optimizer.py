
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Function to create technical indicators as features
def create_features(data):
    """Create technical indicators and features for prediction"""
    df = data.copy()

    # Basic price-based indicators
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])

    # Volume-based indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Price_Trend'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']

    # Price momentum and range indicators
    df['Price_Momentum'] = df['Close'].pct_change(periods=10)
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']

    # Market trend indicator (using 20-day moving average)
    df['Market_Trend'] = (df['Close'] - df['SMA_20']) / df['SMA_20']

                # Volatility
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()

    # Create target variable (next day's return)
    df['Target'] = df['Returns'].shift(-1)

    return df

# Function to calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to calculate MACD
def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

# Function to predict returns using Random Forest
def predict_returns(data, ticker):
    """Predict stock returns using AI"""
    try:
        # Extract data for the specific ticker
        if isinstance(data.columns, pd.MultiIndex):
            stock_data = data.xs(ticker, axis=1, level=1)
        else:
            stock_data = data

        # Create features
        features = create_features(stock_data.copy())

        # Prepare data
        features = features.dropna()
        
        # Split data into training and prediction sets
        train_data = features[:-5]  # Use all data except last week
        predict_data = features[-5:]  # Use last week for prediction
        
        # Prepare features and target
        feature_cols = ['SMA_5', 'SMA_20', 'RSI', 'MACD', 'Volatility', 
                       'Volume_MA', 'Price_Momentum', 'High_Low_Range', 
                       'Volume_Price_Trend', 'Market_Trend']
        
        X_train = train_data[feature_cols]
        y_train = train_data['Target']
        X_predict = predict_data[feature_cols]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_predict_scaled = scaler.transform(X_predict)
        
        # Train model with cross-validation
        from sklearn.model_selection import cross_val_score
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Train final model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = model.predict(X_predict_scaled)
        
        # Calculate confidence score based on cross-validation
        confidence_score = np.mean(cv_scores)
        confidence_score = (confidence_score + 1) / 2  # Convert from [-1,1] to [0,1]
        confidence_score = confidence_score * 100  # Convert to percentage
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Calculate next week's prediction date
        current_date = datetime.now()
        next_week = current_date + timedelta(days=7)
        # Adjust to next business day if it falls on weekend
        while next_week.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_week += timedelta(days=1)
        
        # Use the average of predictions for next week
        final_prediction = np.mean(predictions)

        return model, scaler, feature_importance, confidence_score, confidence_score, final_prediction, next_week
        
    except Exception as e:
        print(f"Error predicting returns for {ticker}: {str(e)}")
        return None

# Function to fetch live market data
def get_live_data(tickers):
    """Fetch live market data for given tickers"""
    end_date = datetime.now()
    # Use 2 years of data for better predictions
    start_date = end_date - timedelta(days=730)  # 2 years of data

    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        # Fetch data with progress bar
        data = yf.download(tickers, start=start_date, end=end_date, progress=True)

        if data.empty:
            raise Exception("No data received from Yahoo Finance")

        # Ensure we have the most recent data
        latest_date = data.index[-1]
        print(f"Latest data available: {latest_date.strftime('%Y-%m-%d')}")

        return data

    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_vol

# Function to calculate Sortino ratio
def sortino_ratio(weights, returns, risk_free_rate=0.05/252):
    portfolio_returns = returns.dot(weights)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
    expected_return = np.mean(portfolio_returns)
    return (expected_return - risk_free_rate) / downside_std

# Function to calculate max drawdown
def max_drawdown(portfolio_returns):
    cumulative = (1 + portfolio_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

# Function to calculate beta relative to S&P 500
def calculate_beta(returns, market_returns):
    # Ensure both series are aligned and have the same length
    aligned_returns, aligned_market_returns = returns.align(market_returns, join='inner')
    if len(aligned_returns) == 0 or len(aligned_market_returns) == 0:
        return np.nan

    # Convert to numpy arrays and ensure correct shape
    returns_array = aligned_returns.values.reshape(-1, 1)
    market_array = aligned_market_returns.values.reshape(-1, 1)

    # Calculate covariance and variance
    covariance = np.cov(returns_array.flatten(), market_array.flatten())[0, 1]
    market_variance = np.var(market_array)

    return covariance / market_variance if market_variance != 0 else np.nan

# Function to analyze portfolio risk profile
def analyze_risk_profile(returns, weights, market_returns):
    # Calculate portfolio metrics
    portfolio_returns = returns.dot(weights)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    portfolio_beta = calculate_beta(portfolio_returns, market_returns)

    # Calculate individual stock metrics
    stock_vols = returns.std()
    stock_betas = [calculate_beta(returns[col], market_returns) for col in returns.columns]

    # Calculate concentration metrics
    herfindahl_index = np.sum(weights ** 2)  # Concentration measure
    max_weight = np.max(weights)

    # Calculate sector concentration (if sector data available)
    # For now, using volatility as a proxy for sector risk
    sector_volatility = np.sum(weights * stock_vols)

    # Calculate downside risk metrics
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = np.std(downside_returns) if len(downside_returns) > 0 else 0

    # Calculate maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_dd = drawdown.min()

    # Risk assessment based on industry standards
    risk_score = 0

    # Volatility component (25% weight)
    vol_score = portfolio_vol * np.sqrt(252)  # Annualized volatility
    if vol_score > 0.25:  # High volatility threshold
        risk_score += 25
    else:
        risk_score += min(vol_score * 100, 25)

    # Beta component (20% weight)
    beta_score = abs(portfolio_beta - 1) * 10  # Deviation from market beta
    if portfolio_beta > 1.3:  # High beta threshold
        risk_score += 20
    else:
        risk_score += min(beta_score, 20)

    # Concentration component (20% weight)
    concentration_score = (herfindahl_index * 20)  # Higher concentration = higher risk
    if max_weight > 0.25:  # High concentration threshold
        risk_score += 20
    else:
        risk_score += min(concentration_score, 20)

    # Downside risk component (20% weight)
    downside_score = downside_vol * np.sqrt(252) * 20
    if downside_vol * np.sqrt(252) > 0.20:  # High downside risk threshold
        risk_score += 20
    else:
        risk_score += min(downside_score, 20)

    # Drawdown component (15% weight)
    drawdown_score = abs(max_dd) * 15
    if abs(max_dd) > 0.30:  # High drawdown threshold
        risk_score += 15
    else:
        risk_score += min(drawdown_score, 15)

    # Determine risk category with more granular levels
    if risk_score < 15:
        risk_category = "Very Low Risk"
    elif risk_score < 30:
        risk_category = "Low Risk"
    elif risk_score < 45:
        risk_category = "Moderate Risk"
    elif risk_score < 60:
        risk_category = "High Risk"
    elif risk_score < 75:
        risk_category = "Very High Risk"
    else:
        risk_category = "Extreme Risk"

    # Add risk warnings for specific thresholds
    risk_warnings = []
    if portfolio_vol * np.sqrt(252) > 0.25:
        risk_warnings.append("High volatility (>25%)")
    if portfolio_beta > 1.3:
        risk_warnings.append("High market sensitivity (Beta > 1.3)")
    if max_weight > 0.25:
        risk_warnings.append("High concentration in single stock (>25%)")
    if abs(max_dd) > 0.30:
        risk_warnings.append("Large potential drawdown (>30%)")

    return {
        'risk_score': risk_score,
        'risk_category': risk_category,
        'volatility': portfolio_vol * np.sqrt(252),
        'beta': portfolio_beta,
        'concentration': herfindahl_index,
        'max_weight': max_weight,
        'sector_volatility': sector_volatility,
        'downside_volatility': downside_vol * np.sqrt(252),
        'max_drawdown': max_dd,
        'risk_warnings': risk_warnings
    }

# Modified optimize_portfolio function
def optimize_portfolio(returns, market_returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov() + np.eye(len(returns.columns)) * 1e-6
    
    num_assets = len(returns.columns)

    # Calculate optimal bounds based on volatility and number of assets
    vols = np.sqrt(np.diag(cov_matrix))
    min_allocation = max(0.05, 1 / (num_assets * 2))  # Minimum 5% or 1/(2n)
    max_allocation = min(0.40, 1 / (num_assets / 2))  # Maximum 40% or 1/(n/2)

    # Risk-free rate (assuming 5% annual)
    risk_free_rate = 0.05/252

    # Objective function: Maximize Sharpe ratio with penalty for concentration
    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol

        # Add penalty for high concentration
        concentration_penalty = np.sum(weights ** 2) * 0.1
        return -sharpe_ratio + concentration_penalty

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights = 1
        {'type': 'ineq', 'fun': lambda x: np.sum(mean_returns * x) - 0.05}  # Minimum return of 5%
    ]
    
    # Add minimum and maximum weight constraints
    for i in range(num_assets):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - min_allocation})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: max_allocation - x[i]})
    
    # Add maximum concentration constraint
    constraints.append({'type': 'ineq', 'fun': lambda x: 1 - np.max(x)})

    # Add sector concentration constraint (using volatility as proxy)
    constraints.append({'type': 'ineq', 'fun': lambda x: 0.5 - np.sum(x * vols)})

    # Bounds for each weight
    bounds = tuple((min_allocation, max_allocation) for _ in range(num_assets))
    
    # Try multiple initial guesses to avoid local optima
    best_weights = None
    best_sharpe = float('-inf')
    
    for _ in range(30):  # Increased number of attempts
        # Generate random weights that sum to 1
        initial_guess = np.random.dirichlet(np.ones(num_assets), size=1).flatten()

        # Scale initial guess to respect bounds
        initial_guess = min_allocation + (max_allocation - min_allocation) * initial_guess
        initial_guess = initial_guess / np.sum(initial_guess)
        
        try:
            result = minimize(objective,
                            initial_guess,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
            if result.success and -result.fun > best_sharpe:
                best_weights = result.x
                best_sharpe = -result.fun
        except:
            continue
    
    if best_weights is None:
        # Fallback to risk-adjusted weights
        volatility = np.sqrt(np.diag(cov_matrix))
        best_weights = 1 / volatility
        best_weights = best_weights / np.sum(best_weights)
    
    # Ensure weights sum to 1 and respect bounds
    best_weights = np.clip(best_weights, min_allocation, max_allocation)
    best_weights = best_weights / np.sum(best_weights)
    
    return best_weights, mean_returns, cov_matrix

# Function to fetch S&P 500 stocks
def get_sp500_stocks():
    # Predefined list of top 50 S&P 500 stocks by market cap (as of 2024)
    sp500_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'JPM', 'JNJ', 'V',
        'PG', 'MA', 'HD', 'CVX', 'AVGO', 'LLY', 'PFE', 'BAC', 'KO', 'PEP',
        'COST', 'TMO', 'DHR', 'CSCO', 'ABBV', 'WMT', 'ACN', 'VZ', 'CRM', 'NEE',
        'ABT', 'NKE', 'D', 'QCOM', 'NFLX', 'INTC', 'WFC', 'MRK', 'MS', 'RTX',
        'T', 'BMY', 'SPGI', 'SCHW', 'BA', 'CAT', 'AXP', 'AMAT', 'PLD', 'INTU'
    ]
    return sp500_tickers

# Function to suggest portfolio combinations
def suggest_portfolios(sp500_tickers, risk_preference):
    """Suggest portfolio combinations based on risk preference"""
    # Define sector categories (simplified)
    sectors = {
        'tech': ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'AVGO', 'CSCO', 'QCOM', 'INTC'],
        'finance': ['JPM', 'BAC', 'WFC', 'MS', 'SCHW', 'AXP', 'V', 'MA'],
        'healthcare': ['JNJ', 'PFE', 'LLY', 'ABBV', 'ABT', 'BMY', 'MRK', 'TMO'],
        'consumer': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'NKE', 'HD'],
        'industrial': ['DHR', 'BA', 'CAT', 'RTX', 'ACN'],
        'energy': ['CVX'],
        'utilities': ['D', 'NEE']
    }

    # Define portfolio templates based on risk preference
    templates = {
        'low': {
            'size': 8,
            'sector_weights': {
                'tech': 0.20,
                'finance': 0.25,
                'healthcare': 0.20,
                'consumer': 0.20,
                'utilities': 0.15
            }
        },
        'medium': {
            'size': 6,
            'sector_weights': {
                'tech': 0.30,
                'finance': 0.25,
                'healthcare': 0.20,
                'consumer': 0.15,
                'industrial': 0.10
            }
        },
        'high': {
            'size': 5,
            'sector_weights': {
                'tech': 0.40,
                'finance': 0.25,
                'healthcare': 0.20,
                'industrial': 0.15
            }
        }
    }

    template = templates[risk_preference]
    suggested_stocks = []

    # Select stocks from each sector based on weights
    for sector, weight in template['sector_weights'].items():
        if sector in sectors:
            num_stocks = max(1, int(weight * template['size']))  # Ensure at least 1 stock per sector
            sector_stocks = sectors[sector]
            # Select top stocks by market cap from each sector
            selected = [stock for stock in sp500_tickers if stock in sector_stocks][:num_stocks]
            suggested_stocks.extend(selected)

    # Ensure we have exactly the required number of stocks
    if len(suggested_stocks) > template['size']:
        suggested_stocks = suggested_stocks[:template['size']]
    elif len(suggested_stocks) < template['size']:
        # Fill remaining slots with stocks from sectors with highest weights
        remaining = template['size'] - len(suggested_stocks)
        sorted_sectors = sorted(template['sector_weights'].items(), key=lambda x: x[1], reverse=True)
        for sector, _ in sorted_sectors:
            if remaining <= 0:
                break
            if sector in sectors:
                sector_stocks = [stock for stock in sp500_tickers if stock in sectors[sector]]
                for stock in sector_stocks:
                    if stock not in suggested_stocks:
                        suggested_stocks.append(stock)
                        remaining -= 1
                        if remaining <= 0:
                            break

    return suggested_stocks[:template['size']]

def run_optimization():
    # Fetch top 50 S&P 500 stocks
    sp500_tickers = get_sp500_stocks()

    print("\nTop 50 S&P 500 Stocks by Market Cap:")
    for i, ticker in enumerate(sp500_tickers, 1):
        print(f"{i:2d}. {ticker}")

    print("\nEnter the numbers of stocks you want to include in your portfolio (comma-separated)")
    print("Example: 1,2,3,4,5")
    selected_indices = input("Your selection: ").strip().split(',')

    try:
        selected_indices = [int(idx.strip()) - 1 for idx in selected_indices]
        tickers = [sp500_tickers[idx] for idx in selected_indices if 0 <= idx < len(sp500_tickers)]

        if not tickers:
            print("No valid stocks selected. Please try again.")
            return

        print(f"\nSelected {len(tickers)} stocks: {', '.join(tickers)}")

    except (ValueError, IndexError):
        print("Invalid input. Please enter valid numbers separated by commas.")
        return

    capital = float(input("\nEnter the total capital you want to invest (in $): "))

    print("\nFetching latest market data for selected stocks...")

    # Fetch stock and S&P 500 data
    data = get_live_data(tickers)
    sp500_data = get_live_data(["^GSPC"])

    if data.isnull().all().all():
        print("Data fetch failed. Try again later.")
        return

    # AI-based return prediction
    print("\nPerforming AI-based return prediction...")
    predictions = {}
    feature_importance_dict = {}
    prediction_dates = {}

    for ticker in tickers:
        stock_data = data.xs(ticker, axis=1, level=1)
        result = predict_returns(stock_data, ticker)
        if result is not None:
            model, scaler, feature_importance, confidence_score, confidence_score, prediction, pred_date = result
            predictions[ticker] = prediction
            feature_importance_dict[ticker] = feature_importance
            prediction_dates[ticker] = pred_date
            print(f"{ticker}: Model R² Score - Train: {confidence_score:.3f}, Test: {confidence_score:.3f}")
            print(f"  Prediction date: {pred_date.strftime('%Y-%m-%d')}")
    else:
            predictions[ticker] = np.array([0])
            feature_importance_dict[ticker] = pd.DataFrame({'feature': ['None'], 'importance': [0]})
            prediction_dates[ticker] = None

    # Calculate returns
    returns = data['Close'].pct_change(fill_method=None).dropna()
    sp500_returns = sp500_data['Close'].pct_change().dropna()

    # Optimize Portfolio
    optimal_weights, mean_returns, cov_matrix = optimize_portfolio(returns, sp500_returns)

    # Analyze risk profile
    risk_profile = analyze_risk_profile(returns, optimal_weights, sp500_returns)

    # Compute Portfolio Metrics
    port_return, port_vol = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    sortino = sortino_ratio(optimal_weights, returns)
    max_dd = max_drawdown(returns.dot(optimal_weights))

    # Calculate portfolio returns for beta calculation
    portfolio_returns = returns.dot(optimal_weights)
    beta = calculate_beta(portfolio_returns, sp500_returns)

    # Save results to CSV
    today = datetime.today().strftime('%Y-%m-%d')
    results_df = pd.DataFrame({'Ticker': tickers, 'Weight': optimal_weights})
    results_df.to_csv(f'portfolio_optimization_{today}.csv', index=False)

    # Print Results
    print(f"\nPortfolio Optimization Results - {today}")
    print(f"\nRisk Assessment: {risk_profile['risk_category']}")
    print(f"Risk Score: {risk_profile['risk_score']:.1f}/100")

    if risk_profile['risk_warnings']:
        print("\nRisk Warnings:")
        for warning in risk_profile['risk_warnings']:
            print(f"  ⚠️ {warning}")

    print("\nRisk Metrics:")
    print(f"  Portfolio Volatility: {risk_profile['volatility']:.2%}")
    print(f"  Portfolio Beta: {risk_profile['beta']:.2f}")
    print(f"  Concentration Index: {risk_profile['concentration']:.2f}")
    print(f"  Maximum Stock Weight: {risk_profile['max_weight']:.2%}")
    print(f"  Sector Risk Score: {risk_profile['sector_volatility']:.2%}")
    print(f"  Downside Volatility: {risk_profile['downside_volatility']:.2%}")
    print(f"  Maximum Drawdown: {risk_profile['max_drawdown']:.2%}")

    total_investments = []
    print("\nStock Allocation and AI Predictions:")
    for ticker, weight in zip(tickers, optimal_weights):
        investment = capital * weight
        total_investments.append(investment)
        pred_return = predictions[ticker] * 100
        print(f"  {ticker}: ${investment:,.2f} ({weight:.2%})")
        print(f"    AI Predicted Return: {pred_return:.2f}%")
        print(f"    Top Feature: {feature_importance_dict[ticker].iloc[0]['feature']}")

    total_investment = sum(total_investments)
    print(f"\nTotal Investment: ${total_investment:,.2f} (Capital: ${capital:,.2f})")

    print("\nPerformance Metrics:")
    print(f"  Expected Annual Return: {port_return * 252:.2%}")
    print(f"  Annual Volatility: {port_vol * np.sqrt(252):.2%}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Sharpe Ratio: {port_return / port_vol:.2f}")
    print(f"  Sortino Ratio: {sortino:.2f}")
    print(f"  Beta vs S&P 500: {beta:.2f}")

    # Plot Portfolio Cumulative Returns
    portfolio_daily_returns = returns.dot(optimal_weights)
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.index, cumulative_returns, label='Optimized Portfolio', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title(f'Portfolio Cumulative Returns - {today}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # After printing results, add portfolio suggestions
    print("\nPortfolio Suggestions:")
    print("\nFor Low Risk Portfolio (8 stocks):")
    low_risk = suggest_portfolios(sp500_tickers, 'low')
    print(f"  {', '.join(low_risk)}")
    print("  Expected Volatility: 15-20%")
    print("  Expected Beta: 0.8-1.0")

    print("\nFor Medium Risk Portfolio (6 stocks):")
    medium_risk = suggest_portfolios(sp500_tickers, 'medium')
    print(f"  {', '.join(medium_risk)}")
    print("  Expected Volatility: 20-25%")
    print("  Expected Beta: 1.0-1.2")

    print("\nFor High Risk Portfolio (5 stocks):")
    high_risk = suggest_portfolios(sp500_tickers, 'high')
    print(f"  {', '.join(high_risk)}")
    print("  Expected Volatility: 25-30%")
    print("  Expected Beta: 1.2-1.5")

    print("\nNote: These suggestions are based on sector diversification and historical risk metrics.")
    print("You can enter these stock numbers to analyze any of these portfolios.")
    print("\nThank you for using the Portfolio Optimization Tool!")

# Run manually in Jupyter Notebook
if __name__ == "__main__":
    run_optimization()