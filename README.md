# Stock Backtester

A comprehensive stock backtesting application that allows you to test trading strategies against historical market data.

## Features

- **Multiple Trading Strategies**: SMA crossover, RSI mean reversion, Bollinger Bands, Momentum
- **Historical Data Integration**: Fetch stock data automatically from Yahoo Finance
- **Performance Analytics**: Detailed metrics including returns, Sharpe ratio, max drawdown
- **Multi-Asset Testing**: Compare strategies across multiple stocks simultaneously
- **Advanced Visualization**: Interactive charts showing portfolio performance and drawdowns
- **Professional Metrics**: Win rate, excess returns, volatility, and risk-adjusted returns

## Tech Stack

- **Backend**: Python with modular architecture
- **Data Processing**: pandas, numpy, yfinance
- **Visualization**: matplotlib with comprehensive charting
- **Technical Analysis**: Custom indicators library with 10+ indicators

## Installation

```bash
# Clone the repository
git clone https://github.com/AI-Bobby-Newb/ClaudeIntegrationTest.git
cd ClaudeIntegrationTest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Simple SMA strategy on Apple stock
python main.py --symbol AAPL --strategy sma --plot

# RSI strategy with custom parameters
python main.py --symbol TSLA --strategy rsi --rsi-period 21 --oversold 25 --plot
```

### Python API Usage

```python
from backtester import BacktestEngine, SMAStrategy

# Create engine and strategy
engine = BacktestEngine(initial_capital=10000)
strategy = SMAStrategy(short_window=20, long_window=50)

# Run backtest
engine.fetch_data('AAPL', '2022-01-01', '2024-01-01')
results = engine.run_strategy(strategy)

# Print results
engine.print_summary()
```

## Command Line Examples

### Single Stock Analysis

```bash
# Basic SMA crossover strategy
python main.py --symbol AAPL --strategy sma --start 2022-01-01 --plot

# Custom SMA windows
python main.py --symbol MSFT --strategy sma --short-window 10 --long-window 30 --plot

# RSI mean reversion
python main.py --symbol GOOGL --strategy rsi --rsi-period 14 --oversold 30 --overbought 70 --plot

# Bollinger Bands strategy
python main.py --symbol NVDA --strategy bollinger --start 2023-01-01 --plot

# Momentum strategy
python main.py --symbol TSLA --strategy momentum --start 2022-01-01 --verbose --plot
```

### Multi-Stock Comparison

```bash
# Compare SMA strategy across tech stocks
python main.py --symbols AAPL MSFT GOOGL NVDA --strategy sma --compare

# RSI strategy comparison with custom parameters
python main.py --symbols TSLA AAPL MSFT --strategy rsi --oversold 25 --overbought 75 --compare

# Test across different sectors
python main.py --symbols AAPL JPM XOM JNJ --strategy sma --start 2021-01-01 --compare
```

### Advanced Usage

```bash
# High capital backtest with detailed output
python main.py --symbol AAPL --strategy sma --capital 100000 --verbose --plot

# Short-term momentum strategy
python main.py --symbol QQQ --strategy momentum --start 2024-01-01 --plot

# Custom date range analysis
python main.py --symbol SPY --strategy sma --start 2020-03-01 --end 2020-12-31 --plot
```

## Strategy Examples

### 1. Simple Moving Average (SMA) Crossover

**Strategy**: Buy when 20-day SMA crosses above 50-day SMA, sell when it crosses below.

```bash
python main.py --symbol AAPL --strategy sma --short-window 20 --long-window 50 --plot
```

**Example Output:**
```
============================================================
📊 BACKTEST RESULTS SUMMARY
============================================================
Initial Capital:        $   10,000.00
Final Value:            $   12,847.32
Total Return:                   28.47%
Buy & Hold Return:              31.20%
Excess Return:                  -2.73%

Volatility (Annual):            18.45%
Sharpe Ratio:                    1.34
Max Drawdown:                  -12.85%

Total Trades:                       8
Winning Trades:                     5
Win Rate:                       62.50%
============================================================
```

### 2. RSI Mean Reversion

**Strategy**: Buy when RSI drops below 30 (oversold), sell when RSI rises above 70 (overbought).

```bash
python main.py --symbol TSLA --strategy rsi --rsi-period 14 --oversold 30 --overbought 70 --plot
```

### 3. Bollinger Bands

**Strategy**: Buy when price touches lower band, sell when price touches upper band.

```bash
python main.py --symbol NVDA --strategy bollinger --start 2023-01-01 --plot
```

### 4. Multi-Stock Comparison

Compare how the same strategy performs across different stocks:

```bash
python main.py --symbols AAPL MSFT GOOGL AMZN TSLA --strategy sma --compare
```

**Example Comparison Output:**
```
📊 COMPARISON RESULTS
================================================================================
Symbol  Total Return (%)  Buy & Hold (%)  Excess Return (%)  Sharpe Ratio  Max Drawdown (%)  Win Rate (%)  Total Trades
  AAPL             28.47           31.20              -2.73          1.34            -12.85         62.50             8
  MSFT             15.23           18.90              -3.67          0.98            -15.42         55.56             9
 GOOGL             22.18           25.67              -3.49          1.12            -18.90         60.00            10
  AMZN             35.67           42.33              -6.66          1.45            -22.17         58.33            12
  TSLA             48.92           52.14              -3.22          1.67            -28.45         63.64            11

📈 SUMMARY STATISTICS
----------------------------------------
Average Total Return:        30.09%
Average Excess Return:       -3.95%
Average Sharpe Ratio:         1.31
Average Win Rate:            60.01%
Best Performer:              TSLA
Worst Performer:             MSFT
```

## Available Strategies

| Strategy | Description | Parameters |
|----------|-------------|------------|
| **SMA** | Simple Moving Average Crossover | `--short-window`, `--long-window` |
| **RSI** | RSI Mean Reversion | `--rsi-period`, `--oversold`, `--overbought` |
| **Bollinger** | Bollinger Bands Mean Reversion | Default: 20-period, 2 std dev |
| **Momentum** | Price Momentum Following | Default: 10-day lookback |

## Performance Metrics Explained

- **Total Return**: Overall percentage gain/loss
- **Buy & Hold Return**: Return from simply buying and holding
- **Excess Return**: Strategy return minus buy & hold return
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## Sample Results

Here's what you might see when running the backtester:

### Portfolio Performance Chart
- **Top Left**: Stock price with buy/sell signals marked
- **Top Right**: Portfolio value over time vs initial capital
- **Bottom Left**: Strategy returns vs buy & hold comparison
- **Bottom Right**: Drawdown periods highlighted in red

### Trade Execution
```bash
2022-03-15: BUY 45 AAPL @ $150.62
2022-05-20: SELL 45 AAPL @ $163.21
2022-08-10: BUY 52 AAPL @ $145.83
2022-11-02: SELL 52 AAPL @ $155.74
```

## Use Cases

### For Traders
- **Strategy Development**: Test ideas before risking real money
- **Performance Analysis**: Compare strategies across different time periods
- **Risk Assessment**: Understand maximum drawdowns and volatility

### For Developers
- **Strategy Framework**: Easily add custom trading strategies
- **Data Pipeline**: Robust data fetching and processing
- **Extensible Design**: Add new indicators and metrics

### For Researchers
- **Academic Studies**: Analyze market efficiency and strategy performance
- **Comparative Analysis**: Test strategies across different markets and periods
- **Statistical Analysis**: Access to detailed performance metrics

## Contributing

We welcome contributions! Here are some ways you can help:

1. **Add New Strategies**: Implement MACD, Stochastic, or machine learning strategies
2. **Improve Visualizations**: Add more chart types or interactive plots
3. **Enhance Metrics**: Add new performance calculations
4. **Bug Fixes**: Report issues or submit fixes

```bash
# Development workflow
git checkout -b feature/new-strategy
# Make your changes
python -m pytest tests/  # Run tests
git commit -m "Add MACD strategy"
git push origin feature/new-strategy
# Open a Pull Request
```

## Roadmap

- [ ] **Web Interface**: Browser-based backtesting dashboard
- [ ] **Real-time Data**: Live market data integration
- [ ] **More Asset Classes**: Crypto, forex, and futures support
- [ ] **Portfolio Optimization**: Automatic parameter tuning
- [ ] **Paper Trading**: Connect to broker APIs for live testing
- [ ] **Machine Learning**: AI-powered strategy development

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research and consider consulting with financial professionals before making investment decisions.

---

**Built with ❤️ for the trading community**
