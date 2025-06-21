# Stock Backtester

A comprehensive stock backtesting application that allows you to test trading strategies against historical market data.

## Features

- **Historical Data Integration**: Fetch stock data from multiple sources
- **Strategy Builder**: Create and customize trading strategies  
- **Performance Analytics**: Detailed metrics including returns, Sharpe ratio, max drawdown
- **Visualization**: Interactive charts and performance graphs
- **Portfolio Management**: Test multiple assets and position sizing
- **Risk Management**: Stop-loss, take-profit, and position limits

## Tech Stack

- **Backend**: Python with Flask/FastAPI
- **Data Processing**: pandas, numpy, yfinance
- **Visualization**: matplotlib, plotly
- **Frontend**: React/Next.js (optional web interface)
- **Database**: SQLite for local storage

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ClaudeIntegrationTest.git
cd ClaudeIntegrationTest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from backtester import Backtester, Strategy

# Create a simple moving average strategy
strategy = Strategy()
strategy.add_indicator('SMA', period=20)
strategy.add_rule('buy_when_price_above_sma')

# Run backtest
bt = Backtester('AAPL', start_date='2020-01-01', initial_capital=10000)
results = bt.run(strategy)
print(results.summary())
```

## Usage

Run a basic backtest:
```bash
python main.py --symbol AAPL --strategy sma_crossover --start 2020-01-01
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
