"""
Backtesting Engine - Core logic for running strategy backtests
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yfinance as yf


class Trade:
    """Represents a single trade."""
    
    def __init__(self, date, trade_type, symbol, price, quantity, commission=0):
        self.date = date
        self.type = trade_type  # 'BUY' or 'SELL'
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.commission = commission
        self.value = price * quantity + commission
    
    def __repr__(self):
        return f"Trade({self.date}, {self.type}, {self.symbol}, {self.quantity}@${self.price:.2f})"


class Portfolio:
    """Manages portfolio state and calculations."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: quantity}
        self.trades = []
        self.portfolio_history = []
    
    def buy(self, symbol: str, price: float, quantity: int, date, commission: float = 0):
        """Execute a buy order."""
        total_cost = (price * quantity) + commission
        
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            
            trade = Trade(date, 'BUY', symbol, price, quantity, commission)
            self.trades.append(trade)
            return True
        return False
    
    def sell(self, symbol: str, price: float, quantity: int, date, commission: float = 0):
        """Execute a sell order."""
        if self.positions.get(symbol, 0) >= quantity:
            total_value = (price * quantity) - commission
            self.cash += total_value
            self.positions[symbol] -= quantity
            
            if self.positions[symbol] == 0:
                del self.positions[symbol]
            
            trade = Trade(date, 'SELL', symbol, price, quantity, commission)
            self.trades.append(trade)
            return True
        return False
    
    def get_position(self, symbol: str) -> int:
        """Get current position for a symbol."""
        return self.positions.get(symbol, 0)
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            self.positions.get(symbol, 0) * price 
            for symbol, price in prices.items()
        )
        return self.cash + positions_value
    
    def record_value(self, date, prices: Dict[str, float]):
        """Record portfolio value for this date."""
        total_value = self.get_portfolio_value(prices)
        self.portfolio_history.append({
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash
        })


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0):
        self.initial_capital = initial_capital
        self.commission = commission
        self.portfolio = Portfolio(initial_capital)
        self.data = {}
        self.results = None
    
    def add_data(self, symbol: str, data: pd.DataFrame):
        """Add price data for a symbol."""
        self.data[symbol] = data.copy()
    
    def fetch_data(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> bool:
        """Fetch data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return False
            
            self.add_data(symbol, data)
            print(f"✓ Fetched {len(data)} days of data for {symbol}")
            return True
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return False
    
    def run_strategy(self, strategy, symbols: List[str] = None):
        """Run a trading strategy against the data."""
        if not symbols:
            symbols = list(self.data.keys())
        
        if not symbols:
            raise ValueError("No symbols available. Add data first.")
        
        # Get the date range from the first symbol
        dates = self.data[symbols[0]].index
        
        print(f"Running backtest from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        
        # Run strategy for each date
        for date in dates:
            # Get current prices for all symbols
            current_prices = {}
            current_data = {}
            
            for symbol in symbols:
                if date in self.data[symbol].index:
                    row = self.data[symbol].loc[date]
                    current_prices[symbol] = row['Close']
                    current_data[symbol] = row
            
            # Record portfolio value
            self.portfolio.record_value(date, current_prices)
            
            # Generate signals from strategy
            signals = strategy.generate_signals(date, current_data, self.data)
            
            # Execute trades based on signals
            for signal in signals:
                self._execute_signal(signal, date, current_prices)
        
        # Calculate final results
        self.results = self._calculate_results()
        return self.results
    
    def _execute_signal(self, signal: Dict, date, prices: Dict[str, float]):
        """Execute a trading signal."""
        symbol = signal['symbol']
        action = signal['action']  # 'BUY' or 'SELL'
        quantity = signal.get('quantity', 0)
        
        if symbol not in prices:
            return False
        
        price = prices[symbol]
        
        if action == 'BUY':
            # If quantity is 0, use all available cash
            if quantity == 0:
                quantity = int(self.portfolio.cash // (price + self.commission))
            
            success = self.portfolio.buy(symbol, price, quantity, date, self.commission)
            if success:
                print(f"{date.strftime('%Y-%m-%d')}: BUY {quantity} {symbol} @ ${price:.2f}")
            return success
            
        elif action == 'SELL':
            current_position = self.portfolio.get_position(symbol)
            
            # If quantity is 0, sell all shares
            if quantity == 0:
                quantity = current_position
            
            # Don't sell more than we have
            quantity = min(quantity, current_position)
            
            if quantity > 0:
                success = self.portfolio.sell(symbol, price, quantity, date, self.commission)
                if success:
                    print(f"{date.strftime('%Y-%m-%d')}: SELL {quantity} {symbol} @ ${price:.2f}")
                return success
        
        return False
    
    def _calculate_results(self) -> Dict:
        """Calculate comprehensive backtest results."""
        if not self.portfolio.portfolio_history:
            return {}
        
        # Convert portfolio history to DataFrame
        df = pd.DataFrame(self.portfolio.portfolio_history)
        df.set_index('date', inplace=True)
        
        # Calculate returns
        df['returns'] = df['total_value'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Final values
        final_value = df['total_value'].iloc[-1]
        total_return = (final_value / self.initial_capital) - 1
        
        # Calculate buy and hold return for comparison (using first symbol)
        if self.data:
            first_symbol = list(self.data.keys())[0]
            first_data = self.data[first_symbol]
            buy_hold_return = (first_data['Close'].iloc[-1] / first_data['Close'].iloc[0]) - 1
        else:
            buy_hold_return = 0
        
        # Risk metrics
        returns_series = df['returns'].dropna()
        volatility = returns_series.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (returns_series.mean() * 252) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        total_trades = len(self.portfolio.trades)
        winning_trades = 0
        losing_trades = 0
        
        # Calculate win/loss for paired trades
        buy_trades = [t for t in self.portfolio.trades if t.type == 'BUY']
        sell_trades = [t for t in self.portfolio.trades if t.type == 'SELL']
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i].price
            sell_price = sell_trades[i].price
            if sell_price > buy_price:
                winning_trades += 1
            else:
                losing_trades += 1
        
        win_rate = winning_trades / max(winning_trades + losing_trades, 1)
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'buy_hold_return': buy_hold_return,
            'buy_hold_return_pct': buy_hold_return * 100,
            'excess_return': total_return - buy_hold_return,
            'excess_return_pct': (total_return - buy_hold_return) * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'portfolio_history': df,
            'trades': self.portfolio.trades
        }
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio value history as DataFrame."""
        if self.results and 'portfolio_history' in self.results:
            return self.results['portfolio_history']
        return pd.DataFrame()
    
    def print_summary(self):
        """Print a summary of backtest results."""
        if not self.results:
            print("No results available. Run a backtest first.")
            return
        
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Initial Capital:        ${self.results['initial_capital']:>12,.2f}")
        print(f"Final Value:            ${self.results['final_value']:>12,.2f}")
        print(f"Total Return:           {self.results['total_return_pct']:>12.2f}%")
        print(f"Buy & Hold Return:      {self.results['buy_hold_return_pct']:>12.2f}%")
        print(f"Excess Return:          {self.results['excess_return_pct']:>12.2f}%")
        print()
        print(f"Volatility (Annual):    {self.results['volatility']*100:>12.2f}%")
        print(f"Sharpe Ratio:           {self.results['sharpe_ratio']:>12.2f}")
        print(f"Max Drawdown:           {self.results['max_drawdown_pct']:>12.2f}%")
        print()
        print(f"Total Trades:           {self.results['total_trades']:>12}")
        print(f"Winning Trades:         {self.results['winning_trades']:>12}")
        print(f"Win Rate:               {self.results['win_rate_pct']:>12.2f}%")
        print("="*60)
