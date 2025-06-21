#!/usr/bin/env python3
"""
Stock Backtester - Main Entry Point
A comprehensive backtesting application for testing trading strategies.
"""

import argparse
import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class SimpleBacktester:
    """Simple backtesting engine for stock strategies."""
    
    def __init__(self, symbol, start_date, end_date=None, initial_capital=10000):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.data = None
        self.trades = []
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.cash = initial_capital
        self.shares = 0
        
    def fetch_data(self):
        """Fetch historical stock data."""
        print(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}...")
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=self.start_date, end=self.end_date)
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            print(f"✓ Successfully fetched {len(self.data)} days of data")
            return True
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return False
    
    def add_indicators(self):
        """Add technical indicators to the data."""
        # Simple Moving Averages
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.data['BB_middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (bb_std * 2)
        self.data['BB_lower'] = self.data['BB_middle'] - (bb_std * 2)
        
    def sma_crossover_strategy(self):
        """Simple Moving Average Crossover Strategy."""
        signals = []
        
        for i in range(len(self.data)):
            if i < 50:  # Need enough data for SMA_50
                signals.append(0)
                continue
                
            current_price = self.data.iloc[i]['Close']
            sma_20 = self.data.iloc[i]['SMA_20']
            sma_50 = self.data.iloc[i]['SMA_50']
            prev_sma_20 = self.data.iloc[i-1]['SMA_20']
            prev_sma_50 = self.data.iloc[i-1]['SMA_50']
            
            # Buy signal: SMA_20 crosses above SMA_50
            if (sma_20 > sma_50 and prev_sma_20 <= prev_sma_50 and self.position == 0):
                self.buy(current_price, self.data.index[i])
                signals.append(1)
            # Sell signal: SMA_20 crosses below SMA_50
            elif (sma_20 < sma_50 and prev_sma_20 >= prev_sma_50 and self.position == 1):
                self.sell(current_price, self.data.index[i])
                signals.append(-1)
            else:
                signals.append(0)
                
        self.data['Signal'] = signals
        
    def buy(self, price, date):
        """Execute buy order."""
        if self.cash > price:
            self.shares = self.cash // price
            self.cash -= self.shares * price
            self.position = 1
            self.trades.append({
                'Date': date,
                'Type': 'BUY',
                'Price': price,
                'Shares': self.shares,
                'Cash': self.cash
            })
            print(f"BUY: {self.shares} shares at ${price:.2f} on {date.strftime('%Y-%m-%d')}")
    
    def sell(self, price, date):
        """Execute sell order."""
        if self.shares > 0:
            self.cash += self.shares * price
            sold_shares = self.shares
            self.shares = 0
            self.position = 0
            self.trades.append({
                'Date': date,
                'Type': 'SELL',
                'Price': price,
                'Shares': sold_shares,
                'Cash': self.cash
            })
            print(f"SELL: {sold_shares} shares at ${price:.2f} on {date.strftime('%Y-%m-%d')}")
    
    def calculate_performance(self):
        """Calculate performance metrics."""
        # Final portfolio value
        final_price = self.data['Close'].iloc[-1]
        portfolio_value = self.cash + (self.shares * final_price)
        
        # Returns
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        # Buy and hold return for comparison
        buy_hold_return = (final_price - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0] * 100
        
        # Calculate daily returns for Sharpe ratio
        self.data['Portfolio_Value'] = self.cash
        for i, row in self.data.iterrows():
            current_shares = self.shares if i >= self.data.index[-1] else 0
            # This is simplified - in reality you'd track portfolio value day by day
            pass
        
        return {
            'Initial Capital': self.initial_capital,
            'Final Portfolio Value': portfolio_value,
            'Total Return (%)': total_return,
            'Buy & Hold Return (%)': buy_hold_return,
            'Number of Trades': len(self.trades),
            'Final Cash': self.cash,
            'Final Shares': self.shares,
            'Alpha (vs Buy & Hold)': total_return - buy_hold_return
        }
    
    def run_backtest(self, strategy='sma_crossover'):
        """Run the complete backtest."""
        if not self.fetch_data():
            return None
            
        self.add_indicators()
        
        if strategy == 'sma_crossover':
            self.sma_crossover_strategy()
        else:
            print(f"Strategy '{strategy}' not implemented yet.")
            return None
            
        return self.calculate_performance()
    
    def plot_results(self):
        """Plot the backtest results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price and moving averages
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=1)
        ax1.plot(self.data.index, self.data['SMA_20'], label='SMA 20', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_50'], label='SMA 50', alpha=0.7)
        
        # Buy/Sell signals
        buy_signals = self.data[self.data['Signal'] == 1]
        sell_signals = self.data[self.data['Signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy')
        ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell')
        
        ax1.set_title(f'{self.symbol} - Price and Trading Signals')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI
        ax2.plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.set_title('RSI Indicator')
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the backtester."""
    parser = argparse.ArgumentParser(description='Stock Backtester')
    parser.add_argument('--symbol', '-s', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--start', type=str, default='2022-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--capital', '-c', type=float, default=10000, help='Initial capital (default: 10000)')
    parser.add_argument('--strategy', type=str, default='sma_crossover', help='Strategy to use (default: sma_crossover)')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 STOCK BACKTESTER")
    print("=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end or 'today'}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Strategy: {args.strategy}")
    print("-" * 50)
    
    # Initialize backtester
    bt = SimpleBacktester(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )
    
    # Run backtest
    results = bt.run_backtest(strategy=args.strategy)
    
    if results:
        print("\n📊 BACKTEST RESULTS")
        print("-" * 50)
        for key, value in results.items():
            if isinstance(value, float):
                if '%' in key:
                    print(f"{key:<25}: {value:>8.2f}%")
                else:
                    print(f"{key:<25}: ${value:>10,.2f}")
            else:
                print(f"{key:<25}: {value:>12}")
        
        print("\n💼 TRADE HISTORY")
        print("-" * 50)
        if bt.trades:
            for trade in bt.trades:
                print(f"{trade['Date'].strftime('%Y-%m-%d')}: {trade['Type']} {trade['Shares']} shares @ ${trade['Price']:.2f}")
        else:
            print("No trades executed.")
        
        if args.plot:
            bt.plot_results()
    
    print("\n" + "=" * 50)
    print("✅ Backtest completed!")
    
if __name__ == "__main__":
    main()
