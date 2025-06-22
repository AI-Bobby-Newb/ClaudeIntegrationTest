#!/usr/bin/env python3
"""
Stock Backtester - Main Entry Point
A comprehensive backtesting application for testing trading strategies.
"""

import argparse
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import talib

# Import our backtesting framework
from backtester import (
    BacktestEngine, 
    SMAStrategy, 
    RSIStrategy, 
    BollingerBandsStrategy, 
    MomentumStrategy,
    MultiStrategyPortfolio
)


def create_strategy(strategy_name: str, **kwargs):
    """Factory function to create strategies."""
    strategies = {
        'sma': SMAStrategy,
        'sma_crossover': SMAStrategy,
        'rsi': RSIStrategy,
        'bollinger': BollingerBandsStrategy,
        'momentum': MomentumStrategy
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        available = list(strategies.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    
    return strategy_class(**kwargs)


def plot_results(engine, symbol):
    """Plot backtest results with portfolio performance."""
    if not engine.results:
        print("No results to plot")
        return
    
    portfolio_history = engine.get_portfolio_history()
    price_data = engine.data[symbol]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Price chart with trades
    ax1.plot(price_data.index, price_data['Close'], label='Close Price', linewidth=1.5)
    
    # Mark buy/sell trades
    for trade in engine.portfolio.trades:
        color = 'green' if trade.type == 'BUY' else 'red'
        marker = '^' if trade.type == 'BUY' else 'v'
        ax1.scatter(trade.date, trade.price, color=color, marker=marker, s=100, alpha=0.8)
    
    ax1.set_title(f'{symbol} - Price and Trading Signals')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Portfolio value over time
    if not portfolio_history.empty:
        ax2.plot(portfolio_history.index, portfolio_history['total_value'], 
                label='Portfolio Value', linewidth=2, color='blue')
        ax2.axhline(y=engine.initial_capital, color='gray', linestyle='--', 
                   alpha=0.5, label='Initial Capital')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Returns comparison
    if not portfolio_history.empty:
        cumulative_returns = portfolio_history['cumulative_returns'] * 100
        
        # Calculate buy & hold returns
        buy_hold_returns = (price_data['Close'] / price_data['Close'].iloc[0] - 1) * 100
        
        ax3.plot(cumulative_returns.index, cumulative_returns, 
                label='Strategy Returns', linewidth=2, color='blue')
        ax3.plot(buy_hold_returns.index, buy_hold_returns, 
                label='Buy & Hold Returns', linewidth=2, color='orange', alpha=0.7)
        ax3.set_title('Cumulative Returns Comparison')
        ax3.set_ylabel('Returns (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Drawdown chart
    if not portfolio_history.empty:
        rolling_max = portfolio_history['total_value'].expanding().max()
        drawdown = (portfolio_history['total_value'] - rolling_max) / rolling_max * 100
        
        ax4.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax4.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax4.set_title('Portfolio Drawdown')
        ax4.set_ylabel('Drawdown (%)')
        ax4.set_xlabel('Date')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_comparison(symbols, strategy_name, start_date, end_date, initial_capital, **strategy_kwargs):
    """Run backtest comparison across multiple symbols."""
    print(f"\n🔄 RUNNING COMPARISON ACROSS {len(symbols)} SYMBOLS")
    print("=" * 60)
    
    results_summary = []
    
    for symbol in symbols:
        print(f"\n📈 Testing {symbol}...")
        
        # Create engine and strategy
        engine = BacktestEngine(initial_capital=initial_capital)
        strategy = create_strategy(strategy_name, **strategy_kwargs)
        
        # Fetch data and run backtest
        if engine.fetch_data(symbol, start_date, end_date):
            try:
                results = engine.run_strategy(strategy)
                
                results_summary.append({
                    'Symbol': symbol,
                    'Total Return (%)': results['total_return_pct'],
                    'Buy & Hold (%)': results['buy_hold_return_pct'],
                    'Excess Return (%)': results['excess_return_pct'],
                    'Sharpe Ratio': results['sharpe_ratio'],
                    'Max Drawdown (%)': results['max_drawdown_pct'],
                    'Win Rate (%)': results['win_rate_pct'],
                    'Total Trades': results['total_trades']
                })
                
            except Exception as e:
                print(f"Error running backtest for {symbol}: {e}")
        else:
            print(f"Failed to fetch data for {symbol}")
    
    # Display comparison results
    if results_summary:
        df = pd.DataFrame(results_summary)
        print(f"\n📊 COMPARISON RESULTS")
        print("=" * 80)
        print(df.to_string(index=False, float_format='%.2f'))
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS")
        print("-" * 40)
        print(f"Average Total Return:    {df['Total Return (%)'].mean():>8.2f}%")
        print(f"Average Excess Return:   {df['Excess Return (%)'].mean():>8.2f}%")
        print(f"Average Sharpe Ratio:    {df['Sharpe Ratio'].mean():>8.2f}")
        print(f"Average Win Rate:        {df['Win Rate (%)'].mean():>8.2f}%")
        print(f"Best Performer:          {df.loc[df['Total Return (%)'].idxmax(), 'Symbol']}")
        print(f"Worst Performer:         {df.loc[df['Total Return (%)'].idxmin(), 'Symbol']}")


def main():
    """Main function to run the backtester."""
    parser = argparse.ArgumentParser(description='Advanced Stock Backtester')
    
    # Basic parameters
    parser.add_argument('--symbol', '-s', type=str, default='AAPL', 
                       help='Stock symbol (default: AAPL)')
    parser.add_argument('--symbols', nargs='+', 
                       help='Multiple symbols for comparison (e.g., --symbols AAPL MSFT GOOGL)')
    parser.add_argument('--start', type=str, default='2022-01-01', 
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, 
                       help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--capital', '-c', type=float, default=10000, 
                       help='Initial capital (default: 10000)')
    
    # Strategy parameters
    parser.add_argument('--strategy', type=str, default='sma', 
                       help='Strategy: sma, rsi, bollinger, momentum (default: sma)')
    parser.add_argument('--short-window', type=int, default=20, 
                       help='Short window for SMA strategy (default: 20)')
    parser.add_argument('--long-window', type=int, default=50, 
                       help='Long window for SMA strategy (default: 50)')
    parser.add_argument('--rsi-period', type=int, default=14, 
                       help='RSI period (default: 14)')
    parser.add_argument('--oversold', type=int, default=30, 
                       help='RSI oversold level (default: 30)')
    parser.add_argument('--overbought', type=int, default=70, 
                       help='RSI overbought level (default: 70)')
    
    # Output options
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--compare', action='store_true', 
                       help='Run comparison across multiple symbols')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine symbols to test
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = [args.symbol.upper()]
    
    # Strategy parameters
    strategy_kwargs = {}
    if args.strategy.lower() in ['sma', 'sma_crossover']:
        strategy_kwargs = {
            'short_window': args.short_window,
            'long_window': args.long_window
        }
    elif args.strategy.lower() == 'rsi':
        strategy_kwargs = {
            'rsi_period': args.rsi_period,
            'oversold': args.oversold,
            'overbought': args.overbought
        }
    
    print("=" * 60)
    print("🚀 ADVANCED STOCK BACKTESTER")
    print("=" * 60)
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Period: {args.start} to {args.end or 'today'}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Symbols: {', '.join(symbols)}")
    if strategy_kwargs:
        print(f"Strategy Parameters: {strategy_kwargs}")
    print("-" * 60)
    
    if args.compare and len(symbols) > 1:
        # Run comparison mode
        run_comparison(symbols, args.strategy, args.start, args.end, 
                      args.capital, **strategy_kwargs)
    else:
        # Run single backtest
        symbol = symbols[0]
        
        # Create engine and strategy
        engine = BacktestEngine(initial_capital=args.capital)
        strategy = create_strategy(args.strategy, **strategy_kwargs)
        
        # Fetch data and run backtest
        if engine.fetch_data(symbol, args.start, args.end):
            print(f"\n🔄 Running {strategy.name} strategy...")
            
            try:
                results = engine.run_strategy(strategy)
                
                # Print detailed results
                engine.print_summary()
                
                if args.verbose and engine.portfolio.trades:
                    print(f"\n💼 DETAILED TRADE HISTORY")
                    print("-" * 60)
                    for trade in engine.portfolio.trades:
                        print(f"{trade.date.strftime('%Y-%m-%d')}: {trade}")
                
                # Show plots if requested
                if args.plot:
                    plot_results(engine, symbol)
                    
            except Exception as e:
                print(f"❌ Error running backtest: {e}")
                sys.exit(1)
        else:
            print(f"❌ Failed to fetch data for {symbol}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✅ Backtest completed successfully!")


if __name__ == "__main__":
    main()
