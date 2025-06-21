"""
Trading Strategies for the Backtesting Engine
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from .indicators import TechnicalIndicators


class Strategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.indicators = TechnicalIndicators()
        self.position = {}  # Track position for each symbol
        self.last_signals = {}  # Track last signal for each symbol
    
    @abstractmethod
    def generate_signals(self, current_date, current_data: Dict, historical_data: Dict) -> List[Dict]:
        """
        Generate trading signals for the current date.
        
        Args:
            current_date: Current date
            current_data: Dict of current data for each symbol {symbol: current_row}
            historical_data: Dict of full historical data {symbol: DataFrame}
            
        Returns:
            List of signal dictionaries with keys: 'symbol', 'action', 'quantity'
        """
        pass
    
    def get_position(self, symbol: str) -> int:
        """Get current position for a symbol (1=long, 0=flat, -1=short)."""
        return self.position.get(symbol, 0)
    
    def set_position(self, symbol: str, position: int):
        """Set position for a symbol."""
        self.position[symbol] = position


class SMAStrategy(Strategy):
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(f"SMA_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, current_date, current_data: Dict, historical_data: Dict) -> List[Dict]:
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_data:
                continue
                
            # Calculate moving averages up to current date
            data_up_to_date = data.loc[:current_date]
            
            if len(data_up_to_date) < self.long_window:
                continue
                
            short_sma = data_up_to_date['Close'].rolling(window=self.short_window).mean()
            long_sma = data_up_to_date['Close'].rolling(window=self.long_window).mean()
            
            current_short = short_sma.iloc[-1]
            current_long = long_sma.iloc[-1]
            prev_short = short_sma.iloc[-2] if len(short_sma) >= 2 else current_short
            prev_long = long_sma.iloc[-2] if len(long_sma) >= 2 else current_long
            
            current_position = self.get_position(symbol)
            
            # Golden Cross: Short SMA crosses above Long SMA (Buy signal)
            if (current_short > current_long and prev_short <= prev_long and current_position == 0):
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': 0  # Use all available cash
                })
                self.set_position(symbol, 1)
                
            # Death Cross: Short SMA crosses below Long SMA (Sell signal)
            elif (current_short < current_long and prev_short >= prev_long and current_position == 1):
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': 0  # Sell all shares
                })
                self.set_position(symbol, 0)
        
        return signals


class RSIStrategy(Strategy):
    """RSI Mean Reversion Strategy."""
    
    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"RSI_{rsi_period}_{oversold}_{overbought}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, current_date, current_data: Dict, historical_data: Dict) -> List[Dict]:
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_data:
                continue
                
            data_up_to_date = data.loc[:current_date]
            
            if len(data_up_to_date) < self.rsi_period + 1:
                continue
            
            # Calculate RSI
            rsi = self.indicators.rsi(data_up_to_date['Close'], self.rsi_period)
            current_rsi = rsi.iloc[-1]
            current_position = self.get_position(symbol)
            
            # Buy when RSI is oversold and we're not already long
            if current_rsi < self.oversold and current_position == 0:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': 0
                })
                self.set_position(symbol, 1)
                
            # Sell when RSI is overbought and we're long
            elif current_rsi > self.overbought and current_position == 1:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': 0
                })
                self.set_position(symbol, 0)
        
        return signals


class BollingerBandsStrategy(Strategy):
    """Bollinger Bands Mean Reversion Strategy."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, current_date, current_data: Dict, historical_data: Dict) -> List[Dict]:
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_data:
                continue
                
            data_up_to_date = data.loc[:current_date]
            
            if len(data_up_to_date) < self.period:
                continue
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(
                data_up_to_date['Close'], self.period, self.std_dev
            )
            
            current_price = current_data[symbol]['Close']
            current_upper = bb_upper.iloc[-1]
            current_lower = bb_lower.iloc[-1]
            current_position = self.get_position(symbol)
            
            # Buy when price touches lower band (oversold)
            if current_price <= current_lower and current_position == 0:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': 0
                })
                self.set_position(symbol, 1)
                
            # Sell when price touches upper band (overbought)
            elif current_price >= current_upper and current_position == 1:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': 0
                })
                self.set_position(symbol, 0)
        
        return signals


class MomentumStrategy(Strategy):
    """Price Momentum Strategy."""
    
    def __init__(self, lookback_period: int = 10, momentum_threshold: float = 0.02):
        super().__init__(f"Momentum_{lookback_period}_{momentum_threshold}")
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
    
    def generate_signals(self, current_date, current_data: Dict, historical_data: Dict) -> List[Dict]:
        signals = []
        
        for symbol, data in historical_data.items():
            if symbol not in current_data:
                continue
                
            data_up_to_date = data.loc[:current_date]
            
            if len(data_up_to_date) < self.lookback_period + 1:
                continue
            
            # Calculate momentum (percentage change over lookback period)
            current_price = current_data[symbol]['Close']
            past_price = data_up_to_date['Close'].iloc[-(self.lookback_period + 1)]
            momentum = (current_price - past_price) / past_price
            
            current_position = self.get_position(symbol)
            
            # Buy on positive momentum
            if momentum > self.momentum_threshold and current_position == 0:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': 0
                })
                self.set_position(symbol, 1)
                
            # Sell on negative momentum
            elif momentum < -self.momentum_threshold and current_position == 1:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': 0
                })
                self.set_position(symbol, 0)
        
        return signals


class MultiStrategyPortfolio(Strategy):
    """Combines multiple strategies with position sizing."""
    
    def __init__(self, strategies: List[Strategy], weights: List[float] = None):
        super().__init__("MultiStrategy")
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
        
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def generate_signals(self, current_date, current_data: Dict, historical_data: Dict) -> List[Dict]:
        all_signals = []
        
        # Get signals from each strategy
        for strategy, weight in zip(self.strategies, self.weights):
            strategy_signals = strategy.generate_signals(current_date, current_data, historical_data)
            
            # Apply weight to position sizes
            for signal in strategy_signals:
                signal['weight'] = weight
                all_signals.append(signal)
        
        # Combine signals for same symbol
        combined_signals = {}
        for signal in all_signals:
            symbol = signal['symbol']
            if symbol not in combined_signals:
                combined_signals[symbol] = {'BUY': 0, 'SELL': 0}
            
            action = signal['action']
            weight = signal.get('weight', 1.0)
            combined_signals[symbol][action] += weight
        
        # Generate final signals
        final_signals = []
        for symbol, actions in combined_signals.items():
            net_signal = actions['BUY'] - actions['SELL']
            
            if net_signal > 0.5:  # Threshold for buy
                final_signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': 0
                })
            elif net_signal < -0.5:  # Threshold for sell
                final_signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': 0
                })
        
        return final_signals
