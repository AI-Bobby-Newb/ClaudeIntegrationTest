"""
Stock Backtester Package

A comprehensive backtesting framework for testing trading strategies.
"""

from .engine import BacktestEngine, Portfolio, Trade
from .strategies import Strategy, SMAStrategy, RSIStrategy, BollingerBandsStrategy, MomentumStrategy, MultiStrategyPortfolio
from .indicators import TechnicalIndicators

__version__ = "1.0.0"
__author__ = "AI-Bobby-Newb"

# Make key classes available at package level
__all__ = [
    'BacktestEngine',
    'Portfolio', 
    'Trade',
    'Strategy',
    'SMAStrategy',
    'RSIStrategy', 
    'BollingerBandsStrategy',
    'MomentumStrategy',
    'MultiStrategyPortfolio',
    'TechnicalIndicators'
]
