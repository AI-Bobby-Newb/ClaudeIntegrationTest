"""
Technical Indicators for Stock Analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple


class TechnicalIndicators:
    """Collection of technical analysis indicators."""
    
    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            prices: Price series
            window: Period for RSI calculation
            
        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            prices: Price series
            window: Period for moving average
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return upper, middle, lower
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_window: %K period
            d_window: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for ATR calculation
            
        Returns:
            ATR values
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Williams %R.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for calculation
            
        Returns:
            Williams %R values (-100 to 0)
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Commodity Channel Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for CCI calculation
            
        Returns:
            CCI values
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def momentum(prices: pd.Series, window: int = 10) -> pd.Series:
        """
        Price Momentum.
        
        Args:
            prices: Price series
            window: Lookback period
            
        Returns:
            Momentum values
        """
        return prices.diff(window)
    
    @staticmethod
    def roc(prices: pd.Series, window: int = 10) -> pd.Series:
        """
        Rate of Change.
        
        Args:
            prices: Price series
            window: Lookback period
            
        Returns:
            ROC values as percentage
        """
        return ((prices / prices.shift(window)) - 1) * 100
    
    @staticmethod
    def support_resistance_levels(prices: pd.Series, window: int = 20) -> dict:
        """
        Identify support and resistance levels.
        
        Args:
            prices: Price series
            window: Period for identifying levels
            
        Returns:
            Dictionary with support and resistance levels
        """
        rolling_max = prices.rolling(window=window).max()
        rolling_min = prices.rolling(window=window).min()
        
        # Find local maxima and minima
        resistance_levels = rolling_max[rolling_max.shift(1) < rolling_max].dropna().unique()
        support_levels = rolling_min[rolling_min.shift(1) > rolling_min].dropna().unique()
        
        return {
            'resistance': sorted(resistance_levels, reverse=True)[:5],  # Top 5
            'support': sorted(support_levels)[:5]  # Bottom 5
        }
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all common technical indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        df = df.copy()
        indicators = TechnicalIndicators()
        
        # Moving averages
        df['SMA_20'] = indicators.sma(df['Close'], 20)
        df['SMA_50'] = indicators.sma(df['Close'], 50)
        df['EMA_12'] = indicators.ema(df['Close'], 12)
        df['EMA_26'] = indicators.ema(df['Close'], 26)
        
        # RSI
        df['RSI'] = indicators.rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = indicators.bollinger_bands(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = indicators.macd(df['Close'])
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = indicators.stochastic(df['High'], df['Low'], df['Close'])
        
        # ATR
        df['ATR'] = indicators.atr(df['High'], df['Low'], df['Close'])
        
        # Williams %R
        df['Williams_R'] = indicators.williams_r(df['High'], df['Low'], df['Close'])
        
        # Momentum indicators
        df['Momentum'] = indicators.momentum(df['Close'])
        df['ROC'] = indicators.roc(df['Close'])
        
        return df
