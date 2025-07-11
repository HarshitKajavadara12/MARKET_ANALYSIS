"""
Technical Indicators Module
Market Research System v1.0

Provides technical analysis indicators for Indian stock market data.
Includes both TA-Lib wrappers and custom implementations.

Created: February 2022
Last Updated: March 2022
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union
import warnings

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using custom implementations.")

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator for Indian stock market analysis.
    Supports both TA-Lib implementations and custom calculations.
    """
    
    def __init__(self):
        self.talib_available = TALIB_AVAILABLE
        
    def sma(self, data: pd.Series, window: int = 20) -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            data: Price series (typically close prices)
            window: Period for moving average (default: 20)
            
        Returns:
            pd.Series: Simple moving average
        """
        if self.talib_available:
            return pd.Series(talib.SMA(data.values, timeperiod=window), index=data.index)
        else:
            return data.rolling(window=window).mean()
    
    def ema(self, data: pd.Series, window: int = 20) -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            data: Price series
            window: Period for EMA (default: 20)
            
        Returns:
            pd.Series: Exponential moving average
        """
        if self.talib_available:
            return pd.Series(talib.EMA(data.values, timeperiod=window), index=data.index)
        else:
            return data.ewm(span=window, adjust=False).mean()
    
    def rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            data: Price series
            window: Period for RSI calculation (default: 14)
            
        Returns:
            pd.Series: RSI values
        """
        if self.talib_available:
            return pd.Series(talib.RSI(data.values, timeperiod=window), index=data.index)
        else:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            pd.DataFrame: MACD line, signal line, and histogram
        """
        if self.talib_available:
            macd_line, signal_line, histogram = talib.MACD(data.values, fastperiod=fast, 
                                                          slowperiod=slow, signalperiod=signal)
            return pd.DataFrame({
                'MACD': macd_line,
                'Signal': signal_line,
                'Histogram': histogram
            }, index=data.index)
        else:
            ema_fast = self.ema(data, fast)
            ema_slow = self.ema(data, slow)
            macd_line = ema_fast - ema_slow
            signal_line = self.ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            return pd.DataFrame({
                'MACD': macd_line,
                'Signal': signal_line,
                'Histogram': histogram
            })
    
    def bollinger_bands(self, data: pd.Series, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands
        
        Args:
            data: Price series
            window: Period for moving average (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            
        Returns:
            pd.DataFrame: Upper band, middle band (SMA), lower band
        """
        if self.talib_available:
            upper, middle, lower = talib.BBANDS(data.values, timeperiod=window, 
                                               nbdevup=std_dev, nbdevdn=std_dev)
            return pd.DataFrame({
                'Upper': upper,
                'Middle': middle,
                'Lower': lower
            }, index=data.index)
        else:
            sma = self.sma(data, window)
            std = data.rolling(window=window).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return pd.DataFrame({
                'Upper': upper,
                'Middle': sma,
                'Lower': lower
            })
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            pd.DataFrame: %K and %D values
        """
        if self.talib_available:
            k_percent, d_percent = talib.STOCH(high.values, low.values, close.values,
                                             fastk_period=k_period, slowk_period=d_period,
                                             slowd_period=d_period)
            return pd.DataFrame({
                'K': k_percent,
                'D': d_percent
            }, index=close.index)
        else:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return pd.DataFrame({
                'K': k_percent,
                'D': d_percent
            })
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for ATR (default: 14)
            
        Returns:
            pd.Series: ATR values
        """
        if self.talib_available:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=window),
                           index=close.index)
        else:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return true_range.rolling(window=window).mean()
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                  window: int = 14) -> pd.Series:
        """
        Williams %R
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for calculation (default: 14)
            
        Returns:
            pd.Series: Williams %R values
        """
        if self.talib_available:
            return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=window),
                           index=close.index)
        else:
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average Directional Index
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for ADX (default: 14)
            
        Returns:
            pd.Series: ADX values
        """
        if self.talib_available:
            return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=window),
                           index=close.index)
        else:
            # Custom ADX implementation
            tr = self.atr(high, low, close, 1)
            
            dm_plus = high.diff()
            dm_minus = -low.diff()
            
            dm_plus[dm_plus < 0] = 0
            dm_minus[dm_minus < 0] = 0
            
            dm_plus[(dm_plus - dm_minus) <= 0] = 0
            dm_minus[(dm_minus - dm_plus) <= 0] = 0
            
            di_plus = 100 * (dm_plus.rolling(window=window).mean() / tr.rolling(window=window).mean())
            di_minus = 100 * (dm_minus.rolling(window=window).mean() / tr.rolling(window=window).mean())
            
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=window).mean()
            
            return adx
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume
        
        Args:
            close: Close prices
            volume: Volume data
            
        Returns:
            pd.Series: OBV values
        """
        if self.talib_available:
            return pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        else:
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            return obv
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            
        Returns:
            pd.Series: VWAP values
        """
        typical_price = (high + low + close) / 3
        cumulative_volume_price = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        vwap = cumulative_volume_price / cumulative_volume
        return vwap
    
    def fibonacci_retracement(self, high: float, low: float) -> dict:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            high: High price level
            low: Low price level
            
        Returns:
            dict: Fibonacci retracement levels
        """
        diff = high - low
        levels = {
            'Level_0': high,
            'Level_23.6': high - 0.236 * diff,
            'Level_38.2': high - 0.382 * diff,
            'Level_50': high - 0.5 * diff,
            'Level_61.8': high - 0.618 * diff,
            'Level_100': low
        }
        return levels
    
    def pivot_points(self, high: float, low: float, close: float) -> dict:
        """
        Calculate pivot points and support/resistance levels
        
        Args:
            high: Previous day high
            low: Previous day low
            close: Previous day close
            
        Returns:
            dict: Pivot points and levels
        """
        pivot = (high + low + close) / 3
        
        levels = {
            'Pivot': pivot,
            'R1': 2 * pivot - low,
            'R2': pivot + (high - low),
            'R3': high + 2 * (pivot - low),
            'S1': 2 * pivot - high,
            'S2': pivot - (high - low),
            'S3': low - 2 * (high - pivot)
        }
        return levels
    
    def ichimoku_cloud(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """
        Ichimoku Cloud components
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            pd.DataFrame: Ichimoku components
        """
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kiju-sen (Base Line): (26-period high + 26-period low)/2
        kiju_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kiju_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        chikou_span = close.shift(-26)
        
        return pd.DataFrame({
            'Tenkan_Sen': tenkan_sen,
            'Kiju_Sen': kiju_sen,
            'Senkou_Span_A': senkou_span_a,
            'Senkou_Span_B': senkou_span_b,
            'Chikou_Span': chikou_span
        })
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for given OHLCV data
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with all technical indicators
        """
        result = data.copy()
        
        # Moving averages
        result['SMA_20'] = self.sma(data['Close'], 20)
        result['SMA_50'] = self.sma(data['Close'], 50)
        result['EMA_12'] = self.ema(data['Close'], 12)
        result['EMA_26'] = self.ema(data['Close'], 26)
        
        # Momentum indicators
        result['RSI'] = self.rsi(data['Close'])
        
        # MACD
        macd_data = self.macd(data['Close'])
        result['MACD'] = macd_data['MACD']
        result['MACD_Signal'] = macd_data['Signal']
        result['MACD_Histogram'] = macd_data['Histogram']
        
        # Bollinger Bands
        bb_data = self.bollinger_bands(data['Close'])
        result['BB_Upper'] = bb_data['Upper']
        result['BB_Middle'] = bb_data['Middle']
        result['BB_Lower'] = bb_data['Lower']
        
        # Stochastic
        stoch_data = self.stochastic(data['High'], data['Low'], data['Close'])
        result['Stoch_K'] = stoch_data['K']
        result['Stoch_D'] = stoch_data['D']
        
        # Other indicators
        result['ATR'] = self.atr(data['High'], data['Low'], data['Close'])
        result['Williams_R'] = self.williams_r(data['High'], data['Low'], data['Close'])
        result['ADX'] = self.adx(data['High'], data['Low'], data['Close'])
        
        if 'Volume' in data.columns:
            result['OBV'] = self.obv(data['Close'], data['Volume'])
            result['VWAP'] = self.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
        
        return result