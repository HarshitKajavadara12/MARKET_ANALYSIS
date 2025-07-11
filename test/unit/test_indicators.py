#!/usr/bin/env python3
"""
Unit tests for technical indicators modules
Market Research System v1.0 (2022)
Focus: Indian Stock Market
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from analysis.technical_indicators import TechnicalIndicators
from analysis.exceptions import IndicatorCalculationError, InsufficientDataError


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for TechnicalIndicators class"""
    
    def setUp(self):
        """Setup test fixtures with Indian stock data"""
        self.indicator_calc = TechnicalIndicators()
        
        # Create sample Indian stock data (RELIANCE.NS)
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2022-01-01', periods=100, freq='D')
        
        # Generate realistic price data with some trend
        base_price = 2500  # Typical Reliance price in 2022
        price_changes = np.random.normal(0, 2, 100).cumsum()
        closes = base_price + price_changes
        
        # Ensure realistic OHLC relationships
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': closes + np.random.uniform(-5, 5, 100),
            'High': closes + np.abs(np.random.uniform(0, 10, 100)),
            'Low': closes - np.abs(np.random.uniform(0, 10, 100)),
            'Close': closes,
            'Volume': np.random.randint(1000000, 10000000, 100)  # Typical Indian stock volumes
        })
        
        # Ensure High >= Close >= Low and High >= Open >= Low
        for i in range(len(self.sample_data)):
            high = max(self.sample_data.iloc[i]['Open'], self.sample_data.iloc[i]['Close'], 
                      self.sample_data.iloc[i]['High'])
            low = min(self.sample_data.iloc[i]['Open'], self.sample_data.iloc[i]['Close'], 
                     self.sample_data.iloc[i]['Low'])
            self.sample_data.iloc[i, self.sample_data.columns.get_loc('High')] = high
            self.sample_data.iloc[i, self.sample_data.columns.get_loc('Low')] = low

    def test_simple_moving_average(self):
        """Test Simple Moving Average calculation"""
        # Test different periods
        sma_5 = self.indicator_calc.sma(self.sample_data, period=5)
        sma_20 = self.indicator_calc.sma(self.sample_data, period=20)
        
        # SMA should return same length as input
        self.assertEqual(len(sma_5), len(self.sample_data))
        self.assertEqual(len(sma_20), len(self.sample_data))
        
        # First (period-1) values should be NaN
        self.assertTrue(pd.isna(sma_5.iloc[:4]).all())
        self.assertTrue(pd.isna(sma_20.iloc[:19]).all())
        
        # SMA values should be finite numbers
        self.assertTrue(np.isfinite(sma_5.iloc[5:]).all())
        self.assertTrue(np.isfinite(sma_20.iloc[20:]).all())
        
        # Manual verification for first valid SMA value
        expected_sma_5 = self.sample_data['Close'].iloc[:5].mean()
        self.assertAlmostEqual(sma_5.iloc[4], expected_sma_5, places=2)

    def test_exponential_moving_average(self):
        """Test Exponential Moving Average calculation"""
        ema_12 = self.indicator_calc.ema(self.sample_data, period=12)
        ema_26 = self.indicator_calc.ema(self.sample_data, period=26)
        
        # EMA should return same length as input
        self.assertEqual(len(ema_12), len(self.sample_data))
        
        # EMA should have fewer NaN values compared to SMA
        nan_count_ema = pd.isna(ema_12).sum()
        self.assertLessEqual(nan_count_ema, 11)
        
        # EMA should react faster to price changes than SMA
        sma_12 = self.indicator_calc.sma(self.sample_data, period=12)
        
        # At least some EMA values should be different from SMA
        differences = np.abs(ema_12.iloc[12:] - sma_12.iloc[12:])
        self.assertGreater(differences.sum(), 0)

    def test_relative_strength_index(self):
        """Test RSI calculation"""
        rsi = self.indicator_calc.rsi(self.sample_data, period=14)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
        
        # RSI should have NaN for first (period) values
        self.assertTrue(pd.isna(rsi.iloc[:14]).all())
        
        # RSI should be finite numbers
        self.assertTrue(np.isfinite(valid_rsi).all())

    def test_macd(self):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = self.indicator_calc.macd(
            self.sample_data, fast=12, slow=26, signal=9
        )
        
        # All components should have same length
        self.assertEqual(len(macd_line), len(self.sample_data))
        self.assertEqual(len(signal_line), len(self.sample_data))
        self.assertEqual(len(histogram), len(self.sample_data))
        
        # MACD histogram should equal MACD line minus signal line
        valid_indices = ~(pd.isna(macd_line) | pd.isna(signal_line))
        calculated_histogram = macd_line[valid_indices] - signal_line[valid_indices]
        actual_histogram = histogram[valid_indices]
        
        np.testing.assert_array_almost_equal(
            calculated_histogram.values, actual_histogram.values, decimal=6
        )

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = self.indicator_calc.bollinger_bands(
            self.sample_data, period=20, std_dev=2
        )
        
        # All bands should have same length
        self.assertEqual(len(upper), len(self.sample_data))
        self.assertEqual(len(middle), len(self.sample_data))
        self.assertEqual(len(lower), len(self.sample_data))
        
        # Upper band should be >= Middle band >= Lower band
        valid_indices = ~(pd.isna(upper) | pd.isna(middle) | pd.isna(lower))
        self.assertTrue((upper[valid_indices] >= middle[valid_indices]).all())
        self.assertTrue((middle[valid_indices] >= lower[valid_indices]).all())
        
        # Middle band should equal SMA
        sma_20 = self.indicator_calc.sma(self.sample_data, period=20)
        np.testing.assert_array_almost_equal(
            middle[valid_indices].values, sma_20[valid_indices].values, decimal=6
        )

    def test_stochastic_oscillator(self):
        """Test Stochastic Oscillator calculation"""
        k_percent, d_percent = self.indicator_calc.stochastic(
            self.sample_data, k_period=14, d_period=3
        )
        
        # Values should be between 0 and 100
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()
        
        self.assertTrue((valid_k >= 0).all())
        self.assertTrue((valid_k <= 100).all())
        self.assertTrue((valid_d >= 0).all())
        self.assertTrue((valid_d <= 100).all())

    def test_williams_r(self):
        """Test Williams %R calculation"""
        williams_r = self.indicator_calc.williams_r(self.sample_data, period=14)
        
        # Williams %R should be between -100 and 0
        valid_wr = williams_r.dropna()
        self.assertTrue((valid_wr >= -100).all())
        self.assertTrue((valid_wr <= 0).all())

    def test_commodity_channel_index(self):
        """Test CCI calculation"""
        cci = self.indicator_calc.cci(self.sample_data, period=20)
        
        # CCI should be calculated without errors
        self.assertEqual(len(cci), len(self.sample_data))
        
        # CCI typically ranges from -200 to +200, but can exceed
        valid_cci = cci.dropna()
        self.assertTrue(np.isfinite(valid_cci).all())

    def test_average_true_range(self):
        """Test ATR calculation"""
        atr = self.indicator_calc.atr(self.sample_data, period=14)
        
        # ATR should be positive
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0).all())
        
        # ATR should have reasonable values (not too large compared to price)
        typical_price = self.sample_data['Close'].mean()
        self.assertTrue((valid_atr < typical_price * 0.5).all())  # ATR shouldn't exceed 50% of price

    def test_on_balance_volume(self):
        """Test OBV calculation"""
        obv = self.indicator_calc.obv(self.sample_data)
        
        # OBV should have same length as input
        self.assertEqual(len(obv), len(self.sample_data))
        
        # OBV should be cumulative (generally increasing or decreasing)
        self.assertTrue(np.isfinite(obv).all())

    def test_money_flow_index(self):
        """Test MFI calculation"""
        mfi = self.indicator_calc.mfi(self.sample_data, period=14)
        
        # MFI should be between 0 and 100
        valid_mfi = mfi.dropna()
        self.assertTrue((valid_mfi >= 0).all())
        self.assertTrue((valid_mfi <= 100).all())

    def test_parabolic_sar(self):
        """Test Parabolic SAR calculation"""
        sar = self.indicator_calc.parabolic_sar(
            self.sample_data, acceleration=0.02, maximum=0.2
        )
        
        # SAR values should be reasonable (close to price range)
        valid_sar = sar.dropna()
        price_min = self.sample_data['Low'].min()
        price_max = self.sample_data['High'].max()
        
        self.assertTrue((valid_sar >= price_min * 0.8).all())
        self.assertTrue((valid_sar <= price_max * 1.2).all())

    def test_ichimoku_cloud(self):
        """Test Ichimoku Cloud calculation"""
        tenkan, kijun, senkou_a, senkou_b, chikou = self.indicator_calc.ichimoku_cloud(
            self.sample_data
        )
        
        # All components should have same length
        self.assertEqual(len(tenkan), len(self.sample_data))
        self.assertEqual(len(kijun), len(self.sample_data))
        self.assertEqual(len(senkou_a), len(self.sample_data))
        self.assertEqual(len(senkou_b), len(self.sample_data))
        self.assertEqual(len(chikou), len(self.sample_data))
        
        # Values should be finite where calculated
        self.assertTrue(np.isfinite(tenkan.dropna()).all())
        self.assertTrue(np.isfinite(kijun.dropna()).all())

    def test_aroon_oscillator(self):
        """Test Aroon Oscillator calculation"""
        aroon_up, aroon_down, aroon_osc = self.indicator_calc.aroon(
            self.sample_data, period=25
        )
        
        # Aroon values should be between 0 and 100
        valid_up = aroon_up.dropna()
        valid_down = aroon_down.dropna()
        
        self.assertTrue((valid_up >= 0).all())
        self.assertTrue((valid_up <= 100).all())
        self.assertTrue((valid_down >= 0).all())
        self.assertTrue((valid_down <= 100).all())
        
        # Aroon oscillator should be between -100 and 100
        valid_osc = aroon_osc.dropna()
        self.assertTrue((valid_osc >= -100).all())
        self.assertTrue((valid_osc <= 100).all())

    def test_pivot_points(self):
        """Test Pivot Points calculation for Indian markets"""
        pivots = self.indicator_calc.pivot_points(self.sample_data)
        
        # Should return dictionary with pivot levels
        self.assertIsInstance(pivots, pd.DataFrame)
        
        # Should have pivot point columns
        expected_cols = ['PP', 'R1', 'R2', 'R3', '
                         

                         not compllted