"""
Unit tests for analysis modules
Market Research System v1.0 (2022)
Focus: Indian Stock Market Analysis
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from analysis.technical_indicators import TechnicalIndicators
from analysis.market_analyzer import MarketAnalyzer
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.trend_analyzer import TrendAnalyzer
from analysis.volatility_analyzer import VolatilityAnalyzer
from analysis.performance_analyzer import PerformanceAnalyzer


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for Technical Indicators module"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample Indian stock data (Reliance Industries)
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic stock price data
        base_price = 2500  # Typical RIL price in 2022
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.01, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 0.99) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        self.sample_data.set_index('Date', inplace=True)
        self.tech_indicators = TechnicalIndicators()
    
    def test_simple_moving_average(self):
        """Test Simple Moving Average calculation"""
        sma_20 = self.tech_indicators.calculate_sma(self.sample_data['Close'], 20)
        
        # Test that SMA is calculated correctly
        self.assertEqual(len(sma_20), len(self.sample_data))
        self.assertTrue(pd.isna(sma_20.iloc[0]))  # First value should be NaN
        self.assertFalse(pd.isna(sma_20.iloc[19]))  # 20th value should not be NaN
        
        # Test manual calculation for verification
        manual_sma = self.sample_data['Close'].iloc[0:20].mean()
        self.assertAlmostEqual(sma_20.iloc[19], manual_sma, places=2)
    
    def test_exponential_moving_average(self):
        """Test Exponential Moving Average calculation"""
        ema_12 = self.tech_indicators.calculate_ema(self.sample_data['Close'], 12)
        
        # Test that EMA is calculated
        self.assertEqual(len(ema_12), len(self.sample_data))
        self.assertFalse(pd.isna(ema_12.iloc[-1]))  # Last value should not be NaN
        
        # EMA should be more responsive than SMA
        sma_12 = self.tech_indicators.calculate_sma(self.sample_data['Close'], 12)
        self.assertNotEqual(ema_12.iloc[-1], sma_12.iloc[-1])
    
    def test_rsi_calculation(self):
        """Test RSI (Relative Strength Index) calculation"""
        rsi = self.tech_indicators.calculate_rsi(self.sample_data['Close'], 14)
        
        # Test RSI bounds (should be between 0 and 100)
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
        
        # Test length
        self.assertEqual(len(rsi), len(self.sample_data))
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = self.tech_indicators.calculate_macd(
            self.sample_data['Close'], 12, 26, 9
        )
        
        # Test that all components are calculated
        self.assertEqual(len(macd_line), len(self.sample_data))
        self.assertEqual(len(signal_line), len(self.sample_data))
        self.assertEqual(len(histogram), len(self.sample_data))
        
        # Histogram should be difference between MACD and Signal
        calculated_histogram = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, calculated_histogram, check_names=False)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        upper_band, middle_band, lower_band = self.tech_indicators.calculate_bollinger_bands(
            self.sample_data['Close'], 20, 2
        )
        
        # Test that bands are in correct order
        valid_data = pd.DataFrame({
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }).dropna()
        
        self.assertTrue((valid_data['upper'] >= valid_data['middle']).all())
        self.assertTrue((valid_data['middle'] >= valid_data['lower']).all())
        
        # Middle band should be SMA
        sma_20 = self.tech_indicators.calculate_sma(self.sample_data['Close'], 20)
        pd.testing.assert_series_equal(middle_band, sma_20, check_names=False)


class TestMarketAnalyzer(unittest.TestCase):
    """Test cases for Market Analyzer module"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample NSE data for multiple stocks
        self.stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS']
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        
        self.market_data = {}
        np.random.seed(42)
        
        for stock in self.stocks:
            base_price = np.random.uniform(1000, 3000)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            self.market_data[stock] = pd.DataFrame({
                'Date': dates,
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }).set_index('Date')
        
        self.analyzer = MarketAnalyzer()
    
    def test_calculate_returns(self):
        """Test returns calculation"""
        returns = self.analyzer.calculate_returns(self.market_data['RELIANCE.NS']['Close'])
        
        # Test length (should be one less than price data)
        self.assertEqual(len(returns), len(self.market_data['RELIANCE.NS']) - 1)
        
        # Test that returns are calculated correctly
        manual_return = (self.market_data['RELIANCE.NS']['Close'].iloc[1] / 
                        self.market_data['RELIANCE.NS']['Close'].iloc[0]) - 1
        self.assertAlmostEqual(returns.iloc[0], manual_return, places=6)
    
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        volatility = self.analyzer.calculate_volatility(
            self.market_data['RELIANCE.NS']['Close'], window=30
        )
        
        # Volatility should be positive
        valid_vol = volatility.dropna()
        self.assertTrue((valid_vol >= 0).all())
        
        # Test annualized volatility
        ann_vol = self.analyzer.calculate_volatility(
            self.market_data['RELIANCE.NS']['Close'], window=30, annualized=True
        )
        self.assertTrue(ann_vol.mean() > volatility.mean())
    
    def test_sector_analysis(self):
        """Test sector analysis functionality"""
        # Create sector mapping
        sector_mapping = {
            'RELIANCE.NS': 'Energy',
            'TCS.NS': 'IT',
            'HDFCBANK.NS': 'Banking',
            'INFY.NS': 'IT',
            'HINDUNILVR.NS': 'FMCG'
        }
        
        sector_performance = self.analyzer.analyze_sector_performance(
            self.market_data, sector_mapping
        )
        
        # Should return performance for each sector
        expected_sectors = set(sector_mapping.values())
        self.assertEqual(set(sector_performance.keys()), expected_sectors)
        
        # Each sector should have performance metrics
        for sector, metrics in sector_performance.items():
            self.assertIn('total_return', metrics)
            self.assertIn('volatility', metrics)
            self.assertIn('sharpe_ratio', metrics)


class TestCorrelationAnalyzer(unittest.TestCase):
    """Test cases for Correlation Analyzer module"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        np.random.seed(42)
        
        # Create correlated stock data
        base_returns = np.random.normal(0.001, 0.02, len(dates))
        
        self.stock_data = pd.DataFrame({
            'RELIANCE.NS': np.cumsum(base_returns + np.random.normal(0, 0.01, len(dates))),
            'TCS.NS': np.cumsum(base_returns * 0.8 + np.random.normal(0, 0.015, len(dates))),
            'HDFCBANK.NS': np.cumsum(base_returns * 0.6 + np.random.normal(0, 0.012, len(dates))),
            'NIFTY50': np.cumsum(base_returns)
        }, index=dates)
        
        self.corr_analyzer = CorrelationAnalyzer()
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation"""
        corr_matrix = self.corr_analyzer.calculate_correlation_matrix(self.stock_data)
        
        # Test that matrix is square
        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
        
        # Test diagonal elements are 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)
        
        # Test symmetry
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.values.T)
        
        # Test correlation bounds
        self.assertTrue((corr_matrix >= -1).all().all())
        self.assertTrue((corr_matrix <= 1).all().all())
    
    def test_rolling_correlation(self):
        """Test rolling correlation calculation"""
        rolling_corr = self.corr_analyzer.calculate_rolling_correlation(
            self.stock_data['RELIANCE.NS'], self.stock_data['NIFTY50'], window=30
        )
        
        # Test length
        self.assertEqual(len(rolling_corr), len(self.stock_data))
        
        # Test correlation bounds
        valid_corr = rolling_corr.dropna()
        self.assertTrue((valid_corr >= -1).all())
        self.assertTrue((valid_corr <= 1).all())
    
    def test_beta_calculation(self):
        """Test beta calculation against market index"""
        beta = self.corr_analyzer.calculate_beta(
            self.stock_data['RELIANCE.NS'], self.stock_data['NIFTY50']
        )
        
        # Beta should be a float
        self.assertIsInstance(beta, float)
        
        # For our test data, beta should be reasonable
        self.assertTrue(-5 < beta < 5)


class TestTrendAnalyzer(unittest.TestCase):
    """Test cases for Trend Analyzer module"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        
        # Create trending data
        trend = np.linspace(2000, 2500, len(dates))  # Upward trend
        noise = np.random.normal(0, 50, len(dates))
        
        self.trending_data = pd.Series(trend + noise, index=dates)
        self.trend_analyzer = TrendAnalyzer()
    
    def test_identify_trend_direction(self):
        """Test trend direction identification"""
        trend_direction = self.trend_analyzer.identify_trend_direction(
            self.trending_data, window=50
        )
        
        # Should identify upward trend for our test data
        self.assertIn(trend_direction, ['upward', 'downward', 'sideways'])
        
        # For our upward trending data, should be 'upward'
        self.assertEqual(trend_direction, 'upward')
    
    def test_calculate_trend_strength(self):
        """Test trend strength calculation"""
        trend_strength = self.trend_analyzer.calculate_trend_strength(
            self.trending_data, window=50
        )
        
        # Trend strength should be between 0 and 1
        self.assertTrue(0 <= trend_strength <= 1)
        
        # For strong trend, should be closer to 1
        self.assertGreater(trend_strength, 0.5)
    
    def test_support_resistance_levels(self):
        """Test support and resistance level detection"""
        support_levels, resistance_levels = self.trend_analyzer.find_support_resistance_levels(
            self.trending_data, window=20
        )
        
        # Should return lists of levels
        self.assertIsInstance(support_levels, list)
        self.assertIsInstance(resistance_levels, list)
        
        # Support levels should be lower than resistance levels on average
        if support_levels and resistance_levels:
            avg_support = np.mean(support_levels)
            avg_resistance = np.mean(resistance_levels)
            self.assertLess(avg_support, avg_resistance)


class TestVolatilityAnalyzer(unittest.TestCase):
    """Test cases for Volatility Analyzer module"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        np.random.seed(42)
        
        # Create data with varying volatility
        returns = np.random.normal(0.001, 0.02, len(dates))
        self.price_data = pd.Series(np.cumsum(returns) + 2000, index=dates)
        
        self.vol_analyzer = VolatilityAnalyzer()
    
    def test_historical_volatility(self):
        """Test historical volatility calculation"""
        hist_vol = self.vol_analyzer.calculate_historical_volatility(
            self.price_data, window=30
        )
        
        # Volatility should be positive
        valid_vol = hist_vol.dropna()
        self.assertTrue((valid_vol >= 0).all())
        
        # Test annualized version
        ann_vol = self.vol_analyzer.calculate_historical_volatility(
            self.price_data, window=30, annualized=True
        )
        self.assertTrue(ann_vol.mean() > hist_vol.mean())
    
    def test_garch_volatility(self):
        """Test GARCH volatility model (simplified)"""
        returns = self.price_data.pct_change().dropna()
        garch_vol = self.vol_analyzer.calculate_garch_volatility(returns)
        
        # Should return volatility series
        self.assertIsInstance(garch_vol, pd.Series)
        self.assertTrue((garch_vol >= 0).all())
    
    def test_volatility_clustering(self):
        """Test volatility clustering detection"""
        returns = self.price_data.pct_change().dropna()
        clustering_score = self.vol_analyzer.detect_volatility_clustering(returns)
        
        # Should return a score between 0 and 1
        self.assertTrue(0 <= clustering_score <= 1)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test cases for Performance Analyzer module"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        np.random.seed(42)
        
        # Create portfolio returns data
        returns = np.random.normal(0.0008, 0.015, len(dates))  # Slightly positive returns
        self.returns_data = pd.Series(returns, index=dates)
        
        # Benchmark returns (NIFTY 50)
        benchmark_returns = np.random.normal(0.0006, 0.012, len(dates))
        self.benchmark_returns = pd.Series(benchmark_returns, index=dates)
        
        self.perf_analyzer = PerformanceAnalyzer()
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        sharpe_ratio = self.perf_analyzer.calculate_sharpe_ratio(
            self.returns_data, risk_free_rate=0.06  # 6% annual risk-free rate
        )
        
        # Sharpe ratio should be a float
        self.assertIsInstance(sharpe_ratio, float)
        
        # For positive returns, Sharpe ratio should be positive
        self.assertGreater(sharpe_ratio, 0)
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        sortino_ratio = self.perf_analyzer.calculate_sortino_ratio(
            self.returns_data, risk_free_rate=0.06
        )
        
        # Sortino ratio should be a float
        self.assertIsInstance(sortino_ratio, float)
        
        # Should be different from Sharpe ratio
        sharpe_ratio = self.perf_analyzer.calculate_sharpe_ratio(
            self.returns_data, risk_free_rate=0.06
        )
        self.assertNotEqual(sortino_ratio, sharpe_ratio)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        cumulative_returns = (1 + self.returns_data).cumprod()
        max_drawdown = self.perf_analyzer.calculate_max_drawdown(cumulative_returns)
        
        # Max drawdown should be negative or zero
        self.assertLessEqual(max_drawdown, 0)
        
        # Should be between -1 and 0
        self.assertGreaterEqual(max_drawdown, -1)
    
    def test_calculate_alpha_beta(self):
        """Test alpha and beta calculation"""
        alpha, beta = self.perf_analyzer.calculate_alpha_beta(
            self.returns_data, self.benchmark_returns, risk_free_rate=0.06
        )
        
        # Alpha and beta should be floats
        self.assertIsInstance(alpha, float)
        self.assertIsInstance(beta, float)
        
        # Beta should be reasonable
        self.assertTrue(-5 < beta < 5)
    
    def test_information_ratio(self):
        """Test information ratio calculation"""
        info_ratio = self.perf_analyzer.calculate_information_ratio(
            self.returns_data, self.benchmark_returns
        )
        
        # Information ratio should be a float
        self.assertIsInstance(info_ratio, float)
        
        # Should be finite
        self.assertTrue(np.isfinite(info_ratio))
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        var_95 = self.perf_analyzer.calculate_var(self.returns_data, confidence_level=0.95)
        var_99 = self.perf_analyzer.calculate_var(self.returns_data, confidence_level=0.99)
        
        # VaR should be negative (representing potential losses)
        self.assertLess(var_95, 0)
        self.assertLess(var_99, 0)
        
        # 99% VaR should be more negative than 95% VaR
        self.assertLess(var_99, var_95)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTechnicalIndicators,
        TestMarketAnalyzer,
        TestCorrelationAnalyzer,
        TestTrendAnalyzer,
        TestVolatilityAnalyzer,
        TestPerformanceAnalyzer
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY - Analysis Modules")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, error in result.failures:
            print(f"- {test}: {error.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, error in result.errors:
            print(f"- {test}: {error.split('\\n')[-2]}")