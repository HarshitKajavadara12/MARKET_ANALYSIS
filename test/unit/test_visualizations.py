"""
Unit tests for visualization modules
Market Research System v1.0 (2022)
Focus: Indian Stock Market Visualization Testing
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
from datetime import datetime, timedelta
import sys
import os
import tempfile

# Set matplotlib backend for testing
matplotlib.use('Agg')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from reporting.visualization import MarketVisualizer
from reporting.chart_utils import ChartUtils


class TestMarketVisualizer(unittest.TestCase):
    """Test cases for Market Visualizer module"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample Indian stock data
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic Nifty 50 data
        base_price = 17000  # Typical Nifty level in 2022
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.nifty_data = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.995, 1.005) for p in prices],
            'High': [p * np.random.uniform(1.005, 1.02) for p in prices],
            'Low': [p * np.random.uniform(0.98, 0.995) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100000000, 500000000, len(dates))
        }).set_index('Date')
        
        # Create sample sector data
        self.sector_data = {
            'Banking': np.random.normal(0.08, 0.15),
            'IT': np.random.normal(0.12, 0.18),
            'Pharma': np.random.normal(0.06, 0.12),
            'FMCG': np.random.normal(0.05, 0.08),
            'Auto': np.random.normal(-0.02, 0.2),
            'Energy': np.random.normal(0.03, 0.25),
            'Metals': np.random.normal(0.15, 0.3),
            'Realty': np.random.normal(-0.05, 0.35)
        }
        
        # Create correlation data
        stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
        correlation_matrix = np.random.rand(len(stocks), len(stocks))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        self.correlation_data = pd.DataFrame(
            correlation_matrix, 
            index=stocks, 
            columns=stocks
        )
        
        self.visualizer = MarketVisualizer()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests"""
        plt.close('all')
    
    def test_create_price_chart(self):
        """Test price chart creation"""
        fig, ax = self.visualizer.create_price_chart(
            self.nifty_data,
            title="NIFTY 50 Price Chart",
            show_volume=True
        )
        
        # Test that figure and axis are created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Test that chart has data
        lines = ax.get_lines()
        self.assertGreater(len(lines), 0)
        
        # Test chart title
        self.assertEqual(ax.get_title(), "NIFTY 50 Price Chart")
        
        # Test axis labels
        self.assertIsNotNone(ax.get_xlabel())
        self.assertIsNotNone(ax.get_ylabel())
        
        # Test that volume subplot is created when requested
        if hasattr(fig, 'axes'):
            self.assertGreaterEqual(len(fig.axes), 1)
    
    def test_create_candlestick_chart(self):
        """Test candlestick chart creation"""
        fig, ax = self.visualizer.create_candlestick_chart(
            self.nifty_data.head(50),  # Use smaller dataset for testing
            title="NIFTY 50 Candlestick Chart"
        )
        
        # Test that figure and axis are created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Test chart title
        self.assertEqual(ax.get_title(), "NIFTY 50 Candlestick Chart")
        
        # Test that rectangles (candlesticks) are drawn
        rectangles = [patch for patch in ax.patches if hasattr(patch, 'get_width')]
        self.assertGreater(len(rectangles), 0)
    
    def test_create_technical_indicators_chart(self):
        """Test technical indicators chart creation"""
        # Add some technical indicators to test data
        close_prices = self.nifty_data['Close']
        
        # Simple moving averages
        sma_20 = close_prices.rolling(window=20).mean()
        sma_50 = close_prices.rolling(window=50).mean()
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        indicators_data = {
            'SMA_20': sma_20,
            'SMA_50': sma_50,
            'RSI': rsi
        }
        
        fig, axes = self.visualizer.create_technical_indicators_chart(
            self.nifty_data, 
            indicators_data,
            title="NIFTY 50 Technical Analysis"
        )
        
        # Test that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Test that multiple subplots are created
        self.assertGreaterEqual(len(fig.axes), 2)  # Price + RSI subplot
        
        # Test that RSI is bounded between 0 and 100
        rsi_ax = fig.axes[1]  # Second subplot should be RSI
        rsi_lines = rsi_ax.get_lines()
        if rsi_lines:
            rsi_data = rsi_lines[0].get_ydata()
            valid_rsi = rsi_data[~np.isnan(rsi_data)]
            if len(valid_rsi) > 0:
                self.assertTrue(np.all(valid_rsi >= 0))
                self.assertTrue(np.all(valid_rsi <= 100))
    
    def test_create_sector_performance_chart(self):
        """Test sector performance chart creation"""
        fig, ax = self.visualizer.create_sector_performance_chart(
            self.sector_data,
            title="Sector Performance (YTD 2022)"
        )
        
        # Test that figure and axis are created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Test chart title
        self.assertEqual(ax.get_title(), "Sector Performance (YTD 2022)")
        
        # Test that bars are

    def test_file_size_operations(self):
        """Test file size utilities"""
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write('Test data for Indian stock market research' * 100)
        
        # Test file size calculation
        file_size = self.file_util.get_file_size(self.test_file)
        self.assertGreater(file_size, 0)
        
        # Test human readable size
        readable_size = self.file_util.get_human_readable_size(file_size)
        self.assertIn('bytes', readable_size.lower())
    
    def test_backup_and_archive(self):
        """Test file backup and archiving"""
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write('NSE market data backup test')
        
        # Test backup
        backup_file = self.file_util.create_backup(self.test_file)
        self.assertTrue(os.path.exists(backup_file))
        
        # Test archive creation
        archive_dir = os.path.join(self.temp_dir, 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        
        archive_file = self.file_util.create_archive(
            [self.test_file], 
            os.path.join(archive_dir, 'market_data.zip')
        )
        self.assertTrue(os.path.exists(archive_file))
    
    def test_file_validation(self):
        """Test file validation utilities"""
        # Test valid file extensions for market data
        valid_extensions = ['.csv', '.json', '.xlsx']
        
        for ext in valid_extensions:
            test_file = f"nse_data{ext}"
            self.assertTrue(self.file_util.is_valid_data_file(test_file))
        
        # Test invalid extensions
        invalid_file = "data.txt"
        self.assertFalse(self.file_util.is_valid_data_file(invalid_file))


class TestMathUtility(unittest.TestCase):
    """Test cases for mathematical utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.math_util = MathUtility()
        
        # Sample Indian stock prices (in INR)
        self.stock_prices = [2650.50, 2660.25, 2645.75, 2670.00, 2655.50]
        self.volumes = [1000000, 1200000, 950000, 1300000, 1100000]
        
        # Sample portfolio data
        self.portfolio = {
            'RELIANCE.NS': {'quantity': 100, 'price': 2650.50},
            'TCS.NS': {'quantity': 50, 'price': 3420.75},
            'HDFCBANK.NS': {'quantity': 75, 'price': 1580.25}
        }
    
    def test_technical_indicators(self):
        """Test basic technical indicator calculations"""
        prices = np.array(self.stock_prices)
        
        # Test Simple Moving Average
        sma_3 = self.math_util.calculate_sma(prices, window=3)
        expected_sma = np.mean(prices[-3:])
        self.assertAlmostEqual(sma_3[-1], expected_sma, places=2)
        
        # Test Exponential Moving Average
        ema_3 = self.math_util.calculate_ema(prices, window=3)
        self.assertIsInstance(ema_3, np.ndarray)
        self.assertEqual(len(ema_3), len(prices))
        
        # Test RSI calculation
        rsi = self.math_util.calculate_rsi(prices, window=4)
        self.assertTrue(0 <= rsi <= 100)
    
    def test_volatility_calculations(self):
        """Test volatility measures"""
        prices = np.array(self.stock_prices)
        returns = self.math_util.calculate_returns(prices)
        
        # Test standard deviation
        volatility = self.math_util.calculate_volatility(returns)
        self.assertGreater(volatility, 0)
        
        # Test annualized volatility (252 trading days in India)
        annual_vol = self.math_util.calculate_annual_volatility(returns)
        expected_annual = volatility * np.sqrt(252)
        self.assertAlmostEqual(annual_vol, expected_annual, places=4)
    
    def test_return_calculations(self):
        """Test various return calculations"""
        prices = np.array(self.stock_prices)
        
        # Test simple returns
        simple_returns = self.math_util.calculate_simple_returns(prices)
        self.assertEqual(len(simple_returns), len(prices) - 1)
        
        # Test log returns
        log_returns = self.math_util.calculate_log_returns(prices)
        self.assertEqual(len(log_returns), len(prices) - 1)
        
        # Test cumulative returns
        cum_returns = self.math_util.calculate_cumulative_returns(simple_returns)
        self.assertIsInstance(cum_returns, np.ndarray)
    
    def test_portfolio_calculations(self):
        """Test portfolio-related calculations"""
        # Test portfolio value
        total_value = self.math_util.calculate_portfolio_value(self.portfolio)
        
        expected_value = (100 * 2650.50) + (50 * 3420.75) + (75 * 1580.25)
        self.assertAlmostEqual(total_value, expected_value, places=2)
        
        # Test portfolio weights
        weights = self.math_util.calculate_portfolio_weights(self.portfolio)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        
        # Test sector allocation (assuming we have sector data)
        sector_data = {
            'RELIANCE.NS': 'Energy',
            'TCS.NS': 'IT',
            'HDFCBANK.NS': 'Banking'
        }
        
        sector_allocation = self.math_util.calculate_sector_allocation(
            self.portfolio, sector_data
        )
        self.assertIsInstance(sector_allocation, dict)
    
    def test_risk_metrics(self):
        """Test risk calculation metrics"""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        
        # Test Value at Risk (VaR)
        var_95 = self.math_util.calculate_var(returns, confidence=0.95)
        self.assertLess(var_95, 0)  # VaR should be negative
        
        # Test Conditional VaR (Expected Shortfall)
        cvar_95 = self.math_util.calculate_cvar(returns, confidence=0.95)
        self.assertLess(cvar_95, var_95)  # CVaR should be worse than VaR
        
        # Test Maximum Drawdown
        prices = np.array(self.stock_prices)
        max_dd = self.math_util.calculate_max_drawdown(prices)
        self.assertLessEqual(max_dd, 0)  # Drawdown should be negative or zero
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        risk_free_rate = 0.06  # 6% annual (typical Indian govt bond rate in 2022)
        
        sharpe = self.math_util.calculate_sharpe_ratio(
            returns, risk_free_rate=risk_free_rate
        )
        self.assertIsInstance(sharpe, float)
    
    def test_correlation_analysis(self):
        """Test correlation calculations"""
        stock1_returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        stock2_returns = np.array([0.01, -0.02, 0.02, -0.01, 0.02])
        
        correlation = self.math_util.calculate_correlation(stock1_returns, stock2_returns)
        self.assertTrue(-1 <= correlation <= 1)
        
        # Test correlation matrix for multiple stocks
        returns_matrix = np.array([stock1_returns, stock2_returns])
        corr_matrix = self.math_util.calculate_correlation_matrix(returns_matrix)
        
        self.assertEqual(corr_matrix.shape, (2, 2))
        self.assertAlmostEqual(corr_matrix[0, 0], 1.0, places=4)  # Self correlation = 1


class TestStringUtility(unittest.TestCase):
    """Test cases for string utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.string_util = StringUtility()
    
    def test_nse_symbol_validation(self):
        """Test NSE symbol format validation"""
        # Valid NSE symbols
        valid_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS']
        
        for symbol in valid_symbols:
            self.assertTrue(self.string_util.is_valid_nse_symbol(symbol))
        
        # Invalid symbols
        invalid_symbols = ['RELIANCE', 'TCS.BO', 'INVALID.XX', '']
        
        for symbol in invalid_symbols:
            self.assertFalse(self.string_util.is_valid_nse_symbol(symbol))
    
    def test_bse_symbol_validation(self):
        """Test BSE symbol format validation"""
        # Valid BSE symbols
        valid_symbols = ['500325.BO', '532540.BO', '500180.BO']  # Reliance, TCS, HDFC Bank
        
        for symbol in valid_symbols:
            self.assertTrue(self.string_util.is_valid_bse_symbol(symbol))
        
        # Invalid symbols
        invalid_symbols = ['RELIANCE.BO', '500325', 'INVALID.BO']
        
        for symbol in invalid_symbols:
            self.assertFalse(self.string_util.is_valid_bse_symbol(symbol))
    
    def test_clean_company_name(self):
        """Test company name cleaning"""
        test_cases = [
            ('Reliance Industries Limited', 'Reliance Industries'),
            ('Tata Consultancy Services Ltd.', 'Tata Consultancy Services'),
            ('HDFC Bank Ltd', 'HDFC Bank'),
            ('Infosys Limited', 'Infosys')
        ]
        
        for original, expected in test_cases:
            cleaned = self.string_util.clean_company_name(original)
            self.assertEqual(cleaned, expected)
    
    def test_format_indian_currency(self):
        """Test Indian currency formatting"""
        test_cases = [
            (1000, '₹1,000'),
            (100000, '₹1,00,000'),
            (10000000, '₹1,00,00,000'),
            (1000.50, '₹1,000.50')
        ]
        
        for amount, expected in test_cases:
            formatted = self.string_util.format_indian_currency(amount)
            self.assertEqual(formatted, expected)
    
    def test_parse_market_cap(self):
        """Test market cap string parsing"""
        test_cases = [
            ('₹10,000 Cr', 1000000000000),  # 10,000 Crores
            ('₹5,50,000 Cr', 5500000000000),  # 5,50,000 Crores
            ('₹1,000 Cr', 100000000000)  # 1,000 Crores
        ]
        
        for cap_str, expected in test_cases:
            parsed = self.string_util.parse_market_cap(cap_str)
            self.assertEqual(parsed, expected)
    
    def test_sector_classification(self):
        """Test sector name standardization"""
        test_cases = [
            ('Information Technology', 'IT'),
            ('Banking & Financial Services', 'Banking'),
            ('Oil & Gas', 'Energy'),
            ('Pharmaceuticals', 'Pharma')
        ]
        
        for sector, expected in test_cases:
            standardized = self.string_util.standardize_sector_name(sector)
            self.assertEqual(standardized, expected)


class TestValidationUtility(unittest.TestCase):
    """Test cases for validation utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validation_util = ValidationUtility()
    
    def test_date_validation(self):
        """Test date validation"""
        # Valid dates
        valid_dates = ['2022-06-15', '2022-01-01', '2022-12-31']
        
        for date_str in valid_dates:
            self.assertTrue(self.validation_util.is_valid_date(date_str))
        
        # Invalid dates
        invalid_dates = ['2022-13-01', '2022-02-30', 'invalid-date', '']
        
        for date_str in invalid_dates:
            self.assertFalse(self.validation_util.is_valid_date(date_str))
    
    def test_price_validation(self):
        """Test stock price validation"""
        # Valid prices
        valid_prices = [100.50, 2650.75, 0.01, 50000.00]
        
        for price in valid_prices:
            self.assertTrue(self.validation_util.is_valid_stock_price(price))
        
        # Invalid prices
        invalid_prices = [-100, 0, -0.01, None, 'invalid']
        
        for price in invalid_prices:
            self.assertFalse(self.validation_util.is_valid_stock_price(price))
    
    def test_volume_validation(self):
        """Test trading volume validation"""
        # Valid volumes
        valid_volumes = [1000, 100000, 5000000]
        
        for volume in valid_volumes:
            self.assertTrue(self.validation_util.is_valid_volume(volume))
        
        # Invalid volumes
        invalid_volumes = [-1000, 0, -1, None, 'invalid']
        
        for volume in invalid_volumes:
            self.assertFalse(self.validation_util.is_valid_volume(volume))
    
    def test_percentage_validation(self):
        """Test percentage validation"""
        # Valid percentages
        valid_percentages = [0.5, -2.5, 10.0, -15.75, 0.0]
        
        for percentage in valid_percentages:
            self.assertTrue(self.validation_util.is_valid_percentage(percentage))
        
        # Invalid percentages (outside reasonable range)
        invalid_percentages = [150, -150, None, 'invalid']
        
        for percentage in invalid_percentages:
            self.assertFalse(self.validation_util.is_valid_percentage(percentage))
    
    def test_market_data_validation(self):
        """Test complete market data validation"""
        # Valid market data
        valid_data = {
            'symbol': 'RELIANCE.NS',
            'date': '2022-06-15',
            'open': 2650.00,
            'high': 2670.50,
            'low': 2640.25,
            'close': 2655.75,
            'volume': 1500000
        }
        
        validation_result = self.validation_util.validate_market_data(valid_data)
        self.assertTrue(validation_result['is_valid'])
        self.assertEqual(len(validation_result['errors']), 0)
        
        # Invalid market data
        invalid_data = {
            'symbol': 'INVALID',
            'date': '2022-13-01',
            'open': -100,
            'high': 2670.50,
            'low': 2680.25,  # Low > High (invalid)
            'close': 2655.75,
            'volume': -1000
        }
        
        validation_result = self.validation_util.validate_market_data(invalid_data)
        self.assertFalse(validation_result['is_valid'])
        self.assertGreater(len(validation_result['errors']), 0)


class TestLoggingUtility(unittest.TestCase):
    """Test cases for logging utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logging_util = LoggingUtility()
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logger_creation(self):
        """Test logger creation with different configurations"""
        # Test basic logger
        logger = self.logging_util.create_logger(
            name='market_research_test',
            log_file=self.log_file,
            level='INFO'
        )
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'market_research_test')
    
    def test_log_levels(self):
        """Test different log levels"""
        logger = self.logging_util.create_logger(
            name='test_levels',
            log_file=self.log_file,
            level='DEBUG'
        )
        
        # Test logging at different levels
        logger.debug('Debug message for NSE data processing')
        logger.info('Info message for market analysis')
        logger.warning('Warning message for unusual volume')
        logger.error('Error message for API failure')
        
        # Check if log file was created
        self.assertTrue(os.path.exists(self.log_file))
        
        # Check if log contains messages
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn('Debug message', log_content)
            self.assertIn('Info message', log_content)
            self.assertIn('Warning message', log_content)
            self.assertIn('Error message', log_content)
    
    def test_structured_logging(self):
        """Test structured logging for market events"""
        logger = self.logging_util.create_structured_logger(
            name='market_events',
            log_file=self.log_file
        )
        
        # Log structured market event
        market_event = {
            'event_type': 'price_alert',
            'symbol': 'RELIANCE.NS',
            'price': 2650.50,
            'change': 2.5,
            'timestamp': '2022-06-15T10:30:00'
        }
        
        logger.info('Market alert triggered', extra=market_event)
        
        # Verify structured log
        self.assertTrue(os.path.exists(self.log_file))
    
    def test_log_rotation(self):
        """Test log file rotation"""
        logger = self.logging_util.create_rotating_logger(
            name='rotating_test',
            log_file=self.log_file,
            max_bytes=1024,  # Small size for testing
            backup_count=3
        )
        
        # Generate enough logs to trigger rotation
        for i in range(100):
            logger.info(f'Market data entry {i}: RELIANCE.NS price update')
        
        # Check if rotation occurred (backup files created)
        log_dir = os.path.dirname(self.log_file)
        log_files = [f for f in os.listdir(log_dir) if f.startswith('test.log')]
        
        # Should have original file plus backups
        self.assertGreater(len(log_files), 1)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDateUtility))
    test_suite.addTest(unittest.makeSuite(TestFileUtility))
    test_suite.addTest(unittest.makeSuite(TestMathUtility))
    test_suite.addTest(unittest.makeSuite(TestStringUtility))
    test_suite.addTest(unittest.makeSuite(TestValidationUtility))
    test_suite.addTest(unittest.makeSuite(TestLoggingUtility))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Market Research System v1.0 (2022) - Test Results")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    print(f"\n{'='*50}")
    print("Focus: Indian Stock Market (NSE/BSE)")
    print("Coverage: Date utils, File ops, Math calculations, Validation")
    print("Next: Run integration tests with real market data")
    print(f"{'='*50}")