"""
Functional tests package for Market Research System v1.0.

This package contains functional tests that validate end-to-end workflows
and business logic of the market research system.

Test Categories:
- Daily workflow tests
- Report generation workflow tests  
- Data validation workflow tests

These tests simulate real-world usage scenarios and validate that the system
meets functional requirements from a user perspective.
"""

import os
import sys
import unittest
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

# Test configuration
TEST_CONFIG = {
    'indian_market_symbols': [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'SBIN.NS'
    ],
    'indian_indices': [
        '^NSEI',      # NIFTY 50
        '^BSESN',     # BSE SENSEX  
        '^NSEBANK',   # NIFTY BANK
        '^CNXIT'      # NIFTY IT
    ],
    'test_date_range': {
        'start': '2022-01-01',
        'end': '2022-12-31'
    },
    'sectors': {
        'Technology': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
        'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS'],
        'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS'],
        'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'],
        'Telecom': ['BHARTIARTL.NS', 'IDEA.NS']
    }
}

def get_test_config():
    """Get test configuration dictionary."""
    return TEST_CONFIG.copy()

def setup_test_environment():
    """Set up common test environment variables."""
    os.environ['MARKET_RESEARCH_TEST_MODE'] = '1'
    os.environ['MARKET_RESEARCH_LOG_LEVEL'] = 'DEBUG'
    return True

def cleanup_test_environment():
    """Clean up test environment variables."""
    test_vars = [
        'MARKET_RESEARCH_TEST_MODE',
        'MARKET_RESEARCH_LOG_LEVEL'
    ]
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]

class BaseFunctionalTest(unittest.TestCase):
    """Base class for functional tests with common setup."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test environment."""
        setup_test_environment()
        cls.config = get_test_config()
        cls.test_start_time = datetime.now()
    
    @classmethod  
    def tearDownClass(cls):
        """Clean up class-level test environment."""
        cleanup_test_environment()
        test_duration = datetime.now() - cls.test_start_time
        print(f"\nTest suite completed in {test_duration.total_seconds():.2f} seconds")
    
    def setUp(self):
        """Set up individual test."""
        self.test_symbols = self.config['indian_market_symbols'][:5]  # Limit for faster tests
        self.test_indices = self.config['indian_indices']
        self.date_range = self.config['test_date_range']
        self.sectors = self.config['sectors']
    
    def assertDataFrameValid(self, df, min_rows=1):
        """Assert that DataFrame is valid and contains expected data."""
        self.assertIsNotNone(df, "DataFrame is None")
        self.assertFalse(df.empty, "DataFrame is empty")
        self.assertGreaterEqual(len(df), min_rows, f"DataFrame has fewer than {min_rows} rows")
    
    def assertValidStockData(self, data, symbol):
        """Assert that stock data is valid."""
        self.assertDataFrameValid(data)
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            self.assertIn(col, data.columns, f"Missing column {col} for {symbol}")
        
        # Check data integrity
        self.assertTrue((data['High'] >= data['Low']).all(), 
                       f"High < Low found in {symbol} data")
        self.assertTrue((data['High'] >= data['Close']).all(), 
                       f"High < Close found in {symbol} data") 
        self.assertTrue((data['Low'] <= data['Close']).all(), 
                       f"Low > Close found in {symbol} data")
        self.assertTrue((data['Volume'] >= 0).all(), 
                       f"Negative volume found in {symbol} data")
        
        # Check for reasonable price ranges (Indian market context)
        self.assertTrue((data['Close'] > 0).all(), 
                       f"Zero or negative prices found in {symbol} data")
        self.assertTrue((data['Close'] < 100000).all(), 
                       f"Unreasonably high prices found in {symbol} data")
    
    def assertValidIndicatorData(self, indicator_data, symbol, indicator_name):
        """Assert that technical indicator data is valid."""
        self.assertIsNotNone(indicator_data, f"{indicator_name} data is None for {symbol}")
        
        # Check for reasonable values based on indicator type
        if indicator_name.upper() in ['RSI']:
            # RSI should be between 0 and 100
            valid_rsi = (indicator_data >= 0) & (indicator_data <= 100)
            self.assertTrue(valid_rsi.all(), 
                           f"Invalid RSI values found for {symbol}")
        
        elif indicator_name.upper() in ['SMA', 'EMA']:
            # Moving averages should be positive for stock prices
            self.assertTrue((indicator_data > 0).all(), 
                           f"Non-positive {indicator_name} values found for {symbol}")
        
        elif indicator_name.upper() in ['MACD']:
            # MACD can be positive or negative, just check it's not all NaN
            self.assertFalse(indicator_data.isna().all(), 
                           f"All {indicator_name} values are NaN for {symbol}")

class IndianMarketTestMixin:
    """Mixin class for Indian market specific test utilities."""
    
    def get_market_hours_ist(self):
        """Get Indian market trading hours in IST."""
        return {
            'market_open': '09:15',
            'market_close': '15:30',
            'timezone': 'Asia/Kolkata'
        }
    
    def is_trading_day(self, date):
        """Check if given date is a trading day for Indian markets."""
        # Basic check - exclude weekends
        # In real implementation, would also exclude Indian holidays
        return date.weekday() < 5
    
    def get_sector_symbols(self, sector_name):
        """Get symbols for a specific sector."""
        return self.config['sectors'].get(sector_name, [])
    
    def validate_indian_symbol_format(self, symbol):
        """Validate that symbol follows Indian market format."""
        # NSE symbols end with .NS, BSE symbols end with .BO
        valid_suffixes = ['.NS', '.BO']
        is_index = symbol.startswith('^')
        
        if is_index:
            return True  # Index symbols have different format
        
        return any(symbol.endswith(suffix) for suffix in valid_suffixes)

def run_functional_tests(test_pattern='test_*.py', verbosity=2):
    """
    Run functional tests with specified pattern.
    
    Args:
        test_pattern (str): Pattern to match test files
        verbosity (int): Test output verbosity level
    
    Returns:
        unittest.TestResult: Test results
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.dirname(__file__),
        pattern=test_pattern,
        top_level_dir=os.path.dirname(__file__)
    )
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)

def create_test_report(test_result, output_file='functional_test_report.txt'):
    """
    Create a test report from test results.
    
    Args:
        test_result: unittest.TestResult object
        output_file (str): Output file path for report
    """
    report_lines = [
        "=" * 60,
        "FUNCTIONAL TEST REPORT",
        "=" * 60,
        f"Tests Run: {test_result.testsRun}",
        f"Failures: {len(test_result.failures)}",
        f"Errors: {len(test_result.errors)}",
        f"Success Rate: {((test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun * 100):.2f}%",
        "",
        "DETAILED RESULTS:",
        "-" * 40
    ]
    
    if test_result.failures:
        report_lines.extend([
            "",
            "FAILURES:",
            "-" * 20
        ])
        for test, traceback in test_result.failures:
            report_lines.extend([
                f"Test: {test}",
                f"Traceback: {traceback}",
                ""
            ])
    
    if test_result.errors:
        report_lines.extend([
            "",
            "ERRORS:", 
            "-" * 20
        ])
        for test, traceback in test_result.errors:
            report_lines.extend([
                f"Test: {test}",
                f"Traceback: {traceback}",
                ""
            ])
    
    # Write report to file
    report_content = "\n".join(report_lines)
    
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(os.path.dirname(__file__), '../../../reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, output_file)
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Test report saved to: {report_path}")
    return report_path

# Version 1 specific test utilities
class V1TestDataGenerator:
    """Generate test data specific to Version 1 capabilities."""
    
    @staticmethod
    def generate_mock_price_data(symbol, days=30):
        """Generate mock price data for testing."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate realistic Indian stock price data
        base_price = np.random.uniform(100, 5000)  # INR price range
        prices = []
        
        for i in range(len(dates)):
            # Simple random walk with mean reversion
            if i == 0:
                price = base_price
            else:
                change = np.random.normal(0, 0.02)  # 2% daily volatility
                price = prices[-1] * (1 + change)
                price = max(price, 1)  # Minimum price of Re 1
            
            prices.append(price)
        
        # Generate OHLCV data
        data = {
            'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(100000, 10000000) for _ in prices]
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # Ensure High >= Low and other constraints
        df['High'] = np.maximum(df['High'], df[['Open', 'Close']].max(axis=1))
        df['Low'] = np.minimum(df['Low'], df[['Open', 'Close']].min(axis=1))
        
        return df

# Export main classes and functions
__all__ = [
    'BaseFunctionalTest',
    'IndianMarketTestMixin', 
    'V1TestDataGenerator',
    'get_test_config',
    'setup_test_environment',
    'cleanup_test_environment',
    'run_functional_tests',
    'create_test_report',
    'TEST_CONFIG'
]