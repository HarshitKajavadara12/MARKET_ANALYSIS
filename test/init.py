"""
Market Research System v1.0 - Tests Package
Indian Stock Market Analysis System (2022)

This package contains comprehensive test suites for all system components.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: Component interaction testing  
- Functional Tests: End-to-end workflow testing

Author: Market Research Team
Created: January 2022
Last Updated: December 2022
"""

import os
import sys
import unittest
import warnings
from pathlib import Path

# Add src directory to path for testing
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Suppress warnings during testing
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Test configuration
TEST_CONFIG = {
    'data_dir': Path(__file__).parent / 'fixtures',
    'output_dir': Path(__file__).parent / 'outputs',
    'log_level': 'INFO',
    'timeout': 30,
    'max_retries': 3
}

# Test data configuration for Indian market
INDIAN_TEST_SYMBOLS = {
    'stocks': ['TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS'],
    'indices': ['^NSEI', '^BSESN', '^NSEBANK'],
    'sectors': ['IT', 'Banking', 'Energy', 'FMCG']
}

# Create test directories if they don't exist
for directory in [TEST_CONFIG['data_dir'], TEST_CONFIG['output_dir']]:
    directory.mkdir(parents=True, exist_ok=True)

def get_test_data_path(filename):
    """Get path to test data file."""
    return TEST_CONFIG['data_dir'] / filename

def get_test_output_path(filename):
    """Get path to test output file."""
    return TEST_CONFIG['output_dir'] / filename

def cleanup_test_files():
    """Clean up test output files."""
    import shutil
    if TEST_CONFIG['output_dir'].exists():
        shutil.rmtree(TEST_CONFIG['output_dir'])
    TEST_CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

class BaseTestCase(unittest.TestCase):
    """Base test case with common setup and utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.test_config = TEST_CONFIG
        cls.indian_symbols = INDIAN_TEST_SYMBOLS
        
    def setUp(self):
        """Set up each test."""
        self.test_data_dir = TEST_CONFIG['data_dir']
        self.test_output_dir = TEST_CONFIG['output_dir']
        
    def tearDown(self):
        """Clean up after each test."""
        pass
        
    def assertDataFrameNotEmpty(self, df, msg=None):
        """Assert that DataFrame is not empty."""
        if df is None or df.empty:
            raise AssertionError(msg or "DataFrame is empty")
            
    def assertValidStockData(self, df, msg=None):
        """Assert that DataFrame contains valid stock data."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise AssertionError(msg or f"Missing required columns: {missing_columns}")
            
        # Check for reasonable price values
        if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            raise AssertionError(msg or "Stock prices should be positive")
            
        # Check High >= Low
        if (df['High'] < df['Low']).any():
            raise AssertionError(msg or "High prices should be >= Low prices")

# Test discovery and runner utilities
def discover_tests(start_dir=None, pattern='test_*.py'):
    """Discover all tests in the test directory."""
    if start_dir is None:
        start_dir = Path(__file__).parent
    
    loader = unittest.TestLoader()
    suite = loader.discover(str(start_dir), pattern=pattern)
    return suite

def run_test_suite(verbosity=2):
    """Run the complete test suite."""
    suite = discover_tests()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_specific_test(test_module, test_class=None, test_method=None):
    """Run a specific test."""
    if test_class and test_method:
        suite = unittest.TestSuite()
        suite.addTest(test_class(test_method))
    elif test_class:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    else:
        suite = unittest.TestLoader().loadTestsFromModule(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

# Performance testing utilities
import time
from functools import wraps

def performance_test(max_time=10.0):
    """Decorator for performance testing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            if execution_time > max_time:
                raise AssertionError(
                    f"Test {func.__name__} took {execution_time:.2f}s, "
                    f"expected < {max_time}s"
                )
            return result
        return wrapper
    return decorator

# Mock data generators for testing
def generate_mock_stock_data(symbol, days=30):
    """Generate mock stock data for testing."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )
    
    # Filter out weekends (basic approach)
    dates = [d for d in dates if d.weekday() < 5]
    
    n_days = len(dates)
    base_price = 100.0
    
    # Generate realistic price movements
    returns = np.random.normal(0.001, 0.02, n_days)  # Small daily returns
    prices = base_price * np.cumprod(1 + returns)
    
    # Create OHLC data
    opens = prices
    closes = prices * (1 + np.random.normal(0, 0.01, n_days))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    volumes = np.random.randint(10000, 1000000, n_days)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }).set_index('Date')

def generate_mock_economic_data():
    """Generate mock economic data for testing."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        freq='M'
    )
    
    return pd.DataFrame({
        'Date': dates,
        'GDP_Growth': np.random.normal(6.5, 1.5, len(dates)),
        'CPI': np.random.normal(5.5, 1.0, len(dates)),
        'Repo_Rate': np.random.normal(4.0, 0.5, len(dates)),
        'USD_INR': np.random.normal(75, 2, len(dates))
    }).set_index('Date')

# Test data validation utilities
def validate_test_environment():
    """Validate that test environment is properly set up."""
    errors = []
    
    # Check if test directories exist
    if not TEST_CONFIG['data_dir'].exists():
        errors.append(f"Test data directory missing: {TEST_CONFIG['data_dir']}")
    
    if not TEST_CONFIG['output_dir'].exists():
        errors.append(f"Test output directory missing: {TEST_CONFIG['output_dir']}")
    
    # Check if source code is accessible
    try:
        import src
    except ImportError:
        errors.append("Source code not accessible for testing")
    
    return errors

# Test result reporting
class TestResultCollector:
    """Collect and format test results."""
    
    def __init__(self):
        self.results = {
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'total_time': 0,
            'details': []
        }
    
    def add_result(self, test_name, status, execution_time, error_msg=None):
        """Add a test result."""
        self.results[status] += 1
        self.results['total_time'] += execution_time
        self.results['details'].append({
            'test': test_name,
            'status': status,
            'time': execution_time,
            'error': error_msg
        })
    
    def get_summary(self):
        """Get test summary."""
        total = sum([
            self.results['passed'],
            self.results['failed'], 
            self.results['errors'],
            self.results['skipped']
        ])
        
        return {
            'total_tests': total,
            'success_rate': (self.results['passed'] / total * 100) if total > 0 else 0,
            'total_time': self.results['total_time'],
            'results': self.results
        }

if __name__ == '__main__':
    # Validate test environment
    validation_errors = validate_test_environment()
    if validation_errors:
        print("Test environment validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Run all tests
    print("Running Market Research System Test Suite...")
    print("=" * 50)
    
    success = run_test_suite()
    
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)