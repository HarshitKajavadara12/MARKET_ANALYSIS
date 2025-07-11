"""
Market Research System v1.0 - Unit Tests Package
Indian Stock Market Analysis System (2022)

Unit tests for individual system components.
Each module is tested in isolation with mocked dependencies.

Test Structure:
- test_data_fetchers.py: Data fetching module tests
- test_data_cleaners.py: Data cleaning module tests  
- test_indicators.py: Technical indicators tests
- test_analyzers.py: Market analysis module tests
- test_visualizations.py: Visualization module tests
- test_report_generators.py: Report generation tests
- test_utils.py: Utility function tests

Author: Market Research Team
Created: February 2022
Last Updated: December 2022
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from tests import BaseTestCase, INDIAN_TEST_SYMBOLS

class UnitTestBase(BaseTestCase):
    """Base class for all unit tests with common setup."""
    
    @classmethod
    def setUpClass(cls):
        """Set up unit test environment."""
        super().setUpClass()
        cls.mock_enabled = True
        cls.test_symbols = INDIAN_TEST_SYMBOLS
        
    def setUp(self):
        """Set up each unit test."""
        super().setUp()
        self.maxDiff = None  # Show full diff for failed assertions
        
    def create_mock_response(self, data_type='stock'):
        """Create mock API response data."""
        if data_type == 'stock':
            return self.create_mock_stock_response()
        elif data_type == 'economic':
            return self.create_mock_economic_response()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def create_mock_stock_response(self):
        """Create mock stock data response."""
        import json
        from datetime import datetime, timedelta
        
        base_date = datetime(2022, 1, 1)
        mock_data = {
            'chart': {
                'result': [{
                    'meta': {
                        'currency': 'INR',
                        'symbol': 'TCS.NS',
                        'exchangeName': 'NSI',
                        'instrumentType': 'EQUITY',
                        'firstTradeDate': 1092931800,
                        'regularMarketTime': int(datetime.now().timestamp()),
                        'gmtoffset': 19800,
                        'timezone': 'IST'
                    },
                    'timestamp': [
                        int((base_date + timedelta(days=i)).timestamp())
                        for i in range(30)
                    ],
                    'indicators': {
                        'quote': [{
                            'open': [3500 + i * 10 for i in range(30)],
                            'high': [3550 + i * 10 for i in range(30)],
                            'low': [3450 + i * 10 for i in range(30)],
                            'close': [3500 + i * 10 for i in range(30)],
                            'volume': [100000 + i * 1000 for i in range(30)]
                        }]
                    }
                }]
            }
        }
        return json.dumps(mock_data)
    
    def create_mock_economic_response(self):
        """Create mock economic data response."""
        import json
        from datetime import datetime, timedelta
        
        mock_data = {
            'observations': [
                {
                    'date': (datetime.now() - timedelta(days=30*i)).strftime('%Y-%m-%d'),
                    'value': str(6.5 + 0.1 * i)
                }
                for i in range(12)
            ]
        }
        return json.dumps(mock_data)

def load_unit_tests():
    """Load all unit test modules."""
    test_modules = [
        'test_data_fetchers',
        'test_data_cleaners', 
        'test_indicators',
        'test_analyzers',
        'test_visualizations',
        'test_report_generators',
        'test_utils'
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module = __import__(f'tests.unit.{module_name}', fromlist=[module_name])
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"Warning: Could not load {module_name}: {e}")
    
    return suite

def run_unit_tests(verbosity=2):
    """Run all unit tests."""
    suite = load_unit_tests()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result

if __name__ == '__main__':
    print("Running Unit Tests for Market Research System...")
    print("=" * 60)
    result = run_unit_tests()
    if result.wasSuccessful():
        print("\nAll unit tests passed!")
    else:
        print(f"\nUnit tests completed with {len(result.failures)} failures and {len(result.errors)} errors")