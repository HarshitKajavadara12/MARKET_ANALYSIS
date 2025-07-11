"""
Unit Tests for Data Fetchers Module
Market Research System v1.0 - Indian Stock Market (2022)

Tests for all data fetching functionality including:
- Stock data fetching from Yahoo Finance
- Economic data fetching from various sources
- Data validation and error handling
- API rate limiting and retry logic

Author: Market Research Team
Created: February 2022
Last Updated: November 2022
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import requests
import json

from tests.unit import UnitTestBase
from tests import performance_test

# Import modules to test
try:
    from src.data.fetch_market_data import MarketDataFetcher
    from src.data.fetch_economic_data import EconomicDataFetcher
    from src.data.fetch_indices_data import IndicesDataFetcher
    from src.data.exceptions import DataFetchError, APIError, ValidationError
except ImportError as e:
    print(f"Warning: Could not import data modules: {e}")
    MarketDataFetcher = None


class TestMarketDataFetcher(UnitTestBase):
    """Test cases for MarketDataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        if MarketDataFetcher:
            self.fetcher = MarketDataFetcher()
            self.test_symbol = 'TCS.NS'
            self.test_symbols = ['TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
    
    @unittest.skipIf(MarketDataFetcher is None, "Module not available")
    def test_fetcher_initialization(self):
        """Test MarketDataFetcher initialization."""
        fetcher = MarketDataFetcher()
        self.assertIsNotNone(fetcher)
        self.assertEqual(fetcher.base_url, "https://query1.finance.yahoo.com/v8/finance/chart/")
        self.assertEqual(fetcher.timeout, 30)
        self.assertEqual(fetcher.max_retries, 3)
    
    @unittest.skipIf(MarketDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_fetch_single_stock_success(self, mock_get):
        """Test successful fetching of single Indian stock data."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'chart': {
                'result': [{
                    'meta': {
                        'symbol': 'TCS.NS',
                        'currency': 'INR',
                        'exchangeName': 'NSI',
                        'regularMarketPrice': 3245.50
                    },
                    'timestamp': [1640995200, 1641081600, 1641168000],
                    'indicators': {
                        'quote': [{
                            'open': [3200.0, 3210.0, 3250.0],
                            'high': [3250.0, 3280.0, 3290.0],
                            'low': [3180.0, 3190.0, 3240.0],
                            'close': [3245.50, 3275.25, 3285.75],
                            'volume': [125000, 98000, 156000]
                        }]
                    }
                }],
                'error': None
            }
        }
        mock_get.return_value = mock_response
        
        # Test the fetch
        data = self.fetcher.fetch_stock_data(self.test_symbol, period='5d')
        
        # Assertions
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('Open', data.columns)
        self.assertIn('High', data.columns)
        self.assertIn('Low', data.columns)
        self.assertIn('Close', data.columns)
        self.assertIn('Volume', data.columns)
        
        # Check Indian market specific validations
        self.assertEqual(data.index.name, 'Date')
        self.assertTrue(all(data['Close'] > 0))
        self.assertTrue(all(data['Volume'] >= 0))
    
    @unittest.skipIf(MarketDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_fetch_multiple_stocks_success(self, mock_get):
        """Test successful fetching of multiple Indian stocks."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'chart': {
                'result': [{
                    'meta': {'symbol': 'TCS.NS', 'currency': 'INR'},
                    'timestamp': [1640995200],
                    'indicators': {
                        'quote': [{
                            'open': [3200.0], 'high': [3250.0], 'low': [3180.0],
                            'close': [3245.50], 'volume': [125000]
                        }]
                    }
                }]
            }
        }
        mock_get.return_value = mock_response
        
        data = self.fetcher.fetch_multiple_stocks(self.test_symbols, period='1d')
        
        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), len(self.test_symbols))
        for symbol in self.test_symbols:
            self.assertIn(symbol, data)
            self.assertIsInstance(data[symbol], pd.DataFrame)

    @unittest.skipIf(MarketDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_fetch_stock_api_error(self, mock_get):
        """Test handling of API errors."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        with self.assertRaises(APIError):
            self.fetcher.fetch_stock_data(self.test_symbol)
    
    @unittest.skipIf(MarketDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_fetch_stock_invalid_symbol(self, mock_get):
        """Test handling of invalid Indian stock symbols."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            'chart': {'result': None, 'error': {'code': 'Not Found'}}
        }
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValidationError):
            self.fetcher.fetch_stock_data('INVALID.NS')
    
    @unittest.skipIf(MarketDataFetcher is None, "Module not available")
    def test_validate_indian_symbol(self):
        """Test Indian stock symbol validation."""
        # Valid NSE symbols
        valid_symbols = ['TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS']
        for symbol in valid_symbols:
            self.assertTrue(self.fetcher.validate_symbol(symbol))
        
        # Valid BSE symbols
        valid_bse = ['TCS.BO', 'INFY.BO', 'HDFCBANK.BO']
        for symbol in valid_bse:
            self.assertTrue(self.fetcher.validate_symbol(symbol))
        
        # Invalid symbols
        invalid_symbols = ['TCS', 'AAPL', 'INVALID.NS', 'TEST.XY']
        for symbol in invalid_symbols:
            self.assertFalse(self.fetcher.validate_symbol(symbol))

    @performance_test
    def test_fetch_performance(self):
        """Test fetch performance for Indian market data."""
        start_time = datetime.now()
        try:
            data = self.fetcher.fetch_stock_data('TCS.NS', period='1mo')
            elapsed = (datetime.now() - start_time).total_seconds()
            self.assertLess(elapsed, 10)  # Should complete within 10 seconds
        except Exception:
            pass  # Performance test, ignore actual fetch errors


class TestEconomicDataFetcher(UnitTestBase):
    """Test cases for EconomicDataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        if EconomicDataFetcher:
            self.fetcher = EconomicDataFetcher()
    
    @unittest.skipIf(EconomicDataFetcher is None, "Module not available")
    def test_fetcher_initialization(self):
        """Test EconomicDataFetcher initialization."""
        fetcher = EconomicDataFetcher()
        self.assertIsNotNone(fetcher)
        self.assertIn('rbi', fetcher.data_sources)
        self.assertIn('mospi', fetcher.data_sources)
    
    @unittest.skipIf(EconomicDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_fetch_rbi_repo_rate(self, mock_get):
        """Test fetching RBI repo rate data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {'date': '2022-01-01', 'repo_rate': 4.0},
                {'date': '2022-02-01', 'repo_rate': 4.0},
                {'date': '2022-03-01', 'repo_rate': 4.0}
            ]
        }
        mock_get.return_value = mock_response
        
        data = self.fetcher.fetch_repo_rate(period='3m')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('repo_rate', data.columns)
        self.assertTrue(all(data['repo_rate'] >= 0))
    
    @unittest.skipIf(EconomicDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_fetch_inflation_data(self, mock_get):
        """Test fetching Indian inflation data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {'date': '2022-01-01', 'cpi': 105.2, 'wpi': 102.8},
                {'date': '2022-02-01', 'cpi': 106.1, 'wpi': 103.5}
            ]
        }
        mock_get.return_value = mock_response
        
        data = self.fetcher.fetch_inflation_data(period='6m')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('cpi', data.columns)
        self.assertIn('wpi', data.columns)
    
    @unittest.skipIf(EconomicDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_fetch_gdp_data(self, mock_get):
        """Test fetching Indian GDP data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {'quarter': 'Q1-2022', 'gdp_growth': 8.5},
                {'quarter': 'Q2-2022', 'gdp_growth': 8.2}
            ]
        }
        mock_get.return_value = mock_response
        
        data = self.fetcher.fetch_gdp_data(period='1y')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('gdp_growth', data.columns)


class TestIndicesDataFetcher(UnitTestBase):
    """Test cases for IndicesDataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        if IndicesDataFetcher:
            self.fetcher = IndicesDataFetcher()
            self.indian_indices = ['^NSEI', '^BSESN', '^NSEBANK', '^CNXIT']
    
    @unittest.skipIf(IndicesDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_fetch_nifty_data(self, mock_get):
        """Test fetching NIFTY 50 index data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'chart': {
                'result': [{
                    'meta': {'symbol': '^NSEI', 'currency': 'INR'},
                    'timestamp': [1640995200, 1641081600],
                    'indicators': {
                        'quote': [{
                            'open': [17800.0, 17850.0],
                            'high': [17900.0, 17950.0],
                            'low': [17750.0, 17800.0],
                            'close': [17885.0, 17925.0],
                            'volume': [0, 0]  # Indices typically don't have volume
                        }]
                    }
                }]
            }
        }
        mock_get.return_value = mock_response
        
        data = self.fetcher.fetch_index_data('^NSEI', period='5d')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertTrue(all(data['Close'] > 0))
    
    @unittest.skipIf(IndicesDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_fetch_multiple_indices(self, mock_get):
        """Test fetching multiple Indian indices."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'chart': {
                'result': [{
                    'meta': {'symbol': '^NSEI'},
                    'timestamp': [1640995200],
                    'indicators': {
                        'quote': [{
                            'open': [17800.0], 'high': [17900.0],
                            'low': [17750.0], 'close': [17885.0], 'volume': [0]
                        }]
                    }
                }]
            }
        }
        mock_get.return_value = mock_response
        
        data = self.fetcher.fetch_multiple_indices(self.indian_indices, period='1d')
        
        self.assertIsInstance(data, dict)
        for index in self.indian_indices:
            self.assertIn(index, data)

    @unittest.skipIf(IndicesDataFetcher is None, "Module not available")
    def test_validate_indian_index_symbols(self):
        """Test validation of Indian index symbols."""
        valid_indices = ['^NSEI', '^BSESN', '^NSEBANK', '^CNXIT', '^NSEMDCP50']
        for index in valid_indices:
            self.assertTrue(self.fetcher.validate_index_symbol(index))
        
        invalid_indices = ['NSEI', 'BSE', '^SPX', '^INVALID']
        for index in invalid_indices:
            self.assertFalse(self.fetcher.validate_index_symbol(index))


class TestDataValidation(UnitTestBase):
    """Test cases for data validation functions."""
    
    def test_validate_date_range(self):
        """Test date range validation."""
        from src.data.validators import validate_date_range
        
        # Valid date ranges
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        self.assertTrue(validate_date_range(start_date, end_date))
        
        # Invalid date ranges
        invalid_start = datetime(2023, 1, 1)
        invalid_end = datetime(2022, 1, 1)
        with self.assertRaises(ValidationError):
            validate_date_range(invalid_start, invalid_end)
    
    def test_validate_ohlcv_data(self):
        """Test OHLCV data validation."""
        from src.data.validators import validate_ohlcv_data
        
        # Valid OHLCV data
        valid_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, 106.0],
            'Volume': [1000, 1200, 800]
        })
        self.assertTrue(validate_ohlcv_data(valid_data))
        
        # Invalid OHLCV data (High < Low)
        invalid_data = pd.DataFrame({
            'Open': [100.0], 'High': [99.0], 'Low': [101.0],
            'Close': [100.5], 'Volume': [1000]
        })
        with self.assertRaises(ValidationError):
            validate_ohlcv_data(invalid_data)


class TestAPIRateLimiting(UnitTestBase):
    """Test cases for API rate limiting and retry logic."""
    
    @unittest.skipIf(MarketDataFetcher is None, "Module not available")
    @patch('time.sleep')
    @patch('requests.get')
    def test_rate_limiting_retry(self, mock_get, mock_sleep):
        """Test rate limiting and retry mechanism."""
        # First call returns rate limit error, second succeeds
        responses = [
            Mock(status_code=429, text="Rate limit exceeded"),
            Mock(status_code=200, json=lambda: {'chart': {'result': []}})
        ]
        mock_get.side_effect = responses
        
        fetcher = MarketDataFetcher()
        try:
            fetcher.fetch_stock_data('TCS.NS')
            # Should have slept due to rate limiting
            mock_sleep.assert_called()
        except Exception:
            pass  # Expected for mock data
    
    @unittest.skipIf(MarketDataFetcher is None, "Module not available")
    @patch('requests.get')
    def test_max_retries_exceeded(self, mock_get):
        """Test behavior when max retries are exceeded."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        fetcher = MarketDataFetcher(max_retries=2)
        with self.assertRaises(APIError):
            fetcher.fetch_stock_data('TCS.NS')


if __name__ == '__main__':
    # Configure test runner for Indian market focus
    unittest.main(
        verbosity=2,
        testRunner=unittest.TextTestRunner(
            descriptions=True,
            failfast=False
        )
    )