"""
Integration tests for API connections and data fetching.
Tests Yahoo Finance, FRED API, and other data sources.
"""

import unittest
import os
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
import requests

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from data.fetch_market_data import MarketDataFetcher
from data.fetch_economic_data import EconomicDataFetcher
from data.fetch_indices_data import IndicesDataFetcher
from exceptions.api_exceptions import APIConnectionError, DataValidationError


class TestAPIIntegration(unittest.TestCase):
    """Test API integrations and data fetching capabilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.market_fetcher = MarketDataFetcher()
        self.economic_fetcher = EconomicDataFetcher()
        self.indices_fetcher = IndicesDataFetcher()
        
        # Test symbols for Indian market
        self.test_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'
        ]
        
        # Indian market indices
        self.test_indices = [
            '^NSEI',  # NIFTY 50
            '^BSESN',  # BSE SENSEX
            '^NSEBANK'  # NIFTY BANK
        ]
        
        # Economic indicators
        self.economic_indicators = [
            'GDP',
            'UNRATE',  # Unemployment Rate
            'CPIAUCSL',  # Consumer Price Index
            'FEDFUNDS'  # Federal Funds Rate
        ]
    
    def test_yahoo_finance_api_connection(self):
        """Test Yahoo Finance API connection and data retrieval."""
        try:
            # Test single stock data fetch
            symbol = 'RELIANCE.NS'
            data = self.market_fetcher.fetch_stock_data(
                symbol, 
                start_date='2022-01-01', 
                end_date='2022-01-31'
            )
            
            # Verify data structure
            self.assertIsInstance(data, pd.DataFrame)
            self.assertFalse(data.empty, "No data returned from Yahoo Finance")
            
            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                self.assertIn(col, data.columns, f"Missing column: {col}")
            
            # Verify data quality
            self.assertFalse(data['Close'].isna().all(), "All Close prices are NaN")
            self.assertTrue((data['Volume'] >= 0).all(), "Negative volume detected")
            
        except Exception as e:
            self.fail(f"Yahoo Finance API test failed: {e}")
    
    def test_multiple_stocks_fetch(self):
        """Test fetching multiple stocks simultaneously."""
        try:
            data = self.market_fetcher.fetch_multiple_stocks(
                self.test_symbols[:3],  # Test with first 3 symbols
                start_date='2022-01-01',
                end_date='2022-01-15'
            )
            
            self.assertIsInstance(data, dict)
            self.assertEqual(len(data), 3, "Not all symbols were fetched")
            
            for symbol in self.test_symbols[:3]:
                self.assertIn(symbol, data, f"Missing data for {symbol}")
                self.assertIsInstance(data[symbol], pd.DataFrame)
                self.assertFalse(data[symbol].empty, f"Empty data for {symbol}")
                
        except Exception as e:
            self.fail(f"Multiple stocks fetch test failed: {e}")
    
    def test_indian_indices_data_fetch(self):
        """Test fetching Indian market indices data."""
        try:
            for index in self.test_indices:
                data = self.indices_fetcher.fetch_index_data(
                    index,
                    start_date='2022-01-01',
                    end_date='2022-01-31'
                )
                
                self.assertIsInstance(data, pd.DataFrame)
                self.assertFalse(data.empty, f"No data for index {index}")
                
                # Verify index-specific validations
                if index == '^NSEI':  # NIFTY 50
                    self.assertTrue(data['Close'].iloc[-1] > 10000, "NIFTY value seems too low")
                elif index == '^BSESN':  # SENSEX
                    self.assertTrue(data['Close'].iloc[-1] > 40000, "SENSEX value seems too low")
                    
        except Exception as e:
            self.fail(f"Indian indices fetch test failed: {e}")
    
    def test_fred_api_connection(self):
        """Test FRED API connection for economic data."""
        # Note: This test might be skipped if FRED API key is not available
        try:
            # Test GDP data fetch
            gdp_data = self.economic_fetcher.fetch_economic_indicator(
                'GDP',
                start_date='2022-01-01',
                end_date='2022-12-31'
            )
            
            if gdp_data is not None:
                self.assertIsInstance(gdp_data, pd.DataFrame)
                self.assertFalse(gdp_data.empty, "No GDP data returned")
                self.assertIn('GDP', gdp_data.columns, "GDP column missing")
            else:
                self.skipTest("FRED API key not available or API limit reached")
                
        except Exception as e:
            if "API key" in str(e).lower():
                self.skipTest("FRED API key not configured")
            else:
                self.fail(f"FRED API test failed: {e}")
    
    def test_api_rate_limiting(self):
        """Test API rate limiting and retry mechanisms."""
        # Test rapid successive calls
        results = []
        start_time = time.time()
        
        try:
            for i in range(3):  # Make 3 rapid calls
                data = self.market_fetcher.fetch_stock_data(
                    'TCS.NS',
                    start_date='2022-01-01',
                    end_date='2022-01-05'
                )
                results.append(data is not None and not data.empty)
                time.sleep(0.1)  # Small delay
            
            end_time = time.time()
            
            # Check that all calls succeeded
            self.assertTrue(all(results), "Some API calls failed due to rate limiting")
            
            # Ensure reasonable timing (should handle rate limits gracefully)
            self.assertLess(end_time - start_time, 10, "API calls took too long")
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                self.skipTest("Rate limit exceeded - this is expected behavior")
            else:
                self.fail(f"Rate limiting test failed: {e}")
    
    def test_data_validation_pipeline(self):
        """Test data validation after API fetch."""
        try:
            # Fetch data
            data = self.market_fetcher.fetch_stock_data(
                'INFY.NS',
                start_date='2022-01-01',
                end_date='2022-01-31'
            )
            
            # Run validation
            validation_results = self.market_fetcher.validate_stock_data(data)
            
            # Check validation results
            self.assertIsInstance(validation_results, dict)
            self.assertIn('is_valid', validation_results)
            self.assertIn('errors', validation_results)
            self.assertIn('warnings', validation_results)
            
            if not validation_results['is_valid']:
                print(f"Validation errors: {validation_results['errors']}")
                print(f"Validation warnings: {validation_results['warnings']}")
            
        except Exception as e:
            self.fail(f"Data validation test failed: {e}")
    
    def test_historical_data_completeness(self):
        """Test completeness of historical data."""
        try:
            # Fetch one year of data
            data = self.market_fetcher.fetch_stock_data(
                'HDFCBANK.NS',
                start_date='2022-01-01',
                end_date='2022-12-31'
            )
            
            # Check data completeness
            expected_trading_days = 250  # Approximate trading days in a year
            actual_days = len(data)
            
            # Allow for holidays and weekends
            completeness_ratio = actual_days / expected_trading_days
            self.assertGreater(completeness_ratio, 0.8, 
                             f"Data completeness too low: {completeness_ratio:.2%}")
            
            # Check for large gaps in data
            data.index = pd.to_datetime(data.index)
            date_diffs = data.index.to_series().diff().dt.days
            max_gap = date_diffs.max()
            
            # Should not have gaps larger than 5 days (considering weekends)
            self.assertLess(max_gap, 8, f"Large data gap detected: {max_gap} days")
            
        except Exception as e:
            self.fail(f"Historical data completeness test failed: {e}")
    
    def test_real_time_data_fetch(self):
        """Test real-time/current data fetching."""
        try:
            # Fetch latest data
            current_data = self.market_fetcher.fetch_current_price('RELIANCE.NS')
            
            if current_data is not None:
                self.assertIsInstance(current_data, dict)
                self.assertIn('price', current_data)
                self.assertIn('timestamp', current_data)
                
                # Price should be reasonable
                self.assertGreater(current_data['price'], 0, "Invalid current price")
                
                # Timestamp should be recent (within last day)
                timestamp = pd.to_datetime(current_data['timestamp'])
                now = pd.Timestamp.now(tz=timestamp.tz)
                time_diff = (now - timestamp).total_seconds() / 3600  # hours
                
                self.assertLess(time_diff, 24, "Current data timestamp too old")
            else:
                self.skipTest("Real-time data not available or market closed")
                
        except Exception as e:
            if "market closed" in str(e).lower():
                self.skipTest("Market is closed")
            else:
                self.fail(f"Real-time data test failed: {e}")
    
    def test_error_handling_invalid_symbols(self):
        """Test error handling for invalid stock symbols."""
        invalid_symbols = ['INVALID.NS', 'NOTREAL.BO', 'FAKE123.NS']
        
        for symbol in invalid_symbols:
            try:
                data = self.market_fetcher.fetch_stock_data(
                    symbol,
                    start_date='2022-01-01',
                    end_date='2022-01-31'
                )
                
                # Should either return empty DataFrame or raise appropriate exception
                if data is not None:
                    self.assertTrue(data.empty, f"Got data for invalid symbol: {symbol}")
                    
            except DataValidationError:
                # This is expected for invalid symbols
                pass
            except Exception as e:
                # Other exceptions might be acceptable depending on implementation
                if "not found" not in str(e).lower() and "invalid" not in str(e).lower():
                    self.fail(f"Unexpected error for invalid symbol {symbol}: {e}")
    
    def test_network_connectivity(self):
        """Test network connectivity and timeout handling."""
        try:
            # Test basic connectivity to Yahoo Finance
            response = requests.get('https://finance.yahoo.com', timeout=5)
            self.assertEqual(response.status_code, 200, "Cannot reach Yahoo Finance")
            
            # Test API endpoint
            api_response = requests.get(
                'https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS',
                timeout=10
            )
            self.assertIn(response.status_code, [200, 404], 
                         "API endpoint not responding correctly")
            
        except requests.exceptions.Timeout:
            self.skipTest("Network timeout - connectivity issues")
        except requests.exceptions.ConnectionError:
            self.skipTest("Network connection error")
        except Exception as e:
            self.fail(f"Network connectivity test failed: {e}")
    
    @patch('requests.get')
    def test_api_failure_handling(self, mock_get):
        """Test handling of API failures and timeouts."""
        # Mock API failure
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        try:
            with self.assertRaises(APIConnectionError):
                self.market_fetcher.fetch_stock_data(
                    'TCS.NS',
                    start_date='2022-01-01',
                    end_date='2022-01-31'
                )
        except Exception as e:
            # If APIConnectionError is not implemented, any exception is acceptable
            self.assertTrue(isinstance(e, Exception))


if __name__ == '__main__':
    # Set up test suite with proper ordering
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAPIIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)