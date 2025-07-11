#!/usr/bin/env python3
"""
Unit tests for data cleaning modules
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

from data.data_cleaner import DataCleaner
from data.exceptions import DataValidationError, DataCleaningError


class TestDataCleaner(unittest.TestCase):
    """Test cases for DataCleaner class"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.data_cleaner = DataCleaner()
        
        # Create sample Indian stock data
        dates = pd.date_range('2022-01-01', periods=100, freq='D')
        self.sample_stock_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 1000, 100),
            'High': np.random.uniform(100, 1100, 100),
            'Low': np.random.uniform(90, 950, 100),
            'Close': np.random.uniform(95, 1050, 100),
            'Volume': np.random.randint(1000, 1000000, 100),
            'Symbol': ['RELIANCE.NS'] * 100
        })
        
        # Introduce some data quality issues
        self.dirty_data = self.sample_stock_data.copy()
        self.dirty_data.loc[10, 'High'] = np.nan  # Missing value
        self.dirty_data.loc[15, 'Low'] = 1200     # Inconsistent price (Low > High)
        self.dirty_data.loc[20, 'Volume'] = -100  # Negative volume
        self.dirty_data.loc[25, 'Close'] = 0      # Zero price
        
        # Create sample economic data
        self.sample_economic_data = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=12, freq='M'),
            'GDP_Growth': [7.2, 7.5, 7.8, 7.1, 6.9, 7.3, 7.6, 7.4, 7.0, 6.8, 7.2, 7.5],
            'Inflation_Rate': [5.1, 5.3, 5.8, 6.2, 6.5, 6.1, 5.9, 5.7, 5.4, 5.2, 5.0, 4.8],
            'Interest_Rate': [4.0, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5]
        })

    def test_remove_missing_values(self):
        """Test removal of missing values"""
        cleaned_data = self.data_cleaner.remove_missing_values(self.dirty_data)
        
        # Should have one less row due to NaN removal
        self.assertEqual(len(cleaned_data), len(self.dirty_data) - 1)
        
        # Should not contain any NaN values
        self.assertFalse(cleaned_data.isnull().any().any())

    def test_fix_price_inconsistencies(self):
        """Test fixing of price inconsistencies (Low > High, etc.)"""
        cleaned_data = self.data_cleaner.fix_price_inconsistencies(self.dirty_data)
        
        # Check that Low <= High for all rows
        self.assertTrue((cleaned_data['Low'] <= cleaned_data['High']).all())
        
        # Check that Low <= Close <= High for most cases
        valid_close_range = (
            (cleaned_data['Close'] >= cleaned_data['Low']) & 
            (cleaned_data['Close'] <= cleaned_data['High'])
        )
        # Allow some flexibility for adjusted data
        self.assertGreater(valid_close_range.sum() / len(cleaned_data), 0.9)

    def test_remove_outliers(self):
        """Test outlier removal using IQR method"""
        # Add extreme outliers
        outlier_data = self.sample_stock_data.copy()
        outlier_data.loc[50, 'Close'] = 10000  # Extreme high price
        outlier_data.loc[51, 'Volume'] = 50000000  # Extreme high volume
        
        cleaned_data = self.data_cleaner.remove_outliers(outlier_data, ['Close', 'Volume'])
        
        # Should have fewer rows after outlier removal
        self.assertLessEqual(len(cleaned_data), len(outlier_data))
        
        # Outliers should be removed
        self.assertNotIn(10000, cleaned_data['Close'].values)

    def test_normalize_volumes(self):
        """Test volume normalization"""
        normalized_data = self.data_cleaner.normalize_volumes(self.sample_stock_data)
        
        # Volume should be converted to proper numeric format
        self.assertTrue(pd.api.types.is_numeric_dtype(normalized_data['Volume']))
        
        # Should not contain negative volumes
        self.assertTrue((normalized_data['Volume'] >= 0).all())

    def test_validate_indian_stock_symbols(self):
        """Test validation of Indian stock symbols"""
        # Valid NSE symbols
        valid_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
        for symbol in valid_symbols:
            self.assertTrue(self.data_cleaner.validate_indian_stock_symbol(symbol))
        
        # Valid BSE symbols
        bse_symbols = ['RELIANCE.BO', 'TCS.BO', 'INFY.BO']
        for symbol in bse_symbols:
            self.assertTrue(self.data_cleaner.validate_indian_stock_symbol(symbol))
        
        # Invalid symbols
        invalid_symbols = ['AAPL', 'MSFT', 'INVALID.XX']
        for symbol in invalid_symbols:
            self.assertFalse(self.data_cleaner.validate_indian_stock_symbol(symbol))

    def test_clean_stock_data_pipeline(self):
        """Test complete stock data cleaning pipeline"""
        cleaned_data = self.data_cleaner.clean_stock_data(self.dirty_data)
        
        # Should return a DataFrame
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        
        # Should not contain NaN values
        self.assertFalse(cleaned_data.isnull().any().any())
        
        # Should have proper price relationships
        self.assertTrue((cleaned_data['Low'] <= cleaned_data['High']).all())
        
        # Should not contain negative or zero values
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            self.assertTrue((cleaned_data[col] > 0).all())
        
        # Volume should be non-negative
        self.assertTrue((cleaned_data['Volume'] >= 0).all())

    def test_clean_economic_data(self):
        """Test economic data cleaning"""
        # Add some issues to economic data
        dirty_economic = self.sample_economic_data.copy()
        dirty_economic.loc[5, 'GDP_Growth'] = np.nan
        dirty_economic.loc[8, 'Inflation_Rate'] = -1  # Negative inflation (deflation)
        
        cleaned_data = self.data_cleaner.clean_economic_data(dirty_economic)
        
        # Should handle missing values appropriately
        self.assertFalse(cleaned_data.isnull().any().any())
        
        # Should maintain reasonable economic indicator ranges
        self.assertTrue((cleaned_data['GDP_Growth'] >= -5).all())  # Reasonable GDP range
        self.assertTrue((cleaned_data['GDP_Growth'] <= 15).all())

    def test_standardize_date_formats(self):
        """Test date format standardization"""
        # Create data with mixed date formats
        mixed_date_data = pd.DataFrame({
            'Date': ['2022-01-01', '01/02/2022', '2022.01.03', '2022-1-4'],
            'Value': [100, 110, 105, 115]
        })
        
        standardized_data = self.data_cleaner.standardize_date_formats(mixed_date_data)
        
        # All dates should be in datetime format
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(standardized_data['Date']))
        
        # Dates should be properly parsed
        self.assertEqual(len(standardized_data), 4)

    def test_handle_stock_splits_and_bonuses(self):
        """Test handling of stock splits and bonus adjustments"""
        # Create data with a stock split scenario
        split_data = self.sample_stock_data.copy()
        
        # Simulate 2:1 stock split on day 50
        split_day = 50
        split_data.loc[split_day:, 'Close'] = split_data.loc[split_day:, 'Close'] / 2
        split_data.loc[split_day:, 'Volume'] = split_data.loc[split_day:, 'Volume'] * 2
        
        adjusted_data = self.data_cleaner.adjust_for_corporate_actions(
            split_data, 
            corporate_actions=[{'date': '2022-02-19', 'type': 'split', 'ratio': 2}]
        )
        
        # Data should be adjusted for splits
        self.assertIsInstance(adjusted_data, pd.DataFrame)
        self.assertEqual(len(adjusted_data), len(split_data))

    def test_remove_non_trading_days(self):
        """Test removal of non-trading days (weekends, holidays)"""
        # Create data including weekends
        all_days_data = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=30, freq='D'),
            'Close': np.random.uniform(100, 200, 30),
            'Volume': np.random.randint(1000, 10000, 30)
        })
        
        trading_days_data = self.data_cleaner.remove_non_trading_days(all_days_data)
        
        # Should have fewer days (no weekends)
        self.assertLessEqual(len(trading_days_data), len(all_days_data))
        
        # Should not contain Saturdays or Sundays
        weekdays = trading_days_data['Date'].dt.dayofweek
        self.assertNotIn(5, weekdays.values)  # Saturday
        self.assertNotIn(6, weekdays.values)  # Sunday

    def test_data_quality_score(self):
        """Test data quality scoring"""
        clean_score = self.data_cleaner.calculate_data_quality_score(self.sample_stock_data)
        dirty_score = self.data_cleaner.calculate_data_quality_score(self.dirty_data)
        
        # Clean data should have higher quality score
        self.assertGreater(clean_score, dirty_score)
        self.assertGreaterEqual(clean_score, 0.8)  # Should be high quality
        self.assertLessEqual(dirty_score, 0.7)     # Should be lower quality

    def test_currency_conversion(self):
        """Test currency conversion for international comparison"""
        # Test INR to USD conversion (approximate rate for 2022)
        inr_data = pd.DataFrame({
            'Price_INR': [7500, 8000, 8500],  # Sample prices in INR
            'Date': pd.date_range('2022-01-01', periods=3)
        })
        
        usd_data = self.data_cleaner.convert_currency(inr_data, 'Price_INR', 'INR', 'USD')
        
        # Should add USD column
        self.assertIn('Price_USD', usd_data.columns)
        
        # USD values should be smaller than INR values
        self.assertTrue((usd_data['Price_USD'] < usd_data['Price_INR']).all())

    def test_sector_classification(self):
        """Test sector classification for Indian stocks"""
        stock_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ITC.NS', 'BHARTIARTL.NS']
        
        classified_data = self.data_cleaner.classify_sectors(stock_symbols)
        
        # Should return sector information
        self.assertIsInstance(classified_data, dict)
        self.assertIn('RELIANCE.NS', classified_data)
        
        # Should have appropriate sectors
        expected_sectors = ['Energy', 'Technology', 'Banking', 'FMCG', 'Telecom']
        actual_sectors = list(classified_data.values())
        for sector in actual_sectors:
            self.assertIn(sector, expected_sectors)

    def test_error_handling(self):
        """Test error handling in data cleaning"""
        # Test with None input
        with self.assertRaises(DataValidationError):
            self.data_cleaner.clean_stock_data(None)
        
        # Test with empty DataFrame
        with self.assertRaises(DataValidationError):
            self.data_cleaner.clean_stock_data(pd.DataFrame())
        
        # Test with missing required columns
        incomplete_data = pd.DataFrame({'Date': [1, 2, 3]})
        with self.assertRaises(DataValidationError):
            self.data_cleaner.clean_stock_data(incomplete_data)


class TestDataCleanerEdgeCases(unittest.TestCase):
    """Test edge cases for data cleaning"""
    
    def setUp(self):
        self.data_cleaner = DataCleaner()
    
    def test_single_row_data(self):
        """Test cleaning with single row of data"""
        single_row = pd.DataFrame({
            'Date': ['2022-01-01'],
            'Open': [100.0],
            'High': [105.0],
            'Low': [95.0],
            'Close': [102.0],
            'Volume': [10000],
            'Symbol': ['RELIANCE.NS']
        })
        
        cleaned = self.data_cleaner.clean_stock_data(single_row)
        self.assertEqual(len(cleaned), 1)
    
    def test_all_missing_data(self):
        """Test handling of completely missing data"""
        all_nan_data = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=5),
            'Open': [np.nan] * 5,
            'High': [np.nan] * 5,
            'Low': [np.nan] * 5,
            'Close': [np.nan] * 5,
            'Volume': [np.nan] * 5,
            'Symbol': ['RELIANCE.NS'] * 5
        })
        
        with self.assertRaises(DataCleaningError):
            self.data_cleaner.clean_stock_data(all_nan_data)
    
    def test_extreme_values(self):
        """Test handling of extreme values"""
        extreme_data = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=3),
            'Open': [1e-6, 1e6, 100],  # Very small and very large values
            'High': [1e-5, 1.1e6, 110],
            'Low': [1e-7, 0.9e6, 90],
            'Close': [1e-6, 1.05e6, 105],
            'Volume': [1, 1e9, 10000],
            'Symbol': ['RELIANCE.NS'] * 3
        })
        
        cleaned = self.data_cleaner.clean_stock_data(extreme_data)
        
        # Should handle extreme values appropriately
        self.assertIsInstance(cleaned, pd.DataFrame)
        self.assertGreater(len(cleaned), 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataCleaner))
    test_suite.addTest(unittest.makeSuite(TestDataCleanerEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")