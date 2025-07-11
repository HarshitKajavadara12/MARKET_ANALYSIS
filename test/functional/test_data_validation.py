"""
Functional tests for data validation workflow in the market research system.
Tests the complete data validation pipeline from raw data to processed data.

Created: 2022
Author: Market Research System v1.0
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import yfinance as yf
import os
import tempfile
import json


class TestDataValidationWorkflow:
    """
    Functional tests for the complete data validation workflow.
    Tests integration between data fetchers, cleaners, and validators.
    """
    
    def setup_method(self):
        """Setup test data and mock objects for each test."""
        self.test_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
        self.test_date_range = {
            'start': '2022-01-01',
            'end': '2022-12-31'
        }
        
        # Create sample stock data
        self.sample_stock_data = pd.DataFrame({
            'Open': [100.0, 101.5, 102.0, 99.5, 98.0],
            'High': [105.0, 103.0, 104.5, 102.0, 100.5],
            'Low': [98.0, 99.0, 100.0, 97.5, 96.0],
            'Close': [102.0, 100.5, 101.0, 98.0, 99.5],
            'Volume': [1000000, 1200000, 950000, 1100000, 1050000],
            'Adj Close': [102.0, 100.5, 101.0, 98.0, 99.5]
        }, index=pd.date_range('2022-01-01', periods=5, freq='D'))
        
        # Create sample economic data
        self.sample_economic_data = pd.DataFrame({
            'GDP_Growth': [7.2, 7.4, 7.1, 6.9, 7.0],
            'Inflation_Rate': [4.5, 4.8, 5.1, 4.9, 4.7],
            'Interest_Rate': [6.0, 6.25, 6.5, 6.75, 7.0]
        }, index=pd.date_range('2022-01-01', periods=5, freq='M'))
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clean up temp directory if it exists
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.functional
    def test_complete_stock_data_validation_workflow(self):
        """
        Test the complete workflow from fetching stock data to validation.
        This tests the integration of all data validation components.
        """
        # Mock the data fetcher
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = self.sample_stock_data
            
            # Import modules (assuming they exist in the src directory)
            from src.data.fetch_market_data import MarketDataFetcher
            from src.data.data_cleaner import DataCleaner
            from src.data.data_validator import DataValidator
            
            # Initialize components
            fetcher = MarketDataFetcher()
            cleaner = DataCleaner()
            validator = DataValidator()
            
            # Step 1: Fetch data
            raw_data = fetcher.fetch_stock_data(
                symbols=self.test_symbols[0],
                start_date=self.test_date_range['start'],
                end_date=self.test_date_range['end']
            )
            
            # Step 2: Clean data
            cleaned_data = cleaner.clean_stock_data(raw_data)
            
            # Step 3: Validate data
            validation_results = validator.validate_stock_data(cleaned_data)
            
            # Assertions
            assert validation_results['is_valid'] is True
            assert 'missing_data_percentage' in validation_results
            assert 'outlier_count' in validation_results
            assert 'data_quality_score' in validation_results
            
            # Check data integrity
            assert not cleaned_data.empty
            assert all(col in cleaned_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
            assert cleaned_data.index.is_monotonic_increasing
    
    @pytest.mark.functional
    def test_economic_data_validation_workflow(self):
        """
        Test the complete workflow for economic data validation.
        """
        with patch('fredapi.Fred') as mock_fred:
            # Mock FRED API response
            mock_fred_instance = Mock()
            mock_fred_instance.get_series.return_value = pd.Series(
                [7.2, 7.4, 7.1, 6.9, 7.0],
                index=pd.date_range('2022-01-01', periods=5, freq='M')
            )
            mock_fred.return_value = mock_fred_instance
            
            from src.data.fetch_economic_data import EconomicDataFetcher
            from src.data.data_validator import DataValidator
            
            # Initialize components
            fetcher = EconomicDataFetcher()
            validator = DataValidator()
            
            # Fetch economic data
            economic_data = fetcher.fetch_gdp_data(
                start_date=self.test_date_range['start'],
                end_date=self.test_date_range['end']
            )
            
            # Validate economic data
            validation_results = validator.validate_economic_data(economic_data)
            
            # Assertions
            assert validation_results['is_valid'] is True
            assert not economic_data.empty
            assert validation_results['data_completeness'] > 0.8
    
    @pytest.mark.functional
    def test_data_validation_with_missing_values(self):
        """
        Test data validation workflow when data contains missing values.
        """
        # Create data with missing values
        corrupted_data = self.sample_stock_data.copy()
        corrupted_data.loc[corrupted_data.index[1], 'Close'] = np.nan
        corrupted_data.loc[corrupted_data.index[3], 'Volume'] = np.nan
        
        from src.data.data_validator import DataValidator
        from src.data.data_cleaner import DataCleaner
        
        validator = DataValidator()
        cleaner = DataCleaner()
        
        # Test validation before cleaning
        validation_before = validator.validate_stock_data(corrupted_data)
        assert validation_before['missing_data_percentage'] > 0
        
        # Clean the data
        cleaned_data = cleaner.handle_missing_values(corrupted_data)
        
        # Test validation after cleaning
        validation_after = validator.validate_stock_data(cleaned_data)
        assert validation_after['missing_data_percentage'] == 0
        assert validation_after['is_valid'] is True
    
    @pytest.mark.functional
    def test_data_validation_with_outliers(self):
        """
        Test data validation workflow when data contains outliers.
        """
        # Create data with outliers
        outlier_data = self.sample_stock_data.copy()
        outlier_data.loc[outlier_data.index[2], 'Close'] = 1000.0  # Extreme outlier
        outlier_data.loc[outlier_data.index[4], 'Volume'] = 50000000  # Volume outlier
        
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Test outlier detection
        validation_results = validator.validate_stock_data(outlier_data)
        
        assert validation_results['outlier_count'] > 0
        assert 'outlier_details' in validation_results
        assert validation_results['data_quality_score'] < 0.9  # Quality should be affected
    
    @pytest.mark.functional
    def test_cross_validation_between_datasets(self):
        """
        Test cross-validation between stock data and economic data.
        """
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Test date range consistency
        stock_dates = self.sample_stock_data.index
        economic_dates = self.sample_economic_data.index
        
        cross_validation = validator.cross_validate_datasets(
            stock_data=self.sample_stock_data,
            economic_data=self.sample_economic_data
        )
        
        assert 'date_range_overlap' in cross_validation
        assert 'consistency_score' in cross_validation
        assert cross_validation['is_consistent'] in [True, False]
    
    @pytest.mark.functional
    def test_data_validation_error_handling(self):
        """
        Test error handling in data validation workflow.
        """
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        validation_results = validator.validate_stock_data(empty_data)
        
        assert validation_results['is_valid'] is False
        assert 'error_message' in validation_results
        
        # Test with invalid data types
        invalid_data = pd.DataFrame({
            'Open': ['invalid', 'data', 'types'],
            'Close': ['test', 'test', 'test']
        })
        
        validation_results = validator.validate_stock_data(invalid_data)
        assert validation_results['is_valid'] is False
    
    @pytest.mark.functional
    def test_data_validation_performance_metrics(self):
        """
        Test that data validation includes performance metrics.
        """
        from src.data.data_validator import DataValidator
        import time
        
        validator = DataValidator()
        
        start_time = time.time()
        validation_results = validator.validate_stock_data(self.sample_stock_data)
        validation_time = time.time() - start_time
        
        # Check that validation completes within reasonable time
        assert validation_time < 5.0  # Should complete within 5 seconds
        
        # Check performance metrics are included
        assert 'validation_timestamp' in validation_results
        assert 'processing_time' in validation_results
    
    @pytest.mark.functional
    def test_data_validation_report_generation(self):
        """
        Test generation of data validation reports.
        """
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Run validation
        validation_results = validator.validate_stock_data(self.sample_stock_data)
        
        # Generate validation report
        report_path = os.path.join(self.temp_dir, 'validation_report.json')
        validator.generate_validation_report(validation_results, report_path)
        
        # Check report file exists and contains expected data
        assert os.path.exists(report_path)
        
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        assert 'validation_summary' in report_data
        assert 'data_quality_metrics' in report_data
        assert 'recommendations' in report_data
    
    @pytest.mark.functional
    def test_batch_data_validation(self):
        """
        Test validation of multiple datasets in batch.
        """
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Create multiple datasets
        datasets = {
            'RELIANCE.NS': self.sample_stock_data,
            'TCS.NS': self.sample_stock_data.copy(),
            'INFY.NS': self.sample_stock_data.copy()
        }
        
        # Run batch validation
        batch_results = validator.validate_multiple_datasets(datasets)
        
        assert len(batch_results) == 3
        for symbol, results in batch_results.items():
            assert symbol in self.test_symbols[:3]
            assert 'is_valid' in results
            assert 'data_quality_score' in results
    
    @pytest.mark.functional
    def test_real_time_data_validation(self):
        """
        Test validation of real-time data streams.
        """
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Simulate real-time data points
        real_time_data = pd.DataFrame({
            'symbol': ['RELIANCE.NS'] * 5,
            'price': [2400.50, 2401.25, 2399.80, 2402.10, 2400.95],
            'volume': [1000, 1500, 800, 1200, 950],
            'timestamp': pd.date_range('2022-01-01 09:15:00', periods=5, freq='1min')
        })
        
        # Validate real-time data
        rt_validation = validator.validate_realtime_data(real_time_data)
        
        assert 'is_valid' in rt_validation
        assert 'latency_check' in rt_validation
        assert 'data_freshness' in rt_validation


class TestDataValidationEdgeCases:
    """
    Test edge cases and error conditions in data validation.
    """
    
    @pytest.mark.functional
    def test_validation_with_weekend_data(self):
        """
        Test validation when data includes weekend dates (should be filtered).
        """
        # Create data with weekend dates
        weekend_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        }, index=pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03']))  # Sat, Sun, Mon
        
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        validation_results = validator.validate_stock_data(weekend_data)
        
        assert 'weekend_data_detected' in validation_results
        assert validation_results['business_days_only'] is True
    
    @pytest.mark.functional
    def test_validation_with_split_adjusted_data(self):
        """
        Test validation of data that includes stock splits.
        """
        # Create data with potential split
        split_data = pd.DataFrame({
            'Open': [1000, 1010, 500, 505],  # Potential 2:1 split
            'Close': [1005, 1015, 502, 508],
            'Volume': [100000, 110000, 220000, 210000],  # Volume should double
            'Adj Close': [502.5, 507.5, 502, 508]
        }, index=pd.date_range('2022-01-01', periods=4, freq='D'))
        
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        validation_results = validator.validate_stock_data(split_data)
        
        assert 'potential_corporate_actions' in validation_results
        assert 'split_detection' in validation_results
    
    @pytest.mark.functional
    def test_validation_memory_efficiency(self):
        """
        Test that validation works efficiently with large datasets.
        """
        import psutil
        import os
        
        # Create large dataset
        large_data = pd.DataFrame(np.random.randn(100000, 6), 
                                columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'],
                                index=pd.date_range('2020-01-01', periods=100000, freq='1min'))
        
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        validation_results = validator.validate_stock_data(large_data)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Validation should not use excessive memory
        assert memory_increase < 500  # Less than 500MB increase
        assert validation_results['is_valid'] in [True, False]


@pytest.mark.integration
class TestDataValidationIntegration:
    """
    Integration tests that test the complete system workflow.
    """
    
    def test_end_to_end_validation_workflow(self):
        """
        Test the complete end-to-end validation workflow.
        """
        with patch('yfinance.download') as mock_download:
            # Mock data download
            sample_data = pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [105, 106, 107],
                'Low': [98, 99, 100],
                'Close': [102, 103, 104],
                'Volume': [1000000, 1100000, 1200000],
                'Adj Close': [102, 103, 104]
            }, index=pd.date_range('2022-01-01', periods=3, freq='D'))
            
            mock_download.return_value = sample_data
            
            # Import all necessary modules
            from src.data.fetch_market_data import MarketDataFetcher
            from src.data.data_cleaner import DataCleaner
            from src.data.data_validator import DataValidator
            from src.data.data_storage import DataStorage
            
            # Initialize all components
            fetcher = MarketDataFetcher()
            cleaner = DataCleaner()
            validator = DataValidator()
            storage = DataStorage()
            
            # Execute complete workflow
            # 1. Fetch data
            raw_data = fetcher.fetch_stock_data('RELIANCE.NS', '2022-01-01', '2022-01-03')
            
            # 2. Clean data
            cleaned_data = cleaner.clean_stock_data(raw_data)
            
            # 3. Validate data
            validation_results = validator.validate_stock_data(cleaned_data)
            
            # 4. Store data if validation passes
            if validation_results['is_valid']:
                storage.save_processed_data(cleaned_data, 'RELIANCE.NS')
            
            # Assertions
            assert not raw_data.empty
            assert not cleaned_data.empty
            assert validation_results['is_valid'] is True
            assert validation_results['data_quality_score'] > 0.7