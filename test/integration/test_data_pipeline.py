#!/usr/bin/env python3
"""
Integration Test - End-to-End Data Pipeline
Market Research System v1.0 - 2022
Tests complete data flow from fetching to storage

This module tests the entire data pipeline including:
- Data fetching from multiple sources
- Data cleaning and validation
- Data transformation and storage
- Error handling and recovery
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from data.fetch_market_data import IndianMarketDataFetcher
from data.fetch_economic_data import EconomicDataFetcher
from data.data_cleaner import DataCleaner
from data.data_validator import DataValidator
from data.data_transformer import DataTransformer
from data.data_storage import DataStorage
from utils.date_utils import DateUtils
from utils.logging_utils import setup_logger
from . import TEST_SYMBOLS, TEST_INDICES, TEST_ECONOMIC_INDICATORS, TEST_START_DATE, TEST_END_DATE


class TestDataPipeline(unittest.TestCase):
    """Test complete data pipeline integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.logger = setup_logger('test_data_pipeline', 'test_data_pipeline.log')
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_symbols = TEST_SYMBOLS[:5]  # Use subset for faster testing
        cls.test_indices = TEST_INDICES
        cls.start_date = TEST_START_DATE
        cls.end_date = '2022-06-30'  # Use shorter range for testing
        
        # Initialize components
        cls.market_fetcher = IndianMarketDataFetcher()
        cls.economic_fetcher = EconomicDataFetcher()
        cls.cleaner = DataCleaner()
        cls.validator = DataValidator()
        cls.transformer = DataTransformer()
        cls.storage = DataStorage(base_path=cls.temp_dir)
        
        cls.logger.info("Test environment set up successfully")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        cls.logger.info("Test environment cleaned up")
    
    def test_01_complete_stock_data_pipeline(self):
        """Test complete stock data pipeline"""
        self.logger.info("Testing complete stock data pipeline")
        
        pipeline_results = {}
        
        for symbol in self.test_symbols:
            try:
                # Step 1: Fetch raw data
                self.logger.info(f"Fetching data for {symbol}")
                raw_data = self.market_fetcher.fetch_stock_data(
                    symbol, self.start_date, self.end_date
                )
                self.assertIsNotNone(raw_data, f"Failed to fetch data for {symbol}")
                self.assertGreater(len(raw_data), 0, f"Empty data for {symbol}")
                
                # Step 2: Clean data
                self.logger.info(f"Cleaning data for {symbol}")
                cleaned_data = self.cleaner.clean_stock_data(raw_data)
                self.assertIsInstance(cleaned_data, pd.DataFrame)
                
                # Step 3: Validate data
                self.logger.info(f"Validating data for {symbol}")
                validation_result = self.validator.validate_stock_data(cleaned_data)
                self.assertTrue(validation_result['is_valid'], 
                              f"Data validation failed for {symbol}: {validation_result['errors']}")
                
                # Step 4: Transform data
                self.logger.info(f"Transforming data for {symbol}")
                transformed_data = self.transformer.add_technical_indicators(cleaned_data)
                self.assertGreater(len(transformed_data.columns), len(cleaned_data.columns),
                                 f"Technical indicators not added for {symbol}")
                
                # Step 5: Store data
                self.logger.info(f"Storing data for {symbol}")
                storage_path = self.storage.store_stock_data(transformed_data, symbol)
                self.assertTrue(os.path.exists(storage_path), f"Data not stored for {symbol}")
                
                # Step 6: Verify stored data
                stored_data = pd.read_csv(storage_path)
                self.assertEqual(len(stored_data), len(transformed_data),
                               f"Stored data length mismatch for {symbol}")
                
                pipeline_results[symbol] = {
                    'raw_rows': len(raw_data),
                    'cleaned_rows': len(cleaned_data),
                    'final_rows': len(transformed_data),
                    'columns': len(transformed_data.columns),
                    'storage_path': storage_path
                }
                
                self.logger.info(f"Pipeline completed successfully for {symbol}")
                
            except Exception as e:
                self.fail(f"Pipeline failed for {symbol}: {str(e)}")
        
        # Verify all symbols processed
        self.assertEqual(len(pipeline_results), len(self.test_symbols),
                        "Not all symbols processed successfully")
        
        self.logger.info(f"Complete stock data pipeline test passed for {len(pipeline_results)} symbols")
    
    def test_02_indices_data_pipeline(self):
        """Test indices data pipeline"""
        self.logger.info("Testing indices data pipeline")
        
        for index in self.test_indices:
            try:
                # Fetch index data
                raw_data = self.market_fetcher.fetch_index_data(
                    index, self.start_date, self.end_date
                )
                self.assertIsNotNone(raw_data, f"Failed to fetch data for {index}")
                
                # Process through pipeline
                cleaned_data = self.cleaner.clean_index_data(raw_data)
                validation_result = self.validator.validate_index_data(cleaned_data)
                self.assertTrue(validation_result['is_valid'],
                              f"Index validation failed for {index}")
                
                transformed_data = self.transformer.add_index_indicators(cleaned_data)
                storage_path = self.storage.store_index_data(transformed_data, index)
                
                self.assertTrue(os.path.exists(storage_path), f"Index data not stored for {index}")
                
                self.logger.info(f"Index pipeline completed for {index}")
                
            except Exception as e:
                self.fail(f"Index pipeline failed for {index}: {str(e)}")
    
    def test_03_economic_data_pipeline(self):
        """Test economic data pipeline"""
        self.logger.info("Testing economic data pipeline")
        
        for indicator in TEST_ECONOMIC_INDICATORS:
            try:
                # Fetch economic data
                raw_data = self.economic_fetcher.fetch_indicator_data(
                    indicator, self.start_date, self.end_date
                )
                
                if raw_data is not None and not raw_data.empty:
                    # Process through pipeline
                    cleaned_data = self.cleaner.clean_economic_data(raw_data)
                    validation_result = self.validator.validate_economic_data(cleaned_data)
                    
                    if validation_result['is_valid']:
                        transformed_data = self.transformer.standardize_economic_data(cleaned_data)
                        storage_path = self.storage.store_economic_data(transformed_data, indicator)
                        
                        self.assertTrue(os.path.exists(storage_path),
                                      f"Economic data not stored for {indicator}")
                        
                        self.logger.info(f"Economic pipeline completed for {indicator}")
                    else:
                        self.logger.warning(f"Economic data validation failed for {indicator}")
                else:
                    self.logger.warning(f"No economic data available for {indicator}")
                    
            except Exception as e:
                self.logger.error(f"Economic pipeline failed for {indicator}: {str(e)}")
    
    def test_04_data_consistency_across_pipeline(self):
        """Test data consistency across pipeline stages"""
        self.logger.info("Testing data consistency across pipeline")
        
        symbol = self.test_symbols[0]
        
        # Fetch and process data
        raw_data = self.market_fetcher.fetch_stock_data(symbol, self.start_date, self.end_date)
        cleaned_data = self.cleaner.clean_stock_data(raw_data)
        transformed_data = self.transformer.add_technical_indicators(cleaned_data)
        
        # Test data consistency
        self.assertEqual(raw_data.index.name, cleaned_data.index.name,
                        "Index name changed during cleaning")
        
        # Check that essential columns are preserved
        essential_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in essential_columns:
            self.assertIn(col, cleaned_data.columns, f"Essential column {col} missing after cleaning")
            self.assertIn(col, transformed_data.columns, f"Essential column {col} missing after transformation")
        
        # Check data types consistency
        for col in essential_columns:
            if col in cleaned_data.columns and col in transformed_data.columns:
                self.assertEqual(cleaned_data[col].dtype.kind, transformed_data[col].dtype.kind,
                               f"Data type changed for {col} during transformation")
        
        self.logger.info("Data consistency test passed")
    
    def test_05_pipeline_error_handling(self):
        """Test pipeline error handling and recovery"""
        self.logger.info("Testing pipeline error handling")
        
        # Test with invalid symbol
        invalid_symbol = 'INVALID.NS'
        
        try:
            raw_data = self.market_fetcher.fetch_stock_data(
                invalid_symbol, self.start_date, self.end_date
            )
            if raw_data is None or raw_data.empty:
                self.logger.info("Correctly handled invalid symbol")
            else:
                self.logger.warning("Invalid symbol returned data unexpectedly")
        except Exception as e:
            self.logger.info(f"Exception properly caught for invalid symbol: {str(e)}")
        
        # Test with invalid date range
        try:
            future_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
            raw_data = self.market_fetcher.fetch_stock_data(
                self.test_symbols[0], future_date, future_date
            )
            if raw_data is None or raw_data.empty:
                self.logger.info("Correctly handled future date range")
        except Exception as e:
            self.logger.info(f"Exception properly caught for invalid date range: {str(e)}")
        
        self.logger.info("Error handling test completed")
    
    def test_06_pipeline_performance(self):
        """Test pipeline performance metrics"""
        self.logger.info("Testing pipeline performance")
        
        symbol = self.test_symbols[0]
        start_time = time.time()
        
        # Run complete pipeline
        raw_data = self.market_fetcher.fetch_stock_data(symbol, self.start_date, self.end_date)
        cleaned_data = self.cleaner.clean_stock_data(raw_data)
        validation_result = self.validator.validate_stock_data(cleaned_data)
        transformed_data = self.transformer.add_technical_indicators(cleaned_data)
        storage_path = self.storage.store_stock_data(transformed_data, symbol)
        
        end_time = time.time()
        pipeline_duration = end_time - start_time
        
        # Performance assertions
        self.assertLess(pipeline_duration, 60, "Pipeline took too long (>60 seconds)")
        
        # Memory usage check
        data_memory_mb = transformed_data.memory_usage(deep=True).sum() / 1024 / 1024
        self.assertLess(data_memory_mb, 100, "Data using too much memory (>100MB)")
        
        self.logger.info(f"Pipeline performance: {pipeline_duration:.2f}s, {data_memory_mb:.2f}MB")
    
    def test_07_concurrent_pipeline_execution(self):
        """Test concurrent pipeline execution"""
        self.logger.info("Testing concurrent pipeline execution")
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        threads = []
        
        def process_symbol(symbol):
            try:
                raw_data = self.market_fetcher.fetch_stock_data(symbol, self.start_date, self.end_date)
                if raw_data is not None and not raw_data.empty:
                    cleaned_data = self.cleaner.clean_stock_data(raw_data)
                    results_queue.put((symbol, 'success', len(cleaned_data)))
                else:
                    results_queue.put((symbol, 'no_data', 0))
            except Exception as e:
                results_queue.put((symbol, 'error', str(e)))
        
        # Start threads for multiple symbols
        for symbol in self.test_symbols[:3]:  # Test with 3 symbols
            thread = threading.Thread(target=process_symbol, args=(symbol,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=120)  # 2-minute timeout
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify results
        self.assertEqual(len(results), 3, "Not all concurrent processes completed")
        
        success_count = sum(1 for r in results if r[1] == 'success')
        self.assertGreater(success_count, 0, "No concurrent processes succeeded")
        
        self.logger.info(f"Concurrent execution test: {success_count}/3 successful")


if __name__ == '__main__':
    # Set up test environment
    os.makedirs('test_reports', exist_ok=True)
    os.makedirs('test_logs', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)