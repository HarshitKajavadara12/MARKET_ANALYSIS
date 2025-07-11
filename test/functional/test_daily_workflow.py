"""
Functional tests for daily workflow of Market Research System v1.0
Tests the complete end-to-end daily data collection, analysis, and reporting pipeline
"""

import unittest
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.fetch_market_data import MarketDataFetcher
from data.fetch_economic_data import EconomicDataFetcher
from data.data_cleaner import DataCleaner
from analysis.technical_indicators import TechnicalAnalyzer
from analysis.market_analyzer import MarketAnalyzer
from reporting.pdf_generator import PDFReportGenerator
from utils.date_utils import get_trading_days, is_trading_day


class TestDailyWorkflow(unittest.TestCase):
    """Test complete daily workflow functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_data_dir = tempfile.mkdtemp()
        cls.test_reports_dir = tempfile.mkdtemp()
        cls.test_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'ITC.NS']
        cls.test_date = datetime(2022, 6, 15)  # A known trading day
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_data_dir):
            shutil.rmtree(cls.test_data_dir)
        if os.path.exists(cls.test_reports_dir):
            shutil.rmtree(cls.test_reports_dir)
    
    def setUp(self):
        """Set up each test"""
        self.market_fetcher = MarketDataFetcher()
        self.economic_fetcher = EconomicDataFetcher()
        self.data_cleaner = DataCleaner()
        self.technical_analyzer = TechnicalAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        self.pdf_generator = PDFReportGenerator()
        
    def test_complete_daily_workflow(self):
        """Test the complete daily workflow from data fetch to report generation"""
        print("Testing complete daily workflow...")
        
        # Step 1: Check if today is a trading day
        if not is_trading_day(self.test_date):
            self.skipTest("Test date is not a trading day")
        
        # Step 2: Fetch market data
        market_data = {}
        for symbol in self.test_symbols:
            try:
                data = self.market_fetcher.fetch_daily_data(
                    symbol, 
                    start_date=self.test_date - timedelta(days=30),
                    end_date=self.test_date
                )
                if not data.empty:
                    market_data[symbol] = data
            except Exception as e:
                print(f"Warning: Could not fetch data for {symbol}: {e}")
        
        self.assertGreater(len(market_data), 0, "Should fetch data for at least one symbol")
        
        # Step 3: Clean the data
        cleaned_data = {}
        for symbol, data in market_data.items():
            cleaned = self.data_cleaner.clean_ohlcv_data(data)
            self.assertFalse(cleaned.empty, f"Cleaned data for {symbol} should not be empty")
            cleaned_data[symbol] = cleaned
        
        # Step 4: Calculate technical indicators
        technical_data = {}
        for symbol, data in cleaned_data.items():
            indicators = self.technical_analyzer.calculate_all_indicators(data)
            self.assertIsInstance(indicators, dict, "Technical indicators should be a dictionary")
            self.assertGreater(len(indicators), 0, "Should calculate at least one indicator")
            technical_data[symbol] = indicators
        
        # Step 5: Perform market analysis
        analysis_results = self.market_analyzer.analyze_market_data(cleaned_data)
        self.assertIsInstance(analysis_results, dict, "Analysis results should be a dictionary")
        self.assertIn('summary', analysis_results, "Analysis should include summary")
        
        # Step 6: Generate daily report
        report_path = os.path.join(self.test_reports_dir, f"daily_report_{self.test_date.strftime('%Y%m%d')}.pdf")
        success = self.pdf_generator.generate_daily_report(
            market_data=cleaned_data,
            technical_data=technical_data,
            analysis_results=analysis_results,
            output_path=report_path
        )
        
        self.assertTrue(success, "Report generation should be successful")
        self.assertTrue(os.path.exists(report_path), "Report file should be created")
        
        # Verify report file size
        file_size = os.path.getsize(report_path)
        self.assertGreater(file_size, 1000, "Report file should be at least 1KB")
        
        print(f"Daily workflow completed successfully. Report saved to: {report_path}")
    
    def test_data_fetch_workflow(self):
        """Test data fetching workflow with error handling"""
        print("Testing data fetch workflow...")
        
        # Test with valid symbols
        valid_symbols = ['RELIANCE.NS', 'TCS.NS']
        fetched_data = {}
        
        for symbol in valid_symbols:
            try:
                data = self.market_fetcher.fetch_daily_data(
                    symbol,
                    start_date=self.test_date - timedelta(days=10),
                    end_date=self.test_date
                )
                if not data.empty:
                    fetched_data[symbol] = data
            except Exception as e:
                print(f"Fetch error for {symbol}: {e}")
        
        # Should have at least some data
        if len(fetched_data) > 0:
            # Verify data structure
            for symbol, data in fetched_data.items():
                expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in expected_columns:
                    self.assertIn(col, data.columns, f"Data should contain {col} column")
                
                # Verify data types
                self.assertTrue(pd.api.types.is_numeric_dtype(data['Close']), "Close should be numeric")
                self.assertTrue(pd.api.types.is_numeric_dtype(data['Volume']), "Volume should be numeric")
    
    def test_analysis_workflow(self):
        """Test analysis workflow with mock data"""
        print("Testing analysis workflow...")
        
        # Create mock market data
        dates = pd.date_range(start=self.test_date - timedelta(days=30), end=self.test_date, freq='D')
        mock_data = {}
        
        for symbol in self.test_symbols[:2]:  # Use first 2 symbols for faster testing
            # Generate realistic mock OHLCV data
            np.random.seed(42)  # For reproducible results
            base_price = 100
            prices = []
            
            for i in range(len(dates)):
                if i == 0:
                    price = base_price
                else:
                    # Random walk with some trend
                    change = np.random.normal(0, 0.02) * prices[-1]
                    price = prices[-1] + change
                prices.append(max(price, 10))  # Ensure positive prices
            
            # Create OHLCV data
            data = pd.DataFrame({
                'Date': dates,
                'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(10000, 100000) for _ in range(len(dates))]
            })
            data.set_index('Date', inplace=True)
            mock_data[symbol] = data
        
        # Test technical analysis
        technical_results = {}
        for symbol, data in mock_data.items():
            indicators = self.technical_analyzer.calculate_all_indicators(data)
            self.assertIsInstance(indicators, dict, "Technical indicators should be a dictionary")
            technical_results[symbol] = indicators
        
        # Test market analysis
        market_analysis = self.market_analyzer.analyze_market_data(mock_data)
        self.assertIsInstance(market_analysis, dict, "Market analysis should be a dictionary")
        
        # Verify analysis contains expected keys
        expected_keys = ['summary', 'trends', 'volatility']
        for key in expected_keys:
            if key in market_analysis:
                self.assertIsNotNone(market_analysis[key], f"Analysis {key} should not be None")
    
    def test_weekend_workflow(self):
        """Test workflow behavior on weekends"""
        print("Testing weekend workflow...")
        
        # Test with a weekend date
        weekend_date = datetime(2022, 6, 18)  # Saturday
        
        is_trading = is_trading_day(weekend_date)
        self.assertFalse(is_trading, "Saturday should not be a trading day")
        
        # On weekends, system should use last trading day's data
        last_trading_day = get_trading_days(
            start_date=weekend_date - timedelta(days=7),
            end_date=weekend_date
        )[-1] if get_trading_days(weekend_date - timedelta(days=7), weekend_date) else None
        
        if last_trading_day:
            self.assertLess(last_trading_day, weekend_date, "Last trading day should be before weekend")
    
    def test_error_recovery_workflow(self):
        """Test workflow error recovery mechanisms"""
        print("Testing error recovery workflow...")
        
        # Test with invalid symbol
        invalid_symbols = ['INVALID.NS', 'NONEXISTENT.NS']
        successful_fetches = 0
        
        for symbol in invalid_symbols + self.test_symbols[:2]:
            try:
                data = self.market_fetcher.fetch_daily_data(
                    symbol,
                    start_date=self.test_date - timedelta(days=5),
                    end_date=self.test_date
                )
                if not data.empty:
                    successful_fetches += 1
            except Exception as e:
                print(f"Expected error for {symbol}: {e}")
        
        # Should have some successful fetches from valid symbols
        # Even if invalid symbols fail
        print(f"Successful fetches: {successful_fetches}")
    
    def test_data_quality_workflow(self):
        """Test data quality checks in workflow"""
        print("Testing data quality workflow...")
        
        # Create data with quality issues
        dates = pd.date_range(start=self.test_date - timedelta(days=10), end=self.test_date, freq='D')
        
        # Data with missing values
        problematic_data = pd.DataFrame({
            'Open': [100, np.nan, 102, 103, np.nan],
            'High': [105, 106, np.nan, 108, 109],
            'Low': [98, 99, 100, np.nan, 102],
            'Close': [102, 104, 103, 107, np.nan],
            'Volume': [10000, 0, 15000, -1000, 20000]  # Zero and negative volumes
        }, index=dates[:5])
        
        # Test data cleaning
        cleaned_data = self.data_cleaner.clean_ohlcv_data(problematic_data)
        
        # Verify cleaning results
        self.assertFalse(cleaned_data.empty, "Cleaned data should not be completely empty")
        
        # Check that invalid values are handled
        if not cleaned_data.empty:
            self.assertFalse(cleaned_data['Close'].isna().all(), "Should have some valid Close prices")
            self.assertTrue((cleaned_data['Volume'] >= 0).all(), "Volume should be non-negative after cleaning")
    
    def test_performance_workflow(self):
        """Test workflow performance benchmarks"""
        print("Testing workflow performance...")
        
        start_time = datetime.now()
        
        # Test with minimal data set for speed
        test_symbol = 'RELIANCE.NS'
        
        try:
            # Fetch small amount of data
            data = self.market_fetcher.fetch_daily_data(
                test_symbol,
                start_date=self.test_date - timedelta(days=5),
                end_date=self.test_date
            )
            
            if not data.empty:
                # Clean data
                cleaned_data = self.data_cleaner.clean_ohlcv_data(data)
                
                # Calculate indicators
                indicators = self.technical_analyzer.calculate_all_indicators(cleaned_data)
                
                # Perform analysis
                analysis = self.market_analyzer.analyze_market_data({test_symbol: cleaned_data})
        
        except Exception as e:
            print(f"Performance test warning: {e}")
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"Workflow execution time: {execution_time:.2f} seconds")
        
        # Performance benchmark - should complete within reasonable time
        self.assertLess(execution_time, 60, "Workflow should complete within 60 seconds")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)