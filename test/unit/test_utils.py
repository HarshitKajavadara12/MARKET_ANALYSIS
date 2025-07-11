"""
Unit Tests for Utility Functions
Market Research System v1.0 (2022)
Focus: Indian Stock Market

Tests for date utilities, file utilities, validation utilities, 
mathematical utilities, and other helper functions
"""

import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from utils.date_utils import DateUtility
from utils.file_utils import FileUtility
from utils.math_utils import MathUtility
from utils.string_utils import StringUtility
from utils.validation_utils import ValidationUtility
from utils.logging_utils import LoggingUtility


class TestDateUtility(unittest.TestCase):
    """Test cases for date utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.date_util = DateUtility()
        self.test_date = datetime(2022, 6, 15, 10, 30, 0)  # Wednesday
    
    def test_is_market_day(self):
        """Test market day validation for NSE"""
        # Test weekday (should be market day)
        monday = datetime(2022, 6, 13)  # Monday
        self.assertTrue(self.date_util.is_nse_market_day(monday))
        
        # Test weekend (should not be market day)
        saturday = datetime(2022, 6, 11)  # Saturday
        self.assertFalse(self.date_util.is_nse_market_day(saturday))
        
        sunday = datetime(2022, 6, 12)  # Sunday
        self.assertFalse(self.date_util.is_nse_market_day(sunday))
    
    def test_is_market_hours(self):
        """Test market hours validation for NSE"""
        # NSE trading hours: 9:15 AM to 3:30 PM IST
        
        # Test during market hours
        market_time = datetime(2022, 6, 15, 10, 30, 0)  # 10:30 AM
        self.assertTrue(self.date_util.is_nse_market_hours(market_time))
        
        # Test before market opens
        pre_market = datetime(2022, 6, 15, 8, 30, 0)  # 8:30 AM
        self.assertFalse(self.date_util.is_nse_market_hours(pre_market))
        
        # Test after market closes
        post_market = datetime(2022, 6, 15, 16, 30, 0)  # 4:30 PM
        self.assertFalse(self.date_util.is_nse_market_hours(post_market))
    
    def test_get_market_holidays(self):
        """Test NSE market holidays retrieval"""
        holidays_2022 = self.date_util.get_nse_holidays(2022)
        
        # Check if returns list
        self.assertIsInstance(holidays_2022, list)
        
        # Check if contains major Indian holidays
        holiday_names = [h['name'] for h in holidays_2022]
        self.assertIn('Republic Day', holiday_names)
        self.assertIn('Independence Day', holiday_names)
        self.assertIn('Gandhi Jayanti', holiday_names)
        self.assertIn('Diwali', holiday_names)
    
    def test_get_previous_market_day(self):
        """Test getting previous market day"""
        # Test from Tuesday (should return Monday)
        tuesday = datetime(2022, 6, 14)
        prev_day = self.date_util.get_previous_market_day(tuesday)
        self.assertEqual(prev_day.strftime('%A'), 'Monday')
        
        # Test from Monday (should return Friday)
        monday = datetime(2022, 6, 13)
        prev_day = self.date_util.get_previous_market_day(monday)
        self.assertEqual(prev_day.strftime('%A'), 'Friday')
    
    def test_get_next_market_day(self):
        """Test getting next market day"""
        # Test from Wednesday (should return Thursday)
        wednesday = datetime(2022, 6, 15)
        next_day = self.date_util.get_next_market_day(wednesday)
        self.assertEqual(next_day.strftime('%A'), 'Thursday')
        
        # Test from Friday (should return Monday)
        friday = datetime(2022, 6, 17)
        next_day = self.date_util.get_next_market_day(friday)
        self.assertEqual(next_day.strftime('%A'), 'Monday')
    
    def test_format_date_for_api(self):
        """Test date formatting for various APIs"""
        test_date = datetime(2022, 6, 15)
        
        # Yahoo Finance format
        yahoo_format = self.date_util.format_for_yahoo(test_date)
        self.assertEqual(yahoo_format, '2022-06-15')
        
        # NSE format
        nse_format = self.date_util.format_for_nse(test_date)
        self.assertEqual(nse_format, '15-06-2022')
        
        # BSE format
        bse_format = self.date_util.format_for_bse(test_date)
        self.assertEqual(bse_format, '15/06/2022')
    
    def test_get_quarter_dates(self):
        """Test quarterly date calculations"""
        q1_dates = self.date_util.get_quarter_dates(2022, 1)
        self.assertEqual(q1_dates['start'], datetime(2022, 1, 1))
        self.assertEqual(q1_dates['end'], datetime(2022, 3, 31))
        
        q4_dates = self.date_util.get_quarter_dates(2022, 4)
        self.assertEqual(q4_dates['start'], datetime(2022, 10, 1))
        self.assertEqual(q4_dates['end'], datetime(2022, 12, 31))
    
    def test_get_financial_year_dates(self):
        """Test financial year date calculations (Indian FY: Apr-Mar)"""
        fy_2022 = self.date_util.get_indian_fy_dates(2022)
        self.assertEqual(fy_2022['start'], datetime(2021, 4, 1))
        self.assertEqual(fy_2022['end'], datetime(2022, 3, 31))


class TestFileUtility(unittest.TestCase):
    """Test cases for file utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.file_util = FileUtility()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.txt')
        
        # Create test data
        self.test_data = {
            'stocks': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
            'prices': [2650.50, 3420.75, 1580.25]
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_json(self):
        """Test JSON file operations"""
        json_file = os.path.join(self.temp_dir, 'test.json')
        
        # Save JSON
        self.file_util.save_json(self.test_data, json_file)
        self.assertTrue(os.path.exists(json_file))
        
        # Load JSON
        loaded_data = self.file_util.load_json(json_file)
        self.assertEqual(loaded_data, self.test_data)
    
    def test_save_and_load_csv(self):
        """Test CSV file operations"""
        csv_file = os.path.join(self.temp_dir, 'test.csv')
        
        # Create test DataFrame
        df = pd.DataFrame({
            'symbol': ['RELIANCE.NS', 'TCS.NS'],
            'price': [2650.50, 3420.75]
        })
        
        # Save CSV
        self.file_util.save_csv(df, csv_file)
        self.assertTrue(os.path.exists(csv_file))
        
        # Load CSV
        loaded_df = self.file_util.load_csv(csv_file)
        pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_create_directory_structure(self):
        """Test directory creation"""
        structure = {
            'data': ['raw', 'processed'],
            'reports': ['daily', 'weekly'],
            'logs': []
        }
        
        base_dir = os.path.join(self.temp_dir, 'market_research')
        self.file_util.create_directory_structure(base_dir, structure)
        
        # Check if directories were created
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'data', 'raw')))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'reports', 'daily')))
        self.assertTrue(os.path.exists(os.path.join(base_dir, 'logs')))
    
    def test_file_size_operations(self):
        """Test file size utilities"""
        # Create test file
        with open(self
                  
                  not complted