"""
Market Research System v1.0 - Validation Utilities
Author: Independent Market Researcher
Created: 2022
Updated: 2025

This module provides comprehensive validation utilities for market data,
ensuring data quality and integrity throughout the research pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import re
import warnings

# Indian Stock Market specific validations
NSE_SYMBOLS_PATTERN = r'^[A-Z0-9&-]{1,10}$'
BSE_SYMBOLS_PATTERN = r'^[0-9]{6}$'
INDIAN_EXCHANGES = ['NSE', 'BSE']
TRADING_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class MarketDataValidator:
    """
    Comprehensive market data validation class for Indian stock market data
    """
    
    def __init__(self):
        self.validation_results = {}
        self.error_log = []
        
    def validate_stock_symbol(self, symbol: str, exchange: str = 'NSE') -> bool:
        """
        Validate Indian stock symbols based on exchange
        
        Args:
            symbol (str): Stock symbol to validate
            exchange (str): Exchange name (NSE/BSE)
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not isinstance(symbol, str):
                return False
                
            symbol = symbol.upper().strip()
            
            if exchange.upper() == 'NSE':
                return bool(re.match(NSE_SYMBOLS_PATTERN, symbol))
            elif exchange.upper() == 'BSE':
                return bool(re.match(BSE_SYMBOLS_PATTERN, symbol))
            else:
                return False
                
        except Exception as e:
            self.error_log.append(f"Symbol validation error: {str(e)}")
            return False
    
    def validate_price_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLCV price data for anomalies and consistency
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            Dict: Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Required columns check
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                results['errors'].append(f"Missing columns: {missing_cols}")
                results['is_valid'] = False
                return results
            
            # Data type validation
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception:
                        results['errors'].append(f"Column {col} cannot be converted to numeric")
                        results['is_valid'] = False
            
            # Check for negative prices
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    results['errors'].append(f"Found {negative_count} negative values in {col}")
                    results['is_valid'] = False
            
            # Check for zero prices
            for col in price_cols:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    results['warnings'].append(f"Found {zero_count} zero values in {col}")
            
            # OHLC consistency check
            ohlc_issues = 0
            ohlc_issues += (df['High'] < df['Low']).sum()
            ohlc_issues += (df['High'] < df['Open']).sum()
            ohlc_issues += (df['High'] < df['Close']).sum()
            ohlc_issues += (df['Low'] > df['Open']).sum()
            ohlc_issues += (df['Low'] > df['Close']).sum()
            
            if ohlc_issues > 0:
                results['errors'].append(f"Found {ohlc_issues} OHLC consistency violations")
                results['is_valid'] = False
            
            # Volume validation
            negative_volume = (df['Volume'] < 0).sum()
            if negative_volume > 0:
                results['errors'].append(f"Found {negative_volume} negative volume values")
                results['is_valid'] = False
            
            # Outlier detection using IQR method
            price_outliers = self._detect_price_outliers(df)
            if price_outliers['count'] > 0:
                results['warnings'].append(f"Detected {price_outliers['count']} price outliers")
                results['stats']['price_outliers'] = price_outliers
            
            # Missing data check
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                results['warnings'].append("Missing data detected")
                results['stats']['missing_data'] = missing_data.to_dict()
            
            # Data completeness check
            results['stats']['total_records'] = len(df)
            results['stats']['date_range'] = {
                'start': str(df.index.min()) if hasattr(df.index, 'min') else 'N/A',
                'end': str(df.index.max()) if hasattr(df.index, 'max') else 'N/A'
            }
            
        except Exception as e:
            results['errors'].append(f"Validation error: {str(e)}")
            results['is_valid'] = False
            
        return results
    
    def validate_trading_hours(self, timestamp: datetime, exchange: str = 'NSE') -> bool:
        """
        Validate if timestamp falls within Indian trading hours
        
        Args:
            timestamp (datetime): Timestamp to validate
            exchange (str): Exchange name
            
        Returns:
            bool: True if within trading hours
        """
        try:
            # NSE trading hours: 9:15 AM to 3:30 PM IST
            if exchange.upper() == 'NSE':
                start_time = timestamp.replace(hour=9, minute=15, second=0, microsecond=0)
                end_time = timestamp.replace(hour=15, minute=30, second=0, microsecond=0)
                
                return (timestamp.weekday() < 5 and  # Monday to Friday
                       start_time <= timestamp <= end_time)
            
            # BSE trading hours: Similar to NSE
            elif exchange.upper() == 'BSE':
                start_time = timestamp.replace(hour=9, minute=15, second=0, microsecond=0)
                end_time = timestamp.replace(hour=15, minute=30, second=0, microsecond=0)
                
                return (timestamp.weekday() < 5 and
                       start_time <= timestamp <= end_time)
            
            return False
            
        except Exception as e:
            self.error_log.append(f"Trading hours validation error: {str(e)}")
            return False
    
    def validate_date_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Validate date range for data fetching
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Dict: Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Check if start date is before end date
            if start_dt >= end_dt:
                results['errors'].append("Start date must be before end date")
                results['is_valid'] = False
            
            # Check if dates are not in future
            today = datetime.now()
            if start_dt > today or end_dt > today:
                results['warnings'].append("Future dates detected")
            
            # Check for reasonable data range (not too old for reliable data)
            earliest_reliable = datetime(2000, 1, 1)
            if start_dt < earliest_reliable:
                results['warnings'].append("Start date is very old, data may be unreliable")
            
            # Check for weekend dates
            if start_dt.weekday() >= 5:
                results['warnings'].append("Start date is a weekend")
            if end_dt.weekday() >= 5:
                results['warnings'].append("End date is a weekend")
                
        except ValueError as e:
            results['errors'].append(f"Invalid date format: {str(e)}")
            results['is_valid'] = False
        except Exception as e:
            results['errors'].append(f"Date validation error: {str(e)}")
            results['is_valid'] = False
            
        return results
    
    def validate_technical_indicators(self, df: pd.DataFrame, indicator_name: str) -> Dict[str, Any]:
        """
        Validate technical indicator calculations
        
        Args:
            df (pd.DataFrame): DataFrame with indicator data
            indicator_name (str): Name of the indicator
            
        Returns:
            Dict: Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            if indicator_name not in df.columns:
                results['errors'].append(f"Indicator {indicator_name} not found in DataFrame")
                results['is_valid'] = False
                return results
            
            indicator_data = df[indicator_name]
            
            # Check for infinite values
            inf_count = np.isinf(indicator_data).sum()
            if inf_count > 0:
                results['errors'].append(f"Found {inf_count} infinite values in {indicator_name}")
                results['is_valid'] = False
            
            # Check for NaN values (expected for initial periods of some indicators)
            nan_count = indicator_data.isnull().sum()
            if nan_count > len(df) * 0.5:  # More than 50% NaN is suspicious
                results['warnings'].append(f"High NaN count ({nan_count}) in {indicator_name}")
            
            # Indicator-specific validations
            if indicator_name.upper() == 'RSI':
                out_of_range = ((indicator_data < 0) | (indicator_data > 100)).sum()
                if out_of_range > 0:
                    results['errors'].append(f"RSI values out of range (0-100): {out_of_range}")
                    results['is_valid'] = False
            
            elif 'MACD' in indicator_name.upper():
                # MACD can be positive or negative, but check for extreme values
                extreme_values = (np.abs(indicator_data) > 1000).sum()
                if extreme_values > 0:
                    results['warnings'].append(f"Extreme MACD values detected: {extreme_values}")
            
            results['stats'] = {
                'mean': float(indicator_data.mean()),
                'std': float(indicator_data.std()),
                'min': float(indicator_data.min()),
                'max': float(indicator_data.max()),
                'null_count': int(nan_count)
            }
            
        except Exception as e:
            results['errors'].append(f"Indicator validation error: {str(e)}")
            results['is_valid'] = False
            
        return results
    
    def _detect_price_outliers(self, df: pd.DataFrame, method: str = 'IQR') -> Dict[str, Any]:
        """
        Detect price outliers using statistical methods
        
        Args:
            df (pd.DataFrame): Price data
            method (str): Method to use ('IQR' or 'Z-score')
            
        Returns:
            Dict: Outlier detection results
        """
        outliers = {
            'count': 0,
            'indices': [],
            'method': method
        }
        
        try:
            if method == 'IQR':
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col not in df.columns:
                        continue
                        
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                    outliers['indices'].extend(col_outliers)
                    
            elif method == 'Z-score':
                from scipy import stats
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col not in df.columns:
                        continue
                        
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    col_outliers = df[z_scores > 3].index.tolist()
                    outliers['indices'].extend(col_outliers)
            
            outliers['indices'] = list(set(outliers['indices']))  # Remove duplicates
            outliers['count'] = len(outliers['indices'])
            
        except Exception as e:
            outliers['error'] = str(e)
            
        return outliers

def validate_config_file(config_path: str) -> Dict[str, Any]:
    """
    Validate configuration file structure and values
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict: Validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        import yaml
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Required sections
        required_sections = ['data_sources', 'api_keys', 'analysis', 'reporting']
        for section in required_sections:
            if section not in config:
                results['errors'].append(f"Missing required section: {section}")
                results['is_valid'] = False
        
        # Validate API keys section
        if 'api_keys' in config:
            api_keys = config['api_keys']
            if not isinstance(api_keys, dict):
                results['errors'].append("api_keys section must be a dictionary")
                results['is_valid'] = False
        
        # Validate data sources
        if 'data_sources' in config:
            data_sources = config['data_sources']
            if 'symbols' in data_sources:
                symbols = data_sources['symbols']
                if not isinstance(symbols, list):
                    results['errors'].append("symbols must be a list")
                    results['is_valid'] = False
                else:
                    validator = MarketDataValidator()
                    invalid_symbols = []
                    for symbol in symbols:
                        if not validator.validate_stock_symbol(symbol):
                            invalid_symbols.append(symbol)
                    
                    if invalid_symbols:
                        results['warnings'].append(f"Invalid symbols found: {invalid_symbols}")
        
    except FileNotFoundError:
        results['errors'].append(f"Configuration file not found: {config_path}")
        results['is_valid'] = False
    except yaml.YAMLError as e:
        results['errors'].append(f"YAML parsing error: {str(e)}")
        results['is_valid'] = False
    except Exception as e:
        results['errors'].append(f"Configuration validation error: {str(e)}")
        results['is_valid'] = False
    
    return results

def validate_api_response(response_data: Any, api_name: str) -> Dict[str, Any]:
    """
    Validate API response data
    
    Args:
        response_data: API response data
        api_name (str): Name of the API
        
    Returns:
        Dict: Validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        if response_data is None:
            results['errors'].append("API response is None")
            results['is_valid'] = False
            return results
        
        if api_name.lower() == 'yfinance':
            if isinstance(response_data, pd.DataFrame):
                if response_data.empty:
                    results['errors'].append("Empty DataFrame received from yfinance")
                    results['is_valid'] = False
                else:
                    # Validate OHLCV data
                    validator = MarketDataValidator()
                    price_validation = validator.validate_price_data(response_data)
                    if not price_validation['is_valid']:
                        results['errors'].extend(price_validation['errors'])
                        results['is_valid'] = False
                    results['warnings'].extend(price_validation['warnings'])
            
        elif api_name.lower() == 'nsepy':
            if isinstance(response_data, pd.DataFrame):
                if response_data.empty:
                    results['errors'].append("Empty DataFrame received from NSEPy")
                    results['is_valid'] = False
        
        elif api_name.lower() == 'newsapi':
            if isinstance(response_data, dict):
                if 'status' in response_data and response_data['status'] != 'ok':
                    results['errors'].append(f"NewsAPI error: {response_data.get('message', 'Unknown error')}")
                    results['is_valid'] = False
                
                if 'articles' not in response_data:
                    results['errors'].append("No articles found in NewsAPI response")
                    results['is_valid'] = False
                elif len(response_data['articles']) == 0:
                    results['warnings'].append("Empty articles list in NewsAPI response")
        
    except Exception as e:
        results['errors'].append(f"API response validation error: {str(e)}")
        results['is_valid'] = False
    
    return results

# Utility functions for common validations
def is_trading_day(date: datetime) -> bool:
    """Check if given date is a trading day (Monday-Friday, excluding holidays)"""
    return date.weekday() < 5

def is_market_open(timestamp: datetime = None) -> bool:
    """Check if Indian market is currently open"""
    if timestamp is None:
        timestamp = datetime.now()
    
    validator = MarketDataValidator()
    return validator.validate_trading_hours(timestamp, 'NSE')

def sanitize_symbol(symbol: str) -> str:
    """Sanitize stock symbol for API calls"""
    if not isinstance(symbol, str):
        raise DataValidationError("Symbol must be a string")
    
    return symbol.upper().strip().replace(" ", "")

def validate_percentage(value: float, min_val: float = -100, max_val: float = 1000) -> bool:
    """Validate percentage values (returns, changes, etc.)"""
    return isinstance(value, (int, float)) and min_val <= value <= max_val