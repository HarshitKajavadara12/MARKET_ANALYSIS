"""
Data Validation Utilities for Market Research System v1.0
Focus: Indian Stock Market Data Validation
Created: January 2022
Author: Independent Market Researcher
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import re
from .exceptions import DataValidationError, InvalidDataFormatError, MissingDataError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation for Indian stock market data
    Validates price data, volume data, economic indicators, and technical indicators
    """
    
    def __init__(self):
        # Indian stock market specific configurations
        self.market_hours = {
            'start': '09:15',
            'end': '15:30'
        }
        
        # NSE/BSE symbol patterns
        self.nse_symbol_pattern = r'^[A-Z]{1,10}$'
        self.bse_symbol_pattern = r'^[0-9]{6}$'
        
        # Price validation thresholds (in INR)
        self.price_thresholds = {
            'min_price': 0.01,
            'max_price': 100000.0,
            'max_daily_change': 0.20  # 20% circuit breaker
        }
        
        # Volume validation thresholds
        self.volume_thresholds = {
            'min_volume': 0,
            'max_volume': 1e10  # 10 billion shares
        }
        
        # Required columns for different data types
        self.required_columns = {
            'stock_data': ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
            'index_data': ['date', 'index_name', 'open', 'high', 'low', 'close'],
            'economic_data': ['date', 'indicator', 'value'],
            'technical_indicators': ['date', 'symbol', 'indicator_name', 'value']
        }
    
    def validate_stock_data(self, data: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate stock price data for Indian markets
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            Dict with validation results and error messages
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if DataFrame is not empty
            if data.empty:
                raise MissingDataError("Stock data DataFrame is empty")
            
            # Validate required columns
            missing_cols = self._check_required_columns(data, 'stock_data')
            if missing_cols:
                validation_result['errors'].extend(missing_cols)
                validation_result['is_valid'] = False
            
            # Validate data types
            dtype_errors = self._validate_data_types(data)
            if dtype_errors:
                validation_result['errors'].extend(dtype_errors)
                validation_result['is_valid'] = False
            
            # Validate Indian stock symbols
            symbol_errors = self._validate_indian_symbols(data)
            if symbol_errors:
                validation_result['errors'].extend(symbol_errors)
                validation_result['is_valid'] = False
            
            # Validate price data
            price_errors = self._validate_price_data(data)
            if price_errors:
                validation_result['errors'].extend(price_errors)
                validation_result['is_valid'] = False
            
            # Validate volume data
            volume_errors = self._validate_volume_data(data)
            if volume_errors:
                validation_result['errors'].extend(volume_errors)
                validation_result['is_valid'] = False
            
            # Validate OHLC consistency
            ohlc_errors = self._validate_ohlc_consistency(data)
            if ohlc_errors:
                validation_result['warnings'].extend(ohlc_errors)
            
            # Validate date consistency
            date_errors = self._validate_date_consistency(data)
            if date_errors:
                validation_result['errors'].extend(date_errors)
                validation_result['is_valid'] = False
            
            # Check for gaps in data
            gap_warnings = self._check_data_gaps(data)
            if gap_warnings:
                validation_result['warnings'].extend(gap_warnings)
            
            logger.info(f"Stock data validation completed. Valid: {validation_result['is_valid']}")
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Stock data validation failed: {str(e)}")
        
        return validation_result
    
    def validate_economic_data(self, data: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate economic indicator data for Indian economy
        
        Args:
            data: DataFrame with economic data
            
        Returns:
            Dict with validation results and error messages
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            if data.empty:
                raise MissingDataError("Economic data DataFrame is empty")
            
            # Check required columns
            missing_cols = self._check_required_columns(data, 'economic_data')
            if missing_cols:
                validation_result['errors'].extend(missing_cols)
                validation_result['is_valid'] = False
            
            # Validate economic indicators
            indicator_errors = self._validate_economic_indicators(data)
            if indicator_errors:
                validation_result['errors'].extend(indicator_errors)
                validation_result['is_valid'] = False
            
            # Validate numeric values
            numeric_errors = self._validate_numeric_values(data, 'value')
            if numeric_errors:
                validation_result['errors'].extend(numeric_errors)
                validation_result['is_valid'] = False
            
            logger.info(f"Economic data validation completed. Valid: {validation_result['is_valid']}")
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Economic data validation error: {str(e)}")
            logger.error(f"Economic data validation failed: {str(e)}")
        
        return validation_result
    
    def validate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate technical indicator data
        
        Args:
            data: DataFrame with technical indicator data
            
        Returns:
            Dict with validation results and error messages
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            if data.empty:
                raise MissingDataError("Technical indicator data DataFrame is empty")
            
            # Check required columns
            missing_cols = self._check_required_columns(data, 'technical_indicators')
            if missing_cols:
                validation_result['errors'].extend(missing_cols)
                validation_result['is_valid'] = False
            
            # Validate indicator values
            indicator_errors = self._validate_indicator_values(data)
            if indicator_errors:
                validation_result['errors'].extend(indicator_errors)
                validation_result['is_valid'] = False
            
            logger.info(f"Technical indicator validation completed. Valid: {validation_result['is_valid']}")
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Technical indicator validation error: {str(e)}")
            logger.error(f"Technical indicator validation failed: {str(e)}")
        
        return validation_result
    
    def _check_required_columns(self, data: pd.DataFrame, data_type: str) -> List[str]:
        """Check if all required columns are present"""
        errors = []
        required = self.required_columns.get(data_type, [])
        missing = [col for col in required if col not in data.columns]
        
        if missing:
            errors.append(f"Missing required columns for {data_type}: {missing}")
        
        return errors
    
    def _validate_data_types(self, data: pd.DataFrame) -> List[str]:
        """Validate data types for stock data"""
        errors = []
        
        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        if data[col].isna().any():
                            errors.append(f"Column '{col}' contains non-numeric values")
                    except Exception:
                        errors.append(f"Column '{col}' cannot be converted to numeric")
        
        # Check date column
        if 'date' in data.columns:
            try:
                pd.to_datetime(data['date'])
            except Exception:
                errors.append("Date column contains invalid date formats")
        
        return errors
    
    def _validate_indian_symbols(self, data: pd.DataFrame) -> List[str]:
        """Validate Indian stock symbols (NSE/BSE format)"""
        errors = []
        
        if 'symbol' not in data.columns:
            return errors
        
        invalid_symbols = []
        for symbol in data['symbol'].unique():
            if isinstance(symbol, str):
                # Check NSE format (alphabetic)
                if not re.match(self.nse_symbol_pattern, symbol):
                    # Check BSE format (numeric)
                    if not re.match(self.bse_symbol_pattern, symbol):
                        invalid_symbols.append(symbol)
            else:
                invalid_symbols.append(str(symbol))
        
        if invalid_symbols:
            errors.append(f"Invalid Indian stock symbols: {invalid_symbols}")
        
        return errors
    
    def _validate_price_data(self, data: pd.DataFrame) -> List[str]:
        """Validate price data within reasonable bounds"""
        errors = []
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in data.columns:
                # Check for negative prices
                if (data[col] < 0).any():
                    errors.append(f"Negative prices found in column '{col}'")
                
                # Check for unreasonably high prices
                if (data[col] > self.price_thresholds['max_price']).any():
                    errors.append(f"Unreasonably high prices in column '{col}'")
                
                # Check for zero prices (except for new listings)
                if (data[col] == 0).any():
                    errors.append(f"Zero prices found in column '{col}'")
        
        return errors
    
    def _validate_volume_data(self, data: pd.DataFrame) -> List[str]:
        """Validate volume data"""
        errors = []
        
        if 'volume' in data.columns:
            # Check for negative volume
            if (data['volume'] < 0).any():
                errors.append("Negative volume values found")
            
            # Check for unreasonably high volume
            if (data['volume'] > self.volume_thresholds['max_volume']).any():
                errors.append("Unreasonably high volume values found")
        
        return errors
    
    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> List[str]:
        """Validate OHLC price consistency"""
        warnings = []
        price_cols = ['open', 'high', 'low', 'close']
        
        if all(col in data.columns for col in price_cols):
            # High should be >= max(open, close) and >= low
            high_issues = data[(data['high'] < data[['open', 'close']].max(axis=1)) | 
                              (data['high'] < data['low'])]
            if not high_issues.empty:
                warnings.append(f"OHLC inconsistency: High price issues in {len(high_issues)} records")
            
            # Low should be <= min(open, close) and <= high
            low_issues = data[(data['low'] > data[['open', 'close']].min(axis=1)) | 
                             (data['low'] > data['high'])]
            if not low_issues.empty:
                warnings.append(f"OHLC inconsistency: Low price issues in {len(low_issues)} records")
        
        return warnings
    
    def _validate_date_consistency(self, data: pd.DataFrame) -> List[str]:
        """Validate date consistency and format"""
        errors = []
        
        if 'date' in data.columns:
            try:
                dates = pd.to_datetime(data['date'])
                
                # Check for future dates
                future_dates = dates > datetime.now()
                if future_dates.any():
                    errors.append(f"Future dates found: {future_dates.sum()} records")
                
                # Check for very old dates (before 1990 for Indian markets)
                old_dates = dates < datetime(1990, 1, 1)
                if old_dates.any():
                    errors.append(f"Dates before 1990 found: {old_dates.sum()} records")
                
            except Exception as e:
                errors.append(f"Date parsing error: {str(e)}")
        
        return errors
    
    def _check_data_gaps(self, data: pd.DataFrame) -> List[str]:
        """Check for gaps in time series data"""
        warnings = []
        
        if 'date' in data.columns and 'symbol' in data.columns:
            try:
                data_sorted = data.sort_values(['symbol', 'date'])
                
                for symbol in data_sorted['symbol'].unique():
                    symbol_data = data_sorted[data_sorted['symbol'] == symbol]
                    dates = pd.to_datetime(symbol_data['date'])
                    
                    # Check for significant gaps (more than 5 business days)
                    date_diffs = dates.diff()
                    large_gaps = date_diffs > timedelta(days=7)
                    
                    if large_gaps.any():
                        gap_count = large_gaps.sum()
                        warnings.append(f"Data gaps detected for {symbol}: {gap_count} gaps > 7 days")
                        
            except Exception as e:
                warnings.append(f"Gap analysis error: {str(e)}")
        
        return warnings
    
    def _validate_economic_indicators(self, data: pd.DataFrame) -> List[str]:
        """Validate economic indicator names and values"""
        errors = []
        
        # List of valid Indian economic indicators
        valid_indicators = [
            'GDP_GROWTH', 'INFLATION_CPI', 'REPO_RATE', 'CRR', 'SLR',
            'INDUSTRIAL_PRODUCTION', 'UNEMPLOYMENT_RATE', 'FOREX_RESERVES',
            'CURRENT_ACCOUNT_DEFICIT', 'FISCAL_DEFICIT', 'WPI'
        ]
        
        if 'indicator' in data.columns:
            invalid_indicators = data[~data['indicator'].isin(valid_indicators)]['indicator'].unique()
            if len(invalid_indicators) > 0:
                errors.append(f"Invalid economic indicators: {list(invalid_indicators)}")
        
        return errors
    
    def _validate_numeric_values(self, data: pd.DataFrame, column: str) -> List[str]:
        """Validate numeric values in specified column"""
        errors = []
        
        if column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                try:
                    pd.to_numeric(data[column], errors='raise')
                except Exception:
                    errors.append(f"Column '{column}' contains non-numeric values")
            
            # Check for infinite values
            if np.isinf(data[column]).any():
                errors.append(f"Infinite values found in column '{column}'")
        
        return errors
    
    def _validate_indicator_values(self, data: pd.DataFrame) -> List[str]:
        """Validate technical indicator values"""
        errors = []
        
        if 'indicator_name' in data.columns and 'value' in data.columns:
            # Validate RSI values (should be between 0 and 100)
            rsi_data = data[data['indicator_name'] == 'RSI']
            if not rsi_data.empty:
                invalid_rsi = rsi_data[(rsi_data['value'] < 0) | (rsi_data['value'] > 100)]
                if not invalid_rsi.empty:
                    errors.append(f"Invalid RSI values: {len(invalid_rsi)} records")
            
            # Validate MACD histogram values (check for extreme values)
            macd_data = data[data['indicator_name'].str.contains('MACD', na=False)]
            if not macd_data.empty:
                extreme_macd = macd_data[abs(macd_data['value']) > 1000]
                if not extreme_macd.empty:
                    errors.append(f"Extreme MACD values: {len(extreme_macd)} records")
        
        return errors
    
    def validate_data_completeness(self, data: pd.DataFrame, expected_symbols: List[str], 
                                 date_range: Tuple[str, str]) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate data completeness for expected symbols and date range
        
        Args:
            data: DataFrame with stock data
            expected_symbols: List of expected stock symbols
            date_range: Tuple of (start_date, end_date) strings
            
        Returns:
            Dict with completeness validation results
        """
        validation_result = {
            'is_complete': True,
            'missing_symbols': [],
            'missing_dates': [],
            'coverage_percentage': 0.0
        }
        
        try:
            start_date, end_date = date_range
            
            # Check missing symbols
            actual_symbols = set(data['symbol'].unique()) if 'symbol' in data.columns else set()
            expected_symbols_set = set(expected_symbols)
            missing_symbols = expected_symbols_set - actual_symbols
            
            if missing_symbols:
                validation_result['missing_symbols'] = list(missing_symbols)
                validation_result['is_complete'] = False
            
            # Check date coverage
            if 'date' in data.columns:
                actual_dates = set(pd.to_datetime(data['date']).dt.date)
                expected_dates = set(pd.date_range(start_date, end_date, freq='B').date)  # Business days
                missing_dates = expected_dates - actual_dates
                
                if missing_dates:
                    validation_result['missing_dates'] = [str(d) for d in sorted(missing_dates)]
                    validation_result['is_complete'] = False
                
                # Calculate coverage percentage
                coverage = len(actual_dates) / len(expected_dates) * 100 if expected_dates else 0
                validation_result['coverage_percentage'] = round(coverage, 2)
            
            logger.info(f"Data completeness check: {validation_result['coverage_percentage']}% coverage")
            
        except Exception as e:
            validation_result['is_complete'] = False
            validation_result['error'] = str(e)
            logger.error(f"Data completeness validation failed: {str(e)}")
        
        return validation_result