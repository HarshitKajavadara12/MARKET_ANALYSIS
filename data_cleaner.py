"""
Market Research System v1.0 - Data Cleaning Utilities
Created: 2022
Author: Independent Market Researcher

This module provides comprehensive data cleaning utilities for financial market data.
Handles missing values, outliers, duplicates, and data quality issues.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from .exceptions import DataCleaningError
from .data_validator import DataValidator

class DataCleaner:
    """
    Comprehensive data cleaning utilities for financial market data
    """
    
    def __init__(self, log_level: str = 'INFO'):
        """
        Initialize the Data Cleaner
        
        Args:
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.validator = DataValidator()
        self.logger = self._setup_logger(log_level)
        self.cleaning_stats = {}
        
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup logger for data cleaning operations"""
        logger = logging.getLogger('DataCleaner')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def clean_ohlcv_data(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """
        Comprehensive cleaning for OHLCV market data
        
        Args:
            data: DataFrame with OHLCV columns
            symbol: Symbol name for logging
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Starting data cleaning for {symbol}")
        original_shape = data.shape
        
        # Initialize cleaning stats
        self.cleaning_stats[symbol] = {
            'original_rows': len(data),
            'operations': [],
            'final_rows': 0,
            'data_quality_score': 0
        }
        
        try:
            # Create a copy to avoid modifying original data
            cleaned_data = data.copy()
            
            # Step 1: Handle missing values
            cleaned_data = self._handle_missing_values(cleaned_data, symbol)
            
            # Step 2: Remove duplicates
            cleaned_data = self._remove_duplicates(cleaned_data, symbol)
            
            # Step 3: Fix OHLC relationships
            cleaned_data = self._fix_ohlc_relationships(cleaned_data, symbol)
            
            # Step 4: Handle outliers
            cleaned_data = self._handle_outliers(cleaned_data, symbol)
            
            # Step 5: Validate data types
            cleaned_data = self._validate_data_types(cleaned_data, symbol)
            
            # Step 6: Handle zero or negative prices
            cleaned_data = self._handle_invalid_prices(cleaned_data, symbol)
            
            # Step 7: Sort by date
            cleaned_data = self._sort_by_date(cleaned_data, symbol)
            
            # Step 8: Calculate data quality score
            quality_score = self._calculate_quality_score(cleaned_data, original_shape[0])
            self.cleaning_stats[symbol]['data_quality_score'] = quality_score
            self.cleaning_stats[symbol]['final_rows'] = len(cleaned_data)
            
            self.logger.info(f"Data cleaning completed for {symbol}. "
                           f"Rows: {original_shape[0]} -> {len(cleaned_data)}, "
                           f"Quality Score: {quality_score:.2f}")
            
            return cleaned_data
            
        except Exception as e:
            error_msg = f"Data cleaning failed for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise DataCleaningError(error_msg)
    
    def _handle_missing_values(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        missing_before = data.isnull().sum().sum()
        
        if missing_before == 0:
            return data
            
        self.logger.info(f"Handling {missing_before} missing values for {symbol}")
        
        # Check which columns have missing values
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        for col in missing_cols:
            missing_count = data[col].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            
            self.logger.debug(f"{col}: {missing_count} missing values ({missing_pct:.1f}%)")
            
            # Handle different types of missing values
            if col in ['Open', 'High', 'Low', 'Close']:
                # Forward fill then backward fill for price data
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                
            elif col == 'Volume':
                # Fill with median volume for the dataset
                median_volume = data[col].median()
                data[col] = data[col].fillna(median_volume)
                
            else:
                # For other columns, use forward fill
                data[col] = data[col].fillna(method='ffill')
        
        # Remove rows that still have missing critical values
        critical_cols = ['Open', 'High', 'Low', 'Close']
        data = data.dropna(subset=critical_cols)
        
        missing_after = data.isnull().sum().sum()
        removed_rows = missing_before - missing_after
        
        self.cleaning_stats[symbol]['operations'].append({
            'operation': 'handle_missing_values',
            'missing_before': missing_before,
            'missing_after': missing_after,
            'rows_affected': removed_rows
        })
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Remove duplicate rows based on date index"""
        duplicates_before = data.duplicated().sum()
        
        if duplicates_before == 0:
            return data
            
        self.logger.info(f"Removing {duplicates_before} duplicate rows for {symbol}")
        
        # Remove duplicates, keeping the last occurrence
        data = data[~data.duplicated(keep='last')]
        
        # If index is datetime, also check for duplicate dates
        if isinstance(data.index, pd.DatetimeIndex):
            date_duplicates = data.index.duplicated().sum()
            if date_duplicates > 0:
                self.logger.warning(f"Found {date_duplicates} duplicate dates for {symbol}")
                data = data[~data.index.duplicated(keep='last')]
        
        self.cleaning_stats[symbol]['operations'].append({
            'operation': 'remove_duplicates',
            'duplicates_removed': duplicates_before
        })
        
        return data
    
    def _fix_ohlc_relationships(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Fix OHLC relationship violations"""
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            return data
        
        violations = 0
        
        # Check and fix High >= max(Open, Close) and High >= Low
        mask_high = (data['High'] < data[['Open', 'Close']].max(axis=1)) | (data['High'] < data['Low'])
        violations += mask_high.sum()
        data.loc[mask_high, 'High'] = data.loc[mask_high, ['Open', 'Close', 'Low']].max(axis=1)
        
        # Check and fix Low <= min(Open, Close) and Low <= High
        mask_low = (data['Low'] > data[['Open', 'Close']].min(axis=1)) | (data['Low'] > data['High'])
        violations += mask_low.sum()
        data.loc[mask_low, 'Low'] = data.loc[mask_low, ['Open', 'Close', 'High']].min(axis=1)
        
        if violations > 0:
            self.logger.info(f"Fixed {violations} OHLC relationship violations for {symbol}")
            
        self.cleaning_stats[symbol]['operations'].append({
            'operation': 'fix_ohlc_relationships',
            'violations_fixed': violations
        })
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, symbol: str, method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers in price and volume data"""
        outliers_removed = 0
        
        # Define columns to check for outliers
        price_cols = ['Open', 'High', 'Low', 'Close']
        volume_cols = ['Volume'] if 'Volume' in data.columns else []
        
        for col in price_cols + volume_cols:
            if col not in data.columns:
                continue
                
            if method == 'iqr':
                outliers_mask = self._detect_outliers_iqr(data[col])
            elif method == 'zscore':
                outliers_mask = self._detect_outliers_zscore(data[col])
            else:
                continue
                
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                self.logger.debug(f"Found {outlier_count} outliers in {col} for {symbol}")
                
                # For price data, use interpolation
                if col in price_cols:
                    data.loc[outliers_mask, col] = np.nan
                    data[col] = data[col].interpolate(method='linear')
                    
                # For volume data, cap at reasonable limits
                elif col in volume_cols:
                    q99 = data[col].quantile(0.99)
                    data.loc[outliers_mask & (data[col] > q99), col] = q99
                    
                outliers_removed += outlier_count
        
        self.cleaning_stats[symbol]['operations'].append({
            'operation': 'handle_outliers',
            'outliers_handled': outliers_removed,
            'method': method
        })
        
        return data
    
    def _detect_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        return pd.Series(z_scores > threshold, index=series.index)
    
    def _validate_data_types(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Ensure proper data types for all columns"""
        type_corrections = 0
        
        # Expected data types for financial data
        expected_types = {
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            'Volume': 'int64',
            'Adj Close': 'float64'
        }
        
        for col, expected_type in expected_types.items():
            if col in data.columns:
                current_type = str(data[col].dtype)
                
                if current_type != expected_type:
                    try:
                        if expected_type == 'int64':
                            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('int64')
                        else:
                            data[col] = pd.to_numeric(data[col], errors='coerce').astype(expected_type)
                        
                        type_corrections += 1
                        self.logger.debug(f"Converted {col} from {current_type} to {expected_type}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to convert {col} to {expected_type}: {str(e)}")
        
        self.cleaning_stats[symbol]['operations'].append({
            'operation': 'validate_data_types',
            'corrections_made': type_corrections
        })
        
        return data
    
    def _handle_invalid_prices(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle zero or negative prices"""
        price_cols = ['Open', 'High', 'Low', 'Close']
        invalid_count = 0
        
        for col in price_cols:
            if col in data.columns:
                # Find invalid prices (zero or negative)
                invalid_mask = (data[col] <= 0)
                invalid_count += invalid_mask.sum()
                
                if invalid_mask.any():
                    self.logger.warning(f"Found {invalid_mask.sum()} invalid prices in {col} for {symbol}")
                    
                    # Replace with forward fill, then backward fill
                    data.loc[invalid_mask, col] = np.nan
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows where all prices are still invalid
        price_data = data[price_cols]
        valid_rows = (price_data > 0).any(axis=1)
        rows_removed = (~valid_rows).sum()
        
        if rows_removed > 0:
            data = data[valid_rows]
            self.logger.warning(f"Removed {rows_removed} rows with all invalid prices for {symbol}")
        
        self.cleaning_stats[symbol]['operations'].append({
            'operation': 'handle_invalid_prices',
            'invalid_prices_fixed': invalid_count,
            'rows_removed': rows_removed
        })
        
        return data
    
    def _sort_by_date(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Sort data by date index"""
        if isinstance(data.index, pd.DatetimeIndex):
            if not data.index.is_monotonic_increasing:
                data = data.sort_index()
                self.logger.debug(f"Sorted data by date for {symbol}")
                
                self.cleaning_stats[symbol]['operations'].append({
                    'operation': 'sort_by_date',
                    'sorted': True
                })
        
        return data
    
    def _calculate_quality_score(self, data: pd.DataFrame, original_rows: int) -> float:
        """Calculate data quality score (0-100)"""
        if len(data) == 0:
            return 0.0
        
        # Factors for quality score
        completeness = len(data) / original_rows  # Data retention rate
        consistency = 1.0  # Start with perfect consistency
        
        # Check for remaining missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        consistency -= missing_ratio * 0.5
        
        # Check OHLC relationships
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            ohlc_violations = (
                (data['High'] < data[['Open', 'Close']].max(axis=1)).sum() +
                (data['Low'] > data[['Open', 'Close']].min(axis=1)).sum()
            )
            violation_ratio = ohlc_violations / len(data)
            consistency -= violation_ratio * 0.3
        
        # Overall quality score
        quality_score = (completeness * 0.6 + consistency * 0.4) * 100
        return max(0, min(100, quality_score))
    
    def clean_economic_data(self, data: pd.DataFrame, indicator: str = "UNKNOWN") -> pd.DataFrame:
        """
        Clean economic indicator data for Indian market
        
        Args:
            data: DataFrame with economic data
            indicator: Indicator name for logging
            
        Returns:
            Cleaned DataFrame with economic data
        """
        self.logger.info(f"Starting economic data cleaning for {indicator}")
        original_shape = data.shape
        
        # Initialize cleaning stats for economic data
        self.cleaning_stats[indicator] = {
            'original_rows': len(data),
            'operations': [],
            'final_rows': 0,
            'data_quality_score': 0
        }
        
        try:
            cleaned_data = data.copy()
            
            # Step 1: Handle missing values in economic data
            cleaned_data = self._handle_economic_missing_values(cleaned_data, indicator)
            
            # Step 2: Remove duplicates
            cleaned_data = self._remove_duplicates(cleaned_data, indicator)
            
            # Step 3: Handle economic data outliers (different from OHLCV)
            cleaned_data = self._handle_economic_outliers(cleaned_data, indicator)
            
            # Step 4: Validate economic data types
            cleaned_data = self._validate_economic_data_types(cleaned_data, indicator)
            
            # Step 5: Handle negative values for certain indicators
            cleaned_data = self._handle_economic_negative_values(cleaned_data, indicator)
            
            # Step 6: Sort by date
            cleaned_data = self._sort_by_date(cleaned_data, indicator)
            
            # Step 7: Calculate quality score
            quality_score = self._calculate_quality_score(cleaned_data, original_shape[0])
            self.cleaning_stats[indicator]['data_quality_score'] = quality_score
            self.cleaning_stats[indicator]['final_rows'] = len(cleaned_data)
            
            self.logger.info(f"Economic data cleaning completed for {indicator}. "
                           f"Rows: {original_shape[0]} -> {len(cleaned_data)}, "
                           f"Quality Score: {quality_score:.2f}")
            
            return cleaned_data
            
        except Exception as e:
            error_msg = f"Economic data cleaning failed for {indicator}: {str(e)}"
            self.logger.error(error_msg)
            raise DataCleaningError(error_msg)
    
    def _handle_economic_missing_values(self, data: pd.DataFrame, indicator: str) -> pd.DataFrame:
        """Handle missing values in economic data - specific to Indian indicators"""
        missing_before = data.isnull().sum().sum()
        
        if missing_before == 0:
            return data
            
        self.logger.info(f"Handling {missing_before} missing values in economic data for {indicator}")
        
        # Indian economic indicators and their handling strategies
        indian_indicators = {
            'GDP': 'interpolate',  # GDP growth rate
            'CPI': 'forward_fill',  # Consumer Price Index
            'WPI': 'forward_fill',  # Wholesale Price Index
            'IIP': 'interpolate',   # Index of Industrial Production
            'REPO_RATE': 'forward_fill',  # RBI Repo Rate
            'NIFTY': 'drop',       # Stock indices - no interpolation
            'SENSEX': 'drop',
            'INR_USD': 'interpolate',  # Currency exchange rate
            'FISCAL_DEFICIT': 'interpolate'
        }
        
        for col in data.columns:
            if data[col].isnull().any():
                missing_count = data[col].isnull().sum()
                missing_pct = (missing_count / len(data)) * 100
                
                self.logger.debug(f"{col}: {missing_count} missing values ({missing_pct:.1f}%)")
                
                # Determine handling strategy
                strategy = 'forward_fill'  # default
                for ind, strat in indian_indicators.items():
                    if ind.lower() in col.lower():
                        strategy = strat
                        break
                
                # Apply strategy
                if strategy == 'interpolate':
                    data[col] = data[col].interpolate(method='linear')
                elif strategy == 'forward_fill':
                    data[col] = data[col].fillna(method='ffill')
                elif strategy == 'drop':
                    # For critical indicators, drop rows with missing values
                    data = data.dropna(subset=[col])
        
        missing_after = data.isnull().sum().sum()
        self.cleaning_stats[indicator]['operations'].append({
            'operation': 'handle_economic_missing_values',
            'missing_before': missing_before,
            'missing_after': missing_after
        })
        
        return data
    
    def _handle_economic_outliers(self, data: pd.DataFrame, indicator: str) -> pd.DataFrame:
        """Handle outliers in economic data with Indian market context"""
        outliers_handled = 0
        
        # Economic indicators have different outlier thresholds
        outlier_thresholds = {
            'GDP': 3.0,  # GDP growth can be volatile
            'CPI': 2.5,  # Inflation should be relatively stable
            'WPI': 2.5,
            'IIP': 3.0,  # Industrial production can be volatile
            'REPO_RATE': 2.0,  # Central bank rates are usually stable
            'NIFTY': 4.0,  # Stock indices can be very volatile
            'SENSEX': 4.0,
            'INR_USD': 3.0  # Currency can be volatile
        }
        
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                # Determine threshold
                threshold = 3.0  # default
                for ind, thresh in outlier_thresholds.items():
                    if ind.lower() in col.lower():
                        threshold = thresh
                        break
                
                # Detect outliers using modified Z-score
                outliers_mask = self._detect_outliers_zscore(data[col], threshold)
                outlier_count = outliers_mask.sum()
                
                if outlier_count > 0:
                    self.logger.debug(f"Found {outlier_count} outliers in {col}")
                    
                    # For economic data, use median replacement for extreme outliers
                    # and interpolation for moderate outliers
                    extreme_outliers = self._detect_outliers_zscore(data[col], threshold + 1.0)
                    moderate_outliers = outliers_mask & ~extreme_outliers
                    
                    # Replace extreme outliers with median
                    if extreme_outliers.any():
                        median_val = data[col].median()
                        data.loc[extreme_outliers, col] = median_val
                    
                    # Interpolate moderate outliers
                    if moderate_outliers.any():
                        data.loc[moderate_outliers, col] = np.nan
                        data[col] = data[col].interpolate(method='linear')
                    
                    outliers_handled += outlier_count
        
        self.cleaning_stats[indicator]['operations'].append({
            'operation': 'handle_economic_outliers',
            'outliers_handled': outliers_handled
        })
        
        return data
    
    def _validate_economic_data_types(self, data: pd.DataFrame, indicator: str) -> pd.DataFrame:
        """Validate and correct data types for economic indicators"""
        type_corrections = 0
        
        # Economic data types mapping
        for col in data.columns:
            if col.lower() in ['date', 'time', 'period']:
                continue  # Skip date columns
                
            # Most economic indicators should be numeric
            if not pd.api.types.is_numeric_dtype(data[col]):
                try:
                    # Try to convert to numeric
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    type_corrections += 1
                    self.logger.debug(f"Converted {col} to numeric type")
                except Exception as e:
                    self.logger.warning(f"Failed to convert {col} to numeric: {str(e)}")
        
        self.cleaning_stats[indicator]['operations'].append({
            'operation': 'validate_economic_data_types',
            'corrections_made': type_corrections
        })
        
        return data
    
    def _handle_economic_negative_values(self, data: pd.DataFrame, indicator: str) -> pd.DataFrame:
        """Handle negative values in economic data where appropriate"""
        corrections = 0
        
        # Indicators that should not have negative values
        non_negative_indicators = ['CPI', 'WPI', 'IIP', 'NIFTY', 'SENSEX']
        
        # Indicators that can have negative values
        can_be_negative = ['GDP', 'FISCAL_DEFICIT', 'TRADE_BALANCE']
        
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                negative_mask = data[col] < 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    # Check if this indicator should allow negative values
                    should_be_non_negative = any(
                        ind.lower() in col.lower() for ind in non_negative_indicators
                    )
                    
                    if should_be_non_negative:
                        self.logger.warning(f"Found {negative_count} negative values in {col} - replacing with interpolation")
                        
                        # Replace negative values with NaN and interpolate
                        data.loc[negative_mask, col] = np.nan
                        data[col] = data[col].interpolate(method='linear')
                        
                        # If still negative after interpolation, use absolute value
                        still_negative = data[col] < 0
                        if still_negative.any():
                            data.loc[still_negative, col] = data.loc[still_negative, col].abs()
                        
                        corrections += negative_count
        
        self.cleaning_stats[indicator]['operations'].append({
            'operation': 'handle_economic_negative_values',
            'corrections_made': corrections
        })
        
        return data
    
    def clean_indian_stock_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Specialized cleaning for Indian stock market data (NSE/BSE)
        
        Args:
            data: DataFrame with Indian stock OHLCV data
            symbol: Stock symbol (e.g., 'RELIANCE.NSE', 'TCS.BSE')
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Starting Indian stock data cleaning for {symbol}")
        
        # Use base OHLCV cleaning
        cleaned_data = self.clean_ohlcv_data(data, symbol)
        
        # Indian market specific adjustments
        cleaned_data = self._handle_indian_market_specifics(cleaned_data, symbol)
        
        return cleaned_data
    
    def _handle_indian_market_specifics(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle Indian stock market specific issues"""
        adjustments = 0
        
        # Indian market trading hours: 9:15 AM to 3:30 PM IST
        # Handle any timestamps outside trading hours
        if isinstance(data.index, pd.DatetimeIndex):
            # Filter for Indian market holidays and weekends
            # Keep only Monday to Friday
            weekday_mask = data.index.weekday < 5
            weekend_rows = (~weekday_mask).sum()
            
            if weekend_rows > 0:
                self.logger.info(f"Removing {weekend_rows} weekend rows for {symbol}")
                data = data[weekday_mask]
                adjustments += weekend_rows
        
        # Handle stock splits and bonus issues (basic detection)
        if 'Volume' in data.columns and len(data) > 1:
            # Detect potential stock splits (volume spike + price drop)
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Price_Change'] = data['Close'].pct_change()
            
            # Potential split: volume > 5x average and price drop > 30%
            split_mask = (
                (data['Volume'] > 5 * data['Volume_MA']) & 
                (data['Price_Change'] < -0.3)
            )
            
            split_events = split_mask.sum()
            if split_events > 0:
                self.logger.warning(f"Detected {split_events} potential stock split events for {symbol}")
            
            # Clean up temporary columns
            data = data.drop(['Volume_MA', 'Price_Change'], axis=1)
        
        # Handle Indian currency rounding (paisa to rupee)
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in data.columns:
                # Round to 2 decimal places (paisa precision)
                data[col] = data[col].round(2)
        
        if adjustments > 0:
            self.cleaning_stats[symbol]['operations'].append({
                'operation': 'indian_market_specifics',
                'adjustments_made': adjustments
            })
        
        return data
    
    def get_cleaning_report(self, symbol: str = None) -> Dict:
        """
        Get comprehensive cleaning report
        
        Args:
            symbol: Specific symbol to get report for, or None for all symbols
            
        Returns:
            Dictionary with cleaning statistics and quality metrics
        """
        if symbol:
            if symbol in self.cleaning_stats:
                return {symbol: self.cleaning_stats[symbol]}
            else:
                return {symbol: "No cleaning stats available"}
        else:
            return self.cleaning_stats.copy()
    
    def export_cleaning_log(self, filepath: str, symbol: str = None) -> None:
        """
        Export cleaning operations log to file
        
        Args:
            filepath: Path to save the log file
            symbol: Specific symbol to export, or None for all symbols
        """
        try:
            report = self.get_cleaning_report(symbol)
            
            with open(filepath, 'w') as f:
                f.write("=" * 50 + "\n")
                f.write("MARKET DATA CLEANING REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                for sym, stats in report.items():
                    if isinstance(stats, dict):
                        f.write(f"Symbol: {sym}\n")
                        f.write(f"Original Rows: {stats['original_rows']}\n")
                        f.write(f"Final Rows: {stats['final_rows']}\n")
                        f.write(f"Data Quality Score: {stats['data_quality_score']:.2f}/100\n")
                        f.write("\nOperations Performed:\n")
                        
                        for i, op in enumerate(stats['operations'], 1):
                            f.write(f"  {i}. {op['operation']}:\n")
                            for key, value in op.items():
                                if key != 'operation':
                                    f.write(f"     {key}: {value}\n")
                        f.write("\n" + "-" * 30 + "\n\n")
            
            self.logger.info(f"Cleaning log exported to {filepath}")
            
        except Exception as e:
            error_msg = f"Failed to export cleaning log: {str(e)}"
            self.logger.error(error_msg)
            raise DataCleaningError(error_msg)
    
    def reset_stats(self) -> None:
        """Reset all cleaning statistics"""
        self.cleaning_stats.clear()
        self.logger.info("Cleaning statistics reset")


# Exception class for data cleaning errors
class DataCleaningError(Exception):
    """Custom exception for data cleaning operations"""
    pass