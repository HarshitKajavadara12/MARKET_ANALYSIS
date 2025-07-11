"""
Data Transformation Utilities for Market Research System v1.0
Focus: Indian Stock Market Data Transformation
Created: February 2022
Author: Independent Market Researcher
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from .exceptions import DataTransformationError, InvalidDataFormatError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    """
    Comprehensive data transformation utilities for Indian stock market data
    Handles normalization, aggregation, feature engineering, and format conversions
    """
    
    def __init__(self):
        # Indian market specific configurations
        self.trading_days_per_year = 250
        self.trading_hours_per_day = 6.25  # 9:15 AM to 3:30 PM
        
        # Currency conversion rates (base: INR)
        self.currency_rates = {
            'USD': 75.0,  # Approximate rate for 2022
            'EUR': 85.0,
            'GBP': 95.0,
            'JPY': 0.65
        }
        
        # Market cap categories (in INR Crores)
        self.market_cap_categories = {
            'large_cap': 20000,    # > 20,000 Crores
            'mid_cap': 5000,       # 5,000 - 20,000 Crores
            'small_cap': 5000      # < 5,000 Crores
        }
        
        # Sector mapping for Indian stocks
        self.sector_mapping = {
            'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT', 'TECHM': 'IT',
            'RELIANCE': 'Energy', 'ONGC': 'Energy', 'IOC': 'Energy', 'BPCL': 'Energy',
            'ICICIBANK': 'Banking', 'HDFCBANK': 'Banking', 'SBIN': 'Banking', 'AXISBANK': 'Banking',
            'ITC': 'FMCG', 'HINDUNILVR': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG',
            'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'M&M': 'Auto', 'BAJAJ-AUTO': 'Auto',
            'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma', 'LUPIN': 'Pharma'
        }
    
    def normalize_price_data(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize stock price data using various methods
        
        Args:
            data: DataFrame with OHLCV data
            method: Normalization method ('minmax', 'zscore', 'robust', 'percentage')
            
        Returns:
            DataFrame with normalized price data
        """
        try:
            normalized_data = data.copy()
            price_columns = ['open', 'high', 'low', 'close']
            
            if method == 'minmax':
                # Min-Max normalization (0-1 scale)
                for col in price_columns:
                    if col in normalized_data.columns:
                        min_val = normalized_data[col].min()
                        max_val = normalized_data[col].max()
                        if max_val != min_val:
                            normalized_data[f'{col}_normalized'] = (normalized_data[col] - min_val) / (max_val - min_val)
                        else:
                            normalized_data[f'{col}_normalized'] = 0.5
            
            elif method == 'zscore':
                # Z-score normalization
                for col in price_columns:
                    if col in normalized_data.columns:
                        mean_val = normalized_data[col].mean()
                        std_val = normalized_data[col].std()
                        if std_val != 0:
                            normalized_data[f'{col}_normalized'] = (normalized_data[col] - mean_val) / std_val
                        else:
                            normalized_data[f'{col}_normalized'] = 0
            
            elif method == 'robust':
                # Robust normalization using median and IQR
                for col in price_columns:
                    if col in normalized_data.columns:
                        median_val = normalized_data[col].median()
                        q75 = normalized_data[col].quantile(0.75)
                        q25 = normalized_data[col].quantile(0.25)
                        iqr = q75 - q25
                        if iqr != 0:
                            normalized_data[f'{col}_normalized'] = (normalized_data[col] - median_val) / iqr
                        else:
                            normalized_data[f'{col}_normalized'] = 0
            
            elif method == 'percentage':
                # Percentage change from first value
                for col in price_columns:
                    if col in normalized_data.columns:
                        first_val = normalized_data[col].iloc[0]
                        if first_val != 0:
                            normalized_data[f'{col}_pct'] = ((normalized_data[col] - first_val) / first_val) * 100
                        else:
                            normalized_data[f'{col}_pct'] = 0
            
            logger.info(f"Price data normalized using {method} method")
            return normalized_data
            
        except Exception as e:
            logger.error(f"Price normalization failed: {str(e)}")
            raise DataTransformationError(f"Failed to normalize price data: {str(e)}")
    
    def aggregate_data(self, data: pd.DataFrame, frequency: str = 'D', 
                      aggregation_method: Dict[str, str] = None) -> pd.DataFrame:
        """
        Aggregate stock data to different time frequencies
        
        Args:
            data: DataFrame with datetime index and OHLCV data
            frequency: Aggregation frequency ('D', 'W', 'M', 'Q', 'Y')
            aggregation_method: Dict specifying aggregation method for each column
            
        Returns:
            DataFrame with aggregated data
        """
        try:
            if aggregation_method is None:
                aggregation_method = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
            
            # Ensure datetime index
            if 'date' in data.columns:
                data_copy = data.set_index('date')
            else:
                data_copy = data.copy()
            
            data_copy.index = pd.to_datetime(data_copy.index)
            
            # Group by symbol if present
            if 'symbol' in data_copy.columns:
                aggregated_data = data_copy.groupby('symbol').resample(frequency).agg(aggregation_method)
                aggregated_data = aggregated_data.reset_index()
            else:
                aggregated_data = data_copy.resample(frequency).agg(aggregation_method)
                aggregated_data = aggregated_data.reset_index()
            
            # Remove rows with NaN values (weekends, holidays)
            aggregated_data = aggregated_data.dropna()
            
            logger.info(f"Data aggregated to {frequency} frequency")
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Data aggregation failed: {str(e)}")
            raise DataTransformationError(f"Failed to aggregate data: {str(e)}")
    
    def calculate_returns(self, data: pd.DataFrame, return_type: str = 'simple', 
                         periods: int = 1) -> pd.DataFrame:
        """
        Calculate various types of returns
        
        Args:
            data: DataFrame with price data
            return_type: Type of return ('simple', 'log', 'cumulative')
            periods: Number of periods for return calculation
            
        Returns:
            DataFrame with calculated returns
        """
        try:
            returns_data = data.copy()
            price_col = 'close' if 'close' in data.columns else 'price'
            
            if price_col not in data.columns:
                raise InvalidDataFormatError("No price column found for return calculation")
            
            if return_type == 'simple':
                returns_data['returns'] = returns_data[price_col].pct_change(periods=periods)
            
            elif return_type == 'log':
                returns_data['returns'] = np.log(returns_data[price_col] / returns_data[price_col].shift(periods))
            
            elif return_type == 'cumulative':
                simple_returns = returns_data[price_col].pct_change(periods=periods)
                returns_data['returns'] = (1 + simple_returns).cumprod() - 1
            
            # Calculate additional return metrics
            returns_data['daily_return'] = returns_data[price_col].pct_change()
            returns_data['rolling_volatility'] = returns_data['daily_return'].rolling(window=20).std() * np.sqrt(self.trading_days_per_year)
            
            # Remove infinite and NaN values
            returns_data = returns_data.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"Returns calculated using {return_type} method with {periods} periods")
            return returns_data
            
        except Exception as e:
            logger.error(f"Return calculation failed: {str(e)}")
            raise DataTransformationError(f"Failed to calculate returns: {str(e)}")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical and fundamental features from stock data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            features_data = data.copy()
            
            # Price-based features
            if all(col in data.columns for col in ['high', 'low', 'close']):
                # Price position within daily range
                features_data['price_position'] = (features_data['close'] - features_data['low']) / (features_data['high'] - features_data['low'])
                
                # Daily price range
                features_data['daily_range'] = (features_data['high'] - features_data['low']) / features_data['close']
                
                # Gap up/down from previous close
                if 'open' in data.columns:
                    features_data['gap'] = (features_data['open'] - features_data['close'].shift(1)) / features_data['close'].shift(1)
            
            # Volume-based features
            # Volume-based features
            if 'volume' in data.columns:
                # Volume moving averages
                features_data['volume_ma_5'] = features_data['volume'].rolling(window=5).mean()
                features_data['volume_ma_20'] = features_data['volume'].rolling(window=20).mean()
                
                # Volume ratio (current vs average)
                features_data['volume_ratio'] = features_data['volume'] / features_data['volume_ma_20']
                
                # Price-Volume relationship
                features_data['pv_trend'] = features_data['close'].pct_change() * features_data['volume_ratio']
            
            # Time-based features for Indian market
            if 'date' in data.columns or isinstance(data.index, pd.DatetimeIndex):
                date_col = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['date'])
                
                # Day of week (Monday=0, Friday=4 for trading days)
                features_data['day_of_week'] = date_col.dayofweek
                
                # Month of year (for seasonal patterns)
                features_data['month'] = date_col.month
                
                # Quarter (for quarterly results impact)
                features_data['quarter'] = date_col.quarter
                
                # Days to earnings season (Q1: Apr-May, Q2: Jul-Aug, Q3: Oct-Nov, Q4: Jan-Feb)
                features_data['earnings_season'] = features_data['month'].apply(
                    lambda x: 1 if x in [1, 2, 4, 5, 7, 8, 10, 11] else 0
                )
            
            # Market microstructure features
            if all(col in data.columns for col in ['high', 'low', 'open', 'close']):
                # Intraday momentum
                features_data['intraday_momentum'] = (features_data['close'] - features_data['open']) / features_data['open']
                
                # Upper/Lower shadows (wicks)
                features_data['upper_shadow'] = (features_data['high'] - np.maximum(features_data['open'], features_data['close'])) / features_data['close']
                features_data['lower_shadow'] = (np.minimum(features_data['open'], features_data['close']) - features_data['low']) / features_data['close']
                
                # Body size (candle body)
                features_data['body_size'] = abs(features_data['close'] - features_data['open']) / features_data['close']
            
            # Indian market specific features
            if 'symbol' in data.columns:
                # Add sector information
                features_data['sector'] = features_data['symbol'].map(self.sector_mapping).fillna('Others')
                
                # Market cap category (will be filled by separate method)
                features_data['market_cap_category'] = 'Unknown'
            
            # Lag features (previous day values)
            lag_columns = ['close', 'volume'] if 'volume' in data.columns else ['close']
            for col in lag_columns:
                if col in features_data.columns:
                    for lag in [1, 2, 3, 5]:  # 1, 2, 3, and 5 day lags
                        features_data[f'{col}_lag_{lag}'] = features_data[col].shift(lag)
            
            # Remove infinite and NaN values from features
            features_data = features_data.replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"Created {len(features_data.columns) - len(data.columns)} new features")
            return features_data
            
        except Exception as e:
            logger.error(f"Feature creation failed: {str(e)}")
            raise DataTransformationError(f"Failed to create features: {str(e)}")
    
    def convert_currency(self, data: pd.DataFrame, from_currency: str = 'INR', 
                        to_currency: str = 'USD') -> pd.DataFrame:
        """
        Convert currency for price columns (useful for international comparisons)
        
        Args:
            data: DataFrame with price data in INR
            from_currency: Source currency (default: INR)
            to_currency: Target currency
            
        Returns:
            DataFrame with converted currency values
        """
        try:
            converted_data = data.copy()
            price_columns = ['open', 'high', 'low', 'close', 'adj_close']
            
            if from_currency == 'INR' and to_currency in self.currency_rates:
                conversion_rate = 1 / self.currency_rates[to_currency]
                
                for col in price_columns:
                    if col in converted_data.columns:
                        converted_data[f'{col}_{to_currency}'] = converted_data[col] * conversion_rate
                
                logger.info(f"Currency converted from {from_currency} to {to_currency}")
            
            elif from_currency in self.currency_rates and to_currency == 'INR':
                conversion_rate = self.currency_rates[from_currency]
                
                for col in price_columns:
                    if col in converted_data.columns:
                        converted_data[f'{col}_INR'] = converted_data[col] * conversion_rate
                
                logger.info(f"Currency converted from {from_currency} to INR")
            
            else:
                logger.warning(f"Currency conversion not supported: {from_currency} to {to_currency}")
            
            return converted_data
            
        except Exception as e:
            logger.error(f"Currency conversion failed: {str(e)}")
            raise DataTransformationError(f"Failed to convert currency: {str(e)}")
    
    def resample_data(self, data: pd.DataFrame, target_frequency: str = '1D') -> pd.DataFrame:
        """
        Resample data to different frequencies with proper handling of Indian market hours
        
        Args:
            data: DataFrame with datetime index
            target_frequency: Target frequency ('1min', '5min', '15min', '1H', '1D')
            
        Returns:
            DataFrame with resampled data
        """
        try:
            # Ensure datetime index
            if 'date' in data.columns:
                data_copy = data.set_index('date')
            elif 'datetime' in data.columns:
                data_copy = data.set_index('datetime')
            else:
                data_copy = data.copy()
            
            data_copy.index = pd.to_datetime(data_copy.index)
            
            # Define aggregation rules for OHLCV data
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Filter only existing columns
            available_agg_rules = {k: v for k, v in agg_rules.items() if k in data_copy.columns}
            
            # Handle other columns (take last value)
            for col in data_copy.columns:
                if col not in available_agg_rules:
                    available_agg_rules[col] = 'last'
            
            # Resample data
            if 'symbol' in data_copy.columns:
                resampled_data = data_copy.groupby('symbol').resample(target_frequency).agg(available_agg_rules)
                resampled_data = resampled_data.reset_index()
            else:
                resampled_data = data_copy.resample(target_frequency).agg(available_agg_rules)
                resampled_data = resampled_data.reset_index()
            
            # Remove rows with NaN values (market closed periods)
            resampled_data = resampled_data.dropna(subset=['close'] if 'close' in resampled_data.columns else [resampled_data.columns[1]])
            
            logger.info(f"Data resampled to {target_frequency} frequency")
            return resampled_data
            
        except Exception as e:
            logger.error(f"Data resampling failed: {str(e)}")
            raise DataTransformationError(f"Failed to resample data: {str(e)}")
    
    def clean_outliers(self, data: pd.DataFrame, method: str = 'iqr', 
                      columns: List[str] = None) -> pd.DataFrame:
        """
        Clean outliers from data using various methods
        
        Args:
            data: DataFrame to clean
            method: Outlier detection method ('iqr', 'zscore', 'isolation')
            columns: List of columns to check for outliers
            
        Returns:
            DataFrame with outliers handled
        """
        try:
            cleaned_data = data.copy()
            
            if columns is None:
                columns = ['open', 'high', 'low', 'close', 'volume']
                columns = [col for col in columns if col in data.columns]
            
            if method == 'iqr':
                # Interquartile Range method
                for col in columns:
                    if col in cleaned_data.columns:
                        Q1 = cleaned_data[col].quantile(0.25)
                        Q3 = cleaned_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Cap outliers instead of removing them
                        cleaned_data[col] = np.clip(cleaned_data[col], lower_bound, upper_bound)
            
            elif method == 'zscore':
                # Z-score method
                for col in columns:
                    if col in cleaned_data.columns:
                        z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                        # Cap values with z-score > 3
                        outlier_mask = z_scores > 3
                        if outlier_mask.any():
                            median_val = cleaned_data[col].median()
                            cleaned_data.loc[outlier_mask, col] = median_val
            
            elif method == 'isolation':
                # Simple isolation method (for version 1, keep it basic)
                for col in columns:
                    if col in cleaned_data.columns:
                        # Remove extreme values (top and bottom 0.5%)
                        lower_percentile = cleaned_data[col].quantile(0.005)
                        upper_percentile = cleaned_data[col].quantile(0.995)
                        cleaned_data[col] = np.clip(cleaned_data[col], lower_percentile, upper_percentile)
            
            logger.info(f"Outliers cleaned using {method} method for columns: {columns}")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Outlier cleaning failed: {str(e)}")
            raise DataTransformationError(f"Failed to clean outliers: {str(e)}")
    
    def format_for_analysis(self, data: pd.DataFrame, analysis_type: str = 'technical') -> pd.DataFrame:
        """
        Format data for specific types of analysis
        
        Args:
            data: Raw data DataFrame
            analysis_type: Type of analysis ('technical', 'fundamental', 'sentiment')
            
        Returns:
            DataFrame formatted for specified analysis
        """
        try:
            formatted_data = data.copy()
            
            if analysis_type == 'technical':
                # Ensure required columns for technical analysis
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in formatted_data.columns]
                
                if missing_cols:
                    logger.warning(f"Missing columns for technical analysis: {missing_cols}")
                
                # Add basic technical indicators
                if 'close' in formatted_data.columns:
                    # Simple moving averages
                    formatted_data['sma_5'] = formatted_data['close'].rolling(window=5).mean()
                    formatted_data['sma_10'] = formatted_data['close'].rolling(window=10).mean()
                    formatted_data['sma_20'] = formatted_data['close'].rolling(window=20).mean()
                    
                    # Price relative to moving averages
                    formatted_data['price_above_sma20'] = (formatted_data['close'] > formatted_data['sma_20']).astype(int)
                
                # Volume analysis
                if 'volume' in formatted_data.columns:
                    formatted_data['volume_ma_10'] = formatted_data['volume'].rolling(window=10).mean()
                    formatted_data['volume_spike'] = (formatted_data['volume'] > 2 * formatted_data['volume_ma_10']).astype(int)
            
            elif analysis_type == 'fundamental':
                # Prepare data for fundamental analysis
                if 'symbol' in formatted_data.columns:
                    # Add sector classification
                    formatted_data['sector'] = formatted_data['symbol'].map(self.sector_mapping).fillna('Others')
                    
                    # Add market cap estimation (placeholder for version 1)
                    formatted_data['estimated_market_cap'] = 'Unknown'
                
                # Add financial ratios placeholders (to be filled with actual data)
                formatted_data['pe_ratio'] = np.nan
                formatted_data['pb_ratio'] = np.nan
                formatted_data['debt_to_equity'] = np.nan
            
            elif analysis_type == 'sentiment':
                # Prepare data for sentiment analysis
                # Add time-based features for sentiment correlation
                if 'date' in formatted_data.columns or isinstance(formatted_data.index, pd.DatetimeIndex):
                    date_col = formatted_data.index if isinstance(formatted_data.index, pd.DatetimeIndex) else pd.to_datetime(formatted_data['date'])
                    
                    # Add day of week for sentiment patterns
                    formatted_data['weekday'] = date_col.dayofweek
                    formatted_data['is_monday'] = (formatted_data['weekday'] == 0).astype(int)
                    formatted_data['is_friday'] = (formatted_data['weekday'] == 4).astype(int)
                
                # Add volatility measures for sentiment correlation
                if 'close' in formatted_data.columns:
                    formatted_data['daily_return'] = formatted_data['close'].pct_change()
                    formatted_data['volatility_5d'] = formatted_data['daily_return'].rolling(window=5).std()
            
            # Ensure data is sorted by date
            if 'date' in formatted_data.columns:
                formatted_data = formatted_data.sort_values('date')
            elif isinstance(formatted_data.index, pd.DatetimeIndex):
                formatted_data = formatted_data.sort_index()
            
            logger.info(f"Data formatted for {analysis_type} analysis")
            return formatted_data
            
        except Exception as e:
            logger.error(f"Data formatting failed: {str(e)}")
            raise DataTransformationError(f"Failed to format data for {analysis_type} analysis: {str(e)}")
    
    def export_transformed_data(self, data: pd.DataFrame, file_path: str, 
                               format_type: str = 'csv') -> bool:
        """
        Export transformed data to various formats
        
        Args:
            data: DataFrame to export
            file_path: Path to save the file
            format_type: Export format ('csv', 'json', 'pickle', 'parquet')
            
        Returns:
            Boolean indicating success
        """
        try:
            if format_type == 'csv':
                data.to_csv(file_path, index=False)
            elif format_type == 'json':
                data.to_json(file_path, orient='records', date_format='iso')
            elif format_type == 'pickle':
                data.to_pickle(file_path)
            elif format_type == 'parquet':
                data.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Data exported to {file_path} in {format_type} format")
            return True
            
        except Exception as e:
            logger.error(f"Data export failed: {str(e)}")
            return False
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_metrics = {
                'total_records': len(data),
                'missing_values': data.isnull().sum().to_dict(),
                'duplicate_records': data.duplicated().sum(),
                'data_types': data.dtypes.to_dict(),
                'date_range': {},
                'completeness_score': 0.0,
                'issues': []
            }
            
            # Check date range
            if 'date' in data.columns:
                date_col = pd.to_datetime(data['date'])
                quality_metrics['date_range'] = {
                    'start_date': date_col.min().strftime('%Y-%m-%d'),
                    'end_date': date_col.max().strftime('%Y-%m-%d'),
                    'total_days': (date_col.max() - date_col.min()).days
                }
            
            # Calculate completeness score
            total_cells = len(data) * len(data.columns)
            missing_cells = data.isnull().sum().sum()
            quality_metrics['completeness_score'] = ((total_cells - missing_cells) / total_cells) * 100
            
            # Check for common issues
            if quality_metrics['duplicate_records'] > 0:
                quality_metrics['issues'].append(f"Found {quality_metrics['duplicate_records']} duplicate records")
            
            if quality_metrics['completeness_score'] < 95:
                quality_metrics['issues'].append(f"Data completeness is {quality_metrics['completeness_score']:.1f}% (below 95%)")
            
            # Check for price data consistency
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= Low
                inconsistent_high_low = (data['high'] < data['low']).sum()
                if inconsistent_high_low > 0:
                    quality_metrics['issues'].append(f"Found {inconsistent_high_low} records where High < Low")
                
                # Close should be between High and Low
                close_out_of_range = ((data['close'] > data['high']) | (data['close'] < data['low'])).sum()
                if close_out_of_range > 0:
                    quality_metrics['issues'].append(f"Found {close_out_of_range} records where Close is outside High-Low range")
            
            logger.info(f"Data quality validation completed. Completeness: {quality_metrics['completeness_score']:.1f}%")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {str(e)}")
            return {'error': str(e)}


# Utility functions for Indian market specific transformations

def convert_to_indian_time(data: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Convert datetime to Indian Standard Time (IST)
    
    Args:
        data: DataFrame with datetime column
        datetime_col: Name of datetime column
        
    Returns:
        DataFrame with IST converted datetime
    """
    try:
        data_copy = data.copy()
        if datetime_col in data_copy.columns:
            data_copy[datetime_col] = pd.to_datetime(data_copy[datetime_col])
            # Convert to IST (UTC+5:30)
            data_copy[datetime_col] = data_copy[datetime_col].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            data_copy['ist_time'] = data_copy[datetime_col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return data_copy
    except Exception as e:
        logger.error(f"IST conversion failed: {str(e)}")
        return data

def filter_trading_hours(data: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Filter data to Indian market trading hours (9:15 AM to 3:30 PM IST)
    
    Args:
        data: DataFrame with datetime column
        datetime_col: Name of datetime column
        
    Returns:
        DataFrame filtered to trading hours
    """
    try:
        data_copy = data.copy()
        if datetime_col in data_copy.columns:
            data_copy[datetime_col] = pd.to_datetime(data_copy[datetime_col])
            
            # Filter for trading hours (9:15 AM to 3:30 PM)
            trading_hours = (data_copy[datetime_col].dt.time >= pd.Timestamp('09:15:00').time()) & \
                           (data_copy[datetime_col].dt.time <= pd.Timestamp('15:30:00').time())
            
            data_copy = data_copy[trading_hours]
            
            # Remove weekends (Saturday=5, Sunday=6)
            weekday_mask = data_copy[datetime_col].dt.dayofweek < 5
            data_copy = data_copy[weekday_mask]
        
        return data_copy
    except Exception as e:
        logger.error(f"Trading hours filtering failed: {str(e)}")
        return data

def add_indian_market_holidays(data: pd.DataFrame, year: int = 2022) -> pd.DataFrame:
    """
    Add Indian market holidays information to the data
    
    Args:
        data: DataFrame with date information
        year: Year for which to add holidays
        
    Returns:
        DataFrame with holiday information
    """
    try:
        # Indian market holidays for 2022 (as examples)
        holidays_2022 = [
            '2022-01-26',  # Republic Day
            '2022-03-01',  # Holi
            '2022-03-18',  # Holi
            '2022-04-14',  # Ram Navami
            '2022-04-15',  # Good Friday
            '2022-05-03',  # Eid ul-Fitr
            '2022-08-09',  # Moharram
            '2022-08-15',  # Independence Day
            '2022-08-31',  # Ganesh Chaturthi
            '2022-10-05',  # Dussehra
            '2022-10-24',  # Diwali
            '2022-10-26',  # Diwali
            '2022-11-08',  # Guru Nanak Jayanti
        ]
        
        data_copy = data.copy()
        
        if 'date' in data_copy.columns:
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            data_copy['is_holiday'] = data_copy['date'].dt.strftime('%Y-%m-%d').isin(holidays_2022)
        
        return data_copy
    except Exception as e:
        logger.error(f"Holiday information addition failed: {str(e)}")
        return data