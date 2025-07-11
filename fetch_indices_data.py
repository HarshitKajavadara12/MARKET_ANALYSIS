"""
Market Research System v1.0 - Indian Market Indices Data Fetcher
Created: 2022
Author: Independent Market Researcher

This module fetches Indian market indices data from Yahoo Finance API.
Supports NSE, BSE indices and major sectoral indices.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import time
import os
from .exceptions import DataFetchError, APIConnectionError
from .data_validator import DataValidator

class IndianIndicesDataFetcher:
    """
    Fetches Indian stock market indices data from Yahoo Finance
    """
    
    # Indian Market Indices with Yahoo Finance symbols
    INDIAN_INDICES = {
        'NIFTY50': '^NSEI',
        'NIFTY_BANK': '^NSEBANK',
        'NIFTY_IT': 'NIFTYIT.NS',
        'NIFTY_PHARMA': 'NIFTYPHARMA.NS',
        'NIFTY_AUTO': 'NIFTYAUTO.NS',
        'NIFTY_METAL': 'NIFTYMETAL.NS',
        'NIFTY_REALTY': 'NIFTYREALTY.NS',
        'NIFTY_ENERGY': 'NIFTYENERGY.NS',
        'NIFTY_FMCG': 'NIFTYFMCG.NS',
        'NIFTY_MID50': 'NIFTYMID50.NS',
        'NIFTY_SMALL250': 'NIFTYSML250.NS',
        'SENSEX': '^BSESN',
        'BSE500': 'BSE500.BO',
        'BSE_MIDCAP': 'BSE-MIDCAP.BO',
        'BSE_SMALLCAP': 'BSE-SMLCAP.BO'
    }
    
    def __init__(self, data_dir: str = "data/raw/indices"):
        """
        Initialize the Indian Indices Data Fetcher
        
        Args:
            data_dir: Directory to store raw indices data
        """
        self.data_dir = data_dir
        self.validator = DataValidator()
        self.logger = self._setup_logger()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the data fetcher"""
        logger = logging.getLogger('IndianIndicesDataFetcher')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def fetch_index_data(self, 
                        index_symbol: str, 
                        period: str = "1y",
                        interval: str = "1d",
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data for a specific Indian market index
        
        Args:
            index_symbol: Index symbol (use keys from INDIAN_INDICES)
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get Yahoo Finance symbol
            if index_symbol in self.INDIAN_INDICES:
                yf_symbol = self.INDIAN_INDICES[index_symbol]
            else:
                yf_symbol = index_symbol
                
            self.logger.info(f"Fetching data for {index_symbol} ({yf_symbol})")
            
            # Create ticker object
            ticker = yf.Ticker(yf_symbol)
            
            # Fetch data based on parameters
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
                
            if data.empty:
                raise DataFetchError(f"No data found for {index_symbol}")
                
            # Add metadata
            data['Symbol'] = index_symbol
            data['YF_Symbol'] = yf_symbol
            data['Fetch_Time'] = datetime.now()
            
            # Validate data
            if not self.validator.validate_ohlcv_data(data):
                self.logger.warning(f"Data validation failed for {index_symbol}")
                
            self.logger.info(f"Successfully fetched {len(data)} records for {index_symbol}")
            return data
            
        except Exception as e:
            error_msg = f"Failed to fetch data for {index_symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise DataFetchError(error_msg)
    
    def fetch_multiple_indices(self, 
                              indices: List[str],
                              period: str = "1y",
                              interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple indices
        
        Args:
            indices: List of index symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary with index symbol as key and DataFrame as value
        """
        results = {}
        failed_indices = []
        
        for index in indices:
            try:
                data = self.fetch_index_data(index, period, interval)
                results[index] = data
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Failed to fetch {index}: {str(e)}")
                failed_indices.append(index)
                
        if failed_indices:
            self.logger.warning(f"Failed to fetch data for: {failed_indices}")
            
        return results
    
    def fetch_all_major_indices(self, 
                               period: str = "1y",
                               interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all major Indian indices
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary with all indices data
        """
        major_indices = ['NIFTY50', 'NIFTY_BANK', 'SENSEX', 'NIFTY_IT', 'NIFTY_PHARMA']
        return self.fetch_multiple_indices(major_indices, period, interval)
    
    def get_index_info(self, index_symbol: str) -> Dict:
        """
        Get basic information about an index
        
        Args:
            index_symbol: Index symbol
            
        Returns:
            Dictionary with index information
        """
        try:
            if index_symbol in self.INDIAN_INDICES:
                yf_symbol = self.INDIAN_INDICES[index_symbol]
            else:
                yf_symbol = index_symbol
                
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            return {
                'symbol': index_symbol,
                'yf_symbol': yf_symbol,
                'name': info.get('longName', 'N/A'),
                'market': info.get('market', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
                'currency': info.get('currency', 'INR'),
                'timezone': info.get('timeZone', 'Asia/Kolkata')
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get info for {index_symbol}: {str(e)}")
            return {}
    
    def save_index_data(self, 
                       data: pd.DataFrame, 
                       index_symbol: str,
                       file_format: str = 'csv') -> str:
        """
        Save index data to file
        
        Args:
            data: DataFrame to save
            index_symbol: Index symbol for filename
            file_format: File format ('csv', 'parquet', 'json')
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{index_symbol}_{timestamp}.{file_format}"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            if file_format == 'csv':
                data.to_csv(filepath)
            elif file_format == 'parquet':
                data.to_parquet(filepath)
            elif file_format == 'json':
                data.to_json(filepath, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            self.logger.info(f"Data saved to {filepath}")
            return filepath
            
        except Exception as e:
            error_msg = f"Failed to save data to {filepath}: {str(e)}"
            self.logger.error(error_msg)
            raise DataFetchError(error_msg)
    
    def get_available_indices(self) -> Dict[str, str]:
        """
        Get list of available Indian indices
        
        Returns:
            Dictionary of available indices with their Yahoo Finance symbols
        """
        return self.INDIAN_INDICES.copy()
    
    def fetch_intraday_data(self, 
                           index_symbol: str,
                           interval: str = "5m",
                           period: str = "1d") -> pd.DataFrame:
        """
        Fetch intraday data for an index
        
        Args:
            index_symbol: Index symbol
            interval: Intraday interval ('1m', '2m', '5m', '15m', '30m', '60m')
            period: Period for intraday ('1d', '2d', '5d')
            
        Returns:
            DataFrame with intraday data
        """
        if interval not in ['1m', '2m', '5m', '15m', '30m', '60m']:
            raise ValueError(f"Invalid intraday interval: {interval}")
            
        return self.fetch_index_data(index_symbol, period=period, interval=interval)
    
    def calculate_index_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns for index data
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with calculated returns
        """
        returns_data = data.copy()
        
        # Daily returns
        returns_data['Daily_Return'] = data['Close'].pct_change()
        
        # Weekly returns (5 trading days)
        returns_data['Weekly_Return'] = data['Close'].pct_change(5)
        
        # Monthly returns (22 trading days)
        returns_data['Monthly_Return'] = data['Close'].pct_change(22)
        
        # Cumulative returns
        returns_data['Cumulative_Return'] = (1 + returns_data['Daily_Return']).cumprod() - 1
        
        return returns_data
    
    def get_market_status(self) -> Dict[str, str]:
        """
        Get current Indian market status
        
        Returns:
            Dictionary with market status information
        """
        try:
            # Using NIFTY as reference for Indian market
            nifty = yf.Ticker(self.INDIAN_INDICES['NIFTY50'])
            data = nifty.history(period="1d", interval="1m")
            
            now = datetime.now()
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            if data.empty:
                status = "CLOSED"
            else:
                if market_open <= now <= market_close and now.weekday() < 5:
                    status = "OPEN"
                else:
                    status = "CLOSED"
                    
            return {
                'status': status,
                'market_open': '09:15 IST',
                'market_close': '15:30 IST',
                'timezone': 'Asia/Kolkata',
                'last_update': now.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get market status: {str(e)}")
            return {
                'status': 'UNKNOWN',
                'error': str(e),
                'last_update': datetime.now().isoformat()
            }


# Example usage and testing
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = IndianIndicesDataFetcher()
    
    # Test fetching single index
    try:
        nifty_data = fetcher.fetch_index_data('NIFTY50', period='1mo')
        print(f"NIFTY50 data shape: {nifty_data.shape}")
        print(nifty_data.head())
        
        # Save data
        filepath = fetcher.save_index_data(nifty_data, 'NIFTY50')
        print(f"Data saved to: {filepath}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test market status
    status = fetcher.get_market_status()
    print(f"Market Status: {status}")
    
    # Test available indices
    indices = fetcher.get_available_indices()
    print(f"Available indices: {list(indices.keys())[:5]}...")