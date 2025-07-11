"""
Market Research System v1.0 - Indian Market Data Fetcher
File: src/data/fetch_market_data.py
Created: February 2022

Fetches Indian stock market data from Yahoo Finance and NSE APIs.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional, Union
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import IndianMarketConfig, DataConfig, SystemConfig
from ..utils.date_utils import get_trading_days, is_trading_day
from ..utils.logging_utils import setup_logger

class IndianMarketDataFetcher:
    """
    Comprehensive data fetcher for Indian stock market data
    Supports NSE, BSE symbols and market indices
    """
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.config = IndianMarketConfig()
        self.data_config = DataConfig()
        self.session = requests.Session()
        
        # Setup session headers to mimic browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        
        self.logger.info("Indian Market Data Fetcher initialized")
    
    def fetch_stock_data(self, 
                        symbol: str, 
                        period: str = "2y", 
                        interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol using yfinance
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NS')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            self.logger.info(f"Fetching data for {symbol}")
            
            # Create yfinance ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None
            
            # Clean and standardize data
            data = self._clean_stock_data(data, symbol)
            
            # Add symbol column
            data['Symbol'] = symbol
            
            self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols: List[str], 
                            period: str = "2y", 
                            interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols (List[str]): List of stock symbols
            period (str): Data period
            interval (str): Data interval
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            try:
                data = self.fetch_stock_data(symbol, period, interval)
                if data is not None:
                    results[symbol] = data
                
                # Add delay to avoid rate limiting
                if i < len(symbols) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully fetched data for {len(results)} out of {len(symbols)} symbols")
        return results
    
    def fetch_nifty_50_data(self, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """Fetch data for all Nifty 50 stocks"""
        nifty_50_symbols = self.config.NIFTY_100_SYMBOLS[:50]  # Top 50 from Nifty 100
        return self.fetch_multiple_stocks(nifty_50_symbols, period)
    
    def fetch_sensex_30_data(self, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """Fetch data for all Sensex 30 stocks"""
        return self.fetch_multiple_stocks(self.config.SENSEX_30_SYMBOLS, period)
    
    def fetch_sector_data(self, sector: str, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all stocks in a specific sector
        
        Args:
            sector (str): Sector name (e.g., 'BANKING', 'IT', 'PHARMA')
            period (str): Data period
        
        Returns:
            Dict[str, pd.DataFrame]: Sector stock data
        """
        if sector not in self.config.SECTORS:
            self.logger.error(f"Sector {sector} not found in configuration")
            return {}
        
        sector_symbols = self.config.SECTORS[sector]
        return self.fetch_multiple_stocks(sector_symbols, period)
    
    def fetch_index_data(self, index_name: str = None, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch Indian market indices data
        
        Args:
            index_name (str): Specific index name or None for all indices
            period (str): Data period
        
        Returns:
            Dict[str, pd.DataFrame]: Index data
        """
        if index_name:
            if index_name not in self.config.INDICES:
                self.logger.error(f"Index {index_name} not found")
                return {}
            indices_to_fetch = {index_name: self.config.INDICES[index_name]}
        else:
            indices_to_fetch = self.config.INDICES
        
        results = {}
        for name, symbol in indices_to_fetch.items():
            try:
                data = self.fetch_stock_data(symbol, period)
                if data is not None:
                    results[name] = data
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                self.logger.error(f"Error fetching index {name}: {str(e)}")
        
        return results
    
    def fetch_real_time_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch real-time stock data
        
        Args:
            symbols (List[str]): List of stock symbols
        
        Returns:
            Dict[str, Dict]: Real-time stock data
        """
        results = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current price data
                current_data = ticker.history(period='1d', interval='1m').tail(1)
                
                if not current_data.empty:
                    results[symbol] = {
                        'current_price': float(current_data['Close'].iloc[-1]),
                        'open_price': float(current_data['Open'].iloc[0]),
                        'high_price': float(current_data['High'].max()),
                        'low_price': float(current_data['Low'].min()),
                        'volume': int(current_data['Volume'].iloc[-1]),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'dividend_yield': info.get('dividendYield', 0),
                        'previous_close': info.get('previousClose', 0),
                        'day_change': float(current_data['Close'].iloc[-1]) - info.get('previousClose', 0),
                        'day_change_percent': ((float(current_data['Close'].iloc[-1]) - info.get('previousClose', 0)) / info.get('previousClose', 1)) * 100,
                        'timestamp': current_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                time.sleep(0.3)  # Rate limiting for real-time data
                
            except Exception as e:
                self.logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
                continue
        
        return results
    
    def fetch_market_summary(self) -> Dict[str, any]:
        """
        Fetch daily market summary for Indian markets
        
        Returns:
            Dict: Market summary data
        """
        try:
            summary = {}
            
            # Fetch major indices
            indices = self.fetch_index_data()
            
            # Calculate market statistics
            if indices:
                summary['indices'] = {}
                for index_name, data in indices.items():
                    if not data.empty:
                        latest = data.tail(1)
                        prev = data.tail(2).head(1)
                        
                        current_price = float(latest['Close'].iloc[0])
                        prev_close = float(prev['Close'].iloc[0]) if not prev.empty else current_price
                        
                        summary['indices'][index_name] = {
                            'current_value': current_price,
                            'change': current_price - prev_close,
                            'change_percent': ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0,
                            'volume': int(latest['Volume'].iloc[0])
                        }
            
            # Add market timing info
            summary['market_status'] = self._get_market_status()
            summary['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error fetching market summary: {str(e)}")
            return {}
    
    def save_data(self, data: pd.DataFrame, symbol: str, data_type: str = "stock") -> bool:
        """
        Save fetched data to CSV files
        
        Args:
            data (pd.DataFrame): Data to save
            symbol (str): Stock symbol
            data_type (str): Type of data ('stock', 'index', 'sector')
        
        Returns:
            bool: Success status
        """
        try:
            # Create directory structure
            base_path = Path(self.data_config.DATA_DIR)
            data_path = base_path / "raw" / data_type
            data_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename with date
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"{symbol.replace('.', '_')}_{date_str}.csv"
            filepath = data_path / filename
            
            # Save data
            data.to_csv(filepath, index=True)
            
            self.logger.info(f"Data saved successfully: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {str(e)}")
            return False
    
    def load_saved_data(self, symbol: str, data_type: str = "stock", 
                       date: str = None) -> Optional[pd.DataFrame]:
        """
        Load previously saved data
        
        Args:
            symbol (str): Stock symbol
            data_type (str): Type of data
            date (str): Specific date (YYYYMMDD format) or None for latest
        
        Returns:
            pd.DataFrame: Loaded data or None
        """
        try:
            base_path = Path(self.data_config.DATA_DIR)
            data_path = base_path / "raw" / data_type
            
            if date:
                filename = f"{symbol.replace('.', '_')}_{date}.csv"
                filepath = data_path / filename
                
                if filepath.exists():
                    return pd.read_csv(filepath, index_col=0, parse_dates=True)
            else:
                # Find latest file for symbol
                pattern = f"{symbol.replace('.', '_')}_*.csv"
                files = list(data_path.glob(pattern))
                
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    return pd.read_csv(latest_file, index_col=0, parse_dates=True)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
    
    def _clean_stock_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and standardize stock data
        
        Args:
            data (pd.DataFrame): Raw stock data
            symbol (str): Stock symbol
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            # Remove rows with all NaN values
            data = data.dropna(how='all')
            
            # Forward fill missing values
            data = data.fillna(method='ffill')
            
            # Remove any remaining NaN values
            data = data.dropna()
            
            # Ensure positive prices
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in data.columns:
                    data[col] = data[col].abs()
            
            # Ensure volume is non-negative
            if 'Volume' in data.columns:
                data['Volume'] = data['Volume'].abs()
            
            # Add basic derived columns
            data['Price_Range'] = data['High'] - data['Low']
            data['Price_Change'] = data['Close'] - data['Open']
            data['Price_Change_Pct'] = (data['Price_Change'] / data['Open']) * 100
            
            # Add trading session info
            data['Trading_Day'] = data.index.date
            data['Day_of_Week'] = data.index.dayofweek
            data['Month'] = data.index.month
            data['Year'] = data.index.year
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning data for {symbol}: {str(e)}")
            return data
    
    def _get_market_status(self) -> str:
        """
        Get current Indian market status
        
        Returns:
            str: Market status ('open', 'closed', 'pre_market', 'post_market')
        """
        now = datetime.now()
        current_time = now.time()
        
        # Indian market hours (9:15 AM - 3:30 PM IST)
        market_open = datetime.strptime('09:15', '%H:%M').time()
        market_close = datetime.strptime('15:30', '%H:%M').time()
        pre_market_start = datetime.strptime('09:00', '%H:%M').time()
        post_market_end = datetime.strptime('16:00', '%H:%M').time()
        
        # Check if it's a trading day (Monday to Friday, excluding holidays)
        if not is_trading_day(now.date()):
            return 'closed'
        
        if market_open <= current_time <= market_close:
            return 'open'
        elif pre_market_start <= current_time < market_open:
            return 'pre_market'
        elif market_close < current_time <= post_market_end:
            return 'post_market'
        else:
            return 'closed'
    
    def get_stock_info(self, symbol: str) -> Dict[str, any]:
        """
        Get detailed stock information
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            Dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information for Indian stocks
            stock_info = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'eps': info.get('trailingEps', 0),
                'book_value': info.get('bookValue', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'currency': info.get('currency', 'INR'),
                'exchange': info.get('exchange', 'NSE'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return stock_info
            
        except Exception as e:
            self.logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {}
    
    def get_historical_data_range(self, symbol: str, start_date: str, 
                                 end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical data for a specific date range
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
        
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                data = self._clean_stock_data(data, symbol)
                data['Symbol'] = symbol
                self.logger.info(f"Fetched {len(data)} records for {symbol} from {start_date} to {end_date}")
                return data
            else:
                self.logger.warning(f"No data found for {symbol} in date range {start_date} to {end_date}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def health_check(self) -> Dict[str, any]:
        """
        Perform system health check
        
        Returns:
            Dict: Health status
        """
        health_status = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_status': 'healthy',
            'data_sources': {},
            'last_successful_fetch': None,
            'errors': []
        }
        
        try:
            # Test Yahoo Finance connectivity
            test_symbol = 'RELIANCE.NS'
            test_data = self.fetch_stock_data(test_symbol, period='5d')
            
            if test_data is not None and not test_data.empty:
                health_status['data_sources']['yahoo_finance'] = 'active'
                health_status['last_successful_fetch'] = test_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            else:
                health_status['data_sources']['yahoo_finance'] = 'error'
                health_status['errors'].append('Yahoo Finance API not responding')
                health_status['system_status'] = 'degraded'
            
            # Test data directory access
            base_path = Path(self.data_config.DATA_DIR)
            if base_path.exists() and base_path.is_dir():
                health_status['data_directory'] = 'accessible'
            else:
                health_status['data_directory'] = 'error'
                health_status['errors'].append('Data directory not accessible')
                health_status['system_status'] = 'degraded'
            
        except Exception as e:
            health_status['system_status'] = 'error'
            health_status['errors'].append(f"Health check failed: {str(e)}")
        
        return health_status


# Utility functions for batch operations
class BatchDataFetcher:
    """Helper class for batch data operations"""
    
    def __init__(self, fetcher: IndianMarketDataFetcher):
        self.fetcher = fetcher
        self.logger = fetcher.logger
    
    def fetch_and_save_nifty_50(self, period: str = "2y") -> Dict[str, bool]:
        """Fetch and save all Nifty 50 data"""
        results = {}
        nifty_data = self.fetcher.fetch_nifty_50_data(period)
        
        for symbol, data in nifty_data.items():
            success = self.fetcher.save_data(data, symbol, "nifty50")
            results[symbol] = success
        
        return results
    
    def fetch_and_save_sector_data(self, sector: str, period: str = "2y") -> Dict[str, bool]:
        """Fetch and save sector data"""
        results = {}
        sector_data = self.fetcher.fetch_sector_data(sector, period)
        
        for symbol, data in sector_data.items():
            success = self.fetcher.save_data(data, symbol, f"sector_{sector.lower()}")
            results[symbol] = success
        
        return results
    
    def daily_data_update(self) -> Dict[str, any]:
        """Perform daily data update for all tracked symbols"""
        update_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'nifty_50': {},
            'indices': {},
            'sectors': {},
            'errors': []
        }
        
        try:
            # Update Nifty 50
            update_results['nifty_50'] = self.fetch_and_save_nifty_50(period="5d")
            
            # Update indices
            indices_data = self.fetcher.fetch_index_data(period="5d")
            for index_name, data in indices_data.items():
                success = self.fetcher.save_data(data, index_name, "indices")
                update_results['indices'][index_name] = success
            
            # Update major sectors
            major_sectors = ['BANKING', 'IT', 'PHARMA', 'AUTO']
            for sector in major_sectors:
                sector_results = self.fetch_and_save_sector_data(sector, period="5d")
                update_results['sectors'][sector] = sector_results
                
        except Exception as e:
            update_results['errors'].append(f"Daily update failed: {str(e)}")
            self.logger.error(f"Daily update error: {str(e)}")
        
        return update_results


# Main execution for testing
if __name__ == "__main__":
    # Initialize the fetcher
    fetcher = IndianMarketDataFetcher()
    
    # Test basic functionality
    print("Testing Indian Market Data Fetcher...")
    
    # Test single stock fetch
    test_symbol = 'RELIANCE.NS'
    data = fetcher.fetch_stock_data(test_symbol, period='1mo')
    if data is not None:
        print(f"✓ Successfully fetched data for {test_symbol}: {len(data)} records")
        print(f"  Latest close price: ₹{data['Close'].iloc[-1]:.2f}")
    
    # Test market summary
    summary = fetcher.fetch_market_summary()
    if summary:
        print("✓ Market summary fetched successfully")
        print(f"  Market status: {summary.get('market_status', 'Unknown')}")
    
    # Test health check
    health = fetcher.health_check()
    print(f"✓ System health: {health['system_status']}")
    
    print("Testing completed!")