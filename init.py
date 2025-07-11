"""
Data Package Initialization
Market Research System v1.0 (2022)
Author: Independent Market Researcher
Focus: Indian Stock Market Analysis

This module initializes the data package and provides centralized access
to all data-related functionality including fetching, cleaning, validation,
transformation, and storage.
"""

from .fetch_economic_data import EconomicDataFetcher
from .fetch_indices_data import IndicesDataFetcher
from .data_cleaner import DataCleaner
from .data_validator import DataValidator
from .data_transformer import DataTransformer
from .data_storage import DataStorage
from .exceptions import (
    DataFetchError,
    DataValidationError,  
    DataCleaningError,
    DataTransformationError,
    DataStorageError,
    APIError,
    RateLimitError
)

# Version information
__version__ = "1.0.0"
__author__ = "Independent Market Researcher"
__email__ = "research@marketanalysis.com"
__description__ = "Data handling modules for Indian stock market research"

# Package-level constants
SUPPORTED_EXCHANGES = ["NSE", "BSE"]
DEFAULT_CURRENCY = "INR"
DEFAULT_TIMEZONE = "Asia/Kolkata"

# Data source configurations
DATA_SOURCES = {
    "economic": {
        "primary": "RBI",  # Reserve Bank of India
        "secondary": "CEIC",  # CEIC economic database
        "fallback": "manual"  # Manual data entry
    },
    "indices": {
        "primary": "yfinance",  # Yahoo Finance
        "secondary": "nsepy",   # NSE Python library
        "fallback": "manual"    # Manual data entry
    },
    "stocks": {
        "primary": "yfinance",  # Yahoo Finance
        "secondary": "nsepy",   # NSE Python library  
        "fallback": "manual"    # Manual data entry
    }
}

# Indian market specific configurations
INDIAN_MARKET_CONFIG = {
    "trading_hours": {
        "start": "09:15",
        "end": "15:30",
        "timezone": "Asia/Kolkata"
    },
    "holidays": [
        # Major Indian stock market holidays (2022)
        "2022-01-26",  # Republic Day
        "2022-03-01",  # Holi
        "2022-03-18",  # Holi
        "2022-04-14",  # Ram Navami
        "2022-04-15",  # Good Friday
        "2022-05-03",  # Eid ul-Fitr
        "2022-08-09",  # Muharram
        "2022-08-15",  # Independence Day
        "2022-08-31",  # Ganesh Chaturthi
        "2022-10-02",  # Gandhi Jayanti
        "2022-10-05",  # Dussehra
        "2022-10-24",  # Diwali Laxmi Pujan
        "2022-10-26",  # Diwali Balipratipada
        "2022-11-08",  # Guru Nanak Jayanti
    ],
    "currency": "INR",
    "decimal_places": 2
}

# Major Indian stock indices
MAJOR_INDICES = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN", 
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_ENERGY": "^CNXENERGY"
}

# Top Indian stocks by market cap (2022)
TOP_INDIAN_STOCKS = {
    # Technology
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS", 
    "WIPRO": "WIPRO.NS",
    "HCL_TECH": "HCLTECH.NS",
    "TECH_MAHINDRA": "TECHM.NS",
    
    # Banking & Finance
    "HDFC_BANK": "HDFCBANK.NS",
    "ICICI_BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "AXIS_BANK": "AXISBANK.NS",
    "KOTAK_BANK": "KOTAKBANK.NS",
    
    # Energy & Oil
    "RELIANCE": "RELIANCE.NS",
    "ONGC": "ONGC.NS",
    "IOC": "IOC.NS",
    "BPCL": "BPCL.NS",
    "NTPC": "NTPC.NS",
    
    # FMCG
    "HINDUSTAN_UNILEVER": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "NESTLE": "NESTLEIND.NS",
    "BRITANNIA": "BRITANNIA.NS",
    
    # Pharmaceuticals
    "SUN_PHARMA": "SUNPHARMA.NS",
    "DR_REDDYS": "DRREDDY.NS",
    "CIPLA": "CIPLA.NS",
    "DIVI": "DIVISLAB.NS",
    
    # Auto
    "MARUTI": "MARUTI.NS",
    "TATA_MOTORS": "TATAMOTORS.NS",
    "MAHINDRA": "M&M.NS",
    "BAJAJ_AUTO": "BAJAJ-AUTO.NS",
    
    # Metals
    "TATA_STEEL": "TATASTEEL.NS",
    "JSW_STEEL": "JSWSTEEL.NS",
    "HINDALCO": "HINDALCO.NS",
    "VEDANTA": "VEDL.NS"
}

# Economic indicators for India
INDIAN_ECONOMIC_INDICATORS = {
    "GDP_GROWTH": "India GDP Growth Rate",
    "INFLATION_CPI": "India Consumer Price Index", 
    "REPO_RATE": "India Repo Rate",
    "INDUSTRIAL_PRODUCTION": "India Industrial Production",
    "UNEMPLOYMENT": "India Unemployment Rate",
    "FOREX_RESERVES": "India Foreign Exchange Reserves",
    "CURRENT_ACCOUNT": "India Current Account Balance",
    "FII_FLOWS": "Foreign Institutional Investment Flows",
    "DII_FLOWS": "Domestic Institutional Investment Flows"
}


class DataManager:
    """
    Central data manager that coordinates all data-related operations.
    Provides a unified interface for data fetching, processing, and storage.
    """
    
    def __init__(self):
        """Initialize the data manager with all required components."""
        self.economic_fetcher = EconomicDataFetcher()
        self.indices_fetcher = IndicesDataFetcher()
        self.cleaner = DataCleaner()  
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.storage = DataStorage()
    
    def get_market_data(self, symbol: str, period: str = "1y"):
        """
        Get complete market data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period for data
            
        Returns:
            dict: Complete market data including prices, indicators, and metadata
        """
        try:
            # Fetch raw data
            if symbol in MAJOR_INDICES.values():
                raw_data = self.indices_fetcher.fetch_index_data(symbol, period)
            else:
                raw_data = self.indices_fetcher.fetch_stock_data(symbol, period)
            
            # Clean and validate data
            clean_data = self.cleaner.clean_price_data(raw_data)
            validated_data = self.validator.validate_price_data(clean_data)
            
            # Transform data (add technical indicators, etc.)
            transformed_data = self.transformer.add_technical_indicators(validated_data)
            
            # Store processed data
            self.storage.store_processed_data(symbol, transformed_data)
            
            return transformed_data
            
        except Exception as e:
            raise DataFetchError(f"Failed to get market data for {symbol}: {str(e)}")
    
    def get_economic_data(self, indicator: str, period: str = "5y"):
        """
        Get economic indicator data.
        
        Args:
            indicator (str): Economic indicator name
            period (str): Time period for data
            
        Returns:
            pandas.DataFrame: Economic indicator data
        """
        try:
            raw_data = self.economic_fetcher.fetch_indicator(indicator, period)
            clean_data = self.cleaner.clean_economic_data(raw_data)
            validated_data = self.validator.validate_economic_data(clean_data)
            
            self.storage.store_economic_data(indicator, validated_data)
            
            return validated_data
            
        except Exception as e:
            raise DataFetchError(f"Failed to get economic data for {indicator}: {str(e)}")


# Global data manager instance
data_manager = DataManager()


def get_supported_symbols():
    """
    Get list of all supported stock symbols.
    
    Returns:
        dict: Dictionary of supported symbols categorized by type
    """
    return {
        "indices": MAJOR_INDICES,
        "stocks": TOP_INDIAN_STOCKS,
        "economic": INDIAN_ECONOMIC_INDICATORS
    }


def is_market_open():
    """
    Check if Indian stock market is currently open.
    
    Returns:
        bool: True if market is open, False otherwise
    """
    from datetime import datetime
    import pytz
    
    # Get current time in Indian timezone
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if it's a holiday
    today_str = now.strftime('%Y-%m-%d')
    if today_str in INDIAN_MARKET_CONFIG["holidays"]:
        return False
    
    # Check trading hours
    start_time = datetime.strptime("09:15", "%H:%M").time()
    end_time = datetime.strptime("15:30", "%H:%M").time()
    current_time = now.time()
    
    return start_time <= current_time <= end_time


# Export all public components
__all__ = [
    # Classes
    "EconomicDataFetcher",
    "IndicesDataFetcher", 
    "DataCleaner",
    "DataValidator",
    "DataTransformer",
    "DataStorage",
    "DataManager",
    
    # Exceptions
    "DataFetchError",
    "DataValidationError",
    "DataCleaningError", 
    "DataTransformationError",
    "DataStorageError",
    "APIError",
    "RateLimitError",
    
    # Constants
    "SUPPORTED_EXCHANGES",
    "MAJOR_INDICES",
    "TOP_INDIAN_STOCKS",
    "INDIAN_ECONOMIC_INDICATORS",
    "INDIAN_MARKET_CONFIG",
    
    # Functions
    "get_supported_symbols",
    "is_market_open",
    
    # Global instances
    "data_manager"
]