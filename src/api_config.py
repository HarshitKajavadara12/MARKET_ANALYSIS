"""
API Configuration Module for Market Research System v1.0
Author: Market Research Team
Created: January 2022
Focus: Indian Stock Market Data Sources

This module handles all API configurations for various data sources
including Yahoo Finance, NSE, BSE, and economic data APIs.
"""

import os
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass


@dataclass
class APIEndpoint:
    """Data class for API endpoint configuration"""
    base_url: str
    timeout: int
    rate_limit: int
    headers: Dict[str, str]


class APIConfig:
    """
    Centralized API configuration management for Indian market data sources
    """
    
    def __init__(self, config_path: str = "config/api_keys/"):
        self.config_path = config_path
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all API configurations from YAML files"""
        try:
            # Yahoo Finance Configuration (Primary source for Indian stocks)
            self.yahoo_finance = {
                'base_url': 'https://query1.finance.yahoo.com/v8/finance/chart/',
                'timeout': 30,
                'rate_limit': 2000,  # requests per hour
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            }
            
            # NSE India Configuration (National Stock Exchange)
            self.nse_india = {
                'base_url': 'https://www.nseindia.com/api/',
                'timeout': 30,
                'rate_limit': 1000,
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br'
                }
            }
            
            # BSE India Configuration (Bombay Stock Exchange) 
            self.bse_india = {
                'base_url': 'https://api.bseindia.com/',
                'timeout': 30,
                'rate_limit': 500,
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                    'Accept': 'application/json'
                }
            }
            
            # Economic Data Configuration (RBI and other sources)
            self.economic_data = {
                'rbi_base_url': 'https://rbi.org.in/Scripts/PublicationsView.aspx',
                'trading_economics_url': 'https://api.tradingeconomics.com/country/india',
                'timeout': 45,
                'rate_limit': 100,
                'headers': {
                    'User-Agent': 'MarketResearchBot/1.0',
                    'Accept': 'application/json'
                }
            }
            
            # Alternative data sources
            self.alternative_sources = {
                'money_control': {
                    'base_url': 'https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php',
                    'timeout': 20,
                    'rate_limit': 1500
                },
                'investing_com': {
                    'base_url': 'https://api.investing.com/api/financialdata/',
                    'timeout': 25,
                    'rate_limit': 800
                }
            }
            
        except Exception as e:
            print(f"Error loading API configurations: {e}")
            self._set_default_configs()
    
    def _set_default_configs(self):
        """Set default configurations if loading fails"""
        self.yahoo_finance = {
            'base_url': 'https://query1.finance.yahoo.com/v8/finance/chart/',
            'timeout': 30,
            'rate_limit': 2000,
            'headers': {'User-Agent': 'Mozilla/5.0'}
        }
        
        self.nse_india = {
            'base_url': 'https://www.nseindia.com/api/',
            'timeout': 30,
            'rate_limit': 1000,
            'headers': {'User-Agent': 'Mozilla/5.0'}
        }
    
    def get_indian_stock_symbols(self) -> Dict[str, str]:
        """
        Returns dictionary of major Indian stock symbols with Yahoo Finance suffixes
        """
        return {
            # Nifty 50 Major Stocks
            'RELIANCE.NS': 'Reliance Industries',
            'TCS.NS': 'Tata Consultancy Services', 
            'HDFCBANK.NS': 'HDFC Bank',
            'INFY.NS': 'Infosys',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'ICICIBANK.NS': 'ICICI Bank',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'BHARTIARTL.NS': 'Bharti Airtel',
            'ITC.NS': 'ITC Limited',
            'SBIN.NS': 'State Bank of India',
            'ASIANPAINT.NS': 'Asian Paints',
            'MARUTI.NS': 'Maruti Suzuki',
            'HCLTECH.NS': 'HCL Technologies',
            'AXISBANK.NS': 'Axis Bank',
            'LT.NS': 'Larsen & Toubro',
            'WIPRO.NS': 'Wipro',
            'ULTRACEMCO.NS': 'UltraTech Cement',
            'DMART.NS': 'Avenue Supermarts',
            'TITAN.NS': 'Titan Company',
            'NESTLEIND.NS': 'Nestle India',
            'TECHM.NS': 'Tech Mahindra',
            'POWERGRID.NS': 'Power Grid Corporation',
            'NTPC.NS': 'NTPC Limited',
            'TATAMOTORS.NS': 'Tata Motors',
            'BAJFINANCE.NS': 'Bajaj Finance',
            'M&M.NS': 'Mahindra & Mahindra',
            'SUNPHARMA.NS': 'Sun Pharmaceutical',
            'JSWSTEEL.NS': 'JSW Steel',
            'TATASTEEL.NS': 'Tata Steel',
            'INDUSINDBK.NS': 'IndusInd Bank',
            'BAJAJFINSV.NS': 'Bajaj Finserv',
            'CIPLA.NS': 'Cipla',
            'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
            'EICHERMOT.NS': 'Eicher Motors',
            'GRASIM.NS': 'Grasim Industries',
            'HEROMOTOCO.NS': 'Hero MotoCorp',
            'BRITANNIA.NS': 'Britannia Industries',
            'COALINDIA.NS': 'Coal India',
            'UPL.NS': 'UPL Limited',
            'BPCL.NS': 'Bharat Petroleum Corporation',
            'ONGC.NS': 'Oil & Natural Gas Corporation',
            'IOC.NS': 'Indian Oil Corporation',
            'HINDALCO.NS': 'Hindalco Industries',
            'DIVISLAB.NS': 'Divi\'s Laboratories',
            'ADANIPORTS.NS': 'Adani Ports and SEZ',
            'TATACONSUM.NS': 'Tata Consumer Products',
            'SHREECEM.NS': 'Shree Cement',
            'APOLLOHOSP.NS': 'Apollo Hospitals',
            'GODREJCP.NS': 'Godrej Consumer Products',
            'BAJAJ-AUTO.NS': 'Bajaj Auto'
        }
    
    def get_indian_indices(self) -> Dict[str, str]:
        """
        Returns dictionary of major Indian market indices
        """
        return {
            '^NSEI': 'Nifty 50',
            '^BSESN': 'BSE Sensex',
            '^NSEBANK': 'Nifty Bank',
            '^NSEIT': 'Nifty IT',
            '^NSEFMCG': 'Nifty FMCG',
            'NIFTYNEXT50.NS': 'Nifty Next 50',
            'NIFTY500.NS': 'Nifty 500',
            'NIFTYSMLCAP100.NS': 'Nifty Smallcap 100',
            'NIFTYMIDCAP100.NS': 'Nifty Midcap 100'
        }
    
    def get_request_headers(self, source: str = 'yahoo') -> Dict[str, str]:
        """
        Get appropriate request headers for specific data source
        
        Args:
            source: Data source name ('yahoo', 'nse', 'bse')
            
        Returns:
            Dictionary of HTTP headers
        """
        headers_map = {
            'yahoo': self.yahoo_finance['headers'],
            'nse': self.nse_india['headers'],
            'bse': self.bse_india['headers']
        }
        
        return headers_map.get(source, self.yahoo_finance['headers'])
    
    def get_api_timeout(self, source: str = 'yahoo') -> int:
        """
        Get timeout setting for specific data source
        
        Args:
            source: Data source name
            
        Returns:
            Timeout value in seconds
        """
        timeout_map = {
            'yahoo': self.yahoo_finance['timeout'],
            'nse': self.nse_india['timeout'],
            'bse': self.bse_india['timeout']
        }
        
        return timeout_map.get(source, 30)
    
    def get_rate_limit(self, source: str = 'yahoo') -> int:
        """
        Get rate limit for specific data source
        
        Args:
            source: Data source name
            
        Returns:
            Rate limit (requests per hour)
        """
        rate_limit_map = {
            'yahoo': self.yahoo_finance['rate_limit'],
            'nse': self.nse_india['rate_limit'],
            'bse': self.bse_india['rate_limit']
        }
        
        return rate_limit_map.get(source, 1000)
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate if API keys are properly configured
        
        Returns:
            Dictionary with validation status for each API
        """
        validation_status = {}
        
        # Check Yahoo Finance (no API key required for basic usage)
        validation_status['yahoo_finance'] = True
        
        # Check NSE India (no API key required for public data)
        validation_status['nse_india'] = True
        
        # Check BSE India
        validation_status['bse_india'] = True
        
        # Check if premium API keys exist
        validation_status['premium_apis'] = os.path.exists(
            os.path.join(self.config_path, 'premium_keys.yaml')
        )
        
        return validation_status
    
    def get_sector_mapping(self) -> Dict[str, list]:
        """
        Returns sector-wise stock grouping for Indian market
        """
        return {
            'Banking & Finance': [
                'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
                'SBIN.NS', 'INDUSINDBK.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS'
            ],
            'Information Technology': [
                'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'
            ],
            'Oil & Gas': [
                'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS'
            ],
            'FMCG': [
                'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
                'TATACONSUM.NS', 'GODREJCP.NS'
            ],
            'Automotive': [
                'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'EICHERMOT.NS',
                'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS'
            ],
            'Pharmaceuticals': [
                'SUNPHARMA.NS', 'CIPLA.NS', 'DRREDDY.NS', 'DIVISLAB.NS',
                'APOLLOHOSP.NS'
            ],
            'Metals & Mining': [
                'TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'COALINDIA.NS'
            ],
            'Cement': [
                'ULTRACEMCO.NS', 'SHREECEM.NS', 'GRASIM.NS'
            ],
            'Infrastructure': [
                'LT.NS', 'POWERGRID.NS', 'NTPC.NS', 'ADANIPORTS.NS'
            ],
            'Consumer Goods': [
                'ASIANPAINT.NS', 'TITAN.NS', 'DMART.NS'
            ],
            'Telecom': [
                'BHARTIARTL.NS'
            ]
        }


# Global configuration instance
api_config = APIConfig()


def get_config():
    """
    Get the global API configuration instance
    
    Returns:
        APIConfig instance
    """
    return api_config


if __name__ == "__main__":
    # Test the configuration
    config = APIConfig()
    
    print("API Configuration Test")
    print("=" * 30)
    
    # Test stock symbols
    symbols = config.get_indian_stock_symbols()
    print(f"Total Indian stocks configured: {len(symbols)}")
    print("Sample stocks:", list(symbols.keys())[:5])
    
    # Test indices
    indices = config.get_indian_indices()
    print(f"Total indices configured: {len(indices)}")
    print("Indices:", list(indices.keys()))
    
    # Test sectors
    sectors = config.get_sector_mapping()
    print(f"Total sectors: {len(sectors)}")
    print("Sectors:", list(sectors.keys()))
    
    # Test validation
    validation = config.validate_api_keys()
    print("API Validation:", validation)