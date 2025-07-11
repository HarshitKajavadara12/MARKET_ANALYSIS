"""
Market Research System v1.0 - Configuration Settings
File: src/config/settings.py
Created: January 2022

Main configuration settings for the Indian market research system.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Create directories if they don't exist
for directory in [DATA_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
    for subdir in ['raw', 'processed', 'cache']:
        if directory == DATA_DIR:
            (directory / subdir).mkdir(exist_ok=True)
    if directory == REPORTS_DIR:
        for subdir in ['daily', 'weekly', 'monthly', 'templates']:
            (directory / subdir).mkdir(exist_ok=True)

# Indian Stock Market Configuration
class IndianMarketConfig:
    """Configuration for Indian stock market data"""
    
    # NSE Top 100 Stocks (Nifty 100)
    NIFTY_100_SYMBOLS = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 'INFY.NS',
        'HDFC.NS', 'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'ASIANPAINT.NS', 'LT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
        'ULTRACEMCO.NS', 'TITAN.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'ONGC.NS',
        'NTPC.NS', 'POWERGRID.NS', 'M&M.NS', 'HCLTECH.NS', 'BAJFINANCE.NS',
        'TECHM.NS', 'DIVISLAB.NS', 'HINDALCO.NS', 'COALINDIA.NS', 'DRREDDY.NS',
        'ADANIPORTS.NS', 'GRASIM.NS', 'CIPLA.NS', 'BRITANNIA.NS', 'BPCL.NS',
        'IOC.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'BAJAJFINSERV.NS', 'UPL.NS',
        'TATACONSUM.NS', 'TATASTEEL.NS', 'BAJAJ-AUTO.NS', 'APOLLOHOSP.NS',
        'SBILIFE.NS', 'INDUSINDBK.NS', 'HDFCLIFE.NS', 'GODREJCP.NS', 'PIDILITIND.NS',
        'DABUR.NS', 'MARICO.NS', 'COLPAL.NS', 'BERGEPAINT.NS', 'VOLTAS.NS',
        'HAVELLS.NS', 'MCDOWELL-N.NS', 'AMBUJACEM.NS', 'ACC.NS', 'LUPIN.NS',
        'BIOCON.NS', 'CONCOR.NS', 'NMDC.NS', 'VEDL.NS', 'JSW.NS',
        'SAIL.NS', 'JINDALSTEL.NS', 'HINDZINC.NS', 'NATIONALUM.NS', 'MOIL.NS',
        'RECLTD.NS', 'PFC.NS', 'LICHSGFIN.NS', 'CHOLAFIN.NS', 'MUTHOOTFIN.NS',
        'SHREECEM.NS', 'RAMCOCEM.NS', 'JKCEMENT.NS', 'HEIDELBERG.NS', 'STAR.NS',
        'DMART.NS', 'TRENT.NS', 'JUBLFOOD.NS', 'PAGEIND.NS', 'ZEEL.NS',
        'BALKRISIND.NS', 'MRF.NS', 'APOLLOTYRE.NS', 'CEAT.NS', 'JK.NS',
        'MINDTREE.NS', 'MPHASIS.NS', 'COFORGE.NS', 'LTTS.NS', 'PERSISTENT.NS',
        'NAUKRI.NS', 'ZOMATO.NS', 'PAYTM.NS', 'POLICYBZR.NS', 'FSBL.NS',
        'RBLBANK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'BANDHANBNK.NS', 'PNB.NS'
    ]
    
    # BSE Sensex 30 Companies
    SENSEX_30_SYMBOLS = [
        'RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'HINDUNILVR.BO', 'INFY.BO',
        'HDFC.BO', 'ICICIBANK.BO', 'SBIN.BO', 'BHARTIARTL.BO', 'ITC.BO',
        'ASIANPAINT.BO', 'LT.BO', 'AXISBANK.BO', 'MARUTI.BO', 'SUNPHARMA.BO',
        'ULTRACEMCO.BO', 'TITAN.BO', 'WIPRO.BO', 'NESTLEIND.BO', 'ONGC.BO',
        'NTPC.BO', 'POWERGRID.BO', 'M&M.BO', 'HCLTECH.BO', 'BAJFINANCE.BO',
        'TECHM.BO', 'DIVISLAB.BO', 'HINDALCO.BO', 'COALINDIA.BO', 'DRREDDY.BO'
    ]
    
    # Indian Market Indices
    INDICES = {
        'NIFTY_50': '^NSEI',
        'SENSEX': '^BSESN',
        'NIFTY_BANK': '^NSEBANK',
        'NIFTY_IT': '^CNXIT',
        'NIFTY_PHARMA': '^CNXPHARMA',
        'NIFTY_FMCG': '^CNXFMCG',
        'NIFTY_AUTO': '^CNXAUTO',
        'NIFTY_METAL': '^CNXMETAL',
        'NIFTY_REALTY': '^CNXREALTY',
        'NIFTY_ENERGY': '^CNXENERGY'
    }
    
    # Sector Classification
    SECTORS = {
        'BANKING': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'INDUSINDBK.NS', 'PNB.NS'],
        'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTTS.NS'],
        'PHARMA': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'LUPIN.NS', 'BIOCON.NS', 'DIVISLAB.NS'],
        'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS', 'MARICO.NS'],
        'AUTO': ['MARUTI.NS', 'M&M.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'TATAMOTORS.NS'],
        'ENERGY': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'COALINDIA.NS', 'NTPC.NS'],
        'METAL': ['TATASTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'JSW.NS', 'SAIL.NS', 'JINDALSTEL.NS'],
        'CEMENT': ['ULTRACEMCO.NS', 'SHREECEM.NS', 'ACC.NS', 'AMBUJACEM.NS', 'RAMCOCEM.NS', 'JKCEMENT.NS']
    }
    
    # Trading hours (IST)
    MARKET_OPEN_TIME = "09:15"
    MARKET_CLOSE_TIME = "15:30"
    PRE_MARKET_START = "09:00"
    POST_MARKET_END = "16:00"

class DataConfig:
    """Data collection and storage configuration"""
    
    # Data sources
    YAHOO_FINANCE_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
    NSE_BASE_URL = "https://www.nseindia.com/api/"
    BSE_BASE_URL = "https://api.bseindia.com/"
    
    # Data refresh intervals (in minutes)
    REALTIME_REFRESH_INTERVAL = 5
    DAILY_REFRESH_INTERVAL = 60
    WEEKLY_REFRESH_INTERVAL = 10080  # 7 days
    
    # Historical data periods
    DEFAULT_HISTORY_PERIOD = "2y"  # 2 years
    MAX_HISTORY_PERIOD = "10y"     # 10 years
    
    # Data quality checks
    MIN_DATA_POINTS = 100
    MAX_MISSING_DATA_RATIO = 0.05  # 5%
    
    # File formats
    RAW_DATA_FORMAT = "csv"
    PROCESSED_DATA_FORMAT = "parquet"
    CACHE_FORMAT = "pkl"

class AnalysisConfig:
    """Analysis parameters and settings"""
    
    # Technical Indicators Parameters
    TECHNICAL_INDICATORS = {
        'SMA': [5, 10, 20, 50, 100, 200],
        'EMA': [5, 10, 12, 20, 26, 50, 100, 200],
        'RSI': [14, 21],
        'MACD': [(12, 26, 9)],
        'BOLLINGER_BANDS': [(20, 2)],
        'STOCHASTIC': [(14, 3, 3)],
        'ADX': [14],
        'CCI': [20],
        'WILLIAMS_R': [14],
        'ROC': [12],
        'ATR': [14],
        'OBV': None,
        'VWAP': None
    }
    
    # Thresholds for signals
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    VOLUME_SPIKE_THRESHOLD = 2.0  # 2x average volume
    PRICE_CHANGE_SIGNIFICANT = 0.05  # 5%
    
    # Correlation analysis
    CORRELATION_PERIOD = 252  # 1 year trading days
    CORRELATION_THRESHOLD = 0.7
    
    # Risk metrics
    VAR_CONFIDENCE_LEVEL = 0.05  # 95% VaR
    SHARPE_RATIO_RISK_FREE_RATE = 0.06  # 6% (approx Indian govt bond rate in 2022)

class ReportingConfig:
    """Reporting and notification settings"""
    
    # Report generation settings
    REPORT_FORMATS = ['pdf', 'html', 'excel']
    DEFAULT_REPORT_FORMAT = 'pdf'
    
    # Chart settings
    CHART_STYLE = 'seaborn-v0_8'
    CHART_DPI = 300
    CHART_FIGSIZE = (12, 8)
    COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Email settings
    EMAIL_SUBJECT_PREFIX = "[Indian Market Research]"
    EMAIL_SIGNATURE = """
    ---
    Indian Market Research System v1.0
    Automated Market Analysis & Insights
    Contact: research@marketanalysis.in
    """
    
    # Report sections
    DAILY_REPORT_SECTIONS = [
        'market_summary',
        'index_performance',
        'top_gainers_losers',
        'sector_analysis',
        'technical_signals',
        'volume_analysis'
    ]
    
    WEEKLY_REPORT_SECTIONS = [
        'weekly_summary',
        'sector_performance',
        'correlation_analysis',
        'volatility_analysis',
        'support_resistance_levels'
    ]

class SystemConfig:
    """System-wide configuration"""
    
    # Logging configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Performance settings
    MAX_CONCURRENT_DOWNLOADS = 10
    REQUEST_TIMEOUT = 30  # seconds
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 5  # seconds
    
    # Cache settings
    CACHE_EXPIRY_HOURS = 24
    MAX_CACHE_SIZE_MB = 500
    
    # Database settings (if using SQLite for local storage)
    DATABASE_PATH = DATA_DIR / "market_data.db"
    DATABASE_TIMEOUT = 30

# Environment-specific settings
def load_environment_config():
    """Load configuration from environment variables"""
    return {
        'YAHOO_API_KEY': os.getenv('YAHOO_FINANCE_API_KEY'),
        'NSE_API_KEY': os.getenv('NSE_API_KEY'),
        'EMAIL_CONFIG': {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'recipients': os.getenv('REPORT_RECIPIENTS', '').split(',')
        },
        'DATA_PATH': os.getenv('DATA_PATH', str(DATA_DIR)),
        'REPORTS_PATH': os.getenv('REPORTS_PATH', str(REPORTS_DIR)),
        'LOGS_PATH': os.getenv('LOGS_PATH', str(LOGS_DIR))
    }

# Initialize configuration
ENV_CONFIG = load_environment_config()

# Export main configuration classes
__all__ = [
    'IndianMarketConfig',
    'DataConfig', 
    'AnalysisConfig',
    'ReportingConfig',
    'SystemConfig',
    'ENV_CONFIG',
    'BASE_DIR',
    'DATA_DIR',
    'REPORTS_DIR',
    'LOGS_DIR'
]