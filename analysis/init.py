"""
Market Research System v1.0 - Analysis Package
Indian Stock Market Analysis Module

This package provides comprehensive market analysis tools including:
- Technical indicators calculation
- Statistical market analysis
- Correlation analysis
- Trend analysis
- Volatility analysis
- Performance metrics
- Statistical utilities

Created: January 2022
Author: Independent Market Researcher
Focus: Indian Stock Market (NSE/BSE)
"""

from .technical_indicators import TechnicalIndicators
from .market_analyzer import MarketAnalyzer
from .correlation_analyzer import CorrelationAnalyzer
from .trend_analyzer import TrendAnalyzer
from .volatility_analyzer import VolatilityAnalyzer
from .performance_analyzer import PerformanceAnalyzer
from .statistical_utils import StatisticalUtils

__version__ = "1.0.0"
__author__ = "Independent Market Researcher"
__email__ = "research@indianmarkets.com"

# Package metadata
__all__ = [
    'TechnicalIndicators',
    'MarketAnalyzer',
    'CorrelationAnalyzer',
    'TrendAnalyzer',
    'VolatilityAnalyzer',
    'PerformanceAnalyzer',
    'StatisticalUtils'
]

# Indian Market specific configurations
INDIAN_MARKET_CONFIG = {
    'trading_days_per_year': 250,
    'risk_free_rate': 0.06,  # 6% as of 2022
    'market_hours': {
        'pre_open': '09:00',
        'normal_open': '09:15',  
        'normal_close': '15:30',
        'post_close': '16:00'
    },
    'major_indices': [
        'NIFTY50',
        'SENSEX',
        'NIFTY_BANK',
        'NIFTY_IT',
        'NIFTY_AUTO',
        'NIFTY_PHARMA',
        'NIFTY_FMCG',
        'NIFTY_METAL',
        'NIFTY_REALTY'
    ],
    'exchanges': ['NSE', 'BSE']
}

def get_indian_market_config():
    """
    Returns Indian market specific configuration
    
    Returns:
        dict: Configuration dictionary for Indian markets
    """
    return INDIAN_MARKET_CONFIG

def initialize_analysis_suite():
    """
    Initialize the complete analysis suite for Indian markets
    
    Returns:
        dict: Dictionary containing all analysis classes
    """
    return {
        'technical': TechnicalIndicators(),
        'market': MarketAnalyzer(),
        'correlation': CorrelationAnalyzer(),
        'trend': TrendAnalyzer(),
        'volatility': VolatilityAnalyzer(),
        'performance': PerformanceAnalyzer(),
        'stats': StatisticalUtils()
    }