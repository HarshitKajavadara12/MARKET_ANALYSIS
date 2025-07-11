"""
Market Research System v1.0 - Custom Exceptions Package
======================================================

This package contains custom exception classes for handling various error
scenarios in the market research system.

Created: 2022
Author: Market Research System
Version: 1.0

Exception Categories:
- Data Exceptions: Errors related to data fetching, validation, and processing
- API Exceptions: Errors related to external API interactions
- Analysis Exceptions: Errors related to technical analysis and calculations

Usage:
    from src.exceptions import DataFetchError, APIConnectionError, AnalysisError
    
    try:
        # Your code here
        pass
    except DataFetchError as e:
        logger.error(f"Data fetch failed: {e}")
    except APIConnectionError as e:
        logger.error(f"API connection failed: {e}")
    except AnalysisError as e:
        logger.error(f"Analysis failed: {e}")
"""

# Import all custom exceptions for easy access
from .data_exceptions import (
    DataFetchError,
    DataValidationError,
    DataProcessingError,
    DataStorageError,
    InvalidDateRangeError,
    MissingDataError,
    DataFormatError,
    DuplicateDataError
)

from .api_exceptions import (
    APIConnectionError,
    APIAuthenticationError,
    APIRateLimitError,
    APIResponseError,
    YahooFinanceError,
    FREDAPIError,
    InvalidAPIKeyError,
    APITimeoutError
)

from .analysis_exceptions import (
    AnalysisError,
    TechnicalIndicatorError,
    InsufficientDataError,
    CalculationError,
    IndicatorParameterError,
    CorrelationAnalysisError,
    TrendAnalysisError,
    VolatilityAnalysisError
)

# Base exception class
class MarketResearchError(Exception):
    """Base exception class for Market Research System."""
    pass

# Export all exceptions
__all__ = [
    # Base exception
    'MarketResearchError',
    
    # Data exceptions
    'DataFetchError',
    'DataValidationError', 
    'DataProcessingError',
    'DataStorageError',
    'InvalidDateRangeError',
    'MissingDataError',
    'DataFormatError',
    'DuplicateDataError',
    
    # API exceptions
    'APIConnectionError',
    'APIAuthenticationError',
    'APIRateLimitError',
    'APIResponseError',
    'YahooFinanceError',
    'FREDAPIError',
    'InvalidAPIKeyError',
    'APITimeoutError',
    
    # Analysis exceptions
    'AnalysisError',
    'TechnicalIndicatorError',
    'InsufficientDataError',
    'CalculationError',
    'IndicatorParameterError',
    'CorrelationAnalysisError',
    'TrendAnalysisError',
    'VolatilityAnalysisError'
]

# Version information
__version__ = "1.0.0"
__author__ = "Market Research System"
__email__ = "info@marketresearch.com"
__status__ = "Production"