"""
Market Research System v1.0 - API Exceptions
===========================================

Custom exception classes for API-related errors in the market research system.
These exceptions handle various scenarios that can occur during interactions
with external APIs like Yahoo Finance, FRED, and other data providers.

Created: 2022
Author: Market Research System
Version: 1.0

Exception Hierarchy:
    APIConnectionError
    APIAuthenticationError
    APIRateLimitError
    APIResponseError
    YahooFinanceError
    FREDAPIError
    InvalidAPIKeyError
    APITimeoutError
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, Union

# Setup logger
logger = logging.getLogger(__name__)


class APIConnectionError(Exception):
    """
    Raised when API connection fails.
    
    This exception is raised when there are network connectivity issues
    or when the API endpoint is unreachable.
    
    Attributes:
        message (str): Error message
        api_name (str): Name of the API that failed
        endpoint (str): API endpoint that failed
        status_code (int): HTTP status code if available
        retry_count (int): Number of retries attempted
    """
    
    def __init__(self, message: str, api_name: str = None, endpoint: str = None,
                 status_code: int = None, retry_count: int = 0):
        self.message = message
        self.api_name = api_name or "Unknown API"
        self.endpoint = endpoint
        self.status_code = status_code
        self.retry_count = retry_count
        self.timestamp = datetime.now()
        
        error_msg = f"API Connection Error: {message}"
        if api_name:
            error_msg += f" [API: {api_name}]"
        if endpoint:
            error_msg += f" [Endpoint: {endpoint}]"
        if status_code:
            error_msg += f" [Status: {status_code}]"
        if retry_count > 0:
            error_msg += f" [Retries: {retry_count}]"
            
        super().__init__(error_msg)
        logger.error(f"APIConnectionError: {error_msg}")


class APIAuthenticationError(Exception):
    """
    Raised when API authentication fails.
    
    This exception is raised when API keys are invalid, expired,
    or authentication credentials are incorrect.
    
    Attributes:
        message (str): Error message
        api_name (str): Name of the API that failed authentication
        auth_type (str): Type of authentication used
        key_info (str): Partial key information (masked)
    """
    
    def __init__(self, message: str, api_name: str = None, 
                 auth_type: str = None, key_info: str = None):
        self.message = message
        self.api_name = api_name or "Unknown API"
        self.auth_type = auth_type or "Unknown"
        self.key_info = key_info
        self.timestamp = datetime.now()
        
        error_msg = f"API Authentication Error: {message}"
        if api_name:
            error_msg += f" [API: {api_name}]"
        if auth_type:
            error_msg += f" [Auth Type: {auth_type}]"
        if key_info:
            error_msg += f" [Key: {key_info}]"
            
        super().__init__(error_msg)
        logger.error(f"APIAuthenticationError: {error_msg}")


class APIRateLimitError(Exception):
    """
    Raised when API rate limits are exceeded.
    
    This exception is raised when the number of API requests
    exceeds the allowed rate limits.
    
    Attributes:
        message (str): Error message
        api_name (str): Name of the API that hit rate limit
        limit (int): Rate limit value
        reset_time (datetime): When the rate limit resets
        requests_made (int): Number of requests made
    """
    
    def __init__(self, message: str, api_name: str = None, limit: int = None,
                 reset_time: datetime = None, requests_made: int = None):
        self.message = message
        self.api_name = api_name or "Unknown API"
        self.limit = limit
        self.reset_time = reset_time
        self.requests_made = requests_made
        self.timestamp = datetime.now()
        
        error_msg = f"API Rate Limit Error: {message}"
        if api_name:
            error_msg += f" [API: {api_name}]"
        if limit:
            error_msg += f" [Limit: {limit}]"
        if reset_time:
            error_msg += f" [Reset: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}]"
        if requests_made:
            error_msg += f" [Requests: {requests_made}]"
            
        super().__init__(error_msg)
        logger.error(f"APIRateLimitError: {error_msg}")


class APIResponseError(Exception):
    """
    Raised when API returns an unexpected response.
    
    This exception is raised when API responses are malformed,
    contain errors, or don't match expected formats.
    
    Attributes:
        message (str): Error message
        api_name (str): Name of the API
        status_code (int): HTTP status code
        response_data (dict): Response data if available
        request_info (dict): Information about the request made
    """
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None,
                 response_data: Dict[str, Any] = None, request_info: Dict[str, Any] = None):
        self.message = message
        self.api_name = api_name or "Unknown API"
        self.status_code = status_code
        self.response_data = response_data or {}
        self.request_info = request_info or {}
        self.timestamp = datetime.now()
        
        error_msg = f"API Response Error: {message}"
        if api_name:
            error_msg += f" [API: {api_name}]"
        if status_code:
            error_msg += f" [Status: {status_code}]"
            
        super().__init__(error_msg)
        logger.error(f"APIResponseError: {error_msg}, Response: {self.response_data}")


class YahooFinanceError(Exception):
    """
    Raised when Yahoo Finance API operations fail.
    
    This exception is specific to Yahoo Finance API errors,
    including symbol not found, data unavailable, etc.
    
    Attributes:
        message (str): Error message
        symbol (str): Stock symbol that caused the error
        data_type (str): Type of data requested
        period (str): Time period requested
        error_code (str): Yahoo Finance specific error code
    """
    
    def __init__(self, message: str, symbol: str = None, data_type: str = None,
                 period: str = None, error_code: str = None):
        self.message = message
        self.symbol = symbol
        self.data_type = data_type or "Unknown"
        self.period = period
        self.error_code = error_code
        self.timestamp = datetime.now()
        
        error_msg = f"Yahoo Finance Error: {message}"
        if symbol:
            error_msg += f" [Symbol: {symbol}]"
        if data_type:
            error_msg += f" [Data Type: {data_type}]"
        if period:
            error_msg += f" [Period: {period}]"
        if error_code:
            error_msg += f" [Error Code: {error_code}]"
            
        super().__init__(error_msg)
        logger.error(f"YahooFinanceError: {error_msg}")


class FREDAPIError(Exception):
    """
    Raised when FRED API operations fail.
    
    This exception is specific to Federal Reserve Economic Data (FRED) API errors,
    including series not found, data unavailable, etc.
    
    Attributes:
        message (str): Error message
        series_id (str): FRED series ID that caused the error
        observation_start (str): Start date for observations
        observation_end (str): End date for observations
        fred_error_code (str): FRED specific error code
    """
    
    def __init__(self, message: str, series_id: str = None, 
                 observation_start: str = None, observation_end: str = None,
                 fred_error_code: str = None):
        self.message = message
        self.series_id = series_id
        self.observation_start = observation_start
        self.observation_end = observation_end
        self.fred_error_code = fred_error_code
        self.timestamp = datetime.now()
        
        error_msg = f"FRED API Error: {message}"
        if series_id:
            error_msg += f" [Series: {series_id}]"
        if observation_start:
            error_msg += f" [Start: {observation_start}]"
        if observation_end:
            error_msg += f" [End: {observation_end}]"
        if fred_error_code:
            error_msg += f" [FRED Code: {fred_error_code}]"
            
        super().__init__(error_msg)
        logger.error(f"FREDAPIError: {error_msg}")


class InvalidAPIKeyError(Exception):
    """
    Raised when API key is invalid or missing.
    
    This exception is raised when API keys are not provided,
    are malformed, or don't have required permissions.
    
    Attributes:
        message (str): Error message
        api_name (str): Name of the API
        key_format (str): Expected key format
        provided_key (str): Partial key information (masked)
    """
    
    def __init__(self, message: str, api_name: str = None, 
                 key_format: str = None, provided_key: str = None):
        self.message = message
        self.api_name = api_name or "Unknown API"
        self.key_format = key_format
        self.provided_key = self._mask_key(provided_key) if provided_key else None
        self.timestamp = datetime.now()
        
        error_msg = f"Invalid API Key Error: {message}"
        if api_name:
            error_msg += f" [API: {api_name}]"
        if key_format:
            error_msg += f" [Expected Format: {key_format}]"
        if self.provided_key:
            error_msg += f" [Provided: {self.provided_key}]"
            
        super().__init__(error_msg)
        logger.error(f"InvalidAPIKeyError: {error_msg}")
    
    def _mask_key(self, key: str) -> str:
        """Mask API key for logging purposes."""
        if not key or len(key) < 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"


class APITimeoutError(Exception):
    """
    Raised when API requests timeout.
    
    This exception is raised when API requests take longer
    than the specified timeout period.
    
    Attributes:
        message (str): Error message
        api_name (str): Name of the API
        timeout_duration (float): Timeout duration in seconds
        endpoint (str): API endpoint that timed out
        request_size (int): Size of the request if available
    """
    
    def __init__(self, message: str, api_name: str = None, 
                 timeout_duration: float = None, endpoint: str = None,
                 request_size: int = None):
        self.message = message
        self.api_name = api_name or "Unknown API"
        self.timeout_duration = timeout_duration
        self.endpoint = endpoint
        self.request_size = request_size
        self.timestamp = datetime.now()
        
        error_msg = f"API Timeout Error: {message}"
        if api_name:
            error_msg += f" [API: {api_name}]"
        if timeout_duration:
            error_msg += f" [Timeout: {timeout_duration}s]"
        if endpoint:
            error_msg += f" [Endpoint: {endpoint}]"
        if request_size:
            error_msg += f" [Request Size: {request_size} bytes]"
            
        super().__init__(error_msg)
        logger.error(f"APITimeoutError: {error_msg}")


# Utility functions for API error handling
def create_api_error_report(exception: Exception) -> Dict[str, Any]:
    """
    Create a standardized error report for API exceptions.
    
    Args:
        exception (Exception): The exception to report
        
    Returns:
        dict: Standardized error report
    """
    error_report = {
        'error_type': type(exception).__name__,
        'message': str(exception),
        'timestamp': datetime.now().isoformat(),
        'api_details': {}
    }
    
    # Add specific details based on exception type
    if hasattr(exception, 'api_name'):
        error_report['api_details']['api_name'] = exception.api_name
    if hasattr(exception, 'endpoint'):
        error_report['api_details']['endpoint'] = exception.endpoint
    if hasattr(exception, 'status_code'):
        error_report['api_details']['status_code'] = exception.status_code
    if hasattr(exception, 'symbol'):
        error_report['api_details']['symbol'] = exception.symbol
    if hasattr(exception, 'series_id'):
        error_report['api_details']['series_id'] = exception.series_id
    if hasattr(exception, 'retry_count'):
        error_report['api_details']['retry_count'] = exception.retry_count
        
    return error_report


def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an API error is retryable.
    
    Args:
        exception (Exception): The exception to check
        
    Returns:
        bool: True if the error is retryable, False otherwise
    """
    retryable_errors = [
        APIConnectionError,
        APITimeoutError,
        APIRateLimitError
    ]
    
    # Check if it's a retryable error type
    if type(exception) in retryable_errors:
        return True
    
    # Check specific status codes for retryable errors
    if hasattr(exception, 'status_code'):
        retryable_status_codes = [429, 500, 502, 503, 504]
        return exception.status_code in retryable_status_codes
    
    return False


def get_retry_delay(exception: Exception) -> int:
    """
    Get recommended retry delay based on exception type.
    
    Args:
        exception (Exception): The exception to get delay for
        
    Returns:
        int: Recommended delay in seconds
    """
    if isinstance(exception, APIRateLimitError):
        # For rate limit errors, wait longer
        return 60
    elif isinstance(exception, APITimeoutError):
        # For timeout errors, wait moderate amount
        return 30
    elif isinstance(exception, APIConnectionError):
        # For connection errors, wait short amount
        return 10
    else:
        # Default retry delay
        return 5