"""
Market Research System v1.0 - Data Exceptions
==============================================

Custom exception classes for data-related errors in the market research system.
These exceptions handle various scenarios that can occur during data fetching,
validation, processing, and storage operations.

Created: 2022
Author: Market Research System
Version: 1.0

Exception Hierarchy:
    DataFetchError
    DataValidationError
    DataProcessingError
    DataStorageError
    InvalidDateRangeError
    MissingDataError
    DataFormatError
    DuplicateDataError
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Setup logger
logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """
    Raised when data fetching operations fail.
    
    This exception is raised when there are issues retrieving data from
    external sources like Yahoo Finance, FRED, or other data providers.
    
    Attributes:
        message (str): Error message
        source (str): Data source that failed
        symbol (str): Stock symbol or identifier related to the error
        timestamp (datetime): When the error occurred
        details (dict): Additional error details
    """
    
    def __init__(self, message: str, source: str = None, symbol: str = None, 
                 details: Dict[str, Any] = None):
        self.message = message
        self.source = source or "Unknown"
        self.symbol = symbol or "Unknown"
        self.timestamp = datetime.now()
        self.details = details or {}
        
        # Create detailed error message
        error_msg = f"Data Fetch Error: {message}"
        if source:
            error_msg += f" [Source: {source}]"
        if symbol:
            error_msg += f" [Symbol: {symbol}]"
            
        super().__init__(error_msg)
        logger.error(f"DataFetchError: {error_msg}, Details: {self.details}")


class DataValidationError(Exception):
    """
    Raised when data validation fails.
    
    This exception is raised when fetched data doesn't meet expected
    validation criteria such as missing required columns, invalid data types,
    or data quality issues.
    
    Attributes:
        message (str): Error message
        validation_rule (str): The validation rule that failed
        data_info (dict): Information about the invalid data
    """
    
    def __init__(self, message: str, validation_rule: str = None, 
                 data_info: Dict[str, Any] = None):
        self.message = message
        self.validation_rule = validation_rule or "Unknown"
        self.data_info = data_info or {}
        self.timestamp = datetime.now()
        
        error_msg = f"Data Validation Error: {message}"
        if validation_rule:
            error_msg += f" [Rule: {validation_rule}]"
            
        super().__init__(error_msg)
        logger.error(f"DataValidationError: {error_msg}, Data Info: {self.data_info}")


class DataProcessingError(Exception):
    """
    Raised when data processing operations fail.
    
    This exception is raised during data cleaning, transformation,
    or calculation operations that encounter unexpected issues.
    
    Attributes:
        message (str): Error message
        operation (str): The processing operation that failed
        data_shape (tuple): Shape of data being processed
        error_details (dict): Additional error information
    """
    
    def __init__(self, message: str, operation: str = None, 
                 data_shape: tuple = None, error_details: Dict[str, Any] = None):
        self.message = message
        self.operation = operation or "Unknown"
        self.data_shape = data_shape
        self.error_details = error_details or {}
        self.timestamp = datetime.now()
        
        error_msg = f"Data Processing Error: {message}"
        if operation:
            error_msg += f" [Operation: {operation}]"
        if data_shape:
            error_msg += f" [Data Shape: {data_shape}]"
            
        super().__init__(error_msg)
        logger.error(f"DataProcessingError: {error_msg}, Details: {self.error_details}")


class DataStorageError(Exception):
    """
    Raised when data storage operations fail.
    
    This exception is raised when there are issues saving data to files,
    databases, or cache storage systems.
    
    Attributes:
        message (str): Error message
        storage_type (str): Type of storage that failed
        file_path (str): File path if applicable
        error_code (int): Error code if available
    """
    
    def __init__(self, message: str, storage_type: str = None, 
                 file_path: str = None, error_code: int = None):
        self.message = message
        self.storage_type = storage_type or "Unknown"
        self.file_path = file_path
        self.error_code = error_code
        self.timestamp = datetime.now()
        
        error_msg = f"Data Storage Error: {message}"
        if storage_type:
            error_msg += f" [Storage: {storage_type}]"
        if file_path:
            error_msg += f" [Path: {file_path}]"
        if error_code:
            error_msg += f" [Code: {error_code}]"
            
        super().__init__(error_msg)
        logger.error(f"DataStorageError: {error_msg}")


class InvalidDateRangeError(Exception):
    """
    Raised when an invalid date range is specified.
    
    This exception is raised when date parameters are invalid,
    such as start date after end date, or dates outside valid ranges.
    
    Attributes:
        message (str): Error message
        start_date (datetime): Start date that caused the error
        end_date (datetime): End date that caused the error
    """
    
    def __init__(self, message: str, start_date: datetime = None, 
                 end_date: datetime = None):
        self.message = message
        self.start_date = start_date
        self.end_date = end_date
        self.timestamp = datetime.now()
        
        error_msg = f"Invalid Date Range Error: {message}"
        if start_date:
            error_msg += f" [Start: {start_date.strftime('%Y-%m-%d')}]"
        if end_date:
            error_msg += f" [End: {end_date.strftime('%Y-%m-%d')}]"
            
        super().__init__(error_msg)
        logger.error(f"InvalidDateRangeError: {error_msg}")


class MissingDataError(Exception):
    """
    Raised when required data is missing.
    
    This exception is raised when expected data columns, files,
    or data points are not available for analysis.
    
    Attributes:
        message (str): Error message
        missing_items (list): List of missing data items
        data_source (str): Source where data is missing
    """
    
    def __init__(self, message: str, missing_items: list = None, 
                 data_source: str = None):
        self.message = message
        self.missing_items = missing_items or []
        self.data_source = data_source or "Unknown"
        self.timestamp = datetime.now()
        
        error_msg = f"Missing Data Error: {message}"
        if data_source:
            error_msg += f" [Source: {data_source}]"
        if missing_items:
            error_msg += f" [Missing: {', '.join(map(str, missing_items))}]"
            
        super().__init__(error_msg)
        logger.error(f"MissingDataError: {error_msg}")


class DataFormatError(Exception):
    """
    Raised when data is in an unexpected format.
    
    This exception is raised when data doesn't match expected formats,
    such as incorrect file formats, unexpected column names, or data types.
    
    Attributes:
        message (str): Error message
        expected_format (str): Expected data format
        actual_format (str): Actual data format found
        file_path (str): File path if applicable
    """
    
    def __init__(self, message: str, expected_format: str = None, 
                 actual_format: str = None, file_path: str = None):
        self.message = message
        self.expected_format = expected_format
        self.actual_format = actual_format
        self.file_path = file_path
        self.timestamp = datetime.now()
        
        error_msg = f"Data Format Error: {message}"
        if expected_format:
            error_msg += f" [Expected: {expected_format}]"
        if actual_format:
            error_msg += f" [Actual: {actual_format}]"
        if file_path:
            error_msg += f" [File: {file_path}]"
            
        super().__init__(error_msg)
        logger.error(f"DataFormatError: {error_msg}")


class DuplicateDataError(Exception):
    """
    Raised when duplicate data is detected where it shouldn't exist.
    
    This exception is raised when duplicate records are found in
    datasets that should contain unique entries.
    
    Attributes:
        message (str): Error message
        duplicate_count (int): Number of duplicates found
        duplicate_keys (list): Keys that are duplicated
        data_source (str): Source containing duplicates
    """
    
    def __init__(self, message: str, duplicate_count: int = 0, 
                 duplicate_keys: list = None, data_source: str = None):
        self.message = message
        self.duplicate_count = duplicate_count
        self.duplicate_keys = duplicate_keys or []
        self.data_source = data_source or "Unknown"
        self.timestamp = datetime.now()
        
        error_msg = f"Duplicate Data Error: {message}"
        if duplicate_count > 0:
            error_msg += f" [Count: {duplicate_count}]"
        if data_source:
            error_msg += f" [Source: {data_source}]"
        if duplicate_keys:
            error_msg += f" [Keys: {', '.join(map(str, duplicate_keys[:5]))}...]"
            
        super().__init__(error_msg)
        logger.error(f"DuplicateDataError: {error_msg}")


# Utility function to create standardized error reporting
def create_data_error_report(exception: Exception) -> Dict[str, Any]:
    """
    Create a standardized error report for data exceptions.
    
    Args:
        exception (Exception): The exception to report
        
    Returns:
        dict: Standardized error report
    """
    error_report = {
        'error_type': type(exception).__name__,
        'message': str(exception),
        'timestamp': datetime.now().isoformat(),
        'details': {}
    }
    
    # Add specific details based on exception type
    if hasattr(exception, 'source'):
        error_report['details']['source'] = exception.source
    if hasattr(exception, 'symbol'):
        error_report['details']['symbol'] = exception.symbol
    if hasattr(exception, 'operation'):
        error_report['details']['operation'] = exception.operation
    if hasattr(exception, 'file_path'):
        error_report['details']['file_path'] = exception.file_path
    if hasattr(exception, 'validation_rule'):
        error_report['details']['validation_rule'] = exception.validation_rule
        
    return error_report