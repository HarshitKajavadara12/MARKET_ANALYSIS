"""
Logging Configuration Module
Market Research System v1.0 (2022)
Author: Independent Market Researcher
Focus: Indian Stock Market Analysis

This module provides centralized logging configuration for the entire system.
Supports multiple log levels, file rotation, and structured logging.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


class LoggerConfig:
    """
    Centralized logging configuration for market research system.
    Provides structured logging with rotation and multiple handlers.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize logging configuration.
        
        Args:
            log_dir (str): Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different log types
        (self.log_dir / "application").mkdir(exist_ok=True)
        (self.log_dir / "system").mkdir(exist_ok=True)
        (self.log_dir / "archived").mkdir(exist_ok=True)
        
    def setup_logger(self, name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
        """
        Setup a logger with file and console handlers.
        
        Args:
            name (str): Logger name
            log_file (str): Log file name (optional)
            level (int): Logging level
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if log_file specified)
        if log_file:
            file_path = self.log_dir / "application" / log_file
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def setup_data_logger(self) -> logging.Logger:
        """Setup logger for data operations."""
        return self.setup_logger(
            name="market_research.data",
            log_file="data_collection.log",
            level=logging.INFO
        )
    
    def setup_analysis_logger(self) -> logging.Logger:
        """Setup logger for analysis operations."""
        return self.setup_logger(
            name="market_research.analysis",
            log_file="analysis.log",
            level=logging.INFO
        )
    
    def setup_reporting_logger(self) -> logging.Logger:
        """Setup logger for reporting operations."""
        return self.setup_logger(
            name="market_research.reporting",
            log_file="reporting.log",
            level=logging.INFO
        )
    
    def setup_system_logger(self) -> logging.Logger:
        """Setup logger for system operations."""
        return self.setup_logger(
            name="market_research.system",
            log_file="system.log",
            level=logging.WARNING
        )
    
    def setup_error_logger(self) -> logging.Logger:
        """Setup logger specifically for errors."""
        logger = self.setup_logger(
            name="market_research.errors",
            log_file="errors.log",
            level=logging.ERROR
        )
        
        # Add email handler for critical errors (can be configured later)
        return logger
    
    def get_performance_logger(self) -> logging.Logger:
        """Setup logger for performance monitoring."""
        return self.setup_logger(
            name="market_research.performance",
            log_file="../system/performance.log",
            level=logging.INFO
        )


# Global logger configuration instance
logger_config = LoggerConfig()

# Pre-configured loggers for common use
data_logger = logger_config.setup_data_logger()
analysis_logger = logger_config.setup_analysis_logger()
reporting_logger = logger_config.setup_reporting_logger()
system_logger = logger_config.setup_system_logger()
error_logger = logger_config.setup_error_logger()


def log_function_call(func):
    """
    Decorator to automatically log function calls.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Decorated function with logging
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"market_research.{func.__module__}")
        logger.info(f"Calling function: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed with error: {str(e)}")
            error_logger.error(f"Function {func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper


def log_api_call(api_name: str, endpoint: str, response_code: int = None):
    """
    Log API calls for monitoring and debugging.
    
    Args:
        api_name (str): Name of the API service
        endpoint (str): API endpoint called
        response_code (int): HTTP response code (optional)
    """
    api_logger = logging.getLogger("market_research.api")
    
    if response_code:
        if response_code == 200:
            api_logger.info(f"API Call Success - {api_name}: {endpoint} (Status: {response_code})")
        else:
            api_logger.warning(f"API Call Issue - {api_name}: {endpoint} (Status: {response_code})")
    else:
        api_logger.info(f"API Call - {api_name}: {endpoint}")


class MarketDataLogger:
    """
    Specialized logger for market data operations.
    Provides context-aware logging for data collection and processing.
    """
    
    def __init__(self):
        self.logger = data_logger
    
    def log_data_fetch_start(self, symbol: str, data_type: str):
        """Log start of data fetching operation."""
        self.logger.info(f"Starting data fetch - Symbol: {symbol}, Type: {data_type}")
    
    def log_data_fetch_success(self, symbol: str, data_type: str, records_count: int):
        """Log successful data fetch."""
        self.logger.info(f"Data fetch successful - Symbol: {symbol}, Type: {data_type}, Records: {records_count}")
    
    def log_data_fetch_error(self, symbol: str, data_type: str, error: str):
        """Log data fetch error."""
        self.logger.error(f"Data fetch failed - Symbol: {symbol}, Type: {data_type}, Error: {error}")
        error_logger.error(f"Data fetch failed - Symbol: {symbol}, Type: {data_type}, Error: {error}")
    
    def log_data_processing(self, operation: str, input_records: int, output_records: int):
        """Log data processing operation."""
        self.logger.info(f"Data processing - Operation: {operation}, Input: {input_records}, Output: {output_records}")
    
    def log_data_validation(self, symbol: str, validation_result: bool, issues: list = None):
        """Log data validation results."""
        if validation_result:
            self.logger.info(f"Data validation passed - Symbol: {symbol}")
        else:
            self.logger.warning(f"Data validation failed - Symbol: {symbol}, Issues: {issues}")


# Global market data logger instance
market_data_logger = MarketDataLogger()


if __name__ == "__main__":
    # Test logging configuration
    test_logger = logger_config.setup_logger("test", "test.log")
    
    test_logger.info("Testing logging configuration")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    # Test market data logger
    market_data_logger.log_data_fetch_start("RELIANCE.NS", "historical_prices")
    market_data_logger.log_data_fetch_success("RELIANCE.NS", "historical_prices", 1000)
    
    print("Logging configuration test completed. Check logs directory.")