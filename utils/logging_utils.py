"""
Market Research System v1.0 - Logging Utilities
Author: Independent Market Researcher
Created: 2022
Updated: 2025

Comprehensive logging system for market research operations, data collection,
analysis, and reporting activities. Supports multiple log levels, file rotation,
and structured logging for better monitoring and debugging.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
import traceback
from pathlib import Path
import threading
from functools import wraps
import time

# Configuration constants
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
DETAILED_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(funcName)s:%(lineno)d - %(message)s'
JSON_LOG_FORMAT = '%(asctime)s'  # Will be handled by JSONFormatter

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to the log level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'thread_name': record.threadName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = record.symbol
        if hasattr(record, 'exchange'):
            log_entry['exchange'] = record.exchange
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'execution_time'):
            log_entry['execution_time'] = record.execution_time
        
        return json.dumps(log_entry)

class MarketResearchLogger:
    """
    Centralized logging system for the market research application
    """
    
    def __init__(self, name: str = 'MarketResearch', log_dir: str = 'logs'):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different log types
        self.app_log_dir = self.log_dir / 'application'
        self.sys_log_dir = self.log_dir / 'system'
        self.trade_log_dir = self.log_dir / 'trading' 
        self.api_log_dir = self.log_dir / 'api'
        
        for log_subdir in [self.app_log_dir, self.sys_log_dir, self.trade_log_dir, self.api_log_dir]:
            log_subdir.mkdir(exist_ok=True)
        
        self.loggers = {}
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup different types of loggers"""
        
        # Main application logger
        self.loggers['app'] = self._create_logger(
            'app',
            self.app_log_dir / 'app.log',
            level=logging.INFO
        )
        
        # Data collection logger
        self.loggers['data'] = self._create_logger(
            'data_collection',
            self.app_log_dir / 'data_collection.log',
            level=logging.DEBUG
        )
        
        # Analysis logger
        self.loggers['analysis'] = self._create_logger(
            'analysis',
            self.app_log_dir / 'analysis.log',
            level=logging.INFO
        )
        
        # Reporting logger
        self.loggers['reporting'] = self._create_logger(
            'reporting',
            self.app_log_dir / 'reporting.log',
            level=logging.INFO
        )
        
        # Error logger (captures all errors)
        self.loggers['error'] = self._create_logger(
            'errors',
            self.app_log_dir / 'errors.log',
            level=logging.ERROR,
            error_only=True
        )
        
        # Performance logger
        self.loggers['performance'] = self._create_logger(
            'performance',
            self.sys_log_dir / 'performance.log',
            level=logging.INFO,
            use_json=True
        )
        
        # API logger
        self.loggers['api'] = self._create_logger(
            'api_calls',
            self.api_log_dir / 'api_calls.log',
            level=logging.DEBUG,
            use_json=True
        )
        
        # Trading signals logger
        self.loggers['signals'] = self._create_logger(
            'trading_signals',
            self.trade_log_dir / 'signals.log',
            level=logging.INFO,
            use_json=True
        )
    
    def _create_logger(self, name: str, log_file: Path, level: int = logging.INFO, 
                      error_only: bool = False, use_json: bool = False) -> logging.Logger:
        """Create a configured logger instance"""
        
        logger = logging.getLogger(f"{self.name}.{name}")
        logger.setLevel(level)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Console handler for important logs
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set formatters
        if use_json:
            file_formatter = JSONFormatter()
            console_formatter = ColoredFormatter(DEFAULT_LOG_FORMAT)
        else:
            file_formatter = logging.Formatter(DETAILED_LOG_FORMAT)
            console_formatter = ColoredFormatter(DEFAULT_LOG_FORMAT)
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Set levels for handlers
        file_handler.setLevel(level)
        
        if error_only:
            console_handler.setLevel(logging.ERROR)
        else:
            console_handler.setLevel(logging.WARNING)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        
        # Only add console handler for main app logger to avoid duplicate console output
        if name == 'app' or error_only:
            logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(self, logger_type: str = 'app') -> logging.Logger:
        """Get a specific logger instance"""
        return self.loggers.get(logger_type, self.loggers['app'])
    
    def log_data_fetch(self, symbol: str, exchange: str, operation: str, 
                      status: str, message: str = "", execution_time: float = None):
        """Log data fetching operations"""
        logger = self.get_logger('data')
        
        extra = {
            'symbol': symbol,
            'exchange': exchange,
            'operation': operation,
            'execution_time': execution_time
        }
        
        if status.lower() == 'success':
            logger.info(f"Data fetch successful: {symbol} from {exchange} - {message}", extra=extra)
        elif status.lower() == 'warning':
            logger.warning(f"Data fetch warning: {symbol} from {exchange} - {message}", extra=extra)
        else:
            logger.error(f"Data fetch failed: {symbol} from {exchange} - {message}", extra=extra)
    
    def log_analysis(self, operation: str, symbols: List[str], status: str, 
                    message: str = "", execution_time: float = None):
        """Log analysis operations"""
        logger = self.get_logger('analysis')
        
        extra = {
            'operation': operation,
            'symbols': symbols,
            'execution_time': execution_time
        }
        
        if status.lower() == 'success':
            logger.info(f"Analysis completed: {operation} - {message}", extra=extra)
        elif status.lower() == 'warning':
            logger.warning(f"Analysis warning: {operation} - {message}", extra=extra)
        else:
            logger.error(f"Analysis failed: {operation} - {message}", extra=extra)
    
    def log_api_call(self, api_name: str, endpoint: str, method: str, 
                    status_code: int, response_size: int = None, execution_time: float = None):
        """Log API calls with detailed information"""
        logger = self.get_logger('api')
        
        extra = {
            'api_name': api_name,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_size': response_size,
            'execution_time': execution_time
        }
        
        if 200 <= status_code < 300:
            logger.info(f"API call successful: {api_name} {method} {endpoint}", extra=extra)
        elif 400 <= status_code < 500:
            logger.warning(f"API client error: {api_name} {method} {endpoint} - {status_code}", extra=extra)
        else:
            logger.error(f"API server error: {api_name} {method} {endpoint} - {status_code}", extra=extra)
    
    def log_trading_signal(self, symbol: str, signal_type: str, action: str, 
                          confidence: float, price: float, timestamp: datetime = None):
        """Log trading signals"""
        logger = self.get_logger('signals')
        
        if timestamp is None:
            timestamp = datetime.now()
        
        extra = {
            'symbol': symbol,
            'signal_type': signal_type,
            'action': action,
            'confidence': confidence,
            'price': price,
            'timestamp': timestamp.isoformat()
        }
        
        logger.info(f"Trading signal generated: {symbol} - {action} ({signal_type}) - Confidence: {confidence:.2f}", extra=extra)
    
    def log_performance_metric(self, metric_name: str, value: float, 
                             symbol: str = None, timeframe: str = None):
        """Log performance metrics"""
        logger = self.get_logger('performance')
        
        extra = {
            'metric_name': metric_name,
            'value': value,
            'symbol': symbol,
            'timeframe': timeframe
        }
        
        logger.info(f"Performance metric: {metric_name} = {value}", extra=extra)
    
    def log_report_generation(self, report_type: str, symbols: List[str], 
                            status: str, file_path: str = None, execution_time: float = None):
        """Log report generation activities"""
        logger = self.get_logger('reporting')
        
        extra = {
            'report_type': report_type,
            'symbols': symbols,
            'file_path': file_path,
            'execution_time': execution_time
        }
        
        if status.lower() == 'success':
            logger.info(f"Report generated successfully: {report_type} - {file_path}", extra=extra)
        else:
            logger.error(f"Report generation failed: {report_type}", extra=extra)
    
    def log_indian_market_specific(self, operation: str, segment: str, 
                                 symbol: str = None, message: str = ""):
        """Log Indian market specific operations (NSE/BSE)"""
        logger = self.get_logger('app')
        
        extra = {
            'operation': operation,
            'segment': segment,  # NSE, BSE, MCX, NCDEX
            'symbol': symbol,
            'market': 'INDIAN'
        }
        
        logger.info(f"Indian Market Operation: {operation} - {segment} - {message}", extra=extra)
    
    def log_system_health(self, component: str, status: str, metrics: Dict[str, Any] = None):
        """Log system health and monitoring information"""
        logger = self.get_logger('performance')
        
        extra = {
            'component': component,
            'status': status,
            'metrics': metrics or {}
        }
        
        if status.lower() == 'healthy':
            logger.info(f"System health check: {component} - {status}", extra=extra)
        else:
            logger.warning(f"System health issue: {component} - {status}", extra=extra)
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        logger = self.get_logger('app')
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleaned_files = 0
        for log_dir in [self.app_log_dir, self.sys_log_dir, self.trade_log_dir, self.api_log_dir]:
            for log_file in log_dir.glob('*.log.*'):  # Rotated log files
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    try:
                        log_file.unlink()
                        cleaned_files += 1
                    except OSError as e:
                        logger.warning(f"Failed to delete old log file {log_file}: {e}")
        
        logger.info(f"Log cleanup completed: {cleaned_files} files removed")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about current log files"""
        stats = {}
        
        for log_type, logger in self.loggers.items():
            log_files = []
            for handler in logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    log_file = Path(handler.baseFilename)
                    if log_file.exists():
                        log_files.append({
                            'file': str(log_file),
                            'size': log_file.stat().st_size,
                            'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                        })
            
            stats[log_type] = {
                'files': log_files,
                'total_size': sum(f['size'] for f in log_files)
            }
        
        return stats

def performance_timer(logger_name: str = 'performance'):
    """Decorator to measure and log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Get the logger instance
                market_logger = MarketResearchLogger()
                logger = market_logger.get_logger(logger_name)
                
                extra = {
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'status': 'success'
                }
                
                logger.info(f"Function {func.__name__} completed in {execution_time:.4f}s", extra=extra)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Get the logger instance
                market_logger = MarketResearchLogger()
                logger = market_logger.get_logger('error')
                
                extra = {
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'status': 'error',
                    'error': str(e)
                }
                
                logger.error(f"Function {func.__name__} failed after {execution_time:.4f}s: {str(e)}", 
                           extra=extra, exc_info=True)
                raise
        
        return wrapper
    return decorator

def log_indian_market_session():
    """Log Indian market session information"""
    market_logger = MarketResearchLogger()
    logger = market_logger.get_logger('app')
    
    now = datetime.now()
    
    # NSE/BSE market hours: 9:15 AM to 3:30 PM
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # Pre-market: 9:00 AM to 9:15 AM
    premarket_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    
    # After-market: 3:40 PM to 4:00 PM
    aftermarket_start = now.replace(hour=15, minute=40, second=0, microsecond=0)
    aftermarket_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Determine market session
    if now.weekday() >= 5:  # Weekend
        session = "CLOSED_WEEKEND"
    elif premarket_start <= now < market_start:
        session = "PRE_MARKET"
    elif market_start <= now <= market_end:
        session = "MARKET_HOURS"
    elif aftermarket_start <= now <= aftermarket_end:
        session = "AFTER_MARKET"
    else:
        session = "CLOSED"
    
    extra = {
        'session': session,
        'market_start': market_start.strftime('%H:%M:%S'),
        'market_end': market_end.strftime('%H:%M:%S'),
        'current_time': now.strftime('%H:%M:%S'),
        'weekday': now.strftime('%A')
    }
    
    logger.info(f"Indian Market Session: {session}", extra=extra)
    return session

# Global logger instance
_global_logger = None

def get_global_logger() -> MarketResearchLogger:
    """Get the global logger instance (singleton pattern)"""
    global _global_logger
    if _global_logger is None:
        _global_logger = MarketResearchLogger()
    return _global_logger

def setup_logging(log_level: str = 'INFO', log_dir: str = 'logs'):
    """Setup logging configuration for the entire application"""
    global _global_logger
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create global logger instance
    _global_logger = MarketResearchLogger(log_dir=log_dir)
    
    # Set root logger level
    logging.getLogger().setLevel(numeric_level)
    
    # Log the initialization
    logger = _global_logger.get_logger('app')
    logger.info(f"Logging system initialized - Level: {log_level}, Directory: {log_dir}")
    
    # Log Indian market session
    log_indian_market_session()
    
    return _global_logger

# Exception handler for uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions by logging them"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = get_global_logger().get_logger('error')
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set the exception handler
sys.excepthook = handle_exception

# Context manager for logging operations
class LoggingContext:
    """Context manager for logging operations with automatic cleanup"""
    
    def __init__(self, operation: str, logger_type: str = 'app', 
                 log_start: bool = True, log_end: bool = True):
        self.operation = operation
        self.logger_type = logger_type
        self.log_start = log_start
        self.log_end = log_end
        self.start_time = None
        self.logger = None
    
    def __enter__(self):
        self.logger = get_global_logger().get_logger(self.logger_type)
        self.start_time = time.time()
        
        if self.log_start:
            self.logger.info(f"Starting operation: {self.operation}")
        
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            if self.log_end:
                self.logger.info(f"Completed operation: {self.operation} in {execution_time:.4f}s")
        else:
            self.logger.error(f"Failed operation: {self.operation} after {execution_time:.4f}s - {str(exc_value)}")
        
        return False  # Don't suppress exceptions

# Indian market specific logging helpers
def log_nse_operation(symbol: str, operation: str, message: str = ""):
    """Log NSE specific operations"""
    logger = get_global_logger()
    logger.log_indian_market_specific(operation, 'NSE', symbol, message)

def log_bse_operation(symbol: str, operation: str, message: str = ""):
    """Log BSE specific operations"""
    logger = get_global_logger()
    logger.log_indian_market_specific(operation, 'BSE', symbol, message)

def log_mcx_operation(symbol: str, operation: str, message: str = ""):
    """Log MCX (commodities) specific operations"""
    logger = get_global_logger()
    logger.log_indian_market_specific(operation, 'MCX', symbol, message)

# Example usage and testing
if __name__ == "__main__":
    # Initialize logging
    market_logger = setup_logging('DEBUG', 'test_logs')
    
    # Test different logging operations
    market_logger.log_data_fetch('RELIANCE', 'NSE', 'FETCH_OHLC', 'success', 
                                execution_time=0.234)
    
    market_logger.log_analysis('TECHNICAL_ANALYSIS', ['RELIANCE', 'TCS'], 'success',
                              'RSI and MACD calculated', execution_time=1.456)
    
    market_logger.log_api_call('NSE_API', '/api/equity-stockIndices', 'GET', 200, 
                              response_size=1024, execution_time=0.567)
    
    market_logger.log_trading_signal('RELIANCE', 'RSI_DIVERGENCE', 'BUY', 0.85, 2500.75)
    
    # Test Indian market operations
    log_nse_operation('RELIANCE', 'DATA_FETCH', 'Successfully fetched real-time data')
    log_bse_operation('SENSEX', 'INDEX_CALCULATION', 'Index values updated')
    
    # Test context manager
    with LoggingContext('TEST_OPERATION', 'app'):
        time.sleep(1)  # Simulate work
        print("Test operation completed")
    
    # Test performance decorator
    @performance_timer('performance')
    def test_function():
        time.sleep(0.5)
        return "Test completed"
    
    result = test_function()
    print(f"Function result: {result}")
    
    # Get and print log statistics
    stats = market_logger.get_log_stats()
    print(f"Log statistics: {json.dumps(stats, indent=2)}")
    
    print("Logging test completed. Check the 'test_logs' directory for log files.")