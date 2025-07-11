"""
Market Research System v1.0 - Utils Package
Created: 2022
Author: Independent Market Researcher

This package contains utility modules for common operations across the system.
"""

from .date_utils import (
    format_date,
    parse_date,
    get_market_days,
    is_market_open,
    get_trading_calendar,
    calculate_business_days,
    get_date_range,
    convert_timezone,
    get_market_hours
)

from .file_utils import (
    ensure_directory,
    save_to_csv,
    load_from_csv,
    save_to_json,
    load_from_json,
    get_file_size,
    backup_file,
    clean_filename,
    get_file_extension,
    list_files_by_pattern
)

from .math_utils import (
    calculate_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    normalize_data,
    calculate_correlation,
    moving_average,
    exponential_moving_average,
    calculate_percentile,
    safe_divide
)

from .string_utils import (
    clean_string,
    format_currency,
    format_percentage,
    format_number,
    extract_ticker_symbol,
    sanitize_filename,
    truncate_string,
    capitalize_words,
    remove_special_chars
)

from .validation_utils import (
    validate_ticker,
    validate_date_range,
    validate_dataframe,
    validate_numeric_input,
    validate_file_path,
    validate_api_response,
    validate_config,
    check_data_quality
)

from .logging_utils import (
    setup_logger,
    log_execution_time,
    log_data_info,
    log_error,
    log_warning,
    log_info,
    create_log_entry,
    setup_file_logging
)

__version__ = "1.0.0"
__author__ = "Independent Market Researcher"
__created__ = "2022"

__all__ = [
    # Date utilities
    'format_date', 'parse_date', 'get_market_days', 'is_market_open',
    'get_trading_calendar', 'calculate_business_days', 'get_date_range',
    'convert_timezone', 'get_market_hours',
    
    # File utilities
    'ensure_directory', 'save_to_csv', 'load_from_csv', 'save_to_json',
    'load_from_json', 'get_file_size', 'backup_file', 'clean_filename',
    'get_file_extension', 'list_files_by_pattern',
    
    # Math utilities
    'calculate_returns', 'calculate_volatility', 'calculate_sharpe_ratio',
    'calculate_max_drawdown', 'normalize_data', 'calculate_correlation',
    'moving_average', 'exponential_moving_average', 'calculate_percentile',
    'safe_divide',
    
    # String utilities
    'clean_string', 'format_currency', 'format_percentage', 'format_number',
    'extract_ticker_symbol', 'sanitize_filename', 'truncate_string',
    'capitalize_words', 'remove_special_chars',
    
    # Validation utilities
    'validate_ticker', 'validate_date_range', 'validate_dataframe',
    'validate_numeric_input', 'validate_file_path', 'validate_api_response',
    'validate_config', 'check_data_quality',
    
    # Logging utilities
    'setup_logger', 'log_execution_time', 'log_data_info', 'log_error',
    'log_warning', 'log_info', 'create_log_entry', 'setup_file_logging'
]