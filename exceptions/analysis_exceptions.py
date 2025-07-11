"""
Custom exceptions for analysis-related operations in Market Research System v1.0
Created: January 2022
Author: Independent Market Researcher
Focus: Indian Stock Market Analysis
"""

class AnalysisException(Exception):
    """Base exception class for all analysis-related errors"""
    def __init__(self, message="Analysis operation failed"):
        self.message = message
        super().__init__(self.message)

class InsufficientDataException(AnalysisException):
    """Raised when there's insufficient data for analysis"""
    def __init__(self, required_periods=None, available_periods=None):
        if required_periods and available_periods:
            message = f"Insufficient data: Required {required_periods} periods, but only {available_periods} available"
        else:
            message = "Insufficient data for analysis"
        super().__init__(message)

class InvalidIndicatorParametersException(AnalysisException):
    """Raised when invalid parameters are provided for technical indicators"""
    def __init__(self, indicator_name, parameter, value):
        message = f"Invalid parameter for {indicator_name}: {parameter}={value}"
        super().__init__(message)

class DataQualityException(AnalysisException):
    """Raised when data quality issues prevent proper analysis"""
    def __init__(self, issue_description):
        message = f"Data quality issue: {issue_description}"
        super().__init__(message)

class CalculationException(AnalysisException):
    """Raised when mathematical calculations fail during analysis"""
    def __init__(self, calculation_type, details=None):
        message = f"Calculation failed: {calculation_type}"
        if details:
            message += f" - {details}"
        super().__init__(message)

class UnsupportedTimeframeException(AnalysisException):
    """Raised when an unsupported timeframe is requested"""
    def __init__(self, timeframe):
        message = f"Unsupported timeframe: {timeframe}. Supported: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo"
        super().__init__(message)

class IndicatorNotAvailableException(AnalysisException):
    """Raised when a requested technical indicator is not available"""
    def __init__(self, indicator_name):
        message = f"Technical indicator not available: {indicator_name}"
        super().__init__(message)

class BacktestException(AnalysisException):
    """Raised when backtesting operations fail"""
    def __init__(self, strategy_name, error_details):
        message = f"Backtesting failed for strategy '{strategy_name}': {error_details}"
        super().__init__(message)

class CorrelationAnalysisException(AnalysisException):
    """Raised when correlation analysis fails"""
    def __init__(self, symbol1, symbol2, error_details):
        message = f"Correlation analysis failed between {symbol1} and {symbol2}: {error_details}"
        super().__init__(message)

class VolatilityCalculationException(AnalysisException):
    """Raised when volatility calculations fail"""
    def __init__(self, method, error_details):
        message = f"Volatility calculation failed using {method}: {error_details}"
        super().__init__(message)

class TrendAnalysisException(AnalysisException):
    """Raised when trend analysis operations fail"""
    def __init__(self, analysis_type, error_details):
        message = f"Trend analysis failed for {analysis_type}: {error_details}"
        super().__init__(message)

class RiskMetricsException(AnalysisException):
    """Raised when risk metrics calculation fails"""
    def __init__(self, metric_name, error_details):
        message = f"Risk metrics calculation failed for {metric_name}: {error_details}"
        super().__init__(message)

class MarketRegimeException(AnalysisException):
    """Raised when market regime analysis fails"""
    def __init__(self, error_details):
        message = f"Market regime analysis failed: {error_details}"
        super().__init__(message)

class SectorAnalysisException(AnalysisException):
    """Raised when sector analysis operations fail"""
    def __init__(self, sector_name, error_details):
        message = f"Sector analysis failed for {sector_name}: {error_details}"
        super().__init__(message)

class ModelTrainingException(AnalysisException):
    """Raised when machine learning model training fails"""
    def __init__(self, model_type, error_details):
        message = f"Model training failed for {model_type}: {error_details}"
        super().__init__(message)

class PredictionException(AnalysisException):
    """Raised when prediction operations fail"""
    def __init__(self, prediction_type, error_details):
        message = f"Prediction failed for {prediction_type}: {error_details}"
        super().__init__(message)

class OptimizationException(AnalysisException):
    """Raised when portfolio optimization fails"""
    def __init__(self, optimization_method, error_details):
        message = f"Portfolio optimization failed using {optimization_method}: {error_details}"
        super().__init__(message)

class DataAlignmentException(AnalysisException):
    """Raised when data alignment issues occur during analysis"""
    def __init__(self, symbols, error_details):
        message = f"Data alignment failed for symbols {symbols}: {error_details}"
        super().__init__(message)

class TimeSeriesException(AnalysisException):
    """Raised when time series analysis fails"""
    def __init__(self, analysis_type, error_details):
        message = f"Time series analysis failed for {analysis_type}: {error_details}"
        super().__init__(message)

# Error handling utilities
def handle_analysis_error(func):
    """Decorator to handle common analysis errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError as e:
            raise CalculationException("Division by zero", str(e))
        except ValueError as e:
            raise InvalidIndicatorParametersException("Unknown", "value", str(e))
        except KeyError as e:
            raise DataQualityException(f"Missing required data column: {e}")
        except Exception as e:
            raise AnalysisException(f"Unexpected analysis error: {str(e)}")
    return wrapper

def validate_data_sufficiency(data, min_periods, operation_name):
    """Validate if data has sufficient periods for analysis"""
    if len(data) < min_periods:
        raise InsufficientDataException(min_periods, len(data))

def validate_indicator_params(params, valid_ranges):
    """Validate technical indicator parameters"""
    for param, value in params.items():
        if param in valid_ranges:
            min_val, max_val = valid_ranges[param]
            if not (min_val <= value <= max_val):
                raise InvalidIndicatorParametersException("Unknown", param, value)