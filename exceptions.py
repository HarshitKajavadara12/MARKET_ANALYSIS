"""
Custom Exceptions for Market Research System v1.0
Defines all custom exceptions used throughout the system.
Created: January 2022
"""

class MarketResearchError(Exception):
    """Base exception class for Market Research System."""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataFetchError(MarketResearchError):
    """Raised when data fetching from external APIs fails."""
    
    def __init__(self, message: str, source: str = None, symbol: str = None):
        self.source = source
        self.symbol = symbol
        error_code = "DATA_FETCH_ERROR"
        
        if source and symbol:
            detailed_message = f"Failed to fetch data for {symbol} from {source}: {message}"
        elif source:
            detailed_message = f"Failed to fetch data from {source}: {message}"
        elif symbol:
            detailed_message = f"Failed to fetch data for {symbol}: {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class DataValidationError(MarketResearchError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: str = None, value: str = None):
        self.field = field
        self.value = value
        error_code = "DATA_VALIDATION_ERROR"
        
        if field and value:
            detailed_message = f"Validation failed for field '{field}' with value '{value}': {message}"
        elif field:
            detailed_message = f"Validation failed for field '{field}': {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class DataStorageError(MarketResearchError):
    """Raised when data storage operations fail."""
    
    def __init__(self, message: str, operation: str = None, file_path: str = None):
        self.operation = operation
        self.file_path = file_path
        error_code = "DATA_STORAGE_ERROR"
        
        if operation and file_path:
            detailed_message = f"Storage operation '{operation}' failed for file '{file_path}': {message}"
        elif operation:
            detailed_message = f"Storage operation '{operation}' failed: {message}"
        elif file_path:
            detailed_message = f"Storage operation failed for file '{file_path}': {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class DataProcessingError(MarketResearchError):
    """Raised when data processing operations fail."""
    
    def __init__(self, message: str, processing_step: str = None, symbol: str = None):
        self.processing_step = processing_step
        self.symbol = symbol
        error_code = "DATA_PROCESSING_ERROR"
        
        if processing_step and symbol:
            detailed_message = f"Processing step '{processing_step}' failed for {symbol}: {message}"
        elif processing_step:
            detailed_message = f"Processing step '{processing_step}' failed: {message}"
        elif symbol:
            detailed_message = f"Data processing failed for {symbol}: {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class APIError(MarketResearchError):
    """Raised when API operations fail."""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None, 
                 response_text: str = None):
        self.api_name = api_name
        self.status_code = status_code
        self.response_text = response_text
        error_code = "API_ERROR"
        
        if api_name and status_code:
            detailed_message = f"API '{api_name}' returned status {status_code}: {message}"
        elif api_name:
            detailed_message = f"API '{api_name}' error: {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, message: str = "API rate limit exceeded", api_name: str = None,
                 retry_after: int = None):
        self.retry_after = retry_after
        error_code = "API_RATE_LIMIT_ERROR"
        
        if retry_after:
            detailed_message = f"{message}. Retry after {retry_after} seconds."
        else:
            detailed_message = message
            
        super().__init__(detailed_message, api_name, 429)


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""
    
    def __init__(self, message: str = "API authentication failed", api_name: str = None):
        error_code = "API_AUTH_ERROR"
        super().__init__(message, api_name, 401)


class AnalysisError(MarketResearchError):
    """Raised when analysis operations fail."""
    
    def __init__(self, message: str, analysis_type: str = None, symbol: str = None):
        self.analysis_type = analysis_type
        self.symbol = symbol
        error_code = "ANALYSIS_ERROR"
        
        if analysis_type and symbol:
            detailed_message = f"Analysis '{analysis_type}' failed for {symbol}: {message}"
        elif analysis_type:
            detailed_message = f"Analysis '{analysis_type}' failed: {message}"
        elif symbol:
            detailed_message = f"Analysis failed for {symbol}: {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class TechnicalIndicatorError(AnalysisError):
    """Raised when technical indicator calculation fails."""
    
    def __init__(self, message: str, indicator_name: str = None, symbol: str = None):
        self.indicator_name = indicator_name
        error_code = "TECHNICAL_INDICATOR_ERROR"
        
        if indicator_name:
            detailed_message = f"Technical indicator '{indicator_name}' calculation failed: {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, indicator_name, symbol)


class ReportGenerationError(MarketResearchError):
    """Raised when report generation fails."""
    
    def __init__(self, message: str, report_type: str = None, output_path: str = None):
        self.report_type = report_type
        self.output_path = output_path
        error_code = "REPORT_GENERATION_ERROR"
        
        if report_type and output_path:
            detailed_message = f"Report generation '{report_type}' failed for output '{output_path}': {message}"
        elif report_type:
            detailed_message = f"Report generation '{report_type}' failed: {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class ConfigurationError(MarketResearchError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_file: str = None, config_key: str = None):
        self.config_file = config_file
        self.config_key = config_key
        error_code = "CONFIGURATION_ERROR"
        
        if config_file and config_key:
            detailed_message = f"Configuration error in file '{config_file}' for key '{config_key}': {message}"
        elif config_file:
            detailed_message = f"Configuration error in file '{config_file}': {message}"
        elif config_key:
            detailed_message = f"Configuration error for key '{config_key}': {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class NSEDataError(DataFetchError):
    """Raised when NSE (National Stock Exchange) data fetching fails."""
    
    def __init__(self, message: str, symbol: str = None, segment: str = None):
        self.segment = segment  # EQ, FO, CD, etc.
        
        if segment and symbol:
            detailed_message = f"NSE {segment} data fetch failed for {symbol}: {message}"
        elif symbol:
            detailed_message = f"NSE data fetch failed for {symbol}: {message}"
        else:
            detailed_message = f"NSE data fetch failed: {message}"
            
        super().__init__(detailed_message, "NSE", symbol)


class BSEDataError(DataFetchError):
    """Raised when BSE (Bombay Stock Exchange) data fetching fails."""
    
    def __init__(self, message: str, symbol: str = None, series: str = None):
        self.series = series  # A, B, T, etc.
        
        if series and symbol:
            detailed_message = f"BSE Series-{series} data fetch failed for {symbol}: {message}"
        elif symbol:
            detailed_message = f"BSE data fetch failed for {symbol}: {message}"
        else:
            detailed_message = f"BSE data fetch failed: {message}"
            
        super().__init__(detailed_message, "BSE", symbol)


class IndianMarketHolidayError(MarketResearchError):
    """Raised when trying to fetch data during Indian market holidays."""
    
    def __init__(self, message: str = "Indian market is closed", date: str = None, 
                 holiday_name: str = None):
        self.date = date
        self.holiday_name = holiday_name
        error_code = "MARKET_HOLIDAY_ERROR"
        
        if date and holiday_name:
            detailed_message = f"Market closed on {date} for {holiday_name}: {message}"
        elif date:
            detailed_message = f"Market closed on {date}: {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class SectorAnalysisError(AnalysisError):
    """Raised when Indian sector-specific analysis fails."""
    
    def __init__(self, message: str, sector: str = None, index: str = None):
        self.sector = sector  # Banking, IT, Pharma, Auto, etc.
        self.index = index    # BANKNIFTY, CNXIT, etc.
        error_code = "SECTOR_ANALYSIS_ERROR"
        
        if sector and index:
            detailed_message = f"Sector analysis failed for {sector} ({index}): {message}"
        elif sector:
            detailed_message = f"Sector analysis failed for {sector}: {message}"
        elif index:
            detailed_message = f"Index analysis failed for {index}: {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, "sector_analysis", sector)


class CorporateActionError(MarketResearchError):
    """Raised when corporate action data processing fails."""
    
    def __init__(self, message: str, symbol: str = None, action_type: str = None, 
                 ex_date: str = None):
        self.symbol = symbol
        self.action_type = action_type  # Dividend, Bonus, Split, Rights, etc.
        self.ex_date = ex_date
        error_code = "CORPORATE_ACTION_ERROR"
        
        if symbol and action_type and ex_date:
            detailed_message = f"Corporate action '{action_type}' processing failed for {symbol} (Ex-date: {ex_date}): {message}"
        elif symbol and action_type:
            detailed_message = f"Corporate action '{action_type}' processing failed for {symbol}: {message}"
        elif symbol:
            detailed_message = f"Corporate action processing failed for {symbol}: {message}"
        else:
            detailed_message = message
            
        super().__init__(detailed_message, error_code)


class FIIDataError(DataFetchError):
    """Raised when FII (Foreign Institutional Investor) data fetching fails."""
    
    def __init__(self, message: str, data_type: str = None, date: str = None):
        self.data_type = data_type  # cash, derivatives, debt
        self.date = date
        
        if data_type and date:
            detailed_message = f"FII {data_type} data fetch failed for {date}: {message}"
        elif data_type:
            detailed_message = f"FII {data_type} data fetch failed: {message}"
        else:
            detailed_message = f"FII data fetch failed: {message}"
            
        super().__init__(detailed_message, "FII_DATA", None)


class DIIDataError(DataFetchError):
    """Raised when DII (Domestic Institutional Investor) data fetching fails."""
    
    def __init__(self, message: str, data_type: str = None, date: str = None):
        self.data_type = data_type  # cash, derivatives
        self.date = date
        
        if data_type and date:
            detailed_message = f"DII {data_type} data fetch failed for {date}: {message}"
        elif data_type:
            detailed_message = f"DII {data_type} data fetch failed: {message}"
        else:
            detailed_message = f"DII data fetch failed: {message}"
            
        super().__init__(detailed_message, "DII_DATA", None)


class VIXDataError(DataFetchError):
    """Raised when India VIX data fetching fails."""
    
    def __init__(self, message: str, date: str = None):
        self.date = date
        
        if date:
            detailed_message = f"India VIX data fetch failed for {date}: {message}"
        else:
            detailed_message = f"India VIX data fetch failed: {message}"
            
        super().__init__(detailed_message, "INDIA_VIX", "VIX")


class DerivativesDataError(DataFetchError):
    """Raised when F&O (Futures & Options) data fetching fails."""
    
    def __init__(self, message: str, symbol: str = None, instrument_type: str = None, 
                 expiry: str = None, strike: str = None):
        self.instrument_type = instrument_type  # FUTIDX, FUTSTK, OPTIDX, OPTSTK
        self.expiry = expiry
        self.strike = strike
        
        if all([symbol, instrument_type, expiry, strike]):
            detailed_message = f"F&O data fetch failed for {symbol} {instrument_type} {expiry} {strike}: {message}"
        elif all([symbol, instrument_type, expiry]):
            detailed_message = f"F&O data fetch failed for {symbol} {instrument_type} {expiry}: {message}"
        elif symbol and instrument_type:
            detailed_message = f"F&O data fetch failed for {symbol} {instrument_type}: {message}"
        elif symbol:
            detailed_message = f"F&O data fetch failed for {symbol}: {message}"
        else:
            detailed_message = f"F&O data fetch failed: {message}"
            
        super().__init__(detailed_message, "FNO", symbol)


class EconomicDataError(DataFetchError):
    """Raised when Indian economic indicator data fetching fails."""
    
    def __init__(self, message: str, indicator: str = None, source: str = None):
        self.indicator = indicator  # GDP, CPI, IIP, etc.
        
        if indicator and source:
            detailed_message = f"Economic indicator '{indicator}' fetch failed from {source}: {message}"
        elif indicator:
            detailed_message = f"Economic indicator '{indicator}' fetch failed: {message}"
        else:
            detailed_message = f"Economic data fetch failed: {message}"
            
        super().__init__(detailed_message, source or "ECONOMIC_DATA", indicator)


# Version 1 specific error codes for logging and debugging
ERROR_CODES = {
    "DATA_FETCH_ERROR": "Data fetching operation failed",
    "DATA_VALIDATION_ERROR": "Data validation failed",
    "DATA_STORAGE_ERROR": "Data storage operation failed",
    "DATA_PROCESSING_ERROR": "Data processing operation failed",
    "API_ERROR": "API operation failed",
    "API_RATE_LIMIT_ERROR": "API rate limit exceeded",
    "API_AUTH_ERROR": "API authentication failed",
    "ANALYSIS_ERROR": "Analysis operation failed",
    "TECHNICAL_INDICATOR_ERROR": "Technical indicator calculation failed",
    "REPORT_GENERATION_ERROR": "Report generation failed",
    "CONFIGURATION_ERROR": "Configuration error",
    "MARKET_HOLIDAY_ERROR": "Market holiday detected",
    "SECTOR_ANALYSIS_ERROR": "Sector analysis failed",
    "CORPORATE_ACTION_ERROR": "Corporate action processing failed"
}


def get_error_description(error_code: str) -> str:
    """Get human-readable description for error code."""
    return ERROR_CODES.get(error_code, "Unknown error")
        