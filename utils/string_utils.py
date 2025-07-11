"""
String Manipulation Utilities for Market Research System v1.0
Provides string processing functions for financial data handling and report generation.
Created: January 2022
"""

import re
import string
import unicodedata
from typing import List, Dict, Union, Optional
import html

def clean_ticker_symbol(ticker: str) -> str:
    """
    Clean and standardize ticker symbols.
    
    Args:
        ticker: Raw ticker symbol
        
    Returns:
        Cleaned ticker symbol
    """
    if not ticker:
        return ""
    
    # Remove whitespace and convert to uppercase
    ticker = ticker.strip().upper()
    
    # Remove common prefixes/suffixes for Indian stocks
    ticker = re.sub(r'\.NS$|\.BO$|\.BSE$|\.NSE$', '', ticker)
    
    # Remove special characters except dots and hyphens
    ticker = re.sub(r'[^\w\.\-]', '', ticker)
    
    return ticker

def format_currency(amount: Union[int, float], currency: str = "₹", 
                   decimal_places: int = 2) -> str:
    """
    Format currency values with Indian number system (lakhs, crores).
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        decimal_places: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    if amount is None:
        return f"{currency}0.00"
    
    # Handle negative values
    negative = amount < 0
    amount = abs(amount)
    
    # Format based on Indian number system
    if amount >= 10000000:  # 1 crore
        formatted = f"{amount/10000000:.{decimal_places}f} Cr"
    elif amount >= 100000:  # 1 lakh
        formatted = f"{amount/100000:.{decimal_places}f} L"
    elif amount >= 1000:  # 1 thousand
        formatted = f"{amount/1000:.{decimal_places}f} K"
    else:
        formatted = f"{amount:.{decimal_places}f}"
    
    result = f"{currency}{formatted}"
    return f"-{result}" if negative else result

def format_percentage(value: Union[int, float], decimal_places: int = 2) -> str:
    """
    Format percentage values.
    
    Args:
        value: Percentage value (as decimal, e.g., 0.05 for 5%)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "0.00%"
    
    percentage = value * 100
    color_indicator = "📈" if percentage > 0 else "📉" if percentage < 0 else "➡️"
    
    return f"{color_indicator} {percentage:.{decimal_places}f}%"

def format_large_number(number: Union[int, float], suffix: str = "") -> str:
    """
    Format large numbers with appropriate suffixes.
    
    Args:
        number: Number to format
        suffix: Optional suffix to append
        
    Returns:
        Formatted number string
    """
    if number is None:
        return "0"
    
    abs_number = abs(number)
    
    if abs_number >= 1e12:
        formatted = f"{number/1e12:.2f}T"
    elif abs_number >= 1e9:
        formatted = f"{number/1e9:.2f}B"
    elif abs_number >= 1e6:
        formatted = f"{number/1e6:.2f}M"
    elif abs_number >= 1e3:
        formatted = f"{number/1e3:.2f}K"
    else:
        formatted = f"{number:.2f}"
    
    return f"{formatted}{suffix}"

def extract_stock_symbols(text: str) -> List[str]:
    """
    Extract stock symbols from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted stock symbols
    """
    if not text:
        return []
    
    # Pattern for Indian stock symbols
    pattern = r'\b[A-Z]{2,6}(?:\.(?:NS|BO|BSE|NSE))?\b'
    
    matches = re.findall(pattern, text.upper())
    
    # Clean and deduplicate
    symbols = []
    for match in matches:
        cleaned = clean_ticker_symbol(match)
        if cleaned and cleaned not in symbols:
            symbols.append(cleaned)
    
    return symbols

def clean_company_name(name: str) -> str:
    """
    Clean and standardize company names.
    
    Args:
        name: Raw company name
        
    Returns:
        Cleaned company name
    """
    if not name:
        return ""
    
    # Remove HTML entities
    name = html.unescape(name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name.strip())
    
    # Standardize common suffixes
    suffixes = {
        'LIMITED': 'Ltd',
        'LTD': 'Ltd',
        'PRIVATE': 'Pvt',
        'PVT': 'Pvt',
        'CORPORATION': 'Corp',
        'CORP': 'Corp',
        'INCORPORATED': 'Inc',
        'INC': 'Inc'
    }
    
    for old, new in suffixes.items():
        name = re.sub(rf'\b{old}\b', new, name, flags=re.IGNORECASE)
    
    return name.title()

def extract_numbers(text: str) -> List[float]:
    """
    Extract numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers
    """
    if not text:
        return []
    
    # Pattern to match numbers (including decimals, negatives, percentages)
    pattern = r'-?\d+(?:\.\d+)?(?:%|K|M|B|T)?'
    
    matches = re.findall(pattern, text)
    numbers = []
    
    for match in matches:
        try:
            # Handle suffixes
            if match.endswith('%'):
                numbers.append(float(match[:-1]) / 100)
            elif match.endswith('K'):
                numbers.append(float(match[:-1]) * 1000)
            elif match.endswith('M'):
                numbers.append(float(match[:-1]) * 1000000)
            elif match.endswith('B'):
                numbers.append(float(match[:-1]) * 1000000000)
            elif match.endswith('T'):
                numbers.append(float(match[:-1]) * 1000000000000)
            else:
                numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "untitled"
    
    # Remove HTML entities
    filename = html.unescape(filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Remove invalid characters
    valid_chars = f"-_.{string.ascii_letters}{string.digits}"
    filename = ''.join(c for c in filename if c in valid_chars)
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename or "untitled"

def generate_report_title(data: Dict, template: str = None) -> str:
    """
    Generate report titles based on data and templates.
    
    Args:
        data: Data dictionary for title generation
        template: Title template
        
    Returns:
        Generated title
    """
    if template is None:
        template = "Market Research Report - {date}"
    
    try:
        # Replace placeholders in template
        title = template.format(**data)
    except KeyError:
        # Fallback to basic title if template fails
        title = f"Market Research Report - {data.get('date', 'N/A')}"
    
    return title

def camel_to_snake(name: str) -> str:
    """
    Convert CamelCase to snake_case.
    
    Args:
        name: CamelCase string
        
    Returns:
        snake_case string
    """
    # Insert underscore before uppercase letters
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snake_to_camel(name: str) -> str:
    """
    Convert snake_case to CamelCase.
    
    Args:
        name: snake_case string
        
    Returns:
        CamelCase string
    """
    components = name.split('_')
    return ''.join(word.capitalize() for word in components)

def extract_date_strings(text: str) -> List[str]:
    """
    Extract date strings from text.
    
    Args:
        text: Input text
        
    Returns:
        List of date strings
    """
    if not text:
        return []
    
    # Various date patterns
    patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
        r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b',  # DD Mon YYYY
    ]
    
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return list(set(dates))  # Remove duplicates

def create_slug(text: str, max_length: int = 50) -> str:
    """
    Create URL-friendly slug from text.
    
    Args:
        text: Input text
        max_length: Maximum length of slug
        
    Returns:
        URL-friendly slug
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and special characters with hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    
    # Remove leading/trailing hyphens
    text = text.strip('-')
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length].rstrip('-')
    
    return text

def mask_sensitive_data(text: str, mask_char: str = '*') -> str:
    """
    Mask sensitive data in text (API keys, passwords, etc.).
    
    Args:
        text: Input text
        mask_char: Character to use for masking
        
    Returns:
        Text with sensitive data masked
    """
    if not text:
        return ""
    
    # Patterns for sensitive data
    patterns = [
        (r'api[_-]?key[_-]?[=:]\s*([^\s,]+)', r'api_key=\1'),  # API keys
        (r'password[_-]?[=:]\s*([^\s,]+)', r'password=\1'),    # Passwords
        (r'token[_-]?[=:]\s*([^\s,]+)', r'token=\1'),          # Tokens
        (r'secret[_-]?[=:]\s*([^\s,]+)', r'secret=\1'),        # Secrets
    ]
    
    masked_text = text
    for pattern, replacement in patterns:
        def mask_match(match):
            key_part = match.group(0).split('=')[0] + '='
            value = match.group(1)
            if len(value) > 6:
                masked_value = value[:2] + mask_char * (len(value) - 4) + value[-2:]
            else:
                masked_value = mask_char * len(value)
            return key_part + masked_value
        
        masked_text = re.sub(pattern, mask_match, masked_text, flags=re.IGNORECASE)
    
    return masked_text

def truncate_text(text: str, max_length: int = 100, 
                 suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append when truncating
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def remove_emoji(text: str) -> str:
    """
    Remove emoji characters from text.
    
    Args:
        text: Input text
        
    Returns:
        Text without emoji
    """
    if not text:
        return ""
    
    # Pattern to match emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'', text)

def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters in text.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Normalize unicode to NFKD form
    normalized = unicodedata.normalize('NFKD', text)
    
    # Remove combining characters
    ascii_text = ''.join(c for c in normalized if not unicodedata.combining(c))
    
    return ascii_text

def format_indian_number(number: Union[int, float]) -> str:
    """
    Format numbers according to Indian numbering system (lakhs, crores).
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string with Indian conventions
    """
    if number is None:
        return "0"
    
    abs_number = abs(number)
    negative = number < 0
    
    if abs_number >= 10000000:  # 1 crore
        formatted = f"{number/10000000:.2f} Crores"
    elif abs_number >= 100000:  # 1 lakh
        formatted = f"{number/100000:.2f} Lakhs"
    elif abs_number >= 1000:
        formatted = f"{number/1000:.2f} Thousands"
    else:
        formatted = f"{number:.2f}"
    
    return formatted

def extract_nse_bse_symbols(text: str) -> Dict[str, List[str]]:
    """
    Extract NSE and BSE stock symbols separately from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with NSE and BSE symbols
    """
    if not text:
        return {"NSE": [], "BSE": []}
    
    # NSE symbols pattern
    nse_pattern = r'\b[A-Z]{2,6}\.NS\b|\b[A-Z]{2,6}(?=\s|$|\W)'
    # BSE symbols pattern  
    bse_pattern = r'\b[A-Z]{2,6}\.BO\b|\b[A-Z]{2,6}\.BSE\b'
    
    nse_matches = re.findall(nse_pattern, text.upper())
    bse_matches = re.findall(bse_pattern, text.upper())
    
    # Clean symbols
    nse_symbols = [clean_ticker_symbol(match) for match in nse_matches]
    bse_symbols = [clean_ticker_symbol(match) for match in bse_matches]
    
    return {
        "NSE": list(set(nse_symbols)),
        "BSE": list(set(bse_symbols))
    }

def format_market_time(timestamp: str, market: str = "NSE") -> str:
    """
    Format timestamp according to Indian market hours.
    
    Args:
        timestamp: Input timestamp
        market: Market name (NSE/BSE)
        
    Returns:
        Formatted time string
    """
    from datetime import datetime
    
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
            
        # Convert to IST (UTC+5:30)
        ist_offset = 5.5 * 3600
        ist_dt = dt + timedelta(seconds=ist_offset)
        
        # Market hours check
        market_open = ist_dt.replace(hour=9, minute=15, second=0)
        market_close = ist_dt.replace(hour=15, minute=30, second=0)
        
        status = ""
        if market_open <= ist_dt <= market_close:
            status = " (Market Open)"
        else:
            status = " (Market Closed)"
            
        return f"{ist_dt.strftime('%d-%m-%Y %H:%M:%S IST')}{status}"
        
    except Exception:
        return timestamp

def clean_financial_text(text: str) -> str:
    """
    Clean financial text data for Indian markets.
    
    Args:
        text: Raw financial text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove common financial document artifacts
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'SEBI.*?regulations?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'BSE.*?Ltd\.?', 'BSE', text, flags=re.IGNORECASE)
    text = re.sub(r'NSE.*?Ltd\.?', 'NSE', text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_financial_metrics(text: str) -> Dict[str, float]:
    """
    Extract common financial metrics from text.
    
    Args:
        text: Input text containing financial data
        
    Returns:
        Dictionary of extracted metrics
    """
    metrics = {}
    
    # Common financial patterns for Indian markets
    patterns = {
        'market_cap': r'market\s+cap(?:italization)?[:\s]+([0-9,]+(?:\.[0-9]+)?)\s*(crores?|lakhs?)?',
        'pe_ratio': r'p/?e\s+ratio[:\s]+([0-9]+(?:\.[0-9]+)?)',
        'pb_ratio': r'p/?b\s+ratio[:\s]+([0-9]+(?:\.[0-9]+)?)',
        'dividend_yield': r'dividend\s+yield[:\s]+([0-9]+(?:\.[0-9]+)?)%?',
        'roe': r'r\.?o\.?e\.?[:\s]+([0-9]+(?:\.[0-9]+)?)%?',
        'debt_equity': r'debt[/-]equity[:\s]+([0-9]+(?:\.[0-9]+)?)',
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, text.lower())
        if match:
            try:
                value = float(match.group(1).replace(',', ''))
                # Convert lakhs/crores to actual numbers
                if len(match.groups()) > 1 and match.group(2):
                    if 'crore' in match.group(2).lower():
                        value *= 10000000
                    elif 'lakh' in match.group(2).lower():
                        value *= 100000
                metrics[metric] = value
            except (ValueError, AttributeError):
                continue
    
    return metrics

def standardize_sector_name(sector: str) -> str:
    """
    Standardize sector names for Indian markets.
    
    Args:
        sector: Raw sector name
        
    Returns:
        Standardized sector name
    """
    if not sector:
        return "Others"
    
    sector = sector.strip().title()
    
    # Common sector mappings
    sector_mapping = {
        'It': 'Information Technology',
        'Fmcg': 'FMCG',
        'Pharma': 'Pharmaceuticals',
        'Auto': 'Automobile',
        'Realty': 'Real Estate',
        'Psu': 'PSU',
        'Nbfc': 'NBFC',
        'Bfsi': 'BFSI',
        'Infra': 'Infrastructure',
        'Telecom': 'Telecommunications',
        'Cement': 'Cement & Construction',
        'Steel': 'Metals & Mining',
        'Oil': 'Oil & Gas',
        'Power': 'Power & Energy',
        'Textiles': 'Textiles & Apparel'
    }
    
    for key, value in sector_mapping.items():
        if key.lower() in sector.lower():
            return value
    
    return sector

def format_volume_indian(volume: Union[int, float]) -> str:
    """
    Format trading volume for Indian markets.
    
    Args:
        volume: Trading volume
        
    Returns:
        Formatted volume string
    """
    if volume is None or volume == 0:
        return "0"
    
    if volume >= 10000000:  # 1 crore
        return f"{volume/10000000:.2f} Cr"
    elif volume >= 100000:  # 1 lakh
        return f"{volume/100000:.2f} L"
    elif volume >= 1000:
        return f"{volume/1000:.2f} K"
    else:
        return f"{int(volume)}"

def validate_indian_stock_symbol(symbol: str) -> bool:
    """
    Validate if a symbol is a valid Indian stock symbol.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol:
        return False
    
    symbol = symbol.upper().strip()
    
    # Basic validation patterns
    patterns = [
        r'^[A-Z]{2,6}$',           # Basic NSE symbols
        r'^[A-Z]{2,6}\.NS$',       # NSE with suffix
        r'^[A-Z]{2,6}\.BO$',       # BSE with suffix
        r'^[A-Z]{2,6}\.BSE$',      # BSE alternate suffix
    ]
    
    return any(re.match(pattern, symbol) for pattern in patterns)

def generate_report_filename(symbol: str, report_type: str, date: str) -> str:
    """
    Generate standardized report filenames.
    
    Args:
        symbol: Stock symbol
        report_type: Type of report
        date: Date string
        
    Returns:
        Formatted filename
    """
    clean_symbol = clean_ticker_symbol(symbol) if symbol else "MARKET"
    clean_type = sanitize_filename(report_type)
    clean_date = date.replace('-', '').replace('/', '').replace(' ', '_')
    
    filename = f"{clean_symbol}_{clean_type}_{clean_date}.pdf"
    return filename.lower()

def parse_indian_currency_text(text: str) -> float:
    """
    Parse Indian currency text and convert to numeric value.
    
    Args:
        text: Currency text (e.g., "₹1.5 Crores", "Rs. 50 Lakhs")
        
    Returns:
        Numeric value
    """
    if not text:
        return 0.0
    
    # Remove currency symbols
    text = re.sub(r'[₹Rs\.\s]', '', text, flags=re.IGNORECASE)
    
    # Extract number and multiplier
    match = re.search(r'([0-9,]+(?:\.[0-9]+)?)\s*(crores?|lakhs?|thousands?)?', text.lower())
    
    if not match:
        return 0.0
    
    try:
        value = float(match.group(1).replace(',', ''))
        multiplier = match.group(2) if len(match.groups()) > 1 else None
        
        if multiplier:
            if 'crore' in multiplier:
                value *= 10000000
            elif 'lakh' in multiplier:
                value *= 100000
            elif 'thousand' in multiplier:
                value *= 1000
                
        return value
    except (ValueError, AttributeError):
        return 0.0

# Version 1 specific utility functions for basic market research
def create_market_summary_header(date: str, market_status: str = "CLOSED") -> str:
    """
    Create header for market summary reports.
    
    Args:
        date: Report date
        market_status: Current market status
        
    Returns:
        Formatted header string
    """
    header = f"""
╔══════════════════════════════════════════════════════════════╗
║                    INDIAN MARKET RESEARCH REPORT             ║
║                         Version 1.0 (2022)                  ║
║══════════════════════════════════════════════════════════════║
║  Date: {date:<25} Market: {market_status:<15} ║
║  Generated: {datetime.now().strftime('%d-%m-%Y %H:%M IST'):<37} ║
╚══════════════════════════════════════════════════════════════╝
    """
    return header.strip()

def format_basic_stock_info(symbol: str, price: float, change: float, volume: int) -> str:
    """
    Format basic stock information for Version 1 reports.
    
    Args:
        symbol: Stock symbol
        price: Current price
        change: Price change
        volume: Trading volume
        
    Returns:
        Formatted stock info string
    """
    change_pct = (change / (price - change)) * 100 if price != change else 0
    change_indicator = "📈" if change > 0 else "📉" if change < 0 else "➡️"
    
    return f"""
Symbol: {symbol}
Price: ₹{price:.2f} {change_indicator} {change:+.2f} ({change_pct:+.2f}%)
Volume: {format_volume_indian(volume)}
    """.strip()

# Import required modules for datetime functionality
from datetime import datetime, timedelta