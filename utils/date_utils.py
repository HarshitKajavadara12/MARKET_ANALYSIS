"""
Market Research System v1.0 - Date Utilities
Created: 2022
Author: Independent Market Researcher

Date and time utility functions for market data handling.
Supports Indian Stock Exchange (NSE/BSE) trading hours and holidays.
"""

import datetime as dt
from datetime import datetime, timedelta, date
import pandas as pd
import pytz
from typing import List, Tuple, Optional, Union
import calendar
import numpy as np

# Indian Stock Exchange timezone
IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC

# Indian Stock Market Hours (NSE/BSE)
MARKET_OPEN_TIME = dt.time(9, 15)  # 9:15 AM IST
MARKET_CLOSE_TIME = dt.time(15, 30)  # 3:30 PM IST

# Indian Stock Market Holidays (2022 - Sample)
INDIAN_MARKET_HOLIDAYS_2022 = [
    '2022-01-26',  # Republic Day
    '2022-03-01',  # Mahashivratri
    '2022-03-18',  # Holi
    '2022-04-14',  # Ram Navami
    '2022-04-15',  # Good Friday
    '2022-05-03',  # Eid ul-Fitr
    '2022-08-09',  # Muharram
    '2022-08-15',  # Independence Day
    '2022-08-31',  # Ganesh Chaturthi
    '2022-10-02',  # Gandhi Jayanti
    '2022-10-05',  # Dussehra
    '2022-10-24',  # Diwali
    '2022-11-08',  # Guru Nanak Jayanti
]

def format_date(date_obj: Union[datetime, date, str], format_string: str = '%Y-%m-%d') -> str:
    """
    Format date object to string.
    
    Args:
        date_obj: Date object to format
        format_string: Format string (default: '%Y-%m-%d')
    
    Returns:
        Formatted date string
    """
    try:
        if isinstance(date_obj, str):
            date_obj = parse_date(date_obj)
        elif isinstance(date_obj, date) and not isinstance(date_obj, datetime):
            date_obj = datetime.combine(date_obj, datetime.min.time())
        
        return date_obj.strftime(format_string)
    except Exception as e:
        raise ValueError(f"Error formatting date: {e}")

def parse_date(date_string: str, format_string: Optional[str] = None) -> datetime:
    """
    Parse date string to datetime object.
    
    Args:
        date_string: Date string to parse
        format_string: Expected format (auto-detect if None)
    
    Returns:
        Parsed datetime object
    """
    try:
        if format_string:
            return datetime.strptime(date_string, format_string)
        
        # Common date formats to try
        formats = [
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S',
            '%Y%m%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        
        # Try pandas parsing as last resort
        return pd.to_datetime(date_string).to_pydatetime()
        
    except Exception as e:
        raise ValueError(f"Unable to parse date '{date_string}': {e}")

def get_market_days(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> List[datetime]:
    """
    Get list of market trading days between two dates (excludes weekends and holidays).
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        List of trading days
    """
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)
    
    # Get all business days (Mon-Fri)
    business_days = pd.bdate_range(start=start_date, end=end_date)
    
    # Remove Indian market holidays
    holidays = [parse_date(holiday) for holiday in INDIAN_MARKET_HOLIDAYS_2022]
    
    trading_days = []
    for day in business_days:
        day_dt = day.to_pydatetime().replace(tzinfo=None)
        if not any(day_dt.date() == holiday.date() for holiday in holidays):
            trading_days.append(day_dt)
    
    return trading_days

def is_market_open(check_datetime: Optional[datetime] = None) -> bool:
    """
    Check if Indian stock market is currently open.
    
    Args:
        check_datetime: Datetime to check (current time if None)
    
    Returns:
        True if market is open
    """
    if check_datetime is None:
        check_datetime = datetime.now(IST)
    elif check_datetime.tzinfo is None:
        check_datetime = IST.localize(check_datetime)
    
    # Convert to IST if needed
    if check_datetime.tzinfo != IST:
        check_datetime = check_datetime.astimezone(IST)
    
    # Check if it's a weekday
    if check_datetime.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if it's a holiday
    date_str = check_datetime.strftime('%Y-%m-%d')
    if date_str in INDIAN_MARKET_HOLIDAYS_2022:
        return False
    
    # Check market hours
    current_time = check_datetime.time()
    return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME

def get_trading_calendar(year: int = 2022) -> List[datetime]:
    """
    Get complete trading calendar for a year.
    
    Args:
        year: Year to get calendar for
    
    Returns:
        List of all trading days in the year
    """
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    return get_market_days(start_date, end_date)

def calculate_business_days(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> int:
    """
    Calculate number of business days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        Number of business days
    """
    trading_days = get_market_days(start_date, end_date)
    return len(trading_days)

def get_date_range(start_date: Union[str, datetime], end_date: Union[str, datetime], 
                  freq: str = 'D') -> List[datetime]:
    """
    Generate date range with specified frequency.
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
    
    Returns:
        List of dates
    """
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    return [d.to_pydatetime() for d in date_range]

def convert_timezone(dt_obj: datetime, from_tz: str = 'UTC', to_tz: str = 'Asia/Kolkata') -> datetime:
    """
    Convert datetime from one timezone to another.
    
    Args:
        dt_obj: Datetime object to convert
        from_tz: Source timezone
        to_tz: Target timezone
    
    Returns:
        Converted datetime object
    """
    try:
        from_timezone = pytz.timezone(from_tz)
        to_timezone = pytz.timezone(to_tz)
        
        # Localize if naive
        if dt_obj.tzinfo is None:
            dt_obj = from_timezone.localize(dt_obj)
        
        return dt_obj.astimezone(to_timezone)
    except Exception as e:
        raise ValueError(f"Error converting timezone: {e}")

def get_market_hours(date_obj: Union[str, datetime] = None) -> Tuple[datetime, datetime]:
    """
    Get market opening and closing times for a specific date.
    
    Args:
        date_obj: Date to get market hours for (today if None)
    
    Returns:
        Tuple of (market_open, market_close) datetime objects
    """
    if date_obj is None:
        date_obj = datetime.now(IST).date()
    elif isinstance(date_obj, str):
        date_obj = parse_date(date_obj).date()
    elif isinstance(date_obj, datetime):
        date_obj = date_obj.date()
    
    market_open = IST.localize(datetime.combine(date_obj, MARKET_OPEN_TIME))
    market_close = IST.localize(datetime.combine(date_obj, MARKET_CLOSE_TIME))
    
    return market_open, market_close

def get_previous_trading_day(reference_date: Union[str, datetime] = None) -> datetime:
    """
    Get the previous trading day.
    
    Args:
        reference_date: Reference date (today if None)
    
    Returns:
        Previous trading day
    """
    if reference_date is None:
        reference_date = datetime.now(IST)
    elif isinstance(reference_date, str):
        reference_date = parse_date(reference_date)
    
    # Start from the day before reference date
    check_date = reference_date - timedelta(days=1)
    
    while True:
        # Check if it's a trading day
        if (check_date.weekday() < 5 and  # Not weekend
            check_date.strftime('%Y-%m-%d') not in INDIAN_MARKET_HOLIDAYS_2022):
            return check_date
        check_date -= timedelta(days=1)

def get_next_trading_day(reference_date: Union[str, datetime] = None) -> datetime:
    """
    Get the next trading day.
    
    Args:
        reference_date: Reference date (today if None)
    
    Returns:
        Next trading day
    """
    if reference_date is None:
        reference_date = datetime.now(IST)
    elif isinstance(reference_date, str):
        reference_date = parse_date(reference_date)
    
    # Start from the day after reference date
    check_date = reference_date + timedelta(days=1)
    
    while True:
        # Check if it's a trading day
        if (check_date.weekday() < 5 and  # Not weekend
            check_date.strftime('%Y-%m-%d') not in INDIAN_MARKET_HOLIDAYS_2022):
            return check_date
        check_date += timedelta(days=1)

def get_quarter_dates(year: int, quarter: int) -> Tuple[datetime, datetime]:
    """
    Get start and end dates for a quarter.
    
    Args:
        year: Year
        quarter: Quarter (1-4)
    
    Returns:
        Tuple of (quarter_start, quarter_end)
    """
    if quarter not in [1, 2, 3, 4]:
        raise ValueError("Quarter must be 1, 2, 3, or 4")
    
    quarter_starts = {
        1: (1, 1),   # Q1: Jan-Mar
        2: (4, 1),   # Q2: Apr-Jun
        3: (7, 1),   # Q3: Jul-Sep
        4: (10, 1)   # Q4: Oct-Dec
    }
    
    quarter_ends = {
        1: (3, 31),  # Q1: Jan-Mar
        2: (6, 30),  # Q2: Apr-Jun
        3: (9, 30),  # Q3: Jul-Sep
        4: (12, 31)  # Q4: Oct-Dec
    }
    
    start_month, start_day = quarter_starts[quarter]
    end_month, end_day = quarter_ends[quarter]
    
    quarter_start = datetime(year, start_month, start_day)
    quarter_end = datetime(year, end_month, end_day)
    
    return quarter_start, quarter_end

def time_until_market_open() -> Optional[timedelta]:
    """
    Calculate time until next market opening.
    
    Returns:
        Timedelta until market opens (None if market is currently open)
    """
    now = datetime.now(IST)
    
    if is_market_open(now):
        return None
    
    # Get next trading day
    next_trading_day = get_next_trading_day(now)
    market_open, _ = get_market_hours(next_trading_day)
    
    return market_open - now

def time_until_market_close() -> Optional[timedelta]:
    """
    Calculate time until market closing.
    
    Returns:
        Timedelta until market closes (None if market is closed)
    """
    now = datetime.now(IST)
    
    if not is_market_open(now):
        return None
    
    _, market_close = get_market_hours(now)
    
    return market_close - now