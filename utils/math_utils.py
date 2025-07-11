"""
Mathematical Utilities for Market Research System v1.0
Provides essential mathematical functions for financial calculations and statistical analysis.
Created: January 2022
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional
import warnings

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

def safe_divide(numerator: Union[float, np.ndarray], denominator: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safely divide two numbers or arrays, handling division by zero.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return when division by zero occurs
        
    Returns:
        Result of division or default value
    """
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(denominator != 0, numerator / denominator, default)
            return result
    except Exception:
        return default

def calculate_returns(prices: Union[List, np.ndarray, pd.Series], 
                     method: str = 'simple') -> np.ndarray:
    """
    Calculate returns from a price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log' returns
        
    Returns:
        Array of returns
    """
    prices = np.array(prices)
    
    if len(prices) < 2:
        return np.array([])
    
    if method == 'simple':
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
    elif method == 'log':
        returns = np.log(prices[1:] / prices[:-1])
    else:
        raise ValueError("Method must be 'simple' or 'log'")
    
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

def calculate_volatility(returns: Union[List, np.ndarray], 
                        annualize: bool = True, 
                        trading_days: int = 252) -> float:
    """
    Calculate volatility from returns.
    
    Args:
        returns: Return series
        annualize: Whether to annualize the volatility
        trading_days: Number of trading days per year
        
    Returns:
        Volatility (standard deviation of returns)
    """
    returns = np.array(returns)
    
    if len(returns) == 0:
        return 0.0
    
    vol = np.std(returns, ddof=1)
    
    if annualize:
        vol *= np.sqrt(trading_days)
    
    return float(vol)

def calculate_sharpe_ratio(returns: Union[List, np.ndarray], 
                          risk_free_rate: float = 0.02,
                          trading_days: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        trading_days: Number of trading days per year
        
    Returns:
        Sharpe ratio
    """
    returns = np.array(returns)
    
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / trading_days)
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    sharpe = (mean_excess / std_excess) * np.sqrt(trading_days)
    return float(sharpe)

def calculate_max_drawdown(prices: Union[List, np.ndarray, pd.Series]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from a price series.
    
    Args:
        prices: Price series
        
    Returns:
        Tuple of (max_drawdown, start_index, end_index)
    """
    prices = np.array(prices)
    
    if len(prices) == 0:
        return 0.0, 0, 0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(prices)
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    
    # Find start of drawdown period
    start_idx = np.argmax(running_max[:max_dd_idx + 1])
    
    return float(abs(max_dd)), int(start_idx), int(max_dd_idx)

def calculate_correlation(x: Union[List, np.ndarray], 
                         y: Union[List, np.ndarray],
                         method: str = 'pearson') -> float:
    """
    Calculate correlation between two series.
    
    Args:
        x: First series
        y: Second series
        method: 'pearson' or 'spearman'
        
    Returns:
        Correlation coefficient
    """
    x, y = np.array(x), np.array(y)
    
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 2:
        return 0.0
    
    if method == 'pearson':
        corr = np.corrcoef(x, y)[0, 1]
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr, _ = spearmanr(x, y)
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")
    
    return float(corr) if not np.isnan(corr) else 0.0

def moving_average(data: Union[List, np.ndarray], window: int, 
                  method: str = 'simple') -> np.ndarray:
    """
    Calculate moving average.
    
    Args:
        data: Data series
        window: Window size
        method: 'simple' or 'exponential'
        
    Returns:
        Moving average array
    """
    data = np.array(data)
    
    if len(data) < window:
        return np.full(len(data), np.nan)
    
    if method == 'simple':
        ma = np.convolve(data, np.ones(window)/window, mode='valid')
        result = np.concatenate([np.full(window-1, np.nan), ma])
    elif method == 'exponential':
        alpha = 2 / (window + 1)
        result = np.zeros_like(data, dtype=float)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    else:
        raise ValueError("Method must be 'simple' or 'exponential'")
    
    return result

def calculate_rsi(prices: Union[List, np.ndarray], period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI values
    """
    prices = np.array(prices)
    
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gains = np.zeros_like(gains)
    avg_losses = np.zeros_like(losses)
    
    # Initial averages
    avg_gains[period-1] = np.mean(gains[:period])
    avg_losses[period-1] = np.mean(losses[:period])
    
    # Exponential moving averages
    for i in range(period, len(gains)):
        avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
    
    # Calculate RSI
    rs = safe_divide(avg_gains, avg_losses, default=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Prepend NaN for the first price (no change calculated)
    rsi = np.concatenate([[np.nan], rsi])
    
    return rsi

def calculate_bollinger_bands(prices: Union[List, np.ndarray], 
                            window: int = 20, 
                            num_std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price series
        window: Moving average window
        num_std: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    prices = np.array(prices)
    
    # Calculate middle band (SMA)
    middle_band = moving_average(prices, window, 'simple')
    
    # Calculate standard deviation
    std_dev = np.zeros_like(prices)
    for i in range(window-1, len(prices)):
        std_dev[i] = np.std(prices[i-window+1:i+1])
    
    # Calculate upper and lower bands
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)
    
    return upper_band, middle_band, lower_band

def z_score(data: Union[List, np.ndarray], window: Optional[int] = None) -> np.ndarray:
    """
    Calculate z-score (standardization).
    
    Args:
        data: Data series
        window: Rolling window size (if None, use entire series)
        
    Returns:
        Z-score values
    """
    data = np.array(data)
    
    if window is None:
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return (data - mean) / std if std != 0 else np.zeros_like(data)
    
    z_scores = np.zeros_like(data)
    for i in range(window-1, len(data)):
        window_data = data[i-window+1:i+1]
        mean = np.mean(window_data)
        std = np.std(window_data, ddof=1)
        z_scores[i] = (data[i] - mean) / std if std != 0 else 0
    
    return z_scores

def percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def compound_annual_growth_rate(start_value: float, end_value: float, 
                               periods: float) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        start_value: Starting value
        end_value: Ending value
        periods: Number of periods (years)
        
    Returns:
        CAGR as decimal
    """
    if start_value <= 0 or end_value <= 0 or periods <= 0:
        return 0.0
    
    return (end_value / start_value) ** (1 / periods) - 1

def normalize_data(data: Union[List, np.ndarray], 
                  method: str = 'minmax') -> np.ndarray:
    """
    Normalize data using different methods.
    
    Args:
        data: Data to normalize
        method: 'minmax', 'zscore', or 'robust'
        
    Returns:
        Normalized data
    """
    data = np.array(data)
    
    if method == 'minmax':
        min_val, max_val = np.min(data), np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        return z_score(data)
    
    elif method == 'robust':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return (data - median) / mad if mad != 0 else np.zeros_like(data)
    
    else:
        raise ValueError("Method must be 'minmax', 'zscore', or 'robust'")

def risk_adjusted_return(returns: Union[List, np.ndarray], 
                        benchmark_returns: Union[List, np.ndarray],
                        risk_free_rate: float = 0.02) -> dict:
    """
    Calculate various risk-adjusted return metrics.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of risk metrics
    """
    returns = np.array(returns)
    benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = np.prod(1 + returns) - 1
    volatility = calculate_volatility(returns)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    
    # Tracking error and information ratio
    if len(benchmark_returns) == len(returns):
        tracking_error = np.std(returns - benchmark_returns, ddof=1) * np.sqrt(252)
        excess_return = np.mean(returns - benchmark_returns) * 252
        info_ratio = excess_return / tracking_error if tracking_error != 0 else 0
    else:
        tracking_error = 0
        info_ratio = 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns, ddof=1) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (np.mean(returns) * 252 - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
    
    return {
        'total_return': float(total_return),
        'annualized_return': float(np.mean(returns) * 252),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'tracking_error': float(tracking_error),
        'information_ratio': float(info_ratio)
    }

# Mathematical constants used in financial calculations
TRADING_DAYS_PER_YEAR = 252
BUSINESS_DAYS_PER_MONTH = 21
SECONDS_PER_DAY = 86400
MINUTES_PER_DAY = 1440

def annualize_metric(metric_value: float, frequency: str) -> float:
    """
    Annualize a metric based on its frequency.
    
    Args:
        metric_value: The metric value to annualize
        frequency: 'daily', 'weekly', 'monthly', 'quarterly'
        
    Returns:
        Annualized metric value
    """
    multipliers = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4
    }
    
    multiplier = multipliers.get(frequency.lower(), 1)
    return metric_value * multiplier