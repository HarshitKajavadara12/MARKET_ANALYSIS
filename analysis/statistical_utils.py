"""
Statistical utility functions for Market Research System v1.0
Created: January 2022
Author: Independent Market Researcher
Focus: Indian Stock Market Analysis

This module provides comprehensive statistical functions for market data analysis,
including descriptive statistics, hypothesis testing, and advanced statistical measures.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro
from typing import Union, List, Tuple, Dict, Optional
import warnings
from ..exceptions.analysis_exceptions import (
    CalculationException, 
    InsufficientDataException,
    DataQualityException,
    handle_analysis_error
)

warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Main class for statistical analysis operations"""
    
    def __init__(self):
        self.min_observations = 30
        
    @handle_analysis_error
    def descriptive_stats(self, data: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Calculate comprehensive descriptive statistics
        
        Args:
            data: Price series or returns data
            
        Returns:
            Dictionary containing all descriptive statistics
        """
        if len(data) < 2:
            raise InsufficientDataException(2, len(data))
            
        # Convert to pandas Series if numpy array
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        # Remove NaN values
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            raise DataQualityException("All data points are NaN")
            
        stats_dict = {
            # Basic statistics
            'count': len(clean_data),
            'mean': clean_data.mean(),
            'median': clean_data.median(),
            'mode': clean_data.mode().iloc[0] if not clean_data.mode().empty else np.nan,
            'std': clean_data.std(),
            'var': clean_data.var(),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'range': clean_data.max() - clean_data.min(),
            
            # Percentiles
            'q1': clean_data.quantile(0.25),
            'q3': clean_data.quantile(0.75),
            'iqr': clean_data.quantile(0.75) - clean_data.quantile(0.25),
            
            # Distribution shape
            'skewness': clean_data.skew(),
            'kurtosis': clean_data.kurtosis(),
            'excess_kurtosis': clean_data.kurtosis() - 3,
            
            # Additional metrics
            'coefficient_variation': clean_data.std() / abs(clean_data.mean()) if clean_data.mean() != 0 else np.inf,
            'mad': self.mean_absolute_deviation(clean_data),
            'semi_variance': self.semi_variance(clean_data),
            'downside_deviation': self.downside_deviation(clean_data)
        }
        
        return stats_dict
    
    @handle_analysis_error
    def returns_statistics(self, prices: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Calculate return-specific statistics
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary containing return statistics
        """
        if len(prices) < 2:
            raise InsufficientDataException(2, len(prices))
            
        # Calculate returns
        returns = self.calculate_returns(prices)
        
        # Get basic descriptive stats
        basic_stats = self.descriptive_stats(returns)
        
        # Add return-specific metrics
        return_stats = {
            **basic_stats,
            'annualized_return': self.annualized_return(returns),
            'annualized_volatility': self.annualized_volatility(returns),
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(returns),
            'max_drawdown': self.max_drawdown(prices),
            'var_95': self.value_at_risk(returns, 0.05),
            'var_99': self.value_at_risk(returns, 0.01),
            'cvar_95': self.conditional_var(returns, 0.05),
            'cvar_99': self.conditional_var(returns, 0.01),
            'gain_loss_ratio': self.gain_loss_ratio(returns),
            'win_rate': self.win_rate(returns)
        }
        
        return return_stats
    
    @handle_analysis_error
    def normality_tests(self, data: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Perform multiple normality tests
        
        Args:
            data: Data series to test
            
        Returns:
            Dictionary with test results
        """
        if len(data) < 8:
            raise InsufficientDataException(8, len(data))
            
        clean_data = pd.Series(data).dropna()
        
        results = {}
        
        # Jarque-Bera test
        try:
            jb_stat, jb_pvalue = jarque_bera(clean_data)
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'is_normal': jb_pvalue > 0.05
            }
        except Exception as e:
            results['jarque_bera'] = {'error': str(e)}
        
        # D'Agostino's normality test
        try:
            da_stat, da_pvalue = normaltest(clean_data)
            results['dagostino'] = {
                'statistic': da_stat,
                'p_value': da_pvalue,
                'is_normal': da_pvalue > 0.05
            }
        except Exception as e:
            results['dagostino'] = {'error': str(e)}
        
        # Shapiro-Wilk test (for smaller samples)
        if len(clean_data) <= 5000:
            try:
                sw_stat, sw_pvalue = shapiro(clean_data)
                results['shapiro_wilk'] = {
                    'statistic': sw_stat,
                    'p_value': sw_pvalue,
                    'is_normal': sw_pvalue > 0.05
                }
            except Exception as e:
                results['shapiro_wilk'] = {'error': str(e)}
        
        return results
    
    @handle_analysis_error
    def correlation_analysis(self, x: Union[pd.Series, np.ndarray], 
                           y: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Comprehensive correlation analysis between two series
        
        Args:
            x, y: Data series to analyze
            
        Returns:
            Dictionary with correlation metrics
        """
        # Align data and remove NaN
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(df) < 3:
            raise InsufficientDataException(3, len(df))
            
        x_clean, y_clean = df['x'], df['y']
        
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(x_clean, y_clean)
        
        # Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(x_clean, y_clean)
        
        # Kendall's tau
        kendall_corr, kendall_p = stats.kendalltau(x_clean, y_clean)
        
        return {
            'pearson': {
                'correlation': pearson_corr,
                'p_value': pearson_p,
                'significant': pearson_p < 0.05
            },
            'spearman': {
                'correlation': spearman_corr,
                'p_value': spearman_p,
                'significant': spearman_p < 0.05
            },
            'kendall': {
                'correlation': kendall_corr,
                'p_value': kendall_p,
                'significant': kendall_p < 0.05
            },
            'sample_size': len(df)
        }
    
    # Utility functions
    @staticmethod
    def calculate_returns(prices: Union[pd.Series, np.ndarray], 
                         method: str = 'simple') -> pd.Series:
        """Calculate returns from price series"""
        prices = pd.Series(prices) if isinstance(prices, np.ndarray) else prices
        
        if method == 'simple':
            return prices.pct_change().dropna()
        elif method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
    
    @staticmethod
    def mean_absolute_deviation(data: pd.Series) -> float:
        """Calculate Mean Absolute Deviation"""
        return np.mean(np.abs(data - data.mean()))
    
    @staticmethod
    def semi_variance(data: pd.Series, benchmark: float = 0) -> float:
        """Calculate semi-variance (downside variance)"""
        downside_returns = data[data < benchmark]
        if len(downside_returns) == 0:
            return 0
        return downside_returns.var()
    
    @staticmethod
    def downside_deviation(data: pd.Series, benchmark: float = 0) -> float:
        """Calculate downside deviation"""
        return np.sqrt(StatisticalAnalyzer.semi_variance(data, benchmark))
    
    @staticmethod
    def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0
        total_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year
        return (1 + total_return) ** (1/years) - 1
    
    @staticmethod
    def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04, 
                    periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/periods_per_year
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.04,
                     periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/periods_per_year
        downside_std = StatisticalAnalyzer.downside_deviation(excess_returns)
        if downside_std == 0:
            return 0
        return excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    
    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = StatisticalAnalyzer.annualized_return(returns)
        max_dd = StatisticalAnalyzer.max_drawdown_from_returns(returns)
        if max_dd == 0:
            return 0
        return annual_return / abs(max_dd)
    
    @staticmethod
    def max_drawdown(prices: Union[pd.Series, np.ndarray]) -> float:
        """Calculate maximum drawdown from price series"""
        prices = pd.Series(prices) if isinstance(prices, np.ndarray) else prices
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def max_drawdown_from_returns(returns: pd.Series) -> float:
        """Calculate maximum drawdown from return series"""
        cumulative = (1 + returns).cumprod()
        return StatisticalAnalyzer.max_drawdown(cumulative)
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return returns.quantile(confidence_level)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = StatisticalAnalyzer.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def gain_loss_ratio(returns: pd.Series) -> float:
        """Calculate gain-to-loss ratio"""
        gains = returns[returns > 0].mean()
        losses = abs(returns[returns < 0].mean())
        if losses == 0:
            return np.inf if gains > 0 else 0
        return gains / losses
    
    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0
        return len(returns[returns > 0]) / len(returns)

# Additional statistical functions
def rolling_statistics(data: pd.Series, window: int, 
                      stat_func: str = 'mean') -> pd.Series:
    """Calculate rolling statistics"""
    if stat_func == 'mean':
        return data.rolling(window=window).mean()
    elif stat_func == 'std':
        return data.rolling(window=window).std()
    elif stat_func == 'var':
        return data.rolling(window=window).var()
    elif stat_func == 'skew':
        return data.rolling(window=window).skew()
    elif stat_func == 'kurt':
        return data.rolling(window=window).kurt()
    else:
        raise ValueError(f"Unsupported statistic: {stat_func}")

def z_score(data: pd.Series, window: int = None) -> pd.Series:
    """Calculate z-score (standardized values)"""
    if window:
        mean = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        return (data - mean) / std
    else:
        return (data - data.mean()) / data.std()

def percentile_rank(data: pd.Series, window: int = None) -> pd.Series:
    """Calculate percentile rank"""
    if window:
        return data.rolling(window=window).rank(pct=True)
    else:
        return data.rank(pct=True)

def seasonal_decompose_stats(data: pd.Series, period: int = 252) -> Dict:
    """Perform basic seasonal decomposition statistics"""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(data.dropna(), period=period, model='additive')
        
        return {
            'trend_strength': 1 - (decomposition.resid.var() / 
                                 (decomposition.resid + decomposition.trend).var()),
            'seasonal_strength': 1 - (decomposition.resid.var() / 
                                    (decomposition.resid + decomposition.seasonal).var()),
            'residual_variance': decomposition.resid.var()
        }
    except Exception as e:
        return {'error': f"Seasonal decomposition failed: {str(e)}"}

# Initialize global analyzer instance
analyzer = StatisticalAnalyzer()