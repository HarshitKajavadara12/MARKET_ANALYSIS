"""
Volatility Analysis Module for Indian Stock Market
Market Research System v1.0 (2022)

This module provides comprehensive volatility analysis capabilities including:
- Historical volatility calculations
- GARCH modeling
- Volatility clustering analysis
- Risk metrics computation
- Volatility forecasting

Author: Market Research System
Date: 2022
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    """
    Comprehensive volatility analysis for Indian stock market data
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize VolatilityAnalyzer
        
        Args:
            data: DataFrame with columns ['date', 'symbol', 'close', 'high', 'low', 'volume']
        """
        self.data = data
        self.results = {}
        
    def calculate_returns(self, price_data: pd.Series, method: str = 'simple') -> pd.Series:
        """
        Calculate returns from price data
        
        Args:
            price_data: Series of price data
            method: 'simple' or 'logarithmic' returns
            
        Returns:
            Series of calculated returns
        """
        try:
            if method == 'simple':
                returns = price_data.pct_change().dropna()
            elif method == 'logarithmic':
                returns = np.log(price_data / price_data.shift(1)).dropna()
            else:
                raise ValueError("Method must be 'simple' or 'logarithmic'")
                
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return pd.Series()
    
    def historical_volatility(self, returns: pd.Series, window: int = 30, 
                            annualized: bool = True) -> pd.Series:
        """
        Calculate rolling historical volatility
        
        Args:
            returns: Series of returns
            window: Rolling window size
            annualized: Whether to annualize the volatility
            
        Returns:
            Series of historical volatility
        """
        try:
            # Calculate rolling standard deviation
            rolling_std = returns.rolling(window=window).std()
            
            if annualized:
                # Annualize volatility (assuming 252 trading days)
                rolling_std = rolling_std * np.sqrt(252)
                
            return rolling_std
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {str(e)}")
            return pd.Series()
    
    def parkinson_volatility(self, high: pd.Series, low: pd.Series, 
                           window: int = 30) -> pd.Series:
        """
        Calculate Parkinson volatility estimator using high-low prices
        
        Args:
            high: Series of high prices
            low: Series of low prices
            window: Rolling window size
            
        Returns:
            Series of Parkinson volatility
        """
        try:
            # Parkinson volatility formula
            hl_ratio = np.log(high / low)
            parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * hl_ratio**2)
            
            # Apply rolling window
            rolling_parkinson = parkinson_vol.rolling(window=window).mean()
            
            # Annualize
            rolling_parkinson = rolling_parkinson * np.sqrt(252)
            
            return rolling_parkinson
            
        except Exception as e:
            logger.error(f"Error calculating Parkinson volatility: {str(e)}")
            return pd.Series()
    
    def garman_klass_volatility(self, open_price: pd.Series, high: pd.Series, 
                               low: pd.Series, close: pd.Series, 
                               window: int = 30) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator
        
        Args:
            open_price: Series of opening prices
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            window: Rolling window size
            
        Returns:
            Series of Garman-Klass volatility
        """
        try:
            # Garman-Klass volatility components
            term1 = 0.5 * np.log(high / low)**2
            term2 = (2 * np.log(2) - 1) * np.log(close / open_price)**2
            
            gk_vol = np.sqrt(term1 - term2)
            
            # Apply rolling window
            rolling_gk = gk_vol.rolling(window=window).mean()
            
            # Annualize
            rolling_gk = rolling_gk * np.sqrt(252)
            
            return rolling_gk
            
        except Exception as e:
            logger.error(f"Error calculating Garman-Klass volatility: {str(e)}")
            return pd.Series()
    
    def volatility_clustering_analysis(self, returns: pd.Series) -> Dict:
        """
        Analyze volatility clustering in returns
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with clustering metrics
        """
        try:
            # Calculate absolute returns for volatility proxy
            abs_returns = np.abs(returns)
            
            # Autocorrelation of absolute returns (volatility clustering indicator)
            autocorr_lags = [1, 5, 10, 20, 50]
            autocorrelations = {}
            
            for lag in autocorr_lags:
                if len(abs_returns) > lag:
                    autocorr = abs_returns.autocorr(lag=lag)
                    autocorrelations[f'lag_{lag}'] = autocorr
            
            # ARCH test statistic (simplified)
            squared_returns = returns**2
            arch_stat = squared_returns.autocorr(lag=1) if len(squared_returns) > 1 else 0
            
            clustering_metrics = {
                'volatility_autocorrelations': autocorrelations,
                'arch_statistic': arch_stat,
                'volatility_persistence': autocorrelations.get('lag_1', 0)
            }
            
            return clustering_metrics
            
        except Exception as e:
            logger.error(f"Error in volatility clustering analysis: {str(e)}")
            return {}
    
    def calculate_volatility_metrics(self, symbol: str, 
                                   lookback_days: int = 252) -> Dict:
        """
        Calculate comprehensive volatility metrics for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with volatility metrics
        """
        try:
            if self.data is None:
                logger.error("No data provided to analyzer")
                return {}
            
            # Filter data for symbol
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date').tail(lookback_days)
            
            if len(symbol_data) < 30:
                logger.warning(f"Insufficient data for {symbol}")
                return {}
            
            # Calculate returns
            returns = self.calculate_returns(symbol_data['close'])
            
            # Basic volatility metrics
            current_vol = returns.std() * np.sqrt(252)  # Annualized
            
            # Rolling volatilities
            vol_30d = self.historical_volatility(returns, window=30)
            vol_60d = self.historical_volatility(returns, window=60)
            vol_252d = self.historical_volatility(returns, window=252)
            
            # Parkinson volatility (if high/low available)
            parkinson_vol = None
            if 'high' in symbol_data.columns and 'low' in symbol_data.columns:
                parkinson_vol = self.parkinson_volatility(
                    symbol_data['high'], 
                    symbol_data['low']
                )
            
            # Volatility percentiles
            vol_percentiles = {
                '25th': vol_30d.quantile(0.25) if not vol_30d.empty else None,
                '50th': vol_30d.median() if not vol_30d.empty else None,
                '75th': vol_30d.quantile(0.75) if not vol_30d.empty else None,
                '95th': vol_30d.quantile(0.95) if not vol_30d.empty else None
            }
            
            # Volatility clustering analysis
            clustering_metrics = self.volatility_clustering_analysis(returns)
            
            # Current volatility ranking
            current_vol_value = vol_30d.iloc[-1] if not vol_30d.empty else current_vol
            vol_rank = (vol_30d <= current_vol_value).mean() * 100 if not vol_30d.empty else 50
            
            # Volatility regime detection
            recent_vol = vol_30d.tail(10).mean() if not vol_30d.empty else current_vol
            long_term_vol = vol_252d.iloc[-1] if not vol_252d.empty else current_vol
            
            if recent_vol > long_term_vol * 1.2:
                vol_regime = "High Volatility"
            elif recent_vol < long_term_vol * 0.8:
                vol_regime = "Low Volatility"
            else:
                vol_regime = "Normal Volatility"
            
            metrics = {
                'symbol': symbol,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'current_volatility': current_vol,
                'volatility_30d': vol_30d.iloc[-1] if not vol_30d.empty else None,
                'volatility_60d': vol_60d.iloc[-1] if not vol_60d.empty else None,
                'volatility_252d': vol_252d.iloc[-1] if not vol_252d.empty else None,
                'parkinson_volatility': parkinson_vol.iloc[-1] if parkinson_vol is not None and not parkinson_vol.empty else None,
                'volatility_percentiles': vol_percentiles,
                'volatility_rank_percentile': vol_rank,
                'volatility_regime': vol_regime,
                'clustering_metrics': clustering_metrics,
                'data_points': len(symbol_data)
            }
            
            self.results[symbol] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics for {symbol}: {str(e)}")
            return {}
    
    def compare_volatilities(self, symbols: List[str]) -> pd.DataFrame:
        """
        Compare volatility metrics across multiple symbols
        
        Args:
            symbols: List of symbols to compare
            
        Returns:
            DataFrame with comparative volatility metrics
        """
        try:
            comparison_data = []
            
            for symbol in symbols:
                metrics = self.calculate_volatility_metrics(symbol)
                if metrics:
                    comparison_data.append({
                        'Symbol': symbol,
                        'Current_Vol': metrics.get('current_volatility', np.nan),
                        'Vol_30D': metrics.get('volatility_30d', np.nan),
                        'Vol_60D': metrics.get('volatility_60d', np.nan),
                        'Vol_252D': metrics.get('volatility_252d', np.nan),
                        'Vol_Rank': metrics.get('volatility_rank_percentile', np.nan),
                        'Vol_Regime': metrics.get('volatility_regime', 'Unknown'),
                        'ARCH_Stat': metrics.get('clustering_metrics', {}).get('arch_statistic', np.nan)
                    })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                
                # Add rankings
                df['Vol_Rank_Position'] = df['Current_Vol'].rank(ascending=False)
                
                return df.sort_values('Current_Vol', ascending=False)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error comparing volatilities: {str(e)}")
            return pd.DataFrame()
    
    def volatility_forecast(self, returns: pd.Series, method: str = 'ewma', 
                          forecast_days: int = 30) -> Dict:
        """
        Forecast future volatility using different methods
        
        Args:
            returns: Series of returns
            method: Forecasting method ('ewma', 'garch_simple')
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            if method == 'ewma':
                # Exponentially Weighted Moving Average
                lambda_param = 0.94  # RiskMetrics standard
                ewma_var = returns.ewm(alpha=1-lambda_param).var()
                current_vol = np.sqrt(ewma_var.iloc[-1] * 252)
                
                # Simple forecast (assumes mean reversion)
                forecast_vol = current_vol
                
            elif method == 'garch_simple':
                # Simplified GARCH(1,1) estimation
                # This is a basic implementation - in practice, use arch package
                squared_returns = returns**2
                long_run_var = squared_returns.mean()
                
                # Simple GARCH parameters (estimated)
                alpha = 0.1
                beta = 0.85
                omega = long_run_var * (1 - alpha - beta)
                
                # Current conditional variance
                current_var = omega + alpha * squared_returns.iloc[-1] + beta * long_run_var
                current_vol = np.sqrt(current_var * 252)
                
                # Mean-reverting forecast
                forecast_vol = np.sqrt(long_run_var * 252)
                
            else:
                raise ValueError("Method must be 'ewma' or 'garch_simple'")
            
            forecast_results = {
                'method': method,
                'current_volatility': current_vol,
                'forecast_volatility': forecast_vol,
                'forecast_horizon_days': forecast_days,
                'forecast_date': (datetime.now() + timedelta(days=forecast_days)).strftime('%Y-%m-%d')
            }
            
            return forecast_results
            
        except Exception as e:
            logger.error(f"Error in volatility forecasting: {str(e)}")
            return {}
    
    def generate_volatility_report(self, symbols: List[str]) -> Dict:
        """
        Generate comprehensive volatility report for multiple symbols
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary with complete volatility analysis
        """
        try:
            report = {
                'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbols_analyzed': len(symbols),
                'individual_analysis': {},
                'comparative_analysis': None,
                'market_volatility_summary': {}
            }
            
            # Individual symbol analysis
            for symbol in symbols:
                symbol_metrics = self.calculate_volatility_metrics(symbol)
                if symbol_metrics:
                    report['individual_analysis'][symbol] = symbol_metrics
            
            # Comparative analysis
            comparison_df = self.compare_volatilities(symbols)
            if not comparison_df.empty:
                report['comparative_analysis'] = comparison_df.to_dict('records')
            
            # Market summary
            if not comparison_df.empty:
                report['market_volatility_summary'] = {
                    'average_volatility': comparison_df['Current_Vol'].mean(),
                    'median_volatility': comparison_df['Current_Vol'].median(),
                    'highest_volatility': {
                        'symbol': comparison_df.loc[comparison_df['Current_Vol'].idxmax(), 'Symbol'],
                        'volatility': comparison_df['Current_Vol'].max()
                    },
                    'lowest_volatility': {
                        'symbol': comparison_df.loc[comparison_df['Current_Vol'].idxmin(), 'Symbol'],
                        'volatility': comparison_df['Current_Vol'].min()
                    },
                    'high_vol_regime_count': (comparison_df['Vol_Regime'] == 'High Volatility').sum(),
                    'normal_vol_regime_count': (comparison_df['Vol_Regime'] == 'Normal Volatility').sum(),
                    'low_vol_regime_count': (comparison_df['Vol_Regime'] == 'Low Volatility').sum()
                }
            
            logger.info(f"Volatility report generated for {len(symbols)} symbols")
            return report
            
        except Exception as e:
            logger.error(f"Error generating volatility report: {str(e)}")
            return {}

def main():
    """
    Example usage of VolatilityAnalyzer
    """
    # Sample Indian stock data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=300, freq='D'),
        'symbol': ['RELIANCE'] * 300,
        'close': 2500 + np.cumsum(np.random.randn(300) * 20),
        'high': None,
        'low': None,
        'volume': np.random.randint(1000000, 10000000, 300)
    })
    
    # Add high and low prices
    sample_data['high'] = sample_data['close'] * (1 + np.abs(np.random.randn(300) * 0.02))
    sample_data['low'] = sample_data['close'] * (1 - np.abs(np.random.randn(300) * 0.02))
    
    # Initialize analyzer
    analyzer = VolatilityAnalyzer(sample_data)
    
    # Calculate volatility metrics
    metrics = analyzer.calculate_volatility_metrics('RELIANCE')
    print("Volatility Metrics:", metrics)
    
    # Generate report
    report = analyzer.generate_volatility_report(['RELIANCE'])
    print("Volatility Report Generated Successfully")

if __name__ == "__main__":
    main()