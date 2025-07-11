"""
Performance Analysis Module for Indian Stock Market
Market Research System v1.0 (2022)

This module provides comprehensive performance analysis capabilities including:
- Return calculations and metrics
- Risk-adjusted performance measures
- Drawdown analysis
- Benchmark comparisons
- Performance attribution
- Portfolio performance evaluation

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

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for Indian stock market data
    """
    
    def __init__(self, data: pd.DataFrame = None, benchmark_data: pd.DataFrame = None):
        """
        Initialize PerformanceAnalyzer
        
        Args:
            data: DataFrame with columns ['date', 'symbol', 'close', 'volume']
            benchmark_data: DataFrame with benchmark index data (e.g., NIFTY 50)
        """
        self.data = data
        self.benchmark_data = benchmark_data
        self.results = {}
        self.risk_free_rate = 0.06  # Indian 10-year G-Sec rate (2022)
        
    def calculate_returns(self, prices: pd.Series, method: str = 'simple') -> pd.Series:
        """
        Calculate returns from price series
        
        Args:
            prices: Series of prices
            method: 'simple' or 'logarithmic'
            
        Returns:
            Series of returns
        """
        try:
            if method == 'simple':
                returns = prices.pct_change().dropna()
            elif method == 'logarithmic':
                returns = np.log(prices / prices.shift(1)).dropna()
            else:
                raise ValueError("Method must be 'simple' or 'logarithmic'")
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return pd.Series()
    
    def cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns
        
        Args:
            returns: Series of returns
            
        Returns:
            Series of cumulative returns
        """
        try:
            return (1 + returns).cumprod() - 1
        except Exception as e:
            logger.error(f"Error calculating cumulative returns: {str(e)}")
            return pd.Series()
    
    def annualized_return(self, returns: pd.Series) -> float:
        """
        Calculate annualized return
        
        Args:
            returns: Series of returns
            
        Returns:
            Annualized return
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            # Calculate total return
            total_return = (1 + returns).prod() - 1
            
            # Calculate number of years
            years = len(returns) / 252  # Assuming 252 trading days per year
            
            if years <= 0:
                return 0.0
            
            # Annualize
            annualized = (1 + total_return) ** (1 / years) - 1
            return annualized
            
        except Exception as e:
            logger.error(f"Error calculating annualized return: {str(e)}")
            return 0.0
    
    def volatility(self, returns: pd.Series, annualized: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)
        
        Args:
            returns: Series of returns
            annualized: Whether to annualize volatility
            
        Returns:
            Volatility
        """
        try:
            vol = returns.std()
            if annualized:
                vol = vol * np.sqrt(252)
            return vol
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
    
    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            # Convert annual risk-free rate to daily
            daily_rf = risk_free_rate / 252
            
            # Calculate excess returns
            excess_returns = returns - daily_rf
            
            # Sharpe ratio
            if excess_returns.std() == 0:
                return 0.0
            
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def sortino_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """
        Calculate Sortino ratio (downside deviation)
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sortino ratio
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            # Convert annual risk-free rate to daily
            daily_rf = risk_free_rate / 252
            
            # Calculate excess returns
            excess_returns = returns - daily_rf
            
            # Downside deviation (only negative excess returns)
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return np.inf  # No downside risk
            
            downside_deviation = downside_returns.std() * np.sqrt(252)
            
            if downside_deviation == 0:
                return 0.0
            
            sortino = (excess_returns.mean() * 252) / downside_deviation
            return sortino
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0
    
    def maximum_drawdown(self, returns: pd.Series) -> Dict:
        """
        Calculate maximum drawdown and related metrics
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with drawdown metrics
        """
        try:
            # Calculate cumulative returns
            cum_returns = self.cumulative_returns(returns)
            
            # Calculate running maximum
            running_max = cum_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cum_returns - running_max) / (1 + running_max)
            
            # Maximum drawdown
            max_dd = drawdown.min()
            
            # Find the dates
            max_dd_date = drawdown.idxmin()
            peak_date = running_max.loc[:max_dd_date].idxmax()
            
            # Recovery date (when drawdown becomes 0 again after max drawdown)
            recovery_date = None
            recovery_mask = (drawdown.loc[max_dd_date:] >= -0.01)  # Within 1% of recovery
            if recovery_mask.any():
                recovery_date = drawdown.loc[max_dd_date:][recovery_mask].index[0]
            
            # Drawdown duration
            if recovery_date:
                drawdown_duration = (recovery_date - peak_date).days
            else:
                drawdown_duration = (drawdown.index[-1] - peak_date).days
            
            return {
                'max_drawdown': max_dd,
                'max_drawdown_date': max_dd_date,
                'peak_date': peak_date,
                'recovery_date': recovery_date,
                'drawdown_duration_days': drawdown_duration,
                'current_drawdown': drawdown.iloc[-1],
                'drawdown_series': drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {str(e)}")
            return {}
    
    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown)
        
        Args:
            returns: Series of returns
            
        Returns:
            Calmar ratio
        """
        try:
            ann_return = self.annualized_return(returns)
            max_dd_info = self.maximum_drawdown(returns)
            max_dd = abs(max_dd_info.get('max_drawdown', 0))
            
            if max_dd == 0:
                return np.inf
            
            return ann_return / max_dd
            
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return 0.0
    
    def beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta (systematic risk)
        
        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns
            
        Returns:
            Beta coefficient
        """
        try:
            # Align the series
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if aligned_data.empty or len(aligned_data.columns) < 2:
                return 1.0  # Default beta
            
            stock_aligned = aligned_data.iloc[:, 0]
            market_aligned = aligned_data.iloc[:, 1]
            
            # Calculate covariance and variance
            covariance = np.cov(stock_aligned, market_aligned)[0, 1]
            market_variance = np.var(market_aligned)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 1.0
    
    def alpha(self, stock_returns: pd.Series, market_returns: pd.Series, 
              risk_free_rate: float = None) -> float:
        """
        Calculate alpha (excess return over CAPM)
        
        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Alpha
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            # Calculate beta
            beta_coeff = self.beta(stock_returns, market_returns)
            
            # Calculate returns
            stock_return = self.annualized_return(stock_returns)
            market_return = self.annualized_return(market_returns)
            
            # Alpha = Stock Return - (Risk Free Rate + Beta * (Market Return - Risk Free Rate))
            alpha = stock_return - (risk_free_rate + beta_coeff * (market_return - risk_free_rate))
            
            return alpha
            
        except Exception as e:
            logger.error(f"Error calculating alpha: {str(e)}")
            return 0.0
    
    def information_ratio(self, stock_returns: pd.Series, 
                         benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio
        
        Args:
            stock_returns: Series of stock returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Information ratio
        """
        try:
            # Align the series
            aligned_data = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
            if aligned_data.empty or len(aligned_data.columns) < 2:
                return 0.0
            
            stock_aligned = aligned_data.iloc[:, 0]
            benchmark_aligned = aligned_data.iloc[:, 1]
            
            # Calculate active returns (excess over benchmark)
            active_returns = stock_aligned - benchmark_aligned
            
            # Information ratio
            if active_returns.std() == 0:
                return 0.0
            
            ir = active_returns.mean() / active_returns.std() * np.sqrt(252)
            return ir
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {str(e)}")
            return 0.0
    
    def treynor_ratio(self, stock_returns: pd.Series, market_returns: pd.Series, 
                      risk_free_rate: float = None) -> float:
        """
        Calculate Treynor ratio (excess return per unit of systematic risk)
        
        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Treynor ratio
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            # Calculate beta
            beta_coeff = self.beta(stock_returns, market_returns)
            
            if beta_coeff == 0:
                return 0.0
            
            # Calculate annualized return
            ann_return = self.annualized_return(stock_returns)
            
            # Treynor ratio
            treynor = (ann_return - risk_free_rate) / beta_coeff
            return treynor
            
        except Exception as e:
            logger.error(f"Error calculating Treynor ratio: {str(e)}")
            return 0.0
    
    def var_calculation(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR) using historical method
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default 5% for 95% VaR)
            
        Returns:
            VaR value
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            # Calculate VaR using percentile method
            var = np.percentile(returns, confidence_level * 100)
            return var
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def conditional_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default 5% for 95% CVaR)
            
        Returns:
            CVaR value
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            # Calculate VaR first
            var = self.var_calculation(returns, confidence_level)
            
            # Calculate CVaR (mean of returns below VaR)
            tail_returns = returns[returns <= var]
            
            if len(tail_returns) == 0:
                return var
            
            cvar = tail_returns.mean()
            return cvar
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            return 0.0
    
    def rolling_performance(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Args:
            returns: Series of returns
            window: Rolling window size (default 252 for 1 year)
            
        Returns:
            DataFrame with rolling metrics
        """
        try:
            rolling_metrics = pd.DataFrame(index=returns.index)
            
            # Rolling returns
            rolling_metrics['rolling_return'] = returns.rolling(window).apply(
                lambda x: (1 + x).prod() - 1, raw=False
            )
            
            # Rolling volatility
            rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
            
            # Rolling Sharpe ratio
            rolling_metrics['rolling_sharpe'] = returns.rolling(window).apply(
                lambda x: self.sharpe_ratio(x), raw=False
            )
            
            # Rolling maximum drawdown
            rolling_metrics['rolling_max_dd'] = returns.rolling(window).apply(
                lambda x: self.maximum_drawdown(x).get('max_drawdown', 0), raw=False
            )
            
            return rolling_metrics.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating rolling performance: {str(e)}")
            return pd.DataFrame()
    
    def performance_summary(self, symbol: str, returns: pd.Series, 
                          benchmark_returns: pd.Series = None) -> Dict:
        """
        Generate comprehensive performance summary
        
        Args:
            symbol: Stock symbol
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            summary = {
                'symbol': symbol,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'period_start': returns.index[0].strftime('%Y-%m-%d') if len(returns) > 0 else None,
                'period_end': returns.index[-1].strftime('%Y-%m-%d') if len(returns) > 0 else None,
                'total_observations': len(returns),
            }
            
            # Return metrics
            summary['total_return'] = (1 + returns).prod() - 1
            summary['annualized_return'] = self.annualized_return(returns)
            summary['volatility'] = self.volatility(returns)
            
            # Risk-adjusted metrics
            summary['sharpe_ratio'] = self.sharpe_ratio(returns)
            summary['sortino_ratio'] = self.sortino_ratio(returns)
            summary['calmar_ratio'] = self.calmar_ratio(returns)
            
            # Drawdown analysis
            dd_info = self.maximum_drawdown(returns)
            summary['max_drawdown'] = dd_info.get('max_drawdown', 0)
            summary['max_drawdown_date'] = dd_info.get('max_drawdown_date')
            summary['current_drawdown'] = dd_info.get('current_drawdown', 0)
            summary['drawdown_duration_days'] = dd_info.get('drawdown_duration_days', 0)
            
            # Risk metrics
            summary['var_95'] = self.var_calculation(returns, 0.05)
            summary['cvar_95'] = self.conditional_var(returns, 0.05)
            
            # Benchmark comparison (if provided)
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                summary['beta'] = self.beta(returns, benchmark_returns)
                summary['alpha'] = self.alpha(returns, benchmark_returns)
                summary['information_ratio'] = self.information_ratio(returns, benchmark_returns)
                summary['treynor_ratio'] = self.treynor_ratio(returns, benchmark_returns)
                
                # Relative performance
                benchmark_total_return = (1 + benchmark_returns).prod() - 1
                summary['excess_return'] = summary['total_return'] - benchmark_total_return
                summary['benchmark_total_return'] = benchmark_total_return
                summary['benchmark_annualized_return'] = self.annualized_return(benchmark_returns)
            
            # Additional statistics
            summary['skewness'] = returns.skew()
            summary['kurtosis'] = returns.kurtosis()
            summary['positive_periods'] = (returns > 0).sum()
            summary['negative_periods'] = (returns < 0).sum()
            summary['win_rate'] = summary['positive_periods'] / len(returns) if len(returns) > 0 else 0
            
            # Best and worst periods
            summary['best_day'] = returns.max()
            summary['worst_day'] = returns.min()
            summary['best_day_date'] = returns.idxmax()
            summary['worst_day_date'] = returns.idxmin()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {str(e)}")
            return {}
    
    def analyze_stock_performance(self, symbol: str, start_date: str = None, 
                                end_date: str = None) -> Dict:
        """
        Analyze performance for a specific stock
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if self.data is None or self.data.empty:
                logger.error("No data available for analysis")
                return {}
            
            # Filter data for the symbol
            stock_data = self.data[self.data['symbol'] == symbol].copy()
            
            if stock_data.empty:
                logger.error(f"No data found for symbol: {symbol}")
                return {}
            
            # Filter by date range
            if start_date:
                stock_data = stock_data[stock_data['date'] >= start_date]
            if end_date:
                stock_data = stock_data[stock_data['date'] <= end_date]
            
            if stock_data.empty:
                logger.error(f"No data found for symbol {symbol} in date range")
                return {}
            
            # Sort by date
            stock_data = stock_data.sort_values('date')
            
            # Calculate returns
            stock_data['returns'] = stock_data['close'].pct_change()
            returns = stock_data['returns'].dropna()
            
            # Get benchmark returns if available
            benchmark_returns = None
            if self.benchmark_data is not None and not self.benchmark_data.empty:
                benchmark_data = self.benchmark_data.copy()
                if start_date:
                    benchmark_data = benchmark_data[benchmark_data['date'] >= start_date]
                if end_date:
                    benchmark_data = benchmark_data[benchmark_data['date'] <= end_date]
                
                benchmark_data = benchmark_data.sort_values('date')
                benchmark_data['returns'] = benchmark_data['close'].pct_change()
                benchmark_returns = benchmark_data['returns'].dropna()
            
            # Generate performance summary
            summary = self.performance_summary(symbol, returns, benchmark_returns)
            
            # Add rolling performance
            rolling_perf = self.rolling_performance(returns)
            summary['rolling_performance'] = rolling_perf
            
            # Store results
            self.results[symbol] = summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Error analyzing stock performance: {str(e)}")
            return {}
    
    def compare_stocks(self, symbols: List[str], start_date: str = None, 
                      end_date: str = None) -> pd.DataFrame:
        """
        Compare performance across multiple stocks
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with comparison metrics
        """
        try:
            comparison_data = []
            
            for symbol in symbols:
                perf = self.analyze_stock_performance(symbol, start_date, end_date)
                if perf:
                    comparison_data.append({
                        'symbol': symbol,
                        'total_return': perf.get('total_return', 0),
                        'annualized_return': perf.get('annualized_return', 0),
                        'volatility': perf.get('volatility', 0),
                        'sharpe_ratio': perf.get('sharpe_ratio', 0),
                        'max_drawdown': perf.get('max_drawdown', 0),
                        'beta': perf.get('beta', 1.0),
                        'alpha': perf.get('alpha', 0),
                        'win_rate': perf.get('win_rate', 0),
                        'var_95': perf.get('var_95', 0),
                        'excess_return': perf.get('excess_return', 0)
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            if not comparison_df.empty:
                # Add rankings
                comparison_df['return_rank'] = comparison_df['total_return'].rank(ascending=False)
                comparison_df['sharpe_rank'] = comparison_df['sharpe_ratio'].rank(ascending=False)
                comparison_df['risk_rank'] = comparison_df['volatility'].rank(ascending=True)
                
                # Overall score (simple average of ranks)
                comparison_df['overall_score'] = (
                    comparison_df['return_rank'] + 
                    comparison_df['sharpe_rank'] + 
                    comparison_df['risk_rank']
                ) / 3
                comparison_df['overall_rank'] = comparison_df['overall_score'].rank(ascending=True)
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing stocks: {str(e)}")
            return pd.DataFrame()
    
    def generate_performance_report(self, symbol: str, output_path: str = None) -> str:
        """
        Generate a formatted performance report
        
        Args:
            symbol: Stock symbol
            output_path: Path to save the report
            
        Returns:
            Report as string
        """
        try:
            if symbol not in self.results:
                self.analyze_stock_performance(symbol)
            
            if symbol not in self.results:
                return "No analysis results available"
            
            perf = self.results[symbol]
            
            report = f"""
PERFORMANCE ANALYSIS REPORT
=========================
Symbol: {perf.get('symbol', 'N/A')}
Analysis Date: {perf.get('analysis_date', 'N/A')}
Period: {perf.get('period_start', 'N/A')} to {perf.get('period_end', 'N/A')}
Total Observations: {perf.get('total_observations', 0)}

RETURN METRICS
--------------
Total Return: {perf.get('total_return', 0):.2%}
Annualized Return: {perf.get('annualized_return', 0):.2%}
Volatility (Annual): {perf.get('volatility', 0):.2%}

RISK-ADJUSTED METRICS
--------------------
Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}
Sortino Ratio: {perf.get('sortino_ratio', 0):.3f}
Calmar Ratio: {perf.get('calmar_ratio', 0):.3f}

DRAWDOWN ANALYSIS
----------------
Maximum Drawdown: {perf.get('max_drawdown', 0):.2%}
Current Drawdown: {perf.get('current_drawdown', 0):.2%}
Max Drawdown Date: {perf.get('max_drawdown_date', 'N/A')}
Drawdown Duration: {perf.get('drawdown_duration_days', 0)} days

RISK METRICS
-----------
VaR (95%): {perf.get('var_95', 0):.2%}
CVaR (95%): {perf.get('cvar_95', 0):.2%}
Skewness: {perf.get('skewness', 0):.3f}
Kurtosis: {perf.get('kurtosis', 0):.3f}

TRADING STATISTICS
-----------------
Win Rate: {perf.get('win_rate', 0):.2%}
Best Day: {perf.get('best_day', 0):.2%} ({perf.get('best_day_date', 'N/A')})
Worst Day: {perf.get('worst_day', 0):.2%} ({perf.get('worst_day_date', 'N/A')})

"""
            
            # Add benchmark comparison if available
            if perf.get('beta') is not None:
                report += f"""
BENCHMARK COMPARISON
-------------------
Beta: {perf.get('beta', 0):.3f}
Alpha: {perf.get('alpha', 0):.2%}
Information Ratio: {perf.get('information_ratio', 0):.3f}
Treynor Ratio: {perf.get('treynor_ratio', 0):.3f}
Excess Return: {perf.get('excess_return', 0):.2%}
Benchmark Return: {perf.get('benchmark_total_return', 0):.2%}
"""
            
            report += f"""
ANALYSIS NOTES
--------------
- Risk-free rate used: {self.risk_free_rate:.2%}
- Analysis based on daily returns
- All returns are geometric returns
- Volatility is annualized using 252 trading days

Report generated by Market Research System v1.0 (2022)
"""
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to: {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return "Error generating report"


# Example usage and testing functions
def main():
    """
    Example usage of PerformanceAnalyzer
    """
    # Sample data creation for testing
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    
    # Sample stock data
    sample_data = []
    symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
    
    for symbol in symbols:
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        volumes = np.random.randint(1000000, 10000000, len(dates))
        
        for i, date in enumerate(dates):
            sample_data.append({
                'date': date,
                'symbol': symbol,
                'close': prices[i],
                'volume': volumes[i]
            })
    
    df = pd.DataFrame(sample_data)
    
    # Sample benchmark data (NIFTY 50)
    benchmark_prices = 100 * np.cumprod(1 + np.random.normal(0.0008, 0.015, len(dates)))
    benchmark_data = pd.DataFrame({
        'date': dates,
        'close': benchmark_prices
    })
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(data=df, benchmark_data=benchmark_data)
    
    # Analyze single stock
    result = analyzer.analyze_stock_performance('RELIANCE.NS')
    print("Analysis completed for RELIANCE.NS")
    print(f"Total Return: {result.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
    
    # Compare multiple stocks
    comparison = analyzer.compare_stocks(['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'])
    print("\nStock Comparison:")
    print(comparison[['symbol', 'total_return', 'sharpe_ratio', 'volatility']].to_string())
    
    # Generate report
    report = analyzer.generate_performance_report('RELIANCE.NS')
    print(f"\nSample Report Preview:\n{report[:500]}...")


if __name__ == "__main__":
    main()