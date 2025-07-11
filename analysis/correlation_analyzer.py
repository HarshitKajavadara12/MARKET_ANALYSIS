"""
Market Research System v1.0 - Correlation Analysis Module
Author: Independent Market Researcher
Created: 2022
Updated: 2022-12-31

This module handles correlation analysis for stocks and market indices.
Focuses on Indian stock market data with NSE/BSE listings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    """
    Comprehensive correlation analysis for market data.
    Handles price correlations, volume correlations, and sector analysis.
    """
    
    def __init__(self):
        self.correlation_data = {}
        self.sector_correlations = {}
        
    def calculate_price_correlation(self, data: pd.DataFrame, 
                                  method: str = 'pearson',
                                  min_periods: int = 20) -> pd.DataFrame:
        """
        Calculate price correlation matrix for multiple stocks.
        
        Args:
            data: DataFrame with stock prices (columns as stocks)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_periods: Minimum periods required for correlation
            
        Returns:
            Correlation matrix DataFrame
        """
        try:
            # Calculate returns for correlation
            returns = data.pct_change().dropna()
            
            # Calculate correlation matrix
            corr_matrix = returns.corr(method=method, min_periods=min_periods)
            
            # Store for later use
            self.correlation_data['price_correlation'] = corr_matrix
            
            return corr_matrix
            
        except Exception as e:
            print(f"Error calculating price correlation: {e}")
            return pd.DataFrame()
    
    def calculate_rolling_correlation(self, stock1: pd.Series, stock2: pd.Series,
                                   window: int = 30) -> pd.Series:
        """
        Calculate rolling correlation between two stocks.
        
        Args:
            stock1: First stock price series
            stock2: Second stock price series
            window: Rolling window size
            
        Returns:
            Rolling correlation series
        """
        try:
            # Calculate returns
            returns1 = stock1.pct_change()
            returns2 = stock2.pct_change()
            
            # Calculate rolling correlation
            rolling_corr = returns1.rolling(window=window).corr(returns2)
            
            return rolling_corr.dropna()
            
        except Exception as e:
            print(f"Error calculating rolling correlation: {e}")
            return pd.Series()
    
    def sector_correlation_analysis(self, data: pd.DataFrame, 
                                  sector_mapping: Dict[str, List[str]]) -> Dict:
        """
        Analyze correlations within and between sectors.
        
        Args:
            data: DataFrame with stock prices
            sector_mapping: Dict mapping sector names to stock symbols
            
        Returns:
            Dictionary with sector correlation analysis
        """
        try:
            results = {}
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Intra-sector correlations
            for sector, stocks in sector_mapping.items():
                available_stocks = [s for s in stocks if s in returns.columns]
                
                if len(available_stocks) > 1:
                    sector_data = returns[available_stocks]
                    sector_corr = sector_data.corr()
                    
                    # Calculate average intra-sector correlation
                    mask = np.triu(np.ones_like(sector_corr, dtype=bool), k=1)
                    avg_corr = sector_corr.where(mask).stack().mean()
                    
                    results[f'{sector}_intra'] = {
                        'correlation_matrix': sector_corr,
                        'average_correlation': avg_corr,
                        'stocks_analyzed': available_stocks
                    }
            
            # Inter-sector correlations
            sector_returns = {}
            for sector, stocks in sector_mapping.items():
                available_stocks = [s for s in stocks if s in returns.columns]
                if available_stocks:
                    # Calculate sector average return
                    sector_returns[sector] = returns[available_stocks].mean(axis=1)
            
            if len(sector_returns) > 1:
                sector_df = pd.DataFrame(sector_returns)
                inter_sector_corr = sector_df.corr()
                results['inter_sector'] = inter_sector_corr
            
            self.sector_correlations = results
            return results
            
        except Exception as e:
            print(f"Error in sector correlation analysis: {e}")
            return {}
    
    def correlation_with_index(self, stock_data: pd.DataFrame, 
                             index_data: pd.Series,
                             index_name: str = 'NIFTY50') -> pd.DataFrame:
        """
        Calculate correlation of individual stocks with market index.
        
        Args:
            stock_data: DataFrame with individual stock prices
            index_data: Series with index prices
            index_name: Name of the index
            
        Returns:
            DataFrame with correlation coefficients and statistics
        """
        try:
            # Calculate returns
            stock_returns = stock_data.pct_change().dropna()
            index_returns = index_data.pct_change().dropna()
            
            # Align data
            common_dates = stock_returns.index.intersection(index_returns.index)
            stock_returns = stock_returns.loc[common_dates]
            index_returns = index_returns.loc[common_dates]
            
            results = []
            
            for stock in stock_returns.columns:
                stock_ret = stock_returns[stock].dropna()
                
                # Align stock and index returns
                aligned_dates = stock_ret.index.intersection(index_returns.index)
                if len(aligned_dates) < 20:  # Minimum data points
                    continue
                    
                stock_aligned = stock_ret.loc[aligned_dates]
                index_aligned = index_returns.loc[aligned_dates]
                
                # Calculate correlation
                correlation = stock_aligned.corr(index_aligned)
                
                # Calculate statistical significance
                t_stat, p_value = stats.pearsonr(stock_aligned, index_aligned)
                
                # Calculate beta
                covariance = np.cov(stock_aligned, index_aligned)[0, 1]
                index_variance = np.var(index_aligned)
                beta = covariance / index_variance if index_variance != 0 else 0
                
                results.append({
                    'Stock': stock,
                    'Correlation': correlation,
                    'P_Value': p_value,
                    'Beta': beta,
                    'Observations': len(aligned_dates),
                    'Significant': p_value < 0.05
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Error calculating index correlation: {e}")
            return pd.DataFrame()
    
    def correlation_stability_test(self, data: pd.DataFrame, 
                                 stock1: str, stock2: str,
                                 periods: List[int] = [30, 60, 120, 252]) -> Dict:
        """
        Test correlation stability across different time periods.
        
        Args:
            data: DataFrame with stock prices
            stock1: First stock symbol
            stock2: Second stock symbol
            periods: List of periods to test
            
        Returns:
            Dictionary with stability analysis
        """
        try:
            if stock1 not in data.columns or stock2 not in data.columns:
                return {}
            
            returns1 = data[stock1].pct_change().dropna()
            returns2 = data[stock2].pct_change().dropna()
            
            # Align data
            common_dates = returns1.index.intersection(returns2.index)
            returns1 = returns1.loc[common_dates]
            returns2 = returns2.loc[common_dates]
            
            results = {
                'stock_pair': f"{stock1}-{stock2}",
                'period_analysis': {},
                'correlation_trend': []
            }
            
            for period in periods:
                if len(returns1) < period:
                    continue
                    
                # Calculate correlation for different periods
                recent_data1 = returns1.tail(period)
                recent_data2 = returns2.tail(period)
                
                correlation = recent_data1.corr(recent_data2)
                
                results['period_analysis'][f'{period}_days'] = {
                    'correlation': correlation,
                    'start_date': recent_data1.index[0].strftime('%Y-%m-%d'),
                    'end_date': recent_data1.index[-1].strftime('%Y-%m-%d')
                }
            
            # Calculate rolling correlation trend
            rolling_corr = returns1.rolling(window=60).corr(returns2).dropna()
            
            # Analyze trend
            if len(rolling_corr) > 1:
                trend_slope = np.polyfit(range(len(rolling_corr)), rolling_corr, 1)[0]
                results['correlation_trend'] = {
                    'slope': trend_slope,
                    'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'stability': 'stable' if abs(trend_slope) < 0.001 else 'unstable'
                }
            
            return results
            
        except Exception as e:
            print(f"Error in correlation stability test: {e}")
            return {}
    
    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame, 
                              title: str = "Correlation Matrix",
                              figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            correlation_matrix: Correlation matrix to plot
            title: Plot title
            figsize: Figure size
        """
        try:
            plt.figure(figsize=figsize)
            
            # Create heatmap
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm',
                       center=0,
                       square=True,
                       linewidths=0.5,
                       fmt='.2f',
                       cbar_kws={"shrink": .8})
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting correlation matrix: {e}")
    
    def plot_rolling_correlation(self, stock1_data: pd.Series, stock2_data: pd.Series,
                               stock1_name: str, stock2_name: str,
                               window: int = 60, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot rolling correlation between two stocks.
        
        Args:
            stock1_data: First stock price series
            stock2_data: Second stock price series
            stock1_name: First stock name
            stock2_name: Second stock name
            window: Rolling window size
            figsize: Figure size
        """
        try:
            rolling_corr = self.calculate_rolling_correlation(
                stock1_data, stock2_data, window
            )
            
            plt.figure(figsize=figsize)
            
            plt.plot(rolling_corr.index, rolling_corr, 
                    linewidth=2, label=f'{window}-day Rolling Correlation')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High Correlation')
            plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='High Negative Correlation')
            
            plt.title(f'Rolling Correlation: {stock1_name} vs {stock2_name}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Correlation Coefficient')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting rolling correlation: {e}")
    
    def generate_correlation_report(self, data: pd.DataFrame,
                                  sector_mapping: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive correlation analysis report.
        
        Args:
            data: DataFrame with stock prices
            sector_mapping: Optional sector mapping for analysis
            
        Returns:
            Dictionary with complete correlation analysis
        """
        try:
            report = {
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'data_period': {
                    'start_date': data.index[0].strftime('%Y-%m-%d'),
                    'end_date': data.index[-1].strftime('%Y-%m-%d'),
                    'total_days': len(data)
                },
                'stocks_analyzed': list(data.columns),
                'analysis_results': {}
            }
            
            # Overall correlation matrix
            corr_matrix = self.calculate_price_correlation(data)
            
            # Key statistics
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix.where(mask).stack()
            
            report['analysis_results']['overall_statistics'] = {
                'average_correlation': correlations.mean(),
                'median_correlation': correlations.median(),
                'max_correlation': correlations.max(),
                'min_correlation': correlations.min(),
                'std_correlation': correlations.std()
            }
            
            # Highest correlations
            high_corr = correlations.nlargest(10)
            report['analysis_results']['highest_correlations'] = {
                f"{pair[0]}-{pair[1]}": corr for pair, corr in high_corr.items()
            }
            
            # Lowest correlations
            low_corr = correlations.nsmallest(10)
            report['analysis_results']['lowest_correlations'] = {
                f"{pair[0]}-{pair[1]}": corr for pair, corr in low_corr.items()
            }
            
            # Sector analysis if provided
            if sector_mapping:
                sector_results = self.sector_correlation_analysis(data, sector_mapping)
                report['analysis_results']['sector_analysis'] = sector_results
            
            return report
            
        except Exception as e:
            print(f"Error generating correlation report: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
    
    # Simulate Indian stock data
    indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'ICICIBANK.NS']
    
    # Create sample data with some correlation
    base_trend = np.cumsum(np.random.randn(len(dates)) * 0.02)
    stock_data = {}
    
    for i, stock in enumerate(indian_stocks):
        # Add some correlation with base trend and individual noise
        individual_trend = base_trend * (0.5 + 0.3 * np.random.rand())
        noise = np.cumsum(np.random.randn(len(dates)) * 0.01)
        stock_data[stock] = 100 + individual_trend + noise
    
    df = pd.DataFrame(stock_data, index=dates)
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer()
    
    # Test correlation analysis
    print("Testing Correlation Analyzer...")
    
    # Calculate correlation matrix
    corr_matrix = analyzer.calculate_price_correlation(df)
    print("Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Test sector mapping
    sector_mapping = {
        'IT': ['TCS.NS', 'INFY.NS'],
        'BANKING': ['HDFC.NS', 'ICICIBANK.NS'],
        'ENERGY': ['RELIANCE.NS']
    }
    
    # Generate comprehensive report
    report = analyzer.generate_correlation_report(df, sector_mapping)
    print("\nCorrelation Analysis Report:")
    print(f"Analysis Date: {report['analysis_date']}")
    print(f"Average Correlation: {report['analysis_results']['overall_statistics']['average_correlation']:.3f}")
    print(f"Data Period: {report['data_period']['start_date']} to {report['data_period']['end_date']}")
    
    print("\nCorrelation Analyzer Module - Ready for Production!")