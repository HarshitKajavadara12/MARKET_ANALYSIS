"""
Market Analyzer Module
Market Research System v1.0

Provides basic statistical analysis for Indian stock market data.
Includes market breadth, sector analysis, and statistical measures.

Created: February 2022
Last Updated: April 2022
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import stats
from datetime import datetime, timedelta

class MarketAnalyzer:
    """
    Comprehensive market analysis for Indian stock market.
    Provides statistical analysis, market breadth indicators, and sector analysis.
    """
    
    def __init__(self):
        """Initialize Market Analyzer with Indian market specific configurations."""
        self.trading_days_per_year = 250
        self.risk_free_rate = 0.06  # Indian 10-year G-Sec rate as of 2022
        
        # Indian market sectors (NSE classification)
        self.indian_sectors = {
            'NIFTY_AUTO': ['MARUTI', 'M&M', 'TATAMOTORS', 'BAJAJ-AUTO', 'EICHERMOT'],
            'NIFTY_BANK': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'NIFTY_IT': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'],
            'NIFTY_PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON'],
            'NIFTY_FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'GODREJCP'],
            'NIFTY_METAL': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'NMDC'],
            'NIFTY_REALTY': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'SOBHA'],
            'NIFTY_ENERGY': ['RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'IOC']
        }
    
    def basic_statistics(self, data: pd.Series) -> Dict:
        """
        Calculate basic statistical measures for price series
        
        Args:
            data: Price series (typically close prices)
            
        Returns:
            Dict: Dictionary containing statistical measures
        """
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std_dev': data.std(),
            'variance': data.var(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'skewness': stats.skew(data.dropna()),
            'kurtosis': stats.kurtosis(data.dropna()),
            'count': len(data.dropna()),
            'quantile_25': data.quantile(0.25),
            'quantile_75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25)
        }
    
    def returns_analysis(self, prices: pd.Series) -> Dict:
        """
        Analyze returns characteristics
        
        Args:
            prices: Price series
            
        Returns:
            Dict: Returns analysis results
        """
        returns = prices.pct_change().dropna()
        
        # Annualized metrics
        annual_return = returns.mean() * self.trading_days_per_year
        annual_volatility = returns.std() * np.sqrt(self.trading_days_per_year)
        
        # Risk metrics
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Downside metrics
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(self.trading_days_per_year)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'daily_return_mean': returns.mean(),
            'daily_return_std': returns.std(),
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'positive_days': len(returns[returns > 0]),
            'negative_days': len(returns[returns < 0]),
            'win_rate': len(returns[returns > 0]) / len(returns),
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'var_95': returns.quantile(0.05),  # 95% VaR
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean()  # Conditional VaR
        }
    
    def market_breadth_analysis(self, stock_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze market breadth indicators
        
        Args:
            stock_data: Dictionary with stock symbols as keys and OHLCV data as values
            
        Returns:
            Dict: Market breadth indicators
        """
        if not stock_data:
            return {}
        
        # Get common date range
        dates = None
        for symbol, data in stock_data.items():
            if dates is None:
                dates = data.index
            else:
                dates = dates.intersection(data.index)
        
        if len(dates) == 0:
            return {}
        
        # Calculate daily returns for all stocks
        returns_data = {}
        for symbol, data in stock_data.items():
            returns_data[symbol] = data.loc[dates, 'Close'].pct_change().dropna()
        
        returns_df = pd.DataFrame(returns_data)
        
        # Market breadth indicators
        breadth_indicators = {}
        
        for date in returns_df.index:
            day_returns = returns_df.loc[date]
            
            advancing = len(day_returns[day_returns > 0])
            declining = len(day_returns[day_returns < 0])
            unchanged = len(day_returns[day_returns == 0])
            
            breadth_indicators[date] = {
                'advancing': advancing,
                'declining': declining,
                'unchanged': unchanged,
                'advance_decline_ratio': advancing / declining if declining > 0 else np.inf,
                'advance_decline_line': advancing - declining,
                'total_stocks': len(day_returns)
            }
        
        breadth_df = pd.DataFrame(breadth_indicators).T
        
        return {
            'breadth_data': breadth_df,
            'avg_advance_decline_ratio': breadth_df['advance_decline_ratio'].mean(),
            'cumulative_advance_decline': breadth_df['advance_decline_line'].cumsum(),
            'bullish_days': len(breadth_df[breadth_df['advancing'] > breadth_df['declining']]),
            'bearish_days': len(breadth_df[breadth_df['declining'] > breadth_df['advancing']]),
            'neutral_days': len(breadth_df[breadth_df['advancing'] == breadth_df['declining']])
        }
    
    def sector_analysis(self, sector_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze sector performance and characteristics
        
        Args:
            sector_data: Dictionary with sector names as keys and price data as values
            
        Returns:
            Dict: Sector analysis results
        """
        sector_analysis = {}
        
        for sector, data in sector_data.items():
            if 'Close' not in data.columns:
                continue
                
            prices = data['Close']
            sector_stats = self.returns_analysis(prices)
            basic_stats = self.basic_statistics(prices)
            
            # Sector specific metrics
            sector_analysis[sector] = {
                'returns_analysis': sector_stats,
                'basic_statistics': basic_stats,
                'current_price': prices.iloc[-1] if len(prices) > 0 else None,
                'period_return': (prices.iloc[-1] / prices.iloc[0] - 1) if len(prices) > 1 else None,
                'price_range_52w': {
                    'high': prices.tail(252).max() if len(prices) >= 252 else prices.max(),
                    'low': prices.tail(252).min() if len(prices) >= 252 else prices.min()
                }
            }
        
        # Comparative analysis
        if len(sector_analysis) > 1:
            comparative_metrics = self._comparative_sector_analysis(sector_analysis)
            sector_analysis['comparative_analysis'] = comparative_metrics
        
        return sector_analysis
    
    def _comparative_sector_analysis(self, sector_data: Dict) -> Dict:
        """
        Perform comparative analysis across sectors
        
        Args:
            sector_data: Sector analysis results
            
        Returns:
            Dict: Comparative analysis results
        """
        metrics = {}
        
        # Extract key metrics for comparison
        for metric in ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']:
            values = {}
            for sector, data in sector_data.items():
                if sector != 'comparative_analysis' and 'returns_analysis' in data:
                    values[sector] = data['returns_analysis'].get(metric, 0)
            
            if values:
                metrics[metric] = {
                    'best_sector': max(values, key=values.get) if metric != 'max_drawdown' else min(values, key=values.get),
                    'worst_sector': min(values, key=values.get) if metric != 'max_drawdown' else max(values, key=values.get),
                    'average': np.mean(list(values.values())),
                    'ranking': dict(sorted(values.items(), key=lambda x: x[1], reverse=(metric != 'max_drawdown')))
                }
        
        return metrics
    
    def correlation_analysis(self, data: Dict[str, pd.Series]) -> Dict:
        """
        Perform correlation analysis between different assets/sectors
        
        Args:
            data: Dictionary with asset names as keys and price series as values
            
        Returns:
            Dict: Correlation analysis results
        """
        if len(data) < 2:
            return {}
        
        # Calculate returns for correlation
        returns_data = {}
        for name, prices in data.items():
            returns_data[name] = prices.pct_change().dropna()
        
        returns_df = pd.DataFrame(returns_data)
        
        # Remove columns with insufficient data
        returns_df = returns_df.dropna(axis=1, thresh=30)  # At least 30 observations
        
        if returns_df.empty or len(returns_df.columns) < 2:
            return {}
        
        correlation_matrix = returns_df.corr()
        
        # Find highly correlated pairs
        high_correlations = []
        low_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])
                
                if corr_value > 0.7:
                    high_correlations.append((pair, corr_value))
                elif corr_value < -0.3:
                    low_correlations.append((pair, corr_value))
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_correlations,
            'low_correlations': low_correlations,
            'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
            'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()
        }
    
    def volatility_analysis(self, prices: pd.Series, window: int = 20) -> Dict:
        """
        Analyze volatility patterns and regimes
        
        Args:
            prices: Price series
            window: Rolling window for volatility calculation
            
        Returns:
            Dict: Volatility analysis results
        """
        returns = prices.pct_change().dropna()
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(self.trading_days_per_year)
        
        # GARCH-like volatility clustering detection
        abs_returns = np.abs(returns)
        volatility_clustering = abs_returns.autocorr(lag=1)
        
        # Volatility regimes (simple approach using quantiles)
        vol_quantiles = rolling_vol.quantile([0.33, 0.67])
        low_vol_threshold = vol_quantiles[0.33]
        high_vol_threshold = vol_quantiles[0.67]
        
        vol_regimes = pd.cut(rolling_vol, 
                           bins=[0, low_vol_threshold, high_vol_threshold, np.inf],
                           labels=['Low', 'Medium', 'High'])
        
        regime_counts = vol_regimes.value_counts()
        
        return {
            'current_volatility': rolling_vol.iloc[-1] if len(rolling_vol) > 0 else None,
            'average_volatility': rolling_vol.mean(),
            'volatility_percentile_current': stats.percentileofscore(rolling_vol.dropna(), rolling_vol.iloc[-1]) if len(rolling_vol) > 0 else None,
            'volatility_clustering': volatility_clustering,
            'low_vol_periods': regime_counts.get('Low', 0),
            'medium_vol_periods': regime_counts.get('Medium', 0),
            'high_vol_periods': regime_counts.get('High', 0),
            'rolling_volatility': rolling_vol,
            'vol_regime_thresholds': {
                'low': low_vol_threshold,
                'high': high_vol_threshold
            }
        }
    
    def momentum_analysis(self, prices: pd.Series) -> Dict:
        """
        Analyze momentum characteristics
        
        Args:
            prices: Price series
            
        Returns:
            Dict: Momentum analysis results
        """
        returns = prices.pct_change().dropna()
        
        # Price momentum (different periods)
        momentum_periods = [5, 10, 20, 50, 100]
        momentum_signals = {}
        
        for period in momentum_periods:
            if len(prices) > period:
                momentum_signals[f'{period}d'] = (prices.iloc[-1] / prices.iloc[-period-1] - 1) * 100
        
        # Return momentum (autocorrelation)
        return_momentum = {}
        for lag in [1, 5, 10]:
            if len(returns) > lag:
                return_momentum[f'lag_{lag}'] = returns.autocorr(lag=lag)
        
        # Trend strength
        price_sma_20 = prices.rolling(20).mean()
        price_sma_50 = prices.rolling(50).mean()
        
        trend_strength = {
            'above_sma20': prices.iloc[-1] > price_sma_20.iloc[-1] if len(price_sma_20.dropna()) > 0 else None,
            'above_sma50': prices.iloc[-1] > price_sma_50.iloc[-1] if len(price_sma_50.dropna()) > 0 else None,
            'sma20_above_sma50': price_sma_20.iloc[-1] > price_sma_50.iloc[-1] if len(price_sma_20.dropna()) > 0 and len(price_sma_50.dropna()) > 0 else None
        }
        
        return {
            'price_momentum': momentum_signals,
            'return_momentum': return_momentum,
            'trend_strength': trend_strength,
            'momentum_score': np.mean([v for v in return_momentum.values() if not np.isnan(v)]) if return_momentum else 0
        }
    
    def support_resistance_levels(self, prices: pd.Series, window: int = 20) -> Dict:
        """
        Identify potential support and resistance levels
        
        Args:
            prices: Price series
            window: Window for local maxima/minima detection
            
        Returns:
            Dict: Support and resistance levels
        """
        if len(prices) < window * 2:
            return {}
        
        # Find local maxima and minima
        highs = prices.rolling(window, center=True).max() == prices
        lows = prices.rolling(window, center=True).min() == prices
        
        resistance_levels = prices[highs].dropna().values
        support_levels = prices[lows].dropna().values
        
        # Cluster nearby levels
        def cluster_levels(levels, threshold=0.02):
            if len(levels) == 0:
                return []
            
            levels = np.sort(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] <= threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clusters.append(np.mean(current_cluster))
            return clusters
        
        clustered_resistance = cluster_levels(resistance_levels)
        clustered_support = cluster_levels(support_levels)
        
        current_price = prices.iloc[-1]
        
        # Find nearest levels
        nearest_resistance = min([r for r in clustered_resistance if r > current_price], default=None)
        nearest_support = max([s for s in clustered_support if s < current_price], default=None)
        
        return {
            'resistance_levels': clustered_resistance,
            'support_levels': clustered_support,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'current_price': current_price,
            'distance_to_resistance': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None,
            'distance_to_support': ((current_price - nearest_support) / current_price * 100) if nearest_support else None
        }
    
    def generate_market_summary(self, analysis_results: Dict) -> str:
        """
        Generate a comprehensive market summary report
        
        Args:
            analysis_results: Dictionary containing various analysis results
            
        Returns:
            str: Formatted market summary
        """
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("MARKET ANALYSIS SUMMARY")
        summary_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("=" * 60)
        
        # Basic Statistics Summary
        if 'basic_stats' in analysis_results:
            stats = analysis_results['basic_stats']
            summary_lines.append("\n📊 BASIC STATISTICS:")
            summary_lines.append(f"  Current Level: {stats.get('max', 'N/A'):.2f}")
            summary_lines.append(f"  Mean: {stats.get('mean', 'N/A'):.2f}")
            summary_lines.append(f"  Volatility (Std Dev): {stats.get('std_dev', 'N/A'):.2f}")
            summary_lines.append(f"  Range: {stats.get('range', 'N/A'):.2f}")
        
        # Returns Analysis Summary
        if 'returns_analysis' in analysis_results:
            returns = analysis_results['returns_analysis']
            summary_lines.append("\n📈 RETURNS ANALYSIS:")
            summary_lines.append(f"  Annual Return: {returns.get('annual_return', 0)*100:.2f}%")
            summary_lines.append(f"  Annual Volatility: {returns.get('annual_volatility', 0)*100:.2f}%")
            summary_lines.append(f"  Sharpe Ratio: {returns.get('sharpe_ratio', 0):.2f}")
            summary_lines.append(f"  Max Drawdown: {returns.get('max_drawdown', 0)*100:.2f}%")
            summary_lines.append(f"  Win Rate: {returns.get('win_rate', 0)*100:.2f}%")
        
        # Market Breadth Summary
        if 'market_breadth' in analysis_results:
            breadth = analysis_results['market_breadth']
            summary_lines.append("\n📊 MARKET BREADTH:")
            summary_lines.append(f"  Avg A/D Ratio: {breadth.get('avg_advance_decline_ratio', 'N/A'):.2f}")
            summary_lines.append(f"  Bullish Days: {breadth.get('bullish_days', 'N/A')}")
            summary_lines.append(f"  Bearish Days: {breadth.get('bearish_days', 'N/A')}")
        
        # Volatility Summary
        if 'volatility_analysis' in analysis_results:
            vol = analysis_results['volatility_analysis']
            summary_lines.append("\n📉 VOLATILITY ANALYSIS:")
            summary_lines.append(f"  Current Volatility: {vol.get('current_volatility', 'N/A'):.2f}%")
            summary_lines.append(f"  Volatility Clustering: {vol.get('volatility_clustering', 'N/A'):.3f}")
            summary_lines.append(f"  High Vol Periods: {vol.get('high_vol_periods', 'N/A')}")
        
        # Support/Resistance Summary
        if 'support_resistance' in analysis_results:
            sr = analysis_results['support_resistance']
            summary_lines.append("\n🎯 SUPPORT & RESISTANCE:")
            summary_lines.append(f"  Current Price: {sr.get('current_price', 'N/A'):.2f}")
            summary_lines.append(f"  Nearest Resistance: {sr.get('nearest_resistance', 'N/A')}")
            summary_lines.append(f"  Nearest Support: {sr.get('nearest_support', 'N/A')}")
        
        summary_lines.append("\n" + "=" * 60)
        summary_lines.append("End of Analysis")
        summary_lines.append("=" * 60)
        
        return "\n".join(summary_lines)
    
    def run_comprehensive_analysis(self, data: Union[pd.Series, pd.DataFrame], 
                                 stock_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        """
        Run comprehensive market analysis
        
        Args:
            data: Main price series or DataFrame
            stock_data: Optional dictionary of individual stock data for breadth analysis
            
        Returns:
            Dict: Comprehensive analysis results
        """
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                prices = data['Close']
            else:
                prices = data.iloc[:, 0]  # Use first column if Close not available
        else:
            prices = data
        
        results = {}
        
        try:
            # Basic statistical analysis
            results['basic_stats'] = self.basic_statistics(prices)
            
            # Returns analysis
            results['returns_analysis'] = self.returns_analysis(prices)
            
            # Volatility analysis
            results['volatility_analysis'] = self.volatility_analysis(prices)
            
            # Momentum analysis
            results['momentum_analysis'] = self.momentum_analysis(prices)
            
            # Support and resistance
            results['support_resistance'] = self.support_resistance_levels(prices)
            
            # Market breadth (if stock data provided)
            if stock_data:
                results['market_breadth'] = self.market_breadth_analysis(stock_data)
            
            # Generate summary
            results['summary_report'] = self.generate_market_summary(results)
            
        except Exception as e:
            results['error'] = f"Analysis error: {str(e)}"
            warnings.warn(f"Analysis completed with errors: {str(e)}")
        
        return results