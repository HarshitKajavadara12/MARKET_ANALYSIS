"""
Market Research System v1.0 - Trend Analysis Module
Author: Independent Market Researcher
Created: 2022
Updated: 2022-12-31

This module handles comprehensive trend analysis for stocks and market indices.
Focuses on Indian stock market data with NSE/BSE listings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TrendAnalyzer:
    """
    Comprehensive trend analysis for market data.
    Identifies trends, trend strength, and trend changes.
    """
    
    def __init__(self):
        self.trend_data = {}
        self.support_resistance_levels = {}
        
    def identify_trend_direction(self, data: pd.Series, 
                               lookback_period: int = 20) -> str:
        """
        Identify the current trend direction using multiple methods.
        
        Args:
            data: Price data series
            lookback_period: Number of periods to analyze
            
        Returns:
            Trend direction: 'uptrend', 'downtrend', or 'sideways'
        """
        try:
            if len(data) < lookback_period:
                return 'insufficient_data'
            
            recent_data = data.tail(lookback_period)
            
            # Method 1: Simple slope analysis
            x = np.arange(len(recent_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_data)
            
            # Method 2: Moving average comparison
            short_ma = data.rolling(window=5).mean().tail(1).iloc[0]
            long_ma = data.rolling(window=20).mean().tail(1).iloc[0]
            
            # Method 3: Price position analysis
            current_price = data.iloc[-1]
            period_high = recent_data.max()
            period_low = recent_data.min()
            price_position = (current_price - period_low) / (period_high - period_low)
            
            # Combine methods for final decision
            slope_threshold = data.std() * 0.01  # Adaptive threshold
            
            trend_signals = []
            
            # Slope signal
            if slope > slope_threshold:
                trend_signals.append(1)  # Uptrend
            elif slope < -slope_threshold:
                trend_signals.append(-1)  # Downtrend
            else:
                trend_signals.append(0)  # Sideways
            
            # Moving average signal
            if short_ma > long_ma * 1.02:  # 2% threshold
                trend_signals.append(1)
            elif short_ma < long_ma * 0.98:
                trend_signals.append(-1)
            else:
                trend_signals.append(0)
            
            # Price position signal
            if price_position > 0.7:
                trend_signals.append(1)
            elif price_position < 0.3:
                trend_signals.append(-1)
            else:
                trend_signals.append(0)
            
            # Final decision
            trend_score = sum(trend_signals)
            
            if trend_score >= 2:
                return 'uptrend'
            elif trend_score <= -2:
                return 'downtrend'
            else:
                return 'sideways'
                
        except Exception as e:
            print(f"Error identifying trend direction: {e}")
            return 'error'
    
    def calculate_trend_strength(self, data: pd.Series, 
                               period: int = 20) -> float:
        """
        Calculate trend strength using R-squared of linear regression.
        
        Args:
            data: Price data series
            period: Period for analysis
            
        Returns:
            Trend strength (0-1, higher means stronger trend)
        """
        try:
            if len(data) < period:
                return 0.0
            
            recent_data = data.tail(period)
            x = np.arange(len(recent_data)).reshape(-1, 1)
            y = recent_data.values
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(x, y)
            
            # Calculate R-squared
            y_pred = model.predict(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return max(0, r_squared)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error calculating trend strength: {e}")
            return 0.0
    
    def detect_trend_changes(self, data: pd.Series, 
                           short_window: int = 10,
                           long_window: int = 30) -> pd.DataFrame:
        """
        Detect trend changes using dual moving average crossovers.
        
        Args:
            data: Price data series
            short_window: Short-term moving average period
            long_window: Long-term moving average period
            
        Returns:
            DataFrame with trend change signals
        """
        try:
            df = pd.DataFrame(index=data.index)
            df['price'] = data
            df['short_ma'] = data.rolling(window=short_window).mean()
            df['long_ma'] = data.rolling(window=long_window).mean()
            
            # Calculate crossovers
            df['signal'] = 0
            df['signal'][short_window:] = np.where(
                df['short_ma'][short_window:] > df['long_ma'][short_window:], 1, 0
            )
            
            # Identify changes
            df['position'] = df['signal'].diff()
            
            # Mark trend changes
            df['trend_change'] = 0
            df.loc[df['position'] == 1, 'trend_change'] = 1  # Bullish crossover
            df.loc[df['position'] == -1, 'trend_change'] = -1  # Bearish crossover
            
            # Add trend labels
            df['trend'] = 'sideways'
            df.loc[df['signal'] == 1, 'trend'] = 'uptrend'
            df.loc[df['signal'] == 0, 'trend'] = 'downtrend'
            
            return df[['price', 'short_ma', 'long_ma', 'trend', 'trend_change']].dropna()
            
        except Exception as e:
            print(f"Error detecting trend changes: {e}")
            return pd.DataFrame()
    
    def find_support_resistance(self, data: pd.Series, 
                              window: int = 20,
                              min_touches: int = 2) -> Dict:
        """
        Identify support and resistance levels.
        
        Args:
            data: Price data series
            window: Window for local extrema detection
            min_touches: Minimum touches required for level validation
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Find local maxima and minima
            highs = []
            lows = []
            
            for i in range(window, len(data) - window):
                # Check for local maximum
                if data.iloc[i] == data.iloc[i-window:i+window+1].max():
                    highs.append((data.index[i], data.iloc[i]))
                
                # Check for local minimum
                if data.iloc[i] == data.iloc[i-window:i+window+1].min():
                    lows.append((data.index[i], data.iloc[i]))
            
            # Cluster levels
            resistance_levels = self._cluster_levels([h[1] for h in highs], data, min_touches)
            support_levels = self._cluster_levels([l[1] for l in lows], data, min_touches)
            
            result = {
                'resistance_levels': resistance_levels,
                'support_levels': support_levels,
                'current_price': data.iloc[-1],
                'analysis_date': data.index[-1]
            }
            
            self.support_resistance_levels = result
            return result
            
        except Exception as e:
            print(f"Error finding support/resistance: {e}")
            return {}
    
    def _cluster_levels(self, levels: List[float], data: pd.Series, 
                       min_touches: int) -> List[Dict]:
        """
        Cluster price levels to identify significant support/resistance.
        
        Args:
            levels: List of price levels
            data: Price data for validation
            min_touches: Minimum touches required
            
        Returns:
            List of validated levels with metadata
        """
        try:
            if not levels:
                return []
            
            # Calculate price tolerance (1% of average price)
            tolerance = data.mean() * 0.01
            
            clustered_levels = []
            used_levels = set()
            
            for level in sorted(levels):
                if level in used_levels:
                    continue
                
                # Find nearby levels
                nearby = [l for l in levels if abs(l - level) <= tolerance]
                
                if len(nearby) >= min_touches:
                    avg_level = np.mean(nearby)
                    
                    # Count actual touches in data
                    touches = sum(1 for price in data if abs(price - avg_level) <= tolerance)
                    
                    clustered_levels.append({
                        'level': avg_level,
                        'touches': touches,
                        'strength': len(nearby),
                        'last_touch': self._find_last_touch(data, avg_level, tolerance)
                    })
                    
                    used_levels.update(nearby)
            
            return sorted(clustered_levels, key=lambda x: x['strength'], reverse=True)
            
        except Exception as e:
            print(f"Error clustering levels: {e}")
            return []
    
    def _find_last_touch(self, data: pd.Series, level: float, 
                        tolerance: float) -> Optional[str]:
        """Find the last time price touched a specific level."""
        try:
            for i in range(len(data)-1, -1, -1):
                if abs(data.iloc[i] - level) <= tolerance:
                    return data.index[i].strftime('%Y-%m-%d')
            return None
        except:
            return None
    
    def trend_momentum_analysis(self, data: pd.Series, 
                              periods: List[int] = [5, 10, 20, 50]) -> Dict:
        """
        Analyze trend momentum across multiple timeframes.
        
        Args:
            data: Price data series
            periods: List of periods to analyze
            
        Returns:
            Dictionary with momentum analysis
        """
        try:
            current_price = data.iloc[-1]
            results = {
                'current_price': current_price,
                'momentum_analysis': {},
                'overall_momentum': 'neutral'
            }
            
            momentum_scores = []
            
            for period in periods:
                if len(data) >= period:
                    # Calculate percentage change
                    past_price = data.iloc[-period]
                    pct_change = ((current_price - past_price) / past_price) * 100
                    
                    # Calculate momentum score
                    if pct_change > 2:
                        momentum_score = 1  # Strong positive
                    elif pct_change > 0.5:
                        momentum_score = 0.5  # Weak positive
                    elif pct_change < -2:
                        momentum_score = -1  # Strong negative
                    elif pct_change < -0.5:
                        momentum_score = -0.5  # Weak negative
                    else:
                        momentum_score = 0  # Neutral
                    
                    momentum_scores.append(momentum_score)
                    
                    results['momentum_analysis'][f'{period}_day'] = {
                        'price_change': current_price - past_price,
                        'pct_change': pct_change,
                        'momentum_score': momentum_score,
                        'trend_direction': 'up' if pct_change > 0 else 'down' if pct_change < 0 else 'flat'
                    }
            
            # Overall momentum assessment
            if momentum_scores:
                avg_momentum = np.mean(momentum_scores)
                if avg_momentum > 0.3:
                    results['overall_momentum'] = 'bullish'
                elif avg_momentum < -0.3:
                    results['overall_momentum'] = 'bearish'
                else:
                    results['overall_momentum'] = 'neutral'
                
                results['momentum_score'] = avg_momentum
            
            return results
            
        except Exception as e:
            print(f"Error in momentum analysis: {e}")
            return {}
    
    def plot_trend_analysis(self, data: pd.Series, title: str = "Trend Analysis",
                          figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot comprehensive trend analysis chart.
        
        Args:
            data: Price data series
            title: Chart title
            figsize: Figure size
        """
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                               gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Main price chart with trend
            ax1.plot(data.index, data, label='Price', linewidth=2, color='black')
            
            # Add moving averages
            ma20 = data.rolling(window=20).mean()
            ma50 = data.rolling(window=50).mean()
            
            ax1.plot(data.index, ma20, label='MA20', alpha=0.7, color='blue')
            if len(data) >= 50:
                ax1.plot(data.index, ma50, label='MA50', alpha=0.7, color='red')
            
            # Add support/resistance levels if available
            if hasattr(self, 'support_resistance_levels') and self.support_resistance_levels:
                sr_data = self.support_resistance_levels
                
                # Plot resistance levels
                for level in sr_data.get('resistance_levels', [])[:3]:  # Top 3 levels
                    ax1.axhline(y=level['level'], color='red', linestyle='--', 
                               alpha=0.6, label=f"Resistance: {level['level']:.2f}")
                
                # Plot support levels
                for level in sr_data.get('support_levels', [])[:3]:  # Top 3 levels
                    ax1.axhline(y=level['level'], color='green', linestyle='--', 
                               alpha=0.6, label=f"Support: {level['level']:.2f}")
            
            ax1.set_title(f'{title} - Price and Trend Analysis')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Trend direction subplot
            trend_data = self.detect_trend_changes(data)
            if not trend_data.empty:
                # Color code trend periods
                for i in range(len(trend_data)):
                    if trend_data.iloc[i]['trend'] == 'uptrend':
                        ax2.scatter(trend_data.index[i], 1, color='green', alpha=0.6, s=10)
                    elif trend_data.iloc[i]['trend'] == 'downtrend':
                        ax2.scatter(trend_data.index[i], -1, color='red', alpha=0.6, s=10)
                    else:
                        ax2.scatter(trend_data.index[i], 0, color='gray', alpha=0.6, s=10)
                
                # Mark trend changes
                change_points = trend_data[trend_data['trend_change'] != 0]
                for idx, row in change_points.iterrows():
                    color = 'green' if row['trend_change'] == 1 else 'red'
                    ax2.axvline(x=idx, color=color, alpha=0.7, linestyle='-', linewidth=2)
            
            ax2.set_ylabel('Trend Direction')
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Downtrend', 'Sideways', 'Uptrend'])
            ax2.grid(True, alpha=0.3)
            
            # Momentum analysis subplot
            momentum_data = self.trend_momentum_analysis(data)
            if momentum_data and 'momentum_analysis' in momentum_data:
                periods = []
                momentum_scores = []
                
                for period, analysis in momentum_data['momentum_analysis'].items():
                    periods.append(period.replace('_day', 'd'))
                    momentum_scores.append(analysis['momentum_score'])
                
                colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in momentum_scores]
                ax3.bar(periods, momentum_scores, color=colors, alpha=0.7)
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax3.set_ylabel('Momentum Score')
                ax3.set_title('Multi-Timeframe Momentum Analysis')
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting trend analysis: {e}")
    
    def generate_trend_report(self, symbol: str, data: pd.Series) -> Dict:
        """
        Generate comprehensive trend analysis report.
        
        Args:
            symbol: Stock symbol
            data: Price data series
            
        Returns:
            Dictionary containing comprehensive trend analysis
        """
        try:
            # Get current trend direction and strength
            current_trend = self.identify_trend_direction(data)
            trend_strength = self.calculate_trend_strength(data)
            
            # Get momentum analysis
            momentum_analysis = self.trend_momentum_analysis(data)
            
            # Get support/resistance levels
            sr_levels = self.find_support_resistance(data)
            
            # Calculate additional metrics
            current_price = data.iloc[-1]
            
            # Price changes
            changes = {
                '1d': ((current_price - data.iloc[-2]) / data.iloc[-2] * 100) if len(data) > 1 else 0,
                '5d': ((current_price - data.iloc[-6]) / data.iloc[-6] * 100) if len(data) > 5 else 0,
                '20d': ((current_price - data.iloc[-21]) / data.iloc[-21] * 100) if len(data) > 20 else 0,
                '50d': ((current_price - data.iloc[-51]) / data.iloc[-51] * 100) if len(data) > 50 else 0
            }
            
            # Volatility analysis
            volatility_20d = data.pct_change().rolling(window=20).std() * np.sqrt(252) * 100
            current_volatility = volatility_20d.iloc[-1] if not volatility_20d.empty else 0
            
            # Risk assessment
            risk_level = 'Low'
            if current_volatility > 30:
                risk_level = 'High'
            elif current_volatility > 20:
                risk_level = 'Medium'
            
            # Generate trend signals
            signals = []
            
            if current_trend == 'uptrend' and trend_strength > 0.7:
                signals.append("Strong uptrend confirmed")
            elif current_trend == 'uptrend' and trend_strength > 0.4:
                signals.append("Weak uptrend - monitor for continuation")
            elif current_trend == 'downtrend' and trend_strength > 0.7:
                signals.append("Strong downtrend - caution advised")
            elif current_trend == 'downtrend' and trend_strength > 0.4:
                signals.append("Weak downtrend - potential reversal")
            else:
                signals.append("Sideways movement - wait for clear direction")
            
            # Check proximity to support/resistance
            if sr_levels:
                for resistance in sr_levels.get('resistance_levels', [])[:2]:
                    if abs(current_price - resistance['level']) / current_price < 0.05:  # Within 5%
                        signals.append(f"Near resistance at {resistance['level']:.2f}")
                
                for support in sr_levels.get('support_levels', [])[:2]:
                    if abs(current_price - support['level']) / current_price < 0.05:  # Within 5%
                        signals.append(f"Near support at {support['level']:.2f}")
            
            report = {
                'symbol': symbol,
                'analysis_date': data.index[-1].strftime('%Y-%m-%d'),
                'current_price': current_price,
                'trend_analysis': {
                    'direction': current_trend,
                    'strength': trend_strength,
                    'confidence': 'High' if trend_strength > 0.7 else 'Medium' if trend_strength > 0.4 else 'Low'
                },
                'price_changes': changes,
                'momentum_analysis': momentum_analysis,
                'support_resistance': sr_levels,
                'volatility': {
                    'current_20d_annualized': current_volatility,
                    'risk_level': risk_level
                },
                'signals': signals,
                'recommendation': self._generate_recommendation(current_trend, trend_strength, 
                                                             momentum_analysis, current_volatility)
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating trend report: {e}")
            return {}
    
    def _generate_recommendation(self, trend: str, strength: float, 
                               momentum: Dict, volatility: float) -> str:
        """
        Generate trading recommendation based on analysis.
        
        Args:
            trend: Current trend direction
            strength: Trend strength (0-1)
            momentum: Momentum analysis data
            volatility: Current volatility
            
        Returns:
            Trading recommendation string
        """
        try:
            # Base recommendation on trend and strength
            if trend == 'uptrend' and strength > 0.7:
                if volatility < 20:
                    return "BUY - Strong uptrend with low volatility"
                else:
                    return "CAUTIOUS BUY - Strong uptrend but high volatility"
            
            elif trend == 'uptrend' and strength > 0.4:
                overall_momentum = momentum.get('overall_momentum', 'neutral')
                if overall_momentum == 'bullish':
                    return "BUY - Weak uptrend with positive momentum"
                else:
                    return "HOLD - Weak uptrend, monitor for continuation"
            
            elif trend == 'downtrend' and strength > 0.7:
                return "SELL - Strong downtrend confirmed"
            
            elif trend == 'downtrend' and strength > 0.4:
                overall_momentum = momentum.get('overall_momentum', 'neutral')
                if overall_momentum == 'bearish':
                    return "SELL - Weak downtrend with negative momentum"
                else:
                    return "HOLD - Weak downtrend, potential reversal"
            
            else:
                return "HOLD - Sideways movement, wait for clear direction"
                
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return "HOLD - Analysis incomplete"
    
    def batch_trend_analysis(self, data_dict: Dict[str, pd.Series]) -> Dict:
        """
        Perform trend analysis on multiple symbols.
        
        Args:
            data_dict: Dictionary with symbol as key and price data as value
            
        Returns:
            Dictionary with analysis results for each symbol
        """
        try:
            results = {}
            
            for symbol, data in data_dict.items():
                print(f"Analyzing trend for {symbol}...")
                try:
                    report = self.generate_trend_report(symbol, data)
                    results[symbol] = report
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
                    results[symbol] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            return {}
    
    def export_analysis_to_csv(self, analysis_results: Dict, 
                              filename: str = "trend_analysis.csv") -> bool:
        """
        Export trend analysis results to CSV file.
        
        Args:
            analysis_results: Results from trend analysis
            filename: Output filename
            
        Returns:
            Success status
        """
        try:
            rows = []
            
            for symbol, data in analysis_results.items():
                if 'error' in data:
                    continue
                
                row = {
                    'Symbol': symbol,
                    'Date': data.get('analysis_date', ''),
                    'Current_Price': data.get('current_price', 0),
                    'Trend_Direction': data.get('trend_analysis', {}).get('direction', ''),
                    'Trend_Strength': data.get('trend_analysis', {}).get('strength', 0),
                    'Confidence': data.get('trend_analysis', {}).get('confidence', ''),
                    'Change_1D': data.get('price_changes', {}).get('1d', 0),
                    'Change_5D': data.get('price_changes', {}).get('5d', 0),
                    'Change_20D': data.get('price_changes', {}).get('20d', 0),
                    'Volatility': data.get('volatility', {}).get('current_20d_annualized', 0),
                    'Risk_Level': data.get('volatility', {}).get('risk_level', ''),
                    'Recommendation': data.get('recommendation', ''),
                    'Key_Signals': '; '.join(data.get('signals', []))
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
            print(f"Analysis exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False


# Usage example and testing functions
if __name__ == "__main__":
    # Example usage
    analyzer = TrendAnalyzer()
    
    # Generate sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    
    # Create trending sample data
    trend = np.linspace(100, 150, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    sample_prices = trend + noise + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    
    sample_data = pd.Series(sample_prices, index=dates)
    
    print("=== Trend Analysis Demo ===")
    print(f"Trend Direction: {analyzer.identify_trend_direction(sample_data)}")
    print(f"Trend Strength: {analyzer.calculate_trend_strength(sample_data):.3f}")
    
    # Generate full report
    report = analyzer.generate_trend_report("SAMPLE", sample_data)
    
    print("\n=== Sample Analysis Report ===")
    for key, value in report.items():
        if key not in ['support_resistance', 'momentum_analysis']:
            print(f"{key}: {value}")
    
    print("\n=== Support/Resistance Levels ===")
    sr_data = report.get('support_resistance', {})
    if sr_data:
        print("Resistance Levels:")
        for level in sr_data.get('resistance_levels', [])[:3]:
            print(f"  {level['level']:.2f} (Strength: {level['strength']}, Touches: {level['touches']})")
        
        print("Support Levels:")
        for level in sr_data.get('support_levels', [])[:3]:
            print(f"  {level['level']:.2f} (Strength: {level['strength']}, Touches: {level['touches']})")