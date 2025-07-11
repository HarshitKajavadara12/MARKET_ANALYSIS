import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from scipy import signal
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import joblib
import os
from dataclasses import dataclass
import logging
from collections import defaultdict
import math

@dataclass
class PatternConfig:
    min_pattern_length: int = 10
    max_pattern_length: int = 100
    support_resistance_threshold: float = 0.02  # 2%
    trend_threshold: float = 0.01  # 1%
    volume_confirmation: bool = True
    pattern_types: List[str] = None
    confidence_threshold: float = 0.7
    indian_market_specific: bool = False  # Flag for Indian market specific patterns
    seasonal_analysis: bool = False  # Flag for seasonal pattern analysis
    festival_patterns: bool = False  # Flag for Indian festival patterns
    gap_analysis: bool = True  # Flag for gap analysis
    
    def __post_init__(self):
        if self.pattern_types is None:
            self.pattern_types = [
                'head_and_shoulders', 'inverse_head_and_shoulders',
                'double_top', 'double_bottom', 'triple_top', 'triple_bottom',
                'ascending_triangle', 'descending_triangle', 'symmetrical_triangle',
                'flag', 'pennant', 'wedge', 'channel',
                'cup_and_handle', 'rounding_bottom', 'rounding_top',
                'gap_up', 'gap_down', 'island_reversal', 'engulfing',
                'doji', 'hammer', 'shooting_star', 'morning_star', 'evening_star'
            ]

class PatternRecognition:
    def __init__(self, config: PatternConfig = None):
        self.config = config or PatternConfig()
        self.patterns_detected = []
        self.support_resistance_levels = []
        self.trend_lines = []
        self.model_path = "models/pattern_models"
        
        # Create directories
        os.makedirs(self.model_path, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Pattern templates and characteristics
        self.pattern_characteristics = {
            'head_and_shoulders': {
                'peaks': 3,
                'valleys': 2,
                'trend': 'bearish',
                'reliability': 0.85
            },
            'inverse_head_and_shoulders': {
                'peaks': 2,
                'valleys': 3,
                'trend': 'bullish',
                'reliability': 0.85
            },
            'double_top': {
                'peaks': 2,
                'valleys': 1,
                'trend': 'bearish',
                'reliability': 0.75
            },
            'double_bottom': {
                'peaks': 1,
                'valleys': 2,
                'trend': 'bullish',
                'reliability': 0.75
            },
            'ascending_triangle': {
                'trend': 'bullish',
                'reliability': 0.70
            },
            'descending_triangle': {
                'trend': 'bearish',
                'reliability': 0.70
            },
            'gap_up': {
                'trend': 'bullish',
                'reliability': 0.65
            },
            'gap_down': {
                'trend': 'bearish',
                'reliability': 0.65
            },
            'island_reversal': {
                'trend': 'reversal',
                'reliability': 0.80
            },
            'engulfing': {
                'trend': 'reversal',
                'reliability': 0.75
            },
            'doji': {
                'trend': 'indecision',
                'reliability': 0.60
            },
            'hammer': {
                'trend': 'bullish',
                'reliability': 0.70
            },
            'shooting_star': {
                'trend': 'bearish',
                'reliability': 0.70
            },
            'morning_star': {
                'trend': 'bullish',
                'reliability': 0.80
            },
            'evening_star': {
                'trend': 'bearish',
                'reliability': 0.80
            }
        }
        
        # Indian market specific festivals and events
        if self.config.indian_market_specific:
            self.indian_festivals = {
                'Diwali': {'date': '2023-11-12', 'reliability': 0.75, 'trend': 'bullish', 'description': 'Traditional festival of lights, often associated with positive market sentiment'},
                'Budget': {'date': '2023-02-01', 'reliability': 0.80, 'trend': 'volatile', 'description': 'Annual Union Budget presentation, high impact on market sectors'},
                'Holi': {'date': '2023-03-08', 'reliability': 0.65, 'trend': 'neutral', 'description': 'Festival of colors, moderate market impact'},
                'Dussehra': {'date': '2023-10-24', 'reliability': 0.70, 'trend': 'bullish', 'description': 'Festival celebrating victory of good over evil, often positive for markets'},
                'Muhurat Trading': {'date': '2023-11-12', 'reliability': 0.85, 'trend': 'bullish', 'description': 'Special trading session on Diwali, traditionally bullish'},
                'Ganesh Chaturthi': {'date': '2023-09-19', 'reliability': 0.60, 'trend': 'neutral', 'description': 'Festival honoring Lord Ganesha'},
                'Akshaya Tritiya': {'date': '2023-04-22', 'reliability': 0.75, 'trend': 'bullish', 'description': 'Auspicious day for new beginnings and investments, especially gold'}
            }
            
            # Indian market specific sectors
            self.indian_sectors = {
                'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTI.NS', 'MINDTREE.NS', 'MPHASIS.NS'],
                'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'INDUSINDBK.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS', 'PNB.NS'],
                'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'BIOCON.NS', 'AUROPHARMA.NS', 'LUPIN.NS', 'ALKEM.NS'],
                'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS'],
                'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS', 'MARICO.NS', 'GODREJCP.NS', 'COLPAL.NS', 'EMAMILTD.NS'],
                'Energy': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'BPCL.NS', 'IOC.NS', 'GAIL.NS', 'ADANIGREEN.NS', 'TATAPOWER.NS'],
                'Metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'JINDALSTEL.NS', 'SAIL.NS', 'NATIONALUM.NS', 'VEDL.NS'],
                'Realty': ['GODREJPROP.NS', 'DLF.NS', 'PRESTIGE.NS', 'OBEROIRLTY.NS', 'BRIGADE.NS', 'SOBHA.NS', 'PHOENIXLTD.NS'],
                'PSU': ['SBIN.NS', 'ONGC.NS', 'COALINDIA.NS', 'NTPC.NS', 'POWERGRID.NS', 'BEL.NS', 'HAL.NS', 'BHEL.NS']
            }
            
            # Indian market specific patterns
            self.indian_specific_patterns = {
                'muhurat_trading_reversal': {
                    'description': 'Price reversal following Diwali Muhurat trading session',
                    'trend': 'both',  # Can be bullish or bearish
                    'reliability': 0.75,
                    'lookback_days': 5,
                    'lookforward_days': 5
                },
                'budget_impact': {
                    'description': 'Significant price movement pattern around Union Budget announcements',
                    'trend': 'both',
                    'reliability': 0.80,
                    'lookback_days': 10,
                    'lookforward_days': 10
                },
                'monsoon_pattern': {
                    'description': 'Seasonal pattern affecting agricultural and FMCG stocks based on monsoon forecasts',
                    'trend': 'both',
                    'reliability': 0.70,
                    'lookback_days': 30,
                    'lookforward_days': 60
                },
                'fii_dii_divergence': {
                    'description': 'Pattern where FII and DII flows move in opposite directions',
                    'trend': 'both',
                    'reliability': 0.85,
                    'min_consecutive_days': 3
                },
                'quarterly_results_gap': {
                    'description': 'Price gaps following quarterly results announcements',
                    'trend': 'both',
                    'reliability': 0.80,
                    'gap_threshold': 0.03  # 3% gap
                },
                'nifty_open_interest_pattern': {
                    'description': 'Price patterns based on significant changes in Nifty futures and options OI',
                    'trend': 'both',
                    'reliability': 0.75,
                    'oi_change_threshold': 0.15  # 15% OI change
                },
                'pre_election_rally': {
                    'description': 'Bullish price pattern typically observed before major elections',
                    'trend': 'bullish',
                    'reliability': 0.70,
                    'lookback_days': 60
                },
                'post_election_correction': {
                    'description': 'Bearish price pattern often seen after election results',
                    'trend': 'bearish',
                    'reliability': 0.65,
                    'lookforward_days': 30
                },
                'rbi_policy_impact': {
                    'description': 'Price patterns in banking and financial stocks following RBI policy announcements',
                    'trend': 'both',
                    'reliability': 0.80,
                    'lookback_days': 5,
                    'lookforward_days': 10
                },
                'operator_activity': {
                    'description': 'Distinctive price and volume patterns indicating potential operator activity',
                    'trend': 'both',
                    'reliability': 0.75,
                    'volume_surge_threshold': 3.0,  # 3x average volume
                    'price_change_threshold': 0.05  # 5% price change
                },
                'upper_circuit_streak': {
                    'description': 'Pattern of consecutive upper circuit hits',
                    'trend': 'bullish',
                    'reliability': 0.90,
                    'min_consecutive_days': 2
                },
                'lower_circuit_streak': {
                    'description': 'Pattern of consecutive lower circuit hits',
                    'trend': 'bearish',
                    'reliability': 0.90,
                    'min_consecutive_days': 2
                },
                'delivery_volume_spike': {
                    'description': 'Unusual spike in delivery-based trading volume',
                    'trend': 'both',
                    'reliability': 0.80,
                    'volume_threshold': 2.5  # 2.5x average delivery volume
                },
                'bulk_deal_impact': {
                    'description': 'Price pattern following reported bulk deals',
                    'trend': 'both',
                    'reliability': 0.70,
                    'lookforward_days': 5
                },
                'promoter_action': {
                    'description': 'Price patterns following promoter buying, selling, or pledge changes',
                    'trend': 'both',
                    'reliability': 0.85,
                    'lookforward_days': 20
                }
            }
    
    def find_peaks_valleys(self, data: pd.Series, prominence: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks and valleys in price data"""
        try:
            if prominence is None:
                prominence = data.std() * 0.5
            
            # Find peaks
            peaks, peak_properties = signal.find_peaks(
                data.values,
                prominence=prominence,
                distance=5  # Minimum distance between peaks
            )
            
            # Find valleys (peaks in inverted data)
            valleys, valley_properties = signal.find_peaks(
                -data.values,
                prominence=prominence,
                distance=5
            )
            
            return peaks, valleys
            
        except Exception as e:
            self.logger.error(f"Peak/valley detection failed: {e}")
            return np.array([]), np.array([])
    
    def calculate_support_resistance(self, data: pd.DataFrame) -> List[Dict]:
        """Calculate support and resistance levels"""
        try:
            levels = []
            prices = data['Close'].values
            volumes = data['Volume'].values if 'Volume' in data.columns else None
            
            # Find peaks and valleys
            peaks, valleys = self.find_peaks_valleys(data['Close'])
            
            # Combine peaks and valleys
            all_points = np.concatenate([peaks, valleys])
            all_prices = prices[all_points]
            
            if len(all_points) < 2:
                return levels
            
            # Group similar price levels
            threshold = data['Close'].mean() * self.config.support_resistance_threshold
            
            # Use clustering to find support/resistance levels
            scaler = StandardScaler()
            price_scaled = scaler.fit_transform(all_prices.reshape(-1, 1))
            
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(price_scaled)
            labels = clustering.labels_
            
            # Process each cluster
            for label in set(labels):
                if label == -1:  # Noise points
                    continue
                
                cluster_indices = np.where(labels == label)[0]
                cluster_prices = all_prices[cluster_indices]
                cluster_points = all_points[cluster_indices]
                
                # Calculate level characteristics
                level_price = np.mean(cluster_prices)
                touches = len(cluster_prices)
                strength = min(touches / 5.0, 1.0)  # Normalize to 0-1
                
                # Determine if support or resistance
                recent_price = prices[-1]
                level_type = 'resistance' if level_price > recent_price else 'support'
                
                # Calculate volume at level (if available)
                avg_volume = 0
                if volumes is not None:
                    level_volumes = volumes[cluster_points]
                    avg_volume = np.mean(level_volumes)
                
                levels.append({
                    'price': round(level_price, 2),
                    'type': level_type,
                    'strength': round(strength, 2),
                    'touches': touches,
                    'avg_volume': int(avg_volume) if avg_volume > 0 else None,
                    'last_touch': data.index[cluster_points[-1]].strftime('%Y-%m-%d')
                })
            
            # Sort by strength
            levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return levels[:10]  # Return top 10 levels
            
        except Exception as e:
            self.logger.error(f"Support/resistance calculation failed: {e}")
            return []
    
    def detect_trend_lines(self, data: pd.DataFrame) -> List[Dict]:
        """Detect trend lines in price data"""
        try:
            trend_lines = []
            prices = data['Close'].values
            dates = np.arange(len(prices))
            
            # Find peaks and valleys
            peaks, valleys = self.find_peaks_valleys(data['Close'])
            
            # Detect uptrend lines (connecting valleys)
            if len(valleys) >= 2:
                for i in range(len(valleys) - 1):
                    for j in range(i + 1, len(valleys)):
                        x1, y1 = valleys[i], prices[valleys[i]]
                        x2, y2 = valleys[j], prices[valleys[j]]
                        
                        # Calculate slope
                        slope = (y2 - y1) / (x2 - x1)
                        
                        # Check if it's an uptrend
                        if slope > 0:
                            # Validate trend line
                            touches = self._validate_trend_line(data, x1, y1, x2, y2, 'support')
                            
                            if touches >= 2:
                                trend_lines.append({
                                    'type': 'uptrend',
                                    'start_date': data.index[x1].strftime('%Y-%m-%d'),
                                    'end_date': data.index[x2].strftime('%Y-%m-%d'),
                                    'start_price': round(y1, 2),
                                    'end_price': round(y2, 2),
                                    'slope': round(slope, 4),
                                    'touches': touches,
                                    'strength': min(touches / 5.0, 1.0)
                                })
            
            # Detect downtrend lines (connecting peaks)
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    for j in range(i + 1, len(peaks)):
                        x1, y1 = peaks[i], prices[peaks[i]]
                        x2, y2 = peaks[j], prices[peaks[j]]
                        
                        # Calculate slope
                        slope = (y2 - y1) / (x2 - x1)
                        
                        # Check if it's a downtrend
                        if slope < 0:
                            # Validate trend line
                            touches = self._validate_trend_line(data, x1, y1, x2, y2, 'resistance')
                            
                            if touches >= 2:
                                trend_lines.append({
                                    'type': 'downtrend',
                                    'start_date': data.index[x1].strftime('%Y-%m-%d'),
                                    'end_date': data.index[x2].strftime('%Y-%m-%d'),
                                    'start_price': round(y1, 2),
                                    'end_price': round(y2, 2),
                                    'slope': round(slope, 4),
                                    'touches': touches,
                                    'strength': min(touches / 5.0, 1.0)
                                })
            
            # Sort by strength
            trend_lines.sort(key=lambda x: x['strength'], reverse=True)
            
            return trend_lines[:5]  # Return top 5 trend lines
            
        except Exception as e:
            self.logger.error(f"Trend line detection failed: {e}")
            return []
    
    def _validate_trend_line(self, data: pd.DataFrame, x1: int, y1: float, x2: int, y2: float, line_type: str) -> int:
        """Validate trend line by counting touches"""
        try:
            touches = 0
            prices = data['Close'].values
            threshold = data['Close'].mean() * 0.01  # 1% threshold
            
            # Calculate line equation: y = mx + b
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Check points between x1 and x2
            for x in range(x1, x2 + 1):
                expected_y = slope * x + intercept
                actual_y = prices[x]
                
                # Check if price touches the line
                if line_type == 'support':
                    if abs(actual_y - expected_y) <= threshold and actual_y >= expected_y - threshold:
                        touches += 1
                elif line_type == 'resistance':
                    if abs(actual_y - expected_y) <= threshold and actual_y <= expected_y + threshold:
                        touches += 1
            
            return touches
            
        except Exception as e:
            return 0
    
    def detect_head_and_shoulders(self, data: pd.DataFrame) -> List[Dict]:
        """Detect head and shoulders pattern"""
        try:
            patterns = []
            prices = data['Close'].values
            
            # Find peaks and valleys
            peaks, valleys = self.find_peaks_valleys(data['Close'])
            
            if len(peaks) < 3 or len(valleys) < 2:
                return patterns
            
            # Look for head and shoulders pattern
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i]
                head = peaks[i + 1]
                right_shoulder = peaks[i + 2]
                
                left_valley = None
                right_valley = None
                
                # Find valleys between peaks
                for v in valleys:
                    if left_shoulder < v < head and left_valley is None:
                        left_valley = v
                    elif head < v < right_shoulder and right_valley is None:
                        right_valley = v
                
                if left_valley is None or right_valley is None:
                    continue
                
                # Check pattern characteristics
                left_shoulder_price = prices[left_shoulder]
                head_price = prices[head]
                right_shoulder_price = prices[right_shoulder]
                left_valley_price = prices[left_valley]
                right_valley_price = prices[right_valley]
                
                # Head should be higher than both shoulders
                if head_price > left_shoulder_price and head_price > right_shoulder_price:
                    # Shoulders should be approximately equal
                    shoulder_diff = abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price
                    
                    if shoulder_diff < 0.05:  # 5% tolerance
                        # Neckline should be approximately horizontal
                        neckline_slope = abs(left_valley_price - right_valley_price) / left_valley_price
                        
                        if neckline_slope < 0.03:  # 3% tolerance
                            # Calculate pattern metrics
                            pattern_height = head_price - max(left_valley_price, right_valley_price)
                            target_price = min(left_valley_price, right_valley_price) - pattern_height
                            
                            confidence = self._calculate_pattern_confidence(
                                'head_and_shoulders',
                                [shoulder_diff, neckline_slope]
                            )
                            
                            patterns.append({
                                'pattern': 'head_and_shoulders',
                                'type': 'bearish',
                                'start_date': data.index[left_shoulder].strftime('%Y-%m-%d'),
                                'end_date': data.index[right_shoulder].strftime('%Y-%m-%d'),
                                'left_shoulder': {
                                    'date': data.index[left_shoulder].strftime('%Y-%m-%d'),
                                    'price': round(left_shoulder_price, 2)
                                },
                                'head': {
                                    'date': data.index[head].strftime('%Y-%m-%d'),
                                    'price': round(head_price, 2)
                                },
                                'right_shoulder': {
                                    'date': data.index[right_shoulder].strftime('%Y-%m-%d'),
                                    'price': round(right_shoulder_price, 2)
                                },
                                'neckline': round((left_valley_price + right_valley_price) / 2, 2),
                                'target_price': round(target_price, 2),
                                'confidence': round(confidence, 2)
                            })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Head and shoulders detection failed: {e}")
            return []
    
    def detect_double_top_bottom(self, data: pd.DataFrame) -> List[Dict]:
        """Detect double top and double bottom patterns"""
        try:
            patterns = []
            prices = data['Close'].values
            
            # Find peaks and valleys
            peaks, valleys = self.find_peaks_valleys(data['Close'])
            
            # Detect double tops
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    peak1 = peaks[i]
                    peak2 = peaks[i + 1]
                    
                    peak1_price = prices[peak1]
                    peak2_price = prices[peak2]
                    
                    # Check if peaks are approximately equal
                    price_diff = abs(peak1_price - peak2_price) / peak1_price
                    
                    if price_diff < 0.03:  # 3% tolerance
                        # Find valley between peaks
                        valley_between = None
                        for v in valleys:
                            if peak1 < v < peak2:
                                valley_between = v
                                break
                        
                        if valley_between is not None:
                            valley_price = prices[valley_between]
                            
                            # Calculate pattern metrics
                            pattern_height = max(peak1_price, peak2_price) - valley_price
                            target_price = valley_price - pattern_height
                            
                            confidence = self._calculate_pattern_confidence(
                                'double_top',
                                [price_diff]
                            )
                            
                            patterns.append({
                                'pattern': 'double_top',
                                'type': 'bearish',
                                'start_date': data.index[peak1].strftime('%Y-%m-%d'),
                                'end_date': data.index[peak2].strftime('%Y-%m-%d'),
                                'first_peak': {
                                    'date': data.index[peak1].strftime('%Y-%m-%d'),
                                    'price': round(peak1_price, 2)
                                },
                                'second_peak': {
                                    'date': data.index[peak2].strftime('%Y-%m-%d'),
                                    'price': round(peak2_price, 2)
                                },
                                'valley': {
                                    'date': data.index[valley_between].strftime('%Y-%m-%d'),
                                    'price': round(valley_price, 2)
                                },
                                'target_price': round(target_price, 2),
                                'confidence': round(confidence, 2)
                            })
            
            # Detect double bottoms
            if len(valleys) >= 2:
                for i in range(len(valleys) - 1):
                    valley1 = valleys[i]
                    valley2 = valleys[i + 1]
                    
                    valley1_price = prices[valley1]
                    valley2_price = prices[valley2]
                    
                    # Check if valleys are approximately equal
                    price_diff = abs(valley1_price - valley2_price) / valley1_price
                    
                    if price_diff < 0.03:  # 3% tolerance
                        # Find peak between valleys
                        peak_between = None
                        for p in peaks:
                            if valley1 < p < valley2:
                                peak_between = p
                                break
                        
                        if peak_between is not None:
                            peak_price = prices[peak_between]
                            
                            # Calculate pattern metrics
                            pattern_height = peak_price - min(valley1_price, valley2_price)
                            target_price = peak_price + pattern_height
                            
                            confidence = self._calculate_pattern_confidence(
                                'double_bottom',
                                [price_diff]
                            )
                            
                            patterns.append({
                                'pattern': 'double_bottom',
                                'type': 'bullish',
                                'start_date': data.index[valley1].strftime('%Y-%m-%d'),
                                'end_date': data.index[valley2].strftime('%Y-%m-%d'),
                                'first_bottom': {
                                    'date': data.index[valley1].strftime('%Y-%m-%d'),
                                    'price': round(valley1_price, 2)
                                },
                                'second_bottom': {
                                    'date': data.index[valley2].strftime('%Y-%m-%d'),
                                    'price': round(valley2_price, 2)
                                },
                                'peak': {
                                    'date': data.index[peak_between].strftime('%Y-%m-%d'),
                                    'price': round(peak_price, 2)
                                },
                                'target_price': round(target_price, 2),
                                'confidence': round(confidence, 2)
                            })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Double top/bottom detection failed: {e}")
            return []
    
    def detect_triangles(self, data: pd.DataFrame) -> List[Dict]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        try:
            patterns = []
            prices = data['Close'].values
            
            # Find peaks and valleys
            peaks, valleys = self.find_peaks_valleys(data['Close'])
            
            if len(peaks) < 2 or len(valleys) < 2:
                return patterns
            
            # Analyze recent peaks and valleys for triangle patterns
            recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
            recent_valleys = valleys[-3:] if len(valleys) >= 3 else valleys
            
            # Check for ascending triangle (horizontal resistance, rising support)
            if len(recent_peaks) >= 2 and len(recent_valleys) >= 2:
                # Check if peaks are approximately horizontal
                peak_prices = [prices[p] for p in recent_peaks]
                peak_slope = self._calculate_slope(recent_peaks, peak_prices)
                
                # Check if valleys are rising
                valley_prices = [prices[v] for v in recent_valleys]
                valley_slope = self._calculate_slope(recent_valleys, valley_prices)
                
                if abs(peak_slope) < 0.001 and valley_slope > 0.001:  # Horizontal resistance, rising support
                    patterns.append({
                        'pattern': 'ascending_triangle',
                        'type': 'bullish',
                        'start_date': data.index[min(recent_peaks[0], recent_valleys[0])].strftime('%Y-%m-%d'),
                        'end_date': data.index[max(recent_peaks[-1], recent_valleys[-1])].strftime('%Y-%m-%d'),
                        'resistance_level': round(np.mean(peak_prices), 2),
                        'support_slope': round(valley_slope, 4),
                        'target_price': round(np.mean(peak_prices) + (np.mean(peak_prices) - np.mean(valley_prices)), 2),
                        'confidence': 0.75
                    })
                
                elif abs(valley_slope) < 0.001 and peak_slope < -0.001:  # Horizontal support, falling resistance
                    patterns.append({
                        'pattern': 'descending_triangle',
                        'type': 'bearish',
                        'start_date': data.index[min(recent_peaks[0], recent_valleys[0])].strftime('%Y-%m-%d'),
                        'end_date': data.index[max(recent_peaks[-1], recent_valleys[-1])].strftime('%Y-%m-%d'),
                        'support_level': round(np.mean(valley_prices), 2),
                        'resistance_slope': round(peak_slope, 4),
                        'target_price': round(np.mean(valley_prices) - (np.mean(peak_prices) - np.mean(valley_prices)), 2),
                        'confidence': 0.75
                    })
                
                elif valley_slope > 0.001 and peak_slope < -0.001:  # Converging lines
                    patterns.append({
                        'pattern': 'symmetrical_triangle',
                        'type': 'neutral',
                        'start_date': data.index[min(recent_peaks[0], recent_valleys[0])].strftime('%Y-%m-%d'),
                        'end_date': data.index[max(recent_peaks[-1], recent_valleys[-1])].strftime('%Y-%m-%d'),
                        'support_slope': round(valley_slope, 4),
                        'resistance_slope': round(peak_slope, 4),
                        'apex_price': round((np.mean(peak_prices) + np.mean(valley_prices)) / 2, 2),
                        'confidence': 0.70
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Triangle detection failed: {e}")
            return []
    
    def _calculate_slope(self, x_values: List[int], y_values: List[float]) -> float:
        """Calculate slope of a line through points"""
        if len(x_values) < 2 or len(y_values) < 2:
            return 0.0
        
        try:
            slope, _, _, _, _ = linregress(x_values, y_values)
            return slope
        except:
            return 0.0
    
    def _calculate_pattern_confidence(self, pattern_type: str, metrics: List[float]) -> float:
        """Calculate confidence score for detected pattern"""
        try:
            base_reliability = self.pattern_characteristics.get(pattern_type, {}).get('reliability', 0.5)
            
            # Adjust confidence based on pattern-specific metrics
            confidence_adjustment = 0
            
            for metric in metrics:
                # Lower metric values generally indicate better pattern quality
                if metric < 0.01:
                    confidence_adjustment += 0.1
                elif metric < 0.03:
                    confidence_adjustment += 0.05
                elif metric > 0.1:
                    confidence_adjustment -= 0.1
            
            final_confidence = min(1.0, max(0.0, base_reliability + confidence_adjustment))
            return final_confidence
            
        except Exception as e:
            return 0.5
    
    def detect_candlestick_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Detect candlestick patterns in the data"""
        try:
            patterns = []
            df = data.copy()
            
            # Calculate basic candlestick properties
            df['body_size'] = abs(df['Close'] - df['Open'])
            df['shadow_upper'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['shadow_lower'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            df['body_to_range'] = df['body_size'] / (df['High'] - df['Low'])
            df['is_bullish'] = df['Close'] > df['Open']
            
            # Detect Doji patterns (small body, shadows on both sides)
            if 'doji' in self.config.pattern_types:
                doji_condition = (df['body_size'] / (df['High'] - df['Low']) < 0.1) & \
                                (df['shadow_upper'] > 0) & \
                                (df['shadow_lower'] > 0)
                
                doji_days = df.index[doji_condition]
                
                for day in doji_days:
                    idx = df.index.get_loc(day)
                    if idx > 0:  # Make sure we have previous data
                        prev_trend = 'uptrend' if df['Close'].iloc[idx-1] > df['Close'].iloc[max(0, idx-5):idx-1].mean() else 'downtrend'
                        
                        patterns.append({
                            'pattern': 'doji',
                            'type': 'indecision',
                            'date': day.strftime('%Y-%m-%d'),
                            'price': round(df['Close'].iloc[idx], 2),
                            'prev_trend': prev_trend,
                            'confidence': 0.6 + (0.2 if df['Volume'].iloc[idx] > df['Volume'].iloc[idx-5:idx].mean() else 0)
                        })
            
            # Detect Hammer patterns (small body at top, long lower shadow)
            if 'hammer' in self.config.pattern_types:
                hammer_condition = (df['body_to_range'] < 0.3) & \
                                  (df['shadow_lower'] > 2 * df['body_size']) & \
                                  (df['shadow_upper'] < 0.5 * df['body_size'])
                
                hammer_days = df.index[hammer_condition]
                
                for day in hammer_days:
                    idx = df.index.get_loc(day)
                    if idx > 0:  # Make sure we have previous data
                        # Hammer is bullish if it appears in a downtrend
                        prev_trend = 'downtrend' if df['Close'].iloc[idx-1] < df['Close'].iloc[max(0, idx-5):idx-1].mean() else 'uptrend'
                        
                        if prev_trend == 'downtrend':
                            patterns.append({
                                'pattern': 'hammer',
                                'type': 'bullish',
                                'date': day.strftime('%Y-%m-%d'),
                                'price': round(df['Close'].iloc[idx], 2),
                                'confidence': 0.7 + (0.1 if df['Volume'].iloc[idx] > df['Volume'].iloc[idx-5:idx].mean() else 0)
                            })
            
            # Detect Shooting Star patterns (small body at bottom, long upper shadow)
            if 'shooting_star' in self.config.pattern_types:
                shooting_star_condition = (df['body_to_range'] < 0.3) & \
                                         (df['shadow_upper'] > 2 * df['body_size']) & \
                                         (df['shadow_lower'] < 0.5 * df['body_size'])
                
                shooting_star_days = df.index[shooting_star_condition]
                
                for day in shooting_star_days:
                    idx = df.index.get_loc(day)
                    if idx > 0:  # Make sure we have previous data
                        # Shooting star is bearish if it appears in an uptrend
                        prev_trend = 'uptrend' if df['Close'].iloc[idx-1] > df['Close'].iloc[max(0, idx-5):idx-1].mean() else 'downtrend'
                        
                        if prev_trend == 'uptrend':
                            patterns.append({
                                'pattern': 'shooting_star',
                                'type': 'bearish',
                                'date': day.strftime('%Y-%m-%d'),
                                'price': round(df['Close'].iloc[idx], 2),
                                'confidence': 0.7 + (0.1 if df['Volume'].iloc[idx] > df['Volume'].iloc[idx-5:idx].mean() else 0)
                            })
            
            # Detect Engulfing patterns
            if 'engulfing' in self.config.pattern_types:
                for i in range(1, len(df)):
                    curr_day = df.iloc[i]
                    prev_day = df.iloc[i-1]
                    
                    # Bullish engulfing
                    if curr_day['is_bullish'] and not prev_day['is_bullish'] and \
                       curr_day['Open'] < prev_day['Close'] and curr_day['Close'] > prev_day['Open']:
                        
                        # Check if in a downtrend
                        if i >= 5 and df['Close'].iloc[i-1] < df['Close'].iloc[i-5:i-1].mean():
                            patterns.append({
                                'pattern': 'bullish_engulfing',
                                'type': 'bullish',
                                'date': df.index[i].strftime('%Y-%m-%d'),
                                'price': round(curr_day['Close'], 2),
                                'confidence': 0.75 + (0.15 if curr_day['Volume'] > df['Volume'].iloc[i-5:i].mean() else 0)
                            })
                    
                    # Bearish engulfing
                    elif not curr_day['is_bullish'] and prev_day['is_bullish'] and \
                         curr_day['Open'] > prev_day['Close'] and curr_day['Close'] < prev_day['Open']:
                        
                        # Check if in an uptrend
                        if i >= 5 and df['Close'].iloc[i-1] > df['Close'].iloc[i-5:i-1].mean():
                            patterns.append({
                                'pattern': 'bearish_engulfing',
                                'type': 'bearish',
                                'date': df.index[i].strftime('%Y-%m-%d'),
                                'price': round(curr_day['Close'], 2),
                                'confidence': 0.75 + (0.15 if curr_day['Volume'] > df['Volume'].iloc[i-5:i].mean() else 0)
                            })
            
            # Detect Morning Star pattern (bullish reversal)
            if 'morning_star' in self.config.pattern_types:
                for i in range(2, len(df)):
                    # First day: bearish candle
                    # Second day: small body (doji-like) with gap down
                    # Third day: bullish candle that closes above midpoint of first day
                    
                    first_day = df.iloc[i-2]
                    second_day = df.iloc[i-1]
                    third_day = df.iloc[i]
                    
                    # Check pattern conditions
                    first_bearish = not first_day['is_bullish'] and first_day['body_size'] > first_day['body_size'].mean()
                    second_small_body = second_day['body_size'] < 0.3 * first_day['body_size']
                    gap_down = second_day['High'] < first_day['Close']
                    third_bullish = third_day['is_bullish']
                    closes_above_midpoint = third_day['Close'] > (first_day['Open'] + first_day['Close']) / 2
                    
                    if first_bearish and second_small_body and gap_down and third_bullish and closes_above_midpoint:
                        # Check if in a downtrend
                        if df['Close'].iloc[i-3:i-1].mean() < df['Close'].iloc[i-8:i-3].mean():
                            # Calculate confidence based on volume and price movement
                            vol_increase = third_day['Volume'] > 1.5 * df['Volume'].iloc[i-5:i].mean()
                            price_move = (third_day['Close'] - second_day['Low']) / second_day['Low'] * 100
                            
                            confidence = 0.75 + (0.1 if vol_increase else 0) + min(0.15, price_move / 100)
                            
                            patterns.append({
                                'pattern': 'morning_star',
                                'type': 'bullish',
                                'date': df.index[i].strftime('%Y-%m-%d'),
                                'start_date': df.index[i-2].strftime('%Y-%m-%d'),
                                'price': round(third_day['Close'], 2),
                                'confidence': round(min(0.95, confidence), 2)
                            })
            
            # Detect Evening Star pattern (bearish reversal)
            if 'evening_star' in self.config.pattern_types:
                for i in range(2, len(df)):
                    # First day: bullish candle
                    # Second day: small body (doji-like) with gap up
                    # Third day: bearish candle that closes below midpoint of first day
                    
                    first_day = df.iloc[i-2]
                    second_day = df.iloc[i-1]
                    third_day = df.iloc[i]
                    
                    # Check pattern conditions
                    first_bullish = first_day['is_bullish'] and first_day['body_size'] > first_day['body_size'].mean()
                    second_small_body = second_day['body_size'] < 0.3 * first_day['body_size']
                    gap_up = second_day['Low'] > first_day['Close']
                    third_bearish = not third_day['is_bullish']
                    closes_below_midpoint = third_day['Close'] < (first_day['Open'] + first_day['Close']) / 2
                    
                    if first_bullish and second_small_body and gap_up and third_bearish and closes_below_midpoint:
                        # Check if in an uptrend
                        if df['Close'].iloc[i-3:i-1].mean() > df['Close'].iloc[i-8:i-3].mean():
                            # Calculate confidence based on volume and price movement
                            vol_increase = third_day['Volume'] > 1.5 * df['Volume'].iloc[i-5:i].mean()
                            price_move = (second_day['High'] - third_day['Close']) / second_day['High'] * 100
                            
                            confidence = 0.75 + (0.1 if vol_increase else 0) + min(0.15, price_move / 100)
                            
                            patterns.append({
                                'pattern': 'evening_star',
                                'type': 'bearish',
                                'date': df.index[i].strftime('%Y-%m-%d'),
                                'start_date': df.index[i-2].strftime('%Y-%m-%d'),
                                'price': round(third_day['Close'], 2),
                                'confidence': round(min(0.95, confidence), 2)
                            })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Candlestick pattern detection failed: {e}")
            return []
    
    def detect_gap_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Detect gap patterns in the data"""
        try:
            patterns = []
            df = data.copy()
            
            # Calculate gaps
            df['prev_high'] = df['High'].shift(1)
            df['prev_low'] = df['Low'].shift(1)
            df['gap_up'] = df['Low'] > df['prev_high']
            df['gap_down'] = df['High'] < df['prev_low']
            df['gap_size'] = np.where(df['gap_up'], df['Low'] - df['prev_high'], 
                                     np.where(df['gap_down'], df['prev_low'] - df['High'], 0))
            df['gap_pct'] = df['gap_size'] / df['prev_high'] * 100
            
            # Detect Gap Up patterns
            if 'gap_up' in self.config.pattern_types:
                gap_up_days = df.index[df['gap_up'] & (abs(df['gap_pct']) > 1.0)]  # Gaps > 1%
                
                for day in gap_up_days:
                    idx = df.index.get_loc(day)
                    if idx > 0:  # Make sure we have previous data
                        # Calculate confidence based on volume and gap size
                        vol_ratio = df['Volume'].iloc[idx] / df['Volume'].iloc[idx-5:idx].mean()
                        confidence = min(0.95, 0.65 + (abs(df['gap_pct'].iloc[idx]) * 0.03) + (vol_ratio * 0.05))
                        
                        patterns.append({
                            'pattern': 'gap_up',
                            'type': 'bullish',
                            'date': day.strftime('%Y-%m-%d'),
                            'price': round(df['Open'].iloc[idx], 2),
                            'gap_pct': round(df['gap_pct'].iloc[idx], 2),
                            'confidence': round(confidence, 2)
                        })
            
            # Detect Gap Down patterns
            if 'gap_down' in self.config.pattern_types:
                gap_down_days = df.index[df['gap_down'] & (abs(df['gap_pct']) > 1.0)]  # Gaps > 1%
                
                for day in gap_down_days:
                    idx = df.index.get_loc(day)
                    if idx > 0:  # Make sure we have previous data
                        # Calculate confidence based on volume and gap size
                        vol_ratio = df['Volume'].iloc[idx] / df['Volume'].iloc[idx-5:idx].mean()
                        confidence = min(0.95, 0.65 + (abs(df['gap_pct'].iloc[idx]) * 0.03) + (vol_ratio * 0.05))
                        
                        patterns.append({
                            'pattern': 'gap_down',
                            'type': 'bearish',
                            'date': day.strftime('%Y-%m-%d'),
                            'price': round(df['Open'].iloc[idx], 2),
                            'gap_pct': round(df['gap_pct'].iloc[idx], 2),
                            'confidence': round(confidence, 2)
                        })
            
            # Detect Island Reversal patterns
            if 'island_reversal' in self.config.pattern_types:
                for i in range(2, len(df)):
                    # Bullish island reversal (gap down followed by gap up)
                    if df['gap_down'].iloc[i-1] and df['gap_up'].iloc[i]:
                        patterns.append({
                            'pattern': 'island_reversal_bullish',
                            'type': 'bullish',
                            'date': df.index[i].strftime('%Y-%m-%d'),
                            'start_date': df.index[i-1].strftime('%Y-%m-%d'),
                            'price': round(df['Close'].iloc[i], 2),
                            'confidence': 0.80
                        })
                    
                    # Bearish island reversal (gap up followed by gap down)
                    elif df['gap_up'].iloc[i-1] and df['gap_down'].iloc[i]:
                        patterns.append({
                            'pattern': 'island_reversal_bearish',
                            'type': 'bearish',
                            'date': df.index[i].strftime('%Y-%m-%d'),
                            'start_date': df.index[i-1].strftime('%Y-%m-%d'),
                            'price': round(df['Close'].iloc[i], 2),
                            'confidence': 0.80
                        })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Gap pattern detection failed: {e}")
            return []
    
    def detect_all_patterns(self, data: pd.DataFrame, symbol: str = None) -> Dict:
        """Detect all chart patterns in the data"""
        try:
            all_patterns = []
            
            # Detect different pattern types
            if 'head_and_shoulders' in self.config.pattern_types:
                hs_patterns = self.detect_head_and_shoulders(data)
                all_patterns.extend(hs_patterns)
            
            if any(p in self.config.pattern_types for p in ['double_top', 'double_bottom']):
                double_patterns = self.detect_double_top_bottom(data)
                all_patterns.extend(double_patterns)
            
            if any(p in self.config.pattern_types for p in ['ascending_triangle', 'descending_triangle', 'symmetrical_triangle']):
                triangle_patterns = self.detect_triangles(data)
                all_patterns.extend(triangle_patterns)
            
            # Detect candlestick patterns
            if any(p in self.config.pattern_types for p in ['doji', 'hammer', 'shooting_star', 'engulfing', 'morning_star', 'evening_star']):
                candlestick_patterns = self.detect_candlestick_patterns(data)
                all_patterns.extend(candlestick_patterns)
            
            # Detect gap patterns
            if self.config.gap_analysis and any(p in self.config.pattern_types for p in ['gap_up', 'gap_down', 'island_reversal']):
                gap_patterns = self.detect_gap_patterns(data)
                all_patterns.extend(gap_patterns)
            
            # Calculate support and resistance levels
            support_resistance = self.calculate_support_resistance(data)
            
            # Detect trend lines
            trend_lines = self.detect_trend_lines(data)
            
            # Filter patterns by confidence
            high_confidence_patterns = [
                p for p in all_patterns 
                if p.get('confidence', 0) >= self.config.confidence_threshold
            ]
            
            # Calculate overall market structure
            market_structure = self._analyze_market_structure(data)
            
            result = {
                'symbol': symbol,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'total_patterns_found': len(all_patterns),
                'high_confidence_patterns': len(high_confidence_patterns),
                'patterns': all_patterns,
                'support_resistance_levels': support_resistance,
                'trend_lines': trend_lines,
                'market_structure': market_structure,
                'analysis_time': datetime.now().isoformat()
            }
            
            # Add Indian market specific analysis if enabled
            if self.config.indian_market_specific and symbol and symbol.endswith('.NS'):
                result['indian_market_context'] = self._add_indian_market_context(symbol, data)
            
            # Add seasonal pattern analysis if enabled
            if self.config.seasonal_analysis and symbol and len(data) >= 252:  # At least 1 year of data
                seasonal_analysis = self.analyze_seasonal_patterns(symbol, data)
                if seasonal_analysis and 'error' not in seasonal_analysis:
                    result['seasonal_patterns'] = seasonal_analysis
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze overall market structure"""
        try:
            prices = data['Close']
            
            # Calculate trend
            short_ma = prices.rolling(window=20).mean()
            long_ma = prices.rolling(window=50).mean()
            
            current_price = prices.iloc[-1]
            short_ma_current = short_ma.iloc[-1]
            long_ma_current = long_ma.iloc[-1]
            
            # Determine trend
            if current_price > short_ma_current > long_ma_current:
                trend = 'strong_uptrend'
            elif current_price > short_ma_current:
                trend = 'uptrend'
            elif current_price < short_ma_current < long_ma_current:
                trend = 'strong_downtrend'
            elif current_price < short_ma_current:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            # Calculate volatility
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Determine volatility regime
            if volatility < 0.15:
                vol_regime = 'low'
            elif volatility < 0.25:
                vol_regime = 'normal'
            elif volatility < 0.40:
                vol_regime = 'high'
            else:
                vol_regime = 'extreme'
            
            # Calculate momentum
            momentum_14 = (current_price - prices.iloc[-14]) / prices.iloc[-14] * 100
            
            return {
                'trend': trend,
                'volatility': round(volatility, 3),
                'volatility_regime': vol_regime,
                'momentum_14d': round(momentum_14, 2),
                'current_price': round(current_price, 2),
                'short_ma': round(short_ma_current, 2),
                'long_ma': round(long_ma_current, 2)
            }
            
        except Exception as e:
            self.logger.warning(f"Market structure analysis failed: {e}")
            return {}
    
    def _add_indian_market_context(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Add Indian market specific analysis"""
        try:
            result = {}
            
            # Extract sector information if available
            if hasattr(self, 'indian_sectors') and symbol in self.indian_sectors:
                result['sector'] = self.indian_sectors[symbol]
            
            # Check for upcoming festivals if seasonal analysis is enabled
            if self.config.seasonal_analysis and hasattr(self, 'indian_festivals'):
                today = datetime.now().date()
                upcoming_festivals = []
                
                for festival, date_str in self.indian_festivals.items():
                    festival_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    days_until = (festival_date - today).days
                    
                    if 0 <= days_until <= 30:  # Only include festivals in the next 30 days
                        upcoming_festivals.append({
                            'name': festival,
                            'date': date_str,
                            'days_until': days_until
                        })
                
                if upcoming_festivals:
                    result['upcoming_festivals'] = upcoming_festivals
            
            # Analyze seasonal patterns if enabled
            if self.config.seasonal_analysis and len(data) > 252:  # At least 1 year of data
                # Extract month from the index and calculate monthly returns
                data_copy = data.copy()
                data_copy['month'] = data_copy.index.month
                monthly_returns = data_copy.groupby('month')['Close'].apply(
                    lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 0 else 0
                ).to_dict()
                
                # Find best and worst months
                best_month = max(monthly_returns.items(), key=lambda x: x[1])
                worst_month = min(monthly_returns.items(), key=lambda x: x[1])
                
                result['seasonal_patterns'] = {
                    'best_month': {
                        'month': best_month[0],
                        'avg_return': round(best_month[1], 2)
                    },
                    'worst_month': {
                        'month': worst_month[0],
                        'avg_return': round(worst_month[1], 2)
                    }
                }
                
                # Calculate day-of-week effect
                if len(data) > 30:  # At least a month of data
                    data_copy['day_of_week'] = data_copy.index.dayofweek
                    day_returns = data_copy.groupby('day_of_week')['Close'].apply(
                        lambda x: (x.pct_change().mean() * 100) if len(x) > 5 else 0
                    ).to_dict()
                    
                    # Map numeric days to names
                    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                                3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                    
                    # Find best and worst days
                    best_day = max(day_returns.items(), key=lambda x: x[1])
                    worst_day = min(day_returns.items(), key=lambda x: x[1])
                    
                    result['seasonal_patterns']['best_day'] = {
                        'day': day_names[best_day[0]],
                        'avg_return': round(best_day[1], 2)
                    }
                    result['seasonal_patterns']['worst_day'] = {
                        'day': day_names[worst_day[0]],
                        'avg_return': round(worst_day[1], 2)
                    }
            
            # Analyze gap patterns if enabled
            if self.config.gap_analysis and len(data) > 20:
                # Calculate overnight gaps
                data_copy = data.copy()
                data_copy['prev_close'] = data_copy['Close'].shift(1)
                data_copy['overnight_gap'] = (data_copy['Open'] - data_copy['prev_close']) / data_copy['prev_close'] * 100
                
                # Calculate statistics on gaps
                avg_gap = data_copy['overnight_gap'].mean()
                gap_std = data_copy['overnight_gap'].std()
                gap_skew = data_copy['overnight_gap'].skew()
                
                # Count significant gaps (>1%)
                significant_gaps_up = (data_copy['overnight_gap'] > 1.0).sum()
                significant_gaps_down = (data_copy['overnight_gap'] < -1.0).sum()
                
                result['gap_analysis'] = {
                    'avg_overnight_gap': round(avg_gap, 2),
                    'gap_volatility': round(gap_std, 2),
                    'gap_skew': round(gap_skew, 2),
                    'significant_gaps_up': int(significant_gaps_up),
                    'significant_gaps_down': int(significant_gaps_down),
                    'gap_up_down_ratio': round(significant_gaps_up / max(1, significant_gaps_down), 2)
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Indian market context analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_pattern_signals(self, patterns: List[Dict]) -> List[Dict]:
        """Generate trading signals from detected patterns"""
        signals = []
        
        for pattern in patterns:
            try:
                signal = {
                    'pattern': pattern['pattern'],
                    'signal_type': pattern.get('type', 'neutral'),
                    'confidence': pattern.get('confidence', 0.5),
                    'entry_price': None,
                    'target_price': pattern.get('target_price'),
                    'stop_loss': None,
                    'risk_reward_ratio': None
                }
                
                # Calculate entry and stop loss based on pattern type
                if pattern['pattern'] in ['head_and_shoulders', 'double_top']:
                    signal['entry_price'] = pattern.get('neckline') or pattern.get('valley', {}).get('price')
                    signal['stop_loss'] = pattern.get('head', {}).get('price') or pattern.get('first_peak', {}).get('price')
                
                elif pattern['pattern'] in ['inverse_head_and_shoulders', 'double_bottom']:
                    signal['entry_price'] = pattern.get('neckline') or pattern.get('peak', {}).get('price')
                    signal['stop_loss'] = pattern.get('head', {}).get('price') or pattern.get('first_bottom', {}).get('price')
                
                elif 'triangle' in pattern['pattern']:
                    if pattern.get('type') == 'bullish':
                        signal['entry_price'] = pattern.get('resistance_level')
                        signal['stop_loss'] = pattern.get('support_level')
                    elif pattern.get('type') == 'bearish':
                        signal['entry_price'] = pattern.get('support_level')
                        signal['stop_loss'] = pattern.get('resistance_level')
                
                # Calculate risk-reward ratio
                if signal['entry_price'] and signal['target_price'] and signal['stop_loss']:
                    risk = abs(signal['entry_price'] - signal['stop_loss'])
                    reward = abs(signal['target_price'] - signal['entry_price'])
                    
                    if risk > 0:
                        signal['risk_reward_ratio'] = round(reward / risk, 2)
                
                signals.append(signal)
                
            except Exception as e:
                self.logger.warning(f"Signal generation failed for pattern {pattern.get('pattern', 'unknown')}: {e}")
                continue
        
        return signals
    
    def save_patterns(self, symbol: str, patterns: Dict):
        """Save detected patterns"""
        try:
            pattern_file = os.path.join(self.model_path, f"{symbol}_patterns.json")
            
            import json
            with open(pattern_file, 'w') as f:
                json.dump(patterns, f, indent=2, default=str)
            
            self.logger.info(f"Patterns saved for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to save patterns for {symbol}: {e}")
    
    def analyze_seasonal_patterns(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns in the data
        
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame with DatetimeIndex
            
        Returns:
            Dictionary with seasonal pattern analysis
        """
        try:
            result = {}
            
            # Ensure we have enough data
            if len(data) < 252:  # At least 1 year of data
                return {"error": "Insufficient data for seasonal analysis"}
            
            # Monthly seasonality
            data_copy = data.copy()
            data_copy['month'] = data_copy.index.month
            data_copy['year'] = data_copy.index.year
            
            # Calculate monthly returns
            monthly_returns = {}
            for month in range(1, 13):
                month_data = data_copy[data_copy['month'] == month]
                if len(month_data) > 0:
                    # Group by year and calculate returns
                    yearly_returns = []
                    for year, year_data in month_data.groupby('year'):
                        if len(year_data) > 0:
                            monthly_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0] - 1) * 100
                            yearly_returns.append(monthly_return)
                    
                    if yearly_returns:
                        avg_return = sum(yearly_returns) / len(yearly_returns)
                        monthly_returns[month] = round(avg_return, 2)
            
            # Find best and worst months
            if monthly_returns:
                best_month = max(monthly_returns.items(), key=lambda x: x[1])
                worst_month = min(monthly_returns.items(), key=lambda x: x[1])
                
                # Convert month numbers to names
                month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                              5: 'May', 6: 'June', 7: 'July', 8: 'August', 
                              9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                
                result['monthly_seasonality'] = {
                    'best_month': {
                        'month': month_names[best_month[0]],
                        'avg_return': best_month[1]
                    },
                    'worst_month': {
                        'month': month_names[worst_month[0]],
                        'avg_return': worst_month[1]
                    },
                    'all_months': {month_names[m]: r for m, r in monthly_returns.items()}
                }
            
            # Day of week effect
            if len(data) > 30:  # At least a month of data
                data_copy['day_of_week'] = data_copy.index.dayofweek
                
                # Calculate average returns by day of week
                day_returns = {}
                for day in range(0, 5):  # 0=Monday, 4=Friday
                    day_data = data_copy[data_copy['day_of_week'] == day]
                    if len(day_data) > 5:
                        day_return = day_data['Close'].pct_change().mean() * 100
                        day_returns[day] = round(day_return, 2)
                
                # Find best and worst days
                if day_returns:
                    best_day = max(day_returns.items(), key=lambda x: x[1])
                    worst_day = min(day_returns.items(), key=lambda x: x[1])
                    
                    # Map numeric days to names
                    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                                3: 'Thursday', 4: 'Friday'}
                    
                    result['day_of_week_effect'] = {
                        'best_day': {
                            'day': day_names[best_day[0]],
                            'avg_return': best_day[1]
                        },
                        'worst_day': {
                            'day': day_names[worst_day[0]],
                            'avg_return': worst_day[1]
                        },
                        'all_days': {day_names[d]: r for d, r in day_returns.items()}
                    }
            
            # Festival effect (if Indian market specific is enabled)
            if self.config.indian_market_specific and self.config.festival_patterns and hasattr(self, 'indian_festivals'):
                # Calculate returns around festivals
                festival_returns = {}
                
                for festival, date_str in self.indian_festivals.items():
                    try:
                        festival_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        # Find the closest trading day in our data
                        closest_date = None
                        min_days_diff = float('inf')
                        
                        for date in data.index:
                            days_diff = abs((date.date() - festival_date).days)
                            if days_diff < min_days_diff:
                                min_days_diff = days_diff
                                closest_date = date
                        
                        if closest_date and min_days_diff <= 5:  # Within 5 days of festival
                            # Calculate pre-festival return (5 days before)
                            pre_idx = data.index.get_loc(closest_date)
                            if pre_idx >= 5:
                                pre_price = data['Close'].iloc[pre_idx - 5]
                                festival_price = data['Close'].iloc[pre_idx]
                                pre_return = (festival_price / pre_price - 1) * 100
                                
                                # Calculate post-festival return (5 days after)
                                if pre_idx + 5 < len(data):
                                    post_price = data['Close'].iloc[pre_idx + 5]
                                    post_return = (post_price / festival_price - 1) * 100
                                    
                                    festival_returns[festival] = {
                                        'pre_festival': round(pre_return, 2),
                                        'post_festival': round(post_return, 2)
                                    }
                    except Exception as e:
                        self.logger.warning(f"Festival analysis failed for {festival}: {e}")
                
                if festival_returns:
                    result['festival_effect'] = festival_returns
            
            return result
            
        except Exception as e:
            self.logger.error(f"Seasonal pattern analysis failed for {symbol}: {e}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Create pattern recognition system
    config = PatternConfig(
        min_pattern_length=20,
        confidence_threshold=0.7,
        pattern_types=['head_and_shoulders', 'double_top', 'double_bottom', 'ascending_triangle',
                      'doji', 'hammer', 'engulfing', 'morning_star', 'evening_star',
                      'gap_up', 'gap_down'],
        seasonal_analysis=True,
        indian_market_specific=True,
        festival_patterns=True,
        gap_analysis=True
    )
    
    recognizer = PatternRecognition(config)
    
    # Generate sample data with patterns
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Create data with head and shoulders pattern
    price = 1000
    prices = []
    
    for i in range(len(dates)):
        if i < 100:  # Rising trend
            price += np.random.normal(2, 10)
        elif i < 150:  # Left shoulder
            price += np.random.normal(-1, 8)
        elif i < 200:  # Head formation
            price += np.random.normal(3, 12)
        elif i < 250:  # Right shoulder
            price += np.random.normal(-2, 10)
        else:  # Decline
            price += np.random.normal(-1, 8)
        
        prices.append(max(price, 500))  # Prevent negative prices
    
    sample_data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100000, 1000000) for _ in prices]
    }, index=dates)
    
    print("Sample data created with patterns:")
    print(sample_data.tail())
    
    # Detect patterns
    print("\nDetecting patterns...")
    pattern_results = recognizer.detect_all_patterns(sample_data, 'TEST_STOCK')
    
    print(f"\nPattern Detection Results:")
    print(f"Total patterns found: {pattern_results['total_patterns_found']}")
    print(f"High confidence patterns: {pattern_results['high_confidence_patterns']}")
    
    print("\nDetected Patterns:")
    for pattern in pattern_results['patterns']:
        print(f"- {pattern['pattern']}: {pattern['type']} (confidence: {pattern['confidence']})")
    
    print(f"\nSupport/Resistance Levels: {len(pattern_results['support_resistance_levels'])}")
    for level in pattern_results['support_resistance_levels'][:3]:
        print(f"- {level['type']}: {level['price']} (strength: {level['strength']})")
    
    print(f"\nTrend Lines: {len(pattern_results['trend_lines'])}")
    for line in pattern_results['trend_lines'][:2]:
        print(f"- {line['type']}: {line['start_price']} -> {line['end_price']} (strength: {line['strength']})")
    
    # Generate signals
    signals = recognizer.get_pattern_signals(pattern_results['patterns'])
    print(f"\nGenerated Signals: {len(signals)}")
    for signal in signals:
        print(f"- {signal['pattern']}: {signal['signal_type']} (R:R = {signal.get('risk_reward_ratio', 'N/A')})")
    
    # Check for seasonal patterns
    if 'seasonal_patterns' in pattern_results:
        print("\nSeasonal Pattern Analysis:")
        seasonal = pattern_results['seasonal_patterns']
        
        if 'monthly_seasonality' in seasonal:
            monthly = seasonal['monthly_seasonality']
            print(f"Best month: {monthly['best_month']['month']} ({monthly['best_month']['avg_return']}%)")
            print(f"Worst month: {monthly['worst_month']['month']} ({monthly['worst_month']['avg_return']}%)")
        
        if 'day_of_week_effect' in seasonal:
            dow = seasonal['day_of_week_effect']
            print(f"Best day: {dow['best_day']['day']} ({dow['best_day']['avg_return']}%)")
            print(f"Worst day: {dow['worst_day']['day']} ({dow['worst_day']['avg_return']}%)")
        
        if 'festival_effect' in seasonal:
            print("\nFestival Effects:")
            for festival, effect in list(seasonal['festival_effect'].items())[:3]:  # Show top 3
                print(f"- {festival}: Pre: {effect['pre_festival']}%, Post: {effect['post_festival']}%")