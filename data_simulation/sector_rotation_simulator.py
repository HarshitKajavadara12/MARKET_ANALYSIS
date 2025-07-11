#!/usr/bin/env python3
"""
Sector Rotation Simulator - Indian Market Research Platform
Simulates sector-wise movement patterns and rotation cycles
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass
from enum import Enum
import math

class MarketCycle(Enum):
    """Market cycle phases"""
    EARLY_BULL = "early_bull"
    LATE_BULL = "late_bull"
    EARLY_BEAR = "early_bear"
    LATE_BEAR = "late_bear"
    RECOVERY = "recovery"
    EXPANSION = "expansion"
    PEAK = "peak"
    CONTRACTION = "contraction"

class SectorStyle(Enum):
    """Sector investment styles"""
    GROWTH = "growth"
    VALUE = "value"
    DEFENSIVE = "defensive"
    CYCLICAL = "cyclical"
    MOMENTUM = "momentum"
    QUALITY = "quality"

@dataclass
class SectorData:
    """Sector characteristics and performance data"""
    name: str
    style: SectorStyle
    beta: float
    dividend_yield: float
    pe_ratio: float
    market_cap_bias: str  # 'large', 'mid', 'small'
    economic_sensitivity: float  # 0-1 scale
    interest_rate_sensitivity: float  # -1 to 1 scale
    inflation_sensitivity: float  # -1 to 1 scale
    global_exposure: float  # 0-1 scale
    
class SectorRotationSimulator:
    """
    Simulates realistic sector rotation patterns in Indian markets
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
            
        self.sectors = self._initialize_sectors()
        self.rotation_patterns = self._initialize_rotation_patterns()
        self.cycle_characteristics = self._initialize_cycle_characteristics()
        self.correlation_matrix = self._initialize_correlation_matrix()
        
    def _initialize_sectors(self) -> Dict[str, SectorData]:
        """Initialize Indian market sectors with their characteristics"""
        return {
            'Banking': SectorData(
                name='Banking',
                style=SectorStyle.VALUE,
                beta=1.2,
                dividend_yield=2.5,
                pe_ratio=12.0,
                market_cap_bias='large',
                economic_sensitivity=0.9,
                interest_rate_sensitivity=0.8,
                inflation_sensitivity=-0.3,
                global_exposure=0.3
            ),
            'IT': SectorData(
                name='IT',
                style=SectorStyle.GROWTH,
                beta=0.8,
                dividend_yield=1.8,
                pe_ratio=25.0,
                market_cap_bias='large',
                economic_sensitivity=0.4,
                interest_rate_sensitivity=-0.6,
                inflation_sensitivity=0.2,
                global_exposure=0.9
            ),
            'Pharma': SectorData(
                name='Pharma',
                style=SectorStyle.DEFENSIVE,
                beta=0.7,
                dividend_yield=1.2,
                pe_ratio=22.0,
                market_cap_bias='large',
                economic_sensitivity=0.2,
                interest_rate_sensitivity=-0.2,
                inflation_sensitivity=0.1,
                global_exposure=0.7
            ),
            'Auto': SectorData(
                name='Auto',
                style=SectorStyle.CYCLICAL,
                beta=1.4,
                dividend_yield=1.5,
                pe_ratio=18.0,
                market_cap_bias='large',
                economic_sensitivity=0.8,
                interest_rate_sensitivity=-0.5,
                inflation_sensitivity=-0.6,
                global_exposure=0.4
            ),
            'FMCG': SectorData(
                name='FMCG',
                style=SectorStyle.DEFENSIVE,
                beta=0.6,
                dividend_yield=2.8,
                pe_ratio=35.0,
                market_cap_bias='large',
                economic_sensitivity=0.3,
                interest_rate_sensitivity=-0.1,
                inflation_sensitivity=-0.4,
                global_exposure=0.2
            ),
            'Energy': SectorData(
                name='Energy',
                style=SectorStyle.VALUE,
                beta=1.3,
                dividend_yield=3.5,
                pe_ratio=10.0,
                market_cap_bias='large',
                economic_sensitivity=0.7,
                interest_rate_sensitivity=0.3,
                inflation_sensitivity=0.8,
                global_exposure=0.8
            ),
            'Metals': SectorData(
                name='Metals',
                style=SectorStyle.CYCLICAL,
                beta=1.6,
                dividend_yield=2.0,
                pe_ratio=8.0,
                market_cap_bias='mid',
                economic_sensitivity=0.9,
                interest_rate_sensitivity=0.2,
                inflation_sensitivity=0.7,
                global_exposure=0.9
            ),
            'Infrastructure': SectorData(
                name='Infrastructure',
                style=SectorStyle.CYCLICAL,
                beta=1.5,
                dividend_yield=1.8,
                pe_ratio=15.0,
                market_cap_bias='large',
                economic_sensitivity=0.8,
                interest_rate_sensitivity=-0.7,
                inflation_sensitivity=-0.5,
                global_exposure=0.3
            ),
            'Telecom': SectorData(
                name='Telecom',
                style=SectorStyle.VALUE,
                beta=1.1,
                dividend_yield=0.5,
                pe_ratio=20.0,
                market_cap_bias='large',
                economic_sensitivity=0.5,
                interest_rate_sensitivity=-0.8,
                inflation_sensitivity=-0.2,
                global_exposure=0.2
            ),
            'Real Estate': SectorData(
                name='Real Estate',
                style=SectorStyle.CYCLICAL,
                beta=1.8,
                dividend_yield=1.0,
                pe_ratio=12.0,
                market_cap_bias='mid',
                economic_sensitivity=0.9,
                interest_rate_sensitivity=-0.9,
                inflation_sensitivity=-0.8,
                global_exposure=0.1
            ),
            'Consumer Durables': SectorData(
                name='Consumer Durables',
                style=SectorStyle.CYCLICAL,
                beta=1.2,
                dividend_yield=1.5,
                pe_ratio=25.0,
                market_cap_bias='large',
                economic_sensitivity=0.7,
                interest_rate_sensitivity=-0.4,
                inflation_sensitivity=-0.5,
                global_exposure=0.3
            ),
            'Capital Goods': SectorData(
                name='Capital Goods',
                style=SectorStyle.CYCLICAL,
                beta=1.4,
                dividend_yield=1.2,
                pe_ratio=20.0,
                market_cap_bias='mid',
                economic_sensitivity=0.8,
                interest_rate_sensitivity=-0.6,
                inflation_sensitivity=-0.4,
                global_exposure=0.5
            )
        }
        
    def _initialize_rotation_patterns(self) -> Dict:
        """Initialize sector rotation patterns for different market cycles"""
        return {
            MarketCycle.EARLY_BULL: {
                'leaders': ['Banking', 'Auto', 'Real Estate', 'Capital Goods'],
                'laggards': ['FMCG', 'Pharma', 'IT'],
                'duration_months': (3, 8),
                'volatility_multiplier': 1.2
            },
            MarketCycle.LATE_BULL: {
                'leaders': ['IT', 'Pharma', 'FMCG', 'Consumer Durables'],
                'laggards': ['Banking', 'Metals', 'Energy'],
                'duration_months': (6, 12),
                'volatility_multiplier': 1.0
            },
            MarketCycle.EARLY_BEAR: {
                'leaders': ['FMCG', 'Pharma', 'IT'],
                'laggards': ['Banking', 'Auto', 'Real Estate', 'Metals'],
                'duration_months': (2, 6),
                'volatility_multiplier': 1.8
            },
            MarketCycle.LATE_BEAR: {
                'leaders': ['Energy', 'FMCG', 'Pharma'],
                'laggards': ['Banking', 'Infrastructure', 'Real Estate'],
                'duration_months': (3, 9),
                'volatility_multiplier': 2.0
            },
            MarketCycle.RECOVERY: {
                'leaders': ['Banking', 'Auto', 'Infrastructure', 'Metals'],
                'laggards': ['FMCG', 'Pharma'],
                'duration_months': (4, 10),
                'volatility_multiplier': 1.5
            },
            MarketCycle.EXPANSION: {
                'leaders': ['IT', 'Banking', 'Consumer Durables', 'Capital Goods'],
                'laggards': ['Energy', 'Telecom'],
                'duration_months': (6, 18),
                'volatility_multiplier': 1.1
            },
            MarketCycle.PEAK: {
                'leaders': ['FMCG', 'Pharma', 'IT'],
                'laggards': ['Banking', 'Auto', 'Metals'],
                'duration_months': (1, 4),
                'volatility_multiplier': 1.3
            },
            MarketCycle.CONTRACTION: {
                'leaders': ['FMCG', 'Pharma', 'Energy'],
                'laggards': ['Banking', 'Real Estate', 'Auto', 'Metals'],
                'duration_months': (3, 12),
                'volatility_multiplier': 1.7
            }
        }
        
    def _initialize_cycle_characteristics(self) -> Dict:
        """Initialize characteristics for different market cycles"""
        return {
            MarketCycle.EARLY_BULL: {
                'market_return': 0.25,  # 25% annual return
                'market_volatility': 0.18,
                'style_preference': [SectorStyle.VALUE, SectorStyle.CYCLICAL],
                'economic_growth': 0.06,
                'interest_rate_trend': 'falling',
                'inflation_trend': 'stable'
            },
            MarketCycle.LATE_BULL: {
                'market_return': 0.15,
                'market_volatility': 0.15,
                'style_preference': [SectorStyle.GROWTH, SectorStyle.QUALITY],
                'economic_growth': 0.04,
                'interest_rate_trend': 'rising',
                'inflation_trend': 'rising'
            },
            MarketCycle.EARLY_BEAR: {
                'market_return': -0.15,
                'market_volatility': 0.25,
                'style_preference': [SectorStyle.DEFENSIVE, SectorStyle.QUALITY],
                'economic_growth': 0.02,
                'interest_rate_trend': 'rising',
                'inflation_trend': 'high'
            },
            MarketCycle.LATE_BEAR: {
                'market_return': -0.25,
                'market_volatility': 0.30,
                'style_preference': [SectorStyle.DEFENSIVE, SectorStyle.VALUE],
                'economic_growth': -0.01,
                'interest_rate_trend': 'falling',
                'inflation_trend': 'falling'
            },
            MarketCycle.RECOVERY: {
                'market_return': 0.30,
                'market_volatility': 0.22,
                'style_preference': [SectorStyle.CYCLICAL, SectorStyle.VALUE],
                'economic_growth': 0.05,
                'interest_rate_trend': 'stable',
                'inflation_trend': 'stable'
            },
            MarketCycle.EXPANSION: {
                'market_return': 0.20,
                'market_volatility': 0.16,
                'style_preference': [SectorStyle.GROWTH, SectorStyle.MOMENTUM],
                'economic_growth': 0.07,
                'interest_rate_trend': 'stable',
                'inflation_trend': 'stable'
            },
            MarketCycle.PEAK: {
                'market_return': 0.05,
                'market_volatility': 0.20,
                'style_preference': [SectorStyle.DEFENSIVE, SectorStyle.QUALITY],
                'economic_growth': 0.03,
                'interest_rate_trend': 'rising',
                'inflation_trend': 'rising'
            },
            MarketCycle.CONTRACTION: {
                'market_return': -0.20,
                'market_volatility': 0.28,
                'style_preference': [SectorStyle.DEFENSIVE],
                'economic_growth': 0.01,
                'interest_rate_trend': 'falling',
                'inflation_trend': 'falling'
            }
        }
        
    def _initialize_correlation_matrix(self) -> pd.DataFrame:
        """Initialize sector correlation matrix"""
        sectors = list(self.sectors.keys())
        n_sectors = len(sectors)
        
        # Create base correlation matrix
        correlations = np.random.uniform(0.2, 0.7, (n_sectors, n_sectors))
        
        # Make symmetric
        correlations = (correlations + correlations.T) / 2
        
        # Set diagonal to 1
        np.fill_diagonal(correlations, 1.0)
        
        # Adjust correlations based on sector characteristics
        for i, sector1 in enumerate(sectors):
            for j, sector2 in enumerate(sectors):
                if i != j:
                    s1_data = self.sectors[sector1]
                    s2_data = self.sectors[sector2]
                    
                    # Higher correlation for similar styles
                    if s1_data.style == s2_data.style:
                        correlations[i, j] *= 1.3
                    
                    # Higher correlation for similar economic sensitivity
                    econ_diff = abs(s1_data.economic_sensitivity - s2_data.economic_sensitivity)
                    correlations[i, j] *= (1.0 + 0.3 * (1 - econ_diff))
                    
                    # Ensure correlations stay within bounds
                    correlations[i, j] = np.clip(correlations[i, j], 0.1, 0.9)
                    
        return pd.DataFrame(correlations, index=sectors, columns=sectors)
        
    def simulate_sector_rotation(self,
                               start_date: datetime,
                               end_date: datetime,
                               initial_cycle: Optional[MarketCycle] = None) -> Dict:
        """Simulate sector rotation over a time period"""
        
        total_days = (end_date - start_date).days
        
        # Initialize simulation data
        simulation = {
            'start_date': start_date,
            'end_date': end_date,
            'cycles': [],
            'daily_performance': [],
            'sector_weights': [],
            'style_performance': []
        }
        
        current_date = start_date
        current_cycle = initial_cycle or self._select_initial_cycle()
        
        # Generate cycle sequence
        while current_date < end_date:
            cycle_chars = self.cycle_characteristics[current_cycle]
            rotation_pattern = self.rotation_patterns[current_cycle]
            
            # Determine cycle duration
            min_months, max_months = rotation_pattern['duration_months']
            duration_months = np.random.uniform(min_months, max_months)
            duration_days = int(duration_months * 30)
            
            cycle_end_date = min(current_date + timedelta(days=duration_days), end_date)
            
            # Add cycle to simulation
            cycle_data = {
                'cycle': current_cycle,
                'start_date': current_date,
                'end_date': cycle_end_date,
                'characteristics': cycle_chars,
                'rotation_pattern': rotation_pattern
            }
            simulation['cycles'].append(cycle_data)
            
            # Generate daily performance for this cycle
            cycle_performance = self._generate_cycle_performance(
                current_date, cycle_end_date, current_cycle
            )
            simulation['daily_performance'].extend(cycle_performance)
            
            # Move to next cycle
            current_date = cycle_end_date
            if current_date < end_date:
                current_cycle = self._transition_cycle(current_cycle)
                
        # Calculate sector weights over time
        simulation['sector_weights'] = self._calculate_sector_weights(simulation)
        
        # Calculate style performance
        simulation['style_performance'] = self._calculate_style_performance(simulation)
        
        return simulation
        
    def _select_initial_cycle(self) -> MarketCycle:
        """Select initial market cycle"""
        cycle_probabilities = {
            MarketCycle.EXPANSION: 0.25,
            MarketCycle.EARLY_BULL: 0.20,
            MarketCycle.LATE_BULL: 0.15,
            MarketCycle.PEAK: 0.10,
            MarketCycle.EARLY_BEAR: 0.10,
            MarketCycle.LATE_BEAR: 0.10,
            MarketCycle.RECOVERY: 0.10
        }
        
        cycles = list(cycle_probabilities.keys())
        probabilities = list(cycle_probabilities.values())
        
        return np.random.choice(cycles, p=probabilities)
        
    def _transition_cycle(self, current_cycle: MarketCycle) -> MarketCycle:
        """Determine next market cycle"""
        transition_matrix = {
            MarketCycle.EARLY_BULL: {
                MarketCycle.LATE_BULL: 0.6,
                MarketCycle.EXPANSION: 0.3,
                MarketCycle.PEAK: 0.1
            },
            MarketCycle.LATE_BULL: {
                MarketCycle.PEAK: 0.5,
                MarketCycle.EARLY_BEAR: 0.3,
                MarketCycle.LATE_BULL: 0.2
            },
            MarketCycle.PEAK: {
                MarketCycle.EARLY_BEAR: 0.7,
                MarketCycle.CONTRACTION: 0.2,
                MarketCycle.LATE_BULL: 0.1
            },
            MarketCycle.EARLY_BEAR: {
                MarketCycle.LATE_BEAR: 0.5,
                MarketCycle.CONTRACTION: 0.3,
                MarketCycle.RECOVERY: 0.2
            },
            MarketCycle.LATE_BEAR: {
                MarketCycle.RECOVERY: 0.6,
                MarketCycle.CONTRACTION: 0.3,
                MarketCycle.LATE_BEAR: 0.1
            },
            MarketCycle.RECOVERY: {
                MarketCycle.EARLY_BULL: 0.5,
                MarketCycle.EXPANSION: 0.4,
                MarketCycle.RECOVERY: 0.1
            },
            MarketCycle.EXPANSION: {
                MarketCycle.LATE_BULL: 0.4,
                MarketCycle.PEAK: 0.3,
                MarketCycle.EXPANSION: 0.3
            },
            MarketCycle.CONTRACTION: {
                MarketCycle.LATE_BEAR: 0.5,
                MarketCycle.RECOVERY: 0.3,
                MarketCycle.CONTRACTION: 0.2
            }
        }
        
        if current_cycle not in transition_matrix:
            return self._select_initial_cycle()
            
        transitions = transition_matrix[current_cycle]
        next_cycles = list(transitions.keys())
        probabilities = list(transitions.values())
        
        return np.random.choice(next_cycles, p=probabilities)
        
    def _generate_cycle_performance(self,
                                  start_date: datetime,
                                  end_date: datetime,
                                  cycle: MarketCycle) -> List[Dict]:
        """Generate daily sector performance for a market cycle"""
        
        performance_data = []
        current_date = start_date
        
        cycle_chars = self.cycle_characteristics[cycle]
        rotation_pattern = self.rotation_patterns[cycle]
        
        # Base parameters
        market_return = cycle_chars['market_return']
        market_volatility = cycle_chars['market_volatility']
        volatility_multiplier = rotation_pattern['volatility_multiplier']
        
        leaders = rotation_pattern['leaders']
        laggards = rotation_pattern['laggards']
        
        cycle_days = (end_date - start_date).days
        
        while current_date < end_date:
            daily_data = {
                'date': current_date,
                'cycle': cycle,
                'sector_returns': {},
                'sector_volatilities': {},
                'market_return': 0,
                'market_volatility': 0
            }
            
            # Generate market return for the day
            daily_market_return = np.random.normal(
                market_return / 252,  # Daily return
                market_volatility / np.sqrt(252)  # Daily volatility
            )
            
            daily_data['market_return'] = daily_market_return
            daily_data['market_volatility'] = market_volatility * volatility_multiplier
            
            # Generate sector-specific returns
            for sector_name, sector_data in self.sectors.items():
                # Base return from market beta
                base_return = daily_market_return * sector_data.beta
                
                # Cycle-specific adjustments
                if sector_name in leaders:
                    cycle_adjustment = np.random.uniform(0.002, 0.008)  # 0.2-0.8% daily boost
                elif sector_name in laggards:
                    cycle_adjustment = np.random.uniform(-0.008, -0.002)  # 0.2-0.8% daily drag
                else:
                    cycle_adjustment = np.random.uniform(-0.002, 0.002)  # Neutral
                    
                # Style-based adjustments
                style_adjustment = self._calculate_style_adjustment(
                    sector_data.style, cycle_chars['style_preference']
                )
                
                # Economic factor adjustments
                economic_adjustment = self._calculate_economic_adjustment(
                    sector_data, cycle_chars
                )
                
                # Random component
                sector_volatility = market_volatility * volatility_multiplier * (0.8 + 0.4 * sector_data.beta)
                random_component = np.random.normal(0, sector_volatility / np.sqrt(252))
                
                # Final sector return
                sector_return = (
                    base_return + 
                    cycle_adjustment + 
                    style_adjustment + 
                    economic_adjustment + 
                    random_component
                )
                
                daily_data['sector_returns'][sector_name] = sector_return
                daily_data['sector_volatilities'][sector_name] = sector_volatility
                
            performance_data.append(daily_data)
            current_date += timedelta(days=1)
            
        return performance_data
        
    def _calculate_style_adjustment(self,
                                  sector_style: SectorStyle,
                                  preferred_styles: List[SectorStyle]) -> float:
        """Calculate style-based performance adjustment"""
        if sector_style in preferred_styles:
            return np.random.uniform(0.001, 0.003)  # 0.1-0.3% daily boost
        else:
            return np.random.uniform(-0.001, 0.001)  # Neutral to slight drag
            
    def _calculate_economic_adjustment(self,
                                     sector_data: SectorData,
                                     cycle_chars: Dict) -> float:
        """Calculate economic factor-based adjustment"""
        adjustment = 0.0
        
        # Economic growth sensitivity
        growth_impact = sector_data.economic_sensitivity * cycle_chars['economic_growth'] * 0.1
        adjustment += growth_impact
        
        # Interest rate sensitivity
        if cycle_chars['interest_rate_trend'] == 'rising':
            rate_impact = sector_data.interest_rate_sensitivity * 0.002
        elif cycle_chars['interest_rate_trend'] == 'falling':
            rate_impact = sector_data.interest_rate_sensitivity * -0.002
        else:
            rate_impact = 0.0
            
        adjustment += rate_impact
        
        # Inflation sensitivity
        if cycle_chars['inflation_trend'] == 'rising':
            inflation_impact = sector_data.inflation_sensitivity * 0.001
        elif cycle_chars['inflation_trend'] == 'falling':
            inflation_impact = sector_data.inflation_sensitivity * -0.001
        else:
            inflation_impact = 0.0
            
        adjustment += inflation_impact
        
        return adjustment
        
    def _calculate_sector_weights(self, simulation: Dict) -> List[Dict]:
        """Calculate optimal sector weights over time"""
        weights_data = []
        
        for daily_perf in simulation['daily_performance']:
            date = daily_perf['date']
            cycle = daily_perf['cycle']
            
            rotation_pattern = self.rotation_patterns[cycle]
            leaders = rotation_pattern['leaders']
            laggards = rotation_pattern['laggards']
            
            # Calculate weights based on cycle preferences
            weights = {}
            total_weight = 0.0
            
            for sector_name in self.sectors.keys():
                if sector_name in leaders:
                    weight = np.random.uniform(0.12, 0.18)  # Overweight leaders
                elif sector_name in laggards:
                    weight = np.random.uniform(0.02, 0.06)  # Underweight laggards
                else:
                    weight = np.random.uniform(0.06, 0.10)  # Neutral weight
                    
                weights[sector_name] = weight
                total_weight += weight
                
            # Normalize weights to sum to 1
            for sector_name in weights:
                weights[sector_name] /= total_weight
                
            weights_data.append({
                'date': date,
                'cycle': cycle,
                'weights': weights
            })
            
        return weights_data
        
    def _calculate_style_performance(self, simulation: Dict) -> List[Dict]:
        """Calculate performance by investment style"""
        style_data = []
        
        for daily_perf in simulation['daily_performance']:
            date = daily_perf['date']
            sector_returns = daily_perf['sector_returns']
            
            # Group sectors by style
            style_returns = {}
            style_counts = {}
            
            for sector_name, sector_return in sector_returns.items():
                sector_style = self.sectors[sector_name].style
                
                if sector_style not in style_returns:
                    style_returns[sector_style] = 0.0
                    style_counts[sector_style] = 0
                    
                style_returns[sector_style] += sector_return
                style_counts[sector_style] += 1
                
            # Calculate average returns by style
            avg_style_returns = {}
            for style, total_return in style_returns.items():
                avg_style_returns[style.value] = total_return / style_counts[style]
                
            style_data.append({
                'date': date,
                'style_returns': avg_style_returns
            })
            
        return style_data
        
    def generate_sector_heatmap_data(self,
                                   simulation: Dict,
                                   period_days: int = 30) -> pd.DataFrame:
        """Generate sector performance heatmap data"""
        
        daily_performance = simulation['daily_performance']
        
        # Group data by periods
        heatmap_data = []
        
        for i in range(0, len(daily_performance), period_days):
            period_data = daily_performance[i:i+period_days]
            
            if not period_data:
                continue
                
            period_start = period_data[0]['date']
            period_end = period_data[-1]['date']
            
            # Calculate cumulative returns for each sector
            sector_cumulative = {}
            
            for sector_name in self.sectors.keys():
                cumulative_return = 0.0
                
                for daily_data in period_data:
                    daily_return = daily_data['sector_returns'].get(sector_name, 0.0)
                    cumulative_return += daily_return
                    
                sector_cumulative[sector_name] = cumulative_return * 100  # Convert to percentage
                
            heatmap_row = {
                'period_start': period_start,
                'period_end': period_end,
                **sector_cumulative
            }
            
            heatmap_data.append(heatmap_row)
            
        return pd.DataFrame(heatmap_data)
        
    def calculate_sector_correlations(self,
                                    simulation: Dict,
                                    rolling_window: int = 60) -> pd.DataFrame:
        """Calculate rolling sector correlations"""
        
        # Extract sector returns into DataFrame
        returns_data = []
        
        for daily_perf in simulation['daily_performance']:
            row = {'date': daily_perf['date']}
            row.update(daily_perf['sector_returns'])
            returns_data.append(row)
            
        df = pd.DataFrame(returns_data)
        df.set_index('date', inplace=True)
        
        # Calculate rolling correlations
        rolling_corr = df.rolling(window=rolling_window).corr()
        
        return rolling_corr
        
    def get_rotation_summary(self, simulation: Dict) -> Dict:
        """Get summary statistics for sector rotation"""
        
        cycles = simulation['cycles']
        daily_performance = simulation['daily_performance']
        
        # Cycle distribution
        cycle_days = {}
        total_days = 0
        
        for cycle_data in cycles:
            cycle = cycle_data['cycle']
            days = (cycle_data['end_date'] - cycle_data['start_date']).days
            
            if cycle not in cycle_days:
                cycle_days[cycle] = 0
            cycle_days[cycle] += days
            total_days += days
            
        cycle_distribution = {k.value: v/total_days for k, v in cycle_days.items()}
        
        # Sector performance summary
        sector_performance = {}
        
        for sector_name in self.sectors.keys():
            returns = [d['sector_returns'].get(sector_name, 0.0) for d in daily_performance]
            
            sector_performance[sector_name] = {
                'total_return': sum(returns) * 100,
                'volatility': np.std(returns) * np.sqrt(252) * 100,
                'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(returns) * 100
            }
            
        # Style performance summary
        style_performance = {}
        style_data = simulation['style_performance']
        
        for style in SectorStyle:
            style_returns = [d['style_returns'].get(style.value, 0.0) for d in style_data]
            
            style_performance[style.value] = {
                'total_return': sum(style_returns) * 100,
                'volatility': np.std(style_returns) * np.sqrt(252) * 100,
                'sharpe_ratio': np.mean(style_returns) / np.std(style_returns) * np.sqrt(252) if np.std(style_returns) > 0 else 0
            }
            
        return {
            'total_days': total_days,
            'cycle_distribution': cycle_distribution,
            'sector_performance': sector_performance,
            'style_performance': style_performance,
            'number_of_cycles': len(cycles)
        }
        
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
        
    def to_dataframe(self, simulation: Dict) -> pd.DataFrame:
        """Convert simulation results to DataFrame"""
        
        data = []
        
        for daily_perf in simulation['daily_performance']:
            row = {
                'date': daily_perf['date'],
                'cycle': daily_perf['cycle'].value,
                'market_return': daily_perf['market_return'],
                'market_volatility': daily_perf['market_volatility']
            }
            
            # Add sector returns
            for sector, return_val in daily_perf['sector_returns'].items():
                row[f'{sector}_return'] = return_val
                
            # Add sector volatilities
            for sector, vol_val in daily_perf['sector_volatilities'].items():
                row[f'{sector}_volatility'] = vol_val
                
            data.append(row)
            
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('date', inplace=True)
            
        return df

# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = SectorRotationSimulator(seed=42)
    
    # Run simulation
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    simulation = simulator.simulate_sector_rotation(start_date, end_date)
    
    # Get summary
    summary = simulator.get_rotation_summary(simulation)
    
    print("Sector Rotation Simulation Summary:")
    print(f"Duration: {summary['total_days']} days")
    print(f"Number of Cycles: {summary['number_of_cycles']}")
    
    print("\nCycle Distribution:")
    for cycle, pct in summary['cycle_distribution'].items():
        print(f"  {cycle}: {pct:.1%}")
        
    print("\nTop Performing Sectors:")
    sector_perf = summary['sector_performance']
    sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1]['total_return'], reverse=True)
    
    for sector, perf in sorted_sectors[:5]:
        print(f"  {sector}: {perf['total_return']:.1f}% (Vol: {perf['volatility']:.1f}%, Sharpe: {perf['sharpe_ratio']:.2f})")
        
    print("\nStyle Performance:")
    for style, perf in summary['style_performance'].items():
        print(f"  {style}: {perf['total_return']:.1f}% (Vol: {perf['volatility']:.1f}%, Sharpe: {perf['sharpe_ratio']:.2f})")
        
    # Generate heatmap data
    heatmap_df = simulator.generate_sector_heatmap_data(simulation, period_days=30)
    print(f"\nGenerated heatmap data: {heatmap_df.shape}")
    
    # Convert to DataFrame
    df = simulator.to_dataframe(simulation)
    print(f"\nSimulation DataFrame shape: {df.shape}")
    
    # Show sector correlation
    print(f"\nSector Correlations (sample):")
    sector_names = list(simulator.sectors.keys())[:5]
    corr_matrix = simulator.correlation_matrix.loc[sector_names, sector_names]
    print(corr_matrix.round(2))