#!/usr/bin/env python3
"""
Market Conditions Generator - Indian Market Research Platform
Generates various market condition scenarios for testing and simulation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random

class MarketRegime(Enum):
    """Market regime types"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    BUBBLE = "bubble"
    CORRECTION = "correction"

class MarketEvent(Enum):
    """Market event types"""
    EARNINGS_SEASON = "earnings_season"
    RBI_POLICY = "rbi_policy"
    BUDGET_ANNOUNCEMENT = "budget_announcement"
    ELECTION = "election"
    GLOBAL_CRISIS = "global_crisis"
    SECTOR_NEWS = "sector_news"
    FII_FLOW = "fii_flow"
    CURRENCY_SHOCK = "currency_shock"
    OIL_PRICE_SHOCK = "oil_price_shock"
    MONSOON_UPDATE = "monsoon_update"

class MarketConditionsGenerator:
    """
    Generates various market conditions and scenarios for simulation
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
            
        self.regime_parameters = self._initialize_regime_parameters()
        self.event_parameters = self._initialize_event_parameters()
        self.sector_correlations = self._initialize_sector_correlations()
        
    def _initialize_regime_parameters(self) -> Dict:
        """Initialize parameters for different market regimes"""
        return {
            MarketRegime.BULL_MARKET: {
                'drift': 0.12,  # 12% annual return
                'volatility': 0.15,  # 15% volatility
                'trend_strength': 0.8,
                'mean_reversion': 0.1,
                'duration_days': (180, 720)  # 6 months to 2 years
            },
            MarketRegime.BEAR_MARKET: {
                'drift': -0.20,  # -20% annual return
                'volatility': 0.25,  # 25% volatility
                'trend_strength': 0.7,
                'mean_reversion': 0.2,
                'duration_days': (90, 540)  # 3 months to 1.5 years
            },
            MarketRegime.SIDEWAYS: {
                'drift': 0.02,  # 2% annual return
                'volatility': 0.12,  # 12% volatility
                'trend_strength': 0.2,
                'mean_reversion': 0.6,
                'duration_days': (120, 360)  # 4 months to 1 year
            },
            MarketRegime.HIGH_VOLATILITY: {
                'drift': 0.05,
                'volatility': 0.35,  # 35% volatility
                'trend_strength': 0.3,
                'mean_reversion': 0.4,
                'duration_days': (30, 180)  # 1 to 6 months
            },
            MarketRegime.LOW_VOLATILITY: {
                'drift': 0.08,
                'volatility': 0.08,  # 8% volatility
                'trend_strength': 0.5,
                'mean_reversion': 0.3,
                'duration_days': (60, 300)  # 2 to 10 months
            },
            MarketRegime.CRISIS: {
                'drift': -0.40,  # -40% annual return
                'volatility': 0.50,  # 50% volatility
                'trend_strength': 0.9,
                'mean_reversion': 0.1,
                'duration_days': (30, 120)  # 1 to 4 months
            },
            MarketRegime.RECOVERY: {
                'drift': 0.25,  # 25% annual return
                'volatility': 0.20,  # 20% volatility
                'trend_strength': 0.8,
                'mean_reversion': 0.2,
                'duration_days': (90, 360)  # 3 months to 1 year
            },
            MarketRegime.BUBBLE: {
                'drift': 0.30,  # 30% annual return
                'volatility': 0.18,  # 18% volatility
                'trend_strength': 0.9,
                'mean_reversion': 0.05,
                'duration_days': (180, 540)  # 6 months to 1.5 years
            },
            MarketRegime.CORRECTION: {
                'drift': -0.15,  # -15% annual return
                'volatility': 0.22,  # 22% volatility
                'trend_strength': 0.7,
                'mean_reversion': 0.3,
                'duration_days': (20, 90)  # 3 weeks to 3 months
            }
        }
        
    def _initialize_event_parameters(self) -> Dict:
        """Initialize parameters for market events"""
        return {
            MarketEvent.EARNINGS_SEASON: {
                'frequency': 4,  # 4 times per year
                'duration_days': 45,
                'volatility_multiplier': 1.3,
                'sector_impact': {
                    'IT': 1.5,
                    'Banking': 1.4,
                    'Pharma': 1.2,
                    'Auto': 1.3
                }
            },
            MarketEvent.RBI_POLICY: {
                'frequency': 6,  # 6 times per year
                'duration_days': 3,
                'volatility_multiplier': 1.8,
                'sector_impact': {
                    'Banking': 2.0,
                    'NBFC': 1.8,
                    'Real Estate': 1.5
                }
            },
            MarketEvent.BUDGET_ANNOUNCEMENT: {
                'frequency': 1,  # Once per year
                'duration_days': 7,
                'volatility_multiplier': 2.0,
                'sector_impact': {
                    'Infrastructure': 1.8,
                    'Defense': 1.6,
                    'Railways': 1.7
                }
            },
            MarketEvent.ELECTION: {
                'frequency': 0.2,  # Once every 5 years
                'duration_days': 60,
                'volatility_multiplier': 1.5,
                'sector_impact': {
                    'PSU': 1.8,
                    'Infrastructure': 1.6,
                    'Defense': 1.5
                }
            },
            MarketEvent.GLOBAL_CRISIS: {
                'frequency': 0.1,  # Once every 10 years
                'duration_days': 180,
                'volatility_multiplier': 3.0,
                'sector_impact': {
                    'IT': 0.7,  # Defensive
                    'Pharma': 0.8,  # Defensive
                    'Banking': 1.5,
                    'Auto': 1.8
                }
            },
            MarketEvent.FII_FLOW: {
                'frequency': 12,  # Monthly
                'duration_days': 5,
                'volatility_multiplier': 1.2,
                'sector_impact': {
                    'Large Cap': 1.3,
                    'IT': 1.4,
                    'Banking': 1.2
                }
            },
            MarketEvent.CURRENCY_SHOCK: {
                'frequency': 2,  # Twice per year
                'duration_days': 10,
                'volatility_multiplier': 1.6,
                'sector_impact': {
                    'IT': 0.8,  # Benefits from weak rupee
                    'Pharma': 0.9,
                    'Oil & Gas': 1.5,  # Hurt by weak rupee
                    'Airlines': 1.4
                }
            },
            MarketEvent.OIL_PRICE_SHOCK: {
                'frequency': 1,  # Once per year
                'duration_days': 30,
                'volatility_multiplier': 1.4,
                'sector_impact': {
                    'Oil & Gas': 1.8,
                    'Airlines': 1.6,
                    'Auto': 1.3,
                    'Chemicals': 1.4
                }
            },
            MarketEvent.MONSOON_UPDATE: {
                'frequency': 2,  # Twice per year
                'duration_days': 7,
                'volatility_multiplier': 1.2,
                'sector_impact': {
                    'FMCG': 1.3,
                    'Agriculture': 1.8,
                    'Fertilizers': 1.5
                }
            }
        }
        
    def _initialize_sector_correlations(self) -> Dict:
        """Initialize sector correlation matrix"""
        sectors = ['Banking', 'IT', 'Pharma', 'Auto', 'FMCG', 'Energy', 'Metals', 'Infrastructure']
        
        # Create correlation matrix
        correlation_matrix = np.random.uniform(0.3, 0.8, (len(sectors), len(sectors)))
        
        # Make it symmetric
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        
        # Set diagonal to 1
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return pd.DataFrame(correlation_matrix, index=sectors, columns=sectors)
        
    def generate_market_scenario(self, 
                               start_date: datetime,
                               end_date: datetime,
                               initial_regime: Optional[MarketRegime] = None) -> Dict:
        """Generate a complete market scenario with regime changes and events"""
        
        total_days = (end_date - start_date).days
        scenario = {
            'start_date': start_date,
            'end_date': end_date,
            'regimes': [],
            'events': [],
            'market_parameters': []
        }
        
        current_date = start_date
        current_regime = initial_regime or self._select_initial_regime()
        
        while current_date < end_date:
            # Determine regime duration
            regime_params = self.regime_parameters[current_regime]
            min_duration, max_duration = regime_params['duration_days']
            duration = np.random.randint(min_duration, max_duration + 1)
            
            regime_end_date = min(current_date + timedelta(days=duration), end_date)
            
            # Add regime to scenario
            scenario['regimes'].append({
                'regime': current_regime,
                'start_date': current_date,
                'end_date': regime_end_date,
                'parameters': regime_params
            })
            
            # Generate events during this regime
            regime_events = self._generate_events_for_period(current_date, regime_end_date)
            scenario['events'].extend(regime_events)
            
            # Generate daily market parameters
            daily_params = self._generate_daily_parameters(
                current_date, regime_end_date, current_regime, regime_events
            )
            scenario['market_parameters'].extend(daily_params)
            
            # Move to next regime
            current_date = regime_end_date
            if current_date < end_date:
                current_regime = self._transition_regime(current_regime)
                
        return scenario
        
    def _select_initial_regime(self) -> MarketRegime:
        """Select initial market regime based on probabilities"""
        regime_probabilities = {
            MarketRegime.BULL_MARKET: 0.25,
            MarketRegime.BEAR_MARKET: 0.15,
            MarketRegime.SIDEWAYS: 0.30,
            MarketRegime.HIGH_VOLATILITY: 0.10,
            MarketRegime.LOW_VOLATILITY: 0.15,
            MarketRegime.CRISIS: 0.02,
            MarketRegime.RECOVERY: 0.03
        }
        
        regimes = list(regime_probabilities.keys())
        probabilities = list(regime_probabilities.values())
        
        return np.random.choice(regimes, p=probabilities)
        
    def _transition_regime(self, current_regime: MarketRegime) -> MarketRegime:
        """Determine next regime based on current regime"""
        transition_matrix = {
            MarketRegime.BULL_MARKET: {
                MarketRegime.BULL_MARKET: 0.4,
                MarketRegime.SIDEWAYS: 0.3,
                MarketRegime.CORRECTION: 0.2,
                MarketRegime.HIGH_VOLATILITY: 0.1
            },
            MarketRegime.BEAR_MARKET: {
                MarketRegime.BEAR_MARKET: 0.3,
                MarketRegime.RECOVERY: 0.4,
                MarketRegime.SIDEWAYS: 0.2,
                MarketRegime.CRISIS: 0.1
            },
            MarketRegime.SIDEWAYS: {
                MarketRegime.SIDEWAYS: 0.4,
                MarketRegime.BULL_MARKET: 0.25,
                MarketRegime.BEAR_MARKET: 0.15,
                MarketRegime.HIGH_VOLATILITY: 0.2
            },
            MarketRegime.HIGH_VOLATILITY: {
                MarketRegime.LOW_VOLATILITY: 0.4,
                MarketRegime.SIDEWAYS: 0.3,
                MarketRegime.CRISIS: 0.15,
                MarketRegime.HIGH_VOLATILITY: 0.15
            },
            MarketRegime.LOW_VOLATILITY: {
                MarketRegime.LOW_VOLATILITY: 0.3,
                MarketRegime.BULL_MARKET: 0.3,
                MarketRegime.SIDEWAYS: 0.25,
                MarketRegime.HIGH_VOLATILITY: 0.15
            },
            MarketRegime.CRISIS: {
                MarketRegime.RECOVERY: 0.6,
                MarketRegime.BEAR_MARKET: 0.3,
                MarketRegime.CRISIS: 0.1
            },
            MarketRegime.RECOVERY: {
                MarketRegime.BULL_MARKET: 0.5,
                MarketRegime.SIDEWAYS: 0.3,
                MarketRegime.RECOVERY: 0.2
            },
            MarketRegime.BUBBLE: {
                MarketRegime.CORRECTION: 0.4,
                MarketRegime.BEAR_MARKET: 0.3,
                MarketRegime.BUBBLE: 0.2,
                MarketRegime.CRISIS: 0.1
            },
            MarketRegime.CORRECTION: {
                MarketRegime.RECOVERY: 0.4,
                MarketRegime.SIDEWAYS: 0.3,
                MarketRegime.BEAR_MARKET: 0.2,
                MarketRegime.CORRECTION: 0.1
            }
        }
        
        if current_regime not in transition_matrix:
            return self._select_initial_regime()
            
        transitions = transition_matrix[current_regime]
        next_regimes = list(transitions.keys())
        probabilities = list(transitions.values())
        
        return np.random.choice(next_regimes, p=probabilities)
        
    def _generate_events_for_period(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate market events for a given period"""
        events = []
        period_days = (end_date - start_date).days
        
        for event_type, params in self.event_parameters.items():
            # Calculate expected number of events in this period
            annual_frequency = params['frequency']
            expected_events = (annual_frequency * period_days) / 365
            
            # Generate actual number of events (Poisson distribution)
            num_events = np.random.poisson(expected_events)
            
            for _ in range(num_events):
                # Random date within the period
                random_days = np.random.randint(0, period_days)
                event_date = start_date + timedelta(days=random_days)
                
                event = {
                    'type': event_type,
                    'date': event_date,
                    'duration_days': params['duration_days'],
                    'volatility_multiplier': params['volatility_multiplier'],
                    'sector_impact': params['sector_impact']
                }
                
                events.append(event)
                
        # Sort events by date
        events.sort(key=lambda x: x['date'])
        
        return events
        
    def _generate_daily_parameters(self, 
                                 start_date: datetime, 
                                 end_date: datetime,
                                 regime: MarketRegime,
                                 events: List[Dict]) -> List[Dict]:
        """Generate daily market parameters"""
        daily_params = []
        current_date = start_date
        base_params = self.regime_parameters[regime]
        
        while current_date < end_date:
            # Base parameters from regime
            daily_drift = base_params['drift'] / 365  # Daily drift
            daily_volatility = base_params['volatility'] / np.sqrt(365)  # Daily volatility
            
            # Adjust for active events
            volatility_multiplier = 1.0
            sector_multipliers = {}
            
            for event in events:
                event_start = event['date']
                event_end = event_start + timedelta(days=event['duration_days'])
                
                if event_start <= current_date <= event_end:
                    volatility_multiplier *= event['volatility_multiplier']
                    
                    # Apply sector-specific impacts
                    for sector, impact in event['sector_impact'].items():
                        if sector not in sector_multipliers:
                            sector_multipliers[sector] = 1.0
                        sector_multipliers[sector] *= impact
                        
            # Apply multipliers
            adjusted_volatility = daily_volatility * volatility_multiplier
            
            daily_param = {
                'date': current_date,
                'regime': regime,
                'base_drift': daily_drift,
                'base_volatility': daily_volatility,
                'adjusted_volatility': adjusted_volatility,
                'volatility_multiplier': volatility_multiplier,
                'sector_multipliers': sector_multipliers,
                'trend_strength': base_params['trend_strength'],
                'mean_reversion': base_params['mean_reversion']
            }
            
            daily_params.append(daily_param)
            current_date += timedelta(days=1)
            
        return daily_params
        
    def generate_stress_test_scenarios(self) -> List[Dict]:
        """Generate stress test scenarios"""
        scenarios = []
        
        # Scenario 1: Market Crash
        scenarios.append({
            'name': 'Market Crash',
            'description': 'Sudden 30% market decline over 2 weeks',
            'regime': MarketRegime.CRISIS,
            'duration_days': 14,
            'parameters': {
                'drift': -0.80,  # -80% annualized
                'volatility': 0.60,
                'trend_strength': 0.95
            }
        })
        
        # Scenario 2: Currency Crisis
        scenarios.append({
            'name': 'Currency Crisis',
            'description': 'Rupee devaluation by 20% in a month',
            'regime': MarketRegime.HIGH_VOLATILITY,
            'duration_days': 30,
            'parameters': {
                'drift': -0.25,
                'volatility': 0.40,
                'sector_impacts': {
                    'IT': 0.7,  # Benefits
                    'Oil & Gas': 1.8,  # Hurt
                    'Airlines': 1.6
                }
            }
        })
        
        # Scenario 3: Banking Crisis
        scenarios.append({
            'name': 'Banking Crisis',
            'description': 'Major banking sector stress',
            'regime': MarketRegime.BEAR_MARKET,
            'duration_days': 90,
            'parameters': {
                'drift': -0.30,
                'volatility': 0.35,
                'sector_impacts': {
                    'Banking': 2.0,
                    'NBFC': 1.8,
                    'Real Estate': 1.5
                }
            }
        })
        
        # Scenario 4: Global Recession
        scenarios.append({
            'name': 'Global Recession',
            'description': 'Worldwide economic downturn',
            'regime': MarketRegime.BEAR_MARKET,
            'duration_days': 180,
            'parameters': {
                'drift': -0.35,
                'volatility': 0.30,
                'sector_impacts': {
                    'IT': 1.4,  # Export dependent
                    'Auto': 1.6,
                    'Metals': 1.8
                }
            }
        })
        
        # Scenario 5: Interest Rate Shock
        scenarios.append({
            'name': 'Interest Rate Shock',
            'description': 'Sudden 200 bps rate hike',
            'regime': MarketRegime.HIGH_VOLATILITY,
            'duration_days': 60,
            'parameters': {
                'drift': -0.15,
                'volatility': 0.25,
                'sector_impacts': {
                    'Banking': 0.8,  # Benefits
                    'Real Estate': 1.6,  # Hurt
                    'Auto': 1.4
                }
            }
        })
        
        return scenarios
        
    def generate_sector_rotation_scenario(self, duration_days: int = 252) -> Dict:
        """Generate sector rotation scenario"""
        sectors = ['Banking', 'IT', 'Pharma', 'Auto', 'FMCG', 'Energy', 'Metals']
        
        # Define rotation phases
        phases = [
            {'name': 'Growth Phase', 'leaders': ['IT', 'Pharma'], 'duration_pct': 0.3},
            {'name': 'Value Phase', 'leaders': ['Banking', 'Energy'], 'duration_pct': 0.25},
            {'name': 'Defensive Phase', 'leaders': ['FMCG', 'Pharma'], 'duration_pct': 0.2},
            {'name': 'Cyclical Phase', 'leaders': ['Auto', 'Metals'], 'duration_pct': 0.25}
        ]
        
        rotation_schedule = []
        current_day = 0
        
        for phase in phases:
            phase_duration = int(duration_days * phase['duration_pct'])
            
            # Generate daily sector performance multipliers
            for day in range(phase_duration):
                daily_multipliers = {}
                
                for sector in sectors:
                    if sector in phase['leaders']:
                        # Leading sectors get positive bias
                        multiplier = np.random.uniform(1.1, 1.3)
                    else:
                        # Lagging sectors get negative bias
                        multiplier = np.random.uniform(0.8, 1.0)
                        
                    daily_multipliers[sector] = multiplier
                    
                rotation_schedule.append({
                    'day': current_day + day,
                    'phase': phase['name'],
                    'sector_multipliers': daily_multipliers
                })
                
            current_day += phase_duration
            
        return {
            'name': 'Sector Rotation',
            'duration_days': duration_days,
            'phases': phases,
            'daily_schedule': rotation_schedule
        }
        
    def get_market_condition_summary(self, scenario: Dict) -> Dict:
        """Get summary statistics for a market scenario"""
        regimes = scenario['regimes']
        events = scenario['events']
        
        # Regime distribution
        regime_days = {}
        total_days = 0
        
        for regime_period in regimes:
            regime = regime_period['regime']
            days = (regime_period['end_date'] - regime_period['start_date']).days
            
            if regime not in regime_days:
                regime_days[regime] = 0
            regime_days[regime] += days
            total_days += days
            
        regime_distribution = {k: v/total_days for k, v in regime_days.items()}
        
        # Event frequency
        event_counts = {}
        for event in events:
            event_type = event['type']
            if event_type not in event_counts:
                event_counts[event_type] = 0
            event_counts[event_type] += 1
            
        # Calculate expected returns and volatility
        total_drift = 0
        total_volatility = 0
        
        for regime_period in regimes:
            regime = regime_period['regime']
            days = (regime_period['end_date'] - regime_period['start_date']).days
            weight = days / total_days
            
            params = self.regime_parameters[regime]
            total_drift += params['drift'] * weight
            total_volatility += params['volatility'] * weight
            
        return {
            'total_days': total_days,
            'regime_distribution': regime_distribution,
            'event_counts': event_counts,
            'expected_annual_return': total_drift,
            'expected_annual_volatility': total_volatility,
            'sharpe_ratio': total_drift / total_volatility if total_volatility > 0 else 0,
            'number_of_regime_changes': len(regimes),
            'total_events': len(events)
        }

# Example usage
if __name__ == "__main__":
    # Create generator
    generator = MarketConditionsGenerator(seed=42)
    
    # Generate a 2-year scenario
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    scenario = generator.generate_market_scenario(start_date, end_date)
    
    # Print summary
    summary = generator.get_market_condition_summary(scenario)
    print("Market Scenario Summary:")
    print(f"Duration: {summary['total_days']} days")
    print(f"Expected Annual Return: {summary['expected_annual_return']:.2%}")
    print(f"Expected Annual Volatility: {summary['expected_annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Number of Regime Changes: {summary['number_of_regime_changes']}")
    print(f"Total Events: {summary['total_events']}")
    
    print("\nRegime Distribution:")
    for regime, pct in summary['regime_distribution'].items():
        print(f"  {regime.value}: {pct:.1%}")
        
    print("\nEvent Counts:")
    for event_type, count in summary['event_counts'].items():
        print(f"  {event_type.value}: {count}")
        
    # Generate stress test scenarios
    stress_scenarios = generator.generate_stress_test_scenarios()
    print(f"\nGenerated {len(stress_scenarios)} stress test scenarios")
    
    # Generate sector rotation scenario
    rotation_scenario = generator.generate_sector_rotation_scenario()
    print(f"\nGenerated sector rotation scenario with {len(rotation_scenario['phases'])} phases")