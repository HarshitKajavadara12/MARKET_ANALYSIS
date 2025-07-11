#!/usr/bin/env python3
"""
Intraday Data Generator - Indian Market Research Platform
Generates realistic minute-by-minute market data for Indian stocks
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass
from enum import Enum

class SessionType(Enum):
    """Trading session types"""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    POST_MARKET = "post_market"
    CLOSED = "closed"

@dataclass
class MarketHours:
    """Market trading hours configuration"""
    pre_market_start: time = time(9, 0)   # 9:00 AM
    pre_market_end: time = time(9, 15)    # 9:15 AM
    regular_start: time = time(9, 15)     # 9:15 AM
    regular_end: time = time(15, 30)      # 3:30 PM
    post_market_start: time = time(15, 40) # 3:40 PM
    post_market_end: time = time(16, 0)   # 4:00 PM

@dataclass
class IntradayTick:
    """Single intraday tick data"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    session_type: SessionType
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_trade_price: float
    last_trade_quantity: int
    total_traded_value: float
    
class IntradayDataGenerator:
    """
    Generates realistic minute-by-minute intraday data for Indian stocks
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
            
        self.market_hours = MarketHours()
        self.session_characteristics = self._initialize_session_characteristics()
        self.volume_profiles = self._initialize_volume_profiles()
        
    def _initialize_session_characteristics(self) -> Dict:
        """Initialize characteristics for different trading sessions"""
        return {
            SessionType.PRE_MARKET: {
                'volatility_multiplier': 0.8,
                'volume_multiplier': 0.3,
                'spread_multiplier': 2.0,
                'trend_strength': 0.6
            },
            SessionType.REGULAR: {
                'volatility_multiplier': 1.0,
                'volume_multiplier': 1.0,
                'spread_multiplier': 1.0,
                'trend_strength': 1.0
            },
            SessionType.POST_MARKET: {
                'volatility_multiplier': 0.6,
                'volume_multiplier': 0.2,
                'spread_multiplier': 1.5,
                'trend_strength': 0.4
            }
        }
        
    def _initialize_volume_profiles(self) -> Dict:
        """Initialize intraday volume profiles"""
        # Create U-shaped volume profile (high at open/close, low at lunch)
        regular_minutes = 375  # 9:15 AM to 3:30 PM
        time_points = np.linspace(0, 1, regular_minutes)
        
        # U-shaped curve with lunch dip
        volume_curve = (
            2.0 * np.exp(-5 * time_points) +  # Opening spike
            1.5 * np.exp(-5 * (1 - time_points)) +  # Closing spike
            0.3 * np.exp(-20 * (time_points - 0.5)**2) +  # Lunch dip (inverted)
            0.5  # Base volume
        )
        
        # Normalize
        volume_curve = volume_curve / np.mean(volume_curve)
        
        return {
            'regular_session': volume_curve,
            'pre_market': np.ones(15) * 0.3,  # 15 minutes, low volume
            'post_market': np.ones(20) * 0.2   # 20 minutes, very low volume
        }
        
    def generate_intraday_data(self, 
                             symbol: str,
                             date: datetime,
                             previous_close: float,
                             daily_return: float = 0.0,
                             volatility: float = 0.02,
                             average_volume: int = 100000) -> List[IntradayTick]:
        """Generate complete intraday data for a symbol"""
        
        ticks = []
        current_price = previous_close
        cumulative_volume = 0
        cumulative_value = 0.0
        
        # Calculate target close price
        target_close = previous_close * (1 + daily_return)
        
        # Generate data for each session
        sessions = [
            (SessionType.PRE_MARKET, self.market_hours.pre_market_start, self.market_hours.pre_market_end),
            (SessionType.REGULAR, self.market_hours.regular_start, self.market_hours.regular_end),
            (SessionType.POST_MARKET, self.market_hours.post_market_start, self.market_hours.post_market_end)
        ]
        
        for session_type, start_time, end_time in sessions:
            session_ticks = self._generate_session_data(
                symbol=symbol,
                date=date,
                session_type=session_type,
                start_time=start_time,
                end_time=end_time,
                initial_price=current_price,
                target_price=target_close if session_type == SessionType.REGULAR else current_price,
                volatility=volatility,
                average_volume=average_volume,
                cumulative_volume=cumulative_volume,
                cumulative_value=cumulative_value
            )
            
            ticks.extend(session_ticks)
            
            if session_ticks:
                current_price = session_ticks[-1].close
                cumulative_volume = session_ticks[-1].volume
                cumulative_value = session_ticks[-1].total_traded_value
                
        return ticks
        
    def _generate_session_data(self,
                             symbol: str,
                             date: datetime,
                             session_type: SessionType,
                             start_time: time,
                             end_time: time,
                             initial_price: float,
                             target_price: float,
                             volatility: float,
                             average_volume: int,
                             cumulative_volume: int,
                             cumulative_value: float) -> List[IntradayTick]:
        """Generate data for a specific trading session"""
        
        ticks = []
        
        # Calculate session duration in minutes
        start_datetime = datetime.combine(date, start_time)
        end_datetime = datetime.combine(date, end_time)
        
        if end_datetime <= start_datetime:
            return ticks
            
        total_minutes = int((end_datetime - start_datetime).total_seconds() / 60)
        
        if total_minutes <= 0:
            return ticks
            
        # Get session characteristics
        session_chars = self.session_characteristics[session_type]
        
        # Adjust parameters for session
        session_volatility = volatility * session_chars['volatility_multiplier']
        session_volume_mult = session_chars['volume_multiplier']
        spread_mult = session_chars['spread_multiplier']
        trend_strength = session_chars['trend_strength']
        
        # Get volume profile for session
        if session_type == SessionType.REGULAR:
            volume_profile = self.volume_profiles['regular_session'][:total_minutes]
        elif session_type == SessionType.PRE_MARKET:
            volume_profile = self.volume_profiles['pre_market'][:total_minutes]
        else:
            volume_profile = self.volume_profiles['post_market'][:total_minutes]
            
        # Ensure volume profile matches session length
        if len(volume_profile) != total_minutes:
            volume_profile = np.interp(
                np.linspace(0, 1, total_minutes),
                np.linspace(0, 1, len(volume_profile)),
                volume_profile
            )
            
        # Generate price path
        price_path = self._generate_price_path(
            initial_price=initial_price,
            target_price=target_price,
            num_steps=total_minutes,
            volatility=session_volatility,
            trend_strength=trend_strength
        )
        
        # Generate ticks
        current_volume = cumulative_volume
        current_value = cumulative_value
        
        for i in range(total_minutes):
            timestamp = start_datetime + timedelta(minutes=i)
            
            # Price data
            if i == 0:
                open_price = initial_price
            else:
                open_price = ticks[-1].close
                
            close_price = price_path[i]
            
            # Generate high/low with some randomness
            price_range = abs(close_price - open_price)
            high_extension = np.random.exponential(price_range * 0.3)
            low_extension = np.random.exponential(price_range * 0.3)
            
            high_price = max(open_price, close_price) + high_extension
            low_price = min(open_price, close_price) - low_extension
            
            # Volume data
            base_volume = int(average_volume * session_volume_mult / total_minutes)
            volume_multiplier = volume_profile[i] if i < len(volume_profile) else 1.0
            minute_volume = max(1, int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)))
            
            current_volume += minute_volume
            minute_value = minute_volume * close_price
            current_value += minute_value
            
            # Market depth (bid/ask)
            spread_pct = 0.001 * spread_mult  # 0.1% base spread
            spread = close_price * spread_pct
            
            bid = close_price - spread / 2
            ask = close_price + spread / 2
            
            # Bid/ask sizes
            bid_size = int(minute_volume * np.random.uniform(0.3, 0.8))
            ask_size = int(minute_volume * np.random.uniform(0.3, 0.8))
            
            # Last trade data
            last_trade_price = close_price + np.random.uniform(-spread/4, spread/4)
            last_trade_quantity = int(minute_volume * np.random.uniform(0.1, 0.3))
            
            tick = IntradayTick(
                timestamp=timestamp,
                symbol=symbol,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=current_volume,
                session_type=session_type,
                bid=round(bid, 2),
                ask=round(ask, 2),
                bid_size=bid_size,
                ask_size=ask_size,
                last_trade_price=round(last_trade_price, 2),
                last_trade_quantity=last_trade_quantity,
                total_traded_value=round(current_value, 2)
            )
            
            ticks.append(tick)
            
        return ticks
        
    def _generate_price_path(self,
                           initial_price: float,
                           target_price: float,
                           num_steps: int,
                           volatility: float,
                           trend_strength: float) -> np.ndarray:
        """Generate realistic price path using mean-reverting process with trend"""
        
        if num_steps <= 0:
            return np.array([initial_price])
            
        # Calculate drift towards target
        total_drift = np.log(target_price / initial_price)
        drift_per_step = total_drift / num_steps
        
        # Generate random shocks
        dt = 1.0 / (252 * 375)  # Minute fraction of trading year
        random_shocks = np.random.normal(0, volatility * np.sqrt(dt), num_steps)
        
        # Mean reversion parameters
        mean_reversion_speed = 0.1 * (1 - trend_strength)
        
        # Initialize price path
        log_prices = np.zeros(num_steps)
        log_prices[0] = np.log(initial_price)
        
        target_log_price = np.log(target_price)
        
        for i in range(1, num_steps):
            # Current log price
            current_log_price = log_prices[i-1]
            
            # Trend component
            trend_component = drift_per_step * trend_strength
            
            # Mean reversion component
            mean_reversion_component = mean_reversion_speed * (target_log_price - current_log_price) * dt
            
            # Random component
            random_component = random_shocks[i]
            
            # Update log price
            log_prices[i] = current_log_price + trend_component + mean_reversion_component + random_component
            
        # Convert back to prices
        prices = np.exp(log_prices)
        
        return prices
        
    def generate_market_depth(self, 
                            current_price: float, 
                            spread_pct: float = 0.001,
                            depth_levels: int = 5) -> Dict:
        """Generate market depth (order book) data"""
        
        spread = current_price * spread_pct
        
        # Generate bid levels
        bid_levels = []
        for i in range(depth_levels):
            level_price = current_price - spread * (0.5 + i * 0.5)
            level_quantity = int(np.random.exponential(1000) * (depth_levels - i))
            bid_levels.append({
                'price': round(level_price, 2),
                'quantity': level_quantity,
                'orders': np.random.randint(1, 10)
            })
            
        # Generate ask levels
        ask_levels = []
        for i in range(depth_levels):
            level_price = current_price + spread * (0.5 + i * 0.5)
            level_quantity = int(np.random.exponential(1000) * (depth_levels - i))
            ask_levels.append({
                'price': round(level_price, 2),
                'quantity': level_quantity,
                'orders': np.random.randint(1, 10)
            })
            
        return {
            'bids': bid_levels,
            'asks': ask_levels,
            'spread': round(spread, 2),
            'spread_pct': round(spread_pct * 100, 3)
        }
        
    def generate_trade_ticks(self,
                           symbol: str,
                           timestamp: datetime,
                           price_range: Tuple[float, float],
                           num_trades: int = 10) -> List[Dict]:
        """Generate individual trade ticks within a minute"""
        
        trades = []
        min_price, max_price = price_range
        
        for i in range(num_trades):
            # Random time within the minute
            seconds_offset = np.random.randint(0, 60)
            trade_time = timestamp + timedelta(seconds=seconds_offset)
            
            # Random price within range
            trade_price = np.random.uniform(min_price, max_price)
            
            # Random quantity (log-normal distribution)
            trade_quantity = max(1, int(np.random.lognormal(4, 1)))
            
            # Trade type (buy/sell based on price movement)
            if trade_price > (min_price + max_price) / 2:
                trade_type = 'BUY'
            else:
                trade_type = 'SELL'
                
            trade = {
                'symbol': symbol,
                'timestamp': trade_time,
                'price': round(trade_price, 2),
                'quantity': trade_quantity,
                'value': round(trade_price * trade_quantity, 2),
                'type': trade_type,
                'trade_id': f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{i:03d}"
            }
            
            trades.append(trade)
            
        # Sort by timestamp
        trades.sort(key=lambda x: x['timestamp'])
        
        return trades
        
    def generate_circuit_breaker_scenario(self,
                                        symbol: str,
                                        date: datetime,
                                        trigger_time: time,
                                        direction: str = 'up',
                                        halt_duration: int = 15) -> List[IntradayTick]:
        """Generate circuit breaker scenario"""
        
        # Circuit breaker limits (5%, 10%, 20% for different categories)
        cb_limits = {
            'lower': [0.05, 0.10, 0.20],
            'upper': [0.05, 0.10, 0.20]
        }
        
        # Simulate price movement leading to circuit breaker
        base_price = 100.0
        limit_pct = cb_limits['upper'][0] if direction == 'up' else cb_limits['lower'][0]
        
        if direction == 'up':
            target_price = base_price * (1 + limit_pct)
        else:
            target_price = base_price * (1 - limit_pct)
            
        # Generate rapid price movement
        trigger_datetime = datetime.combine(date, trigger_time)
        
        # Pre-trigger normal trading
        pre_trigger_ticks = self._generate_session_data(
            symbol=symbol,
            date=date,
            session_type=SessionType.REGULAR,
            start_time=self.market_hours.regular_start,
            end_time=trigger_time,
            initial_price=base_price,
            target_price=base_price * 1.02,  # Small movement
            volatility=0.01,
            average_volume=50000,
            cumulative_volume=0,
            cumulative_value=0.0
        )
        
        # Circuit breaker trigger (rapid movement)
        cb_tick = IntradayTick(
            timestamp=trigger_datetime,
            symbol=symbol,
            open=base_price,
            high=target_price if direction == 'up' else base_price,
            low=target_price if direction == 'down' else base_price,
            close=target_price,
            volume=100000,  # High volume
            session_type=SessionType.REGULAR,
            bid=target_price - 0.05,
            ask=target_price + 0.05,
            bid_size=0,  # No liquidity
            ask_size=0,
            last_trade_price=target_price,
            last_trade_quantity=1000,
            total_traded_value=target_price * 100000
        )
        
        # Halt period (no trading)
        halt_ticks = []
        for i in range(halt_duration):
            halt_time = trigger_datetime + timedelta(minutes=i+1)
            halt_tick = IntradayTick(
                timestamp=halt_time,
                symbol=symbol,
                open=target_price,
                high=target_price,
                low=target_price,
                close=target_price,
                volume=0,  # No volume during halt
                session_type=SessionType.REGULAR,
                bid=0,
                ask=0,
                bid_size=0,
                ask_size=0,
                last_trade_price=target_price,
                last_trade_quantity=0,
                total_traded_value=target_price * 100000
            )
            halt_ticks.append(halt_tick)
            
        return pre_trigger_ticks + [cb_tick] + halt_ticks
        
    def get_session_type(self, timestamp: datetime) -> SessionType:
        """Determine session type for given timestamp"""
        time_of_day = timestamp.time()
        
        if self.market_hours.pre_market_start <= time_of_day < self.market_hours.pre_market_end:
            return SessionType.PRE_MARKET
        elif self.market_hours.regular_start <= time_of_day < self.market_hours.regular_end:
            return SessionType.REGULAR
        elif self.market_hours.post_market_start <= time_of_day < self.market_hours.post_market_end:
            return SessionType.POST_MARKET
        else:
            return SessionType.CLOSED
            
    def to_dataframe(self, ticks: List[IntradayTick]) -> pd.DataFrame:
        """Convert tick data to pandas DataFrame"""
        data = []
        for tick in ticks:
            data.append({
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'open': tick.open,
                'high': tick.high,
                'low': tick.low,
                'close': tick.close,
                'volume': tick.volume,
                'session_type': tick.session_type.value,
                'bid': tick.bid,
                'ask': tick.ask,
                'bid_size': tick.bid_size,
                'ask_size': tick.ask_size,
                'last_trade_price': tick.last_trade_price,
                'last_trade_quantity': tick.last_trade_quantity,
                'total_traded_value': tick.total_traded_value
            })
            
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            
        return df
        
    def generate_multiple_symbols(self,
                                symbols: List[str],
                                date: datetime,
                                market_conditions: Dict) -> Dict[str, List[IntradayTick]]:
        """Generate intraday data for multiple symbols"""
        
        all_data = {}
        
        for symbol in symbols:
            # Get symbol-specific parameters
            symbol_params = market_conditions.get(symbol, {
                'previous_close': 100.0,
                'daily_return': np.random.normal(0, 0.02),
                'volatility': np.random.uniform(0.015, 0.035),
                'average_volume': np.random.randint(50000, 500000)
            })
            
            # Generate data
            ticks = self.generate_intraday_data(
                symbol=symbol,
                date=date,
                previous_close=symbol_params['previous_close'],
                daily_return=symbol_params['daily_return'],
                volatility=symbol_params['volatility'],
                average_volume=symbol_params['average_volume']
            )
            
            all_data[symbol] = ticks
            
        return all_data

# Example usage
if __name__ == "__main__":
    # Create generator
    generator = IntradayDataGenerator(seed=42)
    
    # Generate data for a single symbol
    symbol = "RELIANCE"
    date = datetime(2024, 1, 15)
    previous_close = 2500.0
    
    ticks = generator.generate_intraday_data(
        symbol=symbol,
        date=date,
        previous_close=previous_close,
        daily_return=0.015,  # 1.5% gain
        volatility=0.02,     # 2% volatility
        average_volume=200000
    )
    
    print(f"Generated {len(ticks)} ticks for {symbol}")
    
    # Convert to DataFrame
    df = generator.to_dataframe(ticks)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Total volume: {df['volume'].iloc[-1]:,}")
    
    # Show session distribution
    session_counts = df['session_type'].value_counts()
    print(f"\nSession distribution:")
    for session, count in session_counts.items():
        print(f"  {session}: {count} minutes")
        
    # Generate market depth
    current_price = df['close'].iloc[-1]
    depth = generator.generate_market_depth(current_price)
    print(f"\nMarket Depth (Price: {current_price:.2f}):")
    print(f"Best Bid: {depth['bids'][0]['price']:.2f} ({depth['bids'][0]['quantity']} qty)")
    print(f"Best Ask: {depth['asks'][0]['price']:.2f} ({depth['asks'][0]['quantity']} qty)")
    print(f"Spread: {depth['spread']:.2f} ({depth['spread_pct']:.3f}%)")
    
    # Generate circuit breaker scenario
    cb_ticks = generator.generate_circuit_breaker_scenario(
        symbol="TESTSTOCK",
        date=date,
        trigger_time=time(11, 30),
        direction='up'
    )
    
    print(f"\nGenerated circuit breaker scenario with {len(cb_ticks)} ticks")
    
    # Generate multiple symbols
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    market_conditions = {
        "RELIANCE": {'previous_close': 2500, 'daily_return': 0.01, 'volatility': 0.02, 'average_volume': 200000},
        "TCS": {'previous_close': 3800, 'daily_return': -0.005, 'volatility': 0.018, 'average_volume': 150000},
        "INFY": {'previous_close': 1600, 'daily_return': 0.008, 'volatility': 0.022, 'average_volume': 180000},
        "HDFCBANK": {'previous_close': 1650, 'daily_return': 0.012, 'volatility': 0.025, 'average_volume': 300000},
        "ICICIBANK": {'previous_close': 950, 'daily_return': -0.003, 'volatility': 0.028, 'average_volume': 250000}
    }
    
    multi_data = generator.generate_multiple_symbols(symbols, date, market_conditions)
    print(f"\nGenerated data for {len(multi_data)} symbols")
    
    for symbol, symbol_ticks in multi_data.items():
        if symbol_ticks:
            start_price = symbol_ticks[0].close
            end_price = symbol_ticks[-1].close
            return_pct = (end_price - start_price) / start_price * 100
            print(f"  {symbol}: {len(symbol_ticks)} ticks, Return: {return_pct:.2f}%")