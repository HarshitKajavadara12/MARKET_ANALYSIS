import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import random
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from dataclasses import dataclass
import pytz

@dataclass
class StockConfig:
    symbol: str
    sector: str
    base_price: float
    volatility: float
    market_cap: float
    beta: float
    circuit_limit: float = 0.20  # 20% circuit breaker

class IndianStockSimulator:
    def __init__(self):
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        self.market_open_time = time(9, 15)  # 9:15 AM IST
        self.market_close_time = time(15, 30)  # 3:30 PM IST
        self.pre_market_open = time(9, 0)  # 9:00 AM IST
        self.post_market_close = time(16, 0)  # 4:00 PM IST
        
        # Initialize stock configurations
        self.stock_configs = self._initialize_stock_configs()
        
        # Market indices base values
        self.indices_base = {
            'NIFTY50': 19500,
            'SENSEX': 65000,
            'BANKNIFTY': 44000,
            'NIFTYMIDCAP': 32000,
            'NIFTYSMALLCAP': 10500
        }
        
        # Sector weights and correlations
        self.sector_weights = {
            'Banking': 0.35,
            'IT': 0.18,
            'Oil & Gas': 0.12,
            'Consumer Goods': 0.08,
            'Automobile': 0.06,
            'Pharma': 0.05,
            'Metals': 0.04,
            'Telecom': 0.03,
            'Power': 0.03,
            'Others': 0.06
        }
        
        # Economic factors affecting market
        self.economic_factors = {
            'usd_inr_rate': 83.0,
            'repo_rate': 6.5,
            'inflation_rate': 5.2,
            'fii_flow': 0.0,  # Daily FII flow in crores
            'dii_flow': 0.0   # Daily DII flow in crores
        }
    
    def _initialize_stock_configs(self) -> Dict[str, StockConfig]:
        """Initialize configurations for major Indian stocks"""
        configs = {
            'RELIANCE.NS': StockConfig('RELIANCE', 'Oil & Gas', 2500, 0.25, 1500000, 1.1),
            'TCS.NS': StockConfig('TCS', 'IT', 3600, 0.20, 1300000, 0.8),
            'HDFCBANK.NS': StockConfig('HDFCBANK', 'Banking', 1650, 0.22, 900000, 1.2),
            'INFY.NS': StockConfig('INFOSYS', 'IT', 1450, 0.21, 600000, 0.9),
            'HINDUNILVR.NS': StockConfig('HINDUNILVR', 'Consumer Goods', 2650, 0.18, 620000, 0.7),
            'ICICIBANK.NS': StockConfig('ICICIBANK', 'Banking', 950, 0.24, 650000, 1.3),
            'KOTAKBANK.NS': StockConfig('KOTAKBANK', 'Banking', 1750, 0.23, 350000, 1.1),
            'BHARTIARTL.NS': StockConfig('BHARTIARTL', 'Telecom', 850, 0.26, 480000, 1.0),
            'ITC.NS': StockConfig('ITC', 'Consumer Goods', 450, 0.19, 560000, 0.8),
            'SBIN.NS': StockConfig('SBIN', 'Banking', 580, 0.28, 520000, 1.4),
            'LT.NS': StockConfig('LT', 'Construction', 2800, 0.25, 390000, 1.2),
            'ASIANPAINT.NS': StockConfig('ASIANPAINT', 'Consumer Goods', 3200, 0.22, 310000, 0.9),
            'MARUTI.NS': StockConfig('MARUTI', 'Automobile', 10500, 0.24, 320000, 1.1),
            'TITAN.NS': StockConfig('TITAN', 'Consumer Goods', 3100, 0.26, 280000, 1.0),
            'SUNPHARMA.NS': StockConfig('SUNPHARMA', 'Pharma', 1150, 0.23, 275000, 0.8)
        }
        return configs
    
    def is_market_open(self, current_time: datetime = None) -> bool:
        """Check if market is currently open"""
        if current_time is None:
            current_time = datetime.now(self.ist_timezone)
        
        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if current_time.weekday() >= 5:  # Saturday or Sunday
            return False
        
        current_time_only = current_time.time()
        return self.market_open_time <= current_time_only <= self.market_close_time
    
    def is_pre_market(self, current_time: datetime = None) -> bool:
        """Check if it's pre-market session"""
        if current_time is None:
            current_time = datetime.now(self.ist_timezone)
        
        if current_time.weekday() >= 5:
            return False
        
        current_time_only = current_time.time()
        return self.pre_market_open <= current_time_only < self.market_open_time
    
    def generate_market_sentiment(self) -> Dict[str, float]:
        """Generate overall market sentiment factors"""
        # Global factors
        global_sentiment = np.random.normal(0, 0.5)  # -1 to 1 scale
        
        # FII/DII flows (in crores)
        fii_flow = np.random.normal(0, 2000)  # Average daily FII flow
        dii_flow = np.random.normal(1000, 1500)  # DII usually positive
        
        # Currency impact
        usd_inr_change = np.random.normal(0, 0.2)  # Daily USD/INR change
        
        # Oil prices impact (Brent crude)
        oil_price_change = np.random.normal(0, 2)  # Daily oil price change %
        
        return {
            'global_sentiment': global_sentiment,
            'fii_flow': fii_flow,
            'dii_flow': dii_flow,
            'usd_inr_change': usd_inr_change,
            'oil_price_change': oil_price_change,
            'overall_sentiment': (global_sentiment + (fii_flow + dii_flow)/5000 - abs(usd_inr_change)*2) / 3
        }
    
    def generate_sector_movements(self, market_sentiment: Dict) -> Dict[str, float]:
        """Generate sector-wise movements based on market sentiment"""
        base_movement = market_sentiment['overall_sentiment']
        
        sector_movements = {}
        
        for sector in self.sector_weights.keys():
            # Sector-specific factors
            if sector == 'Banking':
                # Banking sensitive to interest rates and FII flows
                movement = base_movement + market_sentiment['fii_flow']/10000 + np.random.normal(0, 0.3)
            elif sector == 'IT':
                # IT sensitive to USD/INR and global sentiment
                movement = base_movement + market_sentiment['usd_inr_change']*2 + market_sentiment['global_sentiment']*0.5 + np.random.normal(0, 0.25)
            elif sector == 'Oil & Gas':
                # Oil & Gas sensitive to crude prices
                movement = base_movement + market_sentiment['oil_price_change']*0.3 + np.random.normal(0, 0.4)
            elif sector == 'Pharma':
                # Pharma less correlated with market, more stock-specific
                movement = base_movement*0.3 + np.random.normal(0, 0.35)
            else:
                # Other sectors follow market with some noise
                movement = base_movement + np.random.normal(0, 0.3)
            
            sector_movements[sector] = np.clip(movement, -0.1, 0.1)  # Cap at ±10%
        
        return sector_movements
    
    def apply_circuit_breaker(self, price: float, base_price: float, circuit_limit: float) -> float:
        """Apply circuit breaker limits"""
        upper_limit = base_price * (1 + circuit_limit)
        lower_limit = base_price * (1 - circuit_limit)
        
        return np.clip(price, lower_limit, upper_limit)
    
    def generate_intraday_data(self, symbol: str, date: datetime, 
                              market_sentiment: Dict, sector_movements: Dict) -> pd.DataFrame:
        """Generate minute-by-minute intraday data"""
        if symbol not in self.stock_configs:
            raise ValueError(f"Stock {symbol} not configured")
        
        config = self.stock_configs[symbol]
        sector_movement = sector_movements.get(config.sector, 0)
        
        # Generate time series for market hours (9:15 AM to 3:30 PM)
        start_time = datetime.combine(date.date(), self.market_open_time)
        end_time = datetime.combine(date.date(), self.market_close_time)
        
        # Create minute-by-minute timestamps
        timestamps = pd.date_range(start_time, end_time, freq='1min')
        
        # Initialize arrays
        prices = []
        volumes = []
        
        # Starting price (previous day close + gap)
        gap = np.random.normal(sector_movement, config.volatility/10)
        current_price = config.base_price * (1 + gap)
        
        # Apply circuit breaker for opening
        current_price = self.apply_circuit_breaker(current_price, config.base_price, config.circuit_limit)
        
        # Generate minute-by-minute data
        for i, timestamp in enumerate(timestamps):
            # Time-based patterns
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Opening volatility (first 30 minutes)
            if i < 30:
                volatility_multiplier = 2.0
            # Lunch time low volatility (12:00-1:00 PM)
            elif 12 <= hour < 13:
                volatility_multiplier = 0.5
            # Closing volatility (last 30 minutes)
            elif i >= len(timestamps) - 30:
                volatility_multiplier = 1.5
            else:
                volatility_multiplier = 1.0
            
            # Price movement
            price_change = np.random.normal(
                sector_movement/375,  # Spread daily movement across 375 minutes
                config.volatility/100 * volatility_multiplier
            )
            
            new_price = current_price * (1 + price_change)
            
            # Apply circuit breaker
            new_price = self.apply_circuit_breaker(new_price, config.base_price, config.circuit_limit)
            
            prices.append(new_price)
            current_price = new_price
            
            # Volume generation (higher at open/close, lower during lunch)
            base_volume = config.market_cap / 1000  # Base volume
            
            if i < 30 or i >= len(timestamps) - 30:  # Opening/closing
                volume_multiplier = np.random.uniform(1.5, 3.0)
            elif 12 <= hour < 13:  # Lunch time
                volume_multiplier = np.random.uniform(0.3, 0.7)
            else:
                volume_multiplier = np.random.uniform(0.8, 1.2)
            
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
            volumes.append(volume)
        
        # Create OHLCV data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
        
        # Resample to create OHLCV candles (1-minute)
        df.set_index('timestamp', inplace=True)
        ohlcv = df.resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return ohlcv
    
    def generate_stock_data(self, symbol: str, timeframe: str = '1d', 
                           days: int = 100) -> pd.DataFrame:
        """Generate stock data for specified timeframe"""
        if symbol not in self.stock_configs:
            # Try to get real data if available
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=f"{days}d", interval=timeframe)
                return data
            except:
                # Fallback to simulation
                return self._generate_fallback_data(symbol, timeframe, days)
        
        config = self.stock_configs[symbol]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if timeframe in ['1m', '5m', '15m', '1h']:
            return self._generate_intraday_timeframe(symbol, timeframe, days)
        else:
            return self._generate_daily_data(symbol, days)
    
    def _generate_daily_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate daily OHLCV data"""
        config = self.stock_configs[symbol]
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        
        # Filter out weekends
        dates = dates[dates.weekday < 5]
        
        data = []
        current_price = config.base_price
        
        for date in dates:
            # Generate market sentiment for the day
            market_sentiment = self.generate_market_sentiment()
            sector_movements = self.generate_sector_movements(market_sentiment)
            
            # Daily price movement
            sector_movement = sector_movements.get(config.sector, 0)
            daily_change = np.random.normal(sector_movement, config.volatility/15)
            
            # Calculate OHLC
            open_price = current_price
            close_price = open_price * (1 + daily_change)
            
            # Apply circuit breaker
            close_price = self.apply_circuit_breaker(close_price, open_price, config.circuit_limit)
            
            # High and Low
            intraday_range = abs(daily_change) + np.random.uniform(0.005, 0.02)
            high_price = max(open_price, close_price) * (1 + intraday_range/2)
            low_price = min(open_price, close_price) * (1 - intraday_range/2)
            
            # Apply circuit breakers to high/low
            high_price = self.apply_circuit_breaker(high_price, open_price, config.circuit_limit)
            low_price = self.apply_circuit_breaker(low_price, open_price, config.circuit_limit)
            
            # Volume
            base_volume = config.market_cap * 1000
            volume_multiplier = 1 + abs(daily_change) * 5  # Higher volume on big moves
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
            
            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
            
            current_price = close_price
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def _generate_intraday_timeframe(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate intraday data for specified timeframe"""
        # Generate minute data first
        minute_data = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            if date.weekday() < 5:  # Only weekdays
                market_sentiment = self.generate_market_sentiment()
                sector_movements = self.generate_sector_movements(market_sentiment)
                
                daily_data = self.generate_intraday_data(symbol, date, market_sentiment, sector_movements)
                minute_data.append(daily_data)
        
        if not minute_data:
            return pd.DataFrame()
        
        # Combine all days
        combined_data = pd.concat(minute_data)
        
        # Resample to requested timeframe
        timeframe_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1H'
        }
        
        freq = timeframe_map.get(timeframe, '1min')
        
        resampled = combined_data.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled.sort_index()
    
    def _generate_fallback_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate fallback data for unknown symbols"""
        # Create a generic stock configuration
        base_price = np.random.uniform(100, 5000)
        volatility = np.random.uniform(0.15, 0.35)
        
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        dates = dates[dates.weekday < 5]  # Filter weekends
        
        data = []
        current_price = base_price
        
        for date in dates:
            daily_change = np.random.normal(0, volatility/15)
            
            open_price = current_price
            close_price = open_price * (1 + daily_change)
            
            intraday_range = abs(daily_change) + np.random.uniform(0.005, 0.02)
            high_price = max(open_price, close_price) * (1 + intraday_range/2)
            low_price = min(open_price, close_price) * (1 - intraday_range/2)
            
            volume = int(np.random.uniform(100000, 10000000))
            
            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
            
            current_price = close_price
        
        return pd.DataFrame(data, index=dates)
    
    def generate_options_data(self, symbol: str, expiry_date: datetime) -> pd.DataFrame:
        """Generate options chain data"""
        if symbol not in self.stock_configs:
            return pd.DataFrame()
        
        config = self.stock_configs[symbol]
        spot_price = config.base_price
        
        # Generate strike prices around spot
        strikes = []
        for i in range(-10, 11):
            strike = round(spot_price * (1 + i * 0.05), 0)
            strikes.append(strike)
        
        options_data = []
        
        for strike in strikes:
            # Simple Black-Scholes approximation for demo
            time_to_expiry = (expiry_date - datetime.now()).days / 365
            
            # Call options
            call_iv = np.random.uniform(0.15, 0.45)
            call_price = max(spot_price - strike, 0) + np.random.uniform(1, 50)
            
            # Put options
            put_iv = np.random.uniform(0.15, 0.45)
            put_price = max(strike - spot_price, 0) + np.random.uniform(1, 50)
            
            options_data.append({
                'strike': strike,
                'call_price': call_price,
                'call_iv': call_iv,
                'call_oi': np.random.randint(1000, 100000),
                'put_price': put_price,
                'put_iv': put_iv,
                'put_oi': np.random.randint(1000, 100000)
            })
        
        return pd.DataFrame(options_data)
    
    def get_market_depth(self, symbol: str) -> Dict:
        """Generate market depth (order book) data"""
        if symbol not in self.stock_configs:
            return {}
        
        config = self.stock_configs[symbol]
        current_price = config.base_price
        
        # Generate bid/ask levels
        bid_levels = []
        ask_levels = []
        
        for i in range(5):
            bid_price = current_price * (1 - (i + 1) * 0.001)
            ask_price = current_price * (1 + (i + 1) * 0.001)
            
            bid_qty = np.random.randint(100, 10000)
            ask_qty = np.random.randint(100, 10000)
            
            bid_levels.append({'price': bid_price, 'quantity': bid_qty})
            ask_levels.append({'price': ask_price, 'quantity': ask_qty})
        
        return {
            'symbol': symbol,
            'ltp': current_price,
            'bids': bid_levels,
            'asks': ask_levels,
            'total_bid_qty': sum(level['quantity'] for level in bid_levels),
            'total_ask_qty': sum(level['quantity'] for level in ask_levels)
        }
    
    def simulate_corporate_action(self, symbol: str, action_type: str, 
                                 ratio: float = None) -> Dict:
        """Simulate corporate actions like splits, bonuses, dividends"""
        if symbol not in self.stock_configs:
            return {}
        
        config = self.stock_configs[symbol]
        
        if action_type == 'split':
            # Stock split
            new_price = config.base_price / ratio
            return {
                'action': 'split',
                'ratio': f"1:{ratio}",
                'old_price': config.base_price,
                'new_price': new_price,
                'effective_date': datetime.now().date()
            }
        
        elif action_type == 'bonus':
            # Bonus issue
            new_price = config.base_price * ratio / (ratio + 1)
            return {
                'action': 'bonus',
                'ratio': f"{ratio}:1",
                'old_price': config.base_price,
                'new_price': new_price,
                'effective_date': datetime.now().date()
            }
        
        elif action_type == 'dividend':
            # Dividend payment
            dividend_amount = config.base_price * 0.02  # 2% dividend
            ex_dividend_price = config.base_price - dividend_amount
            return {
                'action': 'dividend',
                'dividend_amount': dividend_amount,
                'old_price': config.base_price,
                'ex_dividend_price': ex_dividend_price,
                'record_date': datetime.now().date(),
                'payment_date': (datetime.now() + timedelta(days=30)).date()
            }
        
        return {}

# Example usage
if __name__ == "__main__":
    simulator = IndianStockSimulator()
    
    # Generate daily data for Reliance
    reliance_data = simulator.generate_stock_data('RELIANCE.NS', '1d', 30)
    print("Reliance Daily Data:")
    print(reliance_data.tail())
    
    # Generate intraday data
    intraday_data = simulator.generate_stock_data('TCS.NS', '5m', 1)
    print("\nTCS 5-minute Data:")
    print(intraday_data.tail())
    
    # Get market depth
    depth = simulator.get_market_depth('HDFCBANK.NS')
    print("\nHDFC Bank Market Depth:")
    print(depth)