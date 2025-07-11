import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
import yfinance as yf
from bs4 import BeautifulSoup
import asyncio
import aiohttp

@dataclass
class NSEStock:
    symbol: str
    company_name: str
    sector: str
    series: str
    isin: str
    market_cap: float
    face_value: float

class NSEDataHandler:
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Initialize session with NSE
        self._initialize_session()
        
        # Cache for storing data
        self.cache = {
            'stocks': {},
            'indices': {},
            'market_status': None,
            'last_update': None
        }
        
        # NSE stock list
        self.nse_stocks = self._load_nse_stocks()
        
        # Market timings
        self.market_timings = {
            'pre_open': {'start': '09:00', 'end': '09:15'},
            'normal': {'start': '09:15', 'end': '15:30'},
            'closing': {'start': '15:30', 'end': '16:00'}
        }
        
        # Sector mappings
        self.sector_indices = {
            'NIFTY BANK': 'Banking',
            'NIFTY IT': 'Information Technology',
            'NIFTY PHARMA': 'Pharmaceuticals',
            'NIFTY AUTO': 'Automobile',
            'NIFTY FMCG': 'FMCG',
            'NIFTY METAL': 'Metals',
            'NIFTY ENERGY': 'Energy',
            'NIFTY REALTY': 'Real Estate',
            'NIFTY MEDIA': 'Media'
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_session(self):
        """Initialize session with NSE website"""
        try:
            # Get cookies from NSE homepage
            response = self.session.get(self.base_url, timeout=10)
            if response.status_code == 200:
                self.logger.info("NSE session initialized successfully")
            else:
                self.logger.warning(f"NSE session initialization returned status: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to initialize NSE session: {e}")
    
    def _load_nse_stocks(self) -> Dict[str, NSEStock]:
        """Load NSE stock list with company details"""
        # This would typically load from NSE API or a maintained database
        # For demo purposes, using a predefined list
        stocks = {
            'RELIANCE': NSEStock('RELIANCE', 'Reliance Industries Limited', 'Oil & Gas', 'EQ', 'INE002A01018', 1500000, 10),
            'TCS': NSEStock('TCS', 'Tata Consultancy Services Limited', 'Information Technology', 'EQ', 'INE467B01029', 1300000, 1),
            'HDFCBANK': NSEStock('HDFCBANK', 'HDFC Bank Limited', 'Banking', 'EQ', 'INE040A01034', 900000, 1),
            'INFY': NSEStock('INFY', 'Infosys Limited', 'Information Technology', 'EQ', 'INE009A01021', 600000, 5),
            'HINDUNILVR': NSEStock('HINDUNILVR', 'Hindustan Unilever Limited', 'FMCG', 'EQ', 'INE030A01027', 620000, 1),
            'ICICIBANK': NSEStock('ICICIBANK', 'ICICI Bank Limited', 'Banking', 'EQ', 'INE090A01021', 650000, 2),
            'KOTAKBANK': NSEStock('KOTAKBANK', 'Kotak Mahindra Bank Limited', 'Banking', 'EQ', 'INE237A01028', 350000, 5),
            'BHARTIARTL': NSEStock('BHARTIARTL', 'Bharti Airtel Limited', 'Telecommunications', 'EQ', 'INE397D01024', 480000, 5),
            'ITC': NSEStock('ITC', 'ITC Limited', 'FMCG', 'EQ', 'INE154A01025', 560000, 1),
            'SBIN': NSEStock('SBIN', 'State Bank of India', 'Banking', 'EQ', 'INE062A01020', 520000, 1),
            'LT': NSEStock('LT', 'Larsen & Toubro Limited', 'Construction', 'EQ', 'INE018A01030', 390000, 2),
            'ASIANPAINT': NSEStock('ASIANPAINT', 'Asian Paints Limited', 'Consumer Goods', 'EQ', 'INE021A01026', 310000, 1),
            'MARUTI': NSEStock('MARUTI', 'Maruti Suzuki India Limited', 'Automobile', 'EQ', 'INE585B01010', 320000, 5),
            'TITAN': NSEStock('TITAN', 'Titan Company Limited', 'Consumer Goods', 'EQ', 'INE280A01028', 280000, 1),
            'SUNPHARMA': NSEStock('SUNPHARMA', 'Sun Pharmaceutical Industries Limited', 'Pharmaceuticals', 'EQ', 'INE044A01036', 275000, 1)
        }
        return stocks
    
    def get_market_status(self) -> Dict[str, Union[str, bool]]:
        """Get current market status"""
        try:
            # Try to get real market status from NSE
            url = f"{self.base_url}/api/marketStatus"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'market_status': data.get('marketState', 'UNKNOWN'),
                    'is_open': data.get('marketState') == 'Market is Open',
                    'last_updated': datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.warning(f"Failed to get real market status: {e}")
        
        # Fallback to time-based calculation
        now = datetime.now()
        current_time = now.time()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return {
                'market_status': 'Market is Closed for the weekend',
                'is_open': False,
                'last_updated': now.isoformat()
            }
        
        # Check market hours
        if current_time >= datetime.strptime('09:15', '%H:%M').time() and current_time <= datetime.strptime('15:30', '%H:%M').time():
            status = 'Market is Open'
            is_open = True
        elif current_time >= datetime.strptime('09:00', '%H:%M').time() and current_time < datetime.strptime('09:15', '%H:%M').time():
            status = 'Pre-open Session'
            is_open = False
        elif current_time > datetime.strptime('15:30', '%H:%M').time() and current_time <= datetime.strptime('16:00', '%H:%M').time():
            status = 'Closing Session'
            is_open = False
        else:
            status = 'Market is Closed'
            is_open = False
        
        return {
            'market_status': status,
            'is_open': is_open,
            'last_updated': now.isoformat()
        }
    
    def get_stock_quote(self, symbol: str) -> Dict:
        """Get real-time stock quote"""
        try:
            # Try NSE API first
            url = f"{self.base_url}/api/quote-equity?symbol={symbol}"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_nse_quote(data)
        except Exception as e:
            self.logger.warning(f"Failed to get NSE quote for {symbol}: {e}")
        
        # Fallback to Yahoo Finance
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info
            hist = ticker.history(period="2d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                
                return {
                    'symbol': symbol,
                    'companyName': info.get('longName', symbol),
                    'lastPrice': current_price,
                    'change': current_price - prev_close,
                    'pChange': ((current_price - prev_close) / prev_close) * 100,
                    'previousClose': prev_close,
                    'open': hist['Open'].iloc[-1],
                    'dayHigh': hist['High'].iloc[-1],
                    'dayLow': hist['Low'].iloc[-1],
                    'totalTradedVolume': int(hist['Volume'].iloc[-1]),
                    'totalTradedValue': current_price * hist['Volume'].iloc[-1],
                    'lastUpdateTime': datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
                    'yearHigh': info.get('fiftyTwoWeekHigh', current_price),
                    'yearLow': info.get('fiftyTwoWeekLow', current_price),
                    'marketCap': info.get('marketCap', 0),
                    'pe': info.get('trailingPE', 0),
                    'pb': info.get('priceToBook', 0)
                }
        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
        
        # Final fallback - simulated data
        return self._generate_simulated_quote(symbol)
    
    def _parse_nse_quote(self, data: Dict) -> Dict:
        """Parse NSE API response"""
        try:
            price_info = data.get('priceInfo', {})
            return {
                'symbol': data.get('info', {}).get('symbol', ''),
                'companyName': data.get('info', {}).get('companyName', ''),
                'lastPrice': price_info.get('lastPrice', 0),
                'change': price_info.get('change', 0),
                'pChange': price_info.get('pChange', 0),
                'previousClose': price_info.get('previousClose', 0),
                'open': price_info.get('open', 0),
                'dayHigh': price_info.get('intraDayHighLow', {}).get('max', 0),
                'dayLow': price_info.get('intraDayHighLow', {}).get('min', 0),
                'totalTradedVolume': price_info.get('totalTradedVolume', 0),
                'totalTradedValue': price_info.get('totalTradedValue', 0),
                'lastUpdateTime': price_info.get('lastUpdateTime', ''),
                'yearHigh': price_info.get('weekHighLow', {}).get('max', 0),
                'yearLow': price_info.get('weekHighLow', {}).get('min', 0)
            }
        except Exception as e:
            self.logger.error(f"Error parsing NSE quote: {e}")
            return {}
    
    def _generate_simulated_quote(self, symbol: str) -> Dict:
        """Generate simulated quote data"""
        if symbol in self.nse_stocks:
            stock = self.nse_stocks[symbol]
            base_price = 1000 + hash(symbol) % 5000  # Deterministic base price
        else:
            base_price = np.random.uniform(100, 5000)
        
        # Add some randomness
        current_price = base_price * (1 + np.random.normal(0, 0.02))
        prev_close = base_price
        change = current_price - prev_close
        pchange = (change / prev_close) * 100
        
        return {
            'symbol': symbol,
            'companyName': self.nse_stocks.get(symbol, NSEStock(symbol, symbol, 'Unknown', 'EQ', '', 0, 0)).company_name,
            'lastPrice': round(current_price, 2),
            'change': round(change, 2),
            'pChange': round(pchange, 2),
            'previousClose': round(prev_close, 2),
            'open': round(current_price * (1 + np.random.normal(0, 0.01)), 2),
            'dayHigh': round(current_price * (1 + abs(np.random.normal(0, 0.015))), 2),
            'dayLow': round(current_price * (1 - abs(np.random.normal(0, 0.015))), 2),
            'totalTradedVolume': int(np.random.uniform(100000, 10000000)),
            'totalTradedValue': int(current_price * np.random.uniform(100000, 10000000)),
            'lastUpdateTime': datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
            'yearHigh': round(current_price * (1 + np.random.uniform(0.1, 0.5)), 2),
            'yearLow': round(current_price * (1 - np.random.uniform(0.1, 0.4)), 2),
            'marketCap': self.nse_stocks.get(symbol, NSEStock(symbol, symbol, 'Unknown', 'EQ', '', 100000, 0)).market_cap,
            'pe': round(np.random.uniform(10, 50), 2),
            'pb': round(np.random.uniform(1, 10), 2)
        }
    
    def get_index_data(self, index_name: str = 'NIFTY 50') -> Dict:
        """Get index data"""
        try:
            # Try NSE API
            url = f"{self.base_url}/api/allIndices"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                for index in data.get('data', []):
                    if index.get('index') == index_name:
                        return {
                            'name': index.get('index'),
                            'last': index.get('last'),
                            'variation': index.get('variation'),
                            'percentChange': index.get('percentChange'),
                            'open': index.get('open'),
                            'high': index.get('dayHigh'),
                            'low': index.get('dayLow'),
                            'previousClose': index.get('previousClose'),
                            'yearHigh': index.get('yearHigh'),
                            'yearLow': index.get('yearLow'),
                            'lastUpdateTime': datetime.now().strftime('%d-%b-%Y %H:%M:%S')
                        }
        except Exception as e:
            self.logger.warning(f"Failed to get NSE index data: {e}")
        
        # Fallback to simulated data
        return self._generate_simulated_index(index_name)
    
    def _generate_simulated_index(self, index_name: str) -> Dict:
        """Generate simulated index data"""
        base_values = {
            'NIFTY 50': 19500,
            'NIFTY BANK': 44000,
            'SENSEX': 65000,
            'NIFTY IT': 30000,
            'NIFTY AUTO': 15000,
            'NIFTY PHARMA': 13000,
            'NIFTY FMCG': 18000
        }
        
        base_value = base_values.get(index_name, 10000)
        current_value = base_value * (1 + np.random.normal(0, 0.01))
        change = current_value - base_value
        pchange = (change / base_value) * 100
        
        return {
            'name': index_name,
            'last': round(current_value, 2),
            'variation': round(change, 2),
            'percentChange': round(pchange, 2),
            'open': round(current_value * (1 + np.random.normal(0, 0.005)), 2),
            'high': round(current_value * (1 + abs(np.random.normal(0, 0.008))), 2),
            'low': round(current_value * (1 - abs(np.random.normal(0, 0.008))), 2),
            'previousClose': round(base_value, 2),
            'yearHigh': round(current_value * (1 + np.random.uniform(0.1, 0.3)), 2),
            'yearLow': round(current_value * (1 - np.random.uniform(0.1, 0.25)), 2),
            'lastUpdateTime': datetime.now().strftime('%d-%b-%Y %H:%M:%S')
        }
    
    def get_top_gainers(self, count: int = 10) -> List[Dict]:
        """Get top gaining stocks"""
        try:
            url = f"{self.base_url}/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                stocks = data.get('data', [])
                
                # Sort by percentage change
                gainers = sorted(stocks, key=lambda x: x.get('pChange', 0), reverse=True)[:count]
                
                return [{
                    'symbol': stock.get('symbol'),
                    'lastPrice': stock.get('lastPrice'),
                    'change': stock.get('change'),
                    'pChange': stock.get('pChange')
                } for stock in gainers]
        except Exception as e:
            self.logger.warning(f"Failed to get top gainers: {e}")
        
        # Fallback to simulated data
        return self._generate_simulated_movers(count, 'gainers')
    
    def get_top_losers(self, count: int = 10) -> List[Dict]:
        """Get top losing stocks"""
        try:
            url = f"{self.base_url}/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                stocks = data.get('data', [])
                
                # Sort by percentage change (ascending for losers)
                losers = sorted(stocks, key=lambda x: x.get('pChange', 0))[:count]
                
                return [{
                    'symbol': stock.get('symbol'),
                    'lastPrice': stock.get('lastPrice'),
                    'change': stock.get('change'),
                    'pChange': stock.get('pChange')
                } for stock in losers]
        except Exception as e:
            self.logger.warning(f"Failed to get top losers: {e}")
        
        # Fallback to simulated data
        return self._generate_simulated_movers(count, 'losers')
    
    def _generate_simulated_movers(self, count: int, mover_type: str) -> List[Dict]:
        """Generate simulated top movers"""
        symbols = list(self.nse_stocks.keys())[:count]
        movers = []
        
        for symbol in symbols:
            base_price = 1000 + hash(symbol) % 3000
            
            if mover_type == 'gainers':
                pchange = np.random.uniform(2, 15)
            else:  # losers
                pchange = np.random.uniform(-15, -2)
            
            current_price = base_price * (1 + pchange/100)
            change = current_price - base_price
            
            movers.append({
                'symbol': symbol,
                'lastPrice': round(current_price, 2),
                'change': round(change, 2),
                'pChange': round(pchange, 2)
            })
        
        return movers
    
    def get_sector_performance(self) -> Dict[str, Dict]:
        """Get sector-wise performance"""
        sector_data = {}
        
        for index_name, sector in self.sector_indices.items():
            index_data = self.get_index_data(index_name)
            sector_data[sector] = {
                'index': index_name,
                'value': index_data.get('last', 0),
                'change': index_data.get('variation', 0),
                'pChange': index_data.get('percentChange', 0)
            }
        
        return sector_data
    
    def get_market_breadth(self) -> Dict:
        """Get market breadth data (advances vs declines)"""
        try:
            # This would typically come from NSE API
            # For demo, generating simulated data
            total_stocks = 2000
            advances = np.random.randint(600, 1200)
            declines = np.random.randint(600, 1200)
            unchanged = total_stocks - advances - declines
            
            return {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'advance_decline_ratio': round(advances / declines, 2) if declines > 0 else 0,
                'new_highs': np.random.randint(10, 100),
                'new_lows': np.random.randint(10, 100),
                'total_stocks': total_stocks
            }
        except Exception as e:
            self.logger.error(f"Error getting market breadth: {e}")
            return {}
    
    def get_fii_dii_data(self, days: int = 30) -> pd.DataFrame:
        """Get FII/DII investment data"""
        # Generate simulated FII/DII data
        dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        data = []
        for date in dates:
            fii_equity = np.random.normal(0, 2000)  # FII equity flow in crores
            fii_debt = np.random.normal(500, 1000)  # FII debt flow in crores
            dii_equity = np.random.normal(1500, 1000)  # DII equity flow in crores
            
            data.append({
                'date': date,
                'fii_equity': fii_equity,
                'fii_debt': fii_debt,
                'fii_total': fii_equity + fii_debt,
                'dii_equity': dii_equity,
                'net_flow': fii_equity + fii_debt + dii_equity
            })
        
        return pd.DataFrame(data)
    
    def get_options_data(self, symbol: str, expiry: str = None) -> Dict:
        """Get options chain data"""
        try:
            # This would typically fetch from NSE options API
            # For demo, generating simulated options data
            quote = self.get_stock_quote(symbol)
            spot_price = quote.get('lastPrice', 1000)
            
            strikes = []
            for i in range(-10, 11):
                strike = round(spot_price * (1 + i * 0.025), 0)
                strikes.append(strike)
            
            options_data = {
                'underlying': symbol,
                'spot_price': spot_price,
                'expiry': expiry or (datetime.now() + timedelta(days=30)).strftime('%d-%b-%Y'),
                'calls': [],
                'puts': []
            }
            
            for strike in strikes:
                # Simulate call options
                call_price = max(spot_price - strike, 0) + np.random.uniform(5, 100)
                call_iv = np.random.uniform(15, 45)
                call_oi = np.random.randint(1000, 100000)
                
                # Simulate put options
                put_price = max(strike - spot_price, 0) + np.random.uniform(5, 100)
                put_iv = np.random.uniform(15, 45)
                put_oi = np.random.randint(1000, 100000)
                
                options_data['calls'].append({
                    'strike': strike,
                    'price': round(call_price, 2),
                    'iv': round(call_iv, 2),
                    'oi': call_oi,
                    'volume': np.random.randint(100, 10000)
                })
                
                options_data['puts'].append({
                    'strike': strike,
                    'price': round(put_price, 2),
                    'iv': round(put_iv, 2),
                    'oi': put_oi,
                    'volume': np.random.randint(100, 10000)
                })
            
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error getting options data for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, period: str = '1y', 
                           interval: str = '1d') -> pd.DataFrame:
        """Get historical stock data"""
        try:
            # Use yfinance for historical data
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def search_stocks(self, query: str) -> List[Dict]:
        """Search for stocks by name or symbol"""
        results = []
        query_lower = query.lower()
        
        for symbol, stock in self.nse_stocks.items():
            if (query_lower in symbol.lower() or 
                query_lower in stock.company_name.lower()):
                results.append({
                    'symbol': symbol,
                    'name': stock.company_name,
                    'sector': stock.sector,
                    'series': stock.series
                })
        
        return results[:10]  # Return top 10 matches
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time data for multiple symbols asynchronously"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                task = self._fetch_quote_async(session, symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data = {}
            for symbol, result in zip(symbols, results):
                if not isinstance(result, Exception):
                    data[symbol] = result
                else:
                    self.logger.error(f"Error fetching data for {symbol}: {result}")
                    data[symbol] = self._generate_simulated_quote(symbol)
            
            return data
    
    async def _fetch_quote_async(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        """Fetch quote data asynchronously"""
        try:
            url = f"{self.base_url}/api/quote-equity?symbol={symbol}"
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_nse_quote(data)
        except Exception as e:
            self.logger.warning(f"Async fetch failed for {symbol}: {e}")
        
        return self._generate_simulated_quote(symbol)

# Example usage
if __name__ == "__main__":
    nse = NSEDataHandler()
    
    # Get market status
    status = nse.get_market_status()
    print("Market Status:", status)
    
    # Get stock quote
    quote = nse.get_stock_quote('RELIANCE')
    print("\nRELIANCE Quote:", quote)
    
    # Get index data
    nifty = nse.get_index_data('NIFTY 50')
    print("\nNIFTY 50:", nifty)
    
    # Get top gainers
    gainers = nse.get_top_gainers(5)
    print("\nTop Gainers:", gainers)
    
    # Get FII/DII data
    fii_dii = nse.get_fii_dii_data(10)
    print("\nFII/DII Data:")
    print(fii_dii.head())