#!/usr/bin/env python3
"""
Market Research System v1.0 (2022)
Indian Stock Universe Manager - Update Stock List

This script manages the universe of Indian stocks to track.
Focuses on NSE/BSE listed companies with high market cap and liquidity.
"""

import os
import sys
import json
import yaml
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import logging
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.settings import STOCK_UNIVERSE_FILE, DATA_DIR
from utils.logging_utils import setup_logging
from utils.file_utils import ensure_directory_exists

# Setup logging
logger = setup_logging('update_stock_list')

class IndianStockUniverseManager:
    """Manages the universe of Indian stocks for market research"""
    
    def __init__(self):
        self.nifty_50_stocks = [
            # Technology
            'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
            
            # Banking & Financial Services
            'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'SBIN.NS',
            'HDFC.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
            
            # Oil & Gas
            'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'GAIL.NS',
            
            # Metals & Mining
            'TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS',
            
            # Automobiles
            'MARUTI.NS', 'M&M.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS',
            
            # Consumer Goods
            'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
            
            # Pharmaceuticals
            'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS',
            
            # Cement
            'ULTRACEMCO.NS', 'SHREECEM.NS', 'GRASIM.NS',
            
            # Power
            'NTPC.NS', 'POWERGRID.NS',
            
            # Telecom
            'BHARTIARTL.NS',
            
            # Others
            'LT.NS', 'ASIANPAINT.NS', 'TITAN.NS', 'ADANIPORTS.NS'
        ]
        
        self.nifty_next_50_stocks = [
            # Additional high-cap stocks
            'ADANIENT.NS', 'GODREJCP.NS', 'PIDILITIND.NS', 'DABUR.NS',
            'MARICO.NS', 'COLPAL.NS', 'MCDOWELL-N.NS', 'HAVELLS.NS',
            'BERGEPAINT.NS', 'INDIGO.NS', 'BANKBARODA.NS', 'PNB.NS',
            'CANBK.NS', 'IDEA.NS', 'VEDL.NS', 'HINDZINC.NS',
            'SAIL.NS', 'NMDC.NS', 'BHEL.NS', 'BEL.NS'
        ]
        
        self.sector_mapping = {
            # Technology
            'TCS.NS': 'Technology', 'INFY.NS': 'Technology', 'HCLTECH.NS': 'Technology',
            'WIPRO.NS': 'Technology', 'TECHM.NS': 'Technology',
            
            # Banking
            'HDFCBANK.NS': 'Banking', 'ICICIBANK.NS': 'Banking', 'KOTAKBANK.NS': 'Banking',
            'AXISBANK.NS': 'Banking', 'SBIN.NS': 'Banking', 'BANKBARODA.NS': 'Banking',
            'PNB.NS': 'Banking', 'CANBK.NS': 'Banking',
            
            # Financial Services
            'HDFC.NS': 'Financial Services', 'BAJFINANCE.NS': 'Financial Services',
            'BAJAJFINSV.NS': 'Financial Services',
            
            # Oil & Gas
            'RELIANCE.NS': 'Oil & Gas', 'ONGC.NS': 'Oil & Gas', 'IOC.NS': 'Oil & Gas',
            'BPCL.NS': 'Oil & Gas', 'GAIL.NS': 'Oil & Gas',
            
            # Metals
            'TATASTEEL.NS': 'Metals', 'HINDALCO.NS': 'Metals', 'JSWSTEEL.NS': 'Metals',
            'COALINDIA.NS': 'Metals', 'VEDL.NS': 'Metals', 'HINDZINC.NS': 'Metals',
            'SAIL.NS': 'Metals', 'NMDC.NS': 'Metals',
            
            # Automobiles
            'MARUTI.NS': 'Automobiles', 'M&M.NS': 'Automobiles', 'TATAMOTORS.NS': 'Automobiles',
            'BAJAJ-AUTO.NS': 'Automobiles', 'HEROMOTOCO.NS': 'Automobiles',
            
            # Consumer Goods
            'HINDUNILVR.NS': 'Consumer Goods', 'ITC.NS': 'Consumer Goods',
            'NESTLEIND.NS': 'Consumer Goods', 'BRITANNIA.NS': 'Consumer Goods',
            'GODREJCP.NS': 'Consumer Goods', 'DABUR.NS': 'Consumer Goods',
            'MARICO.NS': 'Consumer Goods', 'COLPAL.NS': 'Consumer Goods',
            
            # Pharmaceuticals
            'SUNPHARMA.NS': 'Pharmaceuticals', 'DRREDDY.NS': 'Pharmaceuticals',
            'CIPLA.NS': 'Pharmaceuticals', 'DIVISLAB.NS': 'Pharmaceuticals',
            
            # Others
            'LT.NS': 'Infrastructure', 'ASIANPAINT.NS': 'Paints', 'TITAN.NS': 'Jewelry',
            'ADANIPORTS.NS': 'Infrastructure', 'ULTRACEMCO.NS': 'Cement',
            'SHREECEM.NS': 'Cement', 'GRASIM.NS': 'Cement', 'NTPC.NS': 'Power',
            'POWERGRID.NS': 'Power', 'BHARTIARTL.NS': 'Telecom', 'IDEA.NS': 'Telecom',
            'INDIGO.NS': 'Aviation', 'BHEL.NS': 'Engineering', 'BEL.NS': 'Defense'
        }
        
        self.indices = {
            'NIFTY': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY_BANK': '^NSEBANK',
            'NIFTY_IT': '^CNXIT',
            'NIFTY_PHARMA': '^CNXPHARMA',
            'NIFTY_AUTO': '^CNXAUTO',
            'NIFTY_METAL': '^CNXMETAL',
            'NIFTY_ENERGY': '^CNXENERGY'
        }
    
    def validate_stock_symbol(self, symbol: str) -> bool:
        """Validate if a stock symbol is valid and tradeable"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Check if we can get basic info
            if not info or 'regularMarketPrice' not in info:
                logger.warning(f"Symbol {symbol} may not be valid or tradeable")
                return False
                
            # Check if it's an Indian stock (currency should be INR)
            if info.get('currency') != 'INR':
                logger.warning(f"Symbol {symbol} is not an Indian stock (currency: {info.get('currency')})")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {str(e)}")
            return False
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a stock"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'sector': self.sector_mapping.get(symbol, info.get('sector', 'Unknown')),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'INR'),
                'exchange': info.get('exchange', 'NSE'),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {str(e)}")
            return None
    
    def update_stock_universe(self) -> Dict[str, Any]:
        """Update the complete stock universe"""
        logger.info("Starting stock universe update...")
        
        all_stocks = self.nifty_50_stocks + self.nifty_next_50_stocks
        valid_stocks = []
        invalid_stocks = []
        stock_details = {}
        
        # Validate and get info for each stock
        for symbol in all_stocks:
            logger.info(f"Processing {symbol}...")
            
            if self.validate_stock_symbol(symbol):
                stock_info = self.get_stock_info(symbol)
                if stock_info:
                    valid_stocks.append(symbol)
                    stock_details[symbol] = stock_info
                else:
                    invalid_stocks.append(symbol)
            else:
                invalid_stocks.append(symbol)
        
        # Create universe structure
        universe = {
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'total_stocks': len(valid_stocks),
                'valid_stocks': len(valid_stocks),
                'invalid_stocks': len(invalid_stocks),
                'version': '1.0',
                'market': 'India',
                'exchanges': ['NSE', 'BSE']
            },
            'indices': self.indices,
            'stocks': {
                'valid': valid_stocks,
                'invalid': invalid_stocks,
                'details': stock_details
            },
            'sectors': self._group_by_sector(stock_details),
            'market_caps': self._group_by_market_cap(stock_details)
        }
        
        logger.info(f"Universe update complete. Valid: {len(valid_stocks)}, Invalid: {len(invalid_stocks)}")
        return universe
    
    def _group_by_sector(self, stock_details: Dict) -> Dict[str, List[str]]:
        """Group stocks by sector"""
        sectors = {}
        for symbol, details in stock_details.items():
            sector = details.get('sector', 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(symbol)
        return sectors
    
    def _group_by_market_cap(self, stock_details: Dict) -> Dict[str, List[str]]:
        """Group stocks by market cap ranges"""
        market_caps = {
            'large_cap': [],      # > 20,000 crores
            'mid_cap': [],        # 5,000 - 20,000 crores
            'small_cap': []       # < 5,000 crores
        }
        
        for symbol, details in stock_details.items():
            market_cap = details.get('market_cap', 0)
            market_cap_cr = market_cap / 10000000  # Convert to crores
            
            if market_cap_cr > 20000:
                market_caps['large_cap'].append(symbol)
            elif market_cap_cr > 5000:
                market_caps['mid_cap'].append(symbol)
            else:
                market_caps['small_cap'].append(symbol)
        
        return market_caps
    
    def save_universe(self, universe: Dict[str, Any]) -> None:
        """Save universe to file"""
        ensure_directory_exists(os.path.dirname(STOCK_UNIVERSE_FILE))
        
        # Save as YAML (human readable)
        with open(STOCK_UNIVERSE_FILE, 'w') as f:
            yaml.dump(universe, f, default_flow_style=False, indent=2)
        
        # Also save as JSON for programmatic access
        json_file = STOCK_UNIVERSE_FILE.replace('.yaml', '.json')
        with open(json_file, 'w') as f:
            json.dump(universe, f, indent=2)
        
        logger.info(f"Universe saved to {STOCK_UNIVERSE_FILE} and {json_file}")
    
    def generate_summary_report(self, universe: Dict[str, Any]) -> str:
        """Generate a summary report of the stock universe"""
        report = f"""
Indian Stock Universe Update Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
========
Total Stocks Processed: {universe['metadata']['total_stocks']}
Valid Stocks: {universe['metadata']['valid_stocks']}
Invalid Stocks: {universe['metadata']['invalid_stocks']}

SECTOR BREAKDOWN:
================
"""
        
        for sector, stocks in universe['stocks']['details'].items():
            if isinstance(stocks, dict):
                continue
        
        sectors = universe.get('sectors', {})
        for sector, stock_list in sectors.items():
            report += f"{sector}: {len(stock_list)} stocks\n"
        
        report += f"""
MARKET CAP BREAKDOWN:
===================
Large Cap (>20,000 Cr): {len(universe.get('market_caps', {}).get('large_cap', []))} stocks
Mid Cap (5,000-20,000 Cr): {len(universe.get('market_caps', {}).get('mid_cap', []))} stocks
Small Cap (<5,000 Cr): {len(universe.get('market_caps', {}).get('small_cap', []))} stocks

INDICES TRACKED:
===============
"""
        for index_name, symbol in universe.get('indices', {}).items():
            report += f"{index_name}: {symbol}\n"
        
        if universe['stocks']['invalid']:
            report += f"""
INVALID/PROBLEMATIC STOCKS:
==========================
{', '.join(universe['stocks']['invalid'])}
"""
        
        return report


def main():
    """Main execution function"""
    try:
        logger.info("=== Indian Stock Universe Update Started ===")
        
        manager = IndianStockUniverseManager()
        
        # Update the universe
        universe = manager.update_stock_universe()
        
        # Save the universe
        manager.save_universe(universe)
        
        # Generate and display summary
        summary = manager.generate_summary_report(universe)
        print(summary)
        logger.info("Summary report generated")
        
        # Save summary to file
        summary_file = os.path.join(DATA_DIR, 'reports', 'stock_universe_summary.txt')
        ensure_directory_exists(os.path.dirname(summary_file))
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"=== Stock Universe Update Completed Successfully ===")
        logger.info(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Error in stock universe update: {str(e)}")
        raise


if __name__ == "__main__":
    main()