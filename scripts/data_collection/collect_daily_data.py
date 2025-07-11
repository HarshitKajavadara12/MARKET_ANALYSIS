#!/usr/bin/env python3
"""
Daily Data Collection Script for Indian Stock Market
Market Research System v1.0 (2022)

This script collects daily stock data for NSE/BSE listed companies
and saves it in structured format for analysis.
"""

import os
import sys
import datetime
import pandas as pd
import yfinance as yf
import logging
from pathlib import Path
import json
import time
import requests
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.settings import INDIAN_STOCK_LIST, DATA_PATHS, API_CONFIG
from utils.date_utils import get_trading_days, is_trading_day
from utils.logging_utils import setup_logger
from data.data_storage import save_daily_data, ensure_directories

class IndianStockDataCollector:
    """Collects daily data for Indian stocks from NSE/BSE"""
    
    def __init__(self):
        self.logger = setup_logger('daily_data_collector')
        self.stock_symbols = INDIAN_STOCK_LIST
        self.data_path = DATA_PATHS['raw']['stocks']['daily']
        self.retry_count = 3
        self.delay_between_requests = 1  # seconds
        
        # Ensure data directories exist
        ensure_directories()
        
        # Indian market indices
        self.indices = {
            'NIFTY50': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK',
            'NIFTYNEXT50': '^NSMIDCP',
            'NIFTYIT': '^CNXIT'
        }
        
    def get_indian_stock_symbol(self, symbol: str) -> str:
        """Convert stock symbol to Yahoo Finance format for Indian stocks"""
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            # Default to NSE (.NS) for most stocks
            return f"{symbol}.NS"
        return symbol
    
    def collect_stock_data(self, symbol: str, date: datetime.date) -> Optional[Dict]:
        """Collect daily data for a single stock"""
        yahoo_symbol = self.get_indian_stock_symbol(symbol)
        
        for attempt in range(self.retry_count):
            try:
                self.logger.info(f"Fetching data for {symbol} (attempt {attempt + 1})")
                
                # Fetch data for the specific date
                stock = yf.Ticker(yahoo_symbol)
                
                # Get data for the last 5 trading days to ensure we get the required date
                end_date = date + datetime.timedelta(days=1)
                start_date = date - datetime.timedelta(days=7)
                
                hist_data = stock.history(start=start_date, end=end_date)
                
                if hist_data.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    return None
                
                # Get the data for the specific date
                target_date = pd.Timestamp(date).tz_localize('Asia/Kolkata')
                
                # Find the closest trading day data
                if target_date.strftime('%Y-%m-%d') in hist_data.index.strftime('%Y-%m-%d'):
                    day_data = hist_data[hist_data.index.strftime('%Y-%m-%d') == target_date.strftime('%Y-%m-%d')]
                else:
                    # Get the most recent data
                    day_data = hist_data.tail(1)
                
                if day_data.empty:
                    self.logger.warning(f"No data for {symbol} on {date}")
                    return None
                
                # Extract the data
                data_row = day_data.iloc[0]
                
                stock_data = {
                    'symbol': symbol,
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(float(data_row['Open']), 2),
                    'high': round(float(data_row['High']), 2),
                    'low': round(float(data_row['Low']), 2),
                    'close': round(float(data_row['Close']), 2),
                    'volume': int(data_row['Volume']),
                    'adj_close': round(float(data_row['Adj Close']), 2),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Add basic calculations
                stock_data['change'] = round(stock_data['close'] - stock_data['open'], 2)
                stock_data['change_percent'] = round((stock_data['change'] / stock_data['open']) * 100, 2)
                stock_data['day_range'] = stock_data['high'] - stock_data['low']
                
                self.logger.info(f"Successfully collected data for {symbol}")
                return stock_data
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol} (attempt {attempt + 1}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.delay_between_requests * (attempt + 1))
                    continue
                else:
                    return None
    
    def collect_index_data(self, date: datetime.date) -> Dict:
        """Collect data for major Indian indices"""
        index_data = {}
        
        for index_name, symbol in self.indices.items():
            try:
                self.logger.info(f"Fetching index data for {index_name}")
                
                ticker = yf.Ticker(symbol)
                end_date = date + datetime.timedelta(days=1)
                start_date = date - datetime.timedelta(days=7)
                
                hist_data = ticker.history(start=start_date, end=end_date)
                
                if not hist_data.empty:
                    # Get the most recent data
                    recent_data = hist_data.tail(1).iloc[0]
                    
                    index_data[index_name] = {
                        'symbol': symbol,
                        'date': date.strftime('%Y-%m-%d'),
                        'open': round(float(recent_data['Open']), 2),
                        'high': round(float(recent_data['High']), 2),
                        'low': round(float(recent_data['Low']), 2),
                        'close': round(float(recent_data['Close']), 2),
                        'volume': int(recent_data['Volume']),
                        'change': round(float(recent_data['Close']) - float(recent_data['Open']), 2),
                        'change_percent': round(((float(recent_data['Close']) - float(recent_data['Open'])) / float(recent_data['Open'])) * 100, 2)
                    }
                    
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                self.logger.error(f"Error fetching index data for {index_name}: {str(e)}")
                continue
        
        return index_data
    
    def collect_daily_data(self, target_date: Optional[datetime.date] = None) -> bool:
        """Main function to collect daily data for all stocks"""
        if target_date is None:
            target_date = datetime.date.today()
        
        # Check if it's a trading day
        if not is_trading_day(target_date):
            self.logger.info(f"{target_date} is not a trading day, skipping data collection")
            return True
        
        self.logger.info(f"Starting daily data collection for {target_date}")
        
        collected_stocks = []
        failed_stocks = []
        
        # Collect stock data
        for i, symbol in enumerate(self.stock_symbols):
            self.logger.info(f"Processing {symbol} ({i+1}/{len(self.stock_symbols)})")
            
            stock_data = self.collect_stock_data(symbol, target_date)
            
            if stock_data:
                collected_stocks.append(stock_data)
            else:
                failed_stocks.append(symbol)
            
            # Add delay to avoid rate limiting
            time.sleep(self.delay_between_requests)
        
        # Collect index data
        self.logger.info("Collecting index data...")
        index_data = self.collect_index_data(target_date)
        
        # Save collected data
        if collected_stocks:
            # Save stock data
            stock_df = pd.DataFrame(collected_stocks)
            stock_file_path = os.path.join(
                self.data_path, 
                f"stocks_{target_date.strftime('%Y%m%d')}.csv"
            )
            stock_df.to_csv(stock_file_path, index=False)
            self.logger.info(f"Saved {len(collected_stocks)} stock records to {stock_file_path}")
            
            # Save index data
            if index_data:
                index_file_path = os.path.join(
                    DATA_PATHS['raw']['indices']['daily'],
                    f"indices_{target_date.strftime('%Y%m%d')}.json"
                )
                with open(index_file_path, 'w') as f:
                    json.dump(index_data, f, indent=2)
                self.logger.info(f"Saved index data to {index_file_path}")
            
            # Generate summary report
            summary = {
                'date': target_date.strftime('%Y-%m-%d'),
                'total_stocks_attempted': len(self.stock_symbols),
                'successful_collections': len(collected_stocks),
                'failed_collections': len(failed_stocks),
                'success_rate': round((len(collected_stocks) / len(self.stock_symbols)) * 100, 2),
                'failed_symbols': failed_stocks,
                'indices_collected': list(index_data.keys()) if index_data else [],
                'collection_timestamp': datetime.datetime.now().isoformat()
            }
            
            summary_file_path = os.path.join(
                DATA_PATHS['raw']['stocks']['daily'],
                f"collection_summary_{target_date.strftime('%Y%m%d')}.json"
            )
            
            with open(summary_file_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Collection Summary: {summary['success_rate']}% success rate")
            self.logger.info(f"Successfully collected: {len(collected_stocks)} stocks")
            self.logger.info(f"Failed to collect: {len(failed_stocks)} stocks")
            
            return True
        else:
            self.logger.error("No data was collected successfully")
            return False

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect daily Indian stock market data')
    parser.add_argument('--date', type=str, help='Date to collect data for (YYYY-MM-DD)', default=None)
    parser.add_argument('--backfill', type=int, help='Number of days to backfill', default=0)
    
    args = parser.parse_args()
    
    collector = IndianStockDataCollector()
    
    if args.date:
        target_date = datetime.datetime.strptime(args.date, '%Y-%m-%d').date()
        success = collector.collect_daily_data(target_date)
    elif args.backfill > 0:
        # Backfill multiple days
        success_count = 0
        for i in range(args.backfill):
            target_date = datetime.date.today() - datetime.timedelta(days=i)
            if collector.collect_daily_data(target_date):
                success_count += 1
        
        print(f"Successfully collected data for {success_count}/{args.backfill} days")
        success = success_count > 0
    else:
        # Collect for today
        success = collector.collect_daily_data()
    
    if success:
        print("Daily data collection completed successfully")
        sys.exit(0)
    else:
        print("Daily data collection failed")
        sys.exit(1)

if __name__ == "__main__":
    main()