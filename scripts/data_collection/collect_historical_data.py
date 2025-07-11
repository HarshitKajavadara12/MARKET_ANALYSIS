#!/usr/bin/env python3
"""
Historical Data Collection Script for Indian Stock Market
Market Research System v1.0 (2022)

This script collects historical stock data for NSE/BSE listed companies
for backtesting and analysis purposes.
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
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.settings import INDIAN_STOCK_LIST, DATA_PATHS, API_CONFIG
from utils.date_utils import get_trading_days, get_date_range
from utils.logging_utils import setup_logger
from data.data_storage import save_historical_data, ensure_directories

class IndianHistoricalDataCollector:
    """Collects historical data for Indian stocks from NSE/BSE"""
    
    def __init__(self):
        self.logger = setup_logger('historical_data_collector')
        self.stock_symbols = INDIAN_STOCK_LIST
        self.data_path = DATA_PATHS['raw']['stocks']['historical']
        self.retry_count = 3
        self.delay_between_requests = 2  # seconds
        
        # Ensure data directories exist
        ensure_directories()
        
        # Indian market indices
        self.indices = {
            'NIFTY50': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK',
            'NIFTYNEXT50': '^NSMIDCP',
            'NIFTYIT': '^CNXIT',
            'NIFTYFMCG': '^CNXFMCG',
            'NIFTYPHARMA': '^CNXPHARMA',
            'NIFTYAUTO': '^CNXAUTO',
            'NIFTYMETAL': '^CNXMETAL',
            'NIFTYENERGY': '^CNXENERGY'
        }
        
    def get_indian_stock_symbol(self, symbol: str) -> str:
        """Convert stock symbol to Yahoo Finance format for Indian stocks"""
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            # Default to NSE (.NS) for most stocks
            return f"{symbol}.NS"
        return symbol
    
    def collect_stock_historical_data(self, symbol: str, start_date: datetime.date, 
                                    end_date: datetime.date, period: str = '1d') -> Optional[pd.DataFrame]:
        """Collect historical data for a single stock"""
        yahoo_symbol = self.get_indian_stock_symbol(symbol)
        
        for attempt in range(self.retry_count):
            try:
                self.logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
                
                stock = yf.Ticker(yahoo_symbol)
                
                # Download historical data
                hist_data = stock.history(start=start_date, end=end_date, interval=period)
                
                if hist_data.empty:
                    self.logger.warning(f"No historical data available for {symbol}")
                    return None
                
                # Clean and process the data
                hist_data = hist_data.reset_index()
                hist_data['Symbol'] = symbol
                hist_data['Date'] = hist_data['Date'].dt.date
                
                # Round numerical values
                numerical_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                for col in numerical_cols:
                    if col in hist_data.columns:
                        hist_data[col] = hist_data[col].round(2)
                
                # Calculate additional metrics
                hist_data['Change'] = (hist_data['Close'] - hist_data['Open']).round(2)
                hist_data['Change_Percent'] = ((hist_data['Change'] / hist_data['Open']) * 100).round(2)
                hist_data['Day_Range'] = (hist_data['High'] - hist_data['Low']).round(2)
                hist_data['Day_Range_Percent'] = ((hist_data['Day_Range'] / hist_data['Open']) * 100).round(2)
                
                # Calculate moving averages
                hist_data['MA_5'] = hist_data['Close'].rolling(window=5).mean().round(2)
                hist_data['MA_10'] = hist_data['Close'].rolling(window=10).mean().round(2)
                hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean().round(2)
                hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean().round(2)
                
                # Calculate volatility (20-day rolling standard deviation)
                hist_data['Volatility_20'] = hist_data['Close'].pct_change().rolling(window=20).std().round(4)
                
                # Reorder columns
                column_order = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Adj Close', 
                              'Volume', 'Change', 'Change_Percent', 'Day_Range', 'Day_Range_Percent',
                              'MA_5', 'MA_10', 'MA_20', 'MA_50', 'Volatility_20']
                
                hist_data = hist_data[column_order]
                
                self.logger.info(f"Successfully collected {len(hist_data)} records for {symbol}")
                return hist_data
                
            except Exception as e:
                self.logger.error(f"Error fetching historical data for {symbol} (attempt {attempt + 1}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.delay_between_requests * (attempt + 1))
                    continue
                else:
                    return None
    
    def collect_index_historical_data(self, start_date: datetime.date, 
                                    end_date: datetime.date) -> Dict[str, pd.DataFrame]:
        """Collect historical data for major Indian indices"""
        index_data = {}
        
        for index_name, symbol in self.indices.items():
            try:
                self.logger.info(f"Fetching historical data for {index_name}")
                
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=start_date, end=end_date)
                
                if not hist_data.empty:
                    # Clean and process the data
                    hist_data = hist_data.reset_index()
                    hist_data['Index'] = index_name
                    hist_data['Symbol'] = symbol
                    hist_data['Date'] = hist_data['Date'].dt.date
                    
                    # Round numerical values
                    numerical_cols = ['Open', 'High', 'Low', 'Close']
                    for col in numerical_cols:
                        if col in hist_data.columns:
                            hist_data[col] = hist_data[col].round(2)
                    
                    # Calculate additional metrics
                    hist_data['Change'] = (hist_data['Close'] - hist_data['Open']).round(2)
                    hist_data['Change_Percent'] = ((hist_data['Change'] / hist_data['Open']) * 100).round(2)
                    hist_data['Day_Range'] = (hist_data['High'] - hist_data['Low']).round(2)
                    
                    # Calculate moving averages
                    hist_data['MA_5'] = hist_data['Close'].rolling(window=5).mean().round(2)
                    hist_data['MA_10'] = hist_data['Close'].rolling(window=10).mean().round(2)
                    hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean().round(2)
                    hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean().round(2)
                    
                    index_data[index_name] = hist_data
                    
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                self.logger.error(f"Error fetching historical data for {index_name}: {str(e)}")
                continue
        
        return index_data
    
    def collect_bulk_historical_data(self, start_date: datetime.date, end_date: datetime.date,
                                   symbols: Optional[List[str]] = None, 
                                   save_individual: bool = True,
                                   save_combined: bool = True) -> Tuple[int, int]:
        """
        Collect historical data for multiple stocks
        
        Returns:
            Tuple of (successful_count, failed_count)
        """
        if symbols is None:
            symbols = self.stock_symbols
        
        self.logger.info(f"Starting bulk historical data collection for {len(symbols)} symbols")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        successful_collections = 0
        failed_collections = 0
        all_stock_data = []
        failed_symbols = []
        
        for i, symbol in enumerate(symbols):
            self.logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
            
            stock_data = self.collect_stock_historical_data(symbol, start_date, end_date)
            
            if stock_data is not None and not stock_data.empty:
                successful_collections += 1
                
                if save_individual:
                    # Save individual stock data
                    filename = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                    file_path = os.path.join(self.data_path, filename)
                    stock_data.to_csv(file_path, index=False)
                    self.logger.info(f"Saved individual data for {symbol} to {file_path}")
                
                if save_combined:
                    all_stock_data.append(stock_data)
                
            else:
                failed_collections += 1
                failed_symbols.append(symbol)
                self.logger.warning(f"Failed to collect data for {symbol}")
            
            # Add delay to avoid rate limiting
            time.sleep(self.delay_between_requests)
        
        # Save combined data
        if save_combined and all_stock_data:
            combined_df = pd.concat(all_stock_data, ignore_index=True)
            combined_filename = f"combined_historical_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            combined_file_path = os.path.join(self.data_path, combined_filename)
            combined_df.to_csv(combined_file_path, index=False)
            self.logger.info(f"Saved combined historical data to {combined_file_path}")
        
        # Collect and save index data
        self.logger.info("Collecting historical index data...")
        index_data = self.collect_index_historical_data(start_date, end_date)
        
        if index_data:
            # Save individual index data
            for index_name, data in index_data.items():
                index_filename = f"{index_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                index_file_path = os.path.join(DATA_PATHS['raw']['indices']['historical'], index_filename)
                data.to_csv(index_file_path, index=False)
                self.logger.info(f"Saved historical data for {index_name}")
            
            # Save combined index data
            if len(index_data) > 1:
                combined_index_df = pd.concat(index_data.values(), ignore_index=True)
                combined_index_filename = f"combined_indices_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                combined_index_path = os.path.join(DATA_PATHS['raw']['indices']['historical'], combined_index_filename)
                combined_index_df.to_csv(combined_index_path, index=False)
        
        # Generate and save summary report
        summary = {
            'collection_period': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'total_days': (end_date - start_date).days + 1
            },
            'collection_stats': {
                'total_symbols_attempted': len(symbols),
                'successful_collections': successful_collections,
                'failed_collections': failed_collections,
                'success_rate': round((successful_collections / len(symbols)) * 100, 2)
            },
            'failed_symbols': failed_symbols,
            'indices_collected': list(index_data.keys()) if index_data else [],
            'files_created': {
                'individual_stock_files': successful_collections if save_individual else 0,
                'combined_stock_file': 1 if save_combined and all_stock_data else 0,
                'index_files': len(index_data) if index_data else 0
            },
            'collection_timestamp': datetime.datetime.now().isoformat(),
            'total_records_collected': sum(len(data) for data in all_stock_data) if all_stock_data else 0
        }
        
        summary_filename = f"historical_collection_summary_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        summary_file_path = os.path.join(self.data_path, summary_filename)
        
        with open(summary_file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Historical data collection completed")
        self.logger.info(f"Success rate: {summary['collection_stats']['success_rate']}%")
        self.logger.info(f"Total records collected: {summary['total_records_collected']}")
        
        return successful_collections, failed_collections
    
    def update_existing_data(self, symbol: str, last_date: datetime.date) -> bool:
        """Update existing historical data with recent data"""
        try:
            today = datetime.date.today()
            if last_date >= today:
                self.logger.info(f"Data for {symbol} is already up to date")
                return True
            
            # Collect data from day after last_date to today
            start_date = last_date + datetime.timedelta(days=1)
            new_data = self.collect_stock_historical_data(symbol, start_date, today)
            
            if new_data is not None and not new_data.empty:
                # Load existing data
                existing_files = [f for f in os.listdir(self.data_path) if f.startswith(symbol) and f.endswith('.csv')]
                
                if existing_files:
                    existing_file = existing_files[0]  # Take the first matching file
                    existing_path = os.path.join(self.data_path, existing_file)
                    existing_data = pd.read_csv(existing_path)
                    existing_data['Date'] = pd.to_datetime(existing_data['Date']).dt.date
                    
                    # Combine old and new data
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                    combined_data = combined_data.drop_duplicates(subset=['Date'], keep='last')
                    combined_data = combined_data.sort_values('Date')
                    
                    # Save updated data
                    combined_data.to_csv(existing_path, index=False)
                    self.logger.info(f"Updated historical data for {symbol} with {len(new_data)} new records")
                    return True
                else:
                    # No existing file, save new data
                    filename = f"{symbol}_{start_date.strftime('%Y%m%d')}_{today.strftime('%Y%m%d')}.csv"
                    file_path = os.path.join(self.data_path, filename)
                    new_data.to_csv(file_path, index=False)
                    self.logger.info(f"Created new historical data file for {symbol}")
                    return True
            else:
                self.logger.warning(f"No new data available for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating data for {symbol}: {str(e)}")
            return False

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect historical Indian stock market data')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--symbols', type=str, nargs='+', help='Specific symbols to collect (optional)')
    parser.add_argument('--no-individual', action='store_true', help='Skip saving individual stock files')
    parser.add_argument('--no-combined', action='store_true', help='Skip saving combined file')
    parser.add_argument('--update', action='store_true', help='Update existing data')
    
    args = parser.parse_args()
    
    collector = IndianHistoricalDataCollector()
    
    start_date = datetime.datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(args.end, '%Y-%m-%d').date() if args.end else datetime.date.today()
    
    if args.update:
        # Update mode - update existing data for all symbols
        success_count = 0
        symbols = args.symbols if args.symbols else collector.stock_symbols
        
        for symbol in symbols:
            if collector.update_existing_data(symbol, start_date):
                success_count += 1
        
        print(f"Successfully updated {success_count}/{len(symbols)} symbols")
    else:
        # Regular collection mode
        success, failed = collector.collect_bulk_historical_data(
            start_date=start_date,
            end_date=end_date,
            symbols=args.symbols,
            save_individual=not args.no_individual,
            save_combined=not args.no_combined
        )
        
        print(f"\n{'='*50}")
        print(f"HISTORICAL DATA COLLECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total symbols processed: {success + failed}")
        print(f"Successful collections: {success}")
        print(f"Failed collections: {failed}")
        print(f"Success rate: {round((success / (success + failed)) * 100, 2)}%")
        print(f"Data collection period: {start_date} to {end_date}")
        print(f"{'='*50}")

def collect_economic_data(start_date: datetime.date, end_date: datetime.date):
    """Collect Indian economic indicators from FRED and other sources"""
    import yfinance as yf
    
    logger = setup_logger('economic_data_collector')
    
    # Economic indicators mapping
    economic_indicators = {
        'USD_INR': 'USDINR=X',  # USD to INR exchange rate
        'GOLD_INR': 'GC=F',     # Gold futures (will convert to INR)
        'CRUDE_OIL': 'CL=F',    # Crude oil futures
        'NIFTY_VIX': '^NSEBANK', # Bank Nifty as proxy for volatility
    }
    
    # Additional manual indicators (these would need to be manually updated)
    rbi_indicators = {
        'repo_rate': 6.50,  # Current repo rate as of 2022
        'inflation_rate': 6.95,  # CPI inflation
        'gdp_growth': 8.7,  # GDP growth rate
        'fiscal_deficit': 6.4,  # Fiscal deficit as % of GDP
    }
    
    economic_data = {}
    
    for indicator, symbol in economic_indicators.items():
        try:
            logger.info(f"Collecting data for {indicator}")
            
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if not hist_data.empty:
                hist_data = hist_data.reset_index()
                hist_data['Indicator'] = indicator
                hist_data['Date'] = hist_data['Date'].dt.date
                
                # Keep only relevant columns
                hist_data = hist_data[['Date', 'Indicator', 'Close']]
                hist_data.rename(columns={'Close': 'Value'}, inplace=True)
                hist_data['Value'] = hist_data['Value'].round(4)
                
                economic_data[indicator] = hist_data
                logger.info(f"Collected {len(hist_data)} records for {indicator}")
            
            time.sleep(2)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error collecting {indicator}: {str(e)}")
    
    # Save economic data
    if economic_data:
        # Save individual files
        econ_data_path = DATA_PATHS['raw']['economic']
        os.makedirs(econ_data_path, exist_ok=True)
        
        for indicator, data in economic_data.items():
            filename = f"{indicator}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            file_path = os.path.join(econ_data_path, filename)
            data.to_csv(file_path, index=False)
            logger.info(f"Saved {indicator} data to {file_path}")
        
        # Save combined economic data
        combined_econ_df = pd.concat(economic_data.values(), ignore_index=True)
        combined_econ_filename = f"combined_economic_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        combined_econ_path = os.path.join(econ_data_path, combined_econ_filename)
        combined_econ_df.to_csv(combined_econ_path, index=False)
        
        # Save RBI indicators as JSON
        rbi_data = {
            'date': end_date.strftime('%Y-%m-%d'),
            'indicators': rbi_indicators,
            'source': 'RBI_Manual_Entry',
            'note': 'These are approximate values as of 2022 and should be updated manually'
        }
        
        rbi_filename = f"rbi_indicators_{end_date.strftime('%Y%m%d')}.json"
        rbi_file_path = os.path.join(econ_data_path, rbi_filename)
        
        with open(rbi_file_path, 'w') as f:
            json.dump(rbi_data, f, indent=2)
        
        logger.info(f"Economic data collection completed. {len(economic_data)} indicators collected.")
        return len(economic_data)
    
    return 0

def validate_collected_data(symbol: str, start_date: datetime.date, end_date: datetime.date) -> Dict:
    """Validate the quality of collected data"""
    logger = setup_logger('data_validator')
    
    try:
        # Find the data file
        data_files = [f for f in os.listdir(DATA_PATHS['raw']['stocks']['historical']) 
                     if f.startswith(symbol) and f.endswith('.csv')]
        
        if not data_files:
            return {'status': 'error', 'message': f'No data file found for {symbol}'}
        
        file_path = os.path.join(DATA_PATHS['raw']['stocks']['historical'], data_files[0])
        df = pd.read_csv(file_path)
        
        # Convert Date column
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Validation checks
        validation_results = {
            'symbol': symbol,
            'file_path': file_path,
            'total_records': len(df),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            },
            'missing_dates': [],
            'data_quality': {},
            'anomalies': [],
            'status': 'passed'
        }
        
        # Check for missing dates (trading days)
        expected_dates = get_trading_days(start_date, end_date)
        actual_dates = set(df['Date'].tolist())
        missing_dates = [d for d in expected_dates if d not in actual_dates]
        
        if missing_dates:
            validation_results['missing_dates'] = [d.strftime('%Y-%m-%d') for d in missing_dates[:10]]  # Show first 10
            validation_results['missing_dates_count'] = len(missing_dates)
        
        # Data quality checks
        validation_results['data_quality'] = {
            'null_values': df.isnull().sum().to_dict(),
            'zero_volume_days': len(df[df['Volume'] == 0]),
            'negative_prices': len(df[(df['Open'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0) | (df['Close'] <= 0)]),
            'high_low_anomalies': len(df[df['High'] < df['Low']]),
            'ohlc_anomalies': len(df[(df['Open'] > df['High']) | (df['Open'] < df['Low']) | 
                                   (df['Close'] > df['High']) | (df['Close'] < df['Low'])])
        }
        
        # Check for price anomalies (sudden jumps > 20%)
        df['daily_return'] = df['Close'].pct_change()
        anomalous_returns = df[abs(df['daily_return']) > 0.20]
        
        if not anomalous_returns.empty:
            validation_results['anomalies'] = [
                {
                    'date': row['Date'].strftime('%Y-%m-%d'),
                    'return': round(row['daily_return'] * 100, 2),
                    'close_price': row['Close']
                }
                for _, row in anomalous_returns.head(5).iterrows()
            ]
        
        # Determine overall status
        issues = []
        if validation_results['data_quality']['negative_prices'] > 0:
            issues.append('negative_prices')
        if validation_results['data_quality']['high_low_anomalies'] > 0:
            issues.append('high_low_anomalies')
        if validation_results['data_quality']['ohlc_anomalies'] > 0:
            issues.append('ohlc_anomalies')
        if len(missing_dates) > 10:  # Allow some missing dates
            issues.append('too_many_missing_dates')
        
        if issues:
            validation_results['status'] = 'warning'
            validation_results['issues'] = issues
        
        logger.info(f"Validation completed for {symbol}. Status: {validation_results['status']}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating data for {symbol}: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def generate_collection_report(start_date: datetime.date, end_date: datetime.date):
    """Generate a comprehensive collection report"""
    logger = setup_logger('report_generator')
    
    try:
        # Find all summary files
        summary_files = [f for f in os.listdir(DATA_PATHS['raw']['stocks']['historical'])
                        if f.startswith('historical_collection_summary_') and f.endswith('.json')]
        
        if not summary_files:
            logger.warning("No collection summary files found")
            return
        
        # Load the most recent summary
        latest_summary_file = max(summary_files)
        summary_path = os.path.join(DATA_PATHS['raw']['stocks']['historical'], latest_summary_file)
        
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        # Generate detailed report
        report = {
            'report_metadata': {
                'generated_at': datetime.datetime.now().isoformat(),
                'report_type': 'Historical Data Collection Report',
                'version': '1.0',
                'period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                }
            },
            'collection_summary': summary_data,
            'data_files_created': [],
            'validation_results': [],
            'recommendations': []
        }
        
        # List all created files
        data_dir = DATA_PATHS['raw']['stocks']['historical']
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        for file in csv_files:
            file_path = os.path.join(data_dir, file)
            file_size = os.path.getsize(file_path)
            
            report['data_files_created'].append({
                'filename': file,
                'size_mb': round(file_size / (1024*1024), 2),
                'created_at': datetime.datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
            })
        
        # Run validation on sample of stocks
        sample_symbols = summary_data['failed_symbols'][:5] if summary_data['failed_symbols'] else []
        if not sample_symbols and 'successful_collections' in summary_data['collection_stats']:
            # Get first 5 successful symbols for validation
            successful_files = [f for f in csv_files if not f.startswith('combined_')][:5]
            sample_symbols = [f.split('_')[0] for f in successful_files]
        
        for symbol in sample_symbols:
            validation_result = validate_collected_data(symbol, start_date, end_date)
            report['validation_results'].append(validation_result)
        
        # Generate recommendations
        success_rate = summary_data['collection_stats']['success_rate']
        
        if success_rate < 80:
            report['recommendations'].append({
                'priority': 'high',
                'category': 'data_collection',
                'message': 'Low success rate detected. Consider checking network connectivity and API limits.'
            })
        
        if summary_data['collection_stats']['failed_collections'] > 0:
            report['recommendations'].append({
                'priority': 'medium',
                'category': 'retry_failed',
                'message': f"Retry collection for {len(summary_data['failed_symbols'])} failed symbols."
            })
        
        if len(report['data_files_created']) > 100:
            report['recommendations'].append({
                'priority': 'low',
                'category': 'optimization',
                'message': 'Consider implementing data compression or archival strategy.'
            })
        
        # Save the report
        report_filename = f"collection_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        report_path = os.path.join(DATA_PATHS['reports'], report_filename)
        
        os.makedirs(DATA_PATHS['reports'], exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Collection report saved to {report_path}")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print(f"DETAILED COLLECTION REPORT")
        print(f"{'='*60}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Total files created: {len(report['data_files_created'])}")
        print(f"Total data size: {sum(f['size_mb'] for f in report['data_files_created']):.2f} MB")
        print(f"Success rate: {success_rate}%")
        print(f"Validation checks: {len(report['validation_results'])} symbols validated")
        print(f"Recommendations: {len(report['recommendations'])} items")
        print(f"Full report saved to: {report_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error generating collection report: {str(e)}")

if __name__ == "__main__":
    # Enhanced main execution with additional features
    main()
    
    # After main execution, run additional tasks
    import argparse
    import sys
    
    # Parse arguments again for post-processing
    parser = argparse.ArgumentParser(description='Collect historical Indian stock market data')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--symbols', type=str, nargs='+', help='Specific symbols to collect (optional)')
    parser.add_argument('--no-individual', action='store_true', help='Skip saving individual stock files')
    parser.add_argument('--no-combined', action='store_true', help='Skip saving combined file')
    parser.add_argument('--update', action='store_true', help='Update existing data')
    parser.add_argument('--collect-economic', action='store_true', help='Also collect economic indicators')
    parser.add_argument('--generate-report', action='store_true', help='Generate detailed collection report')
    parser.add_argument('--validate-sample', action='store_true', help='Validate a sample of collected data')
    
    args = parser.parse_args()
    
    start_date = datetime.datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(args.end, '%Y-%m-%d').date() if args.end else datetime.date.today()
    
    # Collect economic data if requested
    if args.collect_economic:
        print("\nCollecting economic indicators...")
        econ_count = collect_economic_data(start_date, end_date)
        print(f"Collected {econ_count} economic indicators")
    
    # Generate report if requested
    if args.generate_report:
        print("\nGenerating collection report...")
        generate_collection_report(start_date, end_date)
    
    # Validate sample if requested
    if args.validate_sample and args.symbols:
        print("\nValidating sample data...")
        for symbol in args.symbols[:3]:  # Validate first 3 symbols
            result = validate_collected_data(symbol, start_date, end_date)
            print(f"Validation for {symbol}: {result['status']}")
            if result['status'] == 'warning':
                print(f"  Issues: {', '.join(result.get('issues', []))}")
    
    print("\nData collection process completed!")
    print("Next steps:")
    print("1. Run data cleaning scripts")
    print("2. Calculate technical indicators")
    print("3. Generate analysis reports")
    print("4. Set up automated daily updates")