#!/usr/bin/env python3
"""
Daily Analysis Runner for Indian Stock Market Research System v1.0
Created: January 2022
Author: Market Research System

This script orchestrates the daily analysis workflow for Indian stocks including:
- Data collection from NSE/BSE
- Technical indicator calculations
- Market trend analysis
- Report generation
"""

import os
import sys
import logging
import datetime
import traceback
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.data.fetch_market_data import IndianStockDataFetcher
from src.analysis.technical_indicators import TechnicalIndicatorCalculator
from src.analysis.market_analyzer import IndianMarketAnalyzer
from src.reporting.pdf_generator import DailyReportGenerator
from src.utils.logging_utils import setup_logging
from src.config.settings import INDIAN_STOCK_CONFIG


class DailyAnalysisRunner:
    """Main class to run daily analysis workflow for Indian stock market"""
    
    def __init__(self):
        """Initialize the daily analysis runner"""
        self.logger = setup_logging('daily_analysis')
        self.data_fetcher = IndianStockDataFetcher()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.market_analyzer = IndianMarketAnalyzer()
        self.report_generator = DailyReportGenerator()
        
        # Indian stock symbols - Top 50 NSE stocks
        self.nse_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS',
            'SBIN.NS', 'BAJFINANCE.NS', 'LT.NS', 'ASIANPAINT.NS', 'HCLTECH.NS',
            'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TITAN.NS',
            'NESTLEIND.NS', 'WIPRO.NS', 'POWERGRID.NS', 'NTPC.NS', 'TECHM.NS',
            'M&M.NS', 'TATAMOTORS.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'JSWSTEEL.NS',
            'GRASIM.NS', 'TATASTEEL.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'BAJAJFINSV.NS',
            'DIVISLAB.NS', 'CIPLA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'BRITANNIA.NS',
            'BPCL.NS', 'ONGC.NS', 'IOC.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS',
            'SHREE.NS', 'UPL.NS', 'TATACONSUM.NS', 'APOLLOHOSP.NS', 'GODREJCP.NS'
        ]
        
        # Indian market indices
        self.indices = ['^NSEI', '^BSESN', '^NSEBANK', '^CNXIT']  # Nifty50, Sensex, Bank Nifty, CNX IT
        
        self.today = datetime.date.today()
        
    def fetch_daily_data(self):
        """Fetch daily market data for Indian stocks and indices"""
        self.logger.info("Starting daily data collection for Indian markets")
        
        try:
            # Fetch stock data
            stock_data = {}
            for symbol in self.nse_stocks:
                try:
                    data = self.data_fetcher.get_stock_data(symbol, period='1d')
                    if not data.empty:
                        stock_data[symbol] = data
                        self.logger.info(f"Successfully fetched data for {symbol}")
                    else:
                        self.logger.warning(f"No data received for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            
            # Fetch index data
            index_data = {}
            for index in self.indices:
                try:
                    data = self.data_fetcher.get_stock_data(index, period='1d')
                    if not data.empty:
                        index_data[index] = data
                        self.logger.info(f"Successfully fetched data for {index}")
                except Exception as e:
                    self.logger.error(f"Error fetching data for {index}: {str(e)}")
            
            self.logger.info(f"Data collection completed. Stocks: {len(stock_data)}, Indices: {len(index_data)}")
            return stock_data, index_data
            
        except Exception as e:
            self.logger.error(f"Critical error in data fetching: {str(e)}")
            raise
    
    def calculate_technical_indicators(self, stock_data):
        """Calculate technical indicators for all stocks"""
        self.logger.info("Calculating technical indicators")
        
        indicators_data = {}
        
        for symbol, data in stock_data.items():
            try:
                if len(data) < 20:  # Need minimum data for indicators
                    self.logger.warning(f"Insufficient data for {symbol}, skipping indicators")
                    continue
                
                indicators = {
                    'sma_20': self.indicator_calculator.sma(data['Close'], 20),
                    'sma_50': self.indicator_calculator.sma(data['Close'], 50),
                    'ema_12': self.indicator_calculator.ema(data['Close'], 12),
                    'ema_26': self.indicator_calculator.ema(data['Close'], 26),
                    'rsi': self.indicator_calculator.rsi(data['Close']),
                    'macd': self.indicator_calculator.macd(data['Close']),
                    'bb_upper': self.indicator_calculator.bollinger_bands(data['Close'])['upper'],
                    'bb_lower': self.indicator_calculator.bollinger_bands(data['Close'])['lower'],
                    'stoch_k': self.indicator_calculator.stochastic(data['High'], data['Low'], data['Close'])['%K'],
                    'volume_sma': self.indicator_calculator.sma(data['Volume'], 20)
                }
                
                indicators_data[symbol] = indicators
                self.logger.debug(f"Calculated indicators for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
        
        self.logger.info(f"Technical indicators calculated for {len(indicators_data)} stocks")
        return indicators_data
    
    def perform_market_analysis(self, stock_data, index_data, indicators_data):
        """Perform comprehensive market analysis"""
        self.logger.info("Performing market analysis")
        
        try:
            analysis_results = {
                'market_summary': self.market_analyzer.get_market_summary(index_data),
                'top_gainers': self.market_analyzer.get_top_gainers(stock_data, limit=10),
                'top_losers': self.market_analyzer.get_top_losers(stock_data, limit=10),
                'high_volume_stocks': self.market_analyzer.get_high_volume_stocks(stock_data, limit=10),
                'technical_signals': self.market_analyzer.get_technical_signals(stock_data, indicators_data),
                'sector_performance': self.market_analyzer.analyze_sector_performance(stock_data),
                'market_breadth': self.market_analyzer.calculate_market_breadth(stock_data),
                'volatility_analysis': self.market_analyzer.analyze_volatility(index_data)
            }
            
            self.logger.info("Market analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            raise
    
    def generate_daily_report(self, stock_data, index_data, indicators_data, analysis_results):
        """Generate comprehensive daily report"""
        self.logger.info("Generating daily report")
        
        try:
            report_data = {
                'date': self.today,
                'stock_data': stock_data,
                'index_data': index_data,
                'indicators_data': indicators_data,
                'analysis_results': analysis_results,
                'total_stocks_analyzed': len(stock_data),
                'total_indices_tracked': len(index_data)
            }
            
            # Generate PDF report
            report_filename = f"daily_market_report_{self.today.strftime('%Y%m%d')}.pdf"
            report_path = Path('reports/daily') / str(self.today.year) / f"{self.today.month:02d}" / report_filename
            
            # Ensure directory exists
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.report_generator.generate_daily_report(report_data, str(report_path))
            
            self.logger.info(f"Daily report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
    
    def cleanup_old_data(self):
        """Cleanup old temporary data files"""
        self.logger.info("Performing cleanup operations")
        
        try:
            # Clean cache older than 7 days
            cache_dir = Path('data/cache/daily_cache')
            if cache_dir.exists():
                cutoff_date = datetime.date.today() - datetime.timedelta(days=7)
                for file_path in cache_dir.iterdir():
                    if file_path.is_file():
                        file_date = datetime.date.fromtimestamp(file_path.stat().st_mtime)
                        if file_date < cutoff_date:
                            file_path.unlink()
                            self.logger.debug(f"Deleted old cache file: {file_path}")
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")
    
    def run_daily_analysis(self):
        """Main method to run complete daily analysis workflow"""
        self.logger.info(f"Starting daily analysis for {self.today}")
        
        start_time = datetime.datetime.now()
        
        try:
            # Step 1: Fetch daily data
            stock_data, index_data = self.fetch_daily_data()
            
            if not stock_data and not index_data:
                self.logger.error("No data fetched, aborting analysis")
                return False
            
            # Step 2: Calculate technical indicators
            indicators_data = self.calculate_technical_indicators(stock_data)
            
            # Step 3: Perform market analysis
            analysis_results = self.perform_market_analysis(stock_data, index_data, indicators_data)
            
            # Step 4: Generate report
            report_path = self.generate_daily_report(stock_data, index_data, indicators_data, analysis_results)
            
            # Step 5: Cleanup
            self.cleanup_old_data()
            
            # Calculate execution time
            execution_time = datetime.datetime.now() - start_time
            
            self.logger.info(f"Daily analysis completed successfully in {execution_time}")
            self.logger.info(f"Report generated: {report_path}")
            
            # Print summary to console
            print(f"\n{'='*60}")
            print(f"INDIAN STOCK MARKET DAILY ANALYSIS - {self.today}")
            print(f"{'='*60}")
            print(f"Execution Time: {execution_time}")
            print(f"Stocks Analyzed: {len(stock_data)}")
            print(f"Indices Tracked: {len(index_data)}")
            print(f"Report Generated: {report_path}")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in daily analysis: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def main():
    """Main function to run daily analysis"""
    print("Indian Stock Market Research System v1.0")
    print("Daily Analysis Runner")
    print("-" * 40)
    
    try:
        # Check if market is open (basic check for Indian market hours)
        now = datetime.datetime.now()
        if now.weekday() >= 5:  # Saturday or Sunday
            print("Market is closed (Weekend)")
            print("Running analysis with latest available data...")
        
        runner = DailyAnalysisRunner()
        success = runner.run_daily_analysis()
        
        if success:
            print("Daily analysis completed successfully!")
            return 0
        else:
            print("Daily analysis failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)