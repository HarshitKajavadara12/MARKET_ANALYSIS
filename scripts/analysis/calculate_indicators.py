#!/usr/bin/env python3
"""
Technical Indicators Calculator for Indian Stock Market Research System v1.0
Created: March 2022
Author: Market Research System

This script calculates various technical indicators for Indian stocks including:
- Moving Averages (SMA, EMA)
- Momentum Indicators (RSI, MACD, Stochastic)
- Volatility Indicators (Bollinger Bands, ATR)
- Volume Indicators
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.data.fetch_market_data import IndianStockDataFetcher
from src.analysis.technical_indicators import TechnicalIndicatorCalculator
from src.utils.logging_utils import setup_logging
from src.utils.file_utils import save_to_csv, load_from_csv


class IndicatorBatchCalculator:
    """Batch calculator for technical indicators on Indian stocks"""
    
    def __init__(self):
        """Initialize the indicator batch calculator"""
        self.logger = setup_logging('indicator_calculator')
        self.data_fetcher = IndianStockDataFetcher()
        self.calculator = TechnicalIndicatorCalculator()
        
        # Top NSE stocks for calculation
        self.nse_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'HDFC.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS',
            'SBIN.NS', 'BAJFINANCE.NS', 'LT.NS', 'ASIANPAINT.NS', 'HCLTECH.NS',
            'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'TITAN.NS',
            'NESTLEIND.NS', 'WIPRO.NS', 'POWERGRID.NS', 'NTPC.NS', 'TECHM.NS',
            'M&M.NS', 'TATAMOTORS.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'JSWSTEEL.NS'
        ]
        
    def calculate_single_stock_indicators(self, symbol, period='6mo'):
        """Calculate all technical indicators for a single stock"""
        self.logger.info(f"Calculating indicators for {symbol}")
        
        try:
            # Fetch stock data
            data = self.data_fetcher.get_stock_data(symbol, period=period)
            
            if data.empty or len(data) < 50:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate all indicators
            indicators = {}
            
            # Moving Averages
            indicators['SMA_10'] = self.calculator.sma(data['Close'], 10)
            indicators['SMA_20'] = self.calculator.sma(data['Close'], 20)
            indicators['SMA_50'] = self.calculator.sma(data['Close'], 50)
            indicators['SMA_200'] = self.calculator.sma(data['Close'], 200)
            
            indicators['EMA_12'] = self.calculator.ema(data['Close'], 12)
            indicators['EMA_26'] = self.calculator.ema(data['Close'], 26)
            indicators['EMA_50'] = self.calculator.ema(data['Close'], 50)
            
            # Momentum Indicators
            indicators['RSI'] = self.calculator.rsi(data['Close'])
            
            macd_data = self.calculator.macd(data['Close'])
            indicators['MACD'] = macd_data['MACD']
            indicators['MACD_Signal'] = macd_data['Signal']
            indicators['MACD_Histogram'] = macd_data['Histogram']
            
            stoch_data = self.calculator.stochastic(data['High'], data['Low'], data['Close'])
            indicators['Stoch_K'] = stoch_data['%K']
            indicators['Stoch_D'] = stoch_data['%D']
            
            # Volatility Indicators
            bb_data = self.calculator.bollinger_bands(data['Close'])
            indicators['BB_Upper'] = bb_data['upper']
            indicators['BB_Middle'] = bb_data['middle']
            indicators['BB_Lower'] = bb_data['lower']
            indicators['BB_Width'] = bb_data['width']
            
            indicators['ATR'] = self.calculator.atr(data['High'], data['Low'], data['Close'])
            
            # Volume Indicators
            indicators['Volume_SMA'] = self.calculator.sma(data['Volume'], 20)
            indicators['Volume_Ratio'] = data['Volume'] / indicators['Volume_SMA']
            
            # Price-based Indicators
            indicators['Williams_R'] = self.calculator.williams_r(data['High'], data['Low'], data['Close'])
            indicators['CCI'] = self.calculator.cci(data['High'], data['Low'], data['Close'])
            
            # Create comprehensive dataframe
            result_df = pd.DataFrame(index=data.index)
            result_df['Open'] = data['Open']
            result_df['High'] = data['High']
            result_df['Low'] = data['Low']
            result_df['Close'] = data['Close']
            result_df['Volume'] = data['Volume']
            
            # Add all indicators
            for indicator_name, indicator_values in indicators.items():
                result_df[indicator_name] = indicator_values
            
            # Add trading signals
            result_df = self._add_trading_signals(result_df)
            
            self.logger.info(f"Successfully calculated indicators for {symbol}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
            return None
    
    def _add_trading_signals(self, df):
        """Add trading signals based on technical indicators"""
        
        # RSI signals
        df['RSI_Overbought'] = df['RSI'] > 70
        df['RSI_Oversold'] = df['RSI'] < 30
        
        # MACD signals
        df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        df['MACD_Bearish'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        
        # Moving Average signals
        df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
        df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))
        
        # Bollinger Band signals
        df['BB_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(20).quantile(0.1)
        df['BB_Upper_Touch'] = df['Close'] >= df['BB_Upper']
        df['BB_Lower_Touch'] = df['Close'] <= df['BB_Lower']
        
        # Price position relative to moving averages
        df['Above_SMA20'] = df['Close'] > df['SMA_20']
        df['Above_SMA50'] = df['Close'] > df['SMA_50']
        df['Above_EMA12'] = df['Close'] > df['EMA_12']
        
        # Volume signals
        df['High_Volume'] = df['Volume_Ratio'] > 1.5
        df['Low_Volume'] = df['Volume_Ratio'] < 0.5
        
        return df
    
    def calculate_batch_indicators(self, symbols=None, period='6mo', save_individual=True):
        """Calculate indicators for multiple stocks in batch"""
        if symbols is None:
            symbols = self.nse_stocks
        
        self.logger.info(f"Starting batch calculation for {len(symbols)} stocks")
        
        batch_results = {}
        successful_calculations = 0
        failed_calculations = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"Processing {symbol} ({i}/{len(symbols)})")
            
            try:
                result_df = self.calculate_single_stock_indicators(symbol, period)
                
                if result_df is not None:
                    batch_results[symbol] = result_df
                    successful_calculations += 1
                    
                    if save_individual:
                        # Save individual stock indicators
                        output_dir = Path('data/processed/technical_indicators')
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        filename = f"{symbol.replace('.NS', '')}_indicators_{datetime.now().strftime('%Y%m%d')}.csv"
                        filepath = output_dir / filename
                        
                        result_df.to_csv(filepath)
                        self.logger.info(f"Saved indicators for {symbol} to {filepath}")
                else:
                    failed_calculations += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process {symbol}: {str(e)}")
                failed_calculations += 1
        
        self.logger.info(f"Batch calculation completed. Success: {successful_calculations}, Failed: {failed_calculations}")
        
        return batch_results
    
    def generate_indicator_summary(self, batch_results):
        """Generate summary of indicators across all stocks"""
        self.logger.info("Generating indicator summary")
        
        try:
            summary_data = []
            
            for symbol, df in batch_results.items():
                if df.empty:
                    continue
                
                latest_data = df.iloc[-1]  # Latest values
                
                summary_row = {
                    'Symbol': symbol,
                    'Date': latest_data.name,
                    'Close': latest_data['Close'],
                    'RSI': latest_data['RSI'],
                    'MACD': latest_data['MACD'],
                    'MACD_Signal': latest_data['MACD_Signal'],
                    'SMA_20': latest_data['SMA_20'],
                    'SMA_50': latest_data['SMA_50'],
                    'EMA_12': latest_data['EMA_12'],
                    'BB_Upper': latest_data['BB_Upper'],
                    'BB_Lower': latest_data['BB_Lower'],
                    'ATR': latest_data['ATR'],
                    'Volume_Ratio': latest_data['Volume_Ratio'],
                    'RSI_Overbought': latest_data['RSI_Overbought'],
                    'RSI_Oversold': latest_data['RSI_Oversold'],
                    'Above_SMA20': latest_data['Above_SMA20'],
                    'Above_SMA50': latest_data['Above_SMA50'],
                    'High_Volume': latest_data['High_Volume'],
                    'MACD_Bullish': latest_data['MACD_Bullish'],
                    'Golden_Cross': latest_data['Golden_Cross']
                }
                
                summary_data.append(summary_row)
            
            summary_df = pd.DataFrame(summary_data)
            
            # Save summary
            output_dir = Path('data/processed/technical_indicators')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            summary_filename = f"indicators_summary_{datetime.now().strftime('%Y%m%d')}.csv"
            summary_filepath = output_dir / summary_filename
            
            summary_df.to_csv(summary_filepath, index=False)
            
            self.logger.info(f"Indicator summary saved to {summary_filepath}")
            
            return summary_df
            
        except Exception as e:
            self.logger.error(f"Error generating indicator summary: {str(e)}")
            return None
    
    def identify_trading_opportunities(self, batch_results):
        """Identify potential trading opportunities based on technical indicators"""
        self.logger.info("Identifying trading opportunities")
        
        opportunities = {
            'oversold_stocks': [],
            'overbought_stocks': [],
            'golden_cross_stocks': [],
            'bullish_macd_stocks': [],
            'high_volume_breakouts': [],
            'bollinger_squeeze': []
        }
        
        for symbol, df in batch_results.items():
            if df.empty:
                continue
            
            latest = df.iloc[-1]
            
            # Oversold stocks (RSI < 30)
            if latest['RSI_Oversold']:
                opportunities['oversold_stocks'].append({
                    'symbol': symbol,
                    'rsi': latest['RSI'],
                    'close': latest['Close']
                })
            
            # Overbought stocks (RSI > 70)
            if latest['RSI_Overbought']:
                opportunities['overbought_stocks'].append({
                    'symbol': symbol,
                    'rsi': latest['RSI'],
                    'close': latest['Close']
                })
            
            # Golden cross (SMA50 > SMA200)
            if latest['Golden_Cross']:
                opportunities['golden_cross_stocks'].append({
                    'symbol': symbol,
                    'sma_50': latest['SMA_50'],
                    'close': latest['Close']
                })
            
            # Bullish MACD crossover
            if latest['MACD_Bullish']:
                opportunities['bullish_macd_stocks'].append({
                    'symbol': symbol,
                    'macd': latest['MACD'],
                    'signal': latest['MACD_Signal']
                })
            
            # High volume with price above SMA20
            if latest['High_Volume'] and latest['Above_SMA20']:
                opportunities['high_volume_breakouts'].append({
                    'symbol': symbol,
                    'volume_ratio': latest['Volume_Ratio'],
                    'close': latest['Close']
                })
            
            # Bollinger band squeeze
            if latest['BB_Squeeze']:
                opportunities['bollinger_squeeze'].append({
                    'symbol': symbol,
                    'bb_width': latest['BB_Width'],
                    'close': latest['Close']
                })
        
        # Save opportunities
        output_dir = Path('data/processed/technical_indicators')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        opportunities_filename = f"trading_opportunities_{datetime.now().strftime('%Y%m%d')}.csv"
        opportunities_filepath = output_dir / opportunities_filename
        
        # Convert to DataFrame for saving
        all_opportunities = []
        for opp_type, stocks in opportunities.items():
            for stock in stocks:
                stock['opportunity_type'] = opp_type
                all_opportunities.append(stock)
        
        if all_opportunities:
            opportunities_df = pd.DataFrame(all_opportunities)
            opportunities_df.to_csv(opportunities_filepath, index=False)
            self.logger.info(f"Trading opportunities saved to {opportunities_filepath}")
        
        return opportunities


def main():
    """Main function to run indicator calculations"""
    parser = argparse.ArgumentParser(description='Calculate Technical Indicators for Indian Stocks')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to analyze (e.g., RELIANCE.NS TCS.NS)')
    parser.add_argument('--period', default='6mo', help='Data period (1mo, 3mo, 6mo, 1y, 2y)')
    parser.add_argument('--batch', action='store_true', help='Run batch calculation for all stocks')
    parser.add_argument('--summary', action='store_true', help='Generate summary report')
    parser.add_argument('--opportunities', action='store_true', help='Identify trading opportunities')
    
    args = parser.parse_args()
    
    print("Indian Stock Market Technical Indicators Calculator v1.0")
    print("-" * 60)
    
    calculator = IndicatorBatchCalculator()
    
    try:
        if args.batch or not args.symbols:
            # Run batch calculation
            print("Running batch calculation for top NSE stocks...")
            batch_results = calculator.calculate_batch_indicators(period=args.period)
            
            if args.summary:
                print("Generating indicator summary...")
                summary_df = calculator.generate_indicator_summary(batch_results)
                if summary_df is not None:
                    print(f"Summary generated for {len(summary_df)} stocks")
            
            if args.opportunities:
                print("Identifying trading opportunities...")
                opportunities = calculator.identify_trading_opportunities(batch_results)
                total_opportunities = sum(len(stocks) for stocks in opportunities.values())
                print(f"Found {total_opportunities} trading opportunities")
                
                # Print summary of opportunities
                for opp_type, stocks in opportunities.items():
                    if stocks:
                        print(f"\n{opp_type.replace('_', ' ').title()}: {len(stocks)} stocks")
                        for stock in stocks[:3]:  # Show first 3
                            print(f"  - {stock['symbol']}")
        
        else:
            # Calculate for specific symbols
            print(f"Calculating indicators for {len(args.symbols)} stocks...")
            batch_results = calculator.calculate_batch_indicators(
                symbols=args.symbols, 
                period=args.period
            )
            
            if args.summary:
                summary_df = calculator.generate_indicator_summary(batch_results)
                if summary_df is not None:
                    print(summary_df[['Symbol', 'Close', 'RSI', 'Above_SMA20', 'Above_SMA50']])
        
        print("\nIndicator calculation completed successfully!")
        print(f"Results saved to: data/processed/technical_indicators/")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error during calculation: {str(e)}")
        logging.error(f"Main execution error: {str(e)}")


class MarketScreener:
    """Market screening functionality for Indian stocks"""
    
    def __init__(self):
        self.logger = setup_logging('market_screener')
    
    def screen_momentum_stocks(self, batch_results, min_rsi=40, max_rsi=60):
        """Screen for momentum stocks with moderate RSI"""
        momentum_stocks = []
        
        for symbol, df in batch_results.items():
            if df.empty:
                continue
            
            latest = df.iloc[-1]
            
            # Momentum criteria
            if (min_rsi <= latest['RSI'] <= max_rsi and 
                latest['Above_SMA20'] and 
                latest['Above_SMA50'] and
                latest['MACD'] > latest['MACD_Signal'] and
                latest['Volume_Ratio'] > 1.2):
                
                momentum_stocks.append({
                    'symbol': symbol,
                    'close': latest['Close'],
                    'rsi': latest['RSI'],
                    'volume_ratio': latest['Volume_Ratio'],
                    'macd': latest['MACD'],
                    'score': self._calculate_momentum_score(latest)
                })
        
        # Sort by momentum score
        momentum_stocks.sort(key=lambda x: x['score'], reverse=True)
        return momentum_stocks
    
    def screen_value_stocks(self, batch_results):
        """Screen for potential value stocks using technical indicators"""
        value_stocks = []
        
        for symbol, df in batch_results.items():
            if df.empty:
                continue
            
            latest = df.iloc[-1]
            
            # Value criteria (oversold but showing signs of recovery)
            if (latest['RSI'] < 40 and 
                latest['RSI'] > 25 and  # Not extremely oversold
                latest['Close'] > latest['BB_Lower'] and  # Above BB lower band
                latest['Volume_Ratio'] > 1.0):  # Some volume interest
                
                value_stocks.append({
                    'symbol': symbol,
                    'close': latest['Close'],
                    'rsi': latest['RSI'],
                    'bb_position': (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']),
                    'volume_ratio': latest['Volume_Ratio']
                })
        
        return value_stocks
    
    def _calculate_momentum_score(self, latest_data):
        """Calculate momentum score based on multiple indicators"""
        score = 0
        
        # RSI contribution (optimal range gets higher score)
        if 50 <= latest_data['RSI'] <= 70:
            score += 30
        elif 40 <= latest_data['RSI'] < 50:
            score += 20
        
        # MACD contribution
        if latest_data['MACD'] > latest_data['MACD_Signal']:
            score += 25
            if latest_data['MACD_Histogram'] > 0:
                score += 10
        
        # Moving average contribution
        if latest_data['Above_SMA20']:
            score += 15
        if latest_data['Above_SMA50']:
            score += 10
        
        # Volume contribution
        if latest_data['Volume_Ratio'] > 1.5:
            score += 20
        elif latest_data['Volume_Ratio'] > 1.2:
            score += 10
        
        return score


def generate_daily_report(batch_results, output_dir='reports'):
    """Generate daily technical analysis report"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize screener
    screener = MarketScreener()
    
    # Get current date for report
    report_date = datetime.now().strftime('%Y%m%d')
    report_file = output_dir / f'daily_technical_report_{report_date}.txt'
    
    with open(report_file, 'w') as f:
        f.write("INDIAN STOCK MARKET - DAILY TECHNICAL ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Report Date: {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(f"Analysis Period: Last 6 months\n")
        f.write("=" * 60 + "\n\n")
        
        # Market Overview
        f.write("MARKET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        
        total_stocks = len(batch_results)
        stocks_above_sma20 = sum(1 for df in batch_results.values() 
                                if not df.empty and df.iloc[-1]['Above_SMA20'])
        stocks_above_sma50 = sum(1 for df in batch_results.values() 
                                if not df.empty and df.iloc[-1]['Above_SMA50'])
        
        f.write(f"Total Stocks Analyzed: {total_stocks}\n")
        f.write(f"Stocks Above SMA-20: {stocks_above_sma20} ({stocks_above_sma20/total_stocks*100:.1f}%)\n")
        f.write(f"Stocks Above SMA-50: {stocks_above_sma50} ({stocks_above_sma50/total_stocks*100:.1f}%)\n\n")
        
        # Momentum Stocks
        momentum_stocks = screener.screen_momentum_stocks(batch_results)
        f.write("TOP MOMENTUM STOCKS\n")
        f.write("-" * 20 + "\n")
        if momentum_stocks:
            f.write(f"{'Symbol':<15} {'Price':<10} {'RSI':<8} {'Vol Ratio':<10} {'Score':<8}\n")
            f.write("-" * 60 + "\n")
            for stock in momentum_stocks[:10]:  # Top 10
                f.write(f"{stock['symbol']:<15} {stock['close']:<10.2f} "
                       f"{stock['rsi']:<8.1f} {stock['volume_ratio']:<10.2f} "
                       f"{stock['score']:<8.0f}\n")
        else:
            f.write("No momentum stocks found matching criteria.\n")
        f.write("\n")
        
        # Value Opportunities
        value_stocks = screener.screen_value_stocks(batch_results)
        f.write("VALUE OPPORTUNITIES\n")
        f.write("-" * 20 + "\n")
        if value_stocks:
            f.write(f"{'Symbol':<15} {'Price':<10} {'RSI':<8} {'BB Position':<12}\n")
            f.write("-" * 50 + "\n")
            for stock in value_stocks[:10]:  # Top 10
                f.write(f"{stock['symbol']:<15} {stock['close']:<10.2f} "
                       f"{stock['rsi']:<8.1f} {stock['bb_position']:<12.2f}\n")
        else:
            f.write("No value opportunities found.\n")
        f.write("\n")
        
        # Technical Alerts
        f.write("TECHNICAL ALERTS\n")
        f.write("-" * 16 + "\n")
        
        # RSI Extremes
        overbought = [symbol for symbol, df in batch_results.items() 
                     if not df.empty and df.iloc[-1]['RSI_Overbought']]
        oversold = [symbol for symbol, df in batch_results.items() 
                   if not df.empty and df.iloc[-1]['RSI_Oversold']]
        
        f.write(f"Overbought Stocks (RSI > 70): {len(overbought)}\n")
        if overbought:
            f.write(f"  {', '.join(overbought[:5])}\n")
        
        f.write(f"Oversold Stocks (RSI < 30): {len(oversold)}\n")
        if oversold:
            f.write(f"  {', '.join(oversold[:5])}\n")
        f.write("\n")
        
        # MACD Signals
        bullish_macd = [symbol for symbol, df in batch_results.items() 
                       if not df.empty and df.iloc[-1]['MACD_Bullish']]
        
        f.write(f"Bullish MACD Crossovers: {len(bullish_macd)}\n")
        if bullish_macd:
            f.write(f"  {', '.join(bullish_macd[:5])}\n")
        f.write("\n")
        
        # Footer
        f.write("=" * 60 + "\n")
        f.write("Report generated by Indian Stock Market Research System v1.0\n")
        f.write("Disclaimer: This is for educational purposes only. Not investment advice.\n")
        f.write("=" * 60 + "\n")
    
    print(f"Daily report generated: {report_file}")
    return str(report_file)


# Additional utility functions for Version 1
def export_to_excel(batch_results, filename=None):
    """Export all technical indicators to Excel file"""
    if filename is None:
        filename = f"technical_indicators_{datetime.now().strftime('%Y%m%d')}.xlsx"
    
    output_dir = Path('data/processed/technical_indicators')
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for symbol, df in batch_results.items():
                if not df.empty:
                    latest = df.iloc[-1]
                    summary_data.append({
                        'Symbol': symbol,
                        'Close': latest['Close'],
                        'RSI': latest['RSI'],
                        'MACD': latest['MACD'],
                        'SMA_20': latest['SMA_20'],
                        'Above_SMA20': latest['Above_SMA20'],
                        'Volume_Ratio': latest['Volume_Ratio']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual stock sheets (first 10 stocks)
            for i, (symbol, df) in enumerate(list(batch_results.items())[:10]):
                if not df.empty:
                    sheet_name = symbol.replace('.NS', '')[:31]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=sheet_name)
        
        print(f"Excel export completed: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"Error exporting to Excel: {str(e)}")
        return None


if __name__ == "__main__":
    main()