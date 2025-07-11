#!/usr/bin/env python3
"""
Correlation Analysis Script for Indian Stock Market Research System v1.0
Author: Market Research System
Created: 2022
Description: Analyzes correlations between Indian stocks, sectors, and indices
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.settings import INDIAN_STOCK_UNIVERSE, SECTOR_MAPPING, INDIAN_INDICES
from utils.logging_utils import setup_logging
from utils.date_utils import get_trading_days
from utils.file_utils import ensure_directory_exists
from data.data_storage import save_correlation_data, load_correlation_data

class CorrelationAnalyzer:
    """Indian Stock Market Correlation Analysis"""
    
    def __init__(self, period_days=252):
        """
        Initialize correlation analyzer
        
        Args:
            period_days (int): Number of trading days to analyze (default: 252 = 1 year)
        """
        self.period_days = period_days
        self.logger = setup_logging('correlation_analysis')
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=int(period_days * 1.4))  # Buffer for weekends
        
        # Indian market specific data
        self.indian_stocks = INDIAN_STOCK_UNIVERSE
        self.sector_mapping = SECTOR_MAPPING
        self.indian_indices = INDIAN_INDICES
        
        # Results storage
        self.stock_data = {}
        self.correlation_matrix = None
        self.sector_correlations = {}
        self.index_correlations = {}
        
        self.logger.info(f"Initialized Correlation Analyzer for period: {self.start_date.date()} to {self.end_date.date()}")

    def fetch_stock_data(self):
        """Fetch historical data for Indian stocks and indices"""
        self.logger.info("Fetching Indian stock market data...")
        
        all_symbols = []
        all_symbols.extend([f"{stock}.NS" for stock in self.indian_stocks])  # NSE stocks
        all_symbols.extend([f"{index}" for index in self.indian_indices.values()])  # Indices
        
        failed_downloads = []
        successful_downloads = []
        
        for symbol in all_symbols:
            try:
                self.logger.info(f"Downloading data for {symbol}")
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                hist_data = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval='1d'
                )
                
                if len(hist_data) > 0:
                    # Calculate daily returns
                    hist_data['Returns'] = hist_data['Close'].pct_change()
                    hist_data = hist_data.dropna()
                    
                    if len(hist_data) >= 50:  # Minimum data points required
                        self.stock_data[symbol] = hist_data
                        successful_downloads.append(symbol)
                        self.logger.info(f"Successfully downloaded {len(hist_data)} days of data for {symbol}")
                    else:
                        failed_downloads.append(symbol)
                        self.logger.warning(f"Insufficient data for {symbol}: {len(hist_data)} days")
                else:
                    failed_downloads.append(symbol)
                    self.logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                failed_downloads.append(symbol)
                self.logger.error(f"Error downloading {symbol}: {str(e)}")
        
        self.logger.info(f"Data fetch complete. Success: {len(successful_downloads)}, Failed: {len(failed_downloads)}")
        
        if failed_downloads:
            self.logger.warning(f"Failed to download: {failed_downloads}")
        
        return len(successful_downloads) > 0

    def calculate_stock_correlations(self):
        """Calculate correlation matrix for Indian stocks"""
        self.logger.info("Calculating stock correlation matrix...")
        
        # Extract returns data for stocks only (exclude indices for now)
        stock_returns = {}
        
        for symbol, data in self.stock_data.items():
            if symbol.endswith('.NS'):  # NSE stocks
                clean_symbol = symbol.replace('.NS', '')
                stock_returns[clean_symbol] = data['Returns']
        
        if len(stock_returns) < 2:
            self.logger.error("Insufficient stock data for correlation analysis")
            return False
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(stock_returns)
        returns_df = returns_df.dropna()
        
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        
        self.logger.info(f"Calculated correlation matrix for {len(stock_returns)} stocks")
        return True

    def calculate_sector_correlations(self):
        """Calculate sector-wise correlations"""
        self.logger.info("Calculating sector correlations...")
        
        sector_returns = {}
        
        # Group stocks by sector and calculate average returns
        for sector, stocks in self.sector_mapping.items():
            sector_stock_returns = []
            
            for stock in stocks:
                symbol = f"{stock}.NS"
                if symbol in self.stock_data:
                    sector_stock_returns.append(self.stock_data[symbol]['Returns'])
            
            if sector_stock_returns:
                # Calculate equal-weighted sector returns
                sector_df = pd.DataFrame(sector_stock_returns).T
                sector_returns[sector] = sector_df.mean(axis=1)
        
        if len(sector_returns) >= 2:
            sector_df = pd.DataFrame(sector_returns)
            sector_df = sector_df.dropna()
            self.sector_correlations = sector_df.corr()
            self.logger.info(f"Calculated sector correlations for {len(sector_returns)} sectors")
            return True
        else:
            self.logger.warning("Insufficient sector data for correlation analysis")
            return False

    def calculate_index_correlations(self):
        """Calculate correlations with major Indian indices"""
        self.logger.info("Calculating index correlations...")
        
        index_returns = {}
        
        # Extract index returns
        for index_name, index_symbol in self.indian_indices.items():
            if index_symbol in self.stock_data:
                index_returns[index_name] = self.stock_data[index_symbol]['Returns']
        
        if len(index_returns) >= 2:
            index_df = pd.DataFrame(index_returns)
            index_df = index_df.dropna()
            self.index_correlations = index_df.corr()
            self.logger.info(f"Calculated index correlations for {len(index_returns)} indices")
            return True
        else:
            self.logger.warning("Insufficient index data for correlation analysis")
            return False

    def find_high_correlations(self, threshold=0.7):
        """Find pairs with high correlations"""
        self.logger.info(f"Finding correlations above {threshold}")
        
        high_corr_pairs = []
        
        if self.correlation_matrix is not None:
            # Get upper triangle of correlation matrix
            upper_triangle = np.triu(self.correlation_matrix, k=1)
            high_corr_indices = np.where(np.abs(upper_triangle) >= threshold)
            
            for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
                stock1 = self.correlation_matrix.index[i]
                stock2 = self.correlation_matrix.columns[j]
                correlation = self.correlation_matrix.iloc[i, j]
                
                high_corr_pairs.append({
                    'Stock1': stock1,
                    'Stock2': stock2,
                    'Correlation': correlation,
                    'Abs_Correlation': abs(correlation)
                })
        
        # Sort by absolute correlation
        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x['Abs_Correlation'], reverse=True)
        
        self.logger.info(f"Found {len(high_corr_pairs)} high correlation pairs")
        return high_corr_pairs

    def generate_correlation_heatmap(self, correlation_matrix, title, filename):
        """Generate correlation heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title(f'{title}\n{self.start_date.date()} to {self.end_date.date()}', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join('reports', 'correlation_analysis')
        ensure_directory_exists(output_dir)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved correlation heatmap: {filename}")

    def generate_correlation_summary(self):
        """Generate correlation analysis summary"""
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'period': {
                'start_date': self.start_date.date().isoformat(),
                'end_date': self.end_date.date().isoformat(),
                'trading_days': self.period_days
            },
            'data_summary': {
                'total_symbols_processed': len(self.stock_data),
                'successful_stocks': len([s for s in self.stock_data.keys() if s.endswith('.NS')]),
                'successful_indices': len([s for s in self.stock_data.keys() if not s.endswith('.NS')])
            }
        }
        
        # Add correlation statistics
        if self.correlation_matrix is not None:
            corr_values = self.correlation_matrix.values
            upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
            
            summary['stock_correlations'] = {
                'average_correlation': float(np.mean(upper_triangle)),
                'median_correlation': float(np.median(upper_triangle)),
                'max_correlation': float(np.max(upper_triangle)),
                'min_correlation': float(np.min(upper_triangle)),
                'std_correlation': float(np.std(upper_triangle))
            }
        
        if self.sector_correlations is not None and len(self.sector_correlations) > 0:
            sector_values = self.sector_correlations.values
            sector_upper = sector_values[np.triu_indices_from(sector_values, k=1)]
            
            summary['sector_correlations'] = {
                'average_correlation': float(np.mean(sector_upper)),
                'max_correlation': float(np.max(sector_upper)),
                'min_correlation': float(np.min(sector_upper))
            }
        
        return summary

    def save_results(self):
        """Save correlation analysis results"""
        self.logger.info("Saving correlation analysis results...")
        
        output_dir = os.path.join('data', 'processed', 'correlations')
        ensure_directory_exists(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save correlation matrices
        if self.correlation_matrix is not None:
            self.correlation_matrix.to_csv(
                os.path.join(output_dir, f'stock_correlations_{timestamp}.csv')
            )
        
        if self.sector_correlations is not None and len(self.sector_correlations) > 0:
            self.sector_correlations.to_csv(
                os.path.join(output_dir, f'sector_correlations_{timestamp}.csv')
            )
        
        if self.index_correlations is not None and len(self.index_correlations) > 0:
            self.index_correlations.to_csv(
                os.path.join(output_dir, f'index_correlations_{timestamp}.csv')
            )
        
        # Save summary
        summary = self.generate_correlation_summary()
        with open(os.path.join(output_dir, f'correlation_summary_{timestamp}.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("Results saved successfully")

    def run_analysis(self):
        """Run complete correlation analysis"""
        self.logger.info("Starting Indian Stock Market Correlation Analysis")
        
        try:
            # Step 1: Fetch data
            if not self.fetch_stock_data():
                self.logger.error("Failed to fetch sufficient data")
                return False
            
            # Step 2: Calculate stock correlations
            if self.calculate_stock_correlations():
                # Generate stock correlation heatmap
                if len(self.correlation_matrix) <= 20:  # Limit for readability
                    self.generate_correlation_heatmap(
                        self.correlation_matrix,
                        'Indian Stock Correlation Matrix',
                        'indian_stock_correlations.png'
                    )
            
            # Step 3: Calculate sector correlations
            if self.calculate_sector_correlations():
                self.generate_correlation_heatmap(
                    self.sector_correlations,
                    'Indian Sector Correlation Matrix',
                    'indian_sector_correlations.png'
                )
            
            # Step 4: Calculate index correlations
            if self.calculate_index_correlations():
                self.generate_correlation_heatmap(
                    self.index_correlations,
                    'Indian Market Indices Correlation Matrix',
                    'indian_index_correlations.png'
                )
            
            # Step 5: Find high correlations
            high_corr_pairs = self.find_high_correlations(threshold=0.7)
            
            # Step 6: Generate summary report
            summary = self.generate_correlation_summary()
            summary['high_correlation_pairs'] = high_corr_pairs[:10]  # Top 10
            
            # Step 7: Save results
            self.save_results()
            
            # Step 8: Print summary
            print("\n" + "="*60)
            print("INDIAN STOCK MARKET CORRELATION ANALYSIS SUMMARY")
            print("="*60)
            print(f"Analysis Period: {self.start_date.date()} to {self.end_date.date()}")
            print(f"Stocks Analyzed: {summary['data_summary']['successful_stocks']}")
            print(f"Indices Analyzed: {summary['data_summary']['successful_indices']}")
            
            if 'stock_correlations' in summary:
                print(f"\nStock Correlation Statistics:")
                print(f"  Average Correlation: {summary['stock_correlations']['average_correlation']:.3f}")
                print(f"  Median Correlation: {summary['stock_correlations']['median_correlation']:.3f}")
                print(f"  Range: [{summary['stock_correlations']['min_correlation']:.3f}, {summary['stock_correlations']['max_correlation']:.3f}]")
            
            if high_corr_pairs:
                print(f"\nTop 5 Highly Correlated Pairs (>0.7):")
                for i, pair in enumerate(high_corr_pairs[:5], 1):
                    print(f"  {i}. {pair['Stock1']} - {pair['Stock2']}: {pair['Correlation']:.3f}")
            
            print("\nAnalysis completed successfully!")
            print("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return False

def main():
    """Main function"""
    # Create output directories
    os.makedirs('reports/correlation_analysis', exist_ok=True)
    os.makedirs('data/processed/correlations', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run analysis
    analyzer = CorrelationAnalyzer(period_days=252)  # 1 year analysis
    success = analyzer.run_analysis()
    
    if success:
        print("\nCorrelation analysis completed successfully!")
        print("Check the following directories for results:")
        print("- reports/correlation_analysis/ (for charts)")
        print("- data/processed/correlations/ (for data files)")
        print("- logs/ (for detailed logs)")
    else:
        print("\nCorrelation analysis failed. Check logs for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())