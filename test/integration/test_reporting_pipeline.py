"""
Integration tests for the reporting pipeline.
Tests the complete flow from data to report generation.
"""

import unittest
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from reporting.pdf_generator import PDFGenerator
from reporting.visualization import ChartGenerator
from analysis.technical_indicators import TechnicalIndicators
from data.fetch_market_data import MarketDataFetcher


class TestReportingPipeline(unittest.TestCase):
    """Test complete reporting pipeline integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, 'data')
        self.reports_dir = os.path.join(self.test_dir, 'reports')
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Create sample stock data
        self.sample_data = self._create_sample_stock_data()
        
        # Initialize components
        self.pdf_generator = PDFGenerator(output_dir=self.reports_dir)
        self.chart_generator = ChartGenerator()
        self.technical_indicators = TechnicalIndicators()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def _create_sample_stock_data(self):
        """Create sample stock data for testing."""
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        # Create realistic stock data for Indian market
        stock_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
        
        stock_data = {}
        for symbol in stock_symbols:
            # Generate realistic price movements
            base_price = np.random.uniform(100, 3000)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLCV data
            stock_data[symbol] = pd.DataFrame({
                'Date': dates,
                'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
                'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
                'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(100000, 10000000) for _ in prices]
            })
            stock_data[symbol].set_index('Date', inplace=True)
        
        return stock_data
    
    def test_complete_daily_report_pipeline(self):
        """Test complete daily report generation pipeline."""
        # Step 1: Prepare data with technical indicators
        processed_data = {}
        for symbol, data in self.sample_data.items():
            # Calculate technical indicators
            data_with_indicators = self.technical_indicators.add_all_indicators(data.copy())
            processed_data[symbol] = data_with_indicators
        
        # Step 2: Generate charts
        chart_paths = []
        for symbol, data in processed_data.items():
            chart_path = os.path.join(self.test_dir, f'{symbol}_chart.png')
            self.chart_generator.create_price_chart_with_indicators(
                data, symbol, save_path=chart_path
            )
            chart_paths.append(chart_path)
            self.assertTrue(os.path.exists(chart_path))
        
        # Step 3: Create market summary
        market_summary = self._create_market_summary(processed_data)
        
        # Step 4: Generate PDF report
        report_path = self.pdf_generator.generate_daily_report(
            market_summary, processed_data, chart_paths
        )
        
        # Verify report exists and has content
        self.assertTrue(os.path.exists(report_path))
        self.assertGreater(os.path.getsize(report_path), 1000)  # Should be > 1KB
        
    def test_weekly_report_pipeline(self):
        """Test weekly report generation pipeline."""
        # Prepare weekly aggregated data
        weekly_data = {}
        for symbol, data in self.sample_data.items():
            # Resample to weekly data
            weekly = data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            weekly_data[symbol] = self.technical_indicators.add_all_indicators(weekly)
        
        # Generate weekly summary
        weekly_summary = self._create_weekly_summary(weekly_data)
        
        # Create sector performance analysis
        sector_performance = self._create_sector_analysis(weekly_data)
        
        # Generate weekly report
        report_path = self.pdf_generator.generate_weekly_report(
            weekly_summary, weekly_data, sector_performance
        )
        
        # Verify report
        self.assertTrue(os.path.exists(report_path))
        self.assertGreater(os.path.getsize(report_path), 2000)
    
    def test_correlation_analysis_pipeline(self):
        """Test correlation analysis and reporting pipeline."""
        # Calculate correlation matrix
        close_prices = pd.DataFrame()
        for symbol, data in self.sample_data.items():
            close_prices[symbol] = data['Close']
        
        correlation_matrix = close_prices.corr()
        
        # Generate correlation heatmap
        heatmap_path = os.path.join(self.test_dir, 'correlation_heatmap.png')
        self.chart_generator.create_correlation_heatmap(
            correlation_matrix, save_path=heatmap_path
        )
        
        # Generate correlation report
        report_path = self.pdf_generator.generate_correlation_report(
            correlation_matrix, heatmap_path
        )
        
        # Verify outputs
        self.assertTrue(os.path.exists(heatmap_path))
        self.assertTrue(os.path.exists(report_path))
    
    def test_performance_metrics_pipeline(self):
        """Test performance metrics calculation and reporting."""
        performance_metrics = {}
        
        for symbol, data in self.sample_data.items():
            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            
            # Calculate performance metrics
            metrics = {
                'total_return': (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100,
                'volatility': data['Returns'].std() * np.sqrt(252) * 100,
                'sharpe_ratio': (data['Returns'].mean() * 252) / (data['Returns'].std() * np.sqrt(252)),
                'max_drawdown': self._calculate_max_drawdown(data['Close']),
                'win_rate': (data['Returns'] > 0).sum() / len(data['Returns']) * 100
            }
            performance_metrics[symbol] = metrics
        
        # Generate performance report
        report_path = self.pdf_generator.generate_performance_report(performance_metrics)
        
        # Verify report
        self.assertTrue(os.path.exists(report_path))
        
    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Test with corrupt data
        corrupt_data = self.sample_data.copy()
        for symbol in corrupt_data:
            # Introduce NaN values
            corrupt_data[symbol].loc[corrupt_data[symbol].index[50:60], 'Close'] = np.nan
        
        # Pipeline should handle errors gracefully
        try:
            processed_data = {}
            for symbol, data in corrupt_data.items():
                data_with_indicators = self.technical_indicators.add_all_indicators(data.copy())
                processed_data[symbol] = data_with_indicators
            
            # Should not raise exception
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Pipeline failed to handle corrupt data: {e}")
    
    def _create_market_summary(self, data):
        """Create market summary from processed data."""
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_stocks': len(data),
            'gainers': 0,
            'losers': 0,
            'unchanged': 0,
            'avg_volume': 0,
            'market_trend': 'Neutral'
        }
        
        total_volume = 0
        for symbol, df in data.items():
            if len(df) >= 2:
                change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                if change > 0:
                    summary['gainers'] += 1
                elif change < 0:
                    summary['losers'] += 1
                else:
                    summary['unchanged'] += 1
                
                total_volume += df['Volume'].iloc[-1]
        
        summary['avg_volume'] = total_volume / len(data)
        
        # Determine market trend
        if summary['gainers'] > summary['losers']:
            summary['market_trend'] = 'Bullish'
        elif summary['losers'] > summary['gainers']:
            summary['market_trend'] = 'Bearish'
        
        return summary
    
    def _create_weekly_summary(self, weekly_data):
        """Create weekly summary from data."""
        return {
            'week_start': datetime.now().strftime('%Y-%m-%d'),
            'week_end': (datetime.now() + timedelta(days=6)).strftime('%Y-%m-%d'),
            'total_stocks': len(weekly_data),
            'best_performer': 'TCS.NS',  # Simplified
            'worst_performer': 'RELIANCE.NS',  # Simplified
            'avg_weekly_return': 2.5
        }
    
    def _create_sector_analysis(self, data):
        """Create sector performance analysis."""
        # Simplified sector mapping
        sectors = {
            'Technology': ['TCS.NS', 'INFY.NS'],
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS'],
            'Energy': ['RELIANCE.NS']
        }
        
        sector_performance = {}
        for sector, stocks in sectors.items():
            sector_returns = []
            for stock in stocks:
                if stock in data and len(data[stock]) >= 2:
                    week_return = (data[stock]['Close'].iloc[-1] / data[stock]['Close'].iloc[0] - 1) * 100
                    sector_returns.append(week_return)
            
            if sector_returns:
                sector_performance[sector] = {
                    'avg_return': np.mean(sector_returns),
                    'best_stock': stocks[0],  # Simplified
                    'worst_stock': stocks[-1]  # Simplified
                }
        
        return sector_performance
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min() * 100


if __name__ == '__main__':
    unittest.main()