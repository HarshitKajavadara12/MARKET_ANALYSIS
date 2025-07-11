#!/usr/bin/env python3
"""
Monthly Market Research Report Generator
Version 1.0 - 2022
Focuses on Indian Stock Market (NSE/BSE)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import logging
import yaml

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.fetch_market_data import IndianMarketDataFetcher
from analysis.technical_indicators import TechnicalAnalyzer
from analysis.market_analyzer import MarketAnalyzer
from utils.date_utils import get_month_range, format_date
from utils.file_utils import ensure_directory_exists

class MonthlyReportGenerator:
    """Generates comprehensive monthly market research reports for Indian markets"""
    
    def __init__(self, config_path="config/reporting/report_templates.yaml"):
        """Initialize the monthly report generator"""
        self.setup_logging()
        self.load_config(config_path)
        self.data_fetcher = IndianMarketDataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        
        # Indian market symbols
        self.nifty_50_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
            'BAJFINANCE.NS', 'LICI.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
            'ADANIENT.NS', 'ONGC.NS', 'NTPC.NS', 'AXISBANK.NS', 'NESTLEIND.NS',
            'LTIM.NS', 'POWERGRID.NS', 'BAJAJFINSV.NS', 'ULTRACEMCO.NS', 'HCLTECH.NS',
            'COALINDIA.NS', 'WIPRO.NS', 'SUNPHARMA.NS', 'GRASIM.NS', 'JSWSTEEL.NS',
            'TATAMOTORS.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS', 'CIPLA.NS', 'EICHERMOT.NS',
            'TECHM.NS', 'BPCL.NS', 'TATACONSUM.NS', 'DIVISLAB.NS', 'TATASTEEL.NS',
            'M&M.NS', 'HINDALCO.NS', 'ADANIPORTS.NS', 'BRITANNIA.NS', 'HEROMOTOCO.NS',
            'BAJAJ-AUTO.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'INDUSINDBK.NS', 'UPL.NS'
        ]
        
        self.indices = {
            '^NSEI': 'NIFTY 50',
            '^NSEBANK': 'NIFTY Bank',
            '^NSEIT': 'NIFTY IT',
            '^NSMID': 'NIFTY Midcap 100'
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/reporting.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            self.config = self.get_default_config()
            
    def get_default_config(self):
        """Return default configuration"""
        return {
            'monthly_report': {
                'include_sectors': True,
                'include_technical_analysis': True,
                'include_top_performers': True,
                'top_performers_count': 10,
                'chart_style': 'seaborn',
                'colors': {
                    'primary': '#1f77b4',
                    'secondary': '#ff7f0e',
                    'success': '#2ca02c',
                    'danger': '#d62728'
                }
            }
        }
        
    def generate_monthly_report(self, year, month, output_dir="reports/monthly"):
        """Generate comprehensive monthly report"""
        try:
            self.logger.info(f"Generating monthly report for {calendar.month_name[month]} {year}")
            
            # Create output directory
            month_dir = os.path.join(output_dir, str(year), f"{month:02d}")
            ensure_directory_exists(month_dir)
            
            # Get date range for the month
            start_date, end_date = get_month_range(year, month)
            
            # Collect data
            market_data = self.collect_monthly_data(start_date, end_date)
            
            # Perform analysis
            analysis_results = self.perform_monthly_analysis(market_data, start_date, end_date)
            
            # Generate visualizations
            chart_paths = self.generate_monthly_charts(analysis_results, month_dir)
            
            # Create PDF report
            report_filename = f"monthly_report_{year}_{month:02d}.pdf"
            report_path = os.path.join(month_dir, report_filename)
            self.create_pdf_report(analysis_results, chart_paths, report_path, year, month)
            
            self.logger.info(f"Monthly report generated: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating monthly report: {str(e)}")
            raise
            
    def collect_monthly_data(self, start_date, end_date):
        """Collect all required data for monthly analysis"""
        self.logger.info("Collecting monthly market data...")
        
        market_data = {}
        
        # Fetch index data
        for symbol, name in self.indices.items():
            try:
                data = self.data_fetcher.fetch_historical_data(symbol, start_date, end_date)
                market_data[symbol] = {
                    'name': name,
                    'data': data,
                    'symbol': symbol
                }
            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
                
        # Fetch top stocks data
        stock_data = {}
        for symbol in self.nifty_50_symbols[:20]:  # Top 20 for monthly analysis
            try:
                data = self.data_fetcher.fetch_historical_data(symbol, start_date, end_date)
                stock_data[symbol] = data
            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
                
        market_data['stocks'] = stock_data
        
        return market_data
        
    def perform_monthly_analysis(self, market_data, start_date, end_date):
        """Perform comprehensive monthly analysis"""
        self.logger.info("Performing monthly analysis...")
        
        analysis_results = {
            'period': {
                'start_date': start_date,
                'end_date': end_date,
                'month_name': calendar.month_name[start_date.month],
                'year': start_date.year
            },
            'indices_performance': {},
            'stock_performance': {},
            'sector_analysis': {},
            'technical_summary': {},
            'market_summary': {}
        }
        
        # Analyze indices performance
        for symbol, info in market_data.items():
            if symbol == 'stocks':
                continue
                
            data = info['data']
            if len(data) > 0:
                performance = self.calculate_monthly_performance(data)
                technical_indicators = self.technical_analyzer.calculate_monthly_indicators(data)
                
                analysis_results['indices_performance'][symbol] = {
                    'name': info['name'],
                    'performance': performance,
                    'technical': technical_indicators
                }
                
        # Analyze stock performance
        top_performers = []
        worst_performers = []
        
        for symbol, data in market_data['stocks'].items():
            if len(data) > 0:
                performance = self.calculate_monthly_performance(data)
                performance['symbol'] = symbol
                
                if performance['return_pct'] > 0:
                    top_performers.append(performance)
                else:
                    worst_performers.append(performance)
                    
        # Sort performers
        top_performers = sorted(top_performers, key=lambda x: x['return_pct'], reverse=True)[:10]
        worst_performers = sorted(worst_performers, key=lambda x: x['return_pct'])[:10]
        
        analysis_results['stock_performance'] = {
            'top_performers': top_performers,
            'worst_performers': worst_performers
        }
        
        # Market summary
        nifty_data = market_data.get('^NSEI', {}).get('data', pd.DataFrame())
        if len(nifty_data) > 0:
            analysis_results['market_summary'] = self.generate_market_summary(nifty_data)
            
        return analysis_results
        
    def calculate_monthly_performance(self, data):
        """Calculate monthly performance metrics"""
        if len(data) == 0:
            return {}
            
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        high_price = data['High'].max()
        low_price = data['Low'].min()
        
        return {
            'start_price': start_price,
            'end_price': end_price,
            'high_price': high_price,
            'low_price': low_price,
            'return_pct': ((end_price - start_price) / start_price) * 100,
            'volatility': data['Close'].pct_change().std() * np.sqrt(252) * 100,
            'avg_volume': data['Volume'].mean() if 'Volume' in data.columns else 0
        }
        
    def generate_market_summary(self, nifty_data):
        """Generate overall market summary"""
        returns = nifty_data['Close'].pct_change().dropna()
        
        return {
            'total_trading_days': len(nifty_data),
            'positive_days': len(returns[returns > 0]),
            'negative_days': len(returns[returns < 0]),
            'avg_daily_return': returns.mean() * 100,
            'max_single_day_gain': returns.max() * 100,
            'max_single_day_loss': returns.min() * 100,
            'monthly_volatility': returns.std() * np.sqrt(252) * 100
        }
        
    def generate_monthly_charts(self, analysis_results, output_dir):
        """Generate all charts for monthly report"""
        self.logger.info("Generating monthly charts...")
        
        chart_paths = {}
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn' in plt.style.available else 'default')
        
        # 1. Indices Performance Chart
        chart_paths['indices_performance'] = self.create_indices_performance_chart(
            analysis_results['indices_performance'], output_dir
        )
        
        # 2. Top Performers Chart
        chart_paths['top_performers'] = self.create_performers_chart(
            analysis_results['stock_performance']['top_performers'], 
            "Top Performers", output_dir
        )
        
        # 3. Worst Performers Chart
        chart_paths['worst_performers'] = self.create_performers_chart(
            analysis_results['stock_performance']['worst_performers'], 
            "Worst Performers", output_dir
        )
        
        return chart_paths
        
    def create_indices_performance_chart(self, indices_data, output_dir):
        """Create indices performance comparison chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        indices = []
        returns = []
        
        for symbol, data in indices_data.items():
            indices.append(data['name'])
            returns.append(data['performance']['return_pct'])
            
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        bars = ax.bar(indices, returns, color=colors, alpha=0.7)
        ax.set_title('Monthly Performance - Major Indices', fontsize=16, fontweight='bold')
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                   f'{value:.2f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = os.path.join(output_dir, 'indices_performance.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    def create_performers_chart(self, performers_data, title, output_dir):
        """Create top/worst performers chart"""
        if not performers_data:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        symbols = [p['symbol'].replace('.NS', '') for p in performers_data]
        returns = [p['return_pct'] for p in performers_data]
        
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        bars = ax.barh(symbols[::-1], returns[::-1], color=colors[::-1], alpha=0.7)
        ax.set_title(f'{title} - Monthly Returns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Return (%)', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, returns[::-1]):
            width = bar.get_width()
            ax.text(width + (0.2 if width > 0 else -0.2), bar.get_y() + bar.get_height()/2.,
                   f'{value:.2f}%', ha='left' if width > 0 else 'right', va='center')
        
        plt.tight_layout()
        
        chart_path = os.path.join(output_dir, f'{title.lower().replace(" ", "_")}.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    def create_pdf_report(self, analysis_results, chart_paths, output_path, year, month):
        """Create comprehensive PDF report"""
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        
        month_name = calendar.month_name[month]
        title = Paragraph(f"Monthly Market Research Report<br/>{month_name} {year}", title_style)
        story.append(title)
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        # Market summary
        market_summary = analysis_results.get('market_summary', {})
        if market_summary:
            summary_text = f"""
            During {month_name} {year}, the Indian equity markets showed the following characteristics:
            <br/>• Total trading days: {market_summary.get('total_trading_days', 'N/A')}
            <br/>• Positive days: {market_summary.get('positive_days', 'N/A')}
            <br/>• Negative days: {market_summary.get('negative_days', 'N/A')}
            <br/>• Average daily return: {market_summary.get('avg_daily_return', 0):.2f}%
            <br/>• Maximum single day gain: {market_summary.get('max_single_day_gain', 0):.2f}%
            <br/>• Maximum single day loss: {market_summary.get('max_single_day_loss', 0):.2f}%
            """
            story.append(Paragraph(summary_text, styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Indices Performance Section
        story.append(Paragraph("Index Performance", styles['Heading2']))
        
        if chart_paths.get('indices_performance'):
            story.append(Image(chart_paths['indices_performance'], width=7*inch, height=4*inch))
        
        # Create indices performance table
        indices_data = []
        indices_data.append(['Index', 'Start Price', 'End Price', 'Return (%)', 'Volatility (%)'])
        
        for symbol, data in analysis_results['indices_performance'].items():
            perf = data['performance']
            indices_data.append([
                data['name'],
                f"₹{perf.get('start_price', 0):.2f}",
                f"₹{perf.get('end_price', 0):.2f}",
                f"{perf.get('return_pct', 0):.2f}%",
                f"{perf.get('volatility', 0):.2f}%"
            ])
        
        indices_table = Table(indices_data)
        indices_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Spacer(1, 0.2*inch))
        story.append(indices_table)
        story.append(PageBreak())
        
        # Top Performers Section
        story.append(Paragraph("Top Performers", styles['Heading2']))
        
        if chart_paths.get('top_performers'):
            story.append(Image(chart_paths['top_performers'], width=7*inch, height=4*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Worst Performers Section
        story.append(Paragraph("Worst Performers", styles['Heading2']))
        
        if chart_paths.get('worst_performers'):
            story.append(Image(chart_paths['worst_performers'], width=7*inch, height=4*inch))
        
        # Build PDF
        doc.build(story)
        self.logger.info(f"PDF report created: {output_path}")


def main():
    """Main function to generate monthly report"""
    if len(sys.argv) != 3:
        print("Usage: python generate_monthly_report.py <year> <month>")
        print("Example: python generate_monthly_report.py 2022 12")
        sys.exit(1)
        
    try:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        
        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12")
            
        generator = MonthlyReportGenerator()
        report_path = generator.generate_monthly_report(year, month)
        print(f"Monthly report generated successfully: {report_path}")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating monthly report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()