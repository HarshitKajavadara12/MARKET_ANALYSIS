#!/usr/bin/env python3
"""
Daily Report Generator for Indian Stock Market Research System v1.0
Author: Market Research Team
Date: 2022
Description: Generates comprehensive daily market analysis reports for Indian stock markets
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import matplotlib
matplotlib.use('Agg')

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.fetch_market_data import IndianStockDataFetcher
from analysis.technical_indicators import TechnicalAnalyzer
from analysis.market_analyzer import MarketAnalyzer
from utils.logging_utils import setup_logger
from config.settings import INDIAN_STOCK_CONFIG

class DailyReportGenerator:
    """Generates daily market analysis reports for Indian stock markets"""
    
    def __init__(self):
        self.logger = setup_logger('daily_report_generator')
        self.data_fetcher = IndianStockDataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        self.report_date = datetime.now().strftime('%Y-%m-%d')
        self.report_dir = os.path.join('reports', 'daily', str(datetime.now().year), 
                                     f"{datetime.now().month:02d}")
        
        # Create report directory if it doesn't exist
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Indian market indices and top stocks
        self.indices = ['NIFTY', 'SENSEX', 'BANKNIFTY', 'NIFTYIT', 'NIFTYPHARMA']
        self.top_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
            'SBIN.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS', 'WIPRO.NS',
            'ULTRACEMCO.NS', 'TECHM.NS', 'TITAN.NS', 'POWERGRID.NS', 'NESTLEIND.NS'
        ]
        
    def fetch_daily_data(self):
        """Fetch current day's market data"""
        try:
            self.logger.info("Fetching daily market data...")
            
            # Fetch indices data
            self.indices_data = {}
            for index in self.indices:
                try:
                    data = self.data_fetcher.get_historical_data(f"^{index}", period='5d')
                    if not data.empty:
                        self.indices_data[index] = data
                except Exception as e:
                    self.logger.warning(f"Failed to fetch data for {index}: {e}")
            
            # Fetch top stocks data
            self.stocks_data = {}
            for stock in self.top_stocks:
                try:
                    data = self.data_fetcher.get_historical_data(stock, period='30d')
                    if not data.empty:
                        self.stocks_data[stock.replace('.NS', '')] = data
                except Exception as e:
                    self.logger.warning(f"Failed to fetch data for {stock}: {e}")
                    
            self.logger.info("Daily data fetching completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching daily data: {e}")
            return False
    
    def analyze_market_sentiment(self):
        """Analyze overall market sentiment"""
        try:
            sentiment_data = {
                'indices_performance': {},
                'top_gainers': [],
                'top_losers': [],
                'market_breadth': {},
                'volatility_analysis': {}
            }
            
            # Analyze indices performance
            for index, data in self.indices_data.items():
                if len(data) >= 2:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    sentiment_data['indices_performance'][index] = {
                        'current': current,
                        'change_pct': change,
                        'volume': data['Volume'].iloc[-1] if 'Volume' in data else 0
                    }
            
            # Find top gainers and losers
            stock_changes = []
            for stock, data in self.stocks_data.items():
                if len(data) >= 2:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    stock_changes.append((stock, change, current))
            
            # Sort by change percentage
            stock_changes.sort(key=lambda x: x[1], reverse=True)
            sentiment_data['top_gainers'] = stock_changes[:5]
            sentiment_data['top_losers'] = stock_changes[-5:]
            
            # Calculate market breadth
            advances = sum(1 for _, change, _ in stock_changes if change > 0)
            declines = sum(1 for _, change, _ in stock_changes if change < 0)
            unchanged = sum(1 for _, change, _ in stock_changes if change == 0)
            
            sentiment_data['market_breadth'] = {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'advance_decline_ratio': advances / declines if declines > 0 else 0
            }
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {}
    
    def generate_technical_analysis(self):
        """Generate technical analysis for key stocks and indices"""
        technical_summary = {}
        
        try:
            # Analyze major indices
            for index, data in self.indices_data.items():
                if len(data) >= 50:  # Need enough data for technical indicators
                    analysis = self.technical_analyzer.calculate_all_indicators(data)
                    signals = self.technical_analyzer.generate_signals(analysis)
                    
                    technical_summary[index] = {
                        'trend': signals.get('trend', 'Neutral'),
                        'rsi': analysis.get('RSI', [0])[-1] if 'RSI' in analysis else 50,
                        'macd_signal': signals.get('macd_signal', 'Hold'),
                        'support_level': min(data['Low'].tail(20)),
                        'resistance_level': max(data['High'].tail(20))
                    }
            
            # Analyze top stocks
            stock_count = 0
            for stock, data in self.stocks_data.items():
                if len(data) >= 50 and stock_count < 10:  # Limit to top 10 for daily report
                    analysis = self.technical_analyzer.calculate_all_indicators(data)
                    signals = self.technical_analyzer.generate_signals(analysis)
                    
                    technical_summary[stock] = {
                        'trend': signals.get('trend', 'Neutral'),
                        'rsi': analysis.get('RSI', [0])[-1] if 'RSI' in analysis else 50,
                        'macd_signal': signals.get('macd_signal', 'Hold'),
                        'price': data['Close'].iloc[-1],
                        'volume_trend': 'High' if data['Volume'].iloc[-1] > data['Volume'].tail(10).mean() else 'Normal'
                    }
                    stock_count += 1
            
            return technical_summary
            
        except Exception as e:
            self.logger.error(f"Error generating technical analysis: {e}")
            return {}
    
    def create_charts(self):
        """Create charts for the daily report"""
        charts = {}
        
        try:
            # Market performance chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Indices performance
            if self.indices_data:
                indices_names = []
                indices_changes = []
                
                for index, data in self.indices_data.items():
                    if len(data) >= 2:
                        current = data['Close'].iloc[-1]
                        previous = data['Close'].iloc[-2]
                        change = ((current - previous) / previous) * 100
                        indices_names.append(index)
                        indices_changes.append(change)
                
                colors_list = ['green' if x >= 0 else 'red' for x in indices_changes]
                ax1.bar(indices_names, indices_changes, color=colors_list)
                ax1.set_title('Indices Performance (%)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Change (%)')
                ax1.tick_params(axis='x', rotation=45)
            
            # Top gainers/losers
            sentiment = self.analyze_market_sentiment()
            if sentiment.get('top_gainers') and sentiment.get('top_losers'):
                gainers = sentiment['top_gainers'][:5]
                losers = sentiment['top_losers'][:5]
                
                gainer_names = [g[0] for g in gainers]
                gainer_changes = [g[1] for g in gainers]
                
                ax2.barh(gainer_names, gainer_changes, color='green', alpha=0.7)
                ax2.set_title('Top Gainers (%)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Change (%)')
            
            # Market breadth
            if sentiment.get('market_breadth'):
                breadth = sentiment['market_breadth']
                labels = ['Advances', 'Declines', 'Unchanged']
                sizes = [breadth['advances'], breadth['declines'], breadth['unchanged']]
                colors_pie = ['green', 'red', 'gray']
                
                ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
                ax3.set_title('Market Breadth', fontsize=14, fontweight='bold')
            
            # NIFTY trend (if available)
            if 'NIFTY' in self.indices_data:
                nifty_data = self.indices_data['NIFTY'].tail(30)
                ax4.plot(nifty_data.index, nifty_data['Close'], linewidth=2, color='blue')
                ax4.set_title('NIFTY 30-Day Trend', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Price')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.report_dir, f'daily_charts_{self.report_date}.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            charts['daily_overview'] = chart_path
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating charts: {e}")
            return {}
    
    def generate_pdf_report(self, sentiment_data, technical_summary, charts):
        """Generate PDF report with all analysis"""
        try:
            report_filename = f"daily_market_report_{self.report_date}.pdf"
            report_path = os.path.join(self.report_dir, report_filename)
            
            doc = SimpleDocTemplate(report_path, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            # Title
            title = Paragraph(f"Indian Stock Market Daily Report<br/>{self.report_date}", title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            
            if sentiment_data.get('indices_performance'):
                nifty_perf = sentiment_data['indices_performance'].get('NIFTY', {})
                sensex_perf = sentiment_data['indices_performance'].get('SENSEX', {})
                
                summary_text = f"""
                <para>
                • NIFTY: {nifty_perf.get('change_pct', 0):.2f}% | Current: {nifty_perf.get('current', 0):.2f}<br/>
                • SENSEX: {sensex_perf.get('change_pct', 0):.2f}% | Current: {sensex_perf.get('current', 0):.2f}<br/>
                • Market Breadth: {sentiment_data.get('market_breadth', {}).get('advances', 0)} advances, 
                {sentiment_data.get('market_breadth', {}).get('declines', 0)} declines<br/>
                • Top Gainer: {sentiment_data.get('top_gainers', [('N/A', 0)])[0][0]} 
                ({sentiment_data.get('top_gainers', [('N/A', 0)])[0][1]:.2f}%)<br/>
                • Top Loser: {sentiment_data.get('top_losers', [('N/A', 0)])[0][0]} 
                ({sentiment_data.get('top_losers', [('N/A', 0)])[0][1]:.2f}%)
                </para>
                """
                story.append(Paragraph(summary_text, styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Add charts
            if charts.get('daily_overview') and os.path.exists(charts['daily_overview']):
                story.append(Paragraph("Market Overview Charts", heading_style))
                img = Image(charts['daily_overview'], width=7*inch, height=5.6*inch)
                story.append(img)
                story.append(Spacer(1, 20))
            
            # Technical Analysis Summary
            story.append(Paragraph("Technical Analysis Summary", heading_style))
            
            if technical_summary:
                # Create table for technical analysis
                tech_data = [['Symbol', 'Trend', 'RSI', 'MACD Signal', 'Current Price']]
                
                for symbol, analysis in list(technical_summary.items())[:10]:
                    tech_data.append([
                        symbol,
                        analysis.get('trend', 'N/A'),
                        f"{analysis.get('rsi', 0):.1f}",
                        analysis.get('macd_signal', 'N/A'),
                        f"{analysis.get('price', 0):.2f}" if 'price' in analysis else 'N/A'
                    ])
                
                tech_table = Table(tech_data)
                tech_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(tech_table)
                story.append(Spacer(1, 20))
            
            # Top Gainers and Losers
            story.append(Paragraph("Top Performers", heading_style))
            
            if sentiment_data.get('top_gainers') and sentiment_data.get('top_losers'):
                performers_data = [['Top Gainers', 'Change (%)', 'Top Losers', 'Change (%)']]
                
                max_len = max(len(sentiment_data['top_gainers']), len(sentiment_data['top_losers']))
                for i in range(min(5, max_len)):
                    gainer = sentiment_data['top_gainers'][i] if i < len(sentiment_data['top_gainers']) else ('', 0)
                    loser = sentiment_data['top_losers'][i] if i < len(sentiment_data['top_losers']) else ('', 0)
                    
                    performers_data.append([
                        gainer[0],
                        f"{gainer[1]:.2f}%",
                        loser[0],
                        f"{loser[1]:.2f}%"
                    ])
                
                performers_table = Table(performers_data)
                performers_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(performers_table)
                story.append(Spacer(1, 20))
            
            # Footer
            footer_text = f"""
            <para>
            Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            Market Research System v1.0 | Indian Stock Market Analysis<br/>
            Data sources: Yahoo Finance, NSE, BSE | For research purposes only
            </para>
            """
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            self.logger.info(f"Daily report generated: {report_path}")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            return None
    
    def run_daily_report(self):
        """Main function to run daily report generation"""
        try:
            self.logger.info(f"Starting daily report generation for {self.report_date}")
            
            # Fetch data
            if not self.fetch_daily_data():
                self.logger.error("Failed to fetch daily data")
                return False
            
            # Analyze market sentiment
            sentiment_data = self.analyze_market_sentiment()
            
            # Generate technical analysis
            technical_summary = self.generate_technical_analysis()
            
            # Create charts
            charts = self.create_charts()
            
            # Generate PDF report
            report_path = self.generate_pdf_report(sentiment_data, technical_summary, charts)
            
            if report_path:
                self.logger.info(f"Daily report completed successfully: {report_path}")
                
                # Log report summary
                self.log_report_summary(sentiment_data, technical_summary)
                
                return report_path
            else:
                self.logger.error("Failed to generate PDF report")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in daily report generation: {e}")
            return False
    
    def log_report_summary(self, sentiment_data, technical_summary):
        """Log key metrics from the report"""
        try:
            summary_log = {
                'date': self.report_date,
                'indices_analyzed': len(self.indices_data),
                'stocks_analyzed': len(self.stocks_data),
                'market_breadth': sentiment_data.get('market_breadth', {}),
                'technical_signals': len(technical_summary)
            }
            
            # Log to dedicated daily summary file
            summary_file = os.path.join('logs', 'application', 'daily_report_summary.log')
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
            
            with open(summary_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()}: {summary_log}\n")
                
        except Exception as e:
            self.logger.error(f"Error logging report summary: {e}")


def main():
    """Main function"""
    try:
        generator = DailyReportGenerator()
        result = generator.run_daily_report()
        
        if result:
            print(f"Daily report generated successfully: {result}")
            return 0
        else:
            print("Failed to generate daily report")
            return 1
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        return 1


if __name__ == "__main__":
    exit(main())