#!/usr/bin/env python3
"""
Weekly Report Generator for Indian Stock Market Research System v1.0
Author: Market Research Team
Date: 2022
Description: Generates comprehensive weekly market analysis reports for Indian stock markets
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.fetch_market_data import IndianStockDataFetcher
from analysis.technical_indicators import TechnicalAnalyzer
from analysis.market_analyzer import MarketAnalyzer
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.trend_analyzer import TrendAnalyzer
from utils.logging_utils import setup_logger
from config.settings import INDIAN_STOCK_CONFIG

class WeeklyReportGenerator:
    """Generates weekly market analysis reports for Indian stock markets"""
    
    def __init__(self):
        self.logger = setup_logger('weekly_report_generator')
        self.data_fetcher = IndianStockDataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        
        # Calculate week dates
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        self.week_start = week_start.strftime('%Y-%m-%d')
        self.week_end = today.strftime('%Y-%m-%d')
        self.report_date = f"Week_{today.isocalendar()[1]}_2022"
        
        # Create report directory
        self.report_dir = os.path.join('reports', 'weekly', str(today.year), 
                                     f"week_{today.isocalendar()[1]:02d}")
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Extended Indian market coverage
        self.indices = ['NIFTY', 'SENSEX', 'BANKNIFTY', 'NIFTYIT', 'NIFTYPHARMA', 
                       'NIFTYAUTO', 'NIFTYFMCG', 'NIFTYMETAL', 'NIFTYREALTY', 'NIFTYPSE']
        
        self.nifty_50_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
            'SBIN.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS', 'WIPRO.NS',
            'ULTRACEMCO.NS', 'TECHM.NS', 'TITAN.NS', 'POWERGRID.NS', 'NESTLEIND.NS',
            'HCLTECH.NS', 'NTPC.NS', 'COALINDIA.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS',
            'GRASIM.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'BAJAJFINSV.NS', 'BRITANNIA.NS',
            'CIPLA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GODREJCP.NS', 'HEROMOTOCO.NS',
            'HINDALCO.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'IOC.NS', 'ONGC.NS',
            'SHREECEM.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'UPL.NS'
        ]
        
        self.sectors = {
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
            'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
            'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'APOLLOHOSP.NS'],
            'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS'],
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'GODREJCP.NS']
        }
    
    def fetch_weekly_data(self):
        """Fetch week's market data"""
        try:
            self.logger.info("Fetching weekly market data...")
            
            # Fetch indices data (3 months for better analysis)
            self.indices_data = {}
            for index in self.indices:
                try:
                    data = self.data_fetcher.get_historical_data(f"^{index}", period='3mo')
                    if not data.empty:
                        self.indices_data[index] = data
                        self.logger.info(f"Fetched {len(data)} days of data for {index}")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch data for {index}: {e}")
            
            # Fetch NIFTY 50 stocks data
            self.stocks_data = {}
            for stock in self.nifty_50_stocks:
                try:
                    data = self.data_fetcher.get_historical_data(stock, period='3mo')
                    if not data.empty:
                        self.stocks_data[stock.replace('.NS', '')] = data
                except Exception as e:
                    self.logger.warning(f"Failed to fetch data for {stock}: {e}")
            
            # Fetch sector-wise data
            self.sector_data = {}
            for sector, stocks in self.sectors.items():
                sector_stocks = {}
                for stock in stocks:
                    try:
                        data = self.data_fetcher.get_historical_data(stock, period='3mo')
                        if not data.empty:
                            sector_stocks[stock.replace('.NS', '')] = data
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch data for {stock}: {e}")
                self.sector_data[sector] = sector_stocks
            
            self.logger.info("Weekly data fetching completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching weekly data: {e}")
            return False
    
    def analyze_weekly_performance(self):
        """Analyze weekly market performance"""
        try:
            performance_data = {
                'indices_weekly': {},
                'sector_performance': {},
                'top_performers': {'gainers': [], 'losers': []},
                'volatility_analysis': {},
                'volume_analysis': {}
            }
            
            # Weekly indices performance
            for index, data in self.indices_data.items():
                if len(data) >= 7:
                    week_data = data.tail(7)  # Last 7 trading days
                    week_start_price = week_data['Close'].iloc[0]
                    week_end_price = week_data['Close'].iloc[-1]
                    week_change = ((week_end_price - week_start_price) / week_start_price) * 100
                    
                    week_high = week_data['High'].max()
                    week_low = week_data['Low'].min()
                    week_volume = week_data['Volume'].sum()
                    
                    # Technical indicators for the week
                    sma_20 = self.technical_analyzer.calculate_sma(data['Close'], 20).iloc[-1]
                    rsi = self.technical_analyzer.calculate_rsi(data['Close']).iloc[-1]
                    
                    performance_data['indices_weekly'][index] = {
                        'week_change_percent': round(week_change, 2),
                        'week_high': week_high,
                        'week_low': week_low,
                        'week_volume': week_volume,
                        'current_price': week_end_price,
                        'sma_20': sma_20,
                        'rsi': rsi,
                        'trend': 'Bullish' if week_change > 0 else 'Bearish'
                    }
            
            # Sector-wise performance analysis
            for sector, stocks in self.sector_data.items():
                sector_returns = []
                for stock, data in stocks.items():
                    if len(data) >= 7:
                        week_data = data.tail(7)
                        week_return = ((week_data['Close'].iloc[-1] - week_data['Close'].iloc[0]) 
                                     / week_data['Close'].iloc[0]) * 100
                        sector_returns.append(week_return)
                
                if sector_returns:
                    avg_sector_return = np.mean(sector_returns)
                    performance_data['sector_performance'][sector] = {
                        'avg_return': round(avg_sector_return, 2),
                        'stocks_analyzed': len(sector_returns),
                        'best_performer': max(sector_returns) if sector_returns else 0,
                        'worst_performer': min(sector_returns) if sector_returns else 0
                    }
            
            # Top gainers and losers from NIFTY 50
            stock_performance = []
            for stock, data in self.stocks_data.items():
                if len(data) >= 7:
                    week_data = data.tail(7)
                    week_return = ((week_data['Close'].iloc[-1] - week_data['Close'].iloc[0]) 
                                 / week_data['Close'].iloc[0]) * 100
                    
                    # Volume analysis
                    avg_volume = data['Volume'].tail(20).mean()
                    current_volume = week_data['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    stock_performance.append({
                        'stock': stock,
                        'week_return': week_return,
                        'current_price': week_data['Close'].iloc[-1],
                        'volume_ratio': volume_ratio
                    })
            
            # Sort and get top/bottom performers
            stock_performance.sort(key=lambda x: x['week_return'], reverse=True)
            performance_data['top_performers']['gainers'] = stock_performance[:10]
            performance_data['top_performers']['losers'] = stock_performance[-10:]
            
            # Volatility analysis
            for index, data in self.indices_data.items():
                if len(data) >= 20:
                    returns = data['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                    performance_data['volatility_analysis'][index] = round(volatility, 2)
            
            self.performance_data = performance_data
            self.logger.info("Weekly performance analysis completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in weekly performance analysis: {e}")
            return False
    
    def generate_charts(self):
        """Generate charts for the weekly report"""
        try:
            self.logger.info("Generating charts for weekly report...")
            
            # Set style for better looking charts
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            charts_dir = os.path.join(self.report_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # 1. Weekly Performance of Major Indices
            fig, ax = plt.subplots(figsize=(12, 8))
            indices_names = []
            weekly_returns = []
            
            for index, perf in self.performance_data['indices_weekly'].items():
                indices_names.append(index)
                weekly_returns.append(perf['week_change_percent'])
            
            colors = ['green' if x > 0 else 'red' for x in weekly_returns]
            bars = ax.bar(indices_names, weekly_returns, color=colors, alpha=0.7)
            
            ax.set_title(f'Weekly Performance of Indian Market Indices\n{self.week_start} to {self.week_end}', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Weekly Return (%)', fontsize=12)
            ax.set_xlabel('Indices', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, weekly_returns):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                       f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'weekly_indices_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Sector Performance Chart
            fig, ax = plt.subplots(figsize=(12, 8))
            sectors = list(self.performance_data['sector_performance'].keys())
            sector_returns = [self.performance_data['sector_performance'][sector]['avg_return'] 
                            for sector in sectors]
            
            colors = ['green' if x > 0 else 'red' for x in sector_returns]
            bars = ax.bar(sectors, sector_returns, color=colors, alpha=0.7)
            
            ax.set_title(f'Weekly Sector Performance Analysis\n{self.week_start} to {self.week_end}', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Average Return (%)', fontsize=12)
            ax.set_xlabel('Sectors', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, sector_returns):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                       f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'weekly_sector_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. NIFTY Price Chart with Technical Indicators
            if 'NIFTY' in self.indices_data:
                nifty_data = self.indices_data['NIFTY'].tail(30)  # Last 30 days
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
                
                # Price chart
                ax1.plot(nifty_data.index, nifty_data['Close'], label='NIFTY Close', linewidth=2)
                
                # Add SMA lines
                sma_5 = self.technical_analyzer.calculate_sma(nifty_data['Close'], 5)
                sma_20 = self.technical_analyzer.calculate_sma(nifty_data['Close'], 20)
                
                ax1.plot(nifty_data.index, sma_5, label='SMA 5', alpha=0.7)
                ax1.plot(nifty_data.index, sma_20, label='SMA 20', alpha=0.7)
                
                ax1.set_title('NIFTY Price Movement with Technical Indicators (Last 30 Days)', 
                             fontsize=14, fontweight='bold')
                ax1.set_ylabel('Price', fontsize=12)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Volume chart
                ax2.bar(nifty_data.index, nifty_data['Volume'], alpha=0.7, color='blue')
                ax2.set_title('Volume', fontsize=12)
                ax2.set_ylabel('Volume', fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, 'nifty_technical_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. Top Gainers and Losers
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Top Gainers
            gainers = self.performance_data['top_performers']['gainers'][:5]
            gainer_names = [g['stock'] for g in gainers]
            gainer_returns = [g['week_return'] for g in gainers]
            
            ax1.barh(gainer_names, gainer_returns, color='green', alpha=0.7)
            ax1.set_title('Top 5 Weekly Gainers', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Weekly Return (%)', fontsize=12)
            
            for i, v in enumerate(gainer_returns):
                ax1.text(v + 0.1, i, f'{v:.1f}%', va='center')
            
            # Top Losers
            losers = self.performance_data['top_performers']['losers'][:5]
            loser_names = [l['stock'] for l in losers]
            loser_returns = [l['week_return'] for l in losers]
            
            ax2.barh(loser_names, loser_returns, color='red', alpha=0.7)
            ax2.set_title('Top 5 Weekly Losers', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Weekly Return (%)', fontsize=12)
            
            for i, v in enumerate(loser_returns):
                ax2.text(v - 0.1, i, f'{v:.1f}%', va='center', ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'top_gainers_losers.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Volatility Comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            vol_indices = list(self.performance_data['volatility_analysis'].keys())
            volatilities = list(self.performance_data['volatility_analysis'].values())
            
            ax.bar(vol_indices, volatilities, color='orange', alpha=0.7)
            ax.set_title('Annualized Volatility Comparison', fontsize=16, fontweight='bold')
            ax.set_ylabel('Volatility (%)', fontsize=12)
            ax.set_xlabel('Indices', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            for i, v in enumerate(volatilities):
                ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'volatility_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Charts generation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
            return False
    
    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        try:
            self.logger.info("Generating PDF report...")
            
            report_filename = os.path.join(self.report_dir, f'Weekly_Market_Report_{self.report_date}.pdf')
            doc = SimpleDocTemplate(report_filename, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1,  # Center alignment
                textColor=colors.darkblue
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue,
                borderWidth=1,
                borderColor=colors.darkblue,
                borderPadding=5
            )
            
            # Title
            title = Paragraph(f"Weekly Indian Market Research Report<br/>Week {datetime.now().isocalendar()[1]}, 2022", title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            
            # Get NIFTY performance for summary
            nifty_perf = self.performance_data['indices_weekly'].get('NIFTY', {})
            sensex_perf = self.performance_data['indices_weekly'].get('SENSEX', {})
            
            summary_text = f"""
            <b>Market Overview ({self.week_start} to {self.week_end}):</b><br/>
            • NIFTY 50: {nifty_perf.get('week_change_percent', 'N/A')}% ({nifty_perf.get('trend', 'N/A')})<br/>
            • SENSEX: {sensex_perf.get('week_change_percent', 'N/A')}% ({sensex_perf.get('trend', 'N/A')})<br/>
            • Top Performing Sector: {max(self.performance_data['sector_performance'].keys(), 
                                        key=lambda x: self.performance_data['sector_performance'][x]['avg_return']) if self.performance_data['sector_performance'] else 'N/A'}<br/>
            • Market Volatility: {'High' if any(v > 25 for v in self.performance_data['volatility_analysis'].values()) else 'Moderate'}<br/>
            • Technical Outlook: {nifty_perf.get('trend', 'Neutral')} bias maintained
            """
            
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Add charts
            charts_dir = os.path.join(self.report_dir, 'charts')
            
            # Weekly Performance Chart
            story.append(Paragraph("1. Weekly Indices Performance", heading_style))
            if os.path.exists(os.path.join(charts_dir, 'weekly_indices_performance.png')):
                img = Image(os.path.join(charts_dir, 'weekly_indices_performance.png'), width=7*inch, height=4*inch)
                story.append(img)
            story.append(Spacer(1, 20))
            
            # Sector Performance
            story.append(Paragraph("2. Sector Analysis", heading_style))
            if os.path.exists(os.path.join(charts_dir, 'weekly_sector_performance.png')):
                img = Image(os.path.join(charts_dir, 'weekly_sector_performance.png'), width=7*inch, height=4*inch)
                story.append(img)
            
            # Sector performance table
            sector_data = []
            sector_data.append(['Sector', 'Weekly Return (%)', 'Stocks Analyzed', 'Status'])
            
            for sector, data in self.performance_data['sector_performance'].items():
                status = 'Outperforming' if data['avg_return'] > 0 else 'Underperforming'
                sector_data.append([
                    sector,
                    f"{data['avg_return']:.2f}%",
                    str(data['stocks_analyzed']),
                    status
                ])
            
            sector_table = Table(sector_data)
            sector_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(sector_table)
            story.append(PageBreak())
            
            # Technical Analysis
            story.append(Paragraph("3. Technical Analysis - NIFTY 50", heading_style))
            if os.path.exists(os.path.join(charts_dir, 'nifty_technical_analysis.png')):
                img = Image(os.path.join(charts_dir, 'nifty_technical_analysis.png'), width=7*inch, height=5*inch)
                story.append(img)
            
            # Technical summary
            if 'NIFTY' in self.performance_data['indices_weekly']:
                nifty_data = self.performance_data['indices_weekly']['NIFTY']
                tech_summary = f"""
                <b>Technical Indicators Summary:</b><br/>
                • Current Level: {nifty_data['current_price']:.2f}<br/>
                • 20-Day SMA: {nifty_data['sma_20']:.2f}<br/>
                • RSI: {nifty_data['rsi']:.2f}<br/>
                • Weekly High: {nifty_data['week_high']:.2f}<br/>
                • Weekly Low: {nifty_data['week_low']:.2f}<br/>
                • Trend: {nifty_data['trend']}
                """
                story.append(Paragraph(tech_summary, styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Top Performers
            story.append(Paragraph("4. Top Performers & Laggards", heading_style))
            if os.path.exists(os.path.join(charts_dir, 'top_gainers_losers.png')):
                img = Image(os.path.join(charts_dir, 'top_gainers_losers.png'), width=7*inch, height=4*inch)
                story.append(img)
            story.append(Spacer(1, 20))
            
            # Volatility Analysis
            story.append(Paragraph("5. Market Volatility Analysis", heading_style))
            if os.path.exists(os.path.join(charts_dir, 'volatility_analysis.png')):
                img = Image(os.path.join(charts_dir, 'volatility_analysis.png'), width=7*inch, height=4*inch)
                story.append(img)
            story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("6. Weekly Outlook & Recommendations", heading_style))
            
            # Generate basic recommendations based on analysis
            recommendations = self._generate_recommendations()
            story.append(Paragraph(recommendations, styles['Normal']))
            
            # Footer
            story.append(Spacer(1, 30))
            footer_text = f"""
            <i>This report is generated by the Market Research System v1.0 for the week ending {self.week_end}.<br/>
            Data sources: Yahoo Finance, NSE, BSE. For educational and research purposes only.<br/>
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
            """
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"PDF report generated: {report_filename}")
            return report_filename
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            return None
    
    def _generate_recommendations(self):
        """Generate basic recommendations based on analysis"""
        try:
            recommendations = []
            
            # Overall market trend
            nifty_trend = self.performance_data['indices_weekly'].get('NIFTY', {}).get('trend', 'Neutral')
            if nifty_trend == 'Bullish':
                recommendations.append("• Market shows bullish momentum. Consider gradually increasing equity exposure.")
            else:
                recommendations.append("• Market shows bearish/sideways trend. Maintain cautious approach with defensive stocks.")
            
            # Sector recommendations
            best_sector = max(self.performance_data['sector_performance'].keys(), 
                            key=lambda x: self.performance_data['sector_performance'][x]['avg_return']) \
                            if self.performance_data['sector_performance'] else None
            
            worst_sector = min(self.performance_data['sector_performance'].keys(), 
                             key=lambda x: self.performance_data['sector_performance'][x]['avg_return']) \
                             if self.performance_data['sector_performance'] else None
            
            if best_sector:
                recommendations.append(f"• {best_sector} sector showing strength. Look for quality stocks in this space.")
            
            if worst_sector:
                recommendations.append(f"• {worst_sector} sector underperforming. Exercise caution or consider profit booking.")
            
            # Volatility based recommendations
            avg_volatility = np.mean(list(self.performance_data['volatility_analysis'].values())) \
                           if self.performance_data['volatility_analysis'] else 20
            
            if avg_volatility > 25:
                recommendations.append("• High volatility observed. Consider reducing position sizes and using stop-losses.")
            else:
                recommendations.append("• Volatility is moderate. Good environment for systematic investment strategies.")
            
            # Technical recommendations
            nifty_rsi = self.performance_data['indices_weekly'].get('NIFTY', {}).get('rsi', 50)
            if nifty_rsi > 70:
                recommendations.append("• NIFTY RSI indicates overbought conditions. Expect some consolidation.")
            elif nifty_rsi < 30:
                recommendations.append("• NIFTY RSI indicates oversold conditions. Potential buying opportunity.")
            
            return "<br/>".join(recommendations)
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return "• Monitor market conditions closely and maintain disciplined approach to investing."
    
    def generate_report(self):
        """Main method to generate complete weekly report"""
        try:
            self.logger.info("Starting weekly report generation...")
            
            # Step 1: Fetch data
            if not self.fetch_weekly_data():
                self.logger.error("Failed to fetch weekly data")
                return False
            
            # Step 2: Analyze performance
            if not self.analyze_weekly_performance():
                self.logger.error("Failed to analyze weekly performance")
                return False
            
            # Step 3: Generate charts
            if not self.generate_charts():
                self.logger.error("Failed to generate charts")
                return False
            
            # Step 4: Generate PDF report
            report_file = self.generate_pdf_report()
            if not report_file:
                self.logger.error("Failed to generate PDF report")
                return False
            
            self.logger.info(f"Weekly report generation completed successfully!")
            self.logger.info(f"Report saved to: {report_file}")
            
            # Generate summary for logging
            self._log_report_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in report generation: {e}")
            return False
    
    def _log_report_summary(self):
        """Log report summary for tracking"""
        try:
            summary = {
                'report_date': self.report_date,
                'week_period': f"{self.week_start} to {self.week_end}",
                'indices_analyzed': len(self.indices_data),
                'stocks_analyzed': len(self.stocks_data),
                'sectors_analyzed': len(self.sector_data),
                'nifty_performance': self.performance_data['indices_weekly'].get('NIFTY', {}).get('week_change_percent', 'N/A'),
                'top_gainer': self.performance_data['top_performers']['gainers'][0]['stock'] if self.performance_data['top_performers']['gainers'] else 'N/A',
                'top_loser': self.performance_data['top_performers']['losers'][0]['stock'] if self.performance_data['top_performers']['losers'] else 'N/A'
            }
            
            self.logger.info("="*50)
            self.logger.info("WEEKLY REPORT SUMMARY")
            self.logger.info("="*50)
            for key, value in summary.items():
                self.logger.info(f"{key.upper()}: {value}")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Error logging report summary: {e}")

if __name__ == "__main__":
    try:
        # Initialize and generate weekly report
        generator = WeeklyReportGenerator()
        success = generator.generate_report()
        
        if success:
            print("✅ Weekly report generated successfully!")
            print(f"📁 Report saved in: {generator.report_dir}")
        else:
            print("❌ Failed to generate weekly report. Check logs for details.")
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        logging.error(f"Critical error in main: {e}")
                                          not complted