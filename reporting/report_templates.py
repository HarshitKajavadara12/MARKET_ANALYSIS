"""
Market Research System v1.0 - Report Templates
Author: Independent Market Researcher
Created: 2022
Description: Report template definitions for generating market analysis reports
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class ReportTemplates:
    """Report template definitions for various market analysis reports"""
    
    def __init__(self):
        self.template_styles = {
            'title_font_size': 16,
            'header_font_size': 14,
            'body_font_size': 12,
            'chart_width': 800,
            'chart_height': 600,
            'colors': {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'success': '#F18F01',
                'danger': '#C73E1D',
                'background': '#F5F5F5'
            }
        }
    
    def daily_market_summary_template(self, market_data: Dict) -> Dict:
        """Template for daily market summary report"""
        
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        template = {
            'report_type': 'Daily Market Summary',
            'report_date': report_date,
            'title': f'Daily Market Summary - {report_date}',
            'sections': [
                {
                    'name': 'Market Overview',
                    'type': 'overview',
                    'data': market_data.get('overview', {}),
                    'charts': ['market_indices_chart', 'sector_performance_chart']
                },
                {
                    'name': 'Top Performers',
                    'type': 'table',
                    'data': market_data.get('top_gainers', []),
                    'columns': ['Symbol', 'Price', 'Change', 'Change %', 'Volume']
                },
                {
                    'name': 'Worst Performers',
                    'type': 'table',
                    'data': market_data.get('top_losers', []),
                    'columns': ['Symbol', 'Price', 'Change', 'Change %', 'Volume']
                },
                {
                    'name': 'Technical Analysis',
                    'type': 'analysis',
                    'data': market_data.get('technical_indicators', {}),
                    'charts': ['rsi_chart', 'macd_chart', 'bollinger_bands_chart']
                },
                {
                    'name': 'Market Sentiment',
                    'type': 'sentiment',
                    'data': market_data.get('sentiment', {}),
                    'charts': ['sentiment_gauge']
                }
            ],
            'footer': {
                'disclaimer': 'This report is for informational purposes only and should not be considered as investment advice.',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_sources': ['NSE India', 'BSE India', 'Yahoo Finance', 'Economic Times']
            }
        }
        
        return template
    
    def weekly_sector_analysis_template(self, sector_data: Dict) -> Dict:
        """Template for weekly sector analysis report"""
        
        report_date = datetime.now().strftime('%Y-%m-%d')
        week_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        template = {
            'report_type': 'Weekly Sector Analysis',
            'report_date': report_date,
            'period': f'{week_start} to {report_date}',
            'title': f'Weekly Sector Analysis - Week Ending {report_date}',
            'sections': [
                {
                    'name': 'Sector Performance Overview',
                    'type': 'overview',
                    'data': sector_data.get('overview', {}),
                    'charts': ['sector_heatmap', 'sector_performance_bar']
                },
                {
                    'name': 'Banking & Financial Services',
                    'type': 'sector_detail',
                    'data': sector_data.get('banking', {}),
                    'charts': ['banking_stocks_chart']
                },
                {
                    'name': 'Information Technology',
                    'type': 'sector_detail',
                    'data': sector_data.get('it', {}),
                    'charts': ['it_stocks_chart']
                },
                {
                    'name': 'Healthcare & Pharmaceuticals',
                    'type': 'sector_detail',
                    'data': sector_data.get('healthcare', {}),
                    'charts': ['healthcare_stocks_chart']
                },
                {
                    'name': 'Energy & Power',
                    'type': 'sector_detail',
                    'data': sector_data.get('energy', {}),
                    'charts': ['energy_stocks_chart']
                },
                {
                    'name': 'FMCG & Consumer Goods',
                    'type': 'sector_detail',
                    'data': sector_data.get('fmcg', {}),
                    'charts': ['fmcg_stocks_chart']
                },
                {
                    'name': 'Sector Correlation Analysis',
                    'type': 'correlation',
                    'data': sector_data.get('correlation', {}),
                    'charts': ['correlation_matrix']
                }
            ],
            'footer': {
                'disclaimer': 'Sector analysis based on historical data and current market conditions.',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_sources': ['NSE Sectoral Indices', 'BSE Sector Data', 'Company Financials']
            }
        }
        
        return template
    
    def monthly_performance_review_template(self, performance_data: Dict) -> Dict:
        """Template for monthly performance review report"""
        
        current_date = datetime.now()
        report_month = current_date.strftime('%B %Y')
        
        template = {
            'report_type': 'Monthly Performance Review',
            'report_date': current_date.strftime('%Y-%m-%d'),
            'period': report_month,
            'title': f'Monthly Performance Review - {report_month}',
            'sections': [
                {
                    'name': 'Executive Summary',
                    'type': 'executive_summary',
                    'data': performance_data.get('summary', {}),
                    'key_metrics': ['nifty_return', 'sensex_return', 'volatility', 'volume']
                },
                {
                    'name': 'Index Performance',
                    'type': 'index_performance',
                    'data': performance_data.get('indices', {}),
                    'charts': ['monthly_returns_chart', 'cumulative_returns_chart']
                },
                {
                    'name': 'Stock Analysis',
                    'type': 'stock_analysis',
                    'data': performance_data.get('stocks', {}),
                    'charts': ['stock_performance_scatter', 'risk_return_plot']
                },
                {
                    'name': 'Economic Indicators Impact',
                    'type': 'economic_analysis',
                    'data': performance_data.get('economic', {}),
                    'charts': ['economic_indicators_chart']
                },
                {
                    'name': 'Risk Metrics',
                    'type': 'risk_analysis',
                    'data': performance_data.get('risk', {}),
                    'charts': ['volatility_chart', 'drawdown_chart']
                },
                {
                    'name': 'Market Outlook',
                    'type': 'outlook',
                    'data': performance_data.get('outlook', {}),
                    'insights': ['technical_outlook', 'fundamental_outlook', 'sentiment_outlook']
                }
            ],
            'footer': {
                'disclaimer': 'Performance review based on historical data and market analysis.',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_sources': ['NSE', 'BSE', 'RBI', 'Ministry of Statistics']
            }
        }
        
        return template
    
    def custom_research_template(self, research_data: Dict, template_config: Dict) -> Dict:
        """Template for custom research reports"""
        
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        template = {
            'report_type': template_config.get('report_type', 'Custom Research Report'),
            'report_date': report_date,
            'title': template_config.get('title', f'Custom Research Report - {report_date}'),
            'sections': [],
            'footer': {
                'disclaimer': template_config.get('disclaimer', 'Custom research analysis for informational purposes.'),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_sources': template_config.get('data_sources', ['Various Market Sources'])
            }
        }
        
        # Build sections based on configuration
        for section_config in template_config.get('sections', []):
            section = {
                'name': section_config.get('name', 'Untitled Section'),
                'type': section_config.get('type', 'generic'),
                'data': research_data.get(section_config.get('data_key', ''), {}),
                'charts': section_config.get('charts', []),
                'tables': section_config.get('tables', [])
            }
            template['sections'].append(section)
        
        return template
    
    def correlation_analysis_template(self, correlation_data: Dict) -> Dict:
        """Template for correlation analysis report"""
        
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        template = {
            'report_type': 'Correlation Analysis Report',
            'report_date': report_date,
            'title': f'Market Correlation Analysis - {report_date}',
            'sections': [
                {
                    'name': 'Correlation Overview',
                    'type': 'overview',
                    'data': correlation_data.get('overview', {}),
                    'charts': ['correlation_heatmap']
                },
                {
                    'name': 'Stock Correlations',
                    'type': 'correlation_detail',
                    'data': correlation_data.get('stock_correlations', {}),
                    'charts': ['stock_correlation_matrix', 'correlation_network']
                },
                {
                    'name': 'Sector Correlations',
                    'type': 'correlation_detail',
                    'data': correlation_data.get('sector_correlations', {}),
                    'charts': ['sector_correlation_matrix']
                },
                {
                    'name': 'Economic Factor Correlations',
                    'type': 'correlation_detail',
                    'data': correlation_data.get('economic_correlations', {}),
                    'charts': ['economic_correlation_chart']
                },
                {
                    'name': 'Portfolio Implications',
                    'type': 'insights',
                    'data': correlation_data.get('portfolio_insights', {}),
                    'recommendations': correlation_data.get('recommendations', [])
                }
            ],
            'footer': {
                'disclaimer': 'Correlation analysis based on historical price movements.',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_sources': ['NSE', 'BSE', 'Historical Price Data']
            }
        }
        
        return template
    
    def get_template_config(self, template_type: str) -> Dict:
        """Get configuration for specific template type"""
        
        configs = {
            'daily_summary': {
                'page_size': 'A4',
                'orientation': 'portrait',
                'margins': {'top': 2, 'bottom': 2, 'left': 1.5, 'right': 1.5},
                'font_family': 'Arial',
                'include_charts': True,
                'include_tables': True
            },
            'weekly_sector': {
                'page_size': 'A4',
                'orientation': 'portrait',
                'margins': {'top': 2, 'bottom': 2, 'left': 1.5, 'right': 1.5},
                'font_family': 'Arial',
                'include_charts': True,
                'include_tables': True
            },
            'monthly_performance': {
                'page_size': 'A4',
                'orientation': 'portrait',
                'margins': {'top': 2, 'bottom': 2, 'left': 1.5, 'right': 1.5},
                'font_family': 'Arial',
                'include_charts': True,
                'include_tables': True
            },
            'correlation_analysis': {
                'page_size': 'A4',
                'orientation': 'landscape',
                'margins': {'top': 1.5, 'bottom': 1.5, 'left': 1, 'right': 1},
                'font_family': 'Arial',
                'include_charts': True,
                'include_tables': False
            }
        }
        
        return configs.get(template_type, configs['daily_summary'])
    
    def get_indian_stock_symbols(self) -> Dict[str, List[str]]:
        """Get Indian stock symbols organized by sector"""
        
        return {
            'banking': [
                'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 
                'AXISBANK.NS', 'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS'
            ],
            'it': [
                'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 
                'TECHM.NS', 'LTI.NS', 'MINDTREE.NS', 'MPHASIS.NS'
            ],
            'healthcare': [
                'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS',
                'LUPIN.NS', 'BIOCON.NS', 'CADILAHC.NS', 'TORNTPHARM.NS'
            ],
            'energy': [
                'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS',
                'HINDPETRO.NS', 'GAIL.NS', 'POWERGRID.NS', 'NTPC.NS'
            ],
            'fmcg': [
                'HINDUNILV.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
                'DABUR.NS', 'GODREJCP.NS', 'MARICO.NS', 'COLPAL.NS'
            ],
            'auto': [
                'MARUTI.NS', 'HYUNDAI.NS', 'TATAMOTORS.NS', 'M&M.NS',
                'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'TVSMOTORS.NS'
            ]
        }
    
    def get_market_indices(self) -> List[str]:
        """Get Indian market indices symbols"""
        
        return [
            '^NSEI',      # Nifty 50
            '^BSESN',     # BSE Sensex
            '^NSMIDCP',   # Nifty Midcap
            '^CNXIT',     # Nifty IT
            '^CNXBANK',   # Nifty Bank
            '^CNXFMCG',   # Nifty FMCG
            '^CNXPHARMA', # Nifty Pharma
            '^CNXAUTO'    # Nifty Auto
        ]