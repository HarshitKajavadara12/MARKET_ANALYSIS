"""
Market Research System v1.0 - Chart Utilities
Author: Independent Market Researcher
Created: 2022
Description: Chart utility functions for creating various financial charts and visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ChartUtils:
    """Utility class for creating financial charts and visualizations"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.dpi = 100
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6A994E',
            'background': '#F5F5F5',
            'text': '#2D3748'
        }
        
        # Indian market specific colors
        self.indian_colors = {
            'nifty': '#FF6B35',
            'sensex': '#004E89',
            'bank_nifty': '#1F7A8C',
            'it_index': '#BFDBF7',
            'positive': '#27AE60',
            'negative': '#E74C3C'
        }
    
    def create_price_chart(self, data: pd.DataFrame, symbol: str, 
                          indicators: Optional[Dict] = None, 
                          save_path: Optional[str] = None) -> str:
        """Create a price chart with optional technical indicators"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.plot(data.index, data['Close'], color=self.colors['primary'], 
                linewidth=2, label='Close Price')
        
        if indicators:
            if 'SMA_20' in indicators:
                ax1.plot(data.index, indicators['SMA_20'], 
                        color=self.colors['success'], linewidth=1, 
                        label='SMA 20', alpha=0.8)
            
            if 'SMA_50' in indicators:
                ax1.plot(data.index, indicators['SMA_50'], 
                        color=self.colors['danger'], linewidth=1, 
                        label='SMA 50', alpha=0.8)
            
            if 'Bollinger_Upper' in indicators and 'Bollinger_Lower' in indicators:
                ax1.fill_between(data.index, indicators['Bollinger_Upper'], 
                               indicators['Bollinger_Lower'], 
                               alpha=0.2, color=self.colors['info'], 
                               label='Bollinger Bands')
        
        ax1.set_title(f'{symbol} - Price Chart', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        colors = ['green' if close >= open_price else 'red' 
                 for close, open_price in zip(data['Close'], data['Open'])]
        ax2.bar(data.index, data['Volume'], color=colors, alpha=0.6)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def create_candlestick_chart(self, data: pd.DataFrame, symbol: str,
                                save_path: Optional[str] = None) -> str:
        """Create a candlestick chart"""
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Calculate candlestick dimensions
        width = 0.6
        width2 = 0.05
        
        # Determine colors
        up = data[data['Close'] >= data['Open']]
        down = data[data['Close'] < data['Open']]
        
        # Plot up prices
        ax.bar(up.index, up['Close'] - up['Open'], width, 
               bottom=up['Open'], color=self.indian_colors['positive'], alpha=0.8)
        ax.bar(up.index, up['High'] - up['Close'], width2, 
               bottom=up['Close'], color=self.indian_colors['positive'])
        ax.bar(up.index, up['Low'] - up['Open'], width2, 
               bottom=up['Open'], color=self.indian_colors['positive'])
        
        # Plot down prices
        ax.bar(down.index, down['Close'] - down['Open'], width, 
               bottom=down['Open'], color=self.indian_colors['negative'], alpha=0.8)
        ax.bar(down.index, down['High'] - down['Open'], width2, 
               bottom=down['Open'], color=self.indian_colors['negative'])
        ax.bar(down.index, down['Low'] - down['Close'], width2, 
               bottom=down['Close'], color=self.indian_colors['negative'])
        
        ax.set_title(f'{symbol} - Candlestick Chart', fontsize=16, fontweight='bold')
        ax.set_ylabel('Price (₹)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def create_sector_heatmap(self, sector_data: Dict, save_path: Optional[str] = None) -> str:
        """Create a sector performance heatmap"""
        
        # Convert to DataFrame for heatmap
        sectors = list(sector_data.keys())
        performance = list(sector_data.values())
        
        # Create matrix for heatmap
        data_matrix = np.array(performance).reshape(1, -1)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create heatmap
        sns.heatmap(data_matrix, 
                   xticklabels=sectors,
                   yticklabels=['Performance %'],
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=0,
                   square=False,
                   cbar_kws={'label': 'Performance (%)'},
                   ax=ax)
        
        ax.set_title('Sector Performance Heatmap', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def create_correlation_matrix(self, correlation_data: pd.DataFrame, 
                                save_path: Optional[str] = None) -> str:
        """Create correlation matrix heatmap"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create correlation heatmap
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax)
        
        ax.set_title('Stock Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def create_performance_comparison(self, performance_data: Dict, 
                                   save_path: Optional[str] = None) -> str:
        """Create performance comparison bar chart"""
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        symbols = list(performance_data.keys())
        returns = list(performance_data.values())
        
        # Color bars based on performance
        colors = [self.indian_colors['positive'] if ret >= 0 
                 else self.indian_colors['negative'] for ret in returns]
        
        bars = ax.bar(symbols, returns, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                   f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.set_title('Stock Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Returns (%)', fontsize=12)
        ax.set_xlabel('Stocks', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def create_technical_indicators_chart(self, data: pd.DataFrame, 
                                        indicators: Dict, symbol: str,
                                        save_path: Optional[str] = None) -> str:
        """Create technical indicators chart"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # RSI Chart
        if 'RSI' in indicators:
            ax1.plot(data.index, indicators['RSI'], color=self.colors['primary'], linewidth=2)
            ax1.axhline(y=70, color=self.colors['danger'], linestyle='--', alpha=0.7, label='Overbought')
            ax1.axhline(y=30, color=self.colors['success'], linestyle='--', alpha=0.7, label='Oversold')
            ax1.fill_between(data.index, 30, 70, alpha=0.1, color=self.colors['info'])
            ax1.set_title('RSI (Relative Strength Index)', fontweight='bold')
            ax1.set_ylabel('RSI')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # MACD Chart
        if 'MACD' in indicators and 'MACD_Signal' in indicators:
            ax2.plot(data.index, indicators['MACD'], color=self.colors['primary'], 
                    linewidth=2, label='MACD')
            ax2.plot(data.index, indicators['MACD_Signal'], color=self.colors['danger'], 
                    linewidth=2, label='Signal')
            if 'MACD_Histogram' in indicators:
                ax2.bar(data.index, indicators['MACD_Histogram'], 
                       color=self.colors['info'], alpha=0.6, label='Histogram')
            ax2.set_title('MACD', fontweight='bold')
            ax2.set_ylabel('MACD')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Bollinger Bands Chart
        if all(key in indicators for key in ['Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower']):
            ax3.plot(data.index, data['Close'], color=self.colors['primary'], linewidth=2, label='Close')
            ax3.plot(data.index, indicators['Bollinger_Upper'], color=self.colors['danger'], 
                    linewidth=1, label='Upper Band')
            ax3.plot(data.index, indicators['Bollinger_Middle'], color=self.colors['success'], 
                    linewidth=1, label='Middle Band')
            ax3.plot(data.index, indicators['Bollinger_Lower'], color=self.colors['danger'], 
                    linewidth=1, label='Lower Band')
            ax3.fill_between(data.index, indicators['Bollinger_Upper'], indicators['Bollinger_Lower'], 
                           alpha=0.1, color=self.colors['info'])
            ax3.set_title('Bollinger Bands', fontweight='bold')
            ax3.set_ylabel('Price (₹)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Moving Averages Chart (completing the incomplete section)
        if 'SMA_20' in indicators or 'EMA_20' in indicators:
            ax4.plot(data.index, data['Close'], color=self.colors['primary'], 
                    linewidth=2, label='Close')
            if 'SMA_20' in indicators:
                ax4.plot(data.index, indicators['SMA_20'], 
                        color=self.colors['success'], linewidth=1, 
                        label='SMA 20', alpha=0.8)
            if 'SMA_50' in indicators:
                ax4.plot(data.index, indicators['SMA_50'], 
                        color=self.colors['danger'], linewidth=1, 
                        label='SMA 50', alpha=0.8)
            if 'EMA_20' in indicators:
                ax4.plot(data.index, indicators['EMA_20'], 
                        color=self.colors['secondary'], linewidth=1, 
                        label='EMA 20', alpha=0.8)
            ax4.set_title('Moving Averages', fontweight='bold')
            ax4.set_ylabel('Price (₹)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Format all x-axis dates
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'{symbol} - Technical Indicators Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def create_index_comparison_chart(self, index_data: Dict[str, pd.DataFrame], 
                                    save_path: Optional[str] = None) -> str:
        """Create Indian market indices comparison chart"""
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Normalize all indices to start from 100 for comparison
        for index_name, data in index_data.items():
            normalized_data = (data['Close'] / data['Close'].iloc[0]) * 100
            
            # Use Indian market specific colors
            if 'NIFTY' in index_name.upper():
                color = self.indian_colors['nifty']
            elif 'SENSEX' in index_name.upper():
                color = self.indian_colors['sensex']
            elif 'BANK' in index_name.upper():
                color = self.indian_colors['bank_nifty']
            elif 'IT' in index_name.upper():
                color = self.indian_colors['it_index']
            else:
                color = self.colors['primary']
            
            ax.plot(data.index, normalized_data, 
                   color=color, linewidth=2, label=index_name, alpha=0.8)
        
        ax.set_title('Indian Market Indices Comparison (Normalized)', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Normalized Value (Base = 100)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def create_volume_analysis_chart(self, data: pd.DataFrame, symbol: str,
                                   save_path: Optional[str] = None) -> str:
        """Create volume analysis chart with volume indicators"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.fig_size[0], 12),
                                           gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Price chart
        ax1.plot(data.index, data['Close'], color=self.colors['primary'], 
                linewidth=2, label='Close Price')
        ax1.set_title(f'{symbol} - Volume Analysis', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart with color coding
        colors = ['green' if close >= open_price else 'red' 
                 for close, open_price in zip(data['Close'], data['Open'])]
        ax2.bar(data.index, data['Volume'], color=colors, alpha=0.6)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Volume moving average
        volume_ma = data['Volume'].rolling(window=20).mean()
        ax3.plot(data.index, data['Volume'], color=self.colors['info'], 
                alpha=0.6, label='Daily Volume')
        ax3.plot(data.index, volume_ma, color=self.colors['danger'], 
                linewidth=2, label='Volume MA(20)')
        ax3.set_ylabel('Volume', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def create_market_breadth_chart(self, breadth_data: Dict, 
                                  save_path: Optional[str] = None) -> str:
        """Create market breadth analysis chart for Indian market"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Advance-Decline Ratio
        if 'advances' in breadth_data and 'declines' in breadth_data:
            dates = breadth_data['dates']
            ad_ratio = np.array(breadth_data['advances']) / np.array(breadth_data['declines'])
            
            ax1.plot(dates, ad_ratio, color=self.colors['primary'], linewidth=2)
            ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Neutral')
            ax1.fill_between(dates, ad_ratio, 1, where=(ad_ratio >= 1), 
                           color=self.indian_colors['positive'], alpha=0.3, label='Bullish')
            ax1.fill_between(dates, ad_ratio, 1, where=(ad_ratio < 1), 
                           color=self.indian_colors['negative'], alpha=0.3, label='Bearish')
            ax1.set_title('Advance-Decline Ratio', fontweight='bold')
            ax1.set_ylabel('A/D Ratio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # New Highs vs New Lows
        if 'new_highs' in breadth_data and 'new_lows' in breadth_data:
            ax2.bar(dates, breadth_data['new_highs'], color=self.indian_colors['positive'], 
                   alpha=0.7, label='New Highs')
            ax2.bar(dates, [-x for x in breadth_data['new_lows']], 
                   color=self.indian_colors['negative'], alpha=0.7, label='New Lows')
            ax2.set_title('New Highs vs New Lows', fontweight='bold')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Sector Performance
        if 'sector_performance' in breadth_data:
            sectors = list(breadth_data['sector_performance'].keys())
            performance = list(breadth_data['sector_performance'].values())
            
            colors = [self.indian_colors['positive'] if p >= 0 
                     else self.indian_colors['negative'] for p in performance]
            
            bars = ax3.barh(sectors, performance, color=colors, alpha=0.8)
            ax3.set_title('Sector Performance (%)', fontweight='bold')
            ax3.set_xlabel('Performance (%)')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, performance):
                width = bar.get_width()
                ax3.text(width + (0.1 if width >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}%', ha='left' if width >= 0 else 'right', va='center')
        
        # Market Cap Distribution
        if 'market_cap_dist' in breadth_data:
            cap_categories = list(breadth_data['market_cap_dist'].keys())
            cap_values = list(breadth_data['market_cap_dist'].values())
            
            ax4.pie(cap_values, labels=cap_categories, autopct='%1.1f%%',
                   colors=[self.colors['primary'], self.colors['secondary'], 
                          self.colors['success'], self.colors['info']])
            ax4.set_title('Market Cap Distribution', fontweight='bold')
        
        plt.suptitle('Indian Market Breadth Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def create_daily_summary_chart(self, summary_data: Dict, 
                                 save_path: Optional[str] = None) -> str:
        """Create daily market summary chart for Indian market"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Index Performance
        if 'indices' in summary_data:
            indices = list(summary_data['indices'].keys())
            changes = list(summary_data['indices'].values())
            
            colors = [self.indian_colors['positive'] if change >= 0 
                     else self.indian_colors['negative'] for change in changes]
            
            bars = ax1.bar(indices, changes, color=colors, alpha=0.8)
            ax1.set_title('Major Indices Performance', fontweight='bold')
            ax1.set_ylabel('Change (%)')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, changes):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.01 if height >= 0 else -0.03),
                        f'{value:.2f}%', ha='center', 
                        va='bottom' if height >= 0 else 'top')
        
        # Top Gainers
        if 'top_gainers' in summary_data:
            gainers = summary_data['top_gainers']
            stocks = [stock['symbol'] for stock in gainers[:10]]
            gains = [stock['change_pct'] for stock in gainers[:10]]
            
            ax2.barh(stocks, gains, color=self.indian_colors['positive'], alpha=0.8)
            ax2.set_title('Top 10 Gainers', fontweight='bold')
            ax2.set_xlabel('Change (%)')
            ax2.grid(True, alpha=0.3)
        
        # Top Losers  
        if 'top_losers' in summary_data:
            losers = summary_data['top_losers']
            stocks = [stock['symbol'] for stock in losers[:10]]
            losses = [stock['change_pct'] for stock in losers[:10]]
            
            ax3.barh(stocks, losses, color=self.indian_colors['negative'], alpha=0.8)
            ax3.set_title('Top 10 Losers', fontweight='bold')
            ax3.set_xlabel('Change (%)')
            ax3.grid(True, alpha=0.3)
        
        # Volume Leaders
        if 'volume_leaders' in summary_data:
            vol_leaders = summary_data['volume_leaders']
            stocks = [stock['symbol'] for stock in vol_leaders[:10]]
            volumes = [stock['volume'] for stock in vol_leaders[:10]]
            
            ax4.bar(stocks, volumes, color=self.colors['info'], alpha=0.8)
            ax4.set_title('Volume Leaders', fontweight='bold')
            ax4.set_ylabel('Volume (Crores)')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'Indian Market Daily Summary - {datetime.now().strftime("%d %B %Y")}', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def save_chart_as_pdf(self, fig, filename: str, title: str = "Market Analysis Report"):
        """Save chart as PDF with proper formatting for Version 1"""
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(filename) as pdf:
            # Add metadata
            pdf.savefig(fig, bbox_inches='tight', dpi=self.dpi)
            
            # Add report metadata
            d = pdf.infodict()
            d['Title'] = title
            d['Author'] = 'Independent Market Researcher'
            d['Subject'] = 'Indian Market Analysis'
            d['Keywords'] = 'Stock Market, Technical Analysis, Indian Stocks'
            d['Creator'] = 'Market Research System v1.0'
            d['Producer'] = 'Market Research System v1.0'
    
    def get_indian_market_colors(self) -> Dict:
        """Return Indian market specific color scheme"""
        return self.indian_colors.copy()
    
    def set_indian_market_style(self):
        """Set chart style optimized for Indian market reports"""
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })


# Version 1 specific utility functions for 2022 Indian market focus
def create_nifty_analysis_chart(data: pd.DataFrame, save_path: str = None):
    """Specialized function for NIFTY 50 analysis - Version 1 (2022)"""
    chart_utils = ChartUtils()
    
    # Calculate basic indicators for 2022 version
    indicators = {
        'SMA_20': data['Close'].rolling(window=20).mean(),
        'SMA_50': data['Close'].rolling(window=50).mean(),
        'RSI': calculate_rsi(data['Close'], window=14)
    }
    
    return chart_utils.create_price_chart(data, 'NIFTY 50', indicators, save_path)

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI for Version 1 - Basic implementation"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_indian_market_summary(date: str = None) -> Dict:
    """Generate market summary data structure for Indian market - Version 1"""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    return {
        'date': date,
        'indices': {
            'NIFTY 50': 0.0,
            'SENSEX': 0.0,
            'BANK NIFTY': 0.0,
            'NIFTY IT': 0.0
        },
        'top_gainers': [],
        'top_losers': [],
        'volume_leaders': [],
        'market_breadth': {
            'advances': 0,
            'declines': 0,
            'unchanged': 0
        }
    }