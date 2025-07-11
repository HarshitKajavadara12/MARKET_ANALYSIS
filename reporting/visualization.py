"""
Market Visualization Module for Market Research System v1.0
==========================================================

Creates professional charts and graphs using Matplotlib and Seaborn.
Specialized for Indian stock market analysis.

Author: Independent Market Researcher
Created: March 2022
Last Updated: December 2022
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MarketVisualizer:
    """
    Professional market visualization for Indian stock market research
    """
    
    def __init__(self, output_dir: str = "charts/", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize Market Visualizer
        
        Args:
            output_dir: Directory to save chart images
            figsize: Default figure size for charts
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.colors = {
            'bullish': '#2E8B57',      # Sea Green
            'bearish': '#DC143C',       # Crimson
            'neutral': '#4682B4',       # Steel Blue
            'volume': '#FFD700',        # Gold
            'support': '#32CD32',       # Lime Green
            'resistance': '#FF6347',    # Tomato
            'background': '#F5F5F5',    # White Smoke
            'grid': '#DCDCDC'          # Gainsboro
        }
        self._ensure_output_dir()
        self._setup_matplotlib_params()
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def _setup_matplotlib_params(self):
        """Setup matplotlib parameters for professional charts"""
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'axes.facecolor': self.colors['background'],
            'axes.edgecolor': 'black',
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.color': self.colors['grid'],
            'grid.linestyle': '-',
            'grid.linewidth': 0.5,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def plot_price_chart(self, data: pd.DataFrame, symbol: str, 
                        indicators: Optional[Dict] = None, 
                        save_path: Optional[str] = None) -> str:
        """
        Create comprehensive price chart with technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            indicators: Dictionary of technical indicators to plot
            save_path: Custom save path for chart
        
        Returns:
            Path to saved chart
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{symbol}_price_chart.png")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.plot(data.index, data['Close'], color=self.colors['neutral'], 
                linewidth=2, label='Close Price')
        
        # Add moving averages if provided
        if indicators:
            if 'SMA_20' in indicators:
                ax1.plot(data.index, indicators['SMA_20'], 
                        color='orange', linewidth=1.5, label='SMA 20', alpha=0.8)
            if 'SMA_50' in indicators:
                ax1.plot(data.index, indicators['SMA_50'], 
                        color='red', linewidth=1.5, label='SMA 50', alpha=0.8)
            if 'EMA_12' in indicators:
                ax1.plot(data.index, indicators['EMA_12'], 
                        color='green', linewidth=1.5, label='EMA 12', alpha=0.8)
        
        # Support and resistance lines
        if indicators and 'support' in indicators:
            ax1.axhline(y=indicators['support'], color=self.colors['support'], 
                       linestyle='--', alpha=0.7, label='Support')
        if indicators and 'resistance' in indicators:
            ax1.axhline(y=indicators['resistance'], color=self.colors['resistance'], 
                       linestyle='--', alpha=0.7, label='Resistance')
        
        # Volume chart
        colors = [self.colors['bullish'] if close >= open_price else self.colors['bearish'] 
                 for close, open_price in zip(data['Close'], data['Open'])]
        ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7)
        
        # Formatting
        ax1.set_title(f'{symbol} - Price Movement with Technical Indicators', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Volume', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def plot_candlestick_chart(self, data: pd.DataFrame, symbol: str, 
                              save_path: Optional[str] = None) -> str:
        """
        Create candlestick chart for OHLC data
        
        Args:
            data: DataFrame with OHLC data
            symbol: Stock symbol
            save_path: Custom save path
        
        Returns:
            Path to saved chart
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{symbol}_candlestick.png")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate candlestick components
        for i, (date, row) in enumerate(data.iterrows()):
            color = self.colors['bullish'] if row['Close'] >= row['Open'] else self.colors['bearish']
            
            # High-Low line
            ax.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1)
            
            # Body rectangle
            body_height = abs(row['Close'] - row['Open'])
            bottom = min(row['Open'], row['Close'])
            
            rect = plt.Rectangle((i-0.3, bottom), 0.6, body_height, 
                               facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
        
        # Formatting
        ax.set_title(f'{symbol} - Candlestick Chart', fontsize=16, fontweight='bold')
        ax.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        step = max(1, len(data) // 10)
        ax.set_xticks(range(0, len(data), step))
        ax.set_xticklabels([data.index[i].strftime('%Y-%m-%d') for i in range(0, len(data), step)], 
                          rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def plot_sector_performance(self, sector_data: Dict[str, float], 
                               title: str = "Sector Performance", 
                               save_path: Optional[str] = None) -> str:
        """
        Create sector performance bar chart
        
        Args:
            sector_data: Dictionary of sector names and performance values
            title: Chart title
            save_path: Custom save path
        
        Returns:
            Path to saved chart
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "sector_performance.png")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sectors = list(sector_data.keys())
        performance = list(sector_data.values())
        
        # Color based on performance
        colors = [self.colors['bullish'] if perf >= 0 else self.colors['bearish'] 
                 for perf in performance]
        
        bars = ax.bar(sectors, performance, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, performance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                   f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                   fontweight='bold')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sectors', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                title: str = "Stock Correlation Matrix", 
                                save_path: Optional[str] = None) -> str:
        """
        Create correlation heatmap for Indian stocks
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Chart title
            save_path: Custom save path
        
        Returns:
            Path to saved chart
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.2f', ax=ax)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def plot_nifty_vs_stock(self, nifty_data: pd.Series, stock_data: pd.Series, 
                           stock_symbol: str, save_path: Optional[str] = None) -> str:
        """
        Compare individual stock performance with NIFTY 50
        
        Args:
            nifty_data: NIFTY 50 price series
            stock_data: Individual stock price series
            stock_symbol: Stock symbol for labeling
            save_path: Custom save path
        
        Returns:
            Path to saved chart
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{stock_symbol}_vs_nifty.png")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Normalize both series to start from 100 for comparison
        nifty_normalized = (nifty_data / nifty_data.iloc[0]) * 100
        stock_normalized = (stock_data / stock_data.iloc[0]) * 100
        
        # Plot both lines
        ax.plot(nifty_normalized.index, nifty_normalized, 
               color=self.colors['neutral'], linewidth=2, label='NIFTY 50')
        ax.plot(stock_normalized.index, stock_normalized, 
               color=self.colors['bullish'], linewidth=2, label=stock_symbol)
        
        # Add horizontal line at 100 (starting point)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_title(f'{stock_symbol} vs NIFTY 50 Performance Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Normalized Price (Base = 100)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def plot_rsi_chart(self, data: pd.DataFrame, rsi_values: pd.Series, 
                      symbol: str, save_path: Optional[str] = None) -> str:
        """
        Create RSI chart with overbought/oversold levels
        
        Args:
            data: Price DataFrame
            rsi_values: RSI values series
            symbol: Stock symbol
            save_path: Custom save path
        
        Returns:
            Path to saved chart
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{symbol}_rsi_chart.png")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # Price chart
        ax1.plot(data.index, data['Close'], color=self.colors['neutral'], 
                linewidth=2, label='Close Price')
        ax1.set_title(f'{symbol} - Price and RSI Analysis', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI chart
        ax2.plot(rsi_values.index, rsi_values, color='purple', linewidth=2, label='RSI')
        ax2.axhline(y=70, color=self.colors['bearish'], linestyle='--', 
                   alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color=self.colors['bullish'], linestyle='--', 
                   alpha=0.7, label='Oversold (30)')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Midline')
        
        # Fill overbought/oversold areas
        ax2.fill_between(rsi_values.index, 70, 100, alpha=0.2, color=self.colors['bearish'])
        ax2.fill_between(rsi_values.index, 0, 30, alpha=0.2, color=self.colors['bullish'])
        
        ax2.set_title('RSI (14)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RSI', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def plot_volume_analysis(self, data: pd.DataFrame, symbol: str, 
                           save_path: Optional[str] = None) -> str:
        """
        Create volume analysis chart with moving averages
        
        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol
            save_path: Custom save path
        
        Returns:
            Path to saved chart
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{symbol}_volume_analysis.png")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # Price chart
        ax1.plot(data.index, data['Close'], color=self.colors['neutral'], 
                linewidth=2, label='Close Price')
        
        # Volume chart
        colors = [self.colors['bullish'] if close >= open_price else self.colors['bearish'] 
                 for close, open_price in zip(data['Close'], data['Open'])]
        ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7, label='Volume')
        
        # Volume moving average
        volume_ma = data['Volume'].rolling(window=20).mean()
        ax2.plot(data.index, volume_ma, color='orange', 
                linewidth=2, label='Volume MA(20)')
        
        # Formatting
        ax1.set_title(f'{symbol} - Price and Volume Analysis', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Volume with Moving Average', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def plot_indian_market_overview(self, market_data: Dict[str, Dict], 
                                   save_path: Optional[str] = None) -> str:
        """
        Create comprehensive Indian market overview dashboard
        
        Args:
            market_data: Dictionary containing market indices data
                        Format: {'NIFTY50': {'current': 18500, 'change': 1.2, 'change_pct': 0.65}}
            save_path: Custom save path
        
        Returns:
            Path to saved chart
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "indian_market_overview.png")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Indian Stock Market Overview', fontsize=20, fontweight='bold', y=0.95)
        
        # Extract data
        indices = list(market_data.keys())
        current_values = [market_data[idx]['current'] for idx in indices]
        changes = [market_data[idx]['change'] for idx in indices]
        change_pcts = [market_data[idx]['change_pct'] for idx in indices]
        
        # 1. Current Index Values
        bars1 = ax1.bar(indices, current_values, color=self.colors['neutral'], alpha=0.8)
        ax1.set_title('Current Index Values', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Index Value', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, current_values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(current_values)*0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Absolute Changes
        colors2 = [self.colors['bullish'] if change >= 0 else self.colors['bearish'] 
                  for change in changes]
        bars2 = ax2.bar(indices, changes, color=colors2, alpha=0.8)
        ax2.set_title('Absolute Change (Points)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Change (Points)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars2, changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -10),
                    f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
        
        # 3. Percentage Changes
        colors3 = [self.colors['bullish'] if pct >= 0 else self.colors['bearish'] 
                  for pct in change_pcts]
        bars3 = ax3.bar(indices, change_pcts, color=colors3, alpha=0.8)
        ax3.set_title('Percentage Change (%)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Change (%)', fontsize=12)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars3, change_pcts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                    f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
        
        # 4. Market Heat Map (Performance Matrix)
        performance_matrix = np.array([change_pcts])
        im = ax4.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)
        ax4.set_title('Performance Heat Map', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(indices)))
        ax4.set_xticklabels(indices, rotation=45)
        ax4.set_yticks([])
        
        # Add percentage labels on heatmap
        for i, pct in enumerate(change_pcts):
            ax4.text(i, 0, f'{pct:.2f}%', ha='center', va='center', 
                    fontweight='bold', color='white' if abs(pct) > 1 else 'black')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax4, shrink=0.6)
        cbar.set_label('Change (%)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def create_daily_summary_chart(self, summary_data: Dict[str, Any], 
                                  date: str, save_path: Optional[str] = None) -> str:
        """
        Create daily market summary visualization
        
        Args:
            summary_data: Dictionary containing daily market summary
            date: Date string for the summary
            save_path: Custom save path
        
        Returns:
            Path to saved chart
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"daily_summary_{date}.png")
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'Daily Market Summary - {date}', fontsize=18, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Top performers
        ax1 = fig.add_subplot(gs[0, 0])
        if 'top_gainers' in summary_data:
            gainers = summary_data['top_gainers']
            y_pos = np.arange(len(gainers))
            ax1.barh(y_pos, [g['change_pct'] for g in gainers], 
                    color=self.colors['bullish'], alpha=0.8)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([g['symbol'] for g in gainers])
            ax1.set_title('Top Gainers (%)', fontweight='bold')
            ax1.set_xlabel('Change %')
        
        # Top losers
        ax2 = fig.add_subplot(gs[0, 1])
        if 'top_losers' in summary_data:
            losers = summary_data['top_losers']
            y_pos = np.arange(len(losers))
            ax2.barh(y_pos, [l['change_pct'] for l in losers], 
                    color=self.colors['bearish'], alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([l['symbol'] for l in losers])
            ax2.set_title('Top Losers (%)', fontweight='bold')
            ax2.set_xlabel('Change %')
        
        # Volume leaders
        ax3 = fig.add_subplot(gs[0, 2])
        if 'volume_leaders' in summary_data:
            volume_leaders = summary_data['volume_leaders']
            ax3.bar(range(len(volume_leaders)), 
                   [v['volume'] for v in volume_leaders],
                   color=self.colors['volume'], alpha=0.8)
            ax3.set_xticks(range(len(volume_leaders)))
            ax3.set_xticklabels([v['symbol'] for v in volume_leaders], rotation=45)
            ax3.set_title('Volume Leaders', fontweight='bold')
            ax3.set_ylabel('Volume (Cr)')
        
        # Market breadth
        ax4 = fig.add_subplot(gs[1, :])
        if 'market_breadth' in summary_data:
            breadth = summary_data['market_breadth']
            categories = ['Advances', 'Declines', 'Unchanged']
            values = [breadth.get('advances', 0), breadth.get('declines', 0), 
                     breadth.get('unchanged', 0)]
            colors = [self.colors['bullish'], self.colors['bearish'], self.colors['neutral']]
            
            ax4.pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=(0.05, 0.05, 0.05))
            ax4.set_title('Market Breadth Distribution', fontweight='bold')
        
        # Index performance table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        if 'indices_performance' in summary_data:
            indices_data = summary_data['indices_performance']
            table_data = []
            for idx, data in indices_data.items():
                table_data.append([idx, f"{data['current']:.2f}", 
                                 f"{data['change']:+.2f}", f"{data['change_pct']:+.2f}%"])
            
            table = ax5.table(cellText=table_data,
                            colLabels=['Index', 'Current', 'Change', 'Change %'],
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color code the change columns
            for i in range(1, len(table_data) + 1):
                change_pct = float(table_data[i-1][3].rstrip('%'))
                color = self.colors['bullish'] if change_pct >= 0 else self.colors['bearish']
                table[(i, 2)].set_facecolor(color)
                table[(i, 3)].set_facecolor(color)
                table[(i, 2)].set_text_props(weight='bold', color='white')
                table[(i, 3)].set_text_props(weight='bold', color='white')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def save_all_charts(self, data_dict: Dict[str, pd.DataFrame], 
                       indicators_dict: Optional[Dict] = None) -> List[str]:
        """
        Generate and save all charts for multiple stocks
        
        Args:
            data_dict: Dictionary of stock symbols and their data
            indicators_dict: Dictionary of technical indicators for each stock
        
        Returns:
            List of paths to saved charts
        """
        saved_charts = []
        
        for symbol, data in data_dict.items():
            try:
                # Get indicators for this symbol if available
                indicators = indicators_dict.get(symbol) if indicators_dict else None
                
                # Generate different types of charts
                chart_paths = [
                    self.plot_price_chart(data, symbol, indicators),
                    self.plot_candlestick_chart(data, symbol),
                    self.plot_volume_analysis(data, symbol)
                ]
                
                # Add RSI chart if RSI data is available
                if indicators and 'RSI' in indicators:
                    chart_paths.append(
                        self.plot_rsi_chart(data, indicators['RSI'], symbol)
                    )
                
                saved_charts.extend(chart_paths)
                
            except Exception as e:
                print(f"Error generating charts for {symbol}: {str(e)}")
                continue
        
        return saved_charts