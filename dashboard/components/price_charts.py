import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class PriceCharts:
    def __init__(self):
        self.chart_themes = {
            'dark': {
                'bg_color': '#1e1e1e',
                'grid_color': '#404040',
                'text_color': '#ffffff',
                'bullish_color': '#00ff88',
                'bearish_color': '#ff4444'
            },
            'light': {
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'bullish_color': '#26a69a',
                'bearish_color': '#ef5350'
            }
        }
        
        self.timeframes = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1hr': '1H',
            '4hr': '4H',
            'daily': '1D',
            'weekly': '1W',
            'monthly': '1M'
        }
    
    def create_candlestick_chart(self, 
                               data: pd.DataFrame, 
                               symbol: str,
                               timeframe: str = 'daily',
                               theme: str = 'dark',
                               show_volume: bool = True,
                               indicators: List[str] = None,
                               support_resistance: List[float] = None) -> go.Figure:
        """
        Create an advanced candlestick chart with technical indicators
        """
        try:
            # Prepare data
            if timeframe != 'daily' and timeframe in self.timeframes:
                # Resample data for different timeframes
                data = self._resample_data(data, self.timeframes[timeframe])
            
            # Create subplots
            rows = 2 if show_volume else 1
            row_heights = [0.7, 0.3] if show_volume else [1.0]
            
            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=row_heights,
                subplot_titles=[f'{symbol} - {timeframe.upper()}', 'Volume'] if show_volume else [f'{symbol} - {timeframe.upper()}']
            )
            
            # Get theme colors
            colors = self.chart_themes[theme]
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=colors['bullish_color'],
                    decreasing_line_color=colors['bearish_color'],
                    increasing_fillcolor=colors['bullish_color'],
                    decreasing_fillcolor=colors['bearish_color']
                ),
                row=1, col=1
            )
            
            # Add technical indicators
            if indicators:
                self._add_technical_indicators(fig, data, indicators, colors)
            
            # Add support and resistance levels
            if support_resistance:
                self._add_support_resistance(fig, data, support_resistance, colors)
            
            # Add volume chart
            if show_volume and 'Volume' in data.columns:
                volume_colors = [
                    colors['bullish_color'] if close >= open else colors['bearish_color']
                    for close, open in zip(data['Close'], data['Open'])
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color=volume_colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Price Chart - {timeframe.upper()}',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                showlegend=True,
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            # Update x-axis
            fig.update_xaxes(
                rangeslider_visible=False,
                showgrid=True,
                gridcolor=colors['grid_color'],
                gridwidth=1
            )
            
            # Update y-axis
            fig.update_yaxes(
                showgrid=True,
                gridcolor=colors['grid_color'],
                gridwidth=1
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating candlestick chart: {e}")
            return go.Figure()
    
    def create_line_chart(self, 
                         data: pd.DataFrame, 
                         symbol: str,
                         price_type: str = 'Close',
                         theme: str = 'dark',
                         comparison_symbols: List[str] = None) -> go.Figure:
        """
        Create a line chart for price comparison
        """
        try:
            colors = self.chart_themes[theme]
            
            fig = go.Figure()
            
            # Add main symbol
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[price_type],
                    mode='lines',
                    name=symbol,
                    line=dict(color=colors['bullish_color'], width=2)
                )
            )
            
            # Add comparison symbols if provided
            if comparison_symbols:
                color_palette = px.colors.qualitative.Set1
                for i, comp_symbol in enumerate(comparison_symbols):
                    if comp_symbol in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data[comp_symbol],
                                mode='lines',
                                name=comp_symbol,
                                line=dict(color=color_palette[i % len(color_palette)], width=2)
                            )
                        )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Price Trend',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                showlegend=True,
                height=400,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating line chart: {e}")
            return go.Figure()
    
    def create_volume_profile(self, 
                            data: pd.DataFrame, 
                            symbol: str,
                            theme: str = 'dark') -> go.Figure:
        """
        Create a volume profile chart
        """
        try:
            colors = self.chart_themes[theme]
            
            # Calculate volume profile
            price_levels = np.linspace(data['Low'].min(), data['High'].max(), 50)
            volume_profile = []
            
            for i in range(len(price_levels) - 1):
                low_level = price_levels[i]
                high_level = price_levels[i + 1]
                
                # Find volume traded in this price range
                mask = (data['Low'] <= high_level) & (data['High'] >= low_level)
                volume_in_range = data[mask]['Volume'].sum()
                
                volume_profile.append({
                    'price': (low_level + high_level) / 2,
                    'volume': volume_in_range
                })
            
            profile_df = pd.DataFrame(volume_profile)
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=profile_df['volume'],
                    y=profile_df['price'],
                    orientation='h',
                    name='Volume Profile',
                    marker_color=colors['bullish_color'],
                    opacity=0.7
                )
            )
            
            # Add current price line
            current_price = data['Close'].iloc[-1]
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color=colors['bearish_color'],
                annotation_text=f"Current: ₹{current_price:.2f}"
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Volume Profile',
                xaxis_title='Volume',
                yaxis_title='Price (₹)',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=500,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volume profile: {e}")
            return go.Figure()
    
    def create_heatmap(self, 
                      data: Dict[str, pd.DataFrame], 
                      metric: str = 'returns',
                      theme: str = 'dark') -> go.Figure:
        """
        Create a correlation heatmap for multiple symbols
        """
        try:
            colors = self.chart_themes[theme]
            
            # Prepare data for correlation
            returns_data = {}
            
            for symbol, df in data.items():
                if metric == 'returns':
                    returns_data[symbol] = df['Close'].pct_change().dropna()
                elif metric == 'volume':
                    returns_data[symbol] = df['Volume']
                else:
                    returns_data[symbol] = df[metric]
            
            # Create correlation matrix
            corr_df = pd.DataFrame(returns_data).corr()
            
            # Create heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale='RdYlBu_r',
                    zmid=0,
                    text=corr_df.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f'Correlation Heatmap - {metric.title()}',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=500,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
            return go.Figure()
    
    def _resample_data(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframes
        """
        try:
            resampled = data.resample(freq).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            return data
    
    def _add_technical_indicators(self, fig: go.Figure, data: pd.DataFrame, indicators: List[str], colors: Dict):
        """
        Add technical indicators to the chart
        """
        try:
            for indicator in indicators:
                if indicator == 'SMA_20':
                    sma_20 = data['Close'].rolling(window=20).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=sma_20,
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='orange', width=1)
                        ),
                        row=1, col=1
                    )
                
                elif indicator == 'SMA_50':
                    sma_50 = data['Close'].rolling(window=50).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=sma_50,
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='blue', width=1)
                        ),
                        row=1, col=1
                    )
                
                elif indicator == 'EMA_20':
                    ema_20 = data['Close'].ewm(span=20).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=ema_20,
                            mode='lines',
                            name='EMA 20',
                            line=dict(color='purple', width=1)
                        ),
                        row=1, col=1
                    )
                
                elif indicator == 'BOLLINGER':
                    # Bollinger Bands
                    sma_20 = data['Close'].rolling(window=20).mean()
                    std_20 = data['Close'].rolling(window=20).std()
                    upper_band = sma_20 + (std_20 * 2)
                    lower_band = sma_20 - (std_20 * 2)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=upper_band,
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill=None
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=lower_band,
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(128,128,128,0.1)'
                        ),
                        row=1, col=1
                    )
                
                elif indicator == 'VWAP':
                    # Volume Weighted Average Price
                    if 'Volume' in data.columns:
                        vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=vwap,
                                mode='lines',
                                name='VWAP',
                                line=dict(color='yellow', width=2)
                            ),
                            row=1, col=1
                        )
                        
        except Exception as e:
            pass  # Continue without indicators if there's an error
    
    def _add_support_resistance(self, fig: go.Figure, data: pd.DataFrame, levels: List[float], colors: Dict):
        """
        Add support and resistance levels to the chart
        """
        try:
            for level in levels:
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color=colors['text_color'],
                    opacity=0.7,
                    annotation_text=f"₹{level:.2f}",
                    annotation_position="right"
                )
                
        except Exception as e:
            pass  # Continue without levels if there's an error
    
    def create_multi_timeframe_view(self, 
                                  data: pd.DataFrame, 
                                  symbol: str,
                                  theme: str = 'dark') -> go.Figure:
        """
        Create a multi-timeframe view with subplots
        """
        try:
            colors = self.chart_themes[theme]
            
            # Create subplots for different timeframes
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Daily', '4 Hour', '1 Hour', '15 Min'],
                shared_xaxes=False,
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            
            timeframes = ['1D', '4H', '1H', '15T']
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for tf, pos in zip(timeframes, positions):
                # Resample data
                tf_data = self._resample_data(data, tf)
                
                if len(tf_data) > 0:
                    # Add candlestick for each timeframe
                    fig.add_trace(
                        go.Candlestick(
                            x=tf_data.index,
                            open=tf_data['Open'],
                            high=tf_data['High'],
                            low=tf_data['Low'],
                            close=tf_data['Close'],
                            name=tf,
                            increasing_line_color=colors['bullish_color'],
                            decreasing_line_color=colors['bearish_color'],
                            showlegend=False
                        ),
                        row=pos[0], col=pos[1]
                    )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Multi-Timeframe Analysis',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=600,
                showlegend=False,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            # Update all xaxes
            fig.update_xaxes(rangeslider_visible=False)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating multi-timeframe view: {e}")
            return go.Figure()

# Streamlit interface functions
def render_price_charts_panel():
    """
    Render the price charts panel in Streamlit
    """
    st.subheader("📈 Price Charts")
    
    # Initialize chart handler
    chart_handler = PriceCharts()
    
    # Chart configuration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ['Candlestick', 'Line', 'Volume Profile', 'Multi-Timeframe']
        )
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            ['1min', '5min', '15min', '30min', '1hr', '4hr', 'daily', 'weekly']
        )
    
    with col3:
        theme = st.selectbox("Theme", ['dark', 'light'])
    
    with col4:
        show_volume = st.checkbox("Show Volume", value=True)
    
    # Technical indicators selection
    indicators = st.multiselect(
        "Technical Indicators",
        ['SMA_20', 'SMA_50', 'EMA_20', 'BOLLINGER', 'VWAP'],
        default=['SMA_20', 'SMA_50']
    )
    
    return {
        'chart_type': chart_type,
        'timeframe': timeframe,
        'theme': theme,
        'show_volume': show_volume,
        'indicators': indicators,
        'chart_handler': chart_handler
    }

def display_price_chart(data: pd.DataFrame, symbol: str, config: Dict):
    """
    Display the selected price chart
    """
    try:
        chart_handler = config['chart_handler']
        
        if config['chart_type'] == 'Candlestick':
            fig = chart_handler.create_candlestick_chart(
                data=data,
                symbol=symbol,
                timeframe=config['timeframe'],
                theme=config['theme'],
                show_volume=config['show_volume'],
                indicators=config['indicators']
            )
        
        elif config['chart_type'] == 'Line':
            fig = chart_handler.create_line_chart(
                data=data,
                symbol=symbol,
                theme=config['theme']
            )
        
        elif config['chart_type'] == 'Volume Profile':
            fig = chart_handler.create_volume_profile(
                data=data,
                symbol=symbol,
                theme=config['theme']
            )
        
        elif config['chart_type'] == 'Multi-Timeframe':
            fig = chart_handler.create_multi_timeframe_view(
                data=data,
                symbol=symbol,
                theme=config['theme']
            )
        
        else:
            fig = go.Figure()
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying chart: {e}")

# Example usage
if __name__ == "__main__":
    # This would typically be called from the main dashboard
    st.title("Price Charts Demo")
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    price = 1000
    prices = []
    volumes = []
    
    for _ in range(len(dates)):
        price += np.random.normal(0, 20)
        price = max(price, 100)  # Prevent negative prices
        prices.append(price)
        volumes.append(np.random.randint(100000, 1000000))
    
    sample_data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # Render charts
    config = render_price_charts_panel()
    display_price_chart(sample_data, 'SAMPLE_STOCK', config)