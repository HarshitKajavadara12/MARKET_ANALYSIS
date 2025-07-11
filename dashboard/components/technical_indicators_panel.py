import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')
from scipy import signal
import talib

class TechnicalIndicatorsPanel:
    def __init__(self):
        self.chart_themes = {
            'dark': {
                'bg_color': '#1e1e1e',
                'grid_color': '#404040',
                'text_color': '#ffffff',
                'bullish_color': '#00ff88',
                'bearish_color': '#ff4444',
                'neutral_color': '#ffa500',
                'signal_color': '#00bfff'
            },
            'light': {
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'bullish_color': '#26a69a',
                'bearish_color': '#ef5350',
                'neutral_color': '#ff9800',
                'signal_color': '#2196f3'
            }
        }
        
        self.indicator_categories = {
            'trend': ['SMA', 'EMA', 'WMA', 'MACD', 'ADX', 'Parabolic SAR', 'Ichimoku'],
            'momentum': ['RSI', 'Stochastic', 'Williams %R', 'CCI', 'ROC', 'MFI'],
            'volatility': ['Bollinger Bands', 'ATR', 'Keltner Channels', 'Donchian Channels'],
            'volume': ['OBV', 'VWAP', 'A/D Line', 'Chaikin MF', 'Volume Profile']
        }
        
        self.signal_levels = {
            'RSI': {'oversold': 30, 'overbought': 70},
            'Stochastic': {'oversold': 20, 'overbought': 80},
            'Williams %R': {'oversold': -80, 'overbought': -20},
            'CCI': {'oversold': -100, 'overbought': 100}
        }
    
    def calculate_sma(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    def calculate_ema(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    def calculate_wma(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Weighted Moving Average"""
        weights = np.arange(1, window + 1)
        return data.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = self.calculate_ema(data, fast)
            ema_slow = self.calculate_ema(data, slow)
            macd_line = ema_fast - ema_slow
            signal_line = self.calculate_ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except:
            return {'macd': pd.Series(index=data.index, dtype=float),
                   'signal': pd.Series(index=data.index, dtype=float),
                   'histogram': pd.Series(index=data.index, dtype=float)}
    
    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = self.calculate_sma(data, window)
            std = data.rolling(window=window).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band,
                'bandwidth': (upper_band - lower_band) / sma * 100,
                'percent_b': (data - lower_band) / (upper_band - lower_band)
            }
        except:
            return {k: pd.Series(index=data.index, dtype=float) for k in ['upper', 'middle', 'lower', 'bandwidth', 'percent_b']}
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_window).mean()
            
            return {
                'k': k_percent,
                'd': d_percent
            }
        except:
            return {'k': pd.Series(index=close.index, dtype=float),
                   'd': pd.Series(index=close.index, dtype=float)}
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = pd.Series(true_range).rolling(window=window).mean()
            atr.index = close.index
            
            return atr
        except:
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index"""
        try:
            # Calculate directional movement
            high_diff = high.diff()
            low_diff = low.diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)
            
            # Calculate True Range
            atr = self.calculate_atr(high, low, close, window)
            
            # Calculate Directional Indicators
            plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
            
            # Calculate ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=window).mean()
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
        except:
            return {k: pd.Series(index=close.index, dtype=float) for k in ['adx', 'plus_di', 'minus_di']}
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r
        except:
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=window).mean()
            mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            return cci
        except:
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except:
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            return vwap
        except:
            return pd.Series(index=close.index, dtype=float)
    
    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        try:
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=window).sum()
            negative_mf = negative_flow.rolling(window=window).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            return mfi
        except:
            return pd.Series(index=close.index, dtype=float)
    
    def create_indicators_dashboard(self, 
                                  data: pd.DataFrame, 
                                  symbol: str,
                                  selected_indicators: List[str],
                                  theme: str = 'dark') -> go.Figure:
        """Create a comprehensive technical indicators dashboard"""
        try:
            colors = self.chart_themes[theme]
            
            # Determine number of subplots needed
            price_indicators = ['SMA', 'EMA', 'WMA', 'Bollinger Bands', 'VWAP']
            oscillator_indicators = ['RSI', 'Stochastic', 'Williams %R', 'CCI', 'MFI']
            volume_indicators = ['OBV']
            trend_indicators = ['MACD', 'ADX']
            
            subplot_count = 1  # Price chart
            if any(ind in selected_indicators for ind in oscillator_indicators):
                subplot_count += 1
            if any(ind in selected_indicators for ind in volume_indicators):
                subplot_count += 1
            if any(ind in selected_indicators for ind in trend_indicators):
                subplot_count += 1
            
            # Create subplots
            subplot_titles = ['Price Chart']
            if any(ind in selected_indicators for ind in oscillator_indicators):
                subplot_titles.append('Oscillators')
            if any(ind in selected_indicators for ind in volume_indicators):
                subplot_titles.append('Volume')
            if any(ind in selected_indicators for ind in trend_indicators):
                subplot_titles.append('Trend Indicators')
            
            fig = make_subplots(
                rows=subplot_count,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=subplot_titles,
                row_heights=[0.5] + [0.5/(subplot_count-1)]*(subplot_count-1) if subplot_count > 1 else [1.0]
            )
            
            # Add price chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color=colors['bullish_color'],
                    decreasing_line_color=colors['bearish_color']
                ),
                row=1, col=1
            )
            
            current_row = 1
            
            # Add price-based indicators
            for indicator in selected_indicators:
                if indicator in price_indicators:
                    self._add_price_indicator(fig, data, indicator, colors, current_row)
            
            # Add oscillator indicators
            if any(ind in selected_indicators for ind in oscillator_indicators):
                current_row += 1
                for indicator in selected_indicators:
                    if indicator in oscillator_indicators:
                        self._add_oscillator_indicator(fig, data, indicator, colors, current_row)
            
            # Add volume indicators
            if any(ind in selected_indicators for ind in volume_indicators):
                current_row += 1
                for indicator in selected_indicators:
                    if indicator in volume_indicators:
                        self._add_volume_indicator(fig, data, indicator, colors, current_row)
            
            # Add trend indicators
            if any(ind in selected_indicators for ind in trend_indicators):
                current_row += 1
                for indicator in selected_indicators:
                    if indicator in trend_indicators:
                        self._add_trend_indicator(fig, data, indicator, colors, current_row)
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Technical Analysis Dashboard',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                showlegend=True,
                height=200 * subplot_count + 100,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            # Update x-axes
            fig.update_xaxes(rangeslider_visible=False)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating indicators dashboard: {e}")
            return go.Figure()
    
    def _add_price_indicator(self, fig: go.Figure, data: pd.DataFrame, indicator: str, colors: Dict, row: int):
        """Add price-based indicators to the chart"""
        try:
            if indicator == 'SMA':
                for period in [20, 50]:
                    sma = self.calculate_sma(data['Close'], period)
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=sma,
                            mode='lines',
                            name=f'SMA {period}',
                            line=dict(width=1)
                        ),
                        row=row, col=1
                    )
            
            elif indicator == 'EMA':
                for period in [12, 26]:
                    ema = self.calculate_ema(data['Close'], period)
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=ema,
                            mode='lines',
                            name=f'EMA {period}',
                            line=dict(width=1)
                        ),
                        row=row, col=1
                    )
            
            elif indicator == 'Bollinger Bands':
                bb = self.calculate_bollinger_bands(data['Close'])
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=bb['upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill=None
                    ),
                    row=row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=bb['lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)'
                    ),
                    row=row, col=1
                )
            
            elif indicator == 'VWAP' and 'Volume' in data.columns:
                vwap = self.calculate_vwap(data['High'], data['Low'], data['Close'], data['Volume'])
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=vwap,
                        mode='lines',
                        name='VWAP',
                        line=dict(color=colors['signal_color'], width=2)
                    ),
                    row=row, col=1
                )
                
        except Exception as e:
            pass
    
    def _add_oscillator_indicator(self, fig: go.Figure, data: pd.DataFrame, indicator: str, colors: Dict, row: int):
        """Add oscillator indicators to the chart"""
        try:
            if indicator == 'RSI':
                rsi = self.calculate_rsi(data['Close'])
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=rsi,
                        mode='lines',
                        name='RSI',
                        line=dict(color=colors['signal_color'], width=2)
                    ),
                    row=row, col=1
                )
                
                # Add overbought/oversold levels
                fig.add_hline(y=70, line_dash="dash", line_color=colors['bearish_color'], row=row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color=colors['bullish_color'], row=row, col=1)
            
            elif indicator == 'Stochastic':
                stoch = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=stoch['k'],
                        mode='lines',
                        name='Stoch %K',
                        line=dict(color=colors['signal_color'], width=2)
                    ),
                    row=row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=stoch['d'],
                        mode='lines',
                        name='Stoch %D',
                        line=dict(color=colors['neutral_color'], width=2)
                    ),
                    row=row, col=1
                )
                
                # Add overbought/oversold levels
                fig.add_hline(y=80, line_dash="dash", line_color=colors['bearish_color'], row=row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color=colors['bullish_color'], row=row, col=1)
            
            elif indicator == 'Williams %R':
                williams_r = self.calculate_williams_r(data['High'], data['Low'], data['Close'])
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=williams_r,
                        mode='lines',
                        name='Williams %R',
                        line=dict(color=colors['signal_color'], width=2)
                    ),
                    row=row, col=1
                )
                
                # Add overbought/oversold levels
                fig.add_hline(y=-20, line_dash="dash", line_color=colors['bearish_color'], row=row, col=1)
                fig.add_hline(y=-80, line_dash="dash", line_color=colors['bullish_color'], row=row, col=1)
            
            elif indicator == 'CCI':
                cci = self.calculate_cci(data['High'], data['Low'], data['Close'])
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=cci,
                        mode='lines',
                        name='CCI',
                        line=dict(color=colors['signal_color'], width=2)
                    ),
                    row=row, col=1
                )
                
                # Add overbought/oversold levels
                fig.add_hline(y=100, line_dash="dash", line_color=colors['bearish_color'], row=row, col=1)
                fig.add_hline(y=-100, line_dash="dash", line_color=colors['bullish_color'], row=row, col=1)
            
            elif indicator == 'MFI' and 'Volume' in data.columns:
                mfi = self.calculate_mfi(data['High'], data['Low'], data['Close'], data['Volume'])
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=mfi,
                        mode='lines',
                        name='MFI',
                        line=dict(color=colors['signal_color'], width=2)
                    ),
                    row=row, col=1
                )
                
                # Add overbought/oversold levels
                fig.add_hline(y=80, line_dash="dash", line_color=colors['bearish_color'], row=row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color=colors['bullish_color'], row=row, col=1)
                
        except Exception as e:
            pass
    
    def _add_volume_indicator(self, fig: go.Figure, data: pd.DataFrame, indicator: str, colors: Dict, row: int):
        """Add volume indicators to the chart"""
        try:
            if indicator == 'OBV' and 'Volume' in data.columns:
                obv = self.calculate_obv(data['Close'], data['Volume'])
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=obv,
                        mode='lines',
                        name='OBV',
                        line=dict(color=colors['signal_color'], width=2)
                    ),
                    row=row, col=1
                )
                
        except Exception as e:
            pass
    
    def _add_trend_indicator(self, fig: go.Figure, data: pd.DataFrame, indicator: str, colors: Dict, row: int):
        """Add trend indicators to the chart"""
        try:
            if indicator == 'MACD':
                macd = self.calculate_macd(data['Close'])
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=macd['macd'],
                        mode='lines',
                        name='MACD',
                        line=dict(color=colors['signal_color'], width=2)
                    ),
                    row=row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=macd['signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color=colors['neutral_color'], width=2)
                    ),
                    row=row, col=1
                )
                
                # Add histogram
                colors_hist = [colors['bullish_color'] if x >= 0 else colors['bearish_color'] for x in macd['histogram']]
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=macd['histogram'],
                        name='Histogram',
                        marker_color=colors_hist,
                        opacity=0.7
                    ),
                    row=row, col=1
                )
            
            elif indicator == 'ADX':
                adx = self.calculate_adx(data['High'], data['Low'], data['Close'])
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=adx['adx'],
                        mode='lines',
                        name='ADX',
                        line=dict(color=colors['signal_color'], width=2)
                    ),
                    row=row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=adx['plus_di'],
                        mode='lines',
                        name='+DI',
                        line=dict(color=colors['bullish_color'], width=1)
                    ),
                    row=row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=adx['minus_di'],
                        mode='lines',
                        name='-DI',
                        line=dict(color=colors['bearish_color'], width=1)
                    ),
                    row=row, col=1
                )
                
                # Add trend strength levels
                fig.add_hline(y=25, line_dash="dash", line_color=colors['text_color'], row=row, col=1)
                
        except Exception as e:
            pass
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Generate trading signals from technical indicators"""
        signals = {}
        
        try:
            # RSI signals
            rsi = self.calculate_rsi(data['Close'])
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            if current_rsi < 30:
                signals['RSI'] = {'signal': 'BUY', 'strength': 'Strong', 'value': current_rsi}
            elif current_rsi > 70:
                signals['RSI'] = {'signal': 'SELL', 'strength': 'Strong', 'value': current_rsi}
            else:
                signals['RSI'] = {'signal': 'NEUTRAL', 'strength': 'Weak', 'value': current_rsi}
            
            # MACD signals
            macd = self.calculate_macd(data['Close'])
            if not macd['macd'].empty and not macd['signal'].empty:
                current_macd = macd['macd'].iloc[-1]
                current_signal = macd['signal'].iloc[-1]
                
                if current_macd > current_signal:
                    signals['MACD'] = {'signal': 'BUY', 'strength': 'Medium', 'value': current_macd - current_signal}
                else:
                    signals['MACD'] = {'signal': 'SELL', 'strength': 'Medium', 'value': current_macd - current_signal}
            
            # Bollinger Bands signals
            bb = self.calculate_bollinger_bands(data['Close'])
            if not bb['percent_b'].empty:
                current_bb = bb['percent_b'].iloc[-1]
                
                if current_bb < 0:
                    signals['Bollinger'] = {'signal': 'BUY', 'strength': 'Strong', 'value': current_bb}
                elif current_bb > 1:
                    signals['Bollinger'] = {'signal': 'SELL', 'strength': 'Strong', 'value': current_bb}
                else:
                    signals['Bollinger'] = {'signal': 'NEUTRAL', 'strength': 'Weak', 'value': current_bb}
            
            # Stochastic signals
            stoch = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
            if not stoch['k'].empty:
                current_stoch = stoch['k'].iloc[-1]
                
                if current_stoch < 20:
                    signals['Stochastic'] = {'signal': 'BUY', 'strength': 'Strong', 'value': current_stoch}
                elif current_stoch > 80:
                    signals['Stochastic'] = {'signal': 'SELL', 'strength': 'Strong', 'value': current_stoch}
                else:
                    signals['Stochastic'] = {'signal': 'NEUTRAL', 'strength': 'Weak', 'value': current_stoch}
            
        except Exception as e:
            pass
        
        return signals

# Streamlit interface functions
def render_technical_indicators_panel():
    """
    Render the technical indicators panel in Streamlit
    """
    st.subheader("📊 Technical Indicators")
    
    # Initialize indicators handler
    indicators_handler = TechnicalIndicatorsPanel()
    
    # Indicator selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select Indicators:**")
        
        trend_indicators = st.multiselect(
            "Trend Indicators",
            indicators_handler.indicator_categories['trend'],
            default=['SMA', 'EMA', 'MACD']
        )
        
        momentum_indicators = st.multiselect(
            "Momentum Indicators",
            indicators_handler.indicator_categories['momentum'],
            default=['RSI', 'Stochastic']
        )
    
    with col2:
        volatility_indicators = st.multiselect(
            "Volatility Indicators",
            indicators_handler.indicator_categories['volatility'],
            default=['Bollinger Bands']
        )
        
        volume_indicators = st.multiselect(
            "Volume Indicators",
            indicators_handler.indicator_categories['volume'],
            default=['OBV']
        )
    
    # Combine all selected indicators
    selected_indicators = trend_indicators + momentum_indicators + volatility_indicators + volume_indicators
    
    # Theme selection
    theme = st.selectbox("Chart Theme", ['dark', 'light'], key='tech_theme')
    
    return {
        'selected_indicators': selected_indicators,
        'theme': theme,
        'indicators_handler': indicators_handler
    }

def display_technical_indicators(data: pd.DataFrame, symbol: str, config: Dict):
    """
    Display technical indicators dashboard and signals
    """
    try:
        indicators_handler = config['indicators_handler']
        
        if config['selected_indicators']:
            # Create and display indicators dashboard
            fig = indicators_handler.create_indicators_dashboard(
                data=data,
                symbol=symbol,
                selected_indicators=config['selected_indicators'],
                theme=config['theme']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate and display signals
            st.subheader("📈 Trading Signals")
            
            signals = indicators_handler.generate_signals(data)
            
            if signals:
                cols = st.columns(len(signals))
                
                for i, (indicator, signal_data) in enumerate(signals.items()):
                    with cols[i % len(cols)]:
                        signal_color = {
                            'BUY': '🟢',
                            'SELL': '🔴',
                            'NEUTRAL': '🟡'
                        }.get(signal_data['signal'], '⚪')
                        
                        st.metric(
                            f"{signal_color} {indicator}",
                            signal_data['signal'],
                            delta=f"{signal_data['strength']} ({signal_data['value']:.2f})"
                        )
            
            # Display indicator summary table
            st.subheader("📋 Indicator Summary")
            
            summary_data = []
            for indicator, signal_data in signals.items():
                summary_data.append({
                    'Indicator': indicator,
                    'Signal': signal_data['signal'],
                    'Strength': signal_data['strength'],
                    'Value': f"{signal_data['value']:.2f}"
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        else:
            st.info("Please select at least one technical indicator to display.")
        
    except Exception as e:
        st.error(f"Error displaying technical indicators: {e}")

# Example usage
if __name__ == "__main__":
    # This would typically be called from the main dashboard
    st.title("Technical Indicators Demo")
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    price = 1000
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Create trending price movement
        trend = 0.001 if i < len(dates)/2 else -0.001
        price_change = np.random.normal(trend, 0.02) * price
        price += price_change
        price = max(price, 100)  # Prevent negative prices
        prices.append(price)
        volumes.append(np.random.randint(100000, 1000000))
    
    sample_data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # Render technical indicators
    config = render_technical_indicators_panel()
    display_technical_indicators(sample_data, 'SAMPLE_STOCK', config)