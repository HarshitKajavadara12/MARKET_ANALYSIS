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
from scipy import stats
from sklearn.preprocessing import StandardScaler

class VolatilityCharts:
    def __init__(self):
        self.chart_themes = {
            'dark': {
                'bg_color': '#1e1e1e',
                'grid_color': '#404040',
                'text_color': '#ffffff',
                'vol_color': '#ff6b6b',
                'realized_vol_color': '#4ecdc4',
                'implied_vol_color': '#45b7d1',
                'vix_color': '#f9ca24'
            },
            'light': {
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'vol_color': '#e74c3c',
                'realized_vol_color': '#1abc9c',
                'implied_vol_color': '#3498db',
                'vix_color': '#f39c12'
            }
        }
        
        self.volatility_regimes = {
            'low': {'threshold': 15, 'color': '#2ecc71', 'label': 'Low Volatility'},
            'normal': {'threshold': 25, 'color': '#f39c12', 'label': 'Normal Volatility'},
            'high': {'threshold': 40, 'color': '#e74c3c', 'label': 'High Volatility'},
            'extreme': {'threshold': float('inf'), 'color': '#8e44ad', 'label': 'Extreme Volatility'}
        }
    
    def calculate_realized_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate realized volatility using different methods
        """
        try:
            # Simple returns volatility
            returns = data['Close'].pct_change().dropna()
            realized_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
            
            return realized_vol
            
        except Exception as e:
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_parkinson_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate Parkinson volatility (uses high-low range)
        """
        try:
            # Parkinson volatility estimator
            hl_ratio = np.log(data['High'] / data['Low'])
            parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * hl_ratio**2)
            parkinson_vol = parkinson_vol.rolling(window=window).mean() * np.sqrt(252) * 100
            
            return parkinson_vol
            
        except Exception as e:
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_garman_klass_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate Garman-Klass volatility (more efficient estimator)
        """
        try:
            # Garman-Klass volatility estimator
            hl = np.log(data['High'] / data['Low'])
            co = np.log(data['Close'] / data['Open'])
            
            gk_vol = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
            gk_vol = np.sqrt(gk_vol.rolling(window=window).mean() * 252) * 100
            
            return gk_vol
            
        except Exception as e:
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_ewma_volatility(self, data: pd.DataFrame, lambda_param: float = 0.94) -> pd.Series:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) volatility
        """
        try:
            returns = data['Close'].pct_change().dropna()
            
            # EWMA volatility
            ewma_var = returns.ewm(alpha=1-lambda_param).var()
            ewma_vol = np.sqrt(ewma_var * 252) * 100
            
            return ewma_vol
            
        except Exception as e:
            return pd.Series(index=data.index, dtype=float)
    
    def create_volatility_surface(self, 
                                data: pd.DataFrame, 
                                symbol: str,
                                theme: str = 'dark') -> go.Figure:
        """
        Create a 3D volatility surface chart
        """
        try:
            colors = self.chart_themes[theme]
            
            # Calculate different volatility measures
            realized_vol = self.calculate_realized_volatility(data, 20)
            parkinson_vol = self.calculate_parkinson_volatility(data, 20)
            gk_vol = self.calculate_garman_klass_volatility(data, 20)
            
            # Create meshgrid for surface
            dates = data.index[-60:]  # Last 60 days
            vol_types = ['Realized', 'Parkinson', 'Garman-Klass']
            
            # Prepare data for surface
            vol_data = np.array([
                realized_vol.loc[dates].fillna(0).values,
                parkinson_vol.loc[dates].fillna(0).values,
                gk_vol.loc[dates].fillna(0).values
            ])
            
            # Create 3D surface
            fig = go.Figure()
            
            fig.add_trace(
                go.Surface(
                    z=vol_data,
                    x=list(range(len(dates))),
                    y=vol_types,
                    colorscale='Viridis',
                    name='Volatility Surface'
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Volatility Surface',
                scene=dict(
                    xaxis_title='Time',
                    yaxis_title='Volatility Type',
                    zaxis_title='Volatility (%)',
                    bgcolor=colors['bg_color']
                ),
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=600,
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility surface: {e}")
            return go.Figure()
    
    def create_volatility_comparison(self, 
                                   data: pd.DataFrame, 
                                   symbol: str,
                                   theme: str = 'dark') -> go.Figure:
        """
        Create a comparison chart of different volatility measures
        """
        try:
            colors = self.chart_themes[theme]
            
            # Calculate different volatility measures
            realized_vol = self.calculate_realized_volatility(data, 20)
            parkinson_vol = self.calculate_parkinson_volatility(data, 20)
            gk_vol = self.calculate_garman_klass_volatility(data, 20)
            ewma_vol = self.calculate_ewma_volatility(data)
            
            fig = go.Figure()
            
            # Add volatility traces
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=realized_vol,
                    mode='lines',
                    name='Realized Volatility',
                    line=dict(color=colors['realized_vol_color'], width=2)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=parkinson_vol,
                    mode='lines',
                    name='Parkinson Volatility',
                    line=dict(color=colors['vol_color'], width=2)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=gk_vol,
                    mode='lines',
                    name='Garman-Klass Volatility',
                    line=dict(color=colors['implied_vol_color'], width=2)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ewma_vol,
                    mode='lines',
                    name='EWMA Volatility',
                    line=dict(color=colors['vix_color'], width=2, dash='dash')
                )
            )
            
            # Add volatility regime zones
            self._add_volatility_regimes(fig, data.index)
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Volatility Comparison',
                xaxis_title='Date',
                yaxis_title='Volatility (%)',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                showlegend=True,
                height=500,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility comparison: {e}")
            return go.Figure()
    
    def create_volatility_cone(self, 
                             data: pd.DataFrame, 
                             symbol: str,
                             theme: str = 'dark') -> go.Figure:
        """
        Create a volatility cone chart showing percentiles
        """
        try:
            colors = self.chart_themes[theme]
            
            # Calculate rolling volatilities for different windows
            windows = [5, 10, 20, 30, 60, 90, 120, 252]
            percentiles = [5, 25, 50, 75, 95]
            
            cone_data = []
            
            for window in windows:
                vol_series = self.calculate_realized_volatility(data, window).dropna()
                
                if len(vol_series) > 0:
                    cone_point = {'window': window}
                    for pct in percentiles:
                        cone_point[f'p{pct}'] = np.percentile(vol_series, pct)
                    cone_data.append(cone_point)
            
            cone_df = pd.DataFrame(cone_data)
            
            if len(cone_df) == 0:
                return go.Figure()
            
            fig = go.Figure()
            
            # Add percentile lines
            colors_pct = ['#e74c3c', '#f39c12', '#2ecc71', '#f39c12', '#e74c3c']
            
            for i, pct in enumerate(percentiles):
                fig.add_trace(
                    go.Scatter(
                        x=cone_df['window'],
                        y=cone_df[f'p{pct}'],
                        mode='lines+markers',
                        name=f'{pct}th Percentile',
                        line=dict(color=colors_pct[i], width=2)
                    )
                )
            
            # Add current volatility
            current_vol = self.calculate_realized_volatility(data, 20).iloc[-1]
            if not np.isnan(current_vol):
                fig.add_hline(
                    y=current_vol,
                    line_dash="dash",
                    line_color=colors['text_color'],
                    annotation_text=f"Current: {current_vol:.1f}%"
                )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Volatility Cone',
                xaxis_title='Time Window (Days)',
                yaxis_title='Volatility (%)',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                showlegend=True,
                height=500,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility cone: {e}")
            return go.Figure()
    
    def create_volatility_smile(self, 
                              options_data: pd.DataFrame, 
                              symbol: str,
                              theme: str = 'dark') -> go.Figure:
        """
        Create a volatility smile chart for options
        """
        try:
            colors = self.chart_themes[theme]
            
            fig = go.Figure()
            
            # Group by expiry date
            if 'expiry' in options_data.columns and 'strike' in options_data.columns and 'implied_vol' in options_data.columns:
                expiries = options_data['expiry'].unique()
                
                color_palette = px.colors.qualitative.Set1
                
                for i, expiry in enumerate(expiries[:5]):  # Show max 5 expiries
                    expiry_data = options_data[options_data['expiry'] == expiry].sort_values('strike')
                    
                    fig.add_trace(
                        go.Scatter(
                            x=expiry_data['strike'],
                            y=expiry_data['implied_vol'] * 100,
                            mode='lines+markers',
                            name=f'Expiry: {expiry}',
                            line=dict(color=color_palette[i % len(color_palette)], width=2)
                        )
                    )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Volatility Smile',
                xaxis_title='Strike Price',
                yaxis_title='Implied Volatility (%)',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                showlegend=True,
                height=500,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility smile: {e}")
            return go.Figure()
    
    def create_volatility_term_structure(self, 
                                        options_data: pd.DataFrame, 
                                        symbol: str,
                                        theme: str = 'dark') -> go.Figure:
        """
        Create a volatility term structure chart
        """
        try:
            colors = self.chart_themes[theme]
            
            fig = go.Figure()
            
            # Calculate ATM volatility for each expiry
            if 'expiry' in options_data.columns and 'days_to_expiry' in options_data.columns and 'implied_vol' in options_data.columns:
                # Group by expiry and calculate ATM vol
                atm_vols = []
                
                for expiry in options_data['expiry'].unique():
                    expiry_data = options_data[options_data['expiry'] == expiry]
                    
                    # Find ATM option (closest to current price)
                    if len(expiry_data) > 0:
                        atm_vol = expiry_data['implied_vol'].median()  # Use median as proxy for ATM
                        days_to_expiry = expiry_data['days_to_expiry'].iloc[0]
                        
                        atm_vols.append({
                            'days_to_expiry': days_to_expiry,
                            'implied_vol': atm_vol * 100,
                            'expiry': expiry
                        })
                
                if atm_vols:
                    term_df = pd.DataFrame(atm_vols).sort_values('days_to_expiry')
                    
                    fig.add_trace(
                        go.Scatter(
                            x=term_df['days_to_expiry'],
                            y=term_df['implied_vol'],
                            mode='lines+markers',
                            name='ATM Implied Volatility',
                            line=dict(color=colors['implied_vol_color'], width=3),
                            marker=dict(size=8)
                        )
                    )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Volatility Term Structure',
                xaxis_title='Days to Expiry',
                yaxis_title='Implied Volatility (%)',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                showlegend=True,
                height=500,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility term structure: {e}")
            return go.Figure()
    
    def create_volatility_heatmap(self, 
                                data: Dict[str, pd.DataFrame], 
                                theme: str = 'dark') -> go.Figure:
        """
        Create a volatility heatmap for multiple symbols
        """
        try:
            colors = self.chart_themes[theme]
            
            # Calculate volatility for each symbol
            vol_data = {}
            
            for symbol, df in data.items():
                vol_series = self.calculate_realized_volatility(df, 20)
                vol_data[symbol] = vol_series.iloc[-30:]  # Last 30 days
            
            # Create DataFrame
            vol_df = pd.DataFrame(vol_data)
            
            # Create heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=vol_df.values,
                    x=vol_df.columns,
                    y=[d.strftime('%Y-%m-%d') for d in vol_df.index],
                    colorscale='Reds',
                    text=vol_df.round(1).values,
                    texttemplate="%{text}%",
                    textfont={"size": 10},
                    hoverongaps=False
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Volatility Heatmap (Last 30 Days)',
                xaxis_title='Symbols',
                yaxis_title='Date',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=600,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volatility heatmap: {e}")
            return go.Figure()
    
    def _add_volatility_regimes(self, fig: go.Figure, dates: pd.DatetimeIndex):
        """
        Add volatility regime zones to the chart
        """
        try:
            # Add horizontal lines for regime thresholds
            for regime, config in self.volatility_regimes.items():
                if config['threshold'] != float('inf'):
                    fig.add_hline(
                        y=config['threshold'],
                        line_dash="dot",
                        line_color=config['color'],
                        opacity=0.5,
                        annotation_text=config['label'],
                        annotation_position="right"
                    )
                    
        except Exception as e:
            pass  # Continue without regimes if there's an error
    
    def calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive volatility metrics
        """
        try:
            metrics = {}
            
            # Current volatility measures
            realized_vol = self.calculate_realized_volatility(data, 20)
            parkinson_vol = self.calculate_parkinson_volatility(data, 20)
            gk_vol = self.calculate_garman_klass_volatility(data, 20)
            ewma_vol = self.calculate_ewma_volatility(data)
            
            # Current values
            metrics['current_realized_vol'] = realized_vol.iloc[-1] if not realized_vol.empty else 0
            metrics['current_parkinson_vol'] = parkinson_vol.iloc[-1] if not parkinson_vol.empty else 0
            metrics['current_gk_vol'] = gk_vol.iloc[-1] if not gk_vol.empty else 0
            metrics['current_ewma_vol'] = ewma_vol.iloc[-1] if not ewma_vol.empty else 0
            
            # Volatility statistics
            if not realized_vol.empty:
                metrics['vol_mean'] = realized_vol.mean()
                metrics['vol_std'] = realized_vol.std()
                metrics['vol_min'] = realized_vol.min()
                metrics['vol_max'] = realized_vol.max()
                metrics['vol_percentile_25'] = realized_vol.quantile(0.25)
                metrics['vol_percentile_75'] = realized_vol.quantile(0.75)
            
            # Volatility regime
            current_vol = metrics.get('current_realized_vol', 0)
            for regime, config in self.volatility_regimes.items():
                if current_vol <= config['threshold']:
                    metrics['volatility_regime'] = regime
                    metrics['regime_color'] = config['color']
                    metrics['regime_label'] = config['label']
                    break
            
            # Volatility trend
            if len(realized_vol) >= 10:
                recent_vol = realized_vol.iloc[-5:].mean()
                previous_vol = realized_vol.iloc[-10:-5].mean()
                
                if recent_vol > previous_vol * 1.1:
                    metrics['volatility_trend'] = 'increasing'
                elif recent_vol < previous_vol * 0.9:
                    metrics['volatility_trend'] = 'decreasing'
                else:
                    metrics['volatility_trend'] = 'stable'
            
            return metrics
            
        except Exception as e:
            return {}

# Streamlit interface functions
def render_volatility_charts_panel():
    """
    Render the volatility charts panel in Streamlit
    """
    st.subheader("📊 Volatility Analysis")
    
    # Initialize volatility chart handler
    vol_handler = VolatilityCharts()
    
    # Chart configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart_type = st.selectbox(
            "Volatility Chart Type",
            ['Comparison', 'Cone', 'Surface', 'Smile', 'Term Structure', 'Heatmap']
        )
    
    with col2:
        theme = st.selectbox("Theme", ['dark', 'light'], key='vol_theme')
    
    with col3:
        vol_window = st.slider("Volatility Window", 5, 60, 20)
    
    return {
        'chart_type': chart_type,
        'theme': theme,
        'vol_window': vol_window,
        'vol_handler': vol_handler
    }

def display_volatility_chart(data: pd.DataFrame, symbol: str, config: Dict, options_data: pd.DataFrame = None):
    """
    Display the selected volatility chart
    """
    try:
        vol_handler = config['vol_handler']
        
        if config['chart_type'] == 'Comparison':
            fig = vol_handler.create_volatility_comparison(
                data=data,
                symbol=symbol,
                theme=config['theme']
            )
        
        elif config['chart_type'] == 'Cone':
            fig = vol_handler.create_volatility_cone(
                data=data,
                symbol=symbol,
                theme=config['theme']
            )
        
        elif config['chart_type'] == 'Surface':
            fig = vol_handler.create_volatility_surface(
                data=data,
                symbol=symbol,
                theme=config['theme']
            )
        
        elif config['chart_type'] == 'Smile' and options_data is not None:
            fig = vol_handler.create_volatility_smile(
                options_data=options_data,
                symbol=symbol,
                theme=config['theme']
            )
        
        elif config['chart_type'] == 'Term Structure' and options_data is not None:
            fig = vol_handler.create_volatility_term_structure(
                options_data=options_data,
                symbol=symbol,
                theme=config['theme']
            )
        
        else:
            fig = go.Figure()
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display volatility metrics
        metrics = vol_handler.calculate_volatility_metrics(data)
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Volatility",
                    f"{metrics.get('current_realized_vol', 0):.1f}%",
                    delta=f"{metrics.get('volatility_trend', 'stable')}"
                )
            
            with col2:
                st.metric(
                    "Volatility Regime",
                    metrics.get('regime_label', 'Unknown')
                )
            
            with col3:
                st.metric(
                    "Vol Range (Min-Max)",
                    f"{metrics.get('vol_min', 0):.1f}% - {metrics.get('vol_max', 0):.1f}%"
                )
            
            with col4:
                st.metric(
                    "Vol Percentile (25-75)",
                    f"{metrics.get('vol_percentile_25', 0):.1f}% - {metrics.get('vol_percentile_75', 0):.1f}%"
                )
        
    except Exception as e:
        st.error(f"Error displaying volatility chart: {e}")

# Example usage
if __name__ == "__main__":
    # This would typically be called from the main dashboard
    st.title("Volatility Charts Demo")
    
    # Generate sample data with varying volatility
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    price = 1000
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Create periods of different volatility
        if i < 100:
            vol = 0.01  # Low volatility
        elif i < 200:
            vol = 0.03  # High volatility
        else:
            vol = 0.015  # Normal volatility
        
        price_change = np.random.normal(0, vol) * price
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
    
    # Render volatility charts
    config = render_volatility_charts_panel()
    display_volatility_chart(sample_data, 'SAMPLE_STOCK', config)