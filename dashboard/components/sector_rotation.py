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

class SectorRotationTracker:
    def __init__(self):
        self.chart_themes = {
            'dark': {
                'bg_color': '#1e1e1e',
                'grid_color': '#404040',
                'text_color': '#ffffff',
                'outperform_color': '#00ff88',
                'underperform_color': '#ff4444',
                'neutral_color': '#ffa500'
            },
            'light': {
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'outperform_color': '#26a69a',
                'underperform_color': '#ef5350',
                'neutral_color': '#ff9800'
            }
        }
        
        # Indian market sectors
        self.sectors = {
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
            'Oil & Gas': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'GAIL'],
            'Consumer Goods': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
            'Automobile': ['MARUTI', 'M&M', 'TATAMOTORS', 'HEROMOTOCO', 'BAJAJ-AUTO'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON'],
            'Metals': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'COALINDIA'],
            'Telecom': ['BHARTIARTL', 'IDEA', 'TATACOMM'],
            'Power': ['NTPC', 'POWERGRID', 'ADANIPOWER', 'TATAPOWER'],
            'Financial Services': ['HDFC', 'BAJFINANCE', 'SBILIFE', 'HDFCLIFE', 'ICICIGI']
        }
        
        # Sector indices mapping
        self.sector_indices = {
            'Banking': 'NIFTY BANK',
            'IT': 'NIFTY IT',
            'Oil & Gas': 'NIFTY ENERGY',
            'Consumer Goods': 'NIFTY FMCG',
            'Automobile': 'NIFTY AUTO',
            'Pharma': 'NIFTY PHARMA',
            'Metals': 'NIFTY METAL',
            'Telecom': 'NIFTY MEDIA',
            'Power': 'NIFTY ENERGY',
            'Financial Services': 'NIFTY FINANCIAL SERVICES'
        }
        
        # Economic cycle phases
        self.economic_phases = {
            'Early Expansion': ['Financial Services', 'Consumer Goods', 'Automobile'],
            'Mid Expansion': ['IT', 'Telecom', 'Banking'],
            'Late Expansion': ['Oil & Gas', 'Metals', 'Power'],
            'Early Contraction': ['Pharma', 'Consumer Goods', 'Power'],
            'Late Contraction': ['Financial Services', 'IT', 'Banking']
        }
    
    def calculate_sector_performance(self, data: Dict[str, pd.DataFrame], 
                                   timeframe: str = '1M') -> pd.DataFrame:
        """Calculate performance metrics for each sector"""
        try:
            # Convert timeframe to days
            days_map = {'1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365, 'YTD': (datetime.now() - datetime(datetime.now().year, 1, 1)).days}
            days = days_map.get(timeframe, 30)
            
            # Calculate sector performance
            sector_performance = []
            
            for sector, symbols in self.sectors.items():
                # Filter symbols that exist in data
                available_symbols = [s for s in symbols if s in data]
                
                if not available_symbols:
                    continue
                
                # Calculate sector metrics
                returns = []
                volatility = []
                volume_change = []
                strength = []
                
                for symbol in available_symbols:
                    df = data[symbol]
                    if len(df) < 2:
                        continue
                    
                    # Calculate return
                    start_price = df['Close'].iloc[-min(len(df), days)]
                    end_price = df['Close'].iloc[-1]
                    ret = (end_price / start_price - 1) * 100
                    returns.append(ret)
                    
                    # Calculate volatility
                    vol = df['Close'].pct_change().iloc[-min(len(df), days):].std() * np.sqrt(252) * 100
                    volatility.append(vol)
                    
                    # Calculate volume change
                    start_volume = df['Volume'].iloc[-min(len(df), days):(-min(len(df), days)//2)].mean()
                    end_volume = df['Volume'].iloc[(-min(len(df), days)//2):].mean()
                    vol_change = (end_volume / start_volume - 1) * 100 if start_volume > 0 else 0
                    volume_change.append(vol_change)
                    
                    # Calculate strength (momentum)
                    strength_val = ret / (vol if vol > 0 else 1)  # Return to risk ratio
                    strength.append(strength_val)
                
                if returns:
                    sector_performance.append({
                        'Sector': sector,
                        'Return': np.mean(returns),
                        'Volatility': np.mean(volatility),
                        'Volume_Change': np.mean(volume_change),
                        'Strength': np.mean(strength),
                        'Symbols_Count': len(available_symbols)
                    })
            
            # Convert to DataFrame
            df_performance = pd.DataFrame(sector_performance)
            
            # Calculate relative performance vs market
            if not df_performance.empty:
                market_return = df_performance['Return'].mean()
                df_performance['Relative_Return'] = df_performance['Return'] - market_return
                
                # Add ranking
                df_performance['Rank'] = df_performance['Return'].rank(ascending=False)
                
                # Add momentum status
                df_performance['Status'] = 'Neutral'
                df_performance.loc[df_performance['Relative_Return'] > 3, 'Status'] = 'Outperforming'
                df_performance.loc[df_performance['Relative_Return'] < -3, 'Status'] = 'Underperforming'
            
            return df_performance
        
        except Exception as e:
            print(f"Error calculating sector performance: {e}")
            return pd.DataFrame()
    
    def create_sector_rotation_chart(self, performance_data: pd.DataFrame, theme: str = 'dark') -> go.Figure:
        """Create sector rotation chart showing relative performance"""
        try:
            if performance_data.empty:
                return None
            
            # Sort by return
            df = performance_data.sort_values('Return', ascending=False)
            
            # Get theme colors
            colors = self.chart_themes[theme]
            
            # Create figure
            fig = go.Figure()
            
            # Add sector performance bars
            fig.add_trace(go.Bar(
                x=df['Sector'],
                y=df['Return'],
                name='Return (%)',
                marker_color=[colors['outperform_color'] if x > 0 else colors['underperform_color'] for x in df['Return']],
                text=df['Return'].round(2).astype(str) + '%',
                textposition='auto'
            ))
            
            # Add market average line
            fig.add_shape(
                type='line',
                x0=-0.5,
                x1=len(df)-0.5,
                y0=df['Return'].mean(),
                y1=df['Return'].mean(),
                line=dict(color=colors['neutral_color'], width=2, dash='dash'),
                name='Market Average'
            )
            
            # Update layout
            fig.update_layout(
                title='Sector Rotation Analysis',
                xaxis_title='Sectors',
                yaxis_title='Return (%)',
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                xaxis=dict(gridcolor=colors['grid_color']),
                yaxis=dict(gridcolor=colors['grid_color']),
                hovermode='x unified',
                height=500
            )
            
            return fig
        
        except Exception as e:
            print(f"Error creating sector rotation chart: {e}")
            return None
    
    def create_sector_heatmap(self, data: Dict[str, pd.DataFrame], 
                            periods: List[str] = ['1W', '1M', '3M', '6M', '1Y'],
                            theme: str = 'dark') -> go.Figure:
        """Create sector performance heatmap across different timeframes"""
        try:
            # Get theme colors
            colors = self.chart_themes[theme]
            
            # Calculate performance for each period
            period_data = {}
            for period in periods:
                period_data[period] = self.calculate_sector_performance(data, period)
            
            # Prepare heatmap data
            sectors = list(self.sectors.keys())
            heatmap_data = []
            
            for sector in sectors:
                row = [sector]
                for period in periods:
                    if period in period_data and not period_data[period].empty:
                        sector_df = period_data[period]
                        if sector in sector_df['Sector'].values:
                            row.append(sector_df[sector_df['Sector'] == sector]['Return'].values[0])
                        else:
                            row.append(np.nan)
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)
            
            # Convert to DataFrame
            heatmap_df = pd.DataFrame(heatmap_data, columns=['Sector'] + periods)
            heatmap_df = heatmap_df.set_index('Sector')
            
            # Create heatmap
            fig = px.imshow(
                heatmap_df,
                labels=dict(x="Timeframe", y="Sector", color="Return (%)"),
                x=periods,
                y=heatmap_df.index,
                color_continuous_scale=[
                    colors['underperform_color'],
                    colors['neutral_color'],
                    colors['outperform_color']
                ],
                aspect="auto"
            )
            
            # Update layout
            fig.update_layout(
                title='Sector Performance Heatmap',
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                height=500
            )
            
            # Add text annotations
            for i, sector in enumerate(heatmap_df.index):
                for j, period in enumerate(periods):
                    value = heatmap_df.loc[sector, period]
                    if not np.isnan(value):
                        fig.add_annotation(
                            x=period,
                            y=sector,
                            text=f"{value:.2f}%",
                            showarrow=False,
                            font=dict(
                                color='white' if abs(value) > 10 else 'black',
                                size=10
                            )
                        )
            
            return fig
        
        except Exception as e:
            print(f"Error creating sector heatmap: {e}")
            return None
    
    def identify_market_cycle(self, performance_data: pd.DataFrame) -> str:
        """Identify current market cycle based on sector performance"""
        try:
            if performance_data.empty:
                return "Unknown"
            
            # Get top 3 and bottom 3 sectors
            top_sectors = performance_data.sort_values('Return', ascending=False).head(3)['Sector'].tolist()
            bottom_sectors = performance_data.sort_values('Return', ascending=True).head(3)['Sector'].tolist()
            
            # Calculate match score for each economic phase
            phase_scores = {}
            for phase, sectors in self.economic_phases.items():
                # Score based on top performing sectors
                top_match = len([s for s in top_sectors if s in sectors])
                # Score based on bottom performing sectors
                bottom_match = len([s for s in bottom_sectors if s not in sectors])
                # Combined score
                phase_scores[phase] = top_match + bottom_match
            
            # Get phase with highest score
            current_phase = max(phase_scores.items(), key=lambda x: x[1])[0]
            
            return current_phase
        
        except Exception as e:
            print(f"Error identifying market cycle: {e}")
            return "Unknown"
    
    def analyze_sector_flows(self, data: Dict[str, pd.DataFrame], 
                           timeframes: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """Analyze capital flows between sectors over different timeframes"""
        try:
            # Calculate sector performance for each timeframe
            flow_data = []
            
            for days in timeframes:
                # Calculate performance
                performance = []
                for sector, symbols in self.sectors.items():
                    # Filter symbols that exist in data
                    available_symbols = [s for s in symbols if s in data]
                    
                    if not available_symbols:
                        continue
                    
                    # Calculate sector return
                    sector_return = 0
                    for symbol in available_symbols:
                        df = data[symbol]
                        if len(df) < days + 1:
                            continue
                        
                        start_price = df['Close'].iloc[-min(len(df), days+1)]
                        end_price = df['Close'].iloc[-1]
                        ret = (end_price / start_price - 1) * 100
                        sector_return += ret
                    
                    if available_symbols:
                        sector_return /= len(available_symbols)
                        performance.append({
                            'Sector': sector,
                            'Return': sector_return,
                            'Timeframe': f"{days}d"
                        })
                
                flow_data.extend(performance)
            
            # Convert to DataFrame
            df_flows = pd.DataFrame(flow_data)
            
            # Calculate flow direction
            if not df_flows.empty:
                # Pivot to get sectors as rows and timeframes as columns
                df_pivot = df_flows.pivot(index='Sector', columns='Timeframe', values='Return')
                
                # Calculate flow direction (improving or deteriorating)
                timeframe_cols = [f"{t}d" for t in sorted(timeframes)]
                for i in range(len(timeframe_cols)-1):
                    short_term = timeframe_cols[i]
                    long_term = timeframe_cols[i+1]
                    col_name = f"Flow_{short_term}_vs_{long_term}"
                    df_pivot[col_name] = df_pivot[short_term] - df_pivot[long_term]
                
                # Add flow status
                for col in [c for c in df_pivot.columns if c.startswith('Flow_')]:
                    status_col = f"Status_{col.split('_')[1]}"
                    df_pivot[status_col] = 'Neutral'
                    df_pivot.loc[df_pivot[col] > 2, status_col] = 'Improving'
                    df_pivot.loc[df_pivot[col] < -2, status_col] = 'Deteriorating'
                
                return df_pivot.reset_index()
            
            return pd.DataFrame()
        
        except Exception as e:
            print(f"Error analyzing sector flows: {e}")
            return pd.DataFrame()

def render_sector_rotation_panel(data: Dict[str, pd.DataFrame], theme: str = 'dark'):
    """Render sector rotation analysis panel"""
    st.header("📊 Sector Rotation Analysis")
    
    # Initialize tracker
    tracker = SectorRotationTracker()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Current Performance", "Historical Trends", "Sector Flows", "Market Cycle"])
    
    with tab1:
        # Timeframe selection
        timeframe = st.selectbox(
            "Select Timeframe",
            options=["1W", "1M", "3M", "6M", "1Y", "YTD"],
            index=1,
            key="sector_timeframe"
        )
        
        # Calculate sector performance
        performance_data = tracker.calculate_sector_performance(data, timeframe)
        
        if not performance_data.empty:
            # Display sector rotation chart
            fig = tracker.create_sector_rotation_chart(performance_data, theme)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display performance table
            st.subheader("Sector Performance Details")
            display_df = performance_data[['Sector', 'Return', 'Volatility', 'Volume_Change', 'Strength', 'Status']].copy()
            display_df = display_df.sort_values('Return', ascending=False)
            
            # Format columns
            display_df['Return'] = display_df['Return'].round(2).astype(str) + '%'
            display_df['Volatility'] = display_df['Volatility'].round(2).astype(str) + '%'
            display_df['Volume_Change'] = display_df['Volume_Change'].round(2).astype(str) + '%'
            display_df['Strength'] = display_df['Strength'].round(2)
            
            # Apply conditional formatting
            def highlight_status(val):
                if val == 'Outperforming':
                    return 'background-color: #00ff8844'
                elif val == 'Underperforming':
                    return 'background-color: #ff444444'
                return ''
            
            styled_df = display_df.style.applymap(highlight_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("Insufficient data to calculate sector performance.")
    
    with tab2:
        # Create sector heatmap
        st.subheader("Sector Performance Across Timeframes")
        fig = tracker.create_sector_heatmap(data, periods=["1W", "1M", "3M", "6M", "1Y"], theme=theme)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data to create sector heatmap.")
    
    with tab3:
        # Analyze sector flows
        st.subheader("Capital Flow Analysis")
        flow_data = tracker.analyze_sector_flows(data, timeframes=[7, 30, 90])
        
        if not flow_data.empty:
            # Display flow table
            flow_display = flow_data[['Sector', '7d', '30d', '90d', 'Status_7d_vs_30d']].copy()
            flow_display.columns = ['Sector', '1 Week', '1 Month', '3 Months', 'Short-term Trend']
            
            # Format columns
            for col in ['1 Week', '1 Month', '3 Months']:
                flow_display[col] = flow_display[col].round(2).astype(str) + '%'
            
            # Apply conditional formatting
            def highlight_trend(val):
                if val == 'Improving':
                    return 'background-color: #00ff8844'
                elif val == 'Deteriorating':
                    return 'background-color: #ff444444'
                return ''
            
            styled_flow = flow_display.style.applymap(highlight_trend, subset=['Short-term Trend'])
            st.dataframe(styled_flow, use_container_width=True)
            
            # Create flow visualization
            st.subheader("Sector Flow Visualization")
            st.info("This chart shows which sectors are gaining momentum (improving) or losing momentum (deteriorating) in the short term compared to longer timeframes.")
            
            # Simple bar chart showing flow strength
            flow_strength = flow_data[['Sector', 'Flow_7d_vs_30d']].copy()
            flow_strength = flow_strength.sort_values('Flow_7d_vs_30d', ascending=False)
            
            fig = px.bar(
                flow_strength,
                x='Sector',
                y='Flow_7d_vs_30d',
                title='Sector Momentum (1W vs 1M)',
                color='Flow_7d_vs_30d',
                color_continuous_scale=['red', 'yellow', 'green'],
                labels={'Flow_7d_vs_30d': 'Momentum'}
            )
            
            # Update layout
            fig.update_layout(
                plot_bgcolor=tracker.chart_themes[theme]['bg_color'],
                paper_bgcolor=tracker.chart_themes[theme]['bg_color'],
                font=dict(color=tracker.chart_themes[theme]['text_color']),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data to analyze sector flows.")
    
    with tab4:
        # Identify market cycle
        st.subheader("Market Cycle Analysis")
        
        performance_data = tracker.calculate_sector_performance(data, "3M")
        if not performance_data.empty:
            current_phase = tracker.identify_market_cycle(performance_data)
            
            # Display current phase
            st.info(f"Current Market Cycle Phase: **{current_phase}**")
            
            # Display phase characteristics
            st.subheader("Phase Characteristics")
            phase_descriptions = {
                'Early Expansion': "Economy recovering from recession. Interest rates low. Consumer and financial sectors typically lead.",
                'Mid Expansion': "Economic growth accelerating. Technology and industrial sectors typically perform well.",
                'Late Expansion': "Economy at peak growth. Inflation concerns emerge. Materials, energy sectors typically outperform.",
                'Early Contraction': "Economic growth slowing. Defensive sectors like healthcare and utilities typically outperform.",
                'Late Contraction': "Economy in recession. Interest rates falling. Quality companies with strong balance sheets outperform."
            }
            
            st.write(phase_descriptions.get(current_phase, "Unknown phase"))
            
            # Display expected sector performance
            st.subheader("Expected Leading Sectors")
            if current_phase in tracker.economic_phases:
                leading_sectors = tracker.economic_phases[current_phase]
                for sector in leading_sectors:
                    st.write(f"- {sector}")
            else:
                st.write("Unknown phase - cannot determine leading sectors")
            
            # Display next phase prediction
            st.subheader("Next Phase Prediction")
            phases = list(tracker.economic_phases.keys())
            if current_phase in phases:
                current_idx = phases.index(current_phase)
                next_idx = (current_idx + 1) % len(phases)
                next_phase = phases[next_idx]
                st.write(f"Based on the current phase, the market may transition to **{next_phase}** next.")
                st.write(f"Sectors to watch: {', '.join(tracker.economic_phases[next_phase])}")
        else:
            st.warning("Insufficient data to identify market cycle.")