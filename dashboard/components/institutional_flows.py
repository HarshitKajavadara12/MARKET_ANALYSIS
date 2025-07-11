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

class InstitutionalFlowTracker:
    def __init__(self):
        self.chart_themes = {
            'dark': {
                'bg_color': '#1e1e1e',
                'grid_color': '#404040',
                'text_color': '#ffffff',
                'fii_color': '#00a0fc',
                'dii_color': '#ff9500',
                'net_positive_color': '#00ff88',
                'net_negative_color': '#ff4444',
                'neutral_color': '#888888'
            },
            'light': {
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'fii_color': '#1976d2',
                'dii_color': '#ff9800',
                'net_positive_color': '#26a69a',
                'net_negative_color': '#ef5350',
                'neutral_color': '#9e9e9e'
            }
        }
        
        # Market segments
        self.market_segments = ['Equity', 'Debt', 'Hybrid', 'Derivatives']
        
        # Initialize sample data (will be replaced with real data in production)
        self.initialize_sample_data()
    
    def initialize_sample_data(self):
        """Initialize sample institutional flow data for demonstration"""
        # Generate dates for the past 90 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate sample FII data
        np.random.seed(42)  # For reproducibility
        
        # FII flows with trend and some volatility
        fii_trend = np.linspace(-500, 800, len(dates))  # Trend from negative to positive
        fii_noise = np.random.normal(0, 400, len(dates))  # Add some noise
        fii_flows = fii_trend + fii_noise
        
        # DII flows often move opposite to FII
        dii_trend = np.linspace(600, -300, len(dates))  # Opposite trend
        dii_noise = np.random.normal(0, 350, len(dates))
        dii_flows = dii_trend + dii_noise
        
        # Create DataFrame
        self.flow_data = pd.DataFrame({
            'Date': dates,
            'FII_Equity': fii_flows,
            'DII_Equity': dii_flows,
            'FII_Debt': fii_flows * 0.6 + np.random.normal(0, 200, len(dates)),
            'DII_Debt': dii_flows * 0.4 + np.random.normal(0, 150, len(dates)),
            'FII_Derivatives': fii_flows * 1.2 + np.random.normal(0, 500, len(dates)),
            'DII_Derivatives': dii_flows * 0.8 + np.random.normal(0, 400, len(dates))
        })
        
        # Calculate net flows
        self.flow_data['FII_Net'] = self.flow_data['FII_Equity'] + self.flow_data['FII_Debt'] + self.flow_data['FII_Derivatives']
        self.flow_data['DII_Net'] = self.flow_data['DII_Equity'] + self.flow_data['DII_Debt'] + self.flow_data['DII_Derivatives']
        self.flow_data['Total_Net'] = self.flow_data['FII_Net'] + self.flow_data['DII_Net']
        
        # Calculate cumulative flows
        self.flow_data['FII_Cumulative'] = self.flow_data['FII_Net'].cumsum()
        self.flow_data['DII_Cumulative'] = self.flow_data['DII_Net'].cumsum()
        self.flow_data['Total_Cumulative'] = self.flow_data['Total_Net'].cumsum()
    
    def update_flow_data(self, new_data: pd.DataFrame):
        """Update flow data with new information"""
        if not new_data.empty:
            self.flow_data = new_data
    
    def create_daily_flows_chart(self, days: int = 30, segment: str = 'Equity', theme: str = 'dark') -> go.Figure:
        """Create chart showing daily FII and DII flows"""
        try:
            # Filter data for selected segment and days
            df = self.flow_data.tail(days).copy()
            
            # Get theme colors
            colors = self.chart_themes[theme]
            
            # Create figure
            fig = go.Figure()
            
            # Add FII bars
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df[f'FII_{segment}'],
                name='FII',
                marker_color=colors['fii_color'],
                opacity=0.8
            ))
            
            # Add DII bars
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df[f'DII_{segment}'],
                name='DII',
                marker_color=colors['dii_color'],
                opacity=0.8
            ))
            
            # Add net flow line
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[f'FII_{segment}'] + df[f'DII_{segment}'],
                name='Net Flow',
                line=dict(color='white', width=2),
                mode='lines'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Daily {segment} Flows (₹ Crore)',
                xaxis_title='Date',
                yaxis_title='Flow (₹ Crore)',
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                xaxis=dict(gridcolor=colors['grid_color']),
                yaxis=dict(gridcolor=colors['grid_color']),
                barmode='group',
                hovermode='x unified',
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            return fig
        
        except Exception as e:
            print(f"Error creating daily flows chart: {e}")
            return None
    
    def create_cumulative_flows_chart(self, days: int = 90, theme: str = 'dark') -> go.Figure:
        """Create chart showing cumulative FII and DII flows"""
        try:
            # Filter data for selected days
            df = self.flow_data.tail(days).copy()
            
            # Reset cumulative values to start from 0 for the selected period
            first_fii = df['FII_Net'].iloc[0]
            first_dii = df['DII_Net'].iloc[0]
            
            df['FII_Cumulative_Period'] = (df['FII_Net'] - first_fii).cumsum()
            df['DII_Cumulative_Period'] = (df['DII_Net'] - first_dii).cumsum()
            df['Total_Cumulative_Period'] = df['FII_Cumulative_Period'] + df['DII_Cumulative_Period']
            
            # Get theme colors
            colors = self.chart_themes[theme]
            
            # Create figure
            fig = go.Figure()
            
            # Add FII line
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['FII_Cumulative_Period'],
                name='FII Cumulative',
                line=dict(color=colors['fii_color'], width=2),
                mode='lines'
            ))
            
            # Add DII line
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['DII_Cumulative_Period'],
                name='DII Cumulative',
                line=dict(color=colors['dii_color'], width=2),
                mode='lines'
            ))
            
            # Add total net line
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Total_Cumulative_Period'],
                name='Total Net',
                line=dict(color='white', width=2, dash='dash'),
                mode='lines'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Cumulative Flows - Past {days} Days (₹ Crore)',
                xaxis_title='Date',
                yaxis_title='Cumulative Flow (₹ Crore)',
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                xaxis=dict(gridcolor=colors['grid_color']),
                yaxis=dict(gridcolor=colors['grid_color']),
                hovermode='x unified',
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            return fig
        
        except Exception as e:
            print(f"Error creating cumulative flows chart: {e}")
            return None
    
    def create_flow_impact_chart(self, stock_data: Dict[str, pd.DataFrame], 
                               index_symbol: str = 'NIFTY 50',
                               days: int = 30, 
                               theme: str = 'dark') -> go.Figure:
        """Create chart showing correlation between institutional flows and market movement"""
        try:
            # Filter flow data for selected days
            flow_df = self.flow_data.tail(days).copy()
            
            # Get index data
            if index_symbol in stock_data:
                index_df = stock_data[index_symbol].tail(days).copy()
                
                # Ensure dates align
                flow_df['Date'] = pd.to_datetime(flow_df['Date']).dt.date
                index_df['Date'] = pd.to_datetime(index_df.index).dt.date
                
                # Merge data
                merged_df = pd.merge(flow_df, index_df, on='Date', how='inner')
                
                if not merged_df.empty:
                    # Get theme colors
                    colors = self.chart_themes[theme]
                    
                    # Create figure with secondary y-axis
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add FII bars
                    fig.add_trace(
                        go.Bar(
                            x=merged_df['Date'],
                            y=merged_df['FII_Net'],
                            name='FII Net',
                            marker_color=colors['fii_color'],
                            opacity=0.7
                        ),
                        secondary_y=False
                    )
                    
                    # Add DII bars
                    fig.add_trace(
                        go.Bar(
                            x=merged_df['Date'],
                            y=merged_df['DII_Net'],
                            name='DII Net',
                            marker_color=colors['dii_color'],
                            opacity=0.7
                        ),
                        secondary_y=False
                    )
                    
                    # Add index line
                    fig.add_trace(
                        go.Scatter(
                            x=merged_df['Date'],
                            y=merged_df['Close'],
                            name=index_symbol,
                            line=dict(color='white', width=2),
                            mode='lines'
                        ),
                        secondary_y=True
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f'Institutional Flows vs {index_symbol}',
                        plot_bgcolor=colors['bg_color'],
                        paper_bgcolor=colors['bg_color'],
                        font=dict(color=colors['text_color']),
                        xaxis=dict(gridcolor=colors['grid_color']),
                        yaxis=dict(gridcolor=colors['grid_color']),
                        barmode='group',
                        hovermode='x unified',
                        height=500,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    
                    # Set y-axes titles
                    fig.update_yaxes(title_text="Flow (₹ Crore)", secondary_y=False)
                    fig.update_yaxes(title_text=f"{index_symbol} Price", secondary_y=True)
                    
                    return fig
            
            return None
        
        except Exception as e:
            print(f"Error creating flow impact chart: {e}")
            return None
    
    def calculate_flow_metrics(self, days: int = 30) -> Dict:
        """Calculate key metrics from institutional flow data"""
        try:
            # Filter data for selected days
            df = self.flow_data.tail(days).copy()
            
            metrics = {
                'FII': {
                    'total_flow': df['FII_Net'].sum(),
                    'avg_daily_flow': df['FII_Net'].mean(),
                    'positive_days': (df['FII_Net'] > 0).sum(),
                    'negative_days': (df['FII_Net'] < 0).sum(),
                    'largest_inflow': df['FII_Net'].max(),
                    'largest_outflow': df['FII_Net'].min(),
                    'volatility': df['FII_Net'].std(),
                    'trend': 'Neutral'
                },
                'DII': {
                    'total_flow': df['DII_Net'].sum(),
                    'avg_daily_flow': df['DII_Net'].mean(),
                    'positive_days': (df['DII_Net'] > 0).sum(),
                    'negative_days': (df['DII_Net'] < 0).sum(),
                    'largest_inflow': df['DII_Net'].max(),
                    'largest_outflow': df['DII_Net'].min(),
                    'volatility': df['DII_Net'].std(),
                    'trend': 'Neutral'
                },
                'Net': {
                    'total_flow': df['Total_Net'].sum(),
                    'avg_daily_flow': df['Total_Net'].mean(),
                    'positive_days': (df['Total_Net'] > 0).sum(),
                    'negative_days': (df['Total_Net'] < 0).sum(),
                    'correlation': 0.0  # Will be calculated if index data is available
                }
            }
            
            # Determine trends
            # Split the period in half and compare
            half_point = len(df) // 2
            first_half_fii = df['FII_Net'].iloc[:half_point].mean()
            second_half_fii = df['FII_Net'].iloc[half_point:].mean()
            
            if second_half_fii > first_half_fii + 100:
                metrics['FII']['trend'] = 'Improving'
            elif second_half_fii < first_half_fii - 100:
                metrics['FII']['trend'] = 'Deteriorating'
            
            first_half_dii = df['DII_Net'].iloc[:half_point].mean()
            second_half_dii = df['DII_Net'].iloc[half_point:].mean()
            
            if second_half_dii > first_half_dii + 100:
                metrics['DII']['trend'] = 'Improving'
            elif second_half_dii < first_half_dii - 100:
                metrics['DII']['trend'] = 'Deteriorating'
            
            return metrics
        
        except Exception as e:
            print(f"Error calculating flow metrics: {e}")
            return {}
    
    def analyze_flow_patterns(self, days: int = 90) -> Dict:
        """Analyze patterns in institutional flows"""
        try:
            # Filter data for selected days
            df = self.flow_data.tail(days).copy()
            
            patterns = {
                'fii_dii_correlation': df['FII_Net'].corr(df['DII_Net']),
                'fii_consecutive_days': self._find_consecutive_days(df, 'FII_Net'),
                'dii_consecutive_days': self._find_consecutive_days(df, 'DII_Net'),
                'divergence_days': self._find_divergence_days(df),
                'consensus_days': self._find_consensus_days(df),
                'dominant_player': 'Balanced'
            }
            
            # Determine dominant player
            fii_abs_sum = df['FII_Net'].abs().sum()
            dii_abs_sum = df['DII_Net'].abs().sum()
            
            if fii_abs_sum > dii_abs_sum * 1.2:
                patterns['dominant_player'] = 'FII'
            elif dii_abs_sum > fii_abs_sum * 1.2:
                patterns['dominant_player'] = 'DII'
            
            return patterns
        
        except Exception as e:
            print(f"Error analyzing flow patterns: {e}")
            return {}
    
    def _find_consecutive_days(self, df: pd.DataFrame, column: str, threshold: int = 3) -> Dict:
        """Find consecutive days of inflows or outflows"""
        result = {'inflow': 0, 'outflow': 0}
        
        # Calculate consecutive days
        df['positive'] = df[column] > 0
        df['negative'] = df[column] < 0
        
        # Find consecutive positive days
        pos_streak = 0
        max_pos_streak = 0
        
        for pos in df['positive']:
            if pos:
                pos_streak += 1
                max_pos_streak = max(max_pos_streak, pos_streak)
            else:
                pos_streak = 0
        
        # Find consecutive negative days
        neg_streak = 0
        max_neg_streak = 0
        
        for neg in df['negative']:
            if neg:
                neg_streak += 1
                max_neg_streak = max(max_neg_streak, neg_streak)
            else:
                neg_streak = 0
        
        result['inflow'] = max_pos_streak
        result['outflow'] = max_neg_streak
        
        return result
    
    def _find_divergence_days(self, df: pd.DataFrame) -> int:
        """Find days where FII and DII flows moved in opposite directions"""
        return ((df['FII_Net'] > 0) & (df['DII_Net'] < 0) | (df['FII_Net'] < 0) & (df['DII_Net'] > 0)).sum()
    
    def _find_consensus_days(self, df: pd.DataFrame) -> int:
        """Find days where FII and DII flows moved in the same direction"""
        return ((df['FII_Net'] > 0) & (df['DII_Net'] > 0) | (df['FII_Net'] < 0) & (df['DII_Net'] < 0)).sum()

def render_institutional_flows_panel(stock_data: Dict[str, pd.DataFrame], theme: str = 'dark'):
    """Render institutional flows analysis panel"""
    st.header("💹 FII/DII Flow Analysis")
    
    # Initialize tracker
    tracker = InstitutionalFlowTracker()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Daily Flows", "Cumulative Flows", "Market Impact", "Flow Patterns"])
    
    with tab1:
        # Timeframe and segment selection
        col1, col2 = st.columns(2)
        with col1:
            days = st.selectbox(
                "Select Timeframe",
                options=[7, 15, 30, 60, 90],
                index=2,
                key="flow_days"
            )
        
        with col2:
            segment = st.selectbox(
                "Select Market Segment",
                options=tracker.market_segments,
                index=0,
                key="flow_segment"
            )
        
        # Display daily flows chart
        fig = tracker.create_daily_flows_chart(days, segment, theme)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        metrics = tracker.calculate_flow_metrics(days)
        if metrics:
            st.subheader("Flow Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="FII Net Flow (₹ Cr)",
                    value=f"{metrics['FII']['total_flow']:.2f}",
                    delta=f"{metrics['FII']['trend']}"
                )
                st.write(f"Avg Daily: ₹{metrics['FII']['avg_daily_flow']:.2f} Cr")
                st.write(f"Positive Days: {metrics['FII']['positive_days']}")
                st.write(f"Negative Days: {metrics['FII']['negative_days']}")
            
            with col2:
                st.metric(
                    label="DII Net Flow (₹ Cr)",
                    value=f"{metrics['DII']['total_flow']:.2f}",
                    delta=f"{metrics['DII']['trend']}"
                )
                st.write(f"Avg Daily: ₹{metrics['DII']['avg_daily_flow']:.2f} Cr")
                st.write(f"Positive Days: {metrics['DII']['positive_days']}")
                st.write(f"Negative Days: {metrics['DII']['negative_days']}")
            
            with col3:
                st.metric(
                    label="Total Net Flow (₹ Cr)",
                    value=f"{metrics['Net']['total_flow']:.2f}",
                    delta=None
                )
                st.write(f"Avg Daily: ₹{metrics['Net']['avg_daily_flow']:.2f} Cr")
                st.write(f"Positive Days: {metrics['Net']['positive_days']}")
                st.write(f"Negative Days: {metrics['Net']['negative_days']}")
    
    with tab2:
        # Timeframe selection for cumulative chart
        cumulative_days = st.selectbox(
            "Select Timeframe",
            options=[30, 60, 90, 180],
            index=2,
            key="cumulative_days"
        )
        
        # Display cumulative flows chart
        fig = tracker.create_cumulative_flows_chart(cumulative_days, theme)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.info(
            "Cumulative flows show the running total of institutional investments over time. "  
            "Divergence between FII and DII flows can indicate different market perspectives, "  
            "while convergence often signals strong market consensus."
        )
    
    with tab3:
        # Timeframe and index selection
        col1, col2 = st.columns(2)
        with col1:
            impact_days = st.selectbox(
                "Select Timeframe",
                options=[15, 30, 60, 90],
                index=1,
                key="impact_days"
            )
        
        with col2:
            index_symbol = st.selectbox(
                "Select Index",
                options=["NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY MIDCAP 100"],
                index=0,
                key="impact_index"
            )
        
        # Display impact chart
        fig = tracker.create_flow_impact_chart(stock_data, index_symbol, impact_days, theme)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Insufficient data to analyze impact on {index_symbol}.")
        
        # Add correlation analysis
        st.subheader("Flow-Market Correlation Analysis")
        st.write(
            "This analysis shows how institutional flows correlate with market movements. "  
            "A high positive correlation indicates that flows are driving market direction, "  
            "while a negative correlation suggests contrarian positioning."
        )
        
        # Sample correlation metrics (would be calculated from actual data)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("FII-Market Correlation", "0.68", "Strong")
            st.write("FIIs have been trend-following in recent sessions")
        
        with col2:
            st.metric("DII-Market Correlation", "-0.42", "Moderate Negative")
            st.write("DIIs have been counter-cyclical, buying on dips")
    
    with tab4:
        # Analyze flow patterns
        patterns = tracker.analyze_flow_patterns(90)
        
        if patterns:
            st.subheader("Institutional Flow Patterns")
            
            # Display key pattern metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("FII-DII Correlation", f"{patterns['fii_dii_correlation']:.2f}")
                
                if patterns['fii_dii_correlation'] < -0.5:
                    st.write("Strong negative correlation - FIIs and DIIs are taking opposite positions")
                elif patterns['fii_dii_correlation'] > 0.5:
                    st.write("Strong positive correlation - FIIs and DIIs are moving in tandem")
                else:
                    st.write("Weak correlation - FIIs and DIIs are acting independently")
            
            with col2:
                st.metric("Dominant Player", patterns['dominant_player'])
                
                if patterns['dominant_player'] == 'FII':
                    st.write("FIIs are driving market flows with larger transaction volumes")
                elif patterns['dominant_player'] == 'DII':
                    st.write("DIIs are dominating market flows with consistent participation")
                else:
                    st.write("Balanced participation between FIIs and DIIs")
            
            # Display streak information
            st.subheader("Consecutive Flow Streaks")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("FII Buying Streak", patterns['fii_consecutive_days']['inflow'])
            
            with col2:
                st.metric("FII Selling Streak", patterns['fii_consecutive_days']['outflow'])
            
            with col3:
                st.metric("DII Buying Streak", patterns['dii_consecutive_days']['inflow'])
            
            with col4:
                st.metric("DII Selling Streak", patterns['dii_consecutive_days']['outflow'])
            
            # Display divergence/consensus information
            st.subheader("Flow Agreement Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                consensus_pct = patterns['consensus_days'] / 90 * 100
                st.metric("Consensus Days", patterns['consensus_days'], f"{consensus_pct:.1f}%")
                st.write("Days when FIIs and DIIs moved in the same direction")
            
            with col2:
                divergence_pct = patterns['divergence_days'] / 90 * 100
                st.metric("Divergence Days", patterns['divergence_days'], f"{divergence_pct:.1f}%")
                st.write("Days when FIIs and DIIs moved in opposite directions")
            
            # Add interpretation
            st.info(
                "**Flow Pattern Interpretation:** " +
                ("FIIs and DIIs are showing significant disagreement, indicating uncertainty in market direction. " 
                 if patterns['divergence_days'] > patterns['consensus_days'] else 
                 "FIIs and DIIs are largely in agreement, suggesting strong conviction in market direction. ")
            )
        else:
            st.warning("Insufficient data to analyze flow patterns.")