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

class RiskManagementPanel:
    def __init__(self):
        self.chart_themes = {
            'dark': {
                'bg_color': '#1e1e1e',
                'grid_color': '#404040',
                'text_color': '#ffffff',
                'low_risk_color': '#00ff88',
                'medium_risk_color': '#ffa500',
                'high_risk_color': '#ff4444',
                'line_color': '#00a0fc'
            },
            'light': {
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'low_risk_color': '#26a69a',
                'medium_risk_color': '#ff9800',
                'high_risk_color': '#ef5350',
                'line_color': '#1976d2'
            }
        }
        
        # Risk metrics configuration
        self.risk_metrics = {
            'Value at Risk (VaR)': {
                'description': 'Maximum expected loss over a given time horizon at a specific confidence level',
                'confidence_levels': [0.90, 0.95, 0.99],
                'time_horizons': ['1 Day', '1 Week', '1 Month'],
                'methods': ['Historical', 'Parametric', 'Monte Carlo']
            },
            'Expected Shortfall': {
                'description': 'Average loss beyond VaR, also known as Conditional VaR (CVaR)',
                'confidence_levels': [0.90, 0.95, 0.99],
                'time_horizons': ['1 Day', '1 Week', '1 Month'],
                'methods': ['Historical', 'Parametric']
            },
            'Maximum Drawdown': {
                'description': 'Largest peak-to-trough decline in portfolio value',
                'lookback_periods': ['1 Month', '3 Months', '6 Months', '1 Year', 'Since Inception']
            },
            'Volatility': {
                'description': 'Standard deviation of returns, annualized',
                'calculation_methods': ['Simple', 'Exponentially Weighted', 'GARCH'],
                'lookback_periods': ['1 Month', '3 Months', '6 Months', '1 Year']
            },
            'Beta': {
                'description': 'Measure of systematic risk relative to the market',
                'benchmarks': ['NIFTY 50', 'NIFTY 500', 'SENSEX'],
                'lookback_periods': ['1 Month', '3 Months', '6 Months', '1 Year']
            },
            'Sharpe Ratio': {
                'description': 'Risk-adjusted return relative to risk-free rate',
                'risk_free_rates': {'India': 0.0525},  # 5.25% for Indian T-bills
                'lookback_periods': ['1 Month', '3 Months', '6 Months', '1 Year']
            },
            'Sortino Ratio': {
                'description': 'Risk-adjusted return focusing only on downside risk',
                'risk_free_rates': {'India': 0.0525},
                'lookback_periods': ['1 Month', '3 Months', '6 Months', '1 Year']
            },
            'Correlation Matrix': {
                'description': 'Measures how assets move in relation to each other',
                'lookback_periods': ['1 Month', '3 Months', '6 Months', '1 Year']
            }
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'VaR_percent': {'low': 2.0, 'medium': 5.0, 'high': float('inf')},
            'ES_percent': {'low': 3.0, 'medium': 7.0, 'high': float('inf')},
            'MaxDD_percent': {'low': 10.0, 'medium': 20.0, 'high': float('inf')},
            'Volatility_percent': {'low': 15.0, 'medium': 25.0, 'high': float('inf')},
            'Beta': {'low': 0.8, 'medium': 1.2, 'high': float('inf')},
            'Sharpe': {'high': 1.0, 'medium': 0.5, 'low': -float('inf')},
            'Sortino': {'high': 1.2, 'medium': 0.6, 'low': -float('inf')}
        }
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95, 
                     method: str = 'Historical') -> float:
        """Calculate Value at Risk (VaR)"""
        try:
            if method == 'Historical':
                # Historical VaR
                return -np.percentile(returns, 100 * (1 - confidence_level))
            
            elif method == 'Parametric':
                # Parametric VaR (assumes normal distribution)
                from scipy import stats
                mean = returns.mean()
                std = returns.std()
                return -(mean + std * stats.norm.ppf(confidence_level))
            
            elif method == 'Monte Carlo':
                # Simple Monte Carlo VaR
                mean = returns.mean()
                std = returns.std()
                np.random.seed(42)  # For reproducibility
                simulations = 10000
                sim_returns = np.random.normal(mean, std, simulations)
                return -np.percentile(sim_returns, 100 * (1 - confidence_level))
            
            else:
                return 0.0
        
        except Exception as e:
            print(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95,
                                    method: str = 'Historical') -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if method == 'Historical':
                # Historical ES
                var = self.calculate_var(returns, confidence_level, 'Historical')
                return -returns[returns <= -var].mean()
            
            elif method == 'Parametric':
                # Parametric ES (assumes normal distribution)
                from scipy import stats
                mean = returns.mean()
                std = returns.std()
                var = -(mean + std * stats.norm.ppf(confidence_level))
                return -(mean - std * stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level))
            
            else:
                return 0.0
        
        except Exception as e:
            print(f"Error calculating Expected Shortfall: {e}")
            return 0.0
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> float:
        """Calculate Maximum Drawdown"""
        try:
            # Calculate cumulative returns
            cum_returns = (1 + prices.pct_change().fillna(0)).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.cummax()
            
            # Calculate drawdown
            drawdown = (cum_returns / running_max - 1)
            
            # Get maximum drawdown
            max_drawdown = drawdown.min()
            
            return max_drawdown
        
        except Exception as e:
            print(f"Error calculating Maximum Drawdown: {e}")
            return 0.0
    
    def calculate_volatility(self, returns: pd.Series, method: str = 'Simple',
                           annualize: bool = True) -> float:
        """Calculate volatility of returns"""
        try:
            if method == 'Simple':
                # Simple volatility
                vol = returns.std()
            
            elif method == 'Exponentially Weighted':
                # EWMA volatility
                vol = returns.ewm(span=20).std().iloc[-1]
            
            elif method == 'GARCH':
                # Simple GARCH approximation (for full GARCH, use arch package)
                vol = returns.rolling(window=20).std().mean()
            
            else:
                vol = returns.std()
            
            # Annualize if requested (assuming daily returns)
            if annualize:
                vol = vol * np.sqrt(252)  # 252 trading days in a year
            
            return vol
        
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return 0.0
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta (systematic risk)"""
        try:
            # Calculate covariance
            covariance = returns.cov(market_returns)
            
            # Calculate market variance
            market_variance = market_returns.var()
            
            # Calculate beta
            beta = covariance / market_variance if market_variance != 0 else 1.0
            
            return beta
        
        except Exception as e:
            print(f"Error calculating beta: {e}")
            return 1.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0525,
                             annualize: bool = True) -> float:
        """Calculate Sharpe ratio"""
        try:
            # Calculate excess returns
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            # Calculate mean excess return
            mean_excess_return = excess_returns.mean()
            
            # Calculate standard deviation
            std_dev = returns.std()
            
            # Calculate Sharpe ratio
            sharpe = mean_excess_return / std_dev if std_dev != 0 else 0.0
            
            # Annualize if requested
            if annualize:
                sharpe = sharpe * np.sqrt(252)  # 252 trading days in a year
            
            return sharpe
        
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0525,
                              target_return: float = 0.0, annualize: bool = True) -> float:
        """Calculate Sortino ratio"""
        try:
            # Calculate excess returns
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            # Calculate mean excess return
            mean_excess_return = excess_returns.mean()
            
            # Calculate downside deviation (only negative returns)
            downside_returns = returns[returns < target_return]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0001
            
            # Calculate Sortino ratio
            sortino = mean_excess_return / downside_deviation if downside_deviation != 0 else 0.0
            
            # Annualize if requested
            if annualize:
                sortino = sortino * np.sqrt(252)  # 252 trading days in a year
            
            return sortino
        
        except Exception as e:
            print(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_correlation_matrix(self, returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate correlation matrix between assets"""
        try:
            # Create DataFrame from returns dictionary
            returns_df = pd.DataFrame(returns_dict)
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            return corr_matrix
        
        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def create_var_chart(self, returns: pd.Series, confidence_levels: List[float] = [0.9, 0.95, 0.99],
                        method: str = 'Historical', theme: str = 'dark') -> go.Figure:
        """Create VaR visualization chart"""
        try:
            # Get theme colors
            colors = self.chart_themes[theme]
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram of returns
            fig.add_trace(go.Histogram(
                x=returns,
                name='Returns Distribution',
                marker_color=colors['line_color'],
                opacity=0.7,
                nbinsx=50
            ))
            
            # Add VaR lines for each confidence level
            for cl in confidence_levels:
                var = self.calculate_var(returns, cl, method)
                fig.add_vline(
                    x=-var,
                    line_width=2,
                    line_dash="dash",
                    line_color=colors['high_risk_color'],
                    annotation_text=f"VaR ({cl*100:.0f}%): {var*100:.2f}%",
                    annotation_position="top right"
                )
            
            # Update layout
            fig.update_layout(
                title=f'Value at Risk ({method} Method)',
                xaxis_title='Returns',
                yaxis_title='Frequency',
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                xaxis=dict(gridcolor=colors['grid_color']),
                yaxis=dict(gridcolor=colors['grid_color']),
                height=400
            )
            
            return fig
        
        except Exception as e:
            print(f"Error creating VaR chart: {e}")
            return None
    
    def create_drawdown_chart(self, prices: pd.Series, theme: str = 'dark') -> go.Figure:
        """Create drawdown visualization chart"""
        try:
            # Calculate cumulative returns
            cum_returns = (1 + prices.pct_change().fillna(0)).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.cummax()
            
            # Calculate drawdown
            drawdown = (cum_returns / running_max - 1) * 100  # Convert to percentage
            
            # Get maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_date = drawdown.idxmin()
            
            # Get theme colors
            colors = self.chart_themes[theme]
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices,
                    name='Price',
                    line=dict(color=colors['line_color'], width=1)
                ),
                secondary_y=True
            )
            
            # Add drawdown area
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color=colors['high_risk_color'], width=1)
                ),
                secondary_y=False
            )
            
            # Add annotation for maximum drawdown
            fig.add_annotation(
                x=max_drawdown_date,
                y=max_drawdown,
                text=f"Max DD: {max_drawdown:.2f}%",
                showarrow=True,
                arrowhead=1,
                arrowcolor=colors['high_risk_color'],
                arrowsize=1,
                arrowwidth=2,
                ax=-40,
                ay=40
            )
            
            # Update layout
            fig.update_layout(
                title='Drawdown Analysis',
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                xaxis=dict(gridcolor=colors['grid_color']),
                yaxis=dict(gridcolor=colors['grid_color']),
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Drawdown (%)", secondary_y=False)
            fig.update_yaxes(title_text="Price", secondary_y=True)
            
            return fig
        
        except Exception as e:
            print(f"Error creating drawdown chart: {e}")
            return None
    
    def create_correlation_heatmap(self, returns_dict: Dict[str, pd.Series], theme: str = 'dark') -> go.Figure:
        """Create correlation heatmap"""
        try:
            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix(returns_dict)
            
            if corr_matrix.empty:
                return None
            
            # Get theme colors
            colors = self.chart_themes[theme]
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale=[
                    colors['high_risk_color'],
                    colors['bg_color'],
                    colors['low_risk_color']
                ],
                zmin=-1,
                zmax=1,
                aspect="auto"
            )
            
            # Update layout
            fig.update_layout(
                title='Asset Correlation Matrix',
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                height=500
            )
            
            return fig
        
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return None
    
    def create_risk_metrics_table(self, portfolio_data: Dict[str, pd.DataFrame], 
                                weights: Dict[str, float],
                                benchmark: str = 'NIFTY 50',
                                lookback_days: int = 90) -> pd.DataFrame:
        """Calculate and format risk metrics for portfolio"""
        try:
            # Extract prices and calculate returns
            portfolio_returns = {}
            portfolio_prices = {}
            
            for symbol, weight in weights.items():
                if symbol in portfolio_data:
                    df = portfolio_data[symbol].tail(lookback_days + 1).copy()
                    portfolio_prices[symbol] = df['Close']
                    portfolio_returns[symbol] = df['Close'].pct_change().fillna(0)
            
            if not portfolio_returns:
                return pd.DataFrame()
            
            # Calculate portfolio returns
            portfolio_return_series = pd.Series(0.0, index=list(portfolio_returns.values())[0].index)
            for symbol, returns in portfolio_returns.items():
                portfolio_return_series += returns * weights.get(symbol, 0)
            
            # Get benchmark returns if available
            benchmark_returns = None
            if benchmark in portfolio_data:
                benchmark_df = portfolio_data[benchmark].tail(lookback_days + 1).copy()
                benchmark_returns = benchmark_df['Close'].pct_change().fillna(0)
            
            # Calculate risk metrics
            metrics = {}
            
            # VaR metrics
            for cl in [0.9, 0.95, 0.99]:
                for method in ['Historical', 'Parametric']:
                    metrics[f'VaR_{int(cl*100)}_{method}'] = self.calculate_var(
                        portfolio_return_series, cl, method) * 100  # Convert to percentage
            
            # Expected Shortfall
            metrics['ES_95'] = self.calculate_expected_shortfall(
                portfolio_return_series, 0.95, 'Historical') * 100  # Convert to percentage
            
            # Maximum Drawdown
            # Create portfolio price series
            portfolio_price_series = pd.Series(100.0, index=list(portfolio_returns.values())[0].index)
            for i in range(1, len(portfolio_price_series)):
                portfolio_price_series.iloc[i] = portfolio_price_series.iloc[i-1] * (1 + portfolio_return_series.iloc[i])
            
            metrics['MaxDD'] = abs(self.calculate_maximum_drawdown(portfolio_price_series)) * 100  # Convert to percentage
            
            # Volatility
            metrics['Volatility'] = self.calculate_volatility(portfolio_return_series, 'Simple', True) * 100  # Convert to percentage
            
            # Beta (if benchmark available)
            if benchmark_returns is not None:
                metrics['Beta'] = self.calculate_beta(portfolio_return_series, benchmark_returns)
            else:
                metrics['Beta'] = 1.0
            
            # Sharpe Ratio
            metrics['Sharpe'] = self.calculate_sharpe_ratio(portfolio_return_series)
            
            # Sortino Ratio
            metrics['Sortino'] = self.calculate_sortino_ratio(portfolio_return_series)
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Value at Risk (95%, Historical)',
                    'Expected Shortfall (95%)',
                    'Maximum Drawdown',
                    'Annualized Volatility',
                    f'Beta (vs {benchmark})',
                    'Sharpe Ratio',
                    'Sortino Ratio'
                ],
                'Value': [
                    f"{metrics['VaR_95_Historical']:.2f}%",
                    f"{metrics['ES_95']:.2f}%",
                    f"{metrics['MaxDD']:.2f}%",
                    f"{metrics['Volatility']:.2f}%",
                    f"{metrics['Beta']:.2f}",
                    f"{metrics['Sharpe']:.2f}",
                    f"{metrics['Sortino']:.2f}"
                ],
                'Risk Level': [
                    self._get_risk_level('VaR_percent', metrics['VaR_95_Historical']),
                    self._get_risk_level('ES_percent', metrics['ES_95']),
                    self._get_risk_level('MaxDD_percent', metrics['MaxDD']),
                    self._get_risk_level('Volatility_percent', metrics['Volatility']),
                    self._get_risk_level('Beta', metrics['Beta']),
                    self._get_risk_level('Sharpe', metrics['Sharpe']),
                    self._get_risk_level('Sortino', metrics['Sortino'])
                ]
            })
            
            return metrics_df
        
        except Exception as e:
            print(f"Error creating risk metrics table: {e}")
            return pd.DataFrame()
    
    def _get_risk_level(self, metric_type: str, value: float) -> str:
        """Determine risk level based on thresholds"""
        if metric_type in self.risk_thresholds:
            thresholds = self.risk_thresholds[metric_type]
            
            # For metrics where higher is better (Sharpe, Sortino)
            if metric_type in ['Sharpe', 'Sortino']:
                if value >= thresholds['high']:
                    return 'Low'
                elif value >= thresholds['medium']:
                    return 'Medium'
                else:
                    return 'High'
            # For metrics where lower is better
            else:
                if value <= thresholds['low']:
                    return 'Low'
                elif value <= thresholds['medium']:
                    return 'Medium'
                else:
                    return 'High'
        
        return 'Medium'

def render_risk_management_panel(stock_data: Dict[str, pd.DataFrame], theme: str = 'dark'):
    """Render risk management analysis panel"""
    st.header("🛡️ Risk Management")
    
    # Initialize risk panel
    risk_panel = RiskManagementPanel()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Risk", "Value at Risk", "Drawdown Analysis", "Correlation Analysis"])
    
    # Sample portfolio (in production, this would come from user input)
    default_portfolio = {
        'RELIANCE': 0.20,
        'TCS': 0.15,
        'HDFCBANK': 0.15,
        'INFY': 0.10,
        'ITC': 0.10,
        'SBIN': 0.10,
        'HINDUNILVR': 0.10,
        'BHARTIARTL': 0.10
    }
    
    with tab1:
        st.subheader("Portfolio Risk Analysis")
        
        # Portfolio selection
        st.write("**Portfolio Allocation**")
        
        # In a real app, you would allow users to input their portfolio
        # For now, we'll use the sample portfolio
        col1, col2 = st.columns([3, 1])
        
        with col1:
            portfolio_table = pd.DataFrame({
                'Stock': list(default_portfolio.keys()),
                'Weight': [f"{w*100:.1f}%" for w in default_portfolio.values()]
            })
            st.dataframe(portfolio_table, use_container_width=True)
        
        with col2:
            benchmark = st.selectbox(
                "Benchmark",
                options=["NIFTY 50", "NIFTY 500", "SENSEX"],
                index=0,
                key="risk_benchmark"
            )
            
            lookback = st.selectbox(
                "Lookback Period",
                options=["1 Month", "3 Months", "6 Months", "1 Year"],
                index=1,
                key="risk_lookback"
            )
            
            # Convert lookback to days
            lookback_days = {
                "1 Month": 30,
                "3 Months": 90,
                "6 Months": 180,
                "1 Year": 365
            }[lookback]
        
        # Calculate and display risk metrics
        metrics_df = risk_panel.create_risk_metrics_table(
            stock_data, default_portfolio, benchmark, lookback_days)
        
        if not metrics_df.empty:
            # Apply conditional formatting
            def highlight_risk(val):
                if val == 'Low':
                    return 'background-color: #00ff8844'
                elif val == 'Medium':
                    return 'background-color: #ffa50044'
                elif val == 'High':
                    return 'background-color: #ff444444'
                return ''
            
            styled_df = metrics_df.style.applymap(highlight_risk, subset=['Risk Level'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Risk summary
            st.subheader("Risk Assessment Summary")
            
            # Count risk levels
            risk_counts = metrics_df['Risk Level'].value_counts()
            high_count = risk_counts.get('High', 0)
            medium_count = risk_counts.get('Medium', 0)
            low_count = risk_counts.get('Low', 0)
            
            # Determine overall risk
            if high_count >= 3 or (high_count >= 2 and 'High' == metrics_df.loc[0, 'Risk Level']):
                overall_risk = "High"
                risk_color = "#ff4444"
                risk_message = "Your portfolio shows high risk characteristics. Consider diversification and reducing exposure to volatile assets."
            elif high_count >= 1 or medium_count >= 3:
                overall_risk = "Medium"
                risk_color = "#ffa500"
                risk_message = "Your portfolio has moderate risk. Some metrics indicate potential for significant drawdowns under stress conditions."
            else:
                overall_risk = "Low"
                risk_color = "#00ff88"
                risk_message = "Your portfolio demonstrates good risk management with balanced risk-reward characteristics."
            
            # Display overall risk
            st.markdown(f"<h3 style='color: {risk_color}'>Overall Risk: {overall_risk}</h3>", unsafe_allow_html=True)
            st.write(risk_message)
            
            # Risk recommendations
            st.subheader("Risk Management Recommendations")
            
            if metrics_df.loc[3, 'Risk Level'] == 'High':  # Volatility
                st.write("- **High Volatility**: Consider adding more stable, low-beta stocks or defensive sectors")
            
            if metrics_df.loc[4, 'Risk Level'] == 'High':  # Beta
                st.write("- **High Beta**: Your portfolio is more volatile than the market. Consider reducing exposure to high-beta stocks")
            
            if metrics_df.loc[0, 'Risk Level'] == 'High' or metrics_df.loc[1, 'Risk Level'] == 'High':  # VaR or ES
                st.write("- **Tail Risk**: Your portfolio has significant downside risk. Consider adding hedges or reducing position sizes")
            
            if metrics_df.loc[5, 'Risk Level'] == 'High':  # Sharpe
                st.write("- **Poor Risk-Adjusted Returns**: Your portfolio isn't compensating adequately for its risk level")
            
            if overall_risk != "High":
                st.write("- **Regular Rebalancing**: Continue to monitor and rebalance your portfolio to maintain risk targets")
        else:
            st.warning("Insufficient data to calculate risk metrics.")
    
    with tab2:
        st.subheader("Value at Risk (VaR) Analysis")
        
        # VaR settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            var_confidence = st.selectbox(
                "Confidence Level",
                options=["90%", "95%", "99%"],
                index=1,
                key="var_confidence"
            )
            confidence_level = float(var_confidence.strip('%')) / 100
        
        with col2:
            var_method = st.selectbox(
                "Calculation Method",
                options=["Historical", "Parametric", "Monte Carlo"],
                index=0,
                key="var_method"
            )
        
        with col3:
            var_horizon = st.selectbox(
                "Time Horizon",
                options=["1 Day", "1 Week", "1 Month"],
                index=0,
                key="var_horizon"
            )
        
        # Calculate portfolio returns for VaR
        portfolio_returns = pd.Series(0.0, index=stock_data.get('NIFTY 50', pd.DataFrame()).index[-90:])
        
        for symbol, weight in default_portfolio.items():
            if symbol in stock_data:
                df = stock_data[symbol].tail(90).copy()
                returns = df['Close'].pct_change().fillna(0)
                portfolio_returns += returns * weight
        
        # Create VaR chart
        if not portfolio_returns.empty:
            fig = risk_panel.create_var_chart(
                portfolio_returns, [confidence_level], var_method, theme)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display VaR values
            var_value = risk_panel.calculate_var(portfolio_returns, confidence_level, var_method) * 100
            es_value = risk_panel.calculate_expected_shortfall(portfolio_returns, confidence_level, var_method) * 100
            
            # Display VaR interpretation
            st.subheader("VaR Interpretation")
            
            # Portfolio value (assumed for demonstration)
            portfolio_value = 1000000  # ₹10 lakhs
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label=f"Value at Risk ({var_confidence})",
                    value=f"{var_value:.2f}%",
                    delta=f"-₹{portfolio_value * var_value / 100:.0f}"
                )
                
                st.write(f"With {var_confidence} confidence, your portfolio will not lose more than "  
                        f"**{var_value:.2f}%** (₹{portfolio_value * var_value / 100:.0f}) "  
                        f"over the next {var_horizon.lower()}.")
            
            with col2:
                st.metric(
                    label=f"Expected Shortfall ({var_confidence})",
                    value=f"{es_value:.2f}%",
                    delta=f"-₹{portfolio_value * es_value / 100:.0f}"
                )
                
                st.write(f"If the {var_confidence} VaR is breached, the average loss would be "  
                        f"**{es_value:.2f}%** (₹{portfolio_value * es_value / 100:.0f}).")
            
            # VaR limitations
            st.info(
                "**Limitations of VaR:**\n"  
                "- VaR does not tell you how much you could lose beyond the threshold\n"  
                "- Historical VaR assumes past patterns will repeat\n"  
                "- VaR can underestimate risk during market regime changes\n"  
                "- Consider multiple risk metrics for a complete picture"
            )
        else:
            st.warning("Insufficient data to calculate Value at Risk.")
    
    with tab3:
        st.subheader("Drawdown Analysis")
        
        # Select stock for drawdown analysis
        selected_stock = st.selectbox(
            "Select Stock/Index",
            options=["Portfolio"] + list(stock_data.keys()),
            index=0,
            key="drawdown_stock"
        )
        
        # Calculate drawdown
        if selected_stock == "Portfolio":
            # Calculate portfolio prices
            first_date = min([df.index[0] for df in stock_data.values() if not df.empty])
            last_date = max([df.index[-1] for df in stock_data.values() if not df.empty])
            date_range = pd.date_range(start=first_date, end=last_date, freq='B')
            
            portfolio_prices = pd.Series(100.0, index=date_range)  # Start with 100
            portfolio_returns = pd.Series(0.0, index=date_range)
            
            # Calculate portfolio returns
            for symbol, weight in default_portfolio.items():
                if symbol in stock_data and not stock_data[symbol].empty:
                    df = stock_data[symbol].copy()
                    returns = df['Close'].pct_change().fillna(0)
                    # Align dates
                    returns = returns.reindex(date_range)
                    portfolio_returns += returns * weight
            
            # Calculate portfolio prices from returns
            for i in range(1, len(portfolio_prices)):
                portfolio_prices.iloc[i] = portfolio_prices.iloc[i-1] * (1 + portfolio_returns.iloc[i])
            
            # Create drawdown chart
            fig = risk_panel.create_drawdown_chart(portfolio_prices, theme)
        else:
            # Use selected stock prices
            if selected_stock in stock_data and not stock_data[selected_stock].empty:
                prices = stock_data[selected_stock]['Close']
                fig = risk_panel.create_drawdown_chart(prices, theme)
            else:
                fig = None
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown interpretation
            st.subheader("Drawdown Interpretation")
            
            st.write(
                "**Maximum Drawdown** measures the largest peak-to-trough decline in value. "  
                "It helps assess downside risk and recovery time after significant losses."
            )
            
            # Recovery analysis
            st.write("**Recovery Analysis:**")
            st.write(
                "- A 10% drawdown requires an 11.1% gain to recover\n"  
                "- A 20% drawdown requires a 25% gain to recover\n"  
                "- A 50% drawdown requires a 100% gain to recover"
            )
            
            # Risk management strategies
            st.info(
                "**Risk Management Strategies:**\n"  
                "- Set stop-loss orders to limit drawdowns\n"  
                "- Diversify across uncorrelated assets\n"  
                "- Consider hedging strategies during high volatility periods\n"  
                "- Maintain cash reserves for buying opportunities during drawdowns"
            )
        else:
            st.warning("Insufficient data to analyze drawdowns.")
    
    with tab4:
        st.subheader("Correlation Analysis")
        
        # Select stocks for correlation analysis
        default_stocks = list(default_portfolio.keys())[:6]  # Limit to 6 for better visualization
        
        selected_stocks = st.multiselect(
            "Select Stocks for Correlation Analysis",
            options=list(stock_data.keys()),
            default=default_stocks,
            key="correlation_stocks"
        )
        
        # Lookback period
        corr_lookback = st.selectbox(
            "Lookback Period",
            options=["1 Month", "3 Months", "6 Months", "1 Year"],
            index=1,
            key="corr_lookback"
        )
        
        # Convert lookback to days
        corr_days = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }[corr_lookback]
        
        if selected_stocks:
            # Calculate returns for selected stocks
            returns_dict = {}
            
            for symbol in selected_stocks:
                if symbol in stock_data and not stock_data[symbol].empty:
                    df = stock_data[symbol].tail(corr_days + 1).copy()
                    returns_dict[symbol] = df['Close'].pct_change().fillna(0)
            
            # Create correlation heatmap
            if returns_dict:
                fig = risk_panel.create_correlation_heatmap(returns_dict, theme)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation interpretation
                    st.subheader("Correlation Interpretation")
                    
                    # Calculate average correlation
                    corr_matrix = risk_panel.calculate_correlation_matrix(returns_dict)
                    avg_corr = (corr_matrix.sum().sum() - len(corr_matrix)) / (len(corr_matrix) * (len(corr_matrix) - 1))
                    
                    # Find highest and lowest correlations
                    corr_values = []
                    for i in range(len(corr_matrix)):
                        for j in range(i+1, len(corr_matrix)):
                            corr_values.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                    
                    if corr_values:
                        highest_corr = max(corr_values, key=lambda x: x[2])
                        lowest_corr = min(corr_values, key=lambda x: x[2])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Average Correlation", f"{avg_corr:.2f}")
                            
                            if avg_corr > 0.7:
                                st.write("Your selected assets are **highly correlated**, indicating limited diversification benefits.")
                            elif avg_corr > 0.3:
                                st.write("Your selected assets show **moderate correlation**, with some diversification benefits.")
                            else:
                                st.write("Your selected assets have **low correlation**, providing good diversification benefits.")
                        
                        with col2:
                            st.write(f"**Highest Correlation:** {highest_corr[0]} & {highest_corr[1]} ({highest_corr[2]:.2f})")
                            st.write(f"**Lowest Correlation:** {lowest_corr[0]} & {lowest_corr[1]} ({lowest_corr[2]:.2f})")
                    
                    # Diversification recommendations
                    st.subheader("Diversification Recommendations")
                    
                    if avg_corr > 0.6:
                        st.write(
                            "- Consider adding assets from different sectors or asset classes\n"  
                            "- Look for negative or low correlation assets to improve portfolio efficiency\n"  
                            "- International exposure may provide additional diversification"
                        )
                    elif avg_corr > 0.3:
                        st.write(
                            "- Your portfolio has reasonable diversification\n"  
                            "- Consider adding 1-2 uncorrelated assets to further improve risk-adjusted returns\n"  
                            "- Monitor correlations during market stress as they tend to increase"
                        )
                    else:
                        st.write(
                            "- Your portfolio shows excellent diversification\n"  
                            "- Maintain this diversification through regular rebalancing\n"  
                            "- Consider the impact of transaction costs when rebalancing frequently"
                        )
            else:
                st.warning("Insufficient data to calculate correlations.")
        else:
            st.info("Please select at least two stocks for correlation analysis.")