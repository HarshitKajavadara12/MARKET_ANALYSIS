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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class PredictionPanel:
    def __init__(self):
        self.prediction_models = {
            'LSTM': 'Long Short-Term Memory Neural Network',
            'ARIMA': 'AutoRegressive Integrated Moving Average',
            'Random Forest': 'Random Forest Regressor',
            'Linear Regression': 'Linear Regression Model',
            'SVM': 'Support Vector Machine',
            'Ensemble': 'Ensemble of Multiple Models'
        }
        
        self.prediction_horizons = {
            '1D': {'days': 1, 'label': '1 Day'},
            '3D': {'days': 3, 'label': '3 Days'},
            '1W': {'days': 7, 'label': '1 Week'},
            '2W': {'days': 14, 'label': '2 Weeks'},
            '1M': {'days': 30, 'label': '1 Month'},
            '3M': {'days': 90, 'label': '3 Months'}
        }
        
        self.confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
        
        self.chart_themes = {
            'dark': {
                'bg_color': '#1e1e1e',
                'grid_color': '#404040',
                'text_color': '#ffffff',
                'bullish_color': '#00ff88',
                'bearish_color': '#ff4444',
                'neutral_color': '#ffa500',
                'prediction_color': '#00bfff',
                'confidence_color': 'rgba(0, 191, 255, 0.2)'
            },
            'light': {
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'bullish_color': '#26a69a',
                'bearish_color': '#ef5350',
                'neutral_color': '#ff9800',
                'prediction_color': '#2196f3',
                'confidence_color': 'rgba(33, 150, 243, 0.2)'
            }
        }
        
        self.risk_metrics = {
            'VaR_95': 'Value at Risk (95%)',
            'VaR_99': 'Value at Risk (99%)',
            'Expected_Shortfall': 'Expected Shortfall',
            'Maximum_Drawdown': 'Maximum Drawdown',
            'Sharpe_Ratio': 'Sharpe Ratio',
            'Volatility': 'Annualized Volatility'
        }
    
    def generate_sample_predictions(self, 
                                  current_price: float, 
                                  horizon_days: int, 
                                  model_type: str = 'LSTM') -> Dict[str, Union[List, float]]:
        """Generate sample predictions for demonstration"""
        try:
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            
            # Generate base trend
            if model_type == 'LSTM':
                trend_strength = np.random.normal(0.001, 0.005)  # Slight upward bias
                volatility = np.random.uniform(0.015, 0.025)
            elif model_type == 'ARIMA':
                trend_strength = np.random.normal(0.0005, 0.003)
                volatility = np.random.uniform(0.012, 0.020)
            elif model_type == 'Random Forest':
                trend_strength = np.random.normal(0.002, 0.004)
                volatility = np.random.uniform(0.018, 0.028)
            else:
                trend_strength = np.random.normal(0.001, 0.004)
                volatility = np.random.uniform(0.015, 0.025)
            
            # Generate prediction path
            dates = []
            prices = []
            upper_bounds = []
            lower_bounds = []
            
            current_date = datetime.now()
            price = current_price
            
            for i in range(horizon_days + 1):
                if i == 0:
                    dates.append(current_date)
                    prices.append(price)
                    upper_bounds.append(price)
                    lower_bounds.append(price)
                else:
                    # Add trend and random walk
                    daily_return = trend_strength + np.random.normal(0, volatility)
                    price *= (1 + daily_return)
                    
                    # Calculate confidence intervals (expanding with time)
                    confidence_width = volatility * np.sqrt(i) * 1.96  # 95% confidence
                    upper_bound = price * (1 + confidence_width)
                    lower_bound = price * (1 - confidence_width)
                    
                    dates.append(current_date + timedelta(days=i))
                    prices.append(price)
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(lower_bound)
            
            # Calculate model confidence (decreases with time)
            base_confidence = {
                'LSTM': 0.85,
                'ARIMA': 0.75,
                'Random Forest': 0.80,
                'Linear Regression': 0.65,
                'SVM': 0.70,
                'Ensemble': 0.90
            }.get(model_type, 0.75)
            
            confidence = base_confidence * np.exp(-horizon_days / 30)  # Decay over time
            
            return {
                'dates': dates,
                'predictions': prices,
                'upper_bounds': upper_bounds,
                'lower_bounds': lower_bounds,
                'confidence': confidence,
                'model_type': model_type,
                'horizon_days': horizon_days,
                'trend_strength': trend_strength,
                'volatility': volatility
            }
            
        except Exception as e:
            return {
                'dates': [datetime.now()],
                'predictions': [current_price],
                'upper_bounds': [current_price],
                'lower_bounds': [current_price],
                'confidence': 0.5,
                'model_type': model_type,
                'horizon_days': horizon_days,
                'trend_strength': 0.0,
                'volatility': 0.02
            }
    
    def calculate_prediction_metrics(self, 
                                   actual_prices: List[float], 
                                   predicted_prices: List[float]) -> Dict[str, float]:
        """Calculate prediction accuracy metrics"""
        try:
            if len(actual_prices) != len(predicted_prices) or len(actual_prices) == 0:
                return {}
            
            actual = np.array(actual_prices)
            predicted = np.array(predicted_prices)
            
            # Calculate metrics
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual, predicted)
            
            # Calculate percentage errors
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Direction accuracy (how often we predicted the right direction)
            actual_direction = np.diff(actual) > 0
            predicted_direction = np.diff(predicted) > 0
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            return {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'Direction_Accuracy': direction_accuracy
            }
            
        except Exception as e:
            return {}
    
    def calculate_risk_metrics(self, returns: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate risk metrics for predictions"""
        try:
            if not returns:
                return {}
            
            returns_array = np.array(returns)
            
            # Value at Risk
            var_95 = np.percentile(returns_array, (1 - 0.95) * 100)
            var_99 = np.percentile(returns_array, (1 - 0.99) * 100)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = returns_array[returns_array <= var_95].mean()
            
            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Sharpe Ratio (assuming risk-free rate of 6% for India)
            risk_free_rate = 0.06 / 252  # Daily risk-free rate
            excess_returns = returns_array - risk_free_rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            # Annualized Volatility
            volatility = np.std(returns_array) * np.sqrt(252)
            
            return {
                'VaR_95': var_95,
                'VaR_99': var_99,
                'Expected_Shortfall': es_95,
                'Maximum_Drawdown': max_drawdown,
                'Sharpe_Ratio': sharpe_ratio,
                'Volatility': volatility
            }
            
        except Exception as e:
            return {}
    
    def create_prediction_chart(self, 
                              historical_data: pd.DataFrame,
                              prediction_data: Dict,
                              symbol: str,
                              theme: str = 'dark') -> go.Figure:
        """Create prediction visualization chart"""
        try:
            colors = self.chart_themes[theme]
            
            fig = go.Figure()
            
            # Add historical price data
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color=colors['text_color'], width=2)
            ))
            
            # Add prediction line
            fig.add_trace(go.Scatter(
                x=prediction_data['dates'],
                y=prediction_data['predictions'],
                mode='lines+markers',
                name=f'{prediction_data["model_type"]} Prediction',
                line=dict(color=colors['prediction_color'], width=3, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=prediction_data['dates'],
                y=prediction_data['upper_bounds'],
                mode='lines',
                name='Upper Bound (95%)',
                line=dict(color=colors['prediction_color'], width=1, dash='dot'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=prediction_data['dates'],
                y=prediction_data['lower_bounds'],
                mode='lines',
                name='Lower Bound (95%)',
                line=dict(color=colors['prediction_color'], width=1, dash='dot'),
                fill='tonexty',
                fillcolor=colors['confidence_color'],
                showlegend=True
            ))
            
            # Add vertical line to separate historical and predicted data
            current_time = datetime.now()
            fig.add_vline(
                x=current_time,
                line_dash="solid",
                line_color=colors['neutral_color'],
                annotation_text="Current Time"
            )
            
            fig.update_layout(
                title=f'{symbol} Price Prediction - {prediction_data["model_type"]} Model',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=500,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            return fig
            
        except Exception as e:
            return go.Figure()
    
    def create_model_comparison_chart(self, 
                                    predictions_dict: Dict[str, Dict],
                                    theme: str = 'dark') -> go.Figure:
        """Create model comparison chart"""
        try:
            colors = self.chart_themes[theme]
            
            fig = go.Figure()
            
            model_colors = {
                'LSTM': colors['prediction_color'],
                'ARIMA': colors['bullish_color'],
                'Random Forest': colors['bearish_color'],
                'Linear Regression': colors['neutral_color'],
                'SVM': '#9c27b0',
                'Ensemble': '#ff5722'
            }
            
            for model_name, prediction_data in predictions_dict.items():
                fig.add_trace(go.Scatter(
                    x=prediction_data['dates'],
                    y=prediction_data['predictions'],
                    mode='lines+markers',
                    name=model_name,
                    line=dict(
                        color=model_colors.get(model_name, colors['text_color']),
                        width=2
                    ),
                    marker=dict(size=4)
                ))
            
            fig.update_layout(
                title='Model Predictions Comparison',
                xaxis_title='Date',
                yaxis_title='Predicted Price (₹)',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=400,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color']),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            return go.Figure()
    
    def create_accuracy_metrics_chart(self, 
                                    metrics_dict: Dict[str, Dict[str, float]],
                                    theme: str = 'dark') -> go.Figure:
        """Create model accuracy comparison chart"""
        try:
            colors = self.chart_themes[theme]
            
            if not metrics_dict:
                return go.Figure()
            
            # Prepare data for visualization
            models = list(metrics_dict.keys())
            metrics = ['MAE', 'RMSE', 'MAPE', 'Direction_Accuracy', 'R2']
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=metrics,
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
            )
            
            positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
            
            for i, metric in enumerate(metrics):
                if i < len(positions):
                    row, col = positions[i]
                    
                    values = [metrics_dict[model].get(metric, 0) for model in models]
                    
                    fig.add_trace(
                        go.Bar(
                            x=models,
                            y=values,
                            name=metric,
                            marker_color=colors['prediction_color'],
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title='Model Performance Comparison',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=500,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            return go.Figure()
    
    def generate_trading_signals(self, prediction_data: Dict, current_price: float) -> Dict[str, Union[str, float]]:
        """Generate trading signals based on predictions"""
        try:
            if not prediction_data['predictions'] or len(prediction_data['predictions']) < 2:
                return {'signal': 'HOLD', 'confidence': 0.5, 'target_price': current_price, 'stop_loss': current_price}
            
            # Get predicted price for next period
            next_price = prediction_data['predictions'][1]
            predicted_return = (next_price - current_price) / current_price
            
            # Get model confidence
            model_confidence = prediction_data['confidence']
            
            # Generate signal based on predicted return and confidence
            if predicted_return > 0.02 and model_confidence > 0.7:  # Strong buy
                signal = 'STRONG BUY'
                target_price = prediction_data['predictions'][-1]  # End of horizon
                stop_loss = current_price * 0.95  # 5% stop loss
            elif predicted_return > 0.005 and model_confidence > 0.6:  # Buy
                signal = 'BUY'
                target_price = next_price * 1.05
                stop_loss = current_price * 0.97  # 3% stop loss
            elif predicted_return < -0.02 and model_confidence > 0.7:  # Strong sell
                signal = 'STRONG SELL'
                target_price = prediction_data['predictions'][-1]
                stop_loss = current_price * 1.05  # 5% stop loss (for short)
            elif predicted_return < -0.005 and model_confidence > 0.6:  # Sell
                signal = 'SELL'
                target_price = next_price * 0.95
                stop_loss = current_price * 1.03  # 3% stop loss (for short)
            else:
                signal = 'HOLD'
                target_price = current_price
                stop_loss = current_price
            
            return {
                'signal': signal,
                'confidence': model_confidence,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'predicted_return': predicted_return,
                'risk_reward_ratio': abs((target_price - current_price) / (stop_loss - current_price)) if stop_loss != current_price else 1.0
            }
            
        except Exception as e:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'target_price': current_price,
                'stop_loss': current_price,
                'predicted_return': 0.0,
                'risk_reward_ratio': 1.0
            }
    
    def generate_sample_historical_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Generate sample historical data for demonstration"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Generate realistic price movement
            np.random.seed(hash(symbol) % 1000)
            
            initial_price = np.random.uniform(100, 2000)  # Random starting price
            prices = [initial_price]
            
            for i in range(1, days):
                # Add trend and volatility
                daily_return = np.random.normal(0.001, 0.02)  # Slight upward bias with 2% daily volatility
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(new_price, 10))  # Prevent negative prices
            
            # Generate OHLCV data
            data = []
            for i, price in enumerate(prices):
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i-1] if i > 0 else price
                volume = np.random.randint(100000, 1000000)
                
                data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': price,
                    'Volume': volume
                })
            
            df = pd.DataFrame(data, index=dates)
            return df
            
        except Exception as e:
            # Return minimal data if error
            dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
            return pd.DataFrame({
                'Open': [1000] * 10,
                'High': [1010] * 10,
                'Low': [990] * 10,
                'Close': [1000] * 10,
                'Volume': [100000] * 10
            }, index=dates)

# Streamlit interface functions
def render_prediction_panel():
    """
    Render the prediction panel in Streamlit
    """
    st.subheader("🔮 AI Predictions & Forecasting")
    
    # Initialize prediction handler
    prediction_handler = PredictionPanel()
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_models = st.multiselect(
            "Select Models",
            list(prediction_handler.prediction_models.keys()),
            default=['LSTM', 'Random Forest', 'Ensemble']
        )
    
    with col2:
        prediction_horizon = st.selectbox(
            "Prediction Horizon",
            list(prediction_handler.prediction_horizons.keys()),
            format_func=lambda x: prediction_handler.prediction_horizons[x]['label']
        )
    
    with col3:
        theme = st.selectbox("Chart Theme", ['dark', 'light'], key='prediction_theme')
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            show_confidence_intervals = st.checkbox("Show Confidence Intervals", value=True)
            show_risk_metrics = st.checkbox("Show Risk Metrics", value=True)
        
        with col2:
            show_model_comparison = st.checkbox("Compare Models", value=True)
            generate_signals = st.checkbox("Generate Trading Signals", value=True)
    
    return {
        'prediction_handler': prediction_handler,
        'selected_models': selected_models,
        'prediction_horizon': prediction_horizon,
        'theme': theme,
        'show_confidence_intervals': show_confidence_intervals,
        'show_risk_metrics': show_risk_metrics,
        'show_model_comparison': show_model_comparison,
        'generate_signals': generate_signals
    }

def display_predictions(symbol: str, current_price: float, config: Dict):
    """
    Display prediction analysis dashboard
    """
    try:
        prediction_handler = config['prediction_handler']
        
        # Generate sample historical data
        historical_data = prediction_handler.generate_sample_historical_data(symbol)
        
        # Get prediction horizon in days
        horizon_days = prediction_handler.prediction_horizons[config['prediction_horizon']]['days']
        
        # Generate predictions for selected models
        predictions = {}
        for model in config['selected_models']:
            predictions[model] = prediction_handler.generate_sample_predictions(
                current_price, horizon_days, model
            )
        
        if predictions:
            # Display main prediction chart
            st.subheader(f"📈 Price Predictions - {config['prediction_horizon']}")
            
            # Use the first model for main chart
            main_model = list(predictions.keys())[0]
            main_prediction = predictions[main_model]
            
            prediction_fig = prediction_handler.create_prediction_chart(
                historical_data, main_prediction, symbol, config['theme']
            )
            st.plotly_chart(prediction_fig, use_container_width=True)
            
            # Display prediction summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                predicted_price = main_prediction['predictions'][-1]
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                st.metric(
                    f"Predicted Price ({config['prediction_horizon']})",
                    f"₹{predicted_price:.2f}",
                    delta=f"{price_change_pct:+.2f}%"
                )
            
            with col2:
                confidence = main_prediction['confidence']
                st.metric(
                    "Model Confidence",
                    f"{confidence:.1%}",
                    delta=f"{main_model}"
                )
            
            with col3:
                volatility = main_prediction['volatility']
                st.metric(
                    "Predicted Volatility",
                    f"{volatility:.1%}",
                    delta="Daily"
                )
            
            with col4:
                trend = "Bullish" if main_prediction['trend_strength'] > 0 else "Bearish"
                trend_emoji = "📈" if trend == "Bullish" else "📉"
                st.metric(
                    "Trend Direction",
                    f"{trend_emoji} {trend}",
                    delta=f"{main_prediction['trend_strength']:.3f}"
                )
            
            # Model comparison
            if config['show_model_comparison'] and len(predictions) > 1:
                st.subheader("🔄 Model Comparison")
                
                comparison_fig = prediction_handler.create_model_comparison_chart(
                    predictions, config['theme']
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Model summary table
                model_summary = []
                for model_name, pred_data in predictions.items():
                    final_price = pred_data['predictions'][-1]
                    change_pct = ((final_price - current_price) / current_price) * 100
                    
                    model_summary.append({
                        'Model': model_name,
                        'Predicted Price': f"₹{final_price:.2f}",
                        'Change %': f"{change_pct:+.2f}%",
                        'Confidence': f"{pred_data['confidence']:.1%}",
                        'Volatility': f"{pred_data['volatility']:.1%}"
                    })
                
                summary_df = pd.DataFrame(model_summary)
                st.dataframe(summary_df, use_container_width=True)
            
            # Trading signals
            if config['generate_signals']:
                st.subheader("📊 Trading Signals")
                
                signals_data = []
                for model_name, pred_data in predictions.items():
                    signal_info = prediction_handler.generate_trading_signals(pred_data, current_price)
                    signals_data.append({
                        'Model': model_name,
                        'Signal': signal_info['signal'],
                        'Confidence': f"{signal_info['confidence']:.1%}",
                        'Target Price': f"₹{signal_info['target_price']:.2f}",
                        'Stop Loss': f"₹{signal_info['stop_loss']:.2f}",
                        'Risk/Reward': f"{signal_info['risk_reward_ratio']:.2f}"
                    })
                
                # Display signals in columns
                cols = st.columns(len(signals_data))
                for i, signal_data in enumerate(signals_data):
                    with cols[i]:
                        signal_color = {
                            'STRONG BUY': '🟢',
                            'BUY': '🟢',
                            'HOLD': '🟡',
                            'SELL': '🔴',
                            'STRONG SELL': '🔴'
                        }.get(signal_data['Signal'], '⚪')
                        
                        st.metric(
                            f"{signal_color} {signal_data['Model']}",
                            signal_data['Signal'],
                            delta=signal_data['Confidence']
                        )
                        
                        st.write(f"**Target:** {signal_data['Target Price']}")
                        st.write(f"**Stop Loss:** {signal_data['Stop Loss']}")
                        st.write(f"**R/R Ratio:** {signal_data['Risk/Reward']}")
                
                # Signals summary table
                signals_df = pd.DataFrame(signals_data)
                st.dataframe(signals_df, use_container_width=True)
            
            # Risk metrics
            if config['show_risk_metrics']:
                st.subheader("⚠️ Risk Analysis")
                
                # Calculate returns for risk analysis
                returns = []
                for i in range(1, len(main_prediction['predictions'])):
                    ret = (main_prediction['predictions'][i] - main_prediction['predictions'][i-1]) / main_prediction['predictions'][i-1]
                    returns.append(ret)
                
                risk_metrics = prediction_handler.calculate_risk_metrics(returns)
                
                if risk_metrics:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Value at Risk (95%)",
                            f"{risk_metrics.get('VaR_95', 0):.2%}",
                            delta="Daily"
                        )
                        
                        st.metric(
                            "Maximum Drawdown",
                            f"{risk_metrics.get('Maximum_Drawdown', 0):.2%}",
                            delta="Predicted"
                        )
                    
                    with col2:
                        st.metric(
                            "Sharpe Ratio",
                            f"{risk_metrics.get('Sharpe_Ratio', 0):.2f}",
                            delta="Annualized"
                        )
                        
                        st.metric(
                            "Expected Shortfall",
                            f"{risk_metrics.get('Expected_Shortfall', 0):.2%}",
                            delta="95% Confidence"
                        )
                    
                    with col3:
                        st.metric(
                            "Volatility",
                            f"{risk_metrics.get('Volatility', 0):.1%}",
                            delta="Annualized"
                        )
                        
                        st.metric(
                            "Value at Risk (99%)",
                            f"{risk_metrics.get('VaR_99', 0):.2%}",
                            delta="Daily"
                        )
            
            # Prediction insights
            st.subheader("🔍 Prediction Insights")
            
            insights = []
            
            # Consensus analysis
            if len(predictions) > 1:
                all_final_prices = [pred['predictions'][-1] for pred in predictions.values()]
                avg_prediction = np.mean(all_final_prices)
                prediction_std = np.std(all_final_prices)
                
                consensus_change = ((avg_prediction - current_price) / current_price) * 100
                
                if prediction_std / avg_prediction < 0.05:  # Low disagreement
                    insights.append(f"📊 **Strong Model Consensus**: All models agree on direction ({consensus_change:+.1f}%)")
                else:
                    insights.append(f"⚠️ **Model Disagreement**: High variance in predictions (±{prediction_std/avg_prediction:.1%})")
            
            # Confidence analysis
            avg_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
            if avg_confidence > 0.8:
                insights.append("✅ **High Confidence**: Models show strong conviction in predictions")
            elif avg_confidence < 0.6:
                insights.append("⚠️ **Low Confidence**: Predictions should be taken with caution")
            
            # Volatility analysis
            avg_volatility = np.mean([pred['volatility'] for pred in predictions.values()])
            if avg_volatility > 0.03:
                insights.append("📈 **High Volatility**: Expect significant price swings")
            elif avg_volatility < 0.015:
                insights.append("📊 **Low Volatility**: Relatively stable price movement expected")
            
            for insight in insights:
                st.info(insight)
        
        else:
            st.warning("Please select at least one prediction model.")
    
    except Exception as e:
        st.error(f"Error displaying predictions: {e}")

# Example usage
if __name__ == "__main__":
    st.title("AI Predictions Demo")
    
    # Render prediction panel
    config = render_prediction_panel()
    
    # Sample data
    sample_symbol = "RELIANCE"
    sample_price = 2500.0
    
    display_predictions(sample_symbol, sample_price, config)