#!/usr/bin/env python3
"""
Main Dashboard Controller - Indian Market Research Platform
Orchestrates all dashboard components and handles user interactions
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import threading
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dashboard components
from dashboard.components.price_charts import PriceCharts, render_price_charts_panel
from dashboard.components.volatility_charts import VolatilityCharts, render_volatility_charts_panel
from dashboard.components.technical_indicators_panel import TechnicalIndicatorsPanel, render_technical_indicators_panel
from dashboard.components.market_sentiment import MarketSentimentAnalyzer, render_market_sentiment_panel
from dashboard.components.prediction_panel import PredictionPanel, render_prediction_panel
from dashboard.components.alert_system import AlertSystem, render_alert_system_panel

# Import data handlers and models
from indian_markets.nse_data_handler import NSEDataHandler
from data_simulation.indian_stock_simulator import IndianStockSimulator
from models.lstm_predictor import LSTMPredictor
from models.volatility_predictor import VolatilityPredictor
from models.sentiment_analyzer import SentimentAnalyzer
from models.pattern_recognition import PatternRecognition

class MainDashboard:
    """
    Main dashboard controller that manages all components and data flow
    """
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_data_handlers()
        self.setup_models()
        self.setup_components()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.selected_symbol = 'RELIANCE'
            st.session_state.selected_timeframe = '1D'
            st.session_state.watchlist = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
            st.session_state.market_data = {}
            st.session_state.predictions = {}
            st.session_state.alerts = []
            st.session_state.last_update = datetime.now()
            st.session_state.auto_refresh = True
            st.session_state.theme = 'dark'
            st.session_state.layout_mode = 'standard'
            
    def setup_data_handlers(self):
        """Initialize data handlers"""
        try:
            self.nse_handler = NSEDataHandler()
            self.stock_simulator = IndianStockSimulator()
            
            # Generate initial market data
            self.generate_sample_data()
            
        except Exception as e:
            st.error(f"Error initializing data handlers: {e}")
            # Fallback to simulation only
            self.nse_handler = None
            self.stock_simulator = IndianStockSimulator()
            
    def setup_models(self):
        """Initialize ML models"""
        try:
            self.lstm_predictor = LSTMPredictor()
            self.volatility_predictor = VolatilityPredictor()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.pattern_recognition = PatternRecognition()
            
        except Exception as e:
            st.error(f"Error initializing models: {e}")
            self.lstm_predictor = None
            self.volatility_predictor = None
            self.sentiment_analyzer = None
            self.pattern_recognition = None
            
    def setup_components(self):
        """Initialize dashboard components"""
        self.price_charts = PriceCharts()
        self.volatility_charts = VolatilityCharts()
        self.technical_indicators = TechnicalIndicatorsPanel()
        self.market_sentiment = MarketSentimentAnalyzer()
        self.prediction_panel = PredictionPanel()
        self.alert_system = AlertSystem()
        
    def generate_sample_data(self):
        """Generate sample market data for demonstration"""
        symbols = st.session_state.watchlist
        
        for symbol in symbols:
            # Generate historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            data = self.stock_simulator.generate_daily_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                base_price=np.random.uniform(100, 3000)
            )
            
            st.session_state.market_data[symbol] = data
    
    def configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Indian Market Research Platform",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/market-research-platform',
                'Report a bug': 'https://github.com/your-repo/market-research-platform/issues',
                'About': 'Indian Market Research Platform - Advanced Analytics & Predictions'
            }
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-open {
            background-color: #00d4aa;
            animation: pulse 2s infinite;
        }
        
        .status-closed {
            background-color: #ff4757;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .alert-high {
            background-color: #ff4757;
            color: white;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.2rem 0;
        }
        
        .alert-medium {
            background-color: #ffa502;
            color: white;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.2rem 0;
        }
        
        .alert-low {
            background-color: #2ed573;
            color: white;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.2rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header with market status and indices"""
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        with col1:
            st.markdown("### 🇮🇳 Indian Market Research Platform")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
            st.markdown(f"**Current Time:** {current_time}")
        
        # Get real-time index data
        indices_data = self.get_major_indices()
        
        with col2:
            nifty_data = indices_data.get('NIFTY', {'price': 0, 'change': 0})
            st.metric(
                "NIFTY 50",
                f"₹{nifty_data['price']:,.2f}",
                f"{nifty_data['change']:+.2f} ({nifty_data['change_pct']:+.2f}%)"
            )
        
        with col3:
            sensex_data = indices_data.get('SENSEX', {'price': 0, 'change': 0})
            st.metric(
                "SENSEX",
                f"₹{sensex_data['price']:,.2f}",
                f"{sensex_data['change']:+.2f} ({sensex_data['change_pct']:+.2f}%)"
            )
        
        with col4:
            banknifty_data = indices_data.get('BANKNIFTY', {'price': 0, 'change': 0})
            st.metric(
                "BANK NIFTY",
                f"₹{banknifty_data['price']:,.2f}",
                f"{banknifty_data['change']:+.2f} ({banknifty_data['change_pct']:+.2f}%)"
            )
        
        with col5:
            market_status = self.get_market_status()
            status_color = "🟢" if market_status == "OPEN" else "🔴"
            st.markdown(f"**Market Status**")
            st.markdown(f"{status_color} {market_status}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the left sidebar with stock selection and watchlist"""
        st.sidebar.markdown("## 📊 Stock Selection")
        
        # Stock search and selection
        popular_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS'
        ]
        
        selected_stock = st.sidebar.selectbox(
            "Select Stock",
            popular_stocks,
            index=popular_stocks.index(st.session_state.selected_stock) if st.session_state.selected_stock in popular_stocks else 0
        )
        
        if selected_stock != st.session_state.selected_stock:
            st.session_state.selected_stock = selected_stock
            st.rerun()
        
        # Timeframe selection
        timeframes = {
            '1m': '1 Minute',
            '5m': '5 Minutes',
            '15m': '15 Minutes',
            '1h': '1 Hour',
            '1d': '1 Day',
            '1wk': '1 Week'
        }
        
        selected_timeframe = st.sidebar.selectbox(
            "Timeframe",
            list(timeframes.keys()),
            format_func=lambda x: timeframes[x],
            index=list(timeframes.keys()).index(st.session_state.timeframe)
        )
        
        if selected_timeframe != st.session_state.timeframe:
            st.session_state.timeframe = selected_timeframe
            st.rerun()
        
        # Watchlist
        st.sidebar.markdown("## 👀 Watchlist")
        watchlist_data = self.get_watchlist_data()
        
        for stock, data in watchlist_data.items():
            change_color = "green" if data['change'] >= 0 else "red"
            st.sidebar.markdown(
                f"**{stock}**: ₹{data['price']:.2f} "
                f"<span style='color:{change_color}'>({data['change']:+.2f}%)</span>",
                unsafe_allow_html=True
            )
        
        # Sector Performance
        st.sidebar.markdown("## 🏭 Sector Performance")
        sector_data = self.get_sector_performance()
        
        for sector, performance in sector_data.items():
            color = "green" if performance >= 0 else "red"
            st.sidebar.markdown(
                f"**{sector}**: <span style='color:{color}'>{performance:+.2f}%</span>",
                unsafe_allow_html=True
            )
    
    def render_main_content(self):
        """Render the main content area with charts and analysis"""
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "📊 Technical Indicators", "🔮 Predictions", "⚠️ Alerts"])
        
        with tab1:
            self.render_price_analysis()
        
        with tab2:
            self.render_technical_analysis()
        
        with tab3:
            self.render_predictions()
        
        with tab4:
            self.render_alerts()
    
    def render_price_analysis(self):
        """Render price charts and volume analysis"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Main price chart
            st.markdown("### 📈 Price Chart")
            price_data = self.get_stock_data(st.session_state.selected_stock, st.session_state.timeframe)
            
            if not price_data.empty:
                fig = self.create_candlestick_chart(price_data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected stock and timeframe.")
        
        with col2:
            # Stock info and metrics
            st.markdown("### 📊 Stock Metrics")
            stock_info = self.get_stock_info(st.session_state.selected_stock)
            
            if stock_info:
                st.metric("Current Price", f"₹{stock_info['current_price']:.2f}")
                st.metric("Day Change", f"{stock_info['day_change']:+.2f}%")
                st.metric("Volume", f"{stock_info['volume']:,}")
                st.metric("Market Cap", f"₹{stock_info['market_cap']:.2f}Cr")
                st.metric("P/E Ratio", f"{stock_info['pe_ratio']:.2f}")
    
    def render_technical_analysis(self):
        """Render technical indicators and analysis"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Technical Indicators")
            
            # RSI
            rsi_data = self.calculate_rsi(st.session_state.selected_stock)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data['RSI'], name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(title="RSI (14)", yaxis_title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 MACD")
            
            # MACD
            macd_data = self.calculate_macd(st.session_state.selected_stock)
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=macd_data.index, y=macd_data['MACD'], name='MACD'))
            fig_macd.add_trace(go.Scatter(x=macd_data.index, y=macd_data['Signal'], name='Signal'))
            fig_macd.add_bar(x=macd_data.index, y=macd_data['Histogram'], name='Histogram')
            fig_macd.update_layout(title="MACD")
            st.plotly_chart(fig_macd, use_container_width=True)
    
    def render_predictions(self):
        """Render ML predictions and forecasts"""
        st.markdown("### 🔮 AI Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Short-term (1-5 days)")
            short_term_pred = self.get_short_term_prediction(st.session_state.selected_stock)
            
            for day, pred in short_term_pred.items():
                direction = "📈" if pred['direction'] == 'up' else "📉"
                confidence = pred['confidence']
                st.markdown(f"**{day}**: {direction} {pred['price']:.2f} (Confidence: {confidence:.1f}%)")
        
        with col2:
            st.markdown("#### Medium-term (1-4 weeks)")
            medium_term_pred = self.get_medium_term_prediction(st.session_state.selected_stock)
            
            for week, pred in medium_term_pred.items():
                direction = "📈" if pred['direction'] == 'up' else "📉"
                st.markdown(f"**{week}**: {direction} {pred['target']:.2f}")
        
        with col3:
            st.markdown("#### Risk Assessment")
            risk_metrics = self.get_risk_assessment(st.session_state.selected_stock)
            
            st.metric("Volatility", f"{risk_metrics['volatility']:.2f}%")
            st.metric("Beta", f"{risk_metrics['beta']:.2f}")
            st.metric("VaR (95%)", f"{risk_metrics['var']:.2f}%")
            st.metric("Sharpe Ratio", f"{risk_metrics['sharpe']:.2f}")
    
    def render_alerts(self):
        """Render alerts and notifications"""
        st.markdown("### ⚠️ Market Alerts")
        
        alerts = self.alert_system.get_active_alerts(st.session_state.selected_stock)
        
        if alerts:
            for alert in alerts:
                alert_type = alert['type']
                message = alert['message']
                timestamp = alert['timestamp']
                
                if alert_type == 'positive':
                    st.markdown(
                        f'<div class="alert-positive"><strong>🟢 {timestamp}</strong><br>{message}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="alert-negative"><strong>🔴 {timestamp}</strong><br>{message}</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No active alerts for the selected stock.")
    
    def get_major_indices(self) -> Dict:
        """Get real-time data for major Indian indices"""
        try:
            # Simulate real-time data (in production, use actual API)
            indices = {
                'NIFTY': {
                    'price': 19500 + np.random.normal(0, 50),
                    'change': np.random.normal(0, 100),
                    'change_pct': np.random.normal(0, 0.5)
                },
                'SENSEX': {
                    'price': 65000 + np.random.normal(0, 200),
                    'change': np.random.normal(0, 300),
                    'change_pct': np.random.normal(0, 0.5)
                },
                'BANKNIFTY': {
                    'price': 44000 + np.random.normal(0, 100),
                    'change': np.random.normal(0, 200),
                    'change_pct': np.random.normal(0, 0.7)
                }
            }
            return indices
        except Exception as e:
            st.error(f"Error fetching indices data: {e}")
            return {}
    
    def get_market_status(self) -> str:
        """Determine if market is open or closed"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Check if it's a weekday and within market hours
        if now.weekday() < 5 and market_open <= now <= market_close:
            return "OPEN"
        else:
            return "CLOSED"
    
    def get_stock_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get stock price data"""
        try:
            # Use the stock simulator for realistic data
            return self.stock_simulator.generate_stock_data(symbol, timeframe)
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def create_candlestick_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create candlestick chart with volume"""
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        # Add volume as subplot
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{st.session_state.selected_stock} - {st.session_state.timeframe}",
            yaxis_title="Price (₹)",
            yaxis2=dict(title="Volume", overlaying='y', side='right'),
            xaxis_title="Time",
            height=600
        )
        
        return fig
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get basic stock information"""
        try:
            # Simulate stock info (in production, use actual API)
            base_price = 1000 + np.random.normal(0, 500)
            return {
                'current_price': base_price,
                'day_change': np.random.normal(0, 2),
                'volume': int(np.random.normal(1000000, 500000)),
                'market_cap': base_price * 100000 / 10000000,  # In crores
                'pe_ratio': np.random.normal(20, 5)
            }
        except Exception as e:
            st.error(f"Error fetching stock info: {e}")
            return {}
    
    def get_watchlist_data(self) -> Dict:
        """Get watchlist stock data"""
        watchlist = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        data = {}
        
        for stock in watchlist:
            data[stock] = {
                'price': 1000 + np.random.normal(0, 500),
                'change': np.random.normal(0, 3)
            }
        
        return data
    
    def get_sector_performance(self) -> Dict:
        """Get sector performance data"""
        sectors = ['Banking', 'IT', 'Pharma', 'Auto', 'FMCG', 'Energy']
        return {sector: np.random.normal(0, 2) for sector in sectors}
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> pd.DataFrame:
        """Calculate RSI indicator"""
        data = self.get_stock_data(symbol, '1d')
        if data.empty:
            return pd.DataFrame()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return pd.DataFrame({'RSI': rsi})
    
    def calculate_macd(self, symbol: str) -> pd.DataFrame:
        """Calculate MACD indicator"""
        data = self.get_stock_data(symbol, '1d')
        if data.empty:
            return pd.DataFrame()
        
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': histogram
        })
    
    def get_short_term_prediction(self, symbol: str) -> Dict:
        """Get short-term predictions"""
        predictions = {}
        base_price = 1000 + np.random.normal(0, 500)
        
        for i in range(1, 6):
            direction = 'up' if np.random.random() > 0.5 else 'down'
            change = np.random.normal(0, 2)
            price = base_price * (1 + change/100)
            confidence = np.random.uniform(60, 95)
            
            predictions[f"Day {i}"] = {
                'direction': direction,
                'price': price,
                'confidence': confidence
            }
        
        return predictions
    
    def get_medium_term_prediction(self, symbol: str) -> Dict:
        """Get medium-term predictions"""
        predictions = {}
        base_price = 1000 + np.random.normal(0, 500)
        
        for i in range(1, 5):
            direction = 'up' if np.random.random() > 0.5 else 'down'
            change = np.random.normal(0, 5)
            target = base_price * (1 + change/100)
            
            predictions[f"Week {i}"] = {
                'direction': direction,
                'target': target
            }
        
        return predictions
    
    def get_risk_assessment(self, symbol: str) -> Dict:
        """Get risk assessment metrics"""
        return {
            'volatility': np.random.uniform(15, 45),
            'beta': np.random.uniform(0.5, 2.0),
            'var': np.random.uniform(2, 8),
            'sharpe': np.random.uniform(0.5, 2.5)
        }
    
    def run(self):
        """Main dashboard execution"""
        try:
            # Configure page
            self.configure_page()
            
            # Render header
            self.render_header()
            
            # Render sidebar
            self.render_sidebar()
            
            # Render main content
            self.render_main_content()
            
            # Auto refresh
            if st.session_state.auto_refresh:
                time.sleep(5)  # Wait 5 seconds
                st.rerun()
                
        except Exception as e:
            st.error(f"Dashboard error: {e}")
            st.exception(e)

def main():
    """Main entry point"""
    dashboard = MainDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()