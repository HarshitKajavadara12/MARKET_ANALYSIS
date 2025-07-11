import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')
import json
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class AlertType(Enum):
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    NEWS = "news"
    PATTERN = "pattern"
    RISK = "risk"
    EARNINGS = "earnings"
    REGULATORY = "regulatory"

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    DISABLED = "disabled"

@dataclass
class Alert:
    id: str
    symbol: str
    alert_type: AlertType
    priority: AlertPriority
    status: AlertStatus
    title: str
    description: str
    condition: Dict[str, Any]
    created_at: datetime
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    notification_sent: bool = False
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AlertSystem:
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        
        self.alert_templates = {
            'price_above': {
                'title': 'Price Above Target',
                'description': 'Stock price has moved above the specified level',
                'condition_fields': ['target_price']
            },
            'price_below': {
                'title': 'Price Below Target',
                'description': 'Stock price has moved below the specified level',
                'condition_fields': ['target_price']
            },
            'price_change': {
                'title': 'Price Change Alert',
                'description': 'Stock price has changed by specified percentage',
                'condition_fields': ['change_percent', 'timeframe']
            },
            'volume_spike': {
                'title': 'Volume Spike',
                'description': 'Trading volume has exceeded normal levels',
                'condition_fields': ['volume_multiplier', 'avg_period']
            },
            'rsi_overbought': {
                'title': 'RSI Overbought',
                'description': 'RSI has moved into overbought territory',
                'condition_fields': ['rsi_threshold']
            },
            'rsi_oversold': {
                'title': 'RSI Oversold',
                'description': 'RSI has moved into oversold territory',
                'condition_fields': ['rsi_threshold']
            },
            'macd_crossover': {
                'title': 'MACD Crossover',
                'description': 'MACD line has crossed the signal line',
                'condition_fields': ['crossover_type']
            },
            'bollinger_breach': {
                'title': 'Bollinger Band Breach',
                'description': 'Price has breached Bollinger Band levels',
                'condition_fields': ['band_type']
            },
            'support_resistance': {
                'title': 'Support/Resistance Break',
                'description': 'Price has broken key support or resistance level',
                'condition_fields': ['level_price', 'break_type']
            },
            'pattern_detected': {
                'title': 'Chart Pattern Detected',
                'description': 'Technical chart pattern has been identified',
                'condition_fields': ['pattern_type', 'confidence']
            }
        }
        
        self.chart_themes = {
            'dark': {
                'bg_color': '#1e1e1e',
                'grid_color': '#404040',
                'text_color': '#ffffff',
                'alert_colors': {
                    'critical': '#ff1744',
                    'high': '#ff5722',
                    'medium': '#ff9800',
                    'low': '#4caf50'
                }
            },
            'light': {
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'alert_colors': {
                    'critical': '#d32f2f',
                    'high': '#f57c00',
                    'medium': '#ffa000',
                    'low': '#388e3c'
                }
            }
        }
    
    def create_alert(self, 
                    symbol: str,
                    alert_type: AlertType,
                    priority: AlertPriority,
                    title: str,
                    description: str,
                    condition: Dict[str, Any],
                    expires_in_hours: Optional[int] = None) -> str:
        """Create a new alert"""
        try:
            alert_id = str(uuid.uuid4())
            
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.now() + timedelta(hours=expires_in_hours)
            
            alert = Alert(
                id=alert_id,
                symbol=symbol,
                alert_type=alert_type,
                priority=priority,
                status=AlertStatus.ACTIVE,
                title=title,
                description=description,
                condition=condition,
                created_at=datetime.now(),
                expires_at=expires_at
            )
            
            self.alerts.append(alert)
            return alert_id
            
        except Exception as e:
            return ""
    
    def check_price_alerts(self, symbol: str, current_price: float, previous_price: float) -> List[Alert]:
        """Check price-based alerts"""
        triggered_alerts = []
        
        try:
            for alert in self.alerts:
                if (alert.symbol == symbol and 
                    alert.status == AlertStatus.ACTIVE and
                    alert.alert_type == AlertType.PRICE):
                    
                    condition = alert.condition
                    
                    # Price above target
                    if 'target_price' in condition and 'direction' in condition:
                        target = condition['target_price']
                        direction = condition['direction']
                        
                        if direction == 'above' and current_price >= target and previous_price < target:
                            alert.status = AlertStatus.TRIGGERED
                            alert.triggered_at = datetime.now()
                            triggered_alerts.append(alert)
                        
                        elif direction == 'below' and current_price <= target and previous_price > target:
                            alert.status = AlertStatus.TRIGGERED
                            alert.triggered_at = datetime.now()
                            triggered_alerts.append(alert)
                    
                    # Price change percentage
                    if 'change_percent' in condition:
                        change_threshold = condition['change_percent']
                        actual_change = ((current_price - previous_price) / previous_price) * 100
                        
                        if abs(actual_change) >= change_threshold:
                            alert.status = AlertStatus.TRIGGERED
                            alert.triggered_at = datetime.now()
                            alert.metadata = {'actual_change': actual_change}
                            triggered_alerts.append(alert)
            
            return triggered_alerts
            
        except Exception as e:
            return []
    
    def check_technical_alerts(self, symbol: str, technical_data: Dict[str, float]) -> List[Alert]:
        """Check technical indicator alerts"""
        triggered_alerts = []
        
        try:
            for alert in self.alerts:
                if (alert.symbol == symbol and 
                    alert.status == AlertStatus.ACTIVE and
                    alert.alert_type == AlertType.TECHNICAL):
                    
                    condition = alert.condition
                    
                    # RSI alerts
                    if 'rsi_threshold' in condition and 'rsi' in technical_data:
                        threshold = condition['rsi_threshold']
                        current_rsi = technical_data['rsi']
                        alert_type = condition.get('rsi_type', 'overbought')
                        
                        if alert_type == 'overbought' and current_rsi >= threshold:
                            alert.status = AlertStatus.TRIGGERED
                            alert.triggered_at = datetime.now()
                            alert.metadata = {'rsi_value': current_rsi}
                            triggered_alerts.append(alert)
                        
                        elif alert_type == 'oversold' and current_rsi <= threshold:
                            alert.status = AlertStatus.TRIGGERED
                            alert.triggered_at = datetime.now()
                            alert.metadata = {'rsi_value': current_rsi}
                            triggered_alerts.append(alert)
                    
                    # MACD crossover alerts
                    if 'crossover_type' in condition and 'macd' in technical_data and 'macd_signal' in technical_data:
                        crossover_type = condition['crossover_type']
                        macd = technical_data['macd']
                        macd_signal = technical_data['macd_signal']
                        prev_macd = technical_data.get('prev_macd', macd)
                        prev_signal = technical_data.get('prev_macd_signal', macd_signal)
                        
                        # Bullish crossover
                        if (crossover_type == 'bullish' and 
                            macd > macd_signal and prev_macd <= prev_signal):
                            alert.status = AlertStatus.TRIGGERED
                            alert.triggered_at = datetime.now()
                            triggered_alerts.append(alert)
                        
                        # Bearish crossover
                        elif (crossover_type == 'bearish' and 
                              macd < macd_signal and prev_macd >= prev_signal):
                            alert.status = AlertStatus.TRIGGERED
                            alert.triggered_at = datetime.now()
                            triggered_alerts.append(alert)
            
            return triggered_alerts
            
        except Exception as e:
            return []
    
    def check_volume_alerts(self, symbol: str, current_volume: float, avg_volume: float) -> List[Alert]:
        """Check volume-based alerts"""
        triggered_alerts = []
        
        try:
            for alert in self.alerts:
                if (alert.symbol == symbol and 
                    alert.status == AlertStatus.ACTIVE and
                    alert.alert_type == AlertType.VOLUME):
                    
                    condition = alert.condition
                    
                    if 'volume_multiplier' in condition:
                        multiplier = condition['volume_multiplier']
                        
                        if current_volume >= avg_volume * multiplier:
                            alert.status = AlertStatus.TRIGGERED
                            alert.triggered_at = datetime.now()
                            alert.metadata = {
                                'current_volume': current_volume,
                                'avg_volume': avg_volume,
                                'multiplier_achieved': current_volume / avg_volume
                            }
                            triggered_alerts.append(alert)
            
            return triggered_alerts
            
        except Exception as e:
            return []
    
    def check_pattern_alerts(self, symbol: str, detected_patterns: List[Dict[str, Any]]) -> List[Alert]:
        """Check pattern detection alerts"""
        triggered_alerts = []
        
        try:
            for alert in self.alerts:
                if (alert.symbol == symbol and 
                    alert.status == AlertStatus.ACTIVE and
                    alert.alert_type == AlertType.PATTERN):
                    
                    condition = alert.condition
                    target_pattern = condition.get('pattern_type', '')
                    min_confidence = condition.get('confidence', 0.7)
                    
                    for pattern in detected_patterns:
                        if (pattern.get('type', '') == target_pattern and 
                            pattern.get('confidence', 0) >= min_confidence):
                            
                            alert.status = AlertStatus.TRIGGERED
                            alert.triggered_at = datetime.now()
                            alert.metadata = pattern
                            triggered_alerts.append(alert)
                            break
            
            return triggered_alerts
            
        except Exception as e:
            return []
    
    def expire_old_alerts(self):
        """Mark expired alerts as expired"""
        try:
            current_time = datetime.now()
            
            for alert in self.alerts:
                if (alert.expires_at and 
                    current_time > alert.expires_at and 
                    alert.status == AlertStatus.ACTIVE):
                    alert.status = AlertStatus.EXPIRED
        
        except Exception as e:
            pass
    
    def get_active_alerts(self, symbol: Optional[str] = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by symbol"""
        try:
            active_alerts = [alert for alert in self.alerts if alert.status == AlertStatus.ACTIVE]
            
            if symbol:
                active_alerts = [alert for alert in active_alerts if alert.symbol == symbol]
            
            return active_alerts
        
        except Exception as e:
            return []
    
    def get_triggered_alerts(self, hours_back: int = 24) -> List[Alert]:
        """Get recently triggered alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            triggered_alerts = [
                alert for alert in self.alerts 
                if (alert.status == AlertStatus.TRIGGERED and 
                    alert.triggered_at and 
                    alert.triggered_at >= cutoff_time)
            ]
            
            return sorted(triggered_alerts, key=lambda x: x.triggered_at, reverse=True)
        
        except Exception as e:
            return []
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert"""
        try:
            self.alerts = [alert for alert in self.alerts if alert.id != alert_id]
            return True
        except Exception as e:
            return False
    
    def toggle_alert(self, alert_id: str) -> bool:
        """Toggle alert between active and disabled"""
        try:
            for alert in self.alerts:
                if alert.id == alert_id:
                    if alert.status == AlertStatus.ACTIVE:
                        alert.status = AlertStatus.DISABLED
                    elif alert.status == AlertStatus.DISABLED:
                        alert.status = AlertStatus.ACTIVE
                    return True
            return False
        except Exception as e:
            return False
    
    def create_alerts_dashboard(self, theme: str = 'dark') -> go.Figure:
        """Create alerts dashboard visualization"""
        try:
            colors = self.chart_themes[theme]
            
            # Count alerts by priority and status
            alert_counts = {
                'Active': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                'Triggered': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                'Expired': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            }
            
            for alert in self.alerts:
                status_key = alert.status.value.title()
                priority_key = alert.priority.value
                
                if status_key in alert_counts and priority_key in alert_counts[status_key]:
                    alert_counts[status_key][priority_key] += 1
            
            # Create stacked bar chart
            fig = go.Figure()
            
            statuses = list(alert_counts.keys())
            priorities = ['critical', 'high', 'medium', 'low']
            
            for priority in priorities:
                values = [alert_counts[status][priority] for status in statuses]
                
                fig.add_trace(go.Bar(
                    name=priority.title(),
                    x=statuses,
                    y=values,
                    marker_color=colors['alert_colors'][priority]
                ))
            
            fig.update_layout(
                title='Alert Status Overview',
                xaxis_title='Alert Status',
                yaxis_title='Number of Alerts',
                barmode='stack',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=400,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            return go.Figure()
    
    def create_alert_timeline(self, hours_back: int = 24, theme: str = 'dark') -> go.Figure:
        """Create timeline of triggered alerts"""
        try:
            colors = self.chart_themes[theme]
            
            triggered_alerts = self.get_triggered_alerts(hours_back)
            
            if not triggered_alerts:
                return go.Figure()
            
            # Prepare data
            times = [alert.triggered_at for alert in triggered_alerts]
            symbols = [alert.symbol for alert in triggered_alerts]
            titles = [alert.title for alert in triggered_alerts]
            priorities = [alert.priority.value for alert in triggered_alerts]
            
            # Map priorities to colors
            priority_colors = [colors['alert_colors'][priority] for priority in priorities]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=times,
                y=symbols,
                mode='markers',
                marker=dict(
                    size=12,
                    color=priority_colors,
                    line=dict(width=2, color=colors['text_color'])
                ),
                text=titles,
                hovertemplate='<b>%{text}</b><br>Symbol: %{y}<br>Time: %{x}<extra></extra>',
                name='Triggered Alerts'
            ))
            
            fig.update_layout(
                title=f'Alert Timeline (Last {hours_back} Hours)',
                xaxis_title='Time',
                yaxis_title='Symbol',
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                height=400,
                plot_bgcolor=colors['bg_color'],
                paper_bgcolor=colors['bg_color'],
                font=dict(color=colors['text_color'])
            )
            
            return fig
            
        except Exception as e:
            return go.Figure()
    
    def generate_sample_alerts(self):
        """Generate sample alerts for demonstration"""
        try:
            sample_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'SBIN', 'ITC', 'HDFCBANK']
            
            # Clear existing alerts
            self.alerts = []
            
            # Create various types of alerts
            for i, symbol in enumerate(sample_symbols[:5]):
                # Price alerts
                self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.PRICE,
                    priority=AlertPriority.HIGH,
                    title=f"{symbol} Price Above ₹2500",
                    description=f"Alert when {symbol} price moves above ₹2500",
                    condition={'target_price': 2500, 'direction': 'above'},
                    expires_in_hours=24
                )
                
                # Technical alerts
                self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.TECHNICAL,
                    priority=AlertPriority.MEDIUM,
                    title=f"{symbol} RSI Overbought",
                    description=f"Alert when {symbol} RSI goes above 70",
                    condition={'rsi_threshold': 70, 'rsi_type': 'overbought'},
                    expires_in_hours=48
                )
                
                # Volume alerts
                if i < 3:
                    self.create_alert(
                        symbol=symbol,
                        alert_type=AlertType.VOLUME,
                        priority=AlertPriority.LOW,
                        title=f"{symbol} Volume Spike",
                        description=f"Alert when {symbol} volume exceeds 2x average",
                        condition={'volume_multiplier': 2.0},
                        expires_in_hours=12
                    )
            
            # Simulate some triggered alerts
            current_time = datetime.now()
            for i in range(3):
                alert = self.alerts[i]
                alert.status = AlertStatus.TRIGGERED
                alert.triggered_at = current_time - timedelta(hours=i+1)
            
        except Exception as e:
            pass

# Streamlit interface functions
def render_alert_system_panel():
    """
    Render the alert system panel in Streamlit
    """
    st.subheader("🚨 Alert System")
    
    # Initialize alert system
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = AlertSystem()
        st.session_state.alert_system.generate_sample_alerts()
    
    alert_system = st.session_state.alert_system
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("Chart Theme", ['dark', 'light'], key='alert_theme')
        show_timeline = st.checkbox("Show Alert Timeline", value=True)
    
    with col2:
        auto_refresh = st.checkbox("Auto Refresh Alerts", value=False)
        show_expired = st.checkbox("Show Expired Alerts", value=False)
    
    return {
        'alert_system': alert_system,
        'theme': theme,
        'show_timeline': show_timeline,
        'auto_refresh': auto_refresh,
        'show_expired': show_expired
    }

def display_alert_system(config: Dict):
    """
    Display alert system dashboard
    """
    try:
        alert_system = config['alert_system']
        
        # Expire old alerts
        alert_system.expire_old_alerts()
        
        # Alert summary metrics
        active_alerts = alert_system.get_active_alerts()
        triggered_alerts = alert_system.get_triggered_alerts(24)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🟢 Active Alerts",
                len(active_alerts),
                delta="Currently monitoring"
            )
        
        with col2:
            st.metric(
                "🔴 Triggered (24h)",
                len(triggered_alerts),
                delta="Recent alerts"
            )
        
        with col3:
            critical_alerts = [a for a in active_alerts if a.priority == AlertPriority.CRITICAL]
            st.metric(
                "⚠️ Critical Alerts",
                len(critical_alerts),
                delta="High priority"
            )
        
        with col4:
            expired_alerts = [a for a in alert_system.alerts if a.status == AlertStatus.EXPIRED]
            st.metric(
                "⏰ Expired Alerts",
                len(expired_alerts),
                delta="Need cleanup"
            )
        
        # Alert dashboard visualization
        st.subheader("📊 Alert Overview")
        
        dashboard_fig = alert_system.create_alerts_dashboard(config['theme'])
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Alert timeline
        if config['show_timeline']:
            st.subheader("⏰ Recent Alert Timeline")
            
            timeline_fig = alert_system.create_alert_timeline(24, config['theme'])
            if timeline_fig.data:
                st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.info("No alerts triggered in the last 24 hours")
        
        # Active alerts management
        st.subheader("🔧 Alert Management")
        
        tab1, tab2, tab3 = st.tabs(["Active Alerts", "Create New Alert", "Alert History"])
        
        with tab1:
            if active_alerts:
                for alert in active_alerts:
                    with st.expander(f"{alert.title} - {alert.symbol}"):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**Description:** {alert.description}")
                            st.write(f"**Type:** {alert.alert_type.value.title()}")
                            st.write(f"**Priority:** {alert.priority.value.title()}")
                            st.write(f"**Created:** {alert.created_at.strftime('%Y-%m-%d %H:%M')}")
                            if alert.expires_at:
                                st.write(f"**Expires:** {alert.expires_at.strftime('%Y-%m-%d %H:%M')}")
                        
                        with col2:
                            if st.button(f"Toggle", key=f"toggle_{alert.id}"):
                                alert_system.toggle_alert(alert.id)
                                st.rerun()
                        
                        with col3:
                            if st.button(f"Delete", key=f"delete_{alert.id}"):
                                alert_system.delete_alert(alert.id)
                                st.rerun()
                        
                        # Show condition details
                        st.json(alert.condition)
            else:
                st.info("No active alerts. Create some alerts to monitor your stocks!")
        
        with tab2:
            st.write("**Create New Alert**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_symbol = st.text_input("Stock Symbol", value="RELIANCE")
                alert_type = st.selectbox(
                    "Alert Type",
                    [AlertType.PRICE, AlertType.TECHNICAL, AlertType.VOLUME, AlertType.PATTERN],
                    format_func=lambda x: x.value.title()
                )
                priority = st.selectbox(
                    "Priority",
                    [AlertPriority.LOW, AlertPriority.MEDIUM, AlertPriority.HIGH, AlertPriority.CRITICAL],
                    format_func=lambda x: x.value.title()
                )
            
            with col2:
                title = st.text_input("Alert Title")
                description = st.text_area("Description")
                expires_hours = st.number_input("Expires in hours (0 = never)", min_value=0, value=24)
            
            # Dynamic condition fields based on alert type
            st.write("**Alert Conditions:**")
            condition = {}
            
            if alert_type == AlertType.PRICE:
                col1, col2 = st.columns(2)
                with col1:
                    target_price = st.number_input("Target Price (₹)", min_value=0.0, value=1000.0)
                    condition['target_price'] = target_price
                with col2:
                    direction = st.selectbox("Direction", ['above', 'below'])
                    condition['direction'] = direction
            
            elif alert_type == AlertType.TECHNICAL:
                indicator = st.selectbox("Technical Indicator", ['RSI', 'MACD', 'Bollinger Bands'])
                
                if indicator == 'RSI':
                    rsi_threshold = st.number_input("RSI Threshold", min_value=0.0, max_value=100.0, value=70.0)
                    rsi_type = st.selectbox("Alert Type", ['overbought', 'oversold'])
                    condition.update({'rsi_threshold': rsi_threshold, 'rsi_type': rsi_type})
                
                elif indicator == 'MACD':
                    crossover_type = st.selectbox("Crossover Type", ['bullish', 'bearish'])
                    condition['crossover_type'] = crossover_type
            
            elif alert_type == AlertType.VOLUME:
                volume_multiplier = st.number_input("Volume Multiplier", min_value=1.0, value=2.0)
                condition['volume_multiplier'] = volume_multiplier
            
            elif alert_type == AlertType.PATTERN:
                pattern_type = st.selectbox("Pattern Type", 
                    ['Head and Shoulders', 'Double Top', 'Double Bottom', 'Triangle', 'Flag'])
                confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.7)
                condition.update({'pattern_type': pattern_type, 'confidence': confidence})
            
            if st.button("Create Alert"):
                if title and description and new_symbol:
                    alert_id = alert_system.create_alert(
                        symbol=new_symbol,
                        alert_type=alert_type,
                        priority=priority,
                        title=title,
                        description=description,
                        condition=condition,
                        expires_in_hours=expires_hours if expires_hours > 0 else None
                    )
                    
                    if alert_id:
                        st.success(f"Alert created successfully! ID: {alert_id[:8]}...")
                        st.rerun()
                    else:
                        st.error("Failed to create alert")
                else:
                    st.error("Please fill in all required fields")
        
        with tab3:
            st.write("**Recent Alert History**")
            
            if triggered_alerts:
                for alert in triggered_alerts[:10]:  # Show last 10
                    priority_emoji = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢'
                    }.get(alert.priority.value, '⚪')
                    
                    with st.expander(f"{priority_emoji} {alert.title} - {alert.symbol}"):
                        st.write(f"**Triggered:** {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Description:** {alert.description}")
                        st.write(f"**Type:** {alert.alert_type.value.title()}")
                        st.write(f"**Priority:** {alert.priority.value.title()}")
                        
                        if alert.metadata:
                            st.write("**Additional Data:**")
                            st.json(alert.metadata)
            else:
                st.info("No recent alert history")
        
        # Auto-refresh functionality
        if config['auto_refresh']:
            st.info("🔄 Auto-refresh enabled - Alerts are being monitored in real-time")
            # In a real implementation, you would use st.rerun() with a timer
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Alerts"):
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Expired"):
                alert_system.alerts = [a for a in alert_system.alerts if a.status != AlertStatus.EXPIRED]
                st.success("Expired alerts cleared!")
                st.rerun()
        
        with col3:
            if st.button("📊 Generate Sample Data"):
                alert_system.generate_sample_alerts()
                st.success("Sample alerts generated!")
                st.rerun()
    
    except Exception as e:
        st.error(f"Error displaying alert system: {e}")

# Example usage
if __name__ == "__main__":
    st.title("Alert System Demo")
    
    # Render alert system
    config = render_alert_system_panel()
    display_alert_system(config)