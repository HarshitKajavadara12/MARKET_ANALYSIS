#!/usr/bin/env python3
"""
Alert System for Market Research System v1.0
Handles various types of alerts and notifications for market events.
Created: 2022-01-15
Author: Market Research Team
"""

import logging
import smtplib
import json
import requests
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import yaml
import os
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/application/alerts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    alert_type: str
    symbol: str
    message: str
    severity: str
    timestamp: datetime
    data: Dict[str, Any] = None
    
class AlertSystem:
    """
    Comprehensive alert system for market research
    Handles email alerts, system notifications, and log-based alerts
    """
    
    def __init__(self, config_path: str = "config/system/alert_config.yaml"):
        """Initialize alert system with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.alerts_history = []
        self.setup_email_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load alert configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    return yaml.safe_load(file)
            else:
                # Default configuration
                return {
                    'email': {
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'sender_email': 'your_email@gmail.com',
                        'sender_password': 'your_app_password',
                        'recipients': ['recipient@example.com']
                    },
                    'thresholds': {
                        'price_change_percent': 5.0,
                        'volume_spike_multiplier': 3.0,
                        'volatility_threshold': 0.03,
                        'correlation_change': 0.2
                    },
                    'alert_types': {
                        'price_alert': True,
                        'volume_alert': True,
                        'technical_alert': True,
                        'news_alert': True,
                        'system_alert': True
                    }
                }
        except Exception as e:
            logger.error(f"Error loading alert config: {e}")
            return {}
    
    def setup_email_config(self):
        """Setup email configuration"""
        email_config = self.config.get('email', {})
        self.smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = email_config.get('smtp_port', 587)
        self.sender_email = email_config.get('sender_email', '')
        self.sender_password = email_config.get('sender_password', '')
        self.recipients = email_config.get('recipients', [])
    
    def create_alert(self, alert_type: str, symbol: str, message: str, 
                    severity: str = "INFO", data: Dict[str, Any] = None) -> Alert:
        """Create a new alert"""
        alert = Alert(
            alert_type=alert_type,
            symbol=symbol,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            data=data or {}
        )
        
        self.alerts_history.append(alert)
        logger.info(f"Alert created - {alert_type}: {message}")
        
        return alert
    
    def check_price_alerts(self, market_data: pd.DataFrame):
        """Check for price-based alerts"""
        try:
            threshold = self.config.get('thresholds', {}).get('price_change_percent', 5.0)
            
            for symbol in market_data.index:
                row = market_data.loc[symbol]
                
                if 'price_change_percent' in row and abs(row['price_change_percent']) >= threshold:
                    direction = "up" if row['price_change_percent'] > 0 else "down"
                    message = f"{symbol} moved {row['price_change_percent']:.2f}% {direction}"
                    
                    alert = self.create_alert(
                        alert_type="price_alert",
                        symbol=symbol,
                        message=message,
                        severity="HIGH" if abs(row['price_change_percent']) >= threshold * 2 else "MEDIUM",
                        data={
                            'price_change_percent': row['price_change_percent'],
                            'current_price': row.get('close', 0),
                            'volume': row.get('volume', 0)
                        }
                    )
                    
                    self.send_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking price alerts: {e}")
    
    def check_volume_alerts(self, market_data: pd.DataFrame):
        """Check for volume spike alerts"""
        try:
            multiplier = self.config.get('thresholds', {}).get('volume_spike_multiplier', 3.0)
            
            for symbol in market_data.index:
                row = market_data.loc[symbol]
                
                current_volume = row.get('volume', 0)
                avg_volume = row.get('avg_volume_20d', current_volume)
                
                if current_volume > avg_volume * multiplier:
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                    message = f"{symbol} volume spike: {volume_ratio:.1f}x average volume"
                    
                    alert = self.create_alert(
                        alert_type="volume_alert",
                        symbol=symbol,
                        message=message,
                        severity="MEDIUM",
                        data={
                            'current_volume': current_volume,
                            'average_volume': avg_volume,
                            'volume_ratio': volume_ratio
                        }
                    )
                    
                    self.send_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking volume alerts: {e}")
    
    def check_technical_alerts(self, technical_data: pd.DataFrame):
        """Check for technical indicator alerts"""
        try:
            for symbol in technical_data.index:
                row = technical_data.loc[symbol]
                
                # RSI overbought/oversold alerts
                rsi = row.get('rsi_14', 50)
                if rsi >= 80:
                    alert = self.create_alert(
                        alert_type="technical_alert",
                        symbol=symbol,
                        message=f"{symbol} RSI overbought at {rsi:.1f}",
                        severity="MEDIUM",
                        data={'rsi': rsi, 'signal': 'overbought'}
                    )
                    self.send_alert(alert)
                elif rsi <= 20:
                    alert = self.create_alert(
                        alert_type="technical_alert",
                        symbol=symbol,
                        message=f"{symbol} RSI oversold at {rsi:.1f}",
                        severity="MEDIUM",
                        data={'rsi': rsi, 'signal': 'oversold'}
                    )
                    self.send_alert(alert)
                
                # MACD signal line crossover
                macd = row.get('macd', 0)
                macd_signal = row.get('macd_signal', 0)
                macd_prev = row.get('macd_prev', macd)
                signal_prev = row.get('macd_signal_prev', macd_signal)
                
                if macd > macd_signal and macd_prev <= signal_prev:
                    alert = self.create_alert(
                        alert_type="technical_alert",
                        symbol=symbol,
                        message=f"{symbol} MACD bullish crossover",
                        severity="MEDIUM",
                        data={'macd': macd, 'signal': 'bullish_crossover'}
                    )
                    self.send_alert(alert)
                elif macd < macd_signal and macd_prev >= signal_prev:
                    alert = self.create_alert(
                        alert_type="technical_alert",
                        symbol=symbol,
                        message=f"{symbol} MACD bearish crossover",
                        severity="MEDIUM",
                        data={'macd': macd, 'signal': 'bearish_crossover'}
                    )
                    self.send_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking technical alerts: {e}")
    
    def check_system_alerts(self):
        """Check for system health alerts"""
        try:
            # Check disk space
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            if free_percent < 10:
                alert = self.create_alert(
                    alert_type="system_alert",
                    symbol="SYSTEM",
                    message=f"Low disk space: {free_percent:.1f}% remaining",
                    severity="HIGH",
                    data={'free_space_percent': free_percent}
                )
                self.send_alert(alert)
            
            # Check log file sizes
            log_files = [
                'logs/application/app.log',
                'logs/application/data_collection.log',
                'logs/application/analysis.log'
            ]
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    size_mb = os.path.getsize(log_file) / (1024 * 1024)
                    if size_mb > 100:  # Alert if log file > 100MB
                        alert = self.create_alert(
                            alert_type="system_alert",
                            symbol="SYSTEM",
                            message=f"Large log file: {log_file} ({size_mb:.1f}MB)",
                            severity="MEDIUM",
                            data={'log_file': log_file, 'size_mb': size_mb}
                        )
                        self.send_alert(alert)
                        
        except Exception as e:
            logger.error(f"Error checking system alerts: {e}")
    
    def send_alert(self, alert: Alert):
        """Send alert via configured channels"""
        try:
            # Log the alert
            self._log_alert(alert)
            
            # Send email if configured
            if self.config.get('alert_types', {}).get(alert.alert_type, True):
                if alert.severity in ['HIGH', 'CRITICAL']:
                    self._send_email_alert(alert)
            
            # Send to external webhook if configured
            webhook_url = self.config.get('webhook_url')
            if webhook_url:
                self._send_webhook_alert(alert, webhook_url)
                
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to file"""
        log_entry = {
            'timestamp': alert.timestamp.isoformat(),
            'type': alert.alert_type,
            'symbol': alert.symbol,
            'message': alert.message,
            'severity': alert.severity,
            'data': alert.data
        }
        
        # Ensure alerts log directory exists
        os.makedirs('logs/alerts', exist_ok=True)
        
        # Write to alerts log file
        with open('logs/alerts/alerts.log', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            if not self.sender_email or not self.recipients:
                logger.warning("Email not configured, skipping email alert")
                return
                
            msg = MimeMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"Market Alert - {alert.severity}: {alert.symbol}"
            
            body = f"""
Market Research System Alert

Symbol: {alert.symbol}
Type: {alert.alert_type}
Severity: {alert.severity}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert.message}

Additional Data:
{json.dumps(alert.data, indent=2) if alert.data else 'None'}

---
This is an automated alert from Market Research System v1.0
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            if self.sender_password:
                server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, self.recipients, text)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.symbol}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert, webhook_url: str):
        """Send alert to webhook"""
        try:
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'symbol': alert.symbol,
                'message': alert.message,
                'severity': alert.severity,
                'data': alert.data
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Webhook alert sent for {alert.symbol}")
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts_history if alert.timestamp >= cutoff_time]
    
    def get_alerts_by_symbol(self, symbol: str) -> List[Alert]:
        """Get all alerts for a specific symbol"""
        return [alert for alert in self.alerts_history if alert.symbol == symbol]
    
    def get_alerts_summary(self) -> Dict[str, int]:
        """Get summary of alerts by type and severity"""
        summary = {
            'total': len(self.alerts_history),
            'by_type': {},
            'by_severity': {},
            'last_24h': len(self.get_recent_alerts(24))
        }
        
        for alert in self.alerts_history:
            # Count by type
            if alert.alert_type not in summary['by_type']:
                summary['by_type'][alert.alert_type] = 0
            summary['by_type'][alert.alert_type] += 1
            
            # Count by severity
            if alert.severity not in summary['by_severity']:
                summary['by_severity'][alert.severity] = 0
            summary['by_severity'][alert.severity] += 1
        
        return summary
    
    def clear_old_alerts(self, days: int = 30):
        """Clear alerts older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        original_count = len(self.alerts_history)
        self.alerts_history = [
            alert for alert in self.alerts_history 
            if alert.timestamp >= cutoff_time
        ]
        cleared_count = original_count - len(self.alerts_history)
        logger.info(f"Cleared {cleared_count} old alerts")
        return cleared_count

def main():
    """Main function for testing alert system"""
    # Initialize alert system
    alert_system = AlertSystem()
    
    # Test system alerts
    alert_system.check_system_alerts()
    
    # Create sample market data for testing
    sample_data = pd.DataFrame({
        'RELIANCE.NS': {
            'close': 2500,
            'price_change_percent': 6.5,
            'volume': 1000000,
            'avg_volume_20d': 300000,
            'rsi_14': 85,
            'macd': 0.5,
            'macd_signal': -0.2,
            'macd_prev': -0.1,
            'macd_signal_prev': 0.1
        },
        'TCS.NS': {
            'close': 3200,
            'price_change_percent': -4.2,
            'volume': 800000,
            'avg_volume_20d': 250000,
            'rsi_14': 25,
            'macd': -0.3,
            'macd_signal': 0.1,
            'macd_prev': 0.2,
            'macd_signal_prev': -0.1
        }
    }).T
    
    # Test price alerts
    alert_system.check_price_alerts(sample_data)
    
    # Test volume alerts
    alert_system.check_volume_alerts(sample_data)
    
    # Test technical alerts
    alert_system.check_technical_alerts(sample_data)
    
    # Print summary
    summary = alert_system.get_alerts_summary()
    print("\nAlert System Summary:")
    print(json.dumps(summary, indent=2))
    
    # Print recent alerts
    recent_alerts = alert_system.get_recent_alerts(24)
    print(f"\nRecent Alerts ({len(recent_alerts)}):")
    for alert in recent_alerts:
        print(f"  {alert.timestamp.strftime('%H:%M:%S')} - {alert.severity} - {alert.symbol}: {alert.message}")

if __name__ == "__main__":
    main()