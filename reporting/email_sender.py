"""
Email Report Delivery System for Market Research System v1.0
Handles automated email delivery of reports to clients and subscribers
Compatible with various email providers and supports HTML reports
"""

import smtplib
import ssl
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    """Email configuration settings"""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30

@dataclass
class Recipient:
    """Email recipient information"""
    email: str
    name: str = ""
    client_type: str = "standard"  # standard, premium, institutional
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}

@dataclass
class EmailTemplate:
    """Email template structure"""
    template_id: str
    subject: str
    html_body: str
    text_body: str = ""
    attachments: List[str] = None
    
    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []

class EmailSender:
    """
    Professional email delivery system for market research reports
    Supports multiple templates, client segmentation, and delivery tracking
    """
    
    def __init__(self, config: EmailConfig):
        """
        Initialize email sender with configuration
        
        Args:
            config (EmailConfig): Email server configuration
        """
        self.config = config
        self.templates = {}
        self.recipients = {}
        self.delivery_log = []
        self.retry_attempts = 3
        self.retry_delay = 5  # seconds
        
        # Load default templates
        self._load_default_templates()
        
        # Create logs directory
        self.log_dir = Path("logs/email")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def add_recipient(self, recipient: Recipient) -> bool:
        """
        Add recipient to the mailing list
        
        Args:
            recipient (Recipient): Recipient information
            
        Returns:
            bool: Success status
        """
        try:
            self.recipients[recipient.email] = recipient
            logger.info(f"Added recipient: {recipient.email}")
            return True
        except Exception as e:
            logger.error(f"Error adding recipient {recipient.email}: {str(e)}")
            return False
    
    def add_recipients_from_list(self, recipients_data: List[Dict]) -> int:
        """
        Add multiple recipients from list
        
        Args:
            recipients_data (List[Dict]): List of recipient dictionaries
            
        Returns:
            int: Number of successfully added recipients
        """
        added_count = 0
        for recipient_data in recipients_data:
            try:
                recipient = Recipient(**recipient_data)
                if self.add_recipient(recipient):
                    added_count += 1
            except Exception as e:
                logger.error(f"Error adding recipient from data {recipient_data}: {str(e)}")
        
        logger.info(f"Added {added_count} recipients from list")
        return added_count
    
    def load_recipients_from_file(self, file_path: str) -> int:
        """
        Load recipients from JSON file
        
        Args:
            file_path (str): Path to recipients JSON file
            
        Returns:
            int: Number of loaded recipients
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                recipients_data = json.load(f)
            
            return self.add_recipients_from_list(recipients_data)
            
        except Exception as e:
            logger.error(f"Error loading recipients from file {file_path}: {str(e)}")
            return 0
    
    def add_template(self, template: EmailTemplate) -> bool:
        """
        Add email template
        
        Args:
            template (EmailTemplate): Email template
            
        Returns:
            bool: Success status
        """
        try:
            self.templates[template.template_id] = template
            logger.info(f"Added template: {template.template_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding template {template.template_id}: {str(e)}")
            return False
    
    def send_daily_report(self, 
                         report_data: Dict[str, Any],
                         recipients: List[str] = None,
                         attachments: List[str] = None) -> Dict[str, Any]:
        """
        Send daily market research report
        
        Args:
            report_data (Dict): Report data for template
            recipients (List[str]): List of recipient emails (None for all)
            attachments (List[str]): List of attachment file paths
            
        Returns:
            Dict: Delivery results
        """
        template_id = "daily_report"
        subject = f"Daily Market Research Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Prepare report data with defaults
        report_data = self._prepare_report_data(report_data, "daily")
        
        return self._send_template_email(
            template_id=template_id,
            subject=subject,
            template_data=report_data,
            recipients=recipients,
            attachments=attachments
        )
    
    def send_weekly_report(self, 
                          report_data: Dict[str, Any],
                          recipients: List[str] = None,
                          attachments: List[str] = None) -> Dict[str, Any]:
        """
        Send weekly market research report
        
        Args:
            report_data (Dict): Report data for template
            recipients (List[str]): List of recipient emails
            attachments (List[str]): List of attachment file paths
            
        Returns:
            Dict: Delivery results
        """
        template_id = "weekly_report"
        week_start = datetime.now() - timedelta(days=7)
        subject = f"Weekly Market Analysis - Week of {week_start.strftime('%Y-%m-%d')}"
        
        report_data = self._prepare_report_data(report_data, "weekly")
        
        return self._send_template_email(
            template_id=template_id,
            subject=subject,
            template_data=report_data,
            recipients=recipients,
            attachments=attachments
        )
    
    def send_alert_email(self, 
                        alert_type: str,
                        alert_data: Dict[str, Any],
                        recipients: List[str] = None,
                        priority: str = "normal") -> Dict[str, Any]:
        """
        Send market alert email
        
        Args:
            alert_type (str): Type of alert (price_alert, news_alert, etc.)
            alert_data (Dict): Alert data
            recipients (List[str]): List of recipient emails
            priority (str): Email priority (low, normal, high)
            
        Returns:
            Dict: Delivery results
        """
        template_id = "market_alert"
        subject = f"Market Alert: {alert_type.replace('_', ' ').title()}"
        
        if priority == "high":
            subject = f"🚨 URGENT - {subject}"
        elif priority == "low":
            subject = f"ℹ️ INFO - {subject}"
        
        alert_data.update({
            'alert_type': alert_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'priority': priority
        })
        
        return self._send_template_email(
            template_id=template_id,
            subject=subject,
            template_data=alert_data,
            recipients=recipients
        )
    
    def send_custom_email(self, 
                         subject: str,
                         body: str,
                         recipients: List[str],
                         attachments: List[str] = None,
                         is_html: bool = True) -> Dict[str, Any]:
        """
        Send custom email
        
        Args:
            subject (str): Email subject
            body (str): Email body (HTML or text)
            recipients (List[str]): List of recipient emails
            attachments (List[str]): List of attachment file paths
            is_html (bool): Whether body is HTML
            
        Returns:
            Dict: Delivery results
        """
        results = {
            'sent': [],
            'failed': [],
            'total': len(recipients),
            'success_rate': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        for recipient_email in recipients:
            try:
                if self._send_single_email(
                    to_email=recipient_email,
                    subject=subject,
                    body=body,
                    attachments=attachments,
                    is_html=is_html
                ):
                    results['sent'].append(recipient_email)
                else:
                    results['failed'].append(recipient_email)
                    
            except Exception as e:
                logger.error(f"Failed to send email to {recipient_email}: {str(e)}")
                results['failed'].append(recipient_email)
        
        results['success_rate'] = len(results['sent']) / results['total'] * 100
        self._log_delivery_results(results)
        
        return results
    
    def _send_template_email(self, 
                           template_id: str,
                           subject: str,
                           template_data: Dict[str, Any],
                           recipients: List[str] = None,
                           attachments: List[str] = None) -> Dict[str, Any]:
        """
        Send email using template
        
        Args:
            template_id (str): Template identifier
            subject (str): Email subject
            template_data (Dict): Data for template substitution
            recipients (List[str]): List of recipient emails
            attachments (List[str]): List of attachment file paths
            
        Returns:
            Dict: Delivery results
        """
        if template_id not in self.templates:
            logger.error(f"Template {template_id} not found")
            return {'error': f'Template {template_id} not found'}
        
        template = self.templates[template_id]
        
        # Continue from where the code was incomplete
        if recipients is None:
            recipients = list(self.recipients.keys())
        
        results = {
            'sent': [],
            'failed': [],
            'total': len(recipients),
            'success_rate': 0,
            'timestamp': datetime.now().isoformat(),
            'template_used': template_id
        }
        
        # Render template with data
        try:
            html_body = template.html_body.format(**template_data)
            text_body = template.text_body.format(**template_data) if template.text_body else ""
        except KeyError as e:
            logger.error(f"Template rendering error - missing key: {str(e)}")
            return {'error': f'Template rendering failed: missing key {str(e)}'}
        
        # Send to each recipient
        for recipient_email in recipients:
            try:
                recipient = self.recipients.get(recipient_email)
                if recipient:
                    # Personalize for Indian market context
                    personalized_subject = self._personalize_for_indian_market(subject, recipient)
                    personalized_html = self._personalize_content(html_body, recipient)
                    
                    if self._send_single_email(
                        to_email=recipient_email,
                        to_name=recipient.name if recipient else "",
                        subject=personalized_subject,
                        body=personalized_html,
                        text_body=text_body,
                        attachments=attachments,
                        is_html=True
                    ):
                        results['sent'].append(recipient_email)
                    else:
                        results['failed'].append(recipient_email)
                else:
                    logger.warning(f"Recipient {recipient_email} not found in database")
                    results['failed'].append(recipient_email)
                    
            except Exception as e:
                logger.error(f"Failed to send email to {recipient_email}: {str(e)}")
                results['failed'].append(recipient_email)
        
        results['success_rate'] = (len(results['sent']) / results['total'] * 100) if results['total'] > 0 else 0
        self._log_delivery_results(results)
        
        return results
    
    def _send_single_email(self, 
                          to_email: str,
                          subject: str,
                          body: str,
                          to_name: str = "",
                          text_body: str = "",
                          attachments: List[str] = None,
                          is_html: bool = True) -> bool:
        """
        Send single email with retry mechanism
        
        Args:
            to_email (str): Recipient email
            subject (str): Email subject
            body (str): Email body
            to_name (str): Recipient name
            text_body (str): Plain text version
            attachments (List[str]): Attachment file paths
            is_html (bool): Whether body is HTML
            
        Returns:
            bool: Success status
        """
        for attempt in range(self.retry_attempts):
            try:
                # Create message
                msg = MIMEMultipart('alternative')
                msg['From'] = self.config.username
                msg['To'] = f"{to_name} <{to_email}>" if to_name else to_email
                msg['Subject'] = subject
                msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
                
                # Add text version if provided
                if text_body:
                    text_part = MIMEText(text_body, 'plain', 'utf-8')
                    msg.attach(text_part)
                
                # Add HTML version
                if is_html:
                    html_part = MIMEText(body, 'html', 'utf-8')
                    msg.attach(html_part)
                else:
                    text_part = MIMEText(body, 'plain', 'utf-8')
                    msg.attach(text_part)
                
                # Add attachments
                if attachments:
                    for file_path in attachments:
                        self._attach_file(msg, file_path)
                
                # Send email
                with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                    server.set_debuglevel(0)  # Set to 1 for debugging
                    
                    if self.config.use_tls:
                        server.starttls()
                    
                    server.login(self.config.username, self.config.password)
                    server.send_message(msg)
                
                logger.info(f"Email sent successfully to {to_email}")
                self._log_email_sent(to_email, subject, True)
                return True
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {to_email}: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All attempts failed for {to_email}")
                    self._log_email_sent(to_email, subject, False, str(e))
        
        return False
    
    def _attach_file(self, msg: MIMEMultipart, file_path: str):
        """
        Attach file to email message
        
        Args:
            msg (MIMEMultipart): Email message
            file_path (str): Path to file to attach
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Attachment file not found: {file_path}")
                return
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type is None:
                mime_type = 'application/octet-stream'
            
            main_type, sub_type = mime_type.split('/', 1)
            
            with open(file_path, 'rb') as f:
                if main_type == 'image':
                    attachment = MIMEImage(f.read(), _subtype=sub_type)
                else:
                    attachment = MIMEBase(main_type, sub_type)
                    attachment.set_payload(f.read())
                    encoders.encode_base64(attachment)
            
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename="{file_path.name}"'
            )
            
            msg.attach(attachment)
            logger.info(f"Attached file: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error attaching file {file_path}: {str(e)}")
    
    def _personalize_for_indian_market(self, subject: str, recipient: Recipient) -> str:
        """
        Personalize subject for Indian market context (Version 1 - 2022)
        
        Args:
            subject (str): Original subject
            recipient (Recipient): Recipient information
            
        Returns:
            str: Personalized subject
        """
        # Add Indian market context for 2022
        if "Daily" in subject:
            subject = subject.replace("Daily", "Daily NSE/BSE")
        elif "Weekly" in subject:
            subject = subject.replace("Weekly", "Weekly Indian Markets")
        elif "Alert" in subject:
            subject = f"[Indian Markets] {subject}"
        
        # Add recipient name if available
        if recipient and recipient.name:
            return f"{subject} - {recipient.name}"
        
        return subject
    
    def _personalize_content(self, content: str, recipient: Recipient) -> str:
        """
        Personalize content for recipient
        
        Args:
            content (str): Original content
            recipient (Recipient): Recipient information
            
        Returns:
            str: Personalized content
        """
        if not recipient:
            return content
        
        # Replace placeholders with recipient data
        replacements = {
            '{recipient_name}': recipient.name or 'Valued Client',
            '{client_type}': recipient.client_type.title(),
            '{indian_markets}': 'NSE, BSE & MCX',
            '{timezone}': 'IST',
            '{currency}': '₹ (INR)'
        }
        
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        
        return content
    
    def _prepare_report_data(self, report_data: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """
        Prepare report data with Indian market defaults for Version 1 (2022)
        
        Args:
            report_data (Dict): Raw report data
            report_type (str): Type of report (daily, weekly)
            
        Returns:
            Dict: Prepared report data
        """
        # Set Indian market defaults for 2022
        defaults = {
            'report_date': datetime.now().strftime('%d-%m-%Y'),  # Indian date format
            'report_time': datetime.now().strftime('%I:%M %p IST'),
            'market_session': self._get_market_session(),
            'primary_indices': ['NIFTY 50', 'SENSEX', 'NIFTY BANK'],
            'currency': 'INR',
            'timezone': 'Asia/Kolkata',
            'report_type': report_type,
            'system_version': 'v1.0 (2022)',
            'data_source': 'NSE, BSE Historical Data',
            'disclaimer': 'This is an automated research report. Past performance does not guarantee future results.'
        }
        
        # Merge with provided data
        prepared_data = {**defaults, **report_data}
        
        # Add Indian market specific calculations
        if 'indices_data' in prepared_data:
            prepared_data['market_sentiment'] = self._calculate_indian_market_sentiment(
                prepared_data['indices_data']
            )
        
        return prepared_data
    
    def _get_market_session(self) -> str:
        """
        Get current Indian market session
        
        Returns:
            str: Market session status
        """
        now = datetime.now()
        current_time = now.time()
        
        # NSE/BSE trading hours (IST)
        market_open = datetime.strptime('09:15', '%H:%M').time()
        market_close = datetime.strptime('15:30', '%H:%M').time()
        
        if market_open <= current_time <= market_close:
            return 'Market Open'
        elif current_time < market_open:
            return 'Pre-Market'
        else:
            return 'Post-Market'
    
    def _calculate_indian_market_sentiment(self, indices_data: Dict) -> str:
        """
        Calculate basic market sentiment for Indian indices (Version 1 - 2022)
        
        Args:
            indices_data (Dict): Indices performance data
            
        Returns:
            str: Market sentiment
        """
        try:
            # Simple sentiment calculation based on major indices
            nifty_change = indices_data.get('nifty_change', 0)
            sensex_change = indices_data.get('sensex_change', 0)
            
            avg_change = (nifty_change + sensex_change) / 2
            
            if avg_change > 1:
                return 'Bullish'
            elif avg_change < -1:
                return 'Bearish'
            else:
                return 'Neutral'
        except:
            return 'Neutral'
    
    def _load_default_templates(self):
        """
        Load default email templates for Indian market research (Version 1 - 2022)
        """
        # Daily Report Template
        daily_template = EmailTemplate(
            template_id="daily_report",
            subject="Daily Indian Market Research Report - {report_date}",
            html_body="""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #1f4e79; color: white; padding: 15px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .indices {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                    .footer {{ font-size: 12px; color: #666; margin-top: 30px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>Daily Indian Market Research Report</h2>
                    <p>{report_date} | {report_time}</p>
                </div>
                
                <div class="content">
                    <p>Dear {recipient_name},</p>
                    
                    <p>Here's your daily market research report for the Indian stock markets:</p>
                    
                    <div class="indices">
                        <h3>Market Overview ({market_session})</h3>
                        <p><strong>Primary Indices:</strong> {primary_indices}</p>
                        <p><strong>Market Sentiment:</strong> {market_sentiment}</p>
                        <p><strong>Data Source:</strong> {data_source}</p>
                    </div>
                    
                    <h3>Key Highlights</h3>
                    <ul>
                        <li>Market performance summary</li>
                        <li>Sectoral analysis (IT, Banking, Pharma)</li>
                        <li>Volume and volatility insights</li>
                    </ul>
                    
                    <p>For detailed analysis, please refer to the attached reports.</p>
                    
                    <div class="footer">
                        <p>{disclaimer}</p>
                        <p>Generated by: {system_version}</p>
                        <p>Timezone: {timezone} | Currency: {currency}</p>
                    </div>
                </div>
            </body>
            </html>
            """,
            text_body="""
            Daily Indian Market Research Report
            {report_date} | {report_time}
            
            Dear {recipient_name},
            
            Market Overview ({market_session}):
            - Primary Indices: {primary_indices}
            - Market Sentiment: {market_sentiment}
            - Data Source: {data_source}
            
            Key Highlights:
            - Market performance summary
            - Sectoral analysis (IT, Banking, Pharma)
            - Volume and volatility insights
            
            {disclaimer}
            Generated by: {system_version}
            """
        )
        
        # Weekly Report Template
        weekly_template = EmailTemplate(
            template_id="weekly_report",
            subject="Weekly Indian Market Analysis - Week of {report_date}",
            html_body="""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #2d5016; color: white; padding: 15px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .summary {{ background-color: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>Weekly Indian Market Analysis</h2>
                    <p>Week of {report_date}</p>
                </div>
                
                <div class="content">
                    <p>Dear {recipient_name},</p>
                    
                    <div class="summary">
                        <h3>Weekly Summary</h3>
                        <p><strong>Markets Covered:</strong> {primary_indices}</p>
                        <p><strong>Overall Sentiment:</strong> {market_sentiment}</p>
                    </div>
                    
                    <h3>This Week's Analysis</h3>
                    <ul>
                        <li>Weekly performance of major indices</li>
                        <li>Sector rotation analysis</li>
                        <li>FII/DII activity summary</li>
                        <li>Key corporate announcements</li>
                    </ul>
                </div>
            </body>
            </html>
            """
        )
        
        # Market Alert Template
        alert_template = EmailTemplate(
            template_id="market_alert",
            subject="Indian Market Alert: {alert_type}",
            html_body="""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="background-color: #d32f2f; color: white; padding: 15px; text-align: center;">
                    <h2>Market Alert</h2>
                    <p>{timestamp}</p>
                </div>
                
                <div style="padding: 20px;">
                    <p>Dear {recipient_name},</p>
                    
                    <p><strong>Alert Type:</strong> {alert_type}</p>
                    <p><strong>Priority:</strong> {priority}</p>
                    
                    <p>Please review the market conditions and take appropriate action.</p>
                </div>
            </body>
            </html>
            """
        )
        
        # Add templates
        self.add_template(daily_template)
        self.add_template(weekly_template)
        self.add_template(alert_template)
        
        logger.info("Default Indian market templates loaded for Version 1 (2022)")
    
    def _log_delivery_results(self, results: Dict[str, Any]):
        """
        Log email delivery results to file
        
        Args:
            results (Dict): Delivery results
        """
        try:
            log_file = self.log_dir / f"delivery_log_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Load existing logs
            existing_logs = []
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
            
            # Add new results
            existing_logs.append(results)
            
            # Save updated logs
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Delivery results logged to {log_file}")
            
        except Exception as e:
            logger.error(f"Error logging delivery results: {str(e)}")
    
    def _log_email_sent(self, recipient: str, subject: str, success: bool, error: str = None):
        """
        Log individual email sending attempt
        
        Args:
            recipient (str): Recipient email
            subject (str): Email subject
            success (bool): Whether email was sent successfully
            error (str): Error message if failed
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'recipient': recipient,
            'subject': subject,
            'success': success,
            'error': error,
            'version': 'v1.0'
        }
        
        self.delivery_log.append(log_entry)
        
        # Keep only last 1000 entries in memory
        if len(self.delivery_log) > 1000:
            self.delivery_log = self.delivery_log[-1000:]
    
    def get_delivery_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get delivery statistics for specified days
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            Dict: Delivery statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_logs = [
                log for log in self.delivery_log 
                if datetime.fromisoformat(log['timestamp']) > cutoff_date
            ]
            
            total_attempts = len(recent_logs)
            successful = len([log for log in recent_logs if log['success']])
            failed = total_attempts - successful
            
            stats = {
                'period_days': days,
                'total_attempts': total_attempts,
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / total_attempts * 100) if total_attempts > 0 else 0,
                'most_recent': recent_logs[-1] if recent_logs else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating delivery stats: {str(e)}")
            return {'error': str(e)}
    
    def send_system_status_report(self, admin_emails: List[str]) -> Dict[str, Any]:
        """
        Send system status report to administrators (Version 1 - 2022)
        
        Args:
            admin_emails (List[str]): List of admin email addresses
            
        Returns:
            Dict: Send results
        """
        stats = self.get_delivery_stats(7)
        
        subject = f"Market Research System Status - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Market Research System Status Report</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%d-%m-%Y %I:%M %p IST')}</p>
            <p><strong>System Version:</strong> v1.0 (2022)</p>
            
            <h3>Email Delivery Statistics (Last 7 Days)</h3>
            <ul>
                <li>Total Attempts: {stats.get('total_attempts', 0)}</li>
                <li>Successful: {stats.get('successful', 0)}</li>
                <li>Failed: {stats.get('failed', 0)}</li>
                <li>Success Rate: {stats.get('success_rate', 0):.2f}%</li>
            </ul>
            
            <h3>System Information</h3>
            <ul>
                <li>Active Recipients: {len(self.recipients)}</li>
                <li>Available Templates: {len(self.templates)}</li>
                <li>Focus: Indian Stock Markets (NSE/BSE)</li>
            </ul>
            
            <p>System is running normally for Indian market research reporting.</p>
        </body>
        </html>
        """
        
        return self.send_custom_email(
            subject=subject,
            body=body,
            recipients=admin_emails,
            is_html=True
        )