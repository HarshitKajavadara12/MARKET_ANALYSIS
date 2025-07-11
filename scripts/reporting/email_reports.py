#!/usr/bin/env python3
"""
Email Report Distribution System
Version 1.0 - 2022
Automated email delivery for market research reports
"""

import os
import sys
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
import yaml
import logging
import schedule
import time
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.date_utils import format_date
from utils.file_utils import get_latest_report

class EmailReportSender:
    """Handles automated email distribution of market research reports"""
    
    def __init__(self, config_path="config/reporting/email_recipients.yaml"):
        """Initialize email sender with configuration"""
        self.setup_logging()
        self.load_config(config_path)
        self.load_email_templates()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/reporting.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path):
        """Load email configuration and recipient lists"""
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            self.config = self.get_default_config()
            
        # Load SMTP configuration
        self.smtp_config = self.config.get('smtp', {})
        self.recipients = self.config.get('recipients', {})
        
    def get_default_config(self):
        """Return default email configuration"""
        return {
            'smtp': {
                'server': 'smtp.gmail.com',
                'port': 587,
                'use_tls': True,
                'username': os.getenv('EMAIL_USERNAME', ''),
                'password': os.getenv('EMAIL_PASSWORD', ''),
                'from_name': 'Indian Market Research',
                'from_email': os.getenv('EMAIL_FROM', '')
            },
            'recipients': {
                'daily_reports': [
                    {'email': 'client1@example.com', 'name': 'Client 1', 'tier': 'premium'},
                    {'email': 'client2@example.com', 'name': 'Client 2', 'tier': 'basic'}
                ],
                'weekly_reports': [
                    {'email': 'client1@example.com', 'name': 'Client 1', 'tier': 'premium'}
                ],
                'monthly_reports': [
                    {'email': 'client1@example.com', 'name': 'Client 1', 'tier': 'premium'},
                    {'email': 'fund1@example.com', 'name': 'Fund Manager 1', 'tier': 'institutional'}
                ]
            },
            'email_settings': {
                'daily': {
                    'send_time': '09:00',
                    'subject_template': 'Daily Indian Market Research - {date}',
                    'enabled': True
                },
                'weekly': {
                    'send_day': 'Monday',
                    'send_time': '08:00',
                    'subject_template': 'Weekly Indian Market Analysis - Week of {date}',
                    'enabled': True
                },
                'monthly': {
                    'send_day': 1,
                    'send_time': '08:00',
                    'subject_template': 'Monthly Indian Market Report - {month} {year}',
                    'enabled': True
                }
            }
        }
        
    def load_email_templates(self):
        """Load HTML email templates"""
        template_dir = "reports/templates"
        
        self.templates = {
            'daily': self.load_template(os.path.join(template_dir, 'daily_email_template.html')),
            'weekly': self.load_template(os.path.join(template_dir, 'weekly_email_template.html')),
            'monthly': self.load_template(os.path.join(template_dir, 'monthly_email_template.html'))
        }
        
    def load_template(self, template_path):
        """Load HTML template from file"""
        try:
            with open(template_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            self.logger.warning(f"Template not found: {template_path}. Using default.")
            return self.get_default_html_template()
            
    def get_default_html_template(self):
        """Return default HTML email template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #1f4e79; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; line-height: 1.6; }
        .footer { background-color: #f0f0f0; padding: 15px; text-align: center; font-size: 12px; }
        .highlight { background-color: #e6f3ff; padding: 10px; border-left: 4px solid #1f4e79; }
        .disclaimer { font-size: 10px; color: #666; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Indian Market Research Report</h1>
        <p>{report_type} - {date}</p>
    </div>
    <div class="content">
        <p>Dear {recipient_name},</p>
        
        <p>Please find attached your {report_type} Indian market research report for {date}.</p>
        
        <div class="highlight">
            <h3>Report Highlights:</h3>
            {highlights}
        </div>
        
        <p><strong>Key Focus Areas:</strong></p>
        <ul>
            <li>NIFTY 50 and sectoral indices performance</li>
            <li>Technical analysis and trend identification</li>
            <li>Top gainers and losers analysis</li>
            <li>Market breadth and sentiment indicators</li>
            <li>Economic indicators impact</li>
        </ul>
        
        <p>For any questions or clarifications, please don't hesitate to reach out.</p>
        
        <p>Best regards,<br/>
        Indian Market Research Team</p>
    </div>
    <div class="footer">
        <p>This report is generated by our automated market research system.</p>
        <div class="disclaimer">
            <p><strong>Disclaimer:</strong> This report is for informational purposes only and does not constitute investment advice. 
            Past performance does not guarantee future results. Please consult with your financial advisor before making investment decisions.</p>
        </div>
    </div>
</body>
</html>
        """
        
    def send_daily_reports(self, report_date=None):
        """Send daily reports to subscribers"""
        if report_date is None:
            report_date = datetime.now().date()
            
        self.logger.info(f"Sending daily reports for {report_date}")
        
        try:
            # Find the latest daily report
            report_path = get_latest_report('reports/daily', report_date, 'daily')
            
            if not report_path or not os.path.exists(report_path):
                self.logger.warning(f"Daily report not found for {report_date}")
                return False
                
            # Get report highlights
            highlights = self.generate_daily_highlights(report_path)
            
            # Send to daily subscribers
            recipients = self.recipients.get('daily_reports', [])
            subject = self.config['email_settings']['daily']['subject_template'].format(
                date=format_date(report_date)
            )
            
            success_count = 0
            for recipient in recipients:
                try:
                    self.send_report_email(
                        recipient=recipient,
                        subject=subject,
                        report_path=report_path,
                        report_type='Daily',
                        report_date=report_date,
                        highlights=highlights
                    )
                    success_count += 1
                    self.logger.info(f"Daily report sent to {recipient['email']}")


                   except Exception as e:
                    self.logger.error(f"Failed to send daily report to {recipient['email']}: {str(e)}")
                    
            self.logger.info(f"Daily reports sent successfully to {success_count}/{len(recipients)} recipients")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error sending daily reports: {str(e)}")
            return False
            
    def send_weekly_reports(self, report_date=None):
        """Send weekly reports to subscribers"""
        if report_date is None:
            # Get the Monday of current week
            today = datetime.now().date()
            report_date = today - timedelta(days=today.weekday())
            
        self.logger.info(f"Sending weekly reports for week of {report_date}")
        
        try:
            # Find the latest weekly report
            report_path = get_latest_report('reports/weekly', report_date, 'weekly')
            
            if not report_path or not os.path.exists(report_path):
                self.logger.warning(f"Weekly report not found for {report_date}")
                return False
                
            # Get report highlights
            highlights = self.generate_weekly_highlights(report_path)
            
            # Send to weekly subscribers
            recipients = self.recipients.get('weekly_reports', [])
            subject = self.config['email_settings']['weekly']['subject_template'].format(
                date=format_date(report_date)
            )
            
            success_count = 0
            for recipient in recipients:
                try:
                    self.send_report_email(
                        recipient=recipient,
                        subject=subject,
                        report_path=report_path,
                        report_type='Weekly',
                        report_date=report_date,
                        highlights=highlights
                    )
                    success_count += 1
                    self.logger.info(f"Weekly report sent to {recipient['email']}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to send weekly report to {recipient['email']}: {str(e)}")
                    
            self.logger.info(f"Weekly reports sent successfully to {success_count}/{len(recipients)} recipients")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error sending weekly reports: {str(e)}")
            return False
            
    def send_monthly_reports(self, report_date=None):
        """Send monthly reports to subscribers"""
        if report_date is None:
            # Get first day of current month
            today = datetime.now().date()
            report_date = today.replace(day=1)
            
        self.logger.info(f"Sending monthly reports for {report_date.strftime('%B %Y')}")
        
        try:
            # Find the latest monthly report
            report_path = get_latest_report('reports/monthly', report_date, 'monthly')
            
            if not report_path or not os.path.exists(report_path):
                self.logger.warning(f"Monthly report not found for {report_date}")
                return False
                
            # Get report highlights
            highlights = self.generate_monthly_highlights(report_path)
            
            # Send to monthly subscribers
            recipients = self.recipients.get('monthly_reports', [])
            subject = self.config['email_settings']['monthly']['subject_template'].format(
                month=report_date.strftime('%B'),
                year=report_date.year
            )
            
            success_count = 0
            for recipient in recipients:
                try:
                    self.send_report_email(
                        recipient=recipient,
                        subject=subject,
                        report_path=report_path,
                        report_type='Monthly',
                        report_date=report_date,
                        highlights=highlights
                    )
                    success_count += 1
                    self.logger.info(f"Monthly report sent to {recipient['email']}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to send monthly report to {recipient['email']}: {str(e)}")
                    
            self.logger.info(f"Monthly reports sent successfully to {success_count}/{len(recipients)} recipients")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error sending monthly reports: {str(e)}")
            return False
            
    def send_report_email(self, recipient, subject, report_path, report_type, report_date, highlights):
        """Send individual report email"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.smtp_config['from_name']} <{self.smtp_config['from_email']}>"
            msg['To'] = recipient['email']
            
            # Format template
            template_key = report_type.lower()
            html_content = self.templates.get(template_key, self.get_default_html_template())
            
            formatted_html = html_content.format(
                recipient_name=recipient.get('name', 'Valued Client'),
                report_type=report_type,
                date=format_date(report_date),
                highlights=highlights
            )
            
            # Create HTML part
            html_part = MIMEText(formatted_html, 'html')
            msg.attach(html_part)
            
            # Attach PDF report
            if os.path.exists(report_path):
                with open(report_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    
                encoders.encode_base64(part)
                
                filename = os.path.basename(report_path)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                
                msg.attach(part)
            
            # Send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                if self.smtp_config.get('use_tls', True):
                    server.starttls(context=context)
                
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email to {recipient['email']}: {str(e)}")
            raise
            
    def generate_daily_highlights(self, report_path):
        """Generate highlights for daily report (Version 1 - basic text extraction)"""
        try:
            # For Version 1, we'll extract basic highlights from filename and basic analysis
            highlights = """
            <ul>
                <li><strong>NIFTY 50:</strong> Daily movement and key levels analysis</li>
                <li><strong>Bank NIFTY:</strong> Banking sector performance review</li>
                <li><strong>Sectoral Analysis:</strong> Top performing and underperforming sectors</li>
                <li><strong>F&O Analysis:</strong> Futures and Options market insights</li>
                <li><strong>Market Breadth:</strong> Advance-Decline ratio and market sentiment</li>
            </ul>
            """
            return highlights
            
        except Exception as e:
            self.logger.error(f"Error generating daily highlights: {str(e)}")
            return "<p>Daily market analysis and key insights included in the attached report.</p>"
            
    def generate_weekly_highlights(self, report_path):
        """Generate highlights for weekly report"""
        try:
            highlights = """
            <ul>
                <li><strong>Weekly Performance:</strong> NIFTY 50 and major indices weekly returns</li>
                <li><strong>Sectoral Rotation:</strong> Weekly sector performance comparison</li>
                <li><strong>Technical Outlook:</strong> Key support and resistance levels</li>
                <li><strong>FII/DII Activity:</strong> Foreign and domestic institutional flows</li>
                <li><strong>Economic Events:</strong> Impact of weekly economic announcements</li>
            </ul>
            """
            return highlights
            
        except Exception as e:
            self.logger.error(f"Error generating weekly highlights: {str(e)}")
            return "<p>Weekly market analysis and trend insights included in the attached report.</p>"
            
    def generate_monthly_highlights(self, report_path):
        """Generate highlights for monthly report"""
        try:
            highlights = """
            <ul>
                <li><strong>Monthly Returns:</strong> Comprehensive performance across all major indices</li>
                <li><strong>Sector Analysis:</strong> Monthly sectoral winners and losers</li>
                <li><strong>Economic Impact:</strong> Monthly economic data and market correlation</li>
                <li><strong>Corporate Earnings:</strong> Monthly earnings season impact</li>
                <li><strong>Global Correlation:</strong> Indian markets vs global indices</li>
                <li><strong>Investment Themes:</strong> Emerging monthly investment opportunities</li>
            </ul>
            """
            return highlights
            
        except Exception as e:
            self.logger.error(f"Error generating monthly highlights: {str(e)}")
            return "<p>Comprehensive monthly market analysis included in the attached report.</p>"
            
    def setup_scheduled_reports(self):
        """Setup scheduled report sending (Version 1 - basic scheduling)"""
        try:
            # Daily reports - 9:00 AM IST on weekdays
            if self.config['email_settings']['daily']['enabled']:
                schedule.every().monday.at("09:00").do(self.send_daily_reports)
                schedule.every().tuesday.at("09:00").do(self.send_daily_reports)
                schedule.every().wednesday.at("09:00").do(self.send_daily_reports)
                schedule.every().thursday.at("09:00").do(self.send_daily_reports)
                schedule.every().friday.at("09:00").do(self.send_daily_reports)
                
            # Weekly reports - Monday 8:00 AM IST
            if self.config['email_settings']['weekly']['enabled']:
                schedule.every().monday.at("08:00").do(self.send_weekly_reports)
                
            # Monthly reports - 1st of month 8:00 AM IST
            if self.config['email_settings']['monthly']['enabled']:
                schedule.every().day.at("08:00").do(self.check_and_send_monthly)
                
            self.logger.info("Scheduled reports setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up scheduled reports: {str(e)}")
            
    def check_and_send_monthly(self):
        """Check if today is first of month and send monthly reports"""
        today = datetime.now().date()
        if today.day == 1:
            self.send_monthly_reports()
            
    def run_scheduler(self):
        """Run the email scheduler (Version 1 - basic loop)"""
        self.logger.info("Starting email report scheduler...")
        self.setup_scheduled_reports()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Email scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Email scheduler error: {str(e)}")
            
    def send_test_email(self, test_recipient="test@example.com"):
        """Send a test email to verify configuration"""
        try:
            test_recipient_data = {
                'email': test_recipient,
                'name': 'Test User',
                'tier': 'test'
            }
            
            # Create a dummy report path for testing
            test_report_path = "reports/test_report.pdf"
            
            self.send_report_email(
                recipient=test_recipient_data,
                subject="Test Email - Indian Market Research System",
                report_path=test_report_path,
                report_type='Test',
                report_date=datetime.now().date(),
                highlights="<p>This is a test email to verify the email system is working correctly.</p>"
            )
            
            self.logger.info(f"Test email sent successfully to {test_recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Test email failed: {str(e)}")
            return False
            
    def get_recipient_stats(self):
        """Get statistics about recipients and report distribution"""
        stats = {
            'daily_subscribers': len(self.recipients.get('daily_reports', [])),
            'weekly_subscribers': len(self.recipients.get('weekly_reports', [])),
            'monthly_subscribers': len(self.recipients.get('monthly_reports', [])),
            'total_unique_subscribers': len(set([
                r['email'] for report_type in self.recipients.values() 
                for r in report_type
            ]))
        }
        
        return stats

def main():
    """Main function for running email reports system"""
    try:
        # Initialize email sender
        email_sender = EmailReportSender()
        
        # Check command line arguments for manual sending
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'daily':
                email_sender.send_daily_reports()
            elif command == 'weekly':
                email_sender.send_weekly_reports()
            elif command == 'monthly':
                email_sender.send_monthly_reports()
            elif command == 'test':
                test_email = sys.argv[2] if len(sys.argv) > 2 else "test@example.com"
                email_sender.send_test_email(test_email)
            elif command == 'stats':
                stats = email_sender.get_recipient_stats()
                print("\n=== Email Recipient Statistics ===")
                for key, value in stats.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
            elif command == 'schedule':
                email_sender.run_scheduler()
            else:
                print("Usage: python email_reports.py [daily|weekly|monthly|test|stats|schedule]")
        else:
            # Default: run scheduler
            email_sender.run_scheduler()
            
    except Exception as e:
        logging.error(f"Email reports system error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()