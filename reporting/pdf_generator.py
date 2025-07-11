"""
PDF Generator Module for Market Research System v1.0
===================================================

Creates professional PDF reports using ReportLab library.
Supports multiple report types: Daily, Weekly, Monthly, Custom.

Author: Independent Market Researcher
Created: May 2022
Last Updated: December 2022
"""

import os
import io
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, black, blue, red, green
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import PageBreak, Image, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Line
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib import colors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


class PDFGenerator:
    """
    Professional PDF report generator for market research
    """
    
    def __init__(self, output_dir: str = "reports/", page_size=A4):
        """
        Initialize PDF Generator
        
        Args:
            output_dir: Directory to save PDF reports
            page_size: Page size for PDF (A4, letter, etc.)
        """
        self.output_dir = output_dir
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for reports"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkgreen,
            alignment=TA_LEFT
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=5
        ))
        
        # Summary style
        self.styles.add(ParagraphStyle(
            name='Summary',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            textColor=colors.black,
            alignment=TA_JUSTIFY,
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=8
        ))
        
    def create_header(self, title: str, date_str: str) -> List:
        """Create report header with title and date"""
        story = []
        
        # Title
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Date and company info
        header_info = f"""
        <b>Generated on:</b> {date_str}<br/>
        <b>Prepared by:</b> Independent Market Research<br/>
        <b>Focus:</b> Indian Stock Market Analysis
        """
        story.append(Paragraph(header_info, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        return story
    
    def create_executive_summary(self, summary_data: Dict) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Market overview
        market_summary = f"""
        <b>Market Performance:</b> {summary_data.get('market_performance', 'N/A')}<br/>
        <b>Top Performer:</b> {summary_data.get('top_performer', 'N/A')}<br/>
        <b>Worst Performer:</b> {summary_data.get('worst_performer', 'N/A')}<br/>
        <b>Market Sentiment:</b> {summary_data.get('sentiment', 'Neutral')}<br/>
        <b>Key Recommendation:</b> {summary_data.get('recommendation', 'Hold')}
        """
        
        story.append(Paragraph(market_summary, self.styles['Summary']))
        story.append(Spacer(1, 15))
        
        return story
    
    def create_market_data_table(self, df: pd.DataFrame, title: str) -> List:
        """Create formatted table from DataFrame"""
        story = []
        
        story.append(Paragraph(title, self.styles['SectionHeader']))
        
        # Convert DataFrame to list of lists for ReportLab
        data = [list(df.columns)]
        for _, row in df.iterrows():
            data.append([str(val) for val in row.values])
        
        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
        
        return story
    
    def add_chart_to_story(self, chart_path: str, title: str, width: int = 400, height: int = 300) -> List:
        """Add chart image to story"""
        story = []
        
        story.append(Paragraph(title, self.styles['SectionHeader']))
        
        if os.path.exists(chart_path):
            img = Image(chart_path, width=width, height=height)
            story.append(img)
        else:
            story.append(Paragraph(f"Chart not found: {chart_path}", self.styles['Normal']))
            
        story.append(Spacer(1, 15))
        
        return story
    
    def create_sector_analysis(self, sector_data: Dict) -> List:
        """Create sector-wise analysis section"""
        story = []
        
        story.append(Paragraph("Sector Analysis", self.styles['SectionHeader']))
        
        for sector, data in sector_data.items():
            sector_text = f"""
            <b>{sector} Sector:</b><br/>
            Performance: {data.get('performance', 'N/A')}%<br/>
            Top Stock: {data.get('top_stock', 'N/A')}<br/>
            Volume: {data.get('volume', 'N/A')}<br/>
            Outlook: {data.get('outlook', 'Neutral')}
            """
            story.append(Paragraph(sector_text, self.styles['Normal']))
            story.append(Spacer(1, 10))
        
        return story
    
    def create_technical_analysis(self, technical_data: Dict) -> List:
        """Create technical analysis section"""
        story = []
        
        story.append(Paragraph("Technical Analysis", self.styles['SectionHeader']))
        
        # Support and Resistance levels
        support_resistance = f"""
        <b>Key Levels:</b><br/>
        Resistance: {technical_data.get('resistance', 'N/A')}<br/>
        Support: {technical_data.get('support', 'N/A')}<br/>
        Current Trend: {technical_data.get('trend', 'Sideways')}<br/>
        RSI: {technical_data.get('rsi', 'N/A')}<br/>
        MACD: {technical_data.get('macd', 'N/A')}
        """
        
        story.append(Paragraph(support_resistance, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        return story
    
    def create_risk_disclaimer(self) -> List:
        """Create risk disclaimer section"""
        story = []
        
        disclaimer = """
        <b>Risk Disclaimer:</b><br/>
        This report is for informational purposes only and should not be considered as investment advice.
        Past performance does not guarantee future results. Please consult with a qualified financial advisor
        before making investment decisions. The Indian stock market is subject to various risks including
        market volatility, regulatory changes, and economic factors.
        """
        
        story.append(Paragraph("Disclaimer", self.styles['SectionHeader']))
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return story
    
    def generate_daily_report(self, data: Dict, filename: Optional[str] = None) -> str:
        """Generate daily market report"""
        if not filename:
            filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(filepath, pagesize=self.page_size)
        story = []
        
        # Header
        story.extend(self.create_header(
            "Daily Market Research Report", 
            datetime.now().strftime("%B %d, %Y")
        ))
        
        # Executive Summary
        story.extend(self.create_executive_summary(data.get('summary', {})))
        
        # Market Data Table
        if 'market_data' in data and isinstance(data['market_data'], pd.DataFrame):
            story.extend(self.create_market_data_table(
                data['market_data'], "Market Performance"
            ))
        
        # Technical Analysis
        story.extend(self.create_technical_analysis(data.get('technical', {})))
        
        # Sector Analysis
        story.extend(self.create_sector_analysis(data.get('sectors', {})))
        
        # Add page break
        story.append(PageBreak())
        
        # Risk Disclaimer
        story.extend(self.create_risk_disclaimer())
        
        # Build PDF
        doc.build(story)
        
        return filepath
    
    def generate_weekly_report(self, data: Dict, filename: Optional[str] = None) -> str:
        """Generate weekly market report"""
        if not filename:
            filename = f"weekly_report_{datetime.now().strftime('%Y_W%U')}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(filepath, pagesize=self.page_size)
        story = []
        
        # Header
        story.extend(self.create_header(
            "Weekly Market Research Report", 
            f"Week of {datetime.now().strftime('%B %d, %Y')}"
        ))
        
        # Weekly Summary
        weekly_summary = data.get('weekly_summary', {})
        summary_text = f"""
        <b>Week Performance:</b> {weekly_summary.get('performance', 'N/A')}%<br/>
        <b>Best Day:</b> {weekly_summary.get('best_day', 'N/A')}<br/>
        <b>Worst Day:</b> {weekly_summary.get('worst_day', 'N/A')}<br/>
        <b>Average Volume:</b> {weekly_summary.get('avg_volume', 'N/A')}<br/>
        <b>Volatility:</b> {weekly_summary.get('volatility', 'N/A')}%
        """
        
        story.append(Paragraph("Weekly Summary", self.styles['SectionHeader']))
        story.append(Paragraph(summary_text, self.styles['Summary']))
        story.append(Spacer(1, 20))
        
        # Include daily report sections
        story.extend(self.create_sector_analysis(data.get('sectors', {})))
        story.extend(self.create_technical_analysis(data.get('technical', {})))
        
        # Risk Disclaimer
        story.append(PageBreak())
        story.extend(self.create_risk_disclaimer())
        
        doc.build(story)
        return filepath
    
    def generate_monthly_report(self, data: Dict, filename: Optional[str] = None) -> str:
        """Generate monthly market report"""
        if not filename:
            filename = f"monthly_report_{datetime.now().strftime('%Y_%m')}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(filepath, pagesize=self.page_size)
        story = []
        
        # Header
        story.extend(self.create_header(
            "Monthly Market Research Report", 
            datetime.now().strftime("%B %Y")
        ))
        
        # Monthly Performance
        monthly_data = data.get('monthly_summary', {})
        monthly_text = f"""
        <b>Monthly Return:</b> {monthly_data.get('return', 'N/A')}%<br/>
        <b>Best Week:</b> {monthly_data.get('best_week', 'N/A')}<br/>
        <b>Worst Week:</b> {monthly_data.get('worst_week', 'N/A')}<br/>
        <b>Total Volume:</b> {monthly_data.get('total_volume', 'N/A')}<br/>
        <b>Market Cap Change:</b> {monthly_data.get('market_cap_change', 'N/A')}%
        """
        
        story.append(Paragraph("Monthly Overview", self.styles['SectionHeader']))
        story.append(Paragraph(monthly_text, self.styles['Summary']))
        story.append(Spacer(1, 20))
        
        # Detailed Analysis
        story.extend(self.create_sector_analysis(data.get('sectors', {})))
        story.extend(self.create_technical_analysis(data.get('technical', {})))
        
        # Top Performers Table
        if 'top_performers' in data and isinstance(data['top_performers'], pd.DataFrame):
            story.extend(self.create_market_data_table(
                data['top_performers'], "Top Performers of the Month"
            ))
        
        # Risk Disclaimer
        story.append(PageBreak())
        story.extend(self.create_risk_disclaimer())
        
        doc.build(story)
        return filepath
    
    def generate_custom_report(self, data: Dict, title: str, filename: Optional[str] = None) -> str:
        """Generate custom report with flexible content"""
        if not filename:
            filename = f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(filepath, pagesize=self.page_size)
        story = []
        
        # Header
        story.extend(self.create_header(title, datetime.now().strftime("%B %d, %Y")))
        
        # Add sections based on data provided
        for section_name, section_data in data.items():
            if section_name == 'summary':
                story.extend(self.create_executive_summary(section_data))
            elif section_name == 'sectors':
                story.extend(self.create_sector_analysis(section_data))
            elif section_name == 'technical':
                story.extend(self.create_technical_analysis(section_data))
            elif section_name == 'tables' and isinstance(section_data, dict):
                for table_name, df in section_data.items():
                    if isinstance(df, pd.DataFrame):
                        story.extend(self.create_market_data_table(df, table_name))
            elif section_name == 'charts' and isinstance(section_data, dict):
                for chart_title, chart_path in section_data.items():
                    story.extend(self.add_chart_to_story(chart_path, chart_title))
        
        # Risk Disclaimer
        story.append(PageBreak())
        story.extend(self.create_risk_disclaimer())
        
        doc.build(story)
        return filepath