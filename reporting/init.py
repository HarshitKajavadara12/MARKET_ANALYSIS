"""
Market Research System v1.0 - Reporting Package
===============================================

This package provides comprehensive reporting capabilities for market research,
including PDF generation, visualization, and email delivery.

Author: Independent Market Researcher
Created: 2022
Version: 1.0
"""

from .pdf_generator import PDFGenerator
from .visualization import MarketVisualizer
from .report_templates import ReportTemplates
from .chart_utils import ChartUtils
from .table_generator import TableGenerator
from .email_sender import EmailSender

__version__ = "1.0.0"
__author__ = "Independent Market Researcher"
__email__ = "research@marketanalytics.com"

# Package level imports for easy access
__all__ = [
    'PDFGenerator',
    'MarketVisualizer', 
    'ReportTemplates',
    'ChartUtils',
    'TableGenerator',
    'EmailSender'
]

# Default configuration
DEFAULT_CONFIG = {
    'report_output_dir': 'reports/',
    'chart_style': 'seaborn',
    'pdf_page_size': 'A4',
    'email_smtp_server': 'smtp.gmail.com',
    'email_smtp_port': 587
}

def get_version():
    """Return package version"""
    return __version__

def get_default_config():
    """Return default configuration dictionary"""
    return DEFAULT_CONFIG.copy()