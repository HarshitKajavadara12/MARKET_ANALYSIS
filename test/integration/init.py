#!/usr/bin/env python3
"""
Integration Tests Package Initialization
Market Research System v1.0 - 2022
Indian Stock Market Focus

This package contains integration tests for the market research system,
testing end-to-end workflows and component interactions.
"""

__version__ = "1.0.0"
__author__ = "Market Research Team"
__email__ = "research@marketanalysis.com"
__created__ = "2022-01-15"

# Test configuration constants
TEST_SYMBOLS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'HDFC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS'
]

TEST_INDICES = ['^NSEI', '^BSESN']  # NIFTY 50, BSE SENSEX

TEST_ECONOMIC_INDICATORS = [
    'INDCPIALLMINMEI',  # India CPI
    'INDGDPQDSMEI',     # India GDP
    'INDUNEMRTLMEI'     # India Unemployment Rate
]

# Test data date ranges
TEST_START_DATE = '2020-01-01'
TEST_END_DATE = '2022-12-31'

# API test configurations
API_TIMEOUT = 30
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1  # seconds

# Expected data quality thresholds
MIN_DATA_COMPLETENESS = 0.95  # 95% data availability
MAX_OUTLIER_PERCENTAGE = 0.05  # 5% outliers allowed
MIN_CORRELATION_THRESHOLD = 0.1  # Minimum correlation for validation

# Test report configurations
TEST_REPORT_DIR = 'test_reports'
TEST_LOG_DIR = 'test_logs'
INTEGRATION_TEST_TIMEOUT = 300  # 5 minutes per test

print(f"Integration Tests Package v{__version__} initialized")
print(f"Testing {len(TEST_SYMBOLS)} Indian stocks and {len(TEST_INDICES)} indices")
print(f"Economic indicators: {len(TEST_ECONOMIC_INDICATORS)} metrics")