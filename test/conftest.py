"""
Pytest configuration file for the Market Research System v1.0
Provides fixtures, markers, and global test configuration.

Created: 2022
Author: Market Research System Team
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import shutil
from unittest.mock import Mock, patch
import yfinance as yf
import logging
import warnings

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.
    """
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower)"
    )
    config.addinivalue_line(
        "markers", "functional: marks tests as functional tests (end-to-end)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require external API calls"
    )
    config.addinivalue_line(
        "markers", "data: marks tests that work with data files"
    )
    config.addinivalue_line(
        "markers", "indian_market: marks tests specific to Indian stock market"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers based on test location and names.
    """
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "functional" in str(item.fspath):
            item.add_marker(pytest.mark.functional)
        
        # Add markers based on test names
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        if "api" in item.name or "fetch" in item.name:
            item.add_marker(pytest.mark.api)
        
        if "indian" in item.name or "nse" in item.name or ".NS" in str(item):
            item.add_marker(pytest.mark.indian_market)


@pytest.fixture(scope="session")
def test_config():
    """
    Global test configuration fixture.
    """
    return {
        'test_data_path': 'tests/fixtures/',
        'temp_output_path': tempfile.mkdtemp(prefix='market_research_test_'),
        'indian_stock_symbols': [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 
            'ICICIBANK.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS',
            'BHARTIARTL.NS', 'KOTAKBANK.NS'
        ],
        'test_date_range': {
            'start': '2022-01-01',
            'end': '2022-12-31'
        },
        'api_timeout': 30,
        'max_retries': 3
    }


@pytest.fixture(scope="function")
def temp_directory():
    """
    Create a temporary directory for test files.
    Automatically cleaned up after each test.
    """
    temp_dir = tempfile.mkdtemp(prefix='market_test_')
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def sample_stock_data():
    """
    Generate sample Indian stock market data for testing.
    """
    dates = pd.date_range('2022-01-01', periods=252, freq='B')  # Business days only
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic Indian stock prices (Reliance as example)
    base_price = 2400.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    close_prices = np.array(prices)
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, len(dates)))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.02, len(dates)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.02, len(dates)))
    volumes = np.random.lognormal(14, 0.5, len(dates))  # Log-normal distribution for volume
    
    return pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes.astype(int),
        'Adj Close': close_prices
    }, index=dates)


@pytest.fixture(scope="function")
def sample_economic_data():
    """
    Generate sample Indian economic indicators data for testing.
    """
    dates = pd.date_range('2022-01-01', periods=12, freq='M')  # Monthly data
    
    return pd.DataFrame({
        'GDP_Growth': [7.2, 7.4, 7.1, 6.9, 7.0, 7.3, 7.5, 7.2, 6.8, 7.1, 7.4, 7.6],
        'Inflation_CPI': [4.5, 4.8, 5.1, 4.9, 4.7, 5.2, 5.5, 5.8, 6.1, 5.9, 5.4, 5.0],
        'Repo_Rate': [4.0, 4.0, 4.0, 4.0, 4.4, 4.9, 5.4, 5.4, 5.9, 6.25, 6.25, 6.25],
        'Industrial_Production': [2.1, 2.5, 1.8, 2.7, 3.2, 2.9, 3.5, 4.1, 3.8, 3.2, 2.8, 3.0],
        'Nifty_50': [17354, 16247, 17465, 17222, 15735, 15780, 15692, 17102, 17618, 18014, 18618, 18105]
    }, index=dates)


@pytest.fixture(scope="function")
def sample_nse_indices():
    """
    Generate sample NSE indices data for testing.
    """
    dates = pd.date_range('2022-01-01', periods=252, freq='B')
    np.random.seed(123)
    
    # Base values for different indices
    base_values = {
        'NIFTY_50': 17354.05,
        'NIFTY_BANK': 36544.85,
        'NIFTY_IT': 34281.15,
        'NIFTY_AUTO': 13127.20,
        'NIFTY_PHARMA': 12845.30
    }
    
    indices_data = {}
    for index_name, base_value in base_values.items():
        returns = np.random.normal(0.0008, 0.015, len(dates))
        values = [base_value]
        for ret in returns[1:]:
            values.append(values[-1] * (1 + ret))
        indices_data[index_name] = values
    
    return pd.DataFrame(indices_data, index=dates)


@pytest.fixture(scope="function")
def mock_yfinance():
    """
    Mock yfinance API responses for testing.
    """
    with patch('yfinance.download') as mock_download, \
         patch('yfinance.Ticker') as mock_ticker:
        
        # Configure mock download
        def mock_download_func(tickers, start=None, end=None, **kwargs):
            dates = pd.date_range(start or '2022-01-01', end or '2022-12-31', freq='B')
            np.random.seed(42)
            
            if isinstance(tickers, str):
                tickers = [tickers]
            
            data = {}
            for ticker in tickers:
                base_price = 2400.0 if 'RELIANCE' in ticker else 3500.0
                returns = np.random.normal(0.001, 0.02, len(dates))
                prices = [base_price]
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                data[ticker] = pd.DataFrame({
                    'Open': prices,
                    'High': [p * 1.02 for p in prices],
                    'Low': [p * 0.98 for p in prices],
                    'Close': prices,
                    'Volume': np.random.randint(100000, 2000000, len(dates)),
                    'Adj Close': prices
                }, index=dates)
            
            if len(tickers) == 1:
                return data[tickers[0]]
            return pd.concat(data, axis=1)
        
        mock_download.side_effect = mock_download_func
        
        # Configure mock ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'sector': 'Energy',
            'industry': 'Oil & Gas Refining & Marketing',
            'marketCap': 1500000000000,
            'currency': 'INR'
        }
        mock_ticker.return_value = mock_ticker_instance
        
        yield {
            'download': mock_download,
            'ticker': mock_ticker
        }


@pytest.fixture(scope="function")
def mock_fred_api():
    """
    Mock FRED API for economic data testing.
    """
    with patch('fredapi.Fred') as mock_fred:
        mock_fred_instance = Mock()
        
        def mock_get_series(series_id, start=None, end=None):
            dates = pd.date_range(start or '2022-01-01', end or '2022-12-31', freq='M')
            if 'GDP' in series_id:
                return pd.Series(np.random.uniform(6.5, 7.5, len(dates)), index=dates)
            elif 'UNRATE' in series_id:
                return pd.Series(np.random.uniform(6.0, 8.0, len(dates)), index=dates)
            elif 'CPIAUCSL' in series_id:
                return pd.Series(np.random.uniform(4.0, 6.0, len(dates)), index=dates)
            else:
                return pd.Series(np.random.randn(len(dates)), index=dates)
        
        mock_fred_instance.get_series.side_effect = mock_get_series
        mock_fred.return_value = mock_fred_instance
        
        yield mock_fred


@pytest.fixture(scope="function")
def database_connection():
    """
    Mock database connection for testing.
    """
    connection = Mock()
    connection.execute.return_value = Mock()
    connection.fetchall.return_value = []
    connection.commit.return_value = None
    connection.close.return_value = None
    
    return connection


@pytest.fixture(scope="function")
def sample_technical_indicators():
    """
    Generate sample technical indicators data for Indian stocks testing.
    """
    dates = pd.date_range('2022-01-01', periods=252, freq='B')
    np.random.seed(42)
    
    # Sample data for RELIANCE.NS
    close_prices = np.random.normal(2400, 200, len(dates))
    close_prices = np.cumsum(np.random.normal(0, 5, len(dates))) + 2400
    
    # Calculate various technical indicators
    sma_20 = pd.Series(close_prices).rolling(window=20).mean()
    sma_50 = pd.Series(close_prices).rolling(window=50).mean()
    ema_12 = pd.Series(close_prices).ewm(span=12).mean()
    ema_26 = pd.Series(close_prices).ewm(span=26).mean()
    
    # RSI calculation
    delta = pd.Series(close_prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    macd_histogram = macd - macd_signal
    
    return pd.DataFrame({
        'Close': close_prices,
        'SMA_20': sma_20,
        'SMA_50': sma_50,
        'EMA_12': ema_12,
        'EMA_26': ema_26,
        'RSI': rsi,
        'MACD': macd,
        'MACD_Signal': macd_signal,
        'MACD_Histogram': macd_histogram,
        'Bollinger_Upper': sma_20 + (pd.Series(close_prices).rolling(20).std() * 2),
        'Bollinger_Lower': sma_20 - (pd.Series(close_prices).rolling(20).std() * 2)
    }, index=dates)


@pytest.fixture(scope="function")
def sample_indian_market_data():
    """
    Generate comprehensive Indian market data including NSE, BSE indices and major stocks.
    """
    dates = pd.date_range('2022-01-01', periods=252, freq='B')
    np.random.seed(42)
    
    # Major Indian indices
    nifty_50_base = 17354.05
    sensex_base = 58253.82
    bank_nifty_base = 36544.85
    
    # Generate index data
    nifty_returns = np.random.normal(0.0008, 0.015, len(dates))
    sensex_returns = np.random.normal(0.0007, 0.016, len(dates))
    bank_nifty_returns = np.random.normal(0.001, 0.02, len(dates))
    
    nifty_values = [nifty_50_base]
    sensex_values = [sensex_base]
    bank_nifty_values = [bank_nifty_base]
    
    for i in range(1, len(dates)):
        nifty_values.append(nifty_values[-1] * (1 + nifty_returns[i]))
        sensex_values.append(sensex_values[-1] * (1 + sensex_returns[i]))
        bank_nifty_values.append(bank_nifty_values[-1] * (1 + bank_nifty_returns[i]))
    
    return pd.DataFrame({
        'NIFTY_50': nifty_values,
        'SENSEX': sensex_values,
        'BANK_NIFTY': bank_nifty_values,
        'NIFTY_IT': np.random.normal(34281, 1500, len(dates)),
        'NIFTY_AUTO': np.random.normal(13127, 800, len(dates)),
        'NIFTY_PHARMA': np.random.normal(12845, 600, len(dates)),
        'NIFTY_FMCG': np.random.normal(18500, 900, len(dates))
    }, index=dates)


@pytest.fixture(scope="function")
def indian_stock_symbols():
    """
    Provide list of major Indian stock symbols for testing.
    """
    return {
        'large_cap': [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
            'HINDUNILVR.NS', 'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS',
            'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
            'MARUTI.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS'
        ],
        'mid_cap': [
            'PAGEIND.NS', 'GODREJCP.NS', 'MCDOWELL-N.NS', 'DABUR.NS',
            'BIOCON.NS', 'LUPIN.NS', 'MINDTREE.NS', 'MPHASIS.NS'
        ],
        'small_cap': [
            'RBLBANK.NS', 'IDFCFIRSTB.NS', 'FEDERALBNK.NS', 'BANKBARODA.NS'
        ],
        'sectors': {
            'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
            'Energy': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS'],
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'],
            'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS']
        }
    }


@pytest.fixture(scope="function")
def indian_market_timings():
    """
    Provide Indian market timing information for testing.
    """
    return {
        'NSE': {
            'pre_open': '09:00',
            'normal_market': '09:15',
            'market_close': '15:30',
            'post_market': '15:40',
            'timezone': 'Asia/Kolkata'
        },
        'BSE': {
            'pre_open': '09:00',
            'normal_market': '09:15', 
            'market_close': '15:30',
            'post_market': '15:40',
            'timezone': 'Asia/Kolkata'
        },
        'trading_holidays_2022': [
            '2022-01-26',  # Republic Day
            '2022-03-01',  # Maha Shivratri
            '2022-03-18',  # Holi
            '2022-04-14',  # Dr. Baba Saheb Ambedkar Jayanti
            '2022-04-15',  # Good Friday
            '2022-05-03',  # Eid ul Fitr
            '2022-08-09',  # Muharram
            '2022-08-15',  # Independence Day
            '2022-08-31',  # Ganesh Chaturthi
            '2022-10-05',  # Dussehra
            '2022-10-24',  # Diwali Laxmi Pujan
            '2022-10-26',  # Diwali Balipratipada
            '2022-11-08'   # Guru Nanak Jayanti
        ]
    }


@pytest.fixture(scope="function")
def sample_portfolio_data():
    """
    Generate sample Indian portfolio data for testing.
    """
    stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS']
    
    return pd.DataFrame({
        'Symbol': stocks,
        'Quantity': [50, 100, 75, 200, 500],
        'Avg_Buy_Price': [2250.50, 3420.75, 1540.25, 1650.00, 245.80],
        'Current_Price': [2380.25, 3550.60, 1580.40, 1720.30, 255.90],
        'Sector': ['Energy', 'IT', 'Banking', 'IT', 'FMCG'],
        'Market_Cap': ['Large', 'Large', 'Large', 'Large', 'Large'],
        'Currency': ['INR'] * 5
    })


@pytest.fixture(scope="function")
def sample_fundamental_data():
    """
    Generate sample fundamental analysis data for Indian stocks.
    """
    return {
        'RELIANCE.NS': {
            'PE_Ratio': 24.5,
            'PB_Ratio': 1.8,
            'Debt_to_Equity': 0.35,
            'ROE': 12.8,
            'ROA': 5.2,
            'Current_Ratio': 1.4,
            'Quick_Ratio': 0.9,
            'Dividend_Yield': 0.8,
            'EPS': 98.5,
            'Book_Value': 1325.6,
            'Market_Cap': 1450000,  # in crores
            'Revenue_Growth': 8.5,
            'Profit_Growth': 12.3
        },
        'TCS.NS': {
            'PE_Ratio': 28.2,
            'PB_Ratio': 12.5,
            'Debt_to_Equity': 0.02,
            'ROE': 45.2,
            'ROA': 28.5,
            'Current_Ratio': 3.2,
            'Quick_Ratio': 3.1,
            'Dividend_Yield': 1.2,
            'EPS': 125.4,
            'Book_Value': 285.7,
            'Market_Cap': 1320000,
            'Revenue_Growth': 15.2,
            'Profit_Growth': 18.7
        }
    }


@pytest.fixture(scope="function")
def risk_free_rate():
    """
    Provide Indian risk-free rate (10-year G-Sec yield) for 2022.
    """
    dates = pd.date_range('2022-01-01', periods=252, freq='B')
    # Indian 10-year G-Sec yield fluctuated between 6.1% to 7.4% in 2022
    base_rate = 6.8
    variations = np.random.normal(0, 0.1, len(dates))
    rates = np.clip(base_rate + np.cumsum(variations * 0.01), 6.0, 7.5)
    
    return pd.Series(rates / 100, index=dates)  # Convert percentage to decimal


@pytest.fixture(scope="function")
def market_benchmark():
    """
    Provide benchmark returns (NIFTY 50) for testing performance calculations.
    """
    dates = pd.date_range('2022-01-01', periods=252, freq='B')
    np.random.seed(42)
    daily_returns = np.random.normal(0.0008, 0.015, len(dates))
    return pd.Series(daily_returns, index=dates)


@pytest.fixture(scope="function")
def logging_config():
    """
    Configure logging for tests.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('market_research_test')


@pytest.fixture(scope="session", autouse=True)
def cleanup_session():
    """
    Session-level cleanup fixture.
    """
    yield
    # Cleanup any session-level resources
    temp_dirs = [d for d in os.listdir(tempfile.gettempdir()) if d.startswith('market_research_test_')]
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(os.path.join(tempfile.gettempdir(), temp_dir))
        except Exception:
            pass


# Custom markers for Indian market specific tests
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]


# Test data validation helpers
def validate_indian_stock_data(df):
    """
    Validate Indian stock data format and content.
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    assert all(col in df.columns for col in required_columns), "Missing required OHLCV columns"
    assert df.index.freq == 'B' or df.index.freq is None, "Should have business day frequency"
    assert (df['High'] >= df['Low']).all(), "High should be >= Low"
    assert (df['High'] >= df['Close']).all(), "High should be >= Close"
    assert (df['Low'] <= df['Close']).all(), "Low should be <= Close"
    assert (df['Volume'] >= 0).all(), "Volume should be non-negative"
    return True


def validate_technical_indicators(df):
    """
    Validate technical indicators data.
    """
    if 'RSI' in df.columns:
        assert (df['RSI'] >= 0).all() and (df['RSI'] <= 100).all(), "RSI should be between 0-100"
    
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        assert not df['MACD'].isna().all(), "MACD should have valid values"
        assert not df['MACD_Signal'].isna().all(), "MACD Signal should have valid values"
    
    return True