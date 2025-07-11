#!/usr/bin/env python3
"""
Market Research System v1.0 - Environment Setup Script
Indian Stock Market Focus
Created: January 2022
Author: Independent Market Researcher
"""

import os
import sys
import subprocess
import platform
import yaml
from pathlib import Path

class EnvironmentSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        self.reports_dir = self.project_root / "reports"
        
    def create_directory_structure(self):
        """Create all required directories for the project"""
        print("Creating directory structure...")
        
        directories = [
            # Data directories
            "data/raw/stocks/daily",
            "data/raw/stocks/intraday", 
            "data/raw/stocks/historical",
            "data/raw/economic/gdp",
            "data/raw/economic/unemployment",
            "data/raw/economic/inflation",
            "data/raw/economic/interest_rates",
            "data/raw/indices/nifty50",
            "data/raw/indices/sensex",
            "data/raw/indices/banknifty",
            "data/processed/stocks/cleaned",
            "data/processed/stocks/normalized",
            "data/processed/stocks/aggregated",
            "data/processed/economic/cleaned",
            "data/processed/economic/standardized",
            "data/processed/technical_indicators/moving_averages",
            "data/processed/technical_indicators/oscillators",
            "data/processed/technical_indicators/momentum",
            "data/cache/daily_cache",
            "data/cache/weekly_cache",
            "data/backups/daily",
            "data/backups/weekly",
            
            # Log directories
            "logs/application",
            "logs/system",
            "logs/archived/2022",
            
            # Report directories
            "reports/daily/2022",
            "reports/weekly/2022", 
            "reports/monthly/2022",
            "reports/templates",
            "reports/examples",
            "reports/archives",
            
            # Config directories
            "config/api_keys",
            "config/data_sources",
            "config/analysis",
            "config/reporting",
            "config/system"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {directory}")
            
        print("✅ Directory structure created successfully!")
        
    def install_dependencies(self):
        """Install required Python packages"""
        print("Installing Python dependencies...")
        
        # Core dependencies for 2022 setup
        dependencies = [
            "yfinance==0.1.87",  # 2022 version
            "pandas==1.5.3",
            "numpy==1.24.3", 
            "matplotlib==3.6.3",
            "seaborn==0.12.2",
            "scipy==1.10.1",
            "requests==2.28.2",
            "pyyaml==6.0",
            "python-dateutil==2.8.2",
            "pytz==2022.7.1",
            "ta-lib==0.4.25",  # Technical Analysis Library
            "reportlab==3.6.12",  # PDF generation
            "openpyxl==3.0.10",  # Excel support
            "xlrd==2.0.1",
            "beautifulsoup4==4.11.2",
            "lxml==4.9.2",
            "schedule==1.1.0",  # Task scheduling
            "smtplib",  # Email (built-in)
            "configparser",  # Configuration (built-in)
            "logging",  # Logging (built-in)
            "sqlite3",  # Database (built-in)
            "json",  # JSON handling (built-in)
            "csv",  # CSV handling (built-in)
        ]
        
        # Install packages
        for package in dependencies:
            if not self._is_built_in(package):
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"✅ Installed: {package}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {package}: {e}")
        
        print("✅ Dependencies installation completed!")
        
    def _is_built_in(self, package):
        """Check if package is built-in to Python"""
        built_in_modules = ['smtplib', 'configparser', 'logging', 'sqlite3', 'json', 'csv']
        return package in built_in_modules
        
    def create_config_files(self):
        """Create configuration files with Indian market focus"""
        print("Creating configuration files...")
        
        # API Keys configuration
        api_config = {
            'yahoo_finance': {
                'base_url': 'https://query1.finance.yahoo.com',
                'timeout': 30,
                'retry_attempts': 3
            },
            'rbi_data': {
                'base_url': 'https://rbi.org.in/Scripts/BS_ViewBulletin.aspx',
                'timeout': 30
            },
            'nse_data': {
                'base_url': 'https://www.nseindia.com',
                'timeout': 30
            }
        }
        
        with open(self.config_dir / "api_keys" / "api_config.yaml", 'w') as f:
            yaml.dump(api_config, f, default_flow_style=False)
            
        # Indian Stock Symbols (Top 50 Nifty stocks)
        nifty50_stocks = {
            'nifty50_stocks': [
                'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
                'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
                'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS',
                'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS',
                'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
                'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS',
                'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
                'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS',
                'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS',
                'SHREECEM.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS',
                'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS',
                'TITAN.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'UPL.NS',
                'WIPRO.NS', 'ZEEL.NS'
            ],
            'indices': [
                '^NSEI',  # Nifty 50
                '^BSESN',  # BSE Sensex
                '^NSEBANK',  # Bank Nifty
                'NIFTY_FIN_SERVICE.NS',  # Financial Services
                'NIFTY_FMCG.NS',  # FMCG
                'NIFTY_IT.NS',  # IT
                'NIFTY_PHARMA.NS',  # Pharma
                'NIFTY_METAL.NS',  # Metal
                'NIFTY_AUTO.NS',  # Auto
                'NIFTY_REALTY.NS'  # Realty
            ],
            'sectors': {
                'banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS'],
                'it': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'],
                'pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
                'auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'M&M.NS'],
                'fmcg': ['HINDUNILVR.NS', 'ITC.NS', 'BRITANNIA.NS', 'NESTLEIND.NS', 'TATACONSUM.NS'],
                'energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'COALINDIA.NS', 'NTPC.NS'],
                'metals': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'ADANIPORTS.NS']
            }
        }
        
        with open(self.config_dir / "data_sources" / "indian_stocks.yaml", 'w') as f:
            yaml.dump(nifty50_stocks, f, default_flow_style=False)
            
        # Technical Indicators Configuration
        technical_config = {
            'moving_averages': {
                'sma_periods': [5, 10, 20, 50, 100, 200],
                'ema_periods': [12, 26, 50, 100, 200]
            },
            'oscillators': {
                'rsi_period': 14,
                'stoch_k_period': 14,
                'stoch_d_period': 3,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            'bollinger_bands': {
                'period': 20,
                'std_dev': 2
            },
            'volume_indicators': {
                'volume_sma': 20,
                'vwap_period': 14
            }
        }
        
        with open(self.config_dir / "analysis" / "technical_indicators.yaml", 'w') as f:
            yaml.dump(technical_config, f, default_flow_style=False)
            
        # Logging Configuration
        logging_config = {
            'version': 1,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'simple': {
                    'format': '%(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'simple'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': 'logs/application/app.log',
                    'level': 'DEBUG',
                    'formatter': 'detailed'
                }
            },
            'loggers': {
                'market_research': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file']
                }
            }
        }
        
        with open(self.config_dir / "system" / "logging.yaml", 'w') as f:
            yaml.dump(logging_config, f, default_flow_style=False)
            
        print("✅ Configuration files created successfully!")
        
    def create_environment_file(self):
        """Create .env file template"""
        print("Creating environment file...")
        
        env_content = """# Market Research System v1.0 Environment Variables
# Indian Stock Market Focus - Created January 2022

# Database Configuration
DATABASE_URL=sqlite:///data/market_research.db
DATABASE_BACKUP_PATH=data/backups/

# API Configuration
YAHOO_FINANCE_TIMEOUT=30
RBI_DATA_TIMEOUT=30
NSE_DATA_TIMEOUT=30

# Email Configuration (for reports)
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_FROM=your_email@gmail.com

# Report Recipients
DAILY_REPORT_RECIPIENTS=client1@example.com,client2@example.com
WEEKLY_REPORT_RECIPIENTS=client1@example.com,client2@example.com

# Data Refresh Settings
MARKET_HOURS_START=09:15
MARKET_HOURS_END=15:30
TIMEZONE=Asia/Kolkata

# System Settings
LOG_LEVEL=INFO
MAX_LOG_SIZE=10MB
BACKUP_RETENTION_DAYS=30
CACHE_EXPIRY_HOURS=24

# Performance Settings
MAX_CONCURRENT_REQUESTS=10
REQUEST_DELAY_SECONDS=1
RETRY_ATTEMPTS=3

# Indian Market Specific
NSE_TRADING_DAYS=MON,TUE,WED,THU,FRI
BSE_TRADING_DAYS=MON,TUE,WED,THU,FRI
EXCLUDE_HOLIDAYS=True

# Report Generation
REPORT_GENERATION_TIME=18:00
WEEKEND_REPORTS=True
MONTHLY_REPORTS=True
"""
        
        with open(self.project_root / ".env.example", 'w') as f:
            f.write(env_content)
            
        print("✅ Environment file template created!")
        
    def create_gitignore(self):
        """Create .gitignore file"""
        print("Creating .gitignore file...")
        
        gitignore_content = """# Market Research System v1.0 - Git Ignore
# Created: January 2022

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/*.log
logs/**/*.log
*.log

# Data files
data/raw/**/*.csv
data/raw/**/*.json
data/processed/**/*.csv
data/processed/**/*.json
data/cache/**/*
data/backups/**/*

# Reports
reports/daily/**/*.pdf
reports/weekly/**/*.pdf
reports/monthly/**/*.pdf
reports/archives/**/*

# Configuration
.env
config/api_keys/*
!config/api_keys/*.example
config/**/*.yaml
!config/**/*.example.yaml

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Database
*.db
*.sqlite
*.sqlite3

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
"""
        
        with open(self.project_root / ".gitignore", 'w') as f:
            f.write(gitignore_content)
            
        print("✅ .gitignore file created!")
        
    def create_requirements_file(self):
        """Create requirements.txt file"""
        print("Creating requirements.txt file...")
        
        requirements_content = """# Market Research System v1.0 - Python Dependencies
# Indian Stock Market Focus - Created January 2022

# Core Data Analysis
pandas==1.5.3
numpy==1.24.3
scipy==1.10.1

# Financial Data
yfinance==0.1.87
requests==2.28.2
beautifulsoup4==4.11.2
lxml==4.9.2

# Technical Analysis
TA-Lib==0.4.25

# Data Visualization
matplotlib==3.6.3
seaborn==0.12.2

# Report Generation
reportlab==3.6.12
openpyxl==3.0.10
xlrd==2.0.1

# Configuration and Utilities
PyYAML==6.0
python-dateutil==2.8.2
pytz==2022.7.1
schedule==1.1.0

# Database
SQLAlchemy==1.4.46
"""
        
        with open(self.project_root / "requirements.txt", 'w') as f:
            f.write(requirements_content)
            
        print("✅ requirements.txt file created!")
        
    def setup_database(self):
        """Initialize SQLite database"""
        print("Setting up database...")
        
        import sqlite3
        
        db_path = self.project_root / "data" / "market_research.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables for Indian stock data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                indicator_name TEXT NOT NULL,
                value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date, indicator_name)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_indices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(index_symbol, date)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT NOT NULL,
                report_date DATE NOT NULL,
                file_path TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("✅ Database setup completed!")
        
    def run_setup(self):
        """Run complete environment setup"""
        print("🚀 Starting Market Research System v1.0 Environment Setup")
        print("Focus: Indian Stock Market")
        print("Created: January 2022")
        print("=" * 60)
        
        try:
            self.create_directory_structure()
            self.install_dependencies()
            self.create_config_files()
            self.create_environment_file()
            self.create_gitignore()
            self.create_requirements_file()
            self.setup_database()
            
            print("\n" + "=" * 60)
            print("✅ Environment setup completed successfully!")
            print("\nNext steps:")
            print("1. Copy .env.example to .env and configure your settings")
            print("2. Run 'python scripts/utilities/test_api_connections.py' to test APIs")
            print("3. Run 'python scripts/utilities/generate_sample_data.py' to generate test data")
            print("4. Start using the system with daily data collection")
            print("\n📊 Ready to analyze Indian stock markets!")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.run_setup()