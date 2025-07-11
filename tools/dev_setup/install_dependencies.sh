#!/bin/bash

# Market Research System v1.0 - Development Dependencies Installation
# Created: January 2022
# Author: Independent Market Researcher

set -e

echo "🚀 Setting up Market Research System v1.0 Development Environment"
echo "================================================================"

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' || echo "0.0")
min_version="3.8"

if [ "$(printf '%s\n' "$min_version" "$python_version" | sort -V | head -n1)" != "$min_version" ]; then
    echo "❌ Error: Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core Python packages..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -r requirements-dev.txt

# Install TA-Lib (Technical Analysis Library)
echo "📊 Installing TA-Lib for technical indicators..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    sudo apt-get update
    sudo apt-get install -y build-essential wget
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
    pip install TA-Lib
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is required for macOS installation"
        exit 1
    fi
    brew install ta-lib
    pip install TA-Lib
else
    echo "⚠️ Manual TA-Lib installation required for your OS"
    pip install TA-Lib || echo "⚠️ TA-Lib installation failed - install manually"
fi

# Install system dependencies for data processing
echo "🔧 Installing system dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install -y sqlite3 libsqlite3-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install sqlite
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw/stocks/{daily,intraday,historical}
mkdir -p data/raw/economic/{gdp,unemployment,inflation,interest_rates}
mkdir -p data/raw/indices/{nifty50,sensex,banknifty}
mkdir -p data/processed/{stocks,economic,technical_indicators}
mkdir -p data/cache/{daily_cache,weekly_cache}
mkdir -p data/backups/{daily,weekly}
mkdir -p reports/{daily,weekly,monthly}/{2022,2023,2024,2025}
mkdir -p logs/{application,system,archived}
mkdir -p config/{api_keys,data_sources,analysis,reporting,system}

# Set permissions
chmod +x scripts/data_collection/*.py
chmod +x scripts/analysis/*.py
chmod +x scripts/reporting/*.py
chmod +x scripts/maintenance/*.py

# Initialize database
echo "🗄️ Initializing database..."
if [ -f "tools/database/create_tables.sql" ]; then
    sqlite3 data/market_research.db < tools/database/create_tables.sql
    echo "✅ Database tables created"
fi

# Setup git hooks
echo "🔗 Setting up git hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
fi

# Create sample configuration files
echo "⚙️ Creating sample configuration files..."
cp config/api_keys/nse_api.yaml.example config/api_keys/nse_api.yaml 2>/dev/null || true
cp config/api_keys/yahoo_finance.yaml.example config/api_keys/yahoo_finance.yaml 2>/dev/null || true
cp config/api_keys/fred_api.yaml.example config/api_keys/fred_api.yaml 2>/dev/null || true

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import sqlite3
print('✅ Core packages imported successfully')

# Test data fetching
try:
    data = yf.download('RELIANCE.NS', period='5d', interval='1d')
    print('✅ Yahoo Finance API working - sample data fetched')
except Exception as e:
    print(f'⚠️ Yahoo Finance test failed: {e}')

# Test database connection
try:
    conn = sqlite3.connect('data/market_research.db')
    conn.close()
    print('✅ Database connection working')
except Exception as e:
    print(f'⚠️ Database test failed: {e}')
"

echo ""
echo "🎉 Development environment setup completed!"
echo "================================================================"
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Configure API keys in config/api_keys/"
echo "3. Run initial data collection: python scripts/data_collection/collect_daily_data.py"
echo "4. Generate your first report: python scripts/reporting/generate_daily_report.py"
echo ""
echo "📚 Documentation: docs/user_guide/installation.md"
echo "🐛 Issues: Check logs/ directory for any errors"
echo "💡 Support: See docs/user_guide/troubleshooting.md"