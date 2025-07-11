"""
Database Configuration Module for Market Research System v1.0
Author: Market Research Team
Created: January 2022
Focus: SQLite and CSV-based data storage for Indian Stock Market

This module handles database configurations, connections, and table schemas
for storing Indian market research data.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime


class DatabaseConfig:
    """
    Database configuration and management for market research system
    Supports SQLite for structured data and CSV for raw data storage
    """
    
    def __init__(self, base_path: str = "data/"):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "processed" / "market_research.db"
        self.raw_data_path = self.base_path / "raw"
        self.processed_data_path = self.base_path / "processed"
        
        # Ensure directories exist
        self._create_directories()
        
        # Database connection
        self.connection = None
        
        # Table schemas
        self.schemas = self._define_schemas()
        
        # Initialize database
        self._initialize_database()
    
    def _create_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.raw_data_path / "stocks" / "daily",
            self.raw_data_path / "stocks" / "intraday", 
            self.raw_data_path / "stocks" / "historical",
            self.raw_data_path / "economic",
            self.raw_data_path / "indices",
            self.processed_data_path / "stocks" / "cleaned",
            self.processed_data_path / "stocks" / "normalized",
            self.processed_data_path / "technical_indicators",
            self.base_path / "cache",
            self.base_path / "backups" / "daily"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"Database directories created at: {self.base_path}")
    
    def _define_schemas(self) -> Dict[str, str]:
        """
        Define database table schemas for Indian market data
        
        Returns:
            Dictionary of table creation SQL statements
        """
        schemas = {
            # Stock price data table
            'stock_prices': """
                CREATE TABLE IF NOT EXISTS stock_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    adj_close REAL,
                    volume INTEGER,
                    exchange TEXT DEFAULT 'NSE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """,
            
            # Market indices data
            'market_indices': """
                CREATE TABLE IF NOT EXISTS market_indices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    index_symbol TEXT NOT NULL,
                    index_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER,
                    change_percent REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(index_symbol, date)
                )
            """,
            
            # Technical indicators storage
            'technical_indicators': """
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    indicator_name TEXT NOT NULL,
                    indicator_value REAL,
                    period INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date, indicator_name, period)
                )
            """,
            
            # Economic indicators (Indian economy specific)
            'economic_indicators': """
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(indicator_name, date)
                )
            """,
            
            # Stock metadata
            'stock_metadata': """
                CREATE TABLE IF NOT EXISTS stock_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    company_name TEXT NOT NULL,
                    sector TEXT,
                    market_cap_category TEXT,
                    exchange TEXT DEFAULT 'NSE',
                    is_active BOOLEAN DEFAULT 1,
                    added_date DATE DEFAULT CURRENT_DATE,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            # Sector performance tracking
            'sector_performance': """
                CREATE TABLE IF NOT EXISTS sector_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sector_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    performance_percent REAL,
                    market_cap REAL,
                    avg_volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(sector_name, date)
                )
            """,
            
            # Data quality logs
            'data_quality_logs': """
                CREATE TABLE IF NOT EXISTS data_quality_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    check_date DATE NOT NULL,
                    total_records INTEGER,
                    missing_records INTEGER,
                    duplicate_records INTEGER,
                    quality_score REAL,
                    issues TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            # System logs
            'system_logs': """
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_level TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    error_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
        
        return schemas
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            cursor = self.connection.cursor()
            for table_name, schema in self.schemas.items():
                cursor.execute(schema)
                print(f"Table '{table_name}' created/verified")
            
            # Create indices for better performance
            self._create_indices(cursor)
            
            self.connection.commit()
            print(f"Database initialized at: {self.db_path}")
            
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            raise
    
    def _create_indices(self, cursor):
        """Create database indices for better query performance"""
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(date)",
            "CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_indices_symbol_date ON market_indices(index_symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date ON technical_indicators(symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_economic_date ON economic_indicators(date)",
            "CREATE INDEX IF NOT EXISTS idx_sector_perf_date ON sector_performance(date)"
        ]
        
        for index_sql in indices:
            cursor.execute(index_sql)
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get database connection
        
        Returns:
            SQLite connection object
        """
        if self.connection is None:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
        
        return self.connection
    
    def execute_query(self, query: str, params: tuple = None) -> List[sqlite3.Row]:
        """
        Execute SQL query and return results
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of database rows
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            return cursor.fetchall()
        
        except sqlite3.Error as e:
            print(f"Query execution error: {e}")
            raise
    
    def insert_stock_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Insert stock price data into database
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            
        Returns:
            Success status
        """
        try:
            conn = self.get_connection()
            
            # Prepare data for insertion
            data_copy = data.copy()
            data_copy['symbol'] = symbol
            data_copy['exchange'] = 'NSE'
            
            # Insert data
            data_copy.to_sql('stock_prices', conn, if_exists='append', index=False)
            conn.commit()
            
            print(f"Inserted {len(data_copy)} records for {symbol}")
            return True
            
        except Exception as e:
            print(f"Error inserting stock data for {symbol}: {e}")
            return False
    
    def insert_technical_indicators(self, symbol: str, indicators: Dict[str, Any]) -> bool:
        """
        Insert technical indicators into database
        
        Args:
            symbol: Stock symbol
            indicators: Dictionary of indicator data
            
        Returns:
            Success status
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for indicator_name, values in indicators.items():
                if isinstance(values, pd.Series):
                    for date, value in values.items():
                        cursor.execute("""
                            INSERT OR REPLACE INTO technical_indicators 
                            (symbol, date, indicator_name, indicator_value, period)
                            VALUES (?, ?, ?, ?, ?)
                        """, (symbol, date, indicator_name, float(value), 14))  # Default period
            
            conn.commit()
            print(f"Inserted technical indicators for {symbol}")
            return True
            
        except Exception as e:
            print(f"Error inserting technical indicators for {symbol}: {e}")
            return False
    
    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve stock data from database
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with stock data
        """
        query = "SELECT * FROM stock_prices WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        try:
            conn = self.get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error retrieving stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_file_paths(self) -> Dict[str, Path]:
        """
        Get standardized file paths for different data types
        
        Returns:
            Dictionary of file paths
        """
        return {
            'raw_stocks': self.raw_data_path / "stocks",
            'raw_economic': self.raw_data_path / "economic", 
            'raw_indices': self.raw_data_path / "indices",
            'processed_stocks': self.processed_data_path / "stocks" / "cleaned",
            'indicators': self.processed_data_path / "technical_indicators",
            'cache': self.base_path / "cache",
            'backups': self.base_path / "backups"
        }
    
    def save_to_csv(self, data: pd.DataFrame, file_type: str, filename: str) -> bool:
        """
        Save DataFrame to CSV file in appropriate directory
        
        Args:
            data: DataFrame to save
            file_type: Type of data ('raw_stocks', 'processed_stocks', etc.)
            filename: Name of the file
            
        Returns:
            Success status
        """
        try:
            paths = self.get_file_paths()
            if file_type not in paths:
                print(f"Unknown file type: {file_type}")
                return False
            
            file_path = paths[file_type] / f"{filename}.csv"
            data.to_csv(file_path, index=True)
            
            print(f"Data saved to: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving CSV file: {e}")
            return False
    
    def load_from_csv(self, file_type: str, filename: str) -> pd.DataFrame:
        """
        Load DataFrame from CSV file
        
        Args:
            file_type: Type of data
            filename: Name of the file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            paths = self.get_file_paths()
            if file_type not in paths:
                print(f"Unknown file type: {file_type}")
                return pd.DataFrame()
            
            file_path = paths[file_type] / f"{filename}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                return df
            else:
                print(f"File not found: {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return pd.DataFrame()
    
    def log_data_quality(self, table_name: str, quality_metrics: Dict[str, Any]):
        """
        Log data quality metrics to database
        
        Args:
            table_name: Name of the table being checked
            quality_metrics: Dictionary with quality metrics
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO data_quality_logs 
                (table_name, check_date, total_records, missing_records, 
                 duplicate_records, quality_score, issues)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                table_name,
                datetime.now().date(),
                quality_metrics.get('total_records', 0),
                quality_metrics.get('missing_records', 0),
                quality_metrics.get('duplicate_records', 0),
                quality_metrics.get('quality_score', 0.0),
                json.dumps(quality_metrics.get('issues', []))
            ))
            
            conn.commit()
            
        except Exception as e:
            print(f"Error logging data quality: {e}")
    
    def backup_database(self) -> bool:
        """
        Create database backup
        
        Returns:
            Success status
        """
        try:
            backup_path = self.base_path / "backups" / "daily" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            
            # Create backup directory if it doesn't exist
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup using SQLite backup API
            source = self.get_connection()
            backup_conn = sqlite3.connect(str(backup_path))
            
            source.backup(backup_conn)
            backup_conn.close()
            
            print(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            print(f"Error creating database backup: {e}")
            return False
    
    def get_indian_stock_symbols(self) -> List[str]:
        """
        Get list of Indian stock symbols from metadata table
        
        Returns:
            List of stock symbols
        """
        try:
            query = "SELECT symbol FROM stock_metadata WHERE is_active = 1 ORDER BY symbol"
            rows = self.execute_query(query)
            return [row['symbol'] for row in rows]
            
        except Exception as e:
            print(f"Error getting stock symbols: {e}")
            return []
    
    def add_indian_stocks_metadata(self):
        """
        Add popular Indian stocks metadata to database
        This is a one-time setup for Version 1 (2022)
        """
        indian_stocks = [
            # Large Cap Stocks
            ('RELIANCE.NS', 'Reliance Industries Ltd', 'Energy', 'Large Cap'),
            ('TCS.NS', 'Tata Consultancy Services Ltd', 'IT', 'Large Cap'),
            ('HDFCBANK.NS', 'HDFC Bank Ltd', 'Banking', 'Large Cap'),
            ('INFY.NS', 'Infosys Ltd', 'IT', 'Large Cap'),
            ('HINDUNILVR.NS', 'Hindustan Unilever Ltd', 'FMCG', 'Large Cap'),
            ('ICICIBANK.NS', 'ICICI Bank Ltd', 'Banking', 'Large Cap'),
            ('KOTAKBANK.NS', 'Kotak Mahindra Bank Ltd', 'Banking', 'Large Cap'),
            ('BHARTIARTL.NS', 'Bharti Airtel Ltd', 'Telecom', 'Large Cap'),
            ('ITC.NS', 'ITC Ltd', 'FMCG', 'Large Cap'),
            ('SBIN.NS', 'State Bank of India', 'Banking', 'Large Cap'),
            ('LT.NS', 'Larsen & Toubro Ltd', 'Construction', 'Large Cap'),
            ('ASIANPAINT.NS', 'Asian Paints Ltd', 'Chemicals', 'Large Cap'),
            ('MARUTI.NS', 'Maruti Suzuki India Ltd', 'Automobile', 'Large Cap'),
            ('HCLTECH.NS', 'HCL Technologies Ltd', 'IT', 'Large Cap'),
            ('WIPRO.NS', 'Wipro Ltd', 'IT', 'Large Cap'),
            
            # Mid Cap Stocks
            ('GODREJCP.NS', 'Godrej Consumer Products Ltd', 'FMCG', 'Mid Cap'),
            ('PIDILITIND.NS', 'Pidilite Industries Ltd', 'Chemicals', 'Mid Cap'),
            ('DABUR.NS', 'Dabur India Ltd', 'FMCG', 'Mid Cap'),
            ('BIOCON.NS', 'Biocon Ltd', 'Pharma', 'Mid Cap'),
            ('MCDOWELL-N.NS', 'United Spirits Ltd', 'Beverages', 'Mid Cap'),
            
            # Small Cap Stocks  
            ('BATINDIA.NS', 'ITC Ltd', 'FMCG', 'Small Cap'),
            ('MARICO.NS', 'Marico Ltd', 'FMCG', 'Small Cap'),
            ('COLPAL.NS', 'Colgate Palmolive India Ltd', 'FMCG', 'Small Cap')
        ]
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for symbol, company_name, sector, market_cap in indian_stocks:
                cursor.execute("""
                    INSERT OR IGNORE INTO stock_metadata 
                    (symbol, company_name, sector, market_cap_category, exchange, is_active)
                    VALUES (?, ?, ?, ?, 'NSE', 1)
                """, (symbol, company_name, sector, market_cap))
            
            conn.commit()
            print(f"Added {len(indian_stocks)} Indian stock symbols to metadata")
            
        except Exception as e:
            print(f"Error adding stock metadata: {e}")
    
    def add_indian_indices_metadata(self):
        """
        Add Indian market indices metadata
        """
        indian_indices = [
            ('^NSEI', 'NIFTY 50'),
            ('^BSESN', 'BSE SENSEX'),
            ('^NSEBANK', 'NIFTY BANK'),
            ('^NSEIT', 'NIFTY IT'),
            ('^NSEFMCG', 'NIFTY FMCG'),
            ('^NSEPHARMA', 'NIFTY PHARMA'),
            ('^NSEAUTO', 'NIFTY AUTO'),
            ('^NSEREALTY', 'NIFTY REALTY'),
            ('^NSEMETAL', 'NIFTY METAL'),
            ('^NSEENERGY', 'NIFTY ENERGY')
        ]
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for index_symbol, index_name in indian_indices:
                cursor.execute("""
                    INSERT OR IGNORE INTO market_indices 
                    (index_symbol, index_name, date, open, high, low, close, volume, change_percent)
                    VALUES (?, ?, '2022-01-01', 0, 0, 0, 0, 0, 0)
                    ON CONFLICT DO NOTHING
                """, (index_symbol, index_name))
            
            # Remove the placeholder entry and create proper schema
            cursor.execute("DELETE FROM market_indices WHERE date = '2022-01-01' AND close = 0")
            conn.commit()
            
            print(f"Added {len(indian_indices)} Indian market indices")
            
        except Exception as e:
            print(f"Error adding indices metadata: {e}")
    
    def add_indian_economic_indicators(self):
        """
        Initialize Indian economic indicators tracking
        """
        economic_indicators = [
            ('repo_rate', 'RBI Repo Rate', '%'),
            ('inflation_rate', 'Consumer Price Index Inflation', '%'),
            ('gdp_growth', 'GDP Growth Rate', '%'),
            ('fiscal_deficit', 'Fiscal Deficit', '% of GDP'),
            ('current_account_deficit', 'Current Account Deficit', '% of GDP'),
            ('foreign_reserves', 'Foreign Exchange Reserves', 'USD Billion'),
            ('industrial_production', 'Index of Industrial Production', 'Index'),
            ('manufacturing_pmi', 'Manufacturing PMI', 'Index'),
            ('services_pmi', 'Services PMI', 'Index'),
            ('crude_oil_price', 'Crude Oil Price (Brent)', 'USD/Barrel')
        ]
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Add economic indicators as metadata (will be populated by data fetching modules)
            for indicator_name, description, unit in economic_indicators:
                cursor.execute("""
                    INSERT OR IGNORE INTO economic_indicators 
                    (indicator_name, date, value, unit, source)
                    VALUES (?, '2022-01-01', 0, ?, 'RBI/MOSPI/Trading Economics')
                """, (indicator_name, unit))
            
            # Remove placeholder entries
            cursor.execute("DELETE FROM economic_indicators WHERE date = '2022-01-01' AND value = 0")
            conn.commit()
            
            print(f"Added {len(economic_indicators)} economic indicators for tracking")
            
        except Exception as e:
            print(f"Error adding economic indicators: {e}")
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Get database statistics for monitoring
        
        Returns:
            Dictionary with table record counts
        """
        stats = {}
        tables = ['stock_prices', 'market_indices', 'technical_indicators', 
                 'economic_indicators', 'stock_metadata', 'sector_performance']
        
        try:
            conn = self.get_connection()
            
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Clean up old log entries to maintain database performance
        
        Args:
            days_to_keep: Number of days of logs to retain
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Clean up system logs
            cursor.execute("""
                DELETE FROM system_logs 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            # Clean up data quality logs
            cursor.execute("""
                DELETE FROM data_quality_logs 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            conn.commit()
            print(f"Cleaned up logs older than {days_to_keep} days")
            
        except Exception as e:
            print(f"Error cleaning up logs: {e}")
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Database connection closed")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close_connection()


# Version 1 specific initialization function
def initialize_v1_database(base_path: str = "data/") -> DatabaseConfig:
    """
    Initialize Version 1 database with Indian market focus
    
    Args:
        base_path: Base path for data storage
        
    Returns:
        Configured DatabaseConfig instance
    """
    print("=== Market Research System v1.0 Database Initialization ===")
    print("Focus: Indian Stock Market Analysis (2022)")
    print("Technology Stack: SQLite + CSV storage")
    
    # Create database configuration
    db_config = DatabaseConfig(base_path)
    
    # Add Indian market specific data
    print("\nAdding Indian market metadata...")
    db_config.add_indian_stocks_metadata()
    db_config.add_indian_indices_metadata() 
    db_config.add_indian_economic_indicators()
    
    # Display database statistics
    print("\nDatabase Statistics:")
    stats = db_config.get_database_stats()
    for table, count in stats.items():
        print(f"  {table}: {count} records")
    
    print(f"\nDatabase initialized successfully at: {db_config.db_path}")
    print("Ready for Indian market data collection and analysis!")
    
    return db_config


# Example usage for Version 1
if __name__ == "__main__":
    # Initialize the database for Version 1
    db = initialize_v1_database()
    
    # Example: Check if system is ready
    symbols = db.get_indian_stock_symbols()
    print(f"\nTracking {len(symbols)} Indian stocks:")
    print(f"Sample symbols: {symbols[:10]}")
    
    # Create a backup
    db.backup_database()
    
    # Clean up and close
    db.close_connection()