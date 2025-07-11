#!/usr/bin/env python3
"""
Market Research System v1.0 - Data Migration Tool (2022)
Handles data migration between different database versions and formats
Focus: Indian Stock Market Data Migration
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import argparse
from pathlib import Path

# Database connection modules
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

class DatabaseMigrator:
    """Handles data migration between different database systems and versions"""
    
    def __init__(self, source_config: Dict, target_config: Dict, log_level: str = 'INFO'):
        self.source_config = source_config
        self.target_config = target_config
        self.setup_logging(log_level)
        
        # Supported database types
        self.db_handlers = {
            'sqlite': self._handle_sqlite,
            'mysql': self._handle_mysql,
            'postgresql': self._handle_postgresql
        }
        
        # Migration batch size
        self.batch_size = 10000
        
    def setup_logging(self, level: str):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _handle_sqlite(self, config: Dict) -> sqlite3.Connection:
        """Create SQLite connection"""
        db_path = config.get('database', 'market_research.db')
        return sqlite3.connect(db_path)
        
    def _handle_mysql(self, config: Dict) -> 'mysql.connector.connection.MySQLConnection':
        """Create MySQL connection"""
        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL connector not available. Install mysql-connector-python")
        
        return mysql.connector.connect(
            host=config.get('host', 'localhost'),
            port=config.get('port', 3306),
            user=config.get('user'),
            password=config.get('password'),
            database=config.get('database')
        )
        
    def _handle_postgresql(self, config: Dict) -> 'psycopg2.extensions.connection':
        """Create PostgreSQL connection"""
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("PostgreSQL connector not available. Install psycopg2")
        
        return psycopg2.connect(
            host=config.get('host', 'localhost'),
            port=config.get('port', 5432),
            user=config.get('user'),
            password=config.get('password'),
            database=config.get('database')
        )
        
    def get_connection(self, config: Dict):
        """Get database connection based on type"""
        db_type = config.get('type', 'sqlite').lower()
        
        if db_type not in self.db_handlers:
            raise ValueError(f"Unsupported database type: {db_type}")
            
        return self.db_handlers[db_type](config)
        
    def migrate_table_data(self, table_name: str, source_conn, target_conn, 
                          transform_func: Optional[callable] = None) -> int:
        """Migrate data from source table to target table"""
        try:
            self.logger.info(f"Starting migration of table: {table_name}")
            
            # Get source data in batches
            source_cursor = source_conn.cursor()
            
            # Count total records
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            source_cursor.execute(count_query)
            total_records = source_cursor.fetchone()[0]
            
            if total_records == 0:
                self.logger.info(f"No data found in table: {table_name}")
                return 0
                
            self.logger.info(f"Total records to migrate in {table_name}: {total_records}")
            
            # Get column information
            if self.source_config.get('type') == 'sqlite':
                source_cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in source_cursor.fetchall()]
            else:
                source_cursor.execute(f"DESCRIBE {table_name}")
                columns = [col[0] for col in source_cursor.fetchall()]
            
            # Migrate data in batches
            migrated_count = 0
            offset = 0
            
            target_cursor = target_conn.cursor()
            
            while offset < total_records:
                # Fetch batch from source
                select_query = f"SELECT * FROM {table_name} LIMIT {self.batch_size} OFFSET {offset}"
                source_cursor.execute(select_query)
                batch_data = source_cursor.fetchall()
                
                if not batch_data:
                    break
                
                # Transform data if function provided
                if transform_func:
                    batch_data = transform_func(batch_data, columns)
                
                # Insert into target
                placeholders = ','.join(['?' if self.target_config.get('type') == 'sqlite' else '%s'] * len(columns))
                insert_query = f"INSERT OR IGNORE INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
                
                if self.target_config.get('type') == 'mysql':
                    insert_query = insert_query.replace('INSERT OR IGNORE', 'INSERT IGNORE')
                elif self.target_config.get('type') == 'postgresql':
                    insert_query = insert_query.replace('INSERT OR IGNORE', 'INSERT').replace('?', '%s') + ' ON CONFLICT DO NOTHING'
                
                target_cursor.executemany(insert_query, batch_data)
                target_conn.commit()
                
                migrated_count += len(batch_data)
                offset += self.batch_size
                
                # Progress reporting
                progress = (offset / total_records) * 100
                self.logger.info(f"Migration progress for {table_name}: {progress:.1f}% ({migrated_count}/{total_records})")
            
            self.logger.info(f"Successfully migrated {migrated_count} records from table: {table_name}")
            return migrated_count
            
        except Exception as e:
            self.logger.error(f"Error migrating table {table_name}: {str(e)}")
            raise
            
    def migrate_csv_to_db(self, csv_file_path: str, table_name: str, target_conn,
                         column_mapping: Optional[Dict] = None) -> int:
        """Migrate data from CSV file to database table"""
        try:
            self.logger.info(f"Starting CSV migration: {csv_file_path} -> {table_name}")
            
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Apply column mapping if provided
            if column_mapping:
                df = df.rename(columns=column_mapping)
            
            # Clean data
            df = self._clean_dataframe(df)
            
            # Get database type for appropriate insertion method
            db_type = self.target_config.get('type', 'sqlite')
            
            if db_type == 'sqlite':
                df.to_sql(table_name, target_conn, if_exists='append', index=False)
            else:
                # For MySQL/PostgreSQL, use cursor method
                self._insert_dataframe_to_db(df, table_name, target_conn, db_type)
            
            record_count = len(df)
            self.logger.info(f"Successfully migrated {record_count} records from CSV to {table_name}")
            return record_count
            
        except Exception as e:
            self.logger.error(f"Error migrating CSV {csv_file_path} to {table_name}: {str(e)}")
            raise
            
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe for database insertion"""
        # Handle NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Convert datetime columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
        
        # Handle None/NaN values for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Handle None/NaN values for string columns
        string_columns = df.select_dtypes(include=['object']).columns
        df[string_columns] = df[string_columns].fillna('')
        
        return df
        
    def _insert_dataframe_to_db(self, df: pd.DataFrame, table_name: str, 
                               conn, db_type: str):
        """Insert dataframe into database using cursor method"""
        cursor = conn.cursor()
        
        # Prepare column names and placeholders
        columns = list(df.columns)
        placeholder = '%s' if db_type in ['mysql', 'postgresql'] else '?'
        placeholders = ','.join([placeholder] * len(columns))
        
        # Prepare insert query
        insert_query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
        
        if db_type == 'mysql':
            insert_query = insert_query.replace('INSERT INTO', 'INSERT IGNORE INTO')
        elif db_type == 'postgresql':
            insert_query += ' ON CONFLICT DO NOTHING'
        else:  # SQLite
            insert_query = insert_query.replace('INSERT INTO', 'INSERT OR IGNORE INTO')
        
        # Convert dataframe to list of tuples
        data_tuples = [tuple(row) for row in df.to_numpy()]
        
        # Insert in batches
        batch_size = 1000
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i + batch_size]
            cursor.executemany(insert_query, batch)
            conn.commit()
            
            progress = min(i + batch_size, len(data_tuples))
            self.logger.info(f"Inserted {progress}/{len(data_tuples)} records into {table_name}")
            
    def export_table_to_csv(self, table_name: str, source_conn, 
                           output_path: str, where_clause: str = None) -> str:
        """Export table data to CSV file"""
        try:
            self.logger.info(f"Exporting table {table_name} to CSV: {output_path}")
            
            # Build query
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
                
            # Execute query and create dataframe
            df = pd.read_sql_query(query, source_conn)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to CSV
            df.to_csv(output_path, index=False)
            
            record_count = len(df)
            self.logger.info(f"Successfully exported {record_count} records to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting table {table_name} to CSV: {str(e)}")
            raise
            
    def migrate_historical_data_from_yfinance(self, symbols: List[str], 
                                            target_conn, start_date: str = '2020-01-01'):
        """Migrate historical data from yfinance to database"""
        try:
            import yfinance as yf
            
            self.logger.info(f"Starting yfinance data migration for {len(symbols)} symbols")
            
            cursor = target_conn.cursor()
            migrated_symbols = 0
            
            for symbol in symbols:
                try:
                    self.logger.info(f"Fetching data for {symbol}")
                    
                    # Fetch historical data
                    ticker = yf.Ticker(symbol)
                    hist_data = ticker.history(start=start_date, end=datetime.now().strftime('%Y-%m-%d'))
                    
                    if hist_data.empty:
                        self.logger.warning(f"No data found for {symbol}")
                        continue
                    
                    # Prepare data for insertion
                    hist_data.reset_index(inplace=True)
                    hist_data['symbol'] = symbol
                    
                    # Rename columns to match database schema
                    column_mapping = {
                        'Date': 'date',
                        'Open': 'open_price',
                        'High': 'high_price',
                        'Low': 'low_price',
                        'Close': 'close_price',
                        'Volume': 'volume'
                    }
                    hist_data = hist_data.rename(columns=column_mapping)
                    
                    # Select required columns
                    required_columns = ['symbol', 'date', 'open_price', 'high_price', 
                                      'low_price', 'close_price', 'volume']
                    hist_data = hist_data[required_columns]
                    
                    # Insert data
                    db_type = self.target_config.get('type', 'sqlite')
                    if db_type == 'sqlite':
                        hist_data.to_sql('stock_prices', target_conn, if_exists='append', index=False)
                    else:
                        self._insert_dataframe_to_db(hist_data, 'stock_prices', target_conn, db_type)
                    
                    migrated_symbols += 1
                    self.logger.info(f"Successfully migrated data for {symbol} ({len(hist_data)} records)")
                    
                except Exception as e:
                    self.logger.error(f"Error migrating data for {symbol}: {str(e)}")
                    continue
            
            self.logger.info(f"Migration completed. Successfully migrated {migrated_symbols}/{len(symbols)} symbols")
            return migrated_symbols
            
        except ImportError:
            self.logger.error("yfinance not available. Install with: pip install yfinance")
            raise
        except Exception as e:
            self.logger.error(f"Error in yfinance migration: {str(e)}")
            raise

    def migrate_nse_indices_data(self, target_conn, start_date: str = '2020-01-01'):
        """Migrate NSE indices data for Indian market"""
        try:
            # Indian market indices commonly tracked
            indian_indices = [
                '^NSEI',    # NIFTY 50
                '^NSEBANK', # NIFTY BANK
                '^NSEIT',   # NIFTY IT
                '^NSEFMCG', # NIFTY FMCG
                '^NSEPHARMA', # NIFTY PHARMA
                '^NSEAUTO',   # NIFTY AUTO
                '^NSEMETAL',  # NIFTY METAL
                '^NSEREALTY', # NIFTY REALTY
            ]
            
            self.logger.info("Starting NSE indices data migration")
            return self.migrate_historical_data_from_yfinance(indian_indices, target_conn, start_date)
            
        except Exception as e:
            self.logger.error(f"Error migrating NSE indices: {str(e)}")
            raise

    def migrate_indian_stocks_data(self, target_conn, start_date: str = '2020-01-01'):
        """Migrate top Indian stocks data"""
        try:
            # Top Indian stocks (NSE symbols with .NS suffix for yfinance)
            top_indian_stocks = [
                'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
                'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS',
                'SBIN.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'LT.NS',
                'DMART.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
                'POWERGRID.NS', 'NTPC.NS', 'TECHM.NS', 'HCLTECH.NS', 'WIPRO.NS',
                'ONGC.NS', 'TATAMOTORS.NS', 'COALINDIA.NS', 'GRASIM.NS', 'JSWSTEEL.NS',
                'TATASTEEL.NS', 'ADANIPORTS.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
                'HDFCLIFE.NS', 'SBILIFE.NS', 'BRITANNIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
                'BPCL.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'HEROMOTOCO.NS', 'SHREECEM.NS',
                'HINDALCO.NS', 'BAJAJ-AUTO.NS', 'UPL.NS', 'INDUSINDBK.NS', 'GODREJCP.NS'
            ]
            
            self.logger.info(f"Starting migration for {len(top_indian_stocks)} Indian stocks")
            return self.migrate_historical_data_from_yfinance(top_indian_stocks, target_conn, start_date)
            
        except Exception as e:
            self.logger.error(f"Error migrating Indian stocks: {str(e)}")
            raise

    def create_database_schema(self, target_conn):
        """Create database schema for Version 1 (2022) - Indian Market Focus"""
        try:
            cursor = target_conn.cursor()
            db_type = self.target_config.get('type', 'sqlite')
            
            # Define schema based on database type
            if db_type == 'sqlite':
                schema_queries = [
                    """
                    CREATE TABLE IF NOT EXISTS stock_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS economic_indicators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        indicator_name TEXT NOT NULL,
                        date DATE NOT NULL,
                        value REAL,
                        source TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(indicator_name, date)
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS migration_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        migration_type TEXT NOT NULL,
                        table_name TEXT,
                        records_migrated INTEGER,
                        status TEXT,
                        error_message TEXT,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                    """
                    CREATE INDEX IF NOT EXISTS idx_stock_symbol_date ON stock_prices(symbol, date);
                    """,
                    """
                    CREATE INDEX IF NOT EXISTS idx_economic_indicator_date ON economic_indicators(indicator_name, date);
                    """
                ]
            elif db_type == 'mysql':
                schema_queries = [
                    """
                    CREATE TABLE IF NOT EXISTS stock_prices (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        date DATE NOT NULL,
                        open_price DECIMAL(10,2),
                        high_price DECIMAL(10,2),
                        low_price DECIMAL(10,2),
                        close_price DECIMAL(10,2),
                        volume BIGINT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY unique_symbol_date (symbol, date)
                    ) ENGINE=InnoDB
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS economic_indicators (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        indicator_name VARCHAR(100) NOT NULL,
                        date DATE NOT NULL,
                        value DECIMAL(15,4),
                        source VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY unique_indicator_date (indicator_name, date)
                    ) ENGINE=InnoDB
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS migration_log (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        migration_type VARCHAR(50) NOT NULL,
                        table_name VARCHAR(50),
                        records_migrated INT,
                        status VARCHAR(20),
                        error_message TEXT,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB
                    """
                ]
                
            # Execute schema creation
            for query in schema_queries:
                cursor.execute(query)
                
            target_conn.commit()
            self.logger.info("Database schema created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database schema: {str(e)}")
            raise

    def log_migration(self, target_conn, migration_type: str, table_name: str = None, 
                     records_migrated: int = 0, status: str = 'SUCCESS', 
                     error_message: str = None, started_at: datetime = None):
        """Log migration activity"""
        try:
            cursor = target_conn.cursor()
            
            insert_query = """
                INSERT INTO migration_log 
                (migration_type, table_name, records_migrated, status, error_message, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            
            if self.target_config.get('type') in ['mysql', 'postgresql']:
                insert_query = insert_query.replace('?', '%s')
            
            cursor.execute(insert_query, (
                migration_type, table_name, records_migrated, 
                status, error_message, started_at or datetime.now()
            ))
            target_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error logging migration: {str(e)}")

    def backup_database(self, source_conn, backup_path: str):
        """Create backup of existing database"""
        try:
            self.logger.info(f"Creating database backup at: {backup_path}")
            
            db_type = self.source_config.get('type', 'sqlite')
            
            if db_type == 'sqlite':
                # For SQLite, copy the file
                import shutil
                source_path = self.source_config.get('database')
                shutil.copy2(source_path, backup_path)
                
            else:
                # For other databases, export to SQL dump (simplified approach)
                cursor = source_conn.cursor()
                cursor.execute("SHOW TABLES" if db_type == 'mysql' else 
                             "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'" 
                             if db_type == 'postgresql' else 
                             "SELECT name FROM sqlite_master WHERE type='table'")
                
                tables = cursor.fetchall()
                
                with open(backup_path, 'w') as f:
                    for table in tables:
                        table_name = table[0]
                        f.write(f"-- Backup for table: {table_name}\n")
                        
                        cursor.execute(f"SELECT * FROM {table_name}")
                        rows = cursor.fetchall()
                        
                        for row in rows:
                            f.write(f"INSERT INTO {table_name} VALUES {row};\n")
            
            self.logger.info("Database backup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database backup: {str(e)}")
            raise

    def validate_migration(self, source_conn, target_conn, table_name: str) -> Dict[str, Any]:
        """Validate migration by comparing record counts and sample data"""
        try:
            validation_result = {
                'table_name': table_name,
                'source_count': 0,
                'target_count': 0,
                'sample_match': False,
                'status': 'FAILED'
            }
            
            # Get record counts
            source_cursor = source_conn.cursor()
            target_cursor = target_conn.cursor()
            
            source_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            validation_result['source_count'] = source_cursor.fetchone()[0]
            
            target_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            validation_result['target_count'] = target_cursor.fetchone()[0]
            
            # Check if counts match
            if validation_result['source_count'] == validation_result['target_count']:
                # Sample data validation (check first 10 records)
                source_cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
                source_sample = source_cursor.fetchall()
                
                target_cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
                target_sample = target_cursor.fetchall()
                
                validation_result['sample_match'] = source_sample == target_sample
                validation_result['status'] = 'SUCCESS' if validation_result['sample_match'] else 'PARTIAL'
            
            self.logger.info(f"Validation for {table_name}: {validation_result}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating migration for {table_name}: {str(e)}")
            validation_result['status'] = 'ERROR'
            validation_result['error'] = str(e)
            return validation_result


def main():
    """Main function to handle command line arguments and execute migration"""
    parser = argparse.ArgumentParser(description='Market Research System Data Migration Tool v1.0')
    parser.add_argument('--source-config', required=True, help='Source database configuration file')
    parser.add_argument('--target-config', required=True, help='Target database configuration file')
    parser.add_argument('--migration-type', required=True, 
                       choices=['table', 'csv', 'yfinance', 'indian-stocks', 'nse-indices', 'full'],
                       help='Type of migration to perform')
    parser.add_argument('--table-name', help='Table name for table migration')
    parser.add_argument('--csv-file', help='CSV file path for CSV migration')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date for data migration (YYYY-MM-DD)')
    parser.add_argument('--backup', action='store_true', help='Create backup before migration')
    parser.add_argument('--validate', action='store_true', help='Validate migration after completion')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    try:
        # Load configuration files
        with open(args.source_config, 'r') as f:
            source_config = json.load(f)
        
        with open(args.target_config, 'r') as f:
            target_config = json.load(f)
        
        # Initialize migrator
        migrator = DatabaseMigrator(source_config, target_config, args.log_level)
        
        # Get connections
        source_conn = migrator.get_connection(source_config)
        target_conn = migrator.get_connection(target_config)
        
        # Create schema if target is new
        migrator.create_database_schema(target_conn)
        
        # Create backup if requested
        if args.backup:
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            migrator.backup_database(source_conn, backup_path)
        
        # Perform migration based on type
        migration_start = datetime.now()
        migrated_records = 0
        
        try:
            if args.migration_type == 'table' and args.table_name:
                migrated_records = migrator.migrate_table_data(args.table_name, source_conn, target_conn)
                
            elif args.migration_type == 'csv' and args.csv_file:
                table_name = args.table_name or Path(args.csv_file).stem
                migrated_records = migrator.migrate_csv_to_db(args.csv_file, table_name, target_conn)
                
            elif args.migration_type == 'indian-stocks':
                migrated_records = migrator.migrate_indian_stocks_data(target_conn, args.start_date)
                
            elif args.migration_type == 'nse-indices':
                migrated_records = migrator.migrate_nse_indices_data(target_conn, args.start_date)
                
            elif args.migration_type == 'full':
                # Full migration including Indian stocks and indices
                stock_records = migrator.migrate_indian_stocks_data(target_conn, args.start_date)
                index_records = migrator.migrate_nse_indices_data(target_conn, args.start_date)
                migrated_records = stock_records + index_records
            
            # Log successful migration
            migrator.log_migration(
                target_conn, args.migration_type, args.table_name or 'multiple',
                migrated_records, 'SUCCESS', None, migration_start
            )
            
        except Exception as e:
            # Log failed migration
            migrator.log_migration(
                target_conn, args.migration_type, args.table_name or 'multiple',
                0, 'FAILED', str(e), migration_start
            )
            raise
        
        # Validate migration if requested
        if args.validate and args.table_name:
            validation_result = migrator.validate_migration(source_conn, target_conn, args.table_name)
            print(f"Validation Result: {validation_result}")
        
        print(f"Migration completed successfully! Migrated {migrated_records} records.")
        
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        sys.exit(1)
    finally:
        # Close connections
        try:
            source_conn.close()
            target_conn.close()
        except:
            pass


if __name__ == "__main__":
    main()