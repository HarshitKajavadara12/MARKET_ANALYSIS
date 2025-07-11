"""
Data Storage Utilities for Market Research System v1.0
Handles data storage, caching, and backup operations for Indian stock market data.
Created: January 2022
"""

import os
import json
import pandas as pd
import sqlite3
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Any
import csv
from .exceptions import DataStorageError, DataValidationError

logger = logging.getLogger(__name__)

class DataStorage:
    """
    Comprehensive data storage utility for market research system.
    Handles file operations, database operations, and data caching.
    """
    
    def __init__(self, base_path: str = "data", cache_enabled: bool = True):
        """
        Initialize data storage with base path and caching options.
        
        Args:
            base_path: Base directory for data storage
            cache_enabled: Enable/disable data caching
        """
        self.base_path = Path(base_path)
        self.cache_enabled = cache_enabled
        self.cache_dir = self.base_path / "cache"
        self.raw_dir = self.base_path / "raw"
        self.processed_dir = self.base_path / "processed"
        self.backups_dir = self.base_path / "backups"
        
        # Create directory structures
        self._create_directories()
        
        # Initialize SQLite database for metadata
        self.db_path = self.base_path / "market_data.db"
        self._init_database()
        
    def _create_directories(self):
        """Create necessary directory structure."""
        directories = [
            self.base_path,
            self.cache_dir,
            self.raw_dir,
            self.processed_dir,
            self.backups_dir,
            self.raw_dir / "stocks",
            self.raw_dir / "stocks" / "daily",
            self.raw_dir / "stocks" / "intraday",
            self.raw_dir / "stocks" / "historical",
            self.raw_dir / "economic",
            self.raw_dir / "indices",
            self.processed_dir / "stocks",
            self.processed_dir / "economic",
            self.processed_dir / "technical_indicators",
            self.cache_dir / "daily_cache",
            self.cache_dir / "weekly_cache",
            self.backups_dir / "daily",
            self.backups_dir / "weekly"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # Create .gitkeep files to maintain directory structure
            gitkeep_path = directory / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
    
    def _init_database(self):
        """Initialize SQLite database for metadata storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables for metadata
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS data_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        created_date TEXT NOT NULL,
                        last_updated TEXT NOT NULL,
                        records_count INTEGER,
                        data_quality_score REAL,
                        UNIQUE(symbol, data_type, file_path)
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cache_key TEXT UNIQUE NOT NULL,
                        file_path TEXT NOT NULL,
                        created_date TEXT NOT NULL,
                        expiry_date TEXT NOT NULL,
                        size_bytes INTEGER
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS backup_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        backup_type TEXT NOT NULL,
                        backup_path TEXT NOT NULL,
                        created_date TEXT NOT NULL,
                        files_count INTEGER,
                        total_size_bytes INTEGER
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DataStorageError(f"Failed to initialize database: {e}")
    
    def save_stock_data(self, symbol: str, data: pd.DataFrame, data_type: str = "daily") -> str:
        """
        Save stock data to appropriate directory structure.
        
        Args:
            symbol: Stock symbol (NSE format)
            data: Stock data DataFrame
            data_type: Type of data (daily, intraday, historical)
            
        Returns:
            str: Path to saved file
        """
        try:
            # Validate data
            if data.empty:
                raise DataValidationError(f"Empty data provided for {symbol}")
            
            # Create file path
            today = datetime.now().strftime("%Y%m%d")
            filename = f"{symbol}_{data_type}_{today}.csv"
            file_path = self.raw_dir / "stocks" / data_type / filename
            
            # Save data
            data.to_csv(file_path, index=True)
            
            # Update metadata
            self._update_metadata(symbol, data_type, str(file_path), len(data))
            
            logger.info(f"Saved {data_type} data for {symbol}: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save stock data for {symbol}: {e}")
            raise DataStorageError(f"Failed to save stock data: {e}")
    
    def load_stock_data(self, symbol: str, data_type: str = "daily", 
                       date_range: Optional[tuple] = None) -> pd.DataFrame:
        """
        Load stock data from storage.
        
        Args:
            symbol: Stock symbol
            data_type: Type of data to load
            date_range: Optional tuple of (start_date, end_date)
            
        Returns:
            pd.DataFrame: Loaded stock data
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{data_type}"
            if self.cache_enabled:
                cached_data = self._load_from_cache(cache_key)
                if cached_data is not None:
                    return cached_data
            
            # Find latest file for symbol
            data_dir = self.raw_dir / "stocks" / data_type
            pattern = f"{symbol}_{data_type}_*.csv"
            files = list(data_dir.glob(pattern))
            
            if not files:
                raise DataStorageError(f"No data found for {symbol} ({data_type})")
            
            # Load most recent file
            latest_file = sorted(files)[-1]
            data = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            
            # Filter by date range if provided
            if date_range:
                start_date, end_date = date_range
                data = data[start_date:end_date]
            
            # Cache the data
            if self.cache_enabled:
                self._save_to_cache(cache_key, data)
            
            logger.info(f"Loaded {data_type} data for {symbol}: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load stock data for {symbol}: {e}")
            raise DataStorageError(f"Failed to load stock data: {e}")
    
    def save_economic_data(self, indicator: str, data: pd.DataFrame) -> str:
        """
        Save economic indicator data.
        
        Args:
            indicator: Economic indicator name
            data: Economic data DataFrame
            
        Returns:
            str: Path to saved file
        """
        try:
            today = datetime.now().strftime("%Y%m%d")
            filename = f"{indicator}_{today}.csv"
            file_path = self.raw_dir / "economic" / filename
            
            data.to_csv(file_path, index=True)
            self._update_metadata(indicator, "economic", str(file_path), len(data))
            
            logger.info(f"Saved economic data for {indicator}: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save economic data for {indicator}: {e}")
            raise DataStorageError(f"Failed to save economic data: {e}")
    
    def save_processed_data(self, symbol: str, data: pd.DataFrame, 
                          processing_type: str) -> str:
        """
        Save processed data (cleaned, normalized, or with technical indicators).
        
        Args:
            symbol: Stock symbol
            data: Processed data DataFrame
            processing_type: Type of processing (cleaned, normalized, indicators)
            
        Returns:
            str: Path to saved file
        """
        try:
            today = datetime.now().strftime("%Y%m%d")
            filename = f"{symbol}_{processing_type}_{today}.csv"
            
            # Determine subdirectory based on processing type
            if processing_type in ['cleaned', 'normalized', 'aggregated']:
                subdir = "stocks"
            elif processing_type in ['moving_averages', 'oscillators', 'momentum']:
                subdir = "technical_indicators"
            else:
                subdir = "stocks"
            
            file_path = self.processed_dir / subdir / filename
            data.to_csv(file_path, index=True)
            
            self._update_metadata(symbol, f"processed_{processing_type}", 
                                str(file_path), len(data))
            
            logger.info(f"Saved processed data for {symbol} ({processing_type}): {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save processed data for {symbol}: {e}")
            raise DataStorageError(f"Failed to save processed data: {e}")
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame, 
                      expiry_hours: int = 24):
        """Save data to cache with expiry."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update cache metadata
            expiry_date = datetime.now() + timedelta(hours=expiry_hours)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_metadata 
                    (cache_key, file_path, created_date, expiry_date, size_bytes)
                    VALUES (?, ?, ?, ?, ?)
                ''', (cache_key, str(cache_file), datetime.now().isoformat(),
                     expiry_date.isoformat(), cache_file.stat().st_size))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if not expired."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT file_path, expiry_date FROM cache_metadata 
                    WHERE cache_key = ?
                ''', (cache_key,))
                result = cursor.fetchone()
                
                if not result:
                    return None
                
                file_path, expiry_date = result
                expiry = datetime.fromisoformat(expiry_date)
                
                if datetime.now() > expiry:
                    # Cache expired, remove it
                    self._remove_from_cache(cache_key)
                    return None
                
                # Load cached data
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                logger.debug(f"Loaded data from cache: {cache_key}")
                return data
                
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _remove_from_cache(self, cache_key: str):
        """Remove expired cache entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT file_path FROM cache_metadata WHERE cache_key = ?
                ''', (cache_key,))
                result = cursor.fetchone()
                
                if result:
                    file_path = Path(result[0])
                    if file_path.exists():
                        file_path.unlink()
                    
                    cursor.execute('''
                        DELETE FROM cache_metadata WHERE cache_key = ?
                    ''', (cache_key,))
                    conn.commit()
                    
        except Exception as e:
            logger.warning(f"Failed to remove cache entry: {e}")
    
    def _update_metadata(self, symbol: str, data_type: str, file_path: str, 
                        records_count: int):
        """Update metadata for stored data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO data_metadata 
                    (symbol, data_type, file_path, created_date, last_updated, 
                     records_count, data_quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, data_type, file_path, datetime.now().isoformat(),
                     datetime.now().isoformat(), records_count, 1.0))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to update metadata: {e}")
    
    def create_backup(self, backup_type: str = "daily") -> str:
        """
        Create backup of all data files.
        
        Args:
            backup_type: Type of backup (daily, weekly)
            
        Returns:
            str: Path to backup directory
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backups_dir / backup_type / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy raw data
            import shutil
            raw_backup = backup_dir / "raw"
            shutil.copytree(self.raw_dir, raw_backup, 
                          ignore=shutil.ignore_patterns('.gitkeep'))
            
            # Copy processed data
            processed_backup = backup_dir / "processed"
            shutil.copytree(self.processed_dir, processed_backup,
                          ignore=shutil.ignore_patterns('.gitkeep'))
            
            # Calculate backup size
            total_size = sum(f.stat().st_size for f in backup_dir.rglob('*') 
                           if f.is_file())
            files_count = len([f for f in backup_dir.rglob('*') if f.is_file()])
            
            # Update backup metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO backup_metadata 
                    (backup_type, backup_path, created_date, files_count, total_size_bytes)
                    VALUES (?, ?, ?, ?, ?)
                ''', (backup_type, str(backup_dir), datetime.now().isoformat(),
                     files_count, total_size))
                conn.commit()
            
            logger.info(f"Created {backup_type} backup: {backup_dir}")
            return str(backup_dir)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise DataStorageError(f"Failed to create backup: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Clean up data files older than specified days.
        
        Args:
            days_to_keep: Number of days to keep data
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            removed_files = 0
            
            # Clean raw data
            for data_dir in [self.raw_dir, self.processed_dir, self.cache_dir]:
                for file_path in data_dir.rglob('*.csv'):
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()
                        removed_files += 1
                        
                for file_path in data_dir.rglob('*.pkl'):
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()
                        removed_files += 1
            
            # Clean metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM data_metadata 
                    WHERE created_date < ?
                ''', (cutoff_date.isoformat(),))
                
                cursor.execute('''
                    DELETE FROM cache_metadata 
                    WHERE created_date < ?
                ''', (cutoff_date.isoformat(),))
                
                conn.commit()
            
            logger.info(f"Cleaned up {removed_files} old data files")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            raise DataStorageError(f"Failed to cleanup old data: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics and health information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get data statistics
                cursor.execute('''
                    SELECT data_type, COUNT(*), AVG(records_count)
                    FROM data_metadata
                    GROUP BY data_type
                ''')
                data_stats = cursor.fetchall()
                
                # Get cache statistics
                cursor.execute('''
                    SELECT COUNT(*), SUM(size_bytes)
                    FROM cache_metadata
                    WHERE expiry_date > ?
                ''', (datetime.now().isoformat(),))
                cache_stats = cursor.fetchone()
                
                # Get backup statistics
                cursor.execute('''
                    SELECT backup_type, COUNT(*), MAX(created_date)
                    FROM backup_metadata
                    GROUP BY backup_type
                ''')
                backup_stats = cursor.fetchall()
            
            # Calculate directory sizes
            raw_size = sum(f.stat().st_size for f in self.raw_dir.rglob('*') 
                          if f.is_file())
            processed_size = sum(f.stat().st_size for f in self.processed_dir.rglob('*') 
                               if f.is_file())
            cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') 
                           if f.is_file())
            
            return {
                'data_statistics': data_stats,
                'cache_statistics': cache_stats,
                'backup_statistics': backup_stats,
                'storage_sizes': {
                    'raw_data_mb': raw_size / (1024 * 1024),
                    'processed_data_mb': processed_size / (1024 * 1024),
                    'cache_mb': cache_size / (1024 * 1024),
                    'total_mb': (raw_size + processed_size + cache_size) / (1024 * 1024)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            raise DataStorageError(f"Failed to get storage stats: {e}")


class CSVManager:
    """Specialized CSV file management for market data."""
    
    @staticmethod
    def save_with_compression(data: pd.DataFrame, file_path: str, 
                            compression: str = 'gzip'):
        """Save DataFrame to compressed CSV."""
        try:
            if compression == 'gzip':
                data.to_csv(f"{file_path}.gz", compression='gzip', index=True)
            else:
                data.to_csv(file_path, index=True)
                
        except Exception as e:
            raise DataStorageError(f"Failed to save compressed CSV: {e}")
    
    @staticmethod
    def load_with_compression(file_path: str, compression: str = 'gzip') -> pd.DataFrame:
        """Load DataFrame from compressed CSV."""
        try:
            if compression == 'gzip' and file_path.endswith('.gz'):
                return pd.read_csv(file_path, compression='gzip', 
                                 index_col=0, parse_dates=True)
            else:
                return pd.read_csv(file_path, index_col=0, parse_dates=True)
                
        except Exception as e:
            raise DataStorageError(f"Failed to load compressed CSV: {e}")


class DatabaseManager:
    """Database operations for market data storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def create_stock_table(self, symbol: str):
        """Create table for specific stock data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                table_name = f"stock_{symbol.replace('.', '_').replace('-', '_')}"
                
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        date TEXT PRIMARY KEY,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        adj_close REAL
                    )
                ''')
                conn.commit()
                
        except Exception as e:
            raise DataStorageError(f"Failed to create stock table: {e}")
    
    def insert_stock_data(self, symbol: str, data: pd.DataFrame):
        """Insert stock data into database."""
        try:
            table_name = f"stock_{symbol.replace('.', '_').replace('-', '_')}"
            
            with sqlite3.connect(self.db_path) as conn:
                data.to_sql(table_name, conn, if_exists='replace', 
                          index=True, index_label='date')
                
        except Exception as e:
            raise DataStorageError(f"Failed to insert stock data: {e}")