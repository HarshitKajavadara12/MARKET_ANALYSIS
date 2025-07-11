#!/usr/bin/env python3
"""
Market Research System v1.0 - Data Cleanup Script
Clean up old data files to manage storage and maintain system performance

Author: Market Research Team
Date: 2022
Focus: Indian Stock Market (NSE/BSE)
"""

import os
import sys
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import yaml
import json

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logging_utils import setup_logging
from src.utils.date_utils import get_trading_days
from src.utils.file_utils import get_file_size, format_size

class DataCleanup:
    """Handle cleanup of old data files and maintain storage efficiency"""
    
    def __init__(self, config_path="config/system/cleanup_config.yaml"):
        """Initialize cleanup manager with configuration"""
        self.setup_logging()
        self.load_config(config_path)
        self.stats = {
            'files_deleted': 0,
            'space_freed': 0,
            'errors': 0,
            'directories_cleaned': []
        }
        
    def setup_logging(self):
        """Setup logging for cleanup operations"""
        self.logger = setup_logging(
            'data_cleanup', 
            'logs/maintenance/cleanup.log',
            level=logging.INFO
        )
        
    def load_config(self, config_path):
        """Load cleanup configuration"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                # Default configuration for Indian market
                self.config = {
                    'retention_days': {
                        'raw_data': 365,      # Keep raw data for 1 year
                        'processed_data': 180, # Keep processed data for 6 months
                        'cache': 30,          # Keep cache for 1 month
                        'logs': 90,           # Keep logs for 3 months
                        'reports': 365,       # Keep reports for 1 year
                        'backups': 30         # Keep backups for 1 month
                    },
                    'paths': {
                        'data_raw': 'data/raw',
                        'data_processed': 'data/processed',
                        'cache': 'data/cache',
                        'logs': 'logs',
                        'reports': 'reports',
                        'backups': 'data/backups'
                    },
                    'file_patterns': {
                        'temp_files': ['*.tmp', '*.temp', '*~'],
                        'old_logs': ['*.log.*', '*.log.old'],
                        'cache_files': ['*.cache', '*.pkl']
                    },
                    'size_threshold_mb': 1000  # Alert if folder > 1GB
                }
                self.save_default_config(config_path)
                
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            sys.exit(1)
            
    def save_default_config(self, config_path):
        """Save default configuration to file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Created default config at {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            
    def cleanup_old_files(self, directory, retention_days):
        """Clean up files older than retention period"""
        try:
            if not os.path.exists(directory):
                self.logger.warning(f"Directory {directory} does not exist")
                return
                
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            files_deleted = 0
            space_freed = 0
            
            self.logger.info(f"Cleaning {directory} - keeping files newer than {cutoff_date}")
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_date:
                            file_size = get_file_size(file_path)
                            os.remove(file_path)
                            files_deleted += 1
                            space_freed += file_size
                            self.logger.debug(f"Deleted: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error deleting {file_path}: {e}")
                        self.stats['errors'] += 1
            
            # Remove empty directories
            self.remove_empty_dirs(directory)
            
            self.stats['files_deleted'] += files_deleted
            self.stats['space_freed'] += space_freed
            self.stats['directories_cleaned'].append(directory)
            
            self.logger.info(f"Cleaned {directory}: {files_deleted} files, "
                           f"{format_size(space_freed)} freed")
                           
        except Exception as e:
            self.logger.error(f"Error cleaning {directory}: {e}")
            self.stats['errors'] += 1
            
    def remove_empty_dirs(self, directory):
        """Remove empty directories recursively"""
        try:
            for root, dirs, files in os.walk(directory, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            self.logger.debug(f"Removed empty directory: {dir_path}")
                    except OSError:
                        pass  # Directory not empty or permission denied
        except Exception as e:
            self.logger.error(f"Error removing empty directories: {e}")
            
    def cleanup_temp_files(self):
        """Clean up temporary files based on patterns"""
        try:
            patterns = self.config['file_patterns']['temp_files']
            temp_dirs = ['data', 'reports', 'logs']
            
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for root, dirs, files in os.walk(temp_dir):
                        for pattern in patterns:
                            import glob
                            temp_files = glob.glob(os.path.join(root, pattern))
                            for temp_file in temp_files:
                                try:
                                    file_size = get_file_size(temp_file)
                                    os.remove(temp_file)
                                    self.stats['files_deleted'] += 1
                                    self.stats['space_freed'] += file_size
                                    self.logger.debug(f"Deleted temp file: {temp_file}")
                                except Exception as e:
                                    self.logger.error(f"Error deleting {temp_file}: {e}")
                                    self.stats['errors'] += 1
                                    
        except Exception as e:
            self.logger.error(f"Error cleaning temp files: {e}")
            
    def cleanup_cache_files(self):
        """Clean up cache files"""
        try:
            cache_dir = self.config['paths']['cache']
            if os.path.exists(cache_dir):
                # Clean old cache files
                self.cleanup_old_files(cache_dir, self.config['retention_days']['cache'])
                
                # Clean cache pattern files
                patterns = self.config['file_patterns']['cache_files']
                for root, dirs, files in os.walk(cache_dir):
                    for pattern in patterns:
                        import glob
                        cache_files = glob.glob(os.path.join(root, pattern))
                        for cache_file in cache_files:
                            try:
                                # Check if cache file is older than 1 day
                                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                                if (datetime.now() - file_time).days > 1:
                                    file_size = get_file_size(cache_file)
                                    os.remove(cache_file)
                                    self.stats['files_deleted'] += 1
                                    self.stats['space_freed'] += file_size
                                    self.logger.debug(f"Deleted cache file: {cache_file}")
                            except Exception as e:
                                self.logger.error(f"Error deleting {cache_file}: {e}")
                                self.stats['errors'] += 1
                                
        except Exception as e:
            self.logger.error(f"Error cleaning cache files: {e}")
            
    def check_disk_usage(self):
        """Check disk usage and warn if directories are too large"""
        try:
            threshold_bytes = self.config['size_threshold_mb'] * 1024 * 1024
            
            for path_name, path in self.config['paths'].items():
                if os.path.exists(path):
                    total_size = 0
                    file_count = 0
                    
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                total_size += get_file_size(file_path)
                                file_count += 1
                            except Exception:
                                pass
                    
                    if total_size > threshold_bytes:
                        self.logger.warning(
                            f"Directory {path} is large: {format_size(total_size)} "
                            f"({file_count} files)"
                        )
                    else:
                        self.logger.info(
                            f"Directory {path}: {format_size(total_size)} "
                            f"({file_count} files)"
                        )
                        
        except Exception as e:
            self.logger.error(f"Error checking disk usage: {e}")
            
    def cleanup_indian_market_specific(self):
        """Cleanup specific to Indian market data patterns"""
        try:
            # Cleanup old NSE/BSE data files
            nse_patterns = [
                'data/raw/stocks/nse/old_*.csv',
                'data/raw/indices/nifty/temp_*.csv',
                'data/processed/stocks/nse/backup_*.pkl'
            ]
            
            for pattern in nse_patterns:
                import glob
                files = glob.glob(pattern)
                for file in files:
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file))
                        if (datetime.now() - file_time).days > 7:  # Keep for 1 week
                            file_size = get_file_size(file)
                            os.remove(file)
                            self.stats['files_deleted'] += 1
                            self.stats['space_freed'] += file_size
                            self.logger.debug(f"Deleted NSE/BSE temp file: {file}")
                    except Exception as e:
                        self.logger.error(f"Error deleting {file}: {e}")
                        
            # Cleanup old options data (very large files)
            options_dir = 'data/raw/options'
            if os.path.exists(options_dir):
                cutoff_date = datetime.now() - timedelta(days=30)  # Keep options data for 30 days
                for root, dirs, files in os.walk(options_dir):
                    for file in files:
                        if file.endswith('.csv') and 'options_' in file:
                            file_path = os.path.join(root, file)
                            try:
                                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                                if file_time < cutoff_date:
                                    file_size = get_file_size(file_path)
                                    os.remove(file_path)
                                    self.stats['files_deleted'] += 1
                                    self.stats['space_freed'] += file_size
                                    self.logger.debug(f"Deleted old options data: {file_path}")
                            except Exception as e:
                                self.logger.error(f"Error deleting {file_path}: {e}")
                                
        except Exception as e:
            self.logger.error(f"Error in Indian market specific cleanup: {e}")
            
    def run_cleanup(self, dry_run=False):
        """Run complete cleanup process"""
        start_time = datetime.now()
        self.logger.info(f"Starting data cleanup - Dry run: {dry_run}")
        
        if dry_run:
            self.logger.info("DRY RUN MODE - No files will be deleted")
            return
            
        try:
            # Check disk usage before cleanup
            self.logger.info("=== Disk Usage Before Cleanup ===")
            self.check_disk_usage()
            
            # Clean up by data type with different retention periods
            for data_type, retention_days in self.config['retention_days'].items():
                path = self.config['paths'].get(data_type)
                if path:
                    self.cleanup_old_files(path, retention_days)
                    
            # Clean temporary files
            self.cleanup_temp_files()
            
            # Clean cache files
            self.cleanup_cache_files()
            
            # Indian market specific cleanup
            self.cleanup_indian_market_specific()
            
            # Check disk usage after cleanup
            self.logger.info("=== Disk Usage After Cleanup ===")
            self.check_disk_usage()
            
            # Generate cleanup report
            self.generate_cleanup_report()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup process: {e}")
            self.stats['errors'] += 1
            
        duration = datetime.now() - start_time
        self.logger.info(f"Cleanup completed in {duration}")
        
    def generate_cleanup_report(self):
        """Generate cleanup summary report"""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'files_deleted': self.stats['files_deleted'],
                    'space_freed_mb': round(self.stats['space_freed'] / (1024*1024), 2),
                    'space_freed_formatted': format_size(self.stats['space_freed']),
                    'errors': self.stats['errors'],
                    'directories_cleaned': self.stats['directories_cleaned']
                },
                'config_used': self.config
            }
            
            # Save report
            report_dir = 'reports/maintenance'
            os.makedirs(report_dir, exist_ok=True)
            
            report_file = f"{report_dir}/cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            self.logger.info(f"Cleanup report saved: {report_file}")
            
            # Log summary
            self.logger.info("=== CLEANUP SUMMARY ===")
            self.logger.info(f"Files deleted: {self.stats['files_deleted']}")
            self.logger.info(f"Space freed: {format_size(self.stats['space_freed'])}")
            self.logger.info(f"Errors: {self.stats['errors']}")
            self.logger.info(f"Directories cleaned: {len(self.stats['directories_cleaned'])}")
            
        except Exception as e:
            self.logger.error(f"Error generating cleanup report: {e}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Market Research System - Data Cleanup')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--config', default='config/system/cleanup_config.yaml',
                       help='Path to cleanup configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        cleanup = DataCleanup(args.config)
        
        if args.verbose:
            cleanup.logger.setLevel(logging.DEBUG)
            
        cleanup.run_cleanup(dry_run=args.dry_run)
        
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()