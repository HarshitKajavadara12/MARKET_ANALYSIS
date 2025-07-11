#!/usr/bin/env python3
"""
Market Research System v1.0 - Data Backup Script
Automated backup system for critical market research data

Author: Market Research Team
Date: 2022
Focus: Indian Stock Market (NSE/BSE)
"""

import os
import sys
import logging
import shutil
import zipfile
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import yaml
import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logging_utils import setup_logging
from src.utils.date_utils import get_trading_days
from src.utils.file_utils import get_file_size, format_size

class DataBackup:
    """Handle automated backup of market research data"""
    
    def __init__(self, config_path="config/system/backup_config.yaml"):
        """Initialize backup manager with configuration"""
        self.setup_logging()
        self.load_config(config_path)
        self.backup_stats = {
            'files_backed_up': 0,
            'total_size': 0,
            'compression_ratio': 0,
            'backup_time': 0,
            'errors': 0
        }
        
    def setup_logging(self):
        """Setup logging for backup operations"""
        self.logger = setup_logging(
            'data_backup', 
            'logs/maintenance/backup.log',
            level=logging.INFO
        )
        
    def load_config(self, config_path):
        """Load backup configuration"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                # Default configuration for Indian market backup
                self.config = {
                    'backup_paths': {
                        'critical_data': [
                            'data/processed/stocks/nse',
                            'data/processed/stocks/bse', 
                            'data/processed/indices',
                            'data/processed/technical_indicators',
                            'config',
                            'src'
                        ],
                        'reports': [
                            'reports/daily',
                            'reports/weekly',
                            'reports/monthly'
                        ],
                        'analysis': [
                            'notebooks/research',
                            'data/processed/analysis_results'
                        ]
                    },
                    'backup_destination': 'data/backups',
                    'retention': {
                        'daily_backups': 7,    # Keep 7 days
                        'weekly_backups': 4,   # Keep 4 weeks
                        'monthly_backups': 12  # Keep 12 months
                    },
                    'compression': {
                        'method': 'zip',  # zip, tar.gz, tar.bz2
                        'level': 6        # 1-9 compression level
                    },
                    'backup_schedule': {
                        'daily': True,
                        'weekly': True,
                        'monthly': True
                    },
                    'exclude_patterns': [
                        '*.tmp', '*.temp', '*~', '*.log',
                        '__pycache__', '.git', '.pytest_cache',
                        '*.pyc', '.DS_Store'
                    ],
                    'verify_backup': True,
                    'max_parallel_jobs': 4
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
            self.logger.info(f"Created default backup config at {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            
    def should_exclude_file(self, file_path):
        """Check if file should be excluded from backup"""
        file_name = os.path.basename(file_path)
        
        for pattern in self.config['exclude_patterns']:
            if pattern.startswith('*') and file_name.endswith(pattern[1:]):
                return True
            elif pattern.endswith('*') and file_name.startswith(pattern[:-1]):
                return True
            elif pattern in file_path:
                return True
                
        return False
        
    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file for verification"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
            
    def create_zip_backup(self, source_paths, backup_file):
        """Create ZIP backup of specified paths"""
        try:
            compression_level = self.config['compression']['level']
            
            with zipfile.ZipFile(backup_file, 'w', 
                               zipfile.ZIP_DEFLATED, 
                               compresslevel=compression_level) as zipf:
                
                for source_path in source_paths:
                    if os.path.isfile(source_path):
                        if not self.should_exclude_file(source_path):
                            zipf.write(source_path)
                            self.backup_stats['files_backed_up'] += 1
                            
                    elif os.path.isdir(source_path):
                        for root, dirs, files in os.walk(source_path):
                            # Remove excluded directories
                            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.config['exclude_patterns'])]
                            
                            for file in files:
                                file_path = os.path.join(root, file)
                                if not self.should_exclude_file(file_path):
                                    arcname = os.path.relpath(file_path)
                                    zipf.write(file_path, arcname)
                                    self.backup_stats['files_backed_up'] += 1
                                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating ZIP backup: {e}")
            return False
            
    def create_tar_backup(self, source_paths, backup_file, compression='gz'):
        """Create TAR backup of specified paths"""
        try:
            mode = f'w:{compression}' if compression else 'w'
            
            with tarfile.open(backup_file, mode) as tar:
                for source_path in source_paths:
                    if os.path.isfile(source_path):
                        if not self.should_exclude_file(source_path):
                            tar.add(source_path)
                            self.backup_stats['files_backed_up'] += 1
                            
                    elif os.path.isdir(source_path):
                        def exclude_filter(tarinfo):
                            if self.should_exclude_file(tarinfo.name):
                                return None
                            return tarinfo
                            
                        tar.add(source_path, filter=exclude_filter)
                        
                        # Count files added
                        for root, dirs, files in os.walk(source_path):
                            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.config['exclude_patterns'])]
                            for file in files:
                                file_path = os.path.join(root, file)
                                if not self.should_exclude_file(file_path):
                                    self.backup_stats['files_backed_up'] += 1
                                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating TAR backup: {e}")
            return False
            
    def create_backup(self, backup_type, source_paths):
        """Create backup of specified type"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = os.path.join(self.config['backup_destination'], backup_type)
            os.makedirs(backup_dir, exist_ok=True)
            
            # Determine backup filename and method
            compression_method = self.config['compression']['method']
            
            if compression_method == 'zip':
                backup_file = os.path.join(backup_dir, f'market_research_{backup_type}_{timestamp}.zip')
                success = self.create_zip_backup(source_paths, backup_file)
            elif compression_method == 'tar.gz':
                backup_file = os.path.join(backup_dir, f'market_research_{backup_type}_{timestamp}.tar.gz')
                success = self.create_tar_backup(source_paths, backup_file, 'gz')
            elif compression_method == 'tar.bz2':
                backup_file = os.path.join(backup_dir, f'market_research_{backup_type}_{timestamp}.tar.bz2')
                success = self.create_tar_backup(source_paths, backup_file, 'bz2')
            else:
                backup_file = os.path.join(backup_dir, f'market_research_{backup_type}_{timestamp}.tar')
                success = self.create_tar_backup(source_paths, backup_file, None)
                
            if success:
                # Calculate backup statistics
                backup_size = get_file_size(backup_file)
                self.backup_stats['total_size'] = backup_size
                
                # Calculate original size for compression ratio
                original_size = 0
                for source_path in source_paths:
                    if os.path.exists(source_path):
                        if os.path.isfile(source_path):
                            original_size += get_file_size(source_path)
                        else:
                            for root, dirs, files in os.walk(source_path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    if not self.should_exclude_file(file_path):
                                        try:
                                            original_size += get_file_size(file_path)
                                        except Exception:
                                            pass
                
                if original_size > 0:
                    self.backup_stats['compression_ratio'] = (1 - backup_size / original_size) * 100
                
                self.logger.info(f"Backup created: {backup_file}")
                self.logger.info(f"Original size: {format_size(original_size)}")
                self.logger.info(f"Backup size: {format_size(backup_size)}")
                self.logger.info(f"Compression ratio: {self.backup_stats['compression_ratio']:.1f}%")
                
                # Verify backup if configured
                if self.config['verify_backup']:
                    if self.verify_backup(backup_file):
                        self.logger.info("Backup verification successful")
                    else:
                        self.logger.error("Backup verification failed")
                        return False
                        
                return backup_file
                
            else:
                self.logger.error(f"Failed to create backup: {backup_file}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating {backup_type} backup: {e}")
            self.backup_stats['errors'] += 1
            return None
            
    def verify_backup(self, backup_file):
        """Verify backup file integrity"""
        try:
            if backup_file.endswith('.zip'):
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    # Test ZIP file integrity
                    bad_file = zipf.testzip()
                    if bad_file:
                        self.logger.error(f"Corrupted file in backup: {bad_file}")
                        return False
                        
            elif backup_file.endswith(('.tar.gz', '.tar.bz2', '.tar')):
                with tarfile.open(backup_file, 'r:*') as tar:
                    # Verify TAR file can be read
                    members = tar.getmembers()
                    if not members:
                        self.logger.error("Backup file appears to be empty")
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying backup {backup_file}: {e}")
            return False
            
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        try:
            backup_base = self.config['backup_destination']
            retention = self.config['retention']
            
            for backup_type, retention_days in retention.items():
                backup_dir = os.path.join(backup_base, backup_type.replace('_backups', ''))
                
                if os.path.exists(backup_dir):
                    cutoff_date = datetime.now() - timedelta(days=retention_days)
                    
                    for file in os.listdir(backup_dir):
                        file_path = os.path.join(backup_dir, file)
                        if os.path.isfile(file_path):
                            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_time < cutoff_date:
                                try:
                                    os.remove(file_path)
                                    self.logger.info(f"Removed old backup: {file}")
                                except Exception as e:
                                    self.logger.error(f"Error removing {file}: {e}")
                                    
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def get_backup_manifest(self, backup_type):
        """Generate manifest of files to backup"""
        manifest = []
        
        if backup_type == 'daily':
            paths = self.config['backup_paths']['critical_data']
        elif backup_type == 'weekly':
            paths = (self.config['backup_paths']['critical_data'] + 
                    self.config['backup_paths']['reports'])
        elif backup_type == 'monthly':
            paths = (self.config['backup_paths']['critical_data'] + 
                    self.config['backup_paths']['reports'] + 
                    self.config['backup_paths']['analysis'])
        else:
            paths = self.config['backup_paths']['critical_data']
            
        for path in paths:
            if os.path.exists(path):
                manifest.append(path)
            else:
                self.logger.warning(f"Backup path does not exist: {path}")
                
        return manifest
        
    def run_daily_backup(self):
        """Execute daily backup routine"""
        self.logger.info("Starting daily backup routine")
        start_time = datetime.now()
        
        try:
            # Get files to backup
            backup_paths = self.get_backup_manifest('daily')
            
            if not backup_paths:
                self.logger.warning("No files found for daily backup")
                return False
                
            # Create backup
            backup_file = self.create_backup('daily', backup_paths)
            
            if backup_file:
                self.backup_stats['backup_time'] = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Daily backup completed in {self.backup_stats['backup_time']:.2f} seconds")
                
                # Generate backup report
                self.generate_backup_report('daily', backup_file)
                return True
            else:
                self.logger.error("Daily backup failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in daily backup: {e}")
            return False
            
    def run_weekly_backup(self):
        """Execute weekly backup routine"""
        self.logger.info("Starting weekly backup routine")
        start_time = datetime.now()
        
        try:
            # Get files to backup (includes reports)
            backup_paths = self.get_backup_manifest('weekly')
            
            if not backup_paths:
                self.logger.warning("No files found for weekly backup")
                return False
                
            # Create backup
            backup_file = self.create_backup('weekly', backup_paths)
            
            if backup_file:
                self.backup_stats['backup_time'] = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Weekly backup completed in {self.backup_stats['backup_time']:.2f} seconds")
                
                # Generate backup report
                self.generate_backup_report('weekly', backup_file)
                return True
            else:
                self.logger.error("Weekly backup failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in weekly backup: {e}")
            return False
            
    def run_monthly_backup(self):
        """Execute monthly backup routine"""
        self.logger.info("Starting monthly backup routine")
        start_time = datetime.now()
        
        try:
            # Get all files to backup
            backup_paths = self.get_backup_manifest('monthly')
            
            if not backup_paths:
                self.logger.warning("No files found for monthly backup")
                return False
                
            # Create backup
            backup_file = self.create_backup('monthly', backup_paths)
            
            if backup_file:
                self.backup_stats['backup_time'] = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Monthly backup completed in {self.backup_stats['backup_time']:.2f} seconds")
                
                # Generate backup report
                self.generate_backup_report('monthly', backup_file)
                return True
            else:
                self.logger.error("Monthly backup failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in monthly backup: {e}")
            return False
            
    def generate_backup_report(self, backup_type, backup_file):
        """Generate backup completion report"""
        try:
            report_data = {
                'backup_type': backup_type,
                'backup_file': backup_file,
                'timestamp': datetime.now().isoformat(),
                'stats': self.backup_stats.copy()
            }
            
            # Save report as JSON
            report_dir = os.path.join('reports', 'backup_reports')
            os.makedirs(report_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(report_dir, f'backup_report_{backup_type}_{timestamp}.json')
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            self.logger.info(f"Backup report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating backup report: {e}")
            
    def check_backup_health(self):
        """Check health of backup system and recent backups"""
        try:
            backup_base = self.config['backup_destination']
            health_report = {
                'backup_directories': {},
                'recent_backups': {},
                'disk_usage': {},
                'health_status': 'healthy'
            }
            
            # Check each backup type directory
            for backup_type in ['daily', 'weekly', 'monthly']:
                backup_dir = os.path.join(backup_base, backup_type)
                
                if os.path.exists(backup_dir):
                    files = [f for f in os.listdir(backup_dir) if os.path.isfile(os.path.join(backup_dir, f))]
                    health_report['backup_directories'][backup_type] = {
                        'exists': True,
                        'file_count': len(files),
                        'total_size': sum(get_file_size(os.path.join(backup_dir, f)) for f in files)
                    }
                    
                    # Check for recent backups (last 24 hours for daily, 7 days for weekly, 30 days for monthly)
                    recent_threshold = {
                        'daily': 1,
                        'weekly': 7,
                        'monthly': 30
                    }
                    
                    cutoff_time = datetime.now() - timedelta(days=recent_threshold[backup_type])
                    recent_files = []
                    
                    for file in files:
                        file_path = os.path.join(backup_dir, file)
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time > cutoff_time:
                            recent_files.append({
                                'file': file,
                                'timestamp': file_time.isoformat(),
                                'size': get_file_size(file_path)
                            })
                    
                    health_report['recent_backups'][backup_type] = recent_files
                    
                    # Mark as unhealthy if no recent backups
                    if not recent_files:
                        health_report['health_status'] = 'warning'
                        self.logger.warning(f"No recent {backup_type} backups found")
                        
                else:
                    health_report['backup_directories'][backup_type] = {'exists': False}
                    health_report['health_status'] = 'unhealthy'
                    self.logger.error(f"Backup directory missing: {backup_dir}")
                    
            # Check disk usage
            if os.path.exists(backup_base):
                total, used, free = shutil.disk_usage(backup_base)
                health_report['disk_usage'] = {
                    'total': total,
                    'used': used,
                    'free': free,
                    'usage_percent': (used / total) * 100
                }
                
                # Warn if disk usage is high
                if (used / total) > 0.8:  # 80% threshold
                    health_report['health_status'] = 'warning'
                    self.logger.warning(f"High disk usage: {(used/total)*100:.1f}%")
                    
            return health_report
            
        except Exception as e:
            self.logger.error(f"Error checking backup health: {e}")
            return {'health_status': 'error', 'error': str(e)}
            
    def restore_backup(self, backup_file, restore_path):
        """Restore files from backup"""
        try:
            self.logger.info(f"Starting restore from {backup_file} to {restore_path}")
            
            if not os.path.exists(backup_file):
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
                
            # Create restore directory
            os.makedirs(restore_path, exist_ok=True)
            
            # Extract based on file type
            if backup_file.endswith('.zip'):
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    zipf.extractall(restore_path)
                    
            elif backup_file.endswith(('.tar.gz', '.tar.bz2', '.tar')):
                with tarfile.open(backup_file, 'r:*') as tar:
                    tar.extractall(restore_path)
                    
            else:
                self.logger.error(f"Unsupported backup file format: {backup_file}")
                return False
                
            self.logger.info(f"Restore completed successfully to {restore_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during restore: {e}")
            return False

def main():
    """Main function to handle command line arguments and execute backup operations"""
    parser = argparse.ArgumentParser(description='Market Research Data Backup System v1.0')
    parser.add_argument('--type', choices=['daily', 'weekly', 'monthly', 'all'], 
                       default='daily', help='Type of backup to perform')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Clean up old backups based on retention policy')
    parser.add_argument('--health', action='store_true',
                       help='Check backup system health')
    parser.add_argument('--restore', nargs=2, metavar=('BACKUP_FILE', 'RESTORE_PATH'),
                       help='Restore backup to specified path')
    parser.add_argument('--config', default='config/system/backup_config.yaml',
                       help='Path to backup configuration file')
    
    args = parser.parse_args()
    
    # Initialize backup system
    backup_system = DataBackup(args.config)
    
    try:
        if args.health:
            # Check system health
            health = backup_system.check_backup_health()
            print(f"Backup System Health: {health['health_status']}")
            print(json.dumps(health, indent=2))
            
        elif args.restore:
            # Restore backup
            backup_file, restore_path = args.restore
            success = backup_system.restore_backup(backup_file, restore_path)
            sys.exit(0 if success else 1)
            
        elif args.cleanup:
            # Clean up old backups
            backup_system.cleanup_old_backups()
            print("Cleanup completed")
            
        else:
            # Perform backup
            success = False
            
            if args.type == 'daily':
                success = backup_system.run_daily_backup()
            elif args.type == 'weekly':
                success = backup_system.run_weekly_backup()
            elif args.type == 'monthly':
                success = backup_system.run_monthly_backup()
            elif args.type == 'all':
                daily_success = backup_system.run_daily_backup()
                weekly_success = backup_system.run_weekly_backup()
                monthly_success = backup_system.run_monthly_backup()
                success = daily_success and weekly_success and monthly_success
                
            # Clean up old backups after successful backup
            if success:
                backup_system.cleanup_old_backups()
                print(f"{args.type.title()} backup completed successfully")
                
                # Print backup statistics
                stats = backup_system.backup_stats
                print(f"Files backed up: {stats['files_backed_up']}")
                print(f"Total size: {format_size(stats['total_size'])}")
                print(f"Compression ratio: {stats['compression_ratio']:.1f}%")
                print(f"Backup time: {stats['backup_time']:.2f} seconds")
                
            else:
                print(f"{args.type.title()} backup failed")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nBackup operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()