#!/usr/bin/env python3
"""
Market Research System - Health Check Module
Version 1.0 (2022)

This module provides comprehensive system health monitoring for the market research system.
It checks API connectivity, data pipeline status, disk space, memory usage, and service health.
"""

import os
import sys
import json
import time
import psutil
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd
from fredapi import Fred

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.settings import *
from src.utils.logging_utils import setup_logger

class SystemHealthChecker:
    """
    Comprehensive system health monitoring for market research system
    """
    
    def __init__(self):
        self.logger = setup_logger('health_check')
        self.start_time = datetime.now()
        self.health_status = {
            'timestamp': self.start_time.isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {}
        }
        
        # Health thresholds
        self.thresholds = {
            'disk_usage_warning': 80,  # Percentage
            'disk_usage_critical': 90,
            'memory_usage_warning': 80,
            'memory_usage_critical': 90,
            'cpu_usage_warning': 80,
            'cpu_usage_critical': 95,
            'api_response_time_warning': 5.0,  # Seconds
            'api_response_time_critical': 10.0,
            'data_freshness_hours': 24  # Hours
        }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage (CPU, Memory, Disk)"""
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk Usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Determine status levels
            cpu_status = self._get_status_level(cpu_percent, 'cpu_usage')
            memory_status = self._get_status_level(memory_percent, 'memory_usage')
            disk_status = self._get_status_level(disk_percent, 'disk_usage')
            
            overall_status = max([cpu_status, memory_status, disk_status], 
                               key=lambda x: ['OK', 'WARNING', 'CRITICAL'].index(x))
            
            return {
                'status': overall_status,
                'details': {
                    'cpu': {
                        'usage_percent': cpu_percent,
                        'status': cpu_status
                    },
                    'memory': {
                        'usage_percent': memory_percent,
                        'total_gb': round(memory.total / (1024**3), 2),
                        'available_gb': round(memory.available / (1024**3), 2),
                        'status': memory_status
                    },
                    'disk': {
                        'usage_percent': round(disk_percent, 2),
                        'total_gb': round(disk_usage.total / (1024**3), 2),
                        'free_gb': round(disk_usage.free / (1024**3), 2),
                        'status': disk_status
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check system resources: {e}")
            return {
                'status': 'CRITICAL',
                'error': str(e)
            }
    
    def check_api_connectivity(self) -> Dict[str, Any]:
        """Check connectivity to external APIs"""
        api_checks = {}
        
        # Yahoo Finance API Check
        try:
            start_time = time.time()
            ticker = yf.Ticker("RELIANCE.NS")  # Indian stock
            data = ticker.history(period="1d")
            response_time = time.time() - start_time
            
            if not data.empty:
                status = self._get_status_level(response_time, 'api_response_time')
                api_checks['yahoo_finance'] = {
                    'status': status,
                    'response_time': round(response_time, 2),
                    'last_data_point': data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                api_checks['yahoo_finance'] = {
                    'status': 'CRITICAL',
                    'error': 'No data returned'
                }
                
        except Exception as e:
            api_checks['yahoo_finance'] = {
                'status': 'CRITICAL',
                'error': str(e)
            }
        
        # FRED API Check (if API key available)
        try:
            fred_api_key = os.getenv('FRED_API_KEY')
            if fred_api_key:
                start_time = time.time()
                fred = Fred(api_key=fred_api_key)
                data = fred.get_series('GDP', limit=1)
                response_time = time.time() - start_time
                
                if not data.empty:
                    status = self._get_status_level(response_time, 'api_response_time')
                    api_checks['fred'] = {
                        'status': status,
                        'response_time': round(response_time, 2),
                        'last_data_point': data.index[-1].strftime('%Y-%m-%d')
                    }
                else:
                    api_checks['fred'] = {
                        'status': 'CRITICAL',
                        'error': 'No data returned'
                    }
            else:
                api_checks['fred'] = {
                    'status': 'WARNING',
                    'error': 'API key not configured'
                }
                
        except Exception as e:
            api_checks['fred'] = {
                'status': 'CRITICAL',
                'error': str(e)
            }
        
        # Determine overall API status
        statuses = [check.get('status', 'CRITICAL') for check in api_checks.values()]
        if 'CRITICAL' in statuses:
            overall_status = 'CRITICAL'
        elif 'WARNING' in statuses:
            overall_status = 'WARNING'
        else:
            overall_status = 'OK'
        
        return {
            'status': overall_status,
            'apis': api_checks
        }
    
    def check_data_freshness(self) -> Dict[str, Any]:
        """Check if market data is fresh and up-to-date"""
        try:
            data_dir = Path('data/raw/stocks/daily')
            if not data_dir.exists():
                return {
                    'status': 'CRITICAL',
                    'error': 'Data directory not found'
                }
            
            # Find the most recent data file
            data_files = list(data_dir.glob('*.csv'))
            if not data_files:
                return {
                    'status': 'CRITICAL',
                    'error': 'No data files found'
                }
            
            # Get the most recently modified file
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            file_age_hours = (time.time() - latest_file.stat().st_mtime) / 3600
            
            if file_age_hours <= self.thresholds['data_freshness_hours']:
                status = 'OK'
            elif file_age_hours <= self.thresholds['data_freshness_hours'] * 2:
                status = 'WARNING'
            else:
                status = 'CRITICAL'
            
            return {
                'status': status,
                'details': {
                    'latest_file': latest_file.name,
                    'age_hours': round(file_age_hours, 2),
                    'last_modified': datetime.fromtimestamp(
                        latest_file.stat().st_mtime
                    ).isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check data freshness: {e}")
            return {
                'status': 'CRITICAL',
                'error': str(e)
            }
    
    def check_log_files(self) -> Dict[str, Any]:
        """Check log files for errors and warnings"""
        try:
            log_dir = Path('logs/application')
            if not log_dir.exists():
                return {
                    'status': 'WARNING',
                    'error': 'Log directory not found'
                }
            
            # Check error log
            error_log = log_dir / 'errors.log'
            recent_errors = 0
            
            if error_log.exists():
                # Count errors in the last 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                try:
                    with open(error_log, 'r') as f:
                        for line in f:
                            # Simple timestamp parsing (assumes standard format)
                            if line.strip() and cutoff_time.strftime('%Y-%m-%d') in line:
                                if 'ERROR' in line or 'CRITICAL' in line:
                                    recent_errors += 1
                except Exception:
                    pass
            
            # Determine status based on error count
            if recent_errors == 0:
                status = 'OK'
            elif recent_errors <= 5:
                status = 'WARNING'
            else:
                status = 'CRITICAL'
            
            return {
                'status': status,
                'details': {
                    'recent_errors_24h': recent_errors,
                    'error_log_exists': error_log.exists(),
                    'log_directory': str(log_dir)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check log files: {e}")
            return {
                'status': 'WARNING',
                'error': str(e)
            }
    
    def check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity (if database is used)"""
        try:
            # For this basic version, we'll check if required directories exist
            required_dirs = [
                'data/raw',
                'data/processed',
                'data/cache',
                'reports/daily',
                'logs/application'
            ]
            
            missing_dirs = []
            for dir_path in required_dirs:
                if not Path(dir_path).exists():
                    missing_dirs.append(dir_path)
            
            if not missing_dirs:
                status = 'OK'
                details = {'all_directories_exist': True}
            else:
                status = 'WARNING'
                details = {
                    'missing_directories': missing_dirs,
                    'total_missing': len(missing_dirs)
                }
            
            return {
                'status': status,
                'details': details
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check database connectivity: {e}")
            return {
                'status': 'CRITICAL',
                'error': str(e)
            }
    
    def check_scheduled_tasks(self) -> Dict[str, Any]:
        """Check if scheduled tasks are running properly"""
        try:
            # Check if cron jobs or scheduled scripts have run recently
            script_dir = Path('scripts')
            if not script_dir.exists():
                return {
                    'status': 'WARNING',
                    'error': 'Scripts directory not found'
                }
            
            # Look for evidence of recent script execution
            log_files = list(Path('logs/application').glob('*.log'))
            recent_activity = False
            
            for log_file in log_files:
                if log_file.exists():
                    # Check if modified in last 24 hours
                    file_age_hours = (time.time() - log_file.stat().st_mtime) / 3600
                    if file_age_hours <= 24:
                        recent_activity = True
                        break
            
            status = 'OK' if recent_activity else 'WARNING'
            
            return {
                'status': status,
                'details': {
                    'recent_activity_24h': recent_activity,
                    'checked_log_files': len(log_files)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check scheduled tasks: {e}")
            return {
                'status': 'WARNING',
                'error': str(e)
            }
    
    def _get_status_level(self, value: float, metric_type: str) -> str:
        """Determine status level based on thresholds"""
        warning_threshold = self.thresholds.get(f'{metric_type}_warning', 80)
        critical_threshold = self.thresholds.get(f'{metric_type}_critical', 90)
        
        if metric_type.startswith('api_response_time'):
            # For response time, higher is worse
            if value >= critical_threshold:
                return 'CRITICAL'
            elif value >= warning_threshold:
                return 'WARNING'
            else:
                return 'OK'
        else:
            # For usage metrics, higher percentage is worse
            if value >= critical_threshold:
                return 'CRITICAL'
            elif value >= warning_threshold:
                return 'WARNING'
            else:
                return 'OK'
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        self.logger.info("Starting comprehensive health check")
        
        # Run individual checks
        checks = {
            'system_resources': self.check_system_resources(),
            'api_connectivity': self.check_api_connectivity(),
            'data_freshness': self.check_data_freshness(),
            'log_files': self.check_log_files(),
            'database_connectivity': self.check_database_connectivity(),
            'scheduled_tasks': self.check_scheduled_tasks()
        }
        
        # Determine overall system status
        all_statuses = [check.get('status', 'CRITICAL') for check in checks.values()]
        
        if 'CRITICAL' in all_statuses:
            overall_status = 'CRITICAL'
        elif 'WARNING' in all_statuses:
            overall_status = 'WARNING'
        else:
            overall_status = 'OK'
        
        # Update health status
        self.health_status.update({
            'overall_status': overall_status,
            'checks': checks,
            'check_duration_seconds': (datetime.now() - self.start_time).total_seconds()
        })
        
        self.logger.info(f"Health check completed. Overall status: {overall_status}")
        return self.health_status
    
    def save_health_report(self, output_dir: str = 'logs/system') -> str:
        """Save health check results to file"""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'health_check_{timestamp}.json'
            filepath = Path(output_dir) / filename
            
            with open(filepath, 'w') as f:
                json.dump(self.health_status, f, indent=2, default=str)
            
            self.logger.info(f"Health report saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save health report: {e}")
            return ""


def main():
    """Main function to run health checks"""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create health checker
        health_checker = SystemHealthChecker()
        
        # Run all checks
        health_status = health_checker.run_all_checks()
        
        # Save report
        report_path = health_checker.save_health_report()
        
        # Print summary to console
        print(f"\n{'='*60}")
        print(f"MARKET RESEARCH SYSTEM - HEALTH CHECK REPORT")
        print(f"{'='*60}")
        print(f"Timestamp: {health_status['timestamp']}")
        print(f"Overall Status: {health_status['overall_status']}")
        print(f"Check Duration: {health_status['check_duration_seconds']:.2f} seconds")
        print(f"Report Saved: {report_path}")
        print(f"{'='*60}")
        
        # Print individual check results
        for check_name, check_result in health_status['checks'].items():
            status = check_result.get('status', 'UNKNOWN')
            print(f"{check_name.replace('_', ' ').title()}: {status}")
            
            if 'error' in check_result:
                print(f"  Error: {check_result['error']}")
        
        print(f"{'='*60}\n")
        
        # Exit with appropriate code
        if health_status['overall_status'] == 'CRITICAL':
            sys.exit(2)
        elif health_status['overall_status'] == 'WARNING':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()