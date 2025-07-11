#!/usr/bin/env python3
"""
System Health Check Script for Market Research System v1.0
Author: Independent Market Researcher
Date: January 2022
Purpose: Monitor system health, API connections, data integrity, and performance
"""

import os
import sys
import time
import json
import logging
import psutil
import requests
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system/system_health.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemHealthMonitor:
    """Comprehensive system health monitoring for market research system"""
    
    def __init__(self):
        self.health_report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'UNKNOWN',
            'checks': {},
            'alerts': [],
            'recommendations': []
        }
        
    def check_system_resources(self):
        """Check CPU, Memory, and Disk usage"""
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            self.health_report['checks']['system_resources'] = {
                'status': 'PASS',
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check thresholds
            if cpu_percent > 80:
                self.health_report['alerts'].append(f"HIGH CPU Usage: {cpu_percent}%")
            if memory_percent > 85:
                self.health_report['alerts'].append(f"HIGH Memory Usage: {memory_percent}%")
            if disk_percent > 90:
                self.health_report['alerts'].append(f"HIGH Disk Usage: {disk_percent}%")
                
            logger.info(f"System Resources - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
            
        except Exception as e:
            self.health_report['checks']['system_resources'] = {
                'status': 'FAIL',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            logger.error(f"System resource check failed: {e}")
    
    def check_api_connections(self):
        """Test connections to external APIs"""
        api_tests = {
            'yahoo_finance': self._test_yahoo_finance,
            'nse_india': self._test_nse_connection,
            'fred_api': self._test_fred_api
        }
        
        api_results = {}
        
        for api_name, test_func in api_tests.items():
            try:
                result = test_func()
                api_results[api_name] = result
                if not result['status']:
                    self.health_report['alerts'].append(f"API Connection Failed: {api_name}")
            except Exception as e:
                api_results[api_name] = {
                    'status': False,
                    'error': str(e),
                    'response_time': None
                }
                self.health_report['alerts'].append(f"API Test Error: {api_name} - {str(e)}")
        
        self.health_report['checks']['api_connections'] = api_results
        logger.info(f"API Connection tests completed: {len(api_results)} APIs tested")
    
    def _test_yahoo_finance(self):
        """Test Yahoo Finance API connection"""
        start_time = time.time()
        try:
            # Test with a major Indian stock
            ticker = yf.Ticker("RELIANCE.NS")
            info = ticker.info
            response_time = time.time() - start_time
            
            return {
                'status': True,
                'response_time': round(response_time, 2),
                'test_symbol': 'RELIANCE.NS',
                'data_received': bool(info)
            }
        except Exception as e:
            return {
                'status': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _test_nse_connection(self):
        """Test NSE website connectivity"""
        start_time = time.time()
        try:
            response = requests.get('https://www.nseindia.com', timeout=10)
            response_time = time.time() - start_time
            
            return {
                'status': response.status_code == 200,
                'response_time': round(response_time, 2),
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'status': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _test_fred_api(self):
        """Test FRED API for economic data"""
        start_time = time.time()
        try:
            # Test basic FRED connection (no API key required for basic test)
            response = requests.get('https://fred.stlouisfed.org', timeout=10)
            response_time = time.time() - start_time
            
            return {
                'status': response.status_code == 200,
                'response_time': round(response_time, 2),
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'status': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def check_data_integrity(self):
        """Check data files and directory structure"""
        try:
            required_dirs = [
                'data/raw/stocks',
                'data/processed/stocks',
                'data/cache',
                'logs/application',
                'reports/daily'
            ]
            
            integrity_results = {
                'directories': {},
                'recent_data': {},
                'file_counts': {}
            }
            
            # Check required directories
            for dir_path in required_dirs:
                if os.path.exists(dir_path):
                    integrity_results['directories'][dir_path] = 'EXISTS'
                else:
                    integrity_results['directories'][dir_path] = 'MISSING'
                    self.health_report['alerts'].append(f"Missing directory: {dir_path}")
            
            # Check for recent data files
            data_dir = Path('data/raw/stocks')
            if data_dir.exists():
                csv_files = list(data_dir.glob('*.csv'))
                integrity_results['file_counts']['stock_data_files'] = len(csv_files)
                
                # Check if we have recent data (within last 7 days)
                recent_files = []
                cutoff_date = datetime.now() - timedelta(days=7)
                
                for file_path in csv_files:
                    if file_path.stat().st_mtime > cutoff_date.timestamp():
                        recent_files.append(file_path.name)
                
                integrity_results['recent_data']['recent_files_count'] = len(recent_files)
                
                if len(recent_files) == 0:
                    self.health_report['alerts'].append("No recent data files found (within 7 days)")
            
            self.health_report['checks']['data_integrity'] = integrity_results
            logger.info("Data integrity check completed")
            
        except Exception as e:
            self.health_report['checks']['data_integrity'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"Data integrity check failed: {e}")
    
    def check_log_files(self):
        """Check log file sizes and recent entries"""
        try:
            log_files = [
                'logs/application/app.log',
                'logs/application/data_collection.log',
                'logs/application/analysis.log',
                'logs/system/performance.log'
            ]
            
            log_results = {}
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    stat = os.stat(log_file)
                    size_mb = stat.st_size / (1024 * 1024)
                    last_modified = datetime.fromtimestamp(stat.st_mtime)
                    
                    log_results[log_file] = {
                        'exists': True,
                        'size_mb': round(size_mb, 2),
                        'last_modified': last_modified.isoformat(),
                        'hours_since_update': (datetime.now() - last_modified).total_seconds() / 3600
                    }
                    
                    # Check for large log files
                    if size_mb > 100:
                        self.health_report['alerts'].append(f"Large log file: {log_file} ({size_mb:.1f} MB)")
                    
                    # Check for stale log files
                    hours_old = (datetime.now() - last_modified).total_seconds() / 3600
                    if hours_old > 24:
                        self.health_report['alerts'].append(f"Stale log file: {log_file} (last updated {hours_old:.1f} hours ago)")
                        
                else:
                    log_results[log_file] = {
                        'exists': False,
                        'error': 'File not found'
                    }
                    self.health_report['alerts'].append(f"Missing log file: {log_file}")
            
            self.health_report['checks']['log_files'] = log_results
            logger.info("Log file check completed")
            
        except Exception as e:
            self.health_report['checks']['log_files'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"Log file check failed: {e}")
    
    def check_database_connections(self):
        """Check database connectivity (if using databases)"""
        try:
            # For Version 1, we're using file-based storage
            # This check validates file-based data storage
            
            db_results = {
                'file_based_storage': {
                    'status': 'ACTIVE',
                    'type': 'CSV_FILES',
                    'location': 'data/'
                }
            }
            
            # Check if we can write to data directory
            test_file = 'data/.health_check_test'
            try:
                with open(test_file, 'w') as f:
                    f.write('health_check_test')
                os.remove(test_file)
                db_results['file_based_storage']['write_access'] = True
            except:
                db_results['file_based_storage']['write_access'] = False
                self.health_report['alerts'].append("Cannot write to data directory")
            
            self.health_report['checks']['database'] = db_results
            logger.info("Database/Storage check completed")
            
        except Exception as e:
            self.health_report['checks']['database'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            logger.error(f"Database check failed: {e}")
    
    def generate_recommendations(self):
        """Generate recommendations based on health check results"""
        recommendations = []
        
        # Resource-based recommendations
        if 'system_resources' in self.health_report['checks']:
            resources = self.health_report['checks']['system_resources']
            if resources.get('cpu_usage', 0) > 70:
                recommendations.append("Consider optimizing CPU-intensive processes or upgrading hardware")
            if resources.get('memory_usage', 0) > 80:
                recommendations.append("Monitor memory usage and consider increasing RAM")
            if resources.get('disk_usage', 0) > 85:
                recommendations.append("Clean up old data files or increase storage capacity")
        
        # API-based recommendations
        if 'api_connections' in self.health_report['checks']:
            api_results = self.health_report['checks']['api_connections']
            failed_apis = [api for api, result in api_results.items() if not result.get('status', False)]
            if failed_apis:
                recommendations.append(f"Fix API connections for: {', '.join(failed_apis)}")
        
        # Data-based recommendations
        if len(self.health_report['alerts']) > 5:
            recommendations.append("Multiple alerts detected - investigate system issues immediately")
            
        self.health_report['recommendations'] = recommendations
    
    def determine_overall_status(self):
        """Determine overall system status"""
        critical_alerts = [alert for alert in self.health_report['alerts'] 
                          if any(keyword in alert.lower() for keyword in ['fail', 'error', 'missing', 'high'])]
        
        if len(critical_alerts) == 0:
            self.health_report['system_status'] = 'HEALTHY'
        elif len(critical_alerts) <= 2:
            self.health_report['system_status'] = 'WARNING'
        else:
            self.health_report['system_status'] = 'CRITICAL'
    
    def save_health_report(self):
        """Save health report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'logs/system/health_report_{timestamp}.json'
        
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(self.health_report, f, indent=2)
        
        logger.info(f"Health report saved to: {report_file}")
        return report_file
    
    def run_full_health_check(self):
        """Run complete health check"""
        logger.info("Starting comprehensive system health check...")
        
        # Create required directories if they don't exist
        os.makedirs('logs/system', exist_ok=True)
        os.makedirs('logs/application', exist_ok=True)
        os.makedirs('data/raw/stocks', exist_ok=True)
        os.makedirs('data/processed/stocks', exist_ok=True)
        
        # Run all checks
        self.check_system_resources()
        self.check_api_connections()
        self.check_data_integrity()
        self.check_log_files()
        self.check_database_connections()
        
        # Generate insights
        self.generate_recommendations()
        self.determine_overall_status()
        
        # Save report
        report_file = self.save_health_report()
        
        # Print summary
        self.print_summary()
        
        logger.info("System health check completed")
        return self.health_report
    
    def print_summary(self):
        """Print health check summary"""
        print("\n" + "="*60)
        print("MARKET RESEARCH SYSTEM - HEALTH CHECK SUMMARY")
        print("="*60)
        print(f"Timestamp: {self.health_report['timestamp']}")
        print(f"Overall Status: {self.health_report['system_status']}")
        print(f"Total Alerts: {len(self.health_report['alerts'])}")
        
        if self.health_report['alerts']:
            print("\nALERTS:")
            for i, alert in enumerate(self.health_report['alerts'], 1):
                print(f"  {i}. {alert}")
        
        if self.health_report['recommendations']:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(self.health_report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\nCHECK RESULTS:")
        for check_name, result in self.health_report['checks'].items():
            status = result.get('status', 'COMPLETED')
            print(f"  {check_name.upper()}: {status}")
        
        print("="*60)

def main():
    """Main function to run system health check"""
    try:
        monitor = SystemHealthMonitor()
        health_report = monitor.run_full_health_check()
        
        # Exit with appropriate code
        if health_report['system_status'] == 'CRITICAL':
            sys.exit(2)
        elif health_report['system_status'] == 'WARNING':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Health check failed with error: {e}")
        print(f"CRITICAL ERROR: Health check failed - {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()