#!/usr/bin/env python3
"""
Market Research System - Performance Monitor Module
Version 1.0 (2022)

This module provides comprehensive performance monitoring for the market research system.
It tracks system metrics, data processing performance, API response times, and generates
performance reports with trend analysis.
"""

import os
import sys
import json
import time
import psutil
import threading
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logging_utils import setup_logger

@dataclass
class PerformanceMetric:
    """Data class for performance metrics"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    category: str
    details: Optional[Dict[str, Any]] = None

class PerformanceDatabase:
    """SQLite database for storing performance metrics"""
    
    def __init__(self, db_path: str = 'logs/system/performance.db'):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize performance database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    category TEXT NOT NULL,
                    details TEXT
                )
            ''')
            
            # Create indexes for better query performance
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON performance_metrics(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metric_category 
                ON performance_metrics(metric_name, category)
            ''')
    
    def store_metric(self, metric: PerformanceMetric):
        """Store a performance metric"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO performance_metrics 
                (timestamp, metric_name, value, unit, category, details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metric.timestamp.isoformat(),
                metric.metric_name,
                metric.value,
                metric.unit,
                metric.category,
                json.dumps(metric.details) if metric.details else None
            ))
    
    def get_metrics(self, 
                   metric_name: Optional[str] = None,
                   category: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Retrieve performance metrics with filters"""
        
        query = "SELECT timestamp, metric_name, value, unit, category, details FROM performance_metrics WHERE 1=1"
        params = []
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor.fetchall():
                timestamp = datetime.fromisoformat(row[0])
                details = json.loads(row[5]) if row[5] else None
                
                results.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name=row[1],
                    value=row[2],
                    unit=row[3],
                    category=row[4],
                    details=details
                ))
            
            return results

class SystemPerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, db: PerformanceDatabase):
        self.db = db
        self.logger = setup_logger('performance_monitor')
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 60  # seconds
        
        # In-memory storage for real-time monitoring
        self.real_time_metrics = {
            'cpu_usage': deque(maxlen=60),
            'memory_usage': deque(maxlen=60),
            'disk_io': deque(maxlen=60),
            'network_io': deque(maxlen=60)
        }
    
    def collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect current system performance metrics"""
        timestamp = datetime.now()
        metrics = []
        
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name='cpu_usage',
                value=cpu_percent,
                unit='percent',
                category='system',
                details={'cores': psutil.cpu_count()}
            ))
            
            # Memory Usage
            memory = psutil.virtual_memory()
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name='memory_usage',
                value=memory.percent,
                unit='percent',
                category='system',
                details={
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2)
                }
            ))
            
            # Disk Usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name='disk_usage',
                value=disk_percent,
                unit='percent',
                category='system',
                details={
                    'total_gb': round(disk_usage.total / (1024**3), 2),
                    'free_gb': round(disk_usage.free / (1024**3), 2),
                    'used_gb': round(disk_usage.used / (1024**3), 2)
                }
            ))
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name='disk_read_bytes',
                    value=disk_io.read_bytes,
                    unit='bytes',
                    category='system'
                ))
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name='disk_write_bytes',
                    value=disk_io.write_bytes,
                    unit='bytes',
                    category='system'
                ))
            
            # Network I/O
            network_io = psutil.net_io_counters()
            if network_io:
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name='network_bytes_sent',
                    value=network_io.bytes_sent,
                    unit='bytes',
                    category='system'
                ))
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name='network_bytes_recv',
                    value=network_io.bytes_recv,
                    unit='bytes',
                    category='system'
                ))
            
            # Load Average (Unix-like systems)
            try:
                load_avg = os.getloadavg()
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name='load_average_1min',
                    value=load_avg[0],
                    unit='load',
                    category='system'
                ))
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name='load_average_5min',
                    value=load_avg[1],
                    unit='load',
                    category='system'
                ))
                metrics.append(PerformanceMetric(
                    timestamp=timestamp,
                    metric_name='load_average_15min',
                    value=load_avg[2],
                    unit='load',
                    category='system'
                ))
            except (OSError, AttributeError):
                # Not available on Windows
                pass
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring:
            self.logger.warning("Performance monitoring is already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect and store metrics
                metrics = self.collect_system_metrics()
                for metric in metrics:
                    self.db.store_metric(metric)
                    
                    # Store in real-time cache
                    if metric.metric_name in self.real_time_metrics:
                        self.real_time_metrics[metric.metric_name].append({
                            'timestamp': metric.timestamp,
                            'value': metric.value
                        })
                
                # Sleep until next collection
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying

class ApplicationPerformanceMonitor:
    """Monitor application-specific performance metrics"""
    
    def __init__(self, db: PerformanceDatabase):
        self.db = db
        self.logger = setup_logger('app_performance_monitor')
        self.active_operations = {}
    
    def start_operation(self, operation_name: str, operation_id: Optional[str] = None) -> str:
        """Start timing an operation"""
        if operation_id is None:
            operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        self.active_operations[operation_id] = {
            'name': operation_name,
            'start_time': time.time(),
            'start_datetime': datetime.now()
        }
        
        return operation_id
    
    def end_operation(self, operation_id: str, 
                     details: Optional[Dict[str, Any]] = None) -> float:
        """End timing an operation and store the metric"""
        if operation_id not in self.active_operations:
            self.logger.warning(f"Operation {operation_id} not found in active operations")
            return 0.0
        
        operation = self.active_operations.pop(operation_id)
        duration = time.time() - operation['start_time']
        
        # Store performance metric
        metric = PerformanceMetric(
            timestamp=operation['start_datetime'],
            metric_name=f"{operation['name']}_duration",
            value=duration,
            unit='seconds',
            category='application',
            details=details
        )
        
        self.db.store_metric(metric)
        self.logger.debug(f"Operation {operation['name']} completed in {duration:.3f} seconds")
        
        return duration
    
    def record_data_processing_metrics(self, 
                                     operation: str,
                                     records_processed: int,
                                     duration: float,
                                     memory_used: Optional[float] = None):
        """Record data processing performance metrics"""
        timestamp = datetime.now()
        
        # Records per second
        if duration > 0:
            rps = records_processed / duration
            self.db.store_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_name=f"{operation}_records_per_second",
                value=rps,
                unit='records/sec',
                category='data_processing',
                details={
                    'operation': operation,
                    'total_records': records_processed,
                    'duration_seconds': duration
                }
            ))
        
        # Total records processed
        self.db.store_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_name=f"{operation}_records_processed",
            value=records_processed,
            unit='records',
            category='data_processing',
            details={'operation': operation}
        ))
        
        # Memory usage if provided
        if memory_used:
            self.db.store_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_name=f"{operation}_memory_usage",
                value=memory_used,
                unit='mb',
                category='data_processing',
                details={'operation': operation}
            ))
    
    def record_api_performance(self, 
                             api_name: str,
                             response_time: float,
                             status_code: int,
                             data_size: Optional[int] = None):
        """Record API call performance metrics"""
        timestamp = datetime.now()
        
        # Response time
        self.db.store_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_name=f"{api_name}_response_time",
            value=response_time,
            unit='seconds',
            category='api',
            details={
                'api': api_name,
                'status_code': status_code,
                'data_size_bytes': data_size
            }
        ))
        
        # Success/failure tracking
        success = 1 if 200 <= status_code < 300 else 0
        self.db.store_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_name=f"{api_name}_success_rate",
            value=success,
            unit='boolean',
            category='api',
            details={
                'api': api_name,
                'status_code': status_code
            }
        ))

class IndianMarketPerformanceAnalyzer:
    """Specialized performance analyzer for Indian market data"""
    
    def __init__(self, db: PerformanceDatabase):
        self.db = db
        self.logger = setup_logger('indian_market_analyzer')
        
        # Indian market specific timings
        self.market_hours = {
            'pre_market': {'start': '09:00', 'end': '09:15'},
            'normal': {'start': '09:15', 'end': '15:30'},
            'post_market': {'start': '15:40', 'end': '16:00'}
        }
        
        # Indian market segments
        self.market_segments = [
            'NSE_EQUITY', 'BSE_EQUITY', 'NSE_FO', 'NSE_CD',
            'MCX_COMMODITY', 'NCDEX_COMMODITY'
        ]
    
    def analyze_market_data_latency(self, 
                                  segment: str,
                                  data_timestamp: datetime,
                                  received_timestamp: datetime):
        """Analyze latency for Indian market data"""
        latency = (received_timestamp - data_timestamp).total_seconds()
        
        self.db.store_metric(PerformanceMetric(
            timestamp=received_timestamp,
            metric_name=f"{segment.lower()}_data_latency",
            value=latency,
            unit='seconds',
            category='market_data',
            details={
                'segment': segment,
                'market_session': self._get_market_session(data_timestamp),
                'data_timestamp': data_timestamp.isoformat(),
                'received_timestamp': received_timestamp.isoformat()
            }
        ))
    
    def _get_market_session(self, timestamp: datetime) -> str:
        """Determine market session for given timestamp"""
        time_str = timestamp.strftime('%H:%M')
        
        if self.market_hours['pre_market']['start'] <= time_str <= self.market_hours['pre_market']['end']:
            return 'pre_market'
        elif self.market_hours['normal']['start'] <= time_str <= self.market_hours['normal']['end']:
            return 'normal'
        elif self.market_hours['post_market']['start'] <= time_str <= self.market_hours['post_market']['end']:
            return 'post_market'
        else:
            return 'after_hours'
    
    def track_nse_bse_sync_performance(self, nse_data_count: int, bse_data_count: int, sync_duration: float):
        """Track NSE-BSE data synchronization performance"""
        timestamp = datetime.now()
        
        # Data count difference
        count_diff = abs(nse_data_count - bse_data_count)
        self.db.store_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_name='nse_bse_data_sync_difference',
            value=count_diff,
            unit='records',
            category='indian_market',
            details={
                'nse_count': nse_data_count,
                'bse_count': bse_data_count,
                'sync_duration': sync_duration
            }
        ))
        
        # Sync efficiency
        total_records = nse_data_count + bse_data_count
        if sync_duration > 0 and total_records > 0:
            sync_rate = total_records / sync_duration
            self.db.store_metric(PerformanceMetric(
                timestamp=timestamp,
                metric_name='nse_bse_sync_rate',
                value=sync_rate,
                unit='records/sec',
                category='indian_market'
            ))

class PerformanceReportGenerator:
    """Generate comprehensive performance reports"""
    
    def __init__(self, db: PerformanceDatabase):
        self.db = db
        self.logger = setup_logger('performance_reporter')
        self.reports_dir = Path('reports/performance')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> str:
        """Generate daily performance report"""
        if date is None:
            date = datetime.now().date()
        
        start_time = datetime.combine(date, datetime.min.time())
        end_time = datetime.combine(date, datetime.max.time())
        
        # Get all metrics for the day
        metrics = self.db.get_metrics(start_time=start_time, end_time=end_time)
        
        if not metrics:
            self.logger.warning(f"No performance metrics found for {date}")
            return ""
        
        # Organize metrics by category
        categorized_metrics = {}
        for metric in metrics:
            if metric.category not in categorized_metrics:
                categorized_metrics[metric.category] = []
            categorized_metrics[metric.category].append(metric)
        
        # Generate report
        report_content = self._create_daily_report_content(date, categorized_metrics)
        
        # Save report
        report_filename = f"daily_performance_{date.strftime('%Y%m%d')}.txt"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Daily performance report generated: {report_path}")
        return str(report_path)
    
    def _create_daily_report_content(self, date: datetime.date, categorized_metrics: Dict[str, List[PerformanceMetric]]) -> str:
        """Create formatted daily report content"""
        report_lines = [
            "=" * 80,
            f"DAILY PERFORMANCE REPORT - {date.strftime('%B %d, %Y')}",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40
        ]
        
        # System Performance Summary
        if 'system' in categorized_metrics:
            system_metrics = categorized_metrics['system']
            report_lines.extend(self._analyze_system_performance(system_metrics))
        
        # API Performance Summary
        if 'api' in categorized_metrics:
            api_metrics = categorized_metrics['api']
            report_lines.extend(self._analyze_api_performance(api_metrics))
        
        # Data Processing Summary
        if 'data_processing' in categorized_metrics:
            processing_metrics = categorized_metrics['data_processing']
            report_lines.extend(self._analyze_processing_performance(processing_metrics))
        
        # Indian Market Specific Summary
        if 'indian_market' in categorized_metrics:
            market_metrics = categorized_metrics['indian_market']
            report_lines.extend(self._analyze_indian_market_performance(market_metrics))
        
        # Market Data Summary
        if 'market_data' in categorized_metrics:
            market_data_metrics = categorized_metrics['market_data']
            report_lines.extend(self._analyze_market_data_performance(market_data_metrics))
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def _analyze_system_performance(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Analyze system performance metrics"""
        lines = [
            "",
            "SYSTEM PERFORMANCE",
            "-" * 20
        ]
        
        # Group by metric name
        metric_groups = {}
        for metric in metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric.value)
        
        # Calculate statistics for each metric
        for metric_name, values in metric_groups.items():
            if values:
                avg_val = np.mean(values)
                max_val = np.max(values)
                min_val = np.min(values)
                
                lines.append(f"  {metric_name.replace('_', ' ').title()}:")
                lines.append(f"    Average: {avg_val:.2f}")
                lines.append(f"    Maximum: {max_val:.2f}")
                lines.append(f"    Minimum: {min_val:.2f}")
                
                # Performance alerts
                if 'cpu_usage' in metric_name and max_val > 80:
                    lines.append(f"    ⚠️  HIGH CPU USAGE DETECTED!")
                elif 'memory_usage' in metric_name and max_val > 85:
                    lines.append(f"    ⚠️  HIGH MEMORY USAGE DETECTED!")
                elif 'disk_usage' in metric_name and max_val > 90:
                    lines.append(f"    ⚠️  HIGH DISK USAGE DETECTED!")
                
                lines.append("")
        
        return lines
    
    def _analyze_api_performance(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Analyze API performance metrics"""
        lines = [
            "",
            "API PERFORMANCE",
            "-" * 20
        ]
        
        # Separate response times and success rates
        response_times = {}
        success_rates = {}
        
        for metric in metrics:
            if 'response_time' in metric.metric_name:
                api_name = metric.metric_name.replace('_response_time', '')
                if api_name not in response_times:
                    response_times[api_name] = []
                response_times[api_name].append(metric.value)
            elif 'success_rate' in metric.metric_name:
                api_name = metric.metric_name.replace('_success_rate', '')
                if api_name not in success_rates:
                    success_rates[api_name] = []
                success_rates[api_name].append(metric.value)
        
        # Analyze each API
        all_apis = set(list(response_times.keys()) + list(success_rates.keys()))
        
        for api_name in sorted(all_apis):
            lines.append(f"  {api_name.upper()} API:")
            
            # Response time analysis
            if api_name in response_times:
                rt_values = response_times[api_name]
                avg_rt = np.mean(rt_values) * 1000  # Convert to milliseconds
                max_rt = np.max(rt_values) * 1000
                lines.append(f"    Avg Response Time: {avg_rt:.1f}ms")
                lines.append(f"    Max Response Time: {max_rt:.1f}ms")
                
                if avg_rt > 5000:  # > 5 seconds
                    lines.append(f"    ⚠️  SLOW API RESPONSE DETECTED!")
            
            # Success rate analysis
            if api_name in success_rates:
                sr_values = success_rates[api_name]
                success_percentage = (np.mean(sr_values) * 100)
                lines.append(f"    Success Rate: {success_percentage:.1f}%")
                
                if success_percentage < 95:
                    lines.append(f"    ⚠️  LOW SUCCESS RATE DETECTED!")
            
            lines.append("")
        
        return lines
    
    def _analyze_processing_performance(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Analyze data processing performance metrics"""
        lines = [
            "",
            "DATA PROCESSING PERFORMANCE",
            "-" * 30
        ]
        
        # Group by operation
        operations = {}
        for metric in metrics:
            if metric.details and 'operation' in metric.details:
                op_name = metric.details['operation']
                if op_name not in operations:
                    operations[op_name] = {'records_per_second': [], 'records_processed': [], 'memory_usage': []}
                
                if 'records_per_second' in metric.metric_name:
                    operations[op_name]['records_per_second'].append(metric.value)
                elif 'records_processed' in metric.metric_name:
                    operations[op_name]['records_processed'].append(metric.value)
                elif 'memory_usage' in metric.metric_name:
                    operations[op_name]['memory_usage'].append(metric.value)
        
        for op_name, op_metrics in operations.items():
            lines.append(f"  {op_name.replace('_', ' ').title()} Operation:")
            
            if op_metrics['records_per_second']:
                avg_rps = np.mean(op_metrics['records_per_second'])
                lines.append(f"    Avg Processing Rate: {avg_rps:.1f} records/sec")
            
            if op_metrics['records_processed']:
                total_records = np.sum(op_metrics['records_processed'])
                lines.append(f"    Total Records Processed: {int(total_records):,}")
            
            if op_metrics['memory_usage']:
                avg_memory = np.mean(op_metrics['memory_usage'])
                lines.append(f"    Avg Memory Usage: {avg_memory:.1f} MB")
            
            lines.append("")
        
        return lines
    
    def _analyze_indian_market_performance(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Analyze Indian market specific performance metrics"""
        lines = [
            "",
            "INDIAN MARKET PERFORMANCE",
            "-" * 30
        ]
        
        # NSE-BSE sync performance
        sync_differences = []
        sync_rates = []
        
        for metric in metrics:
            if 'nse_bse_data_sync_difference' in metric.metric_name:
                sync_differences.append(metric.value)
            elif 'nse_bse_sync_rate' in metric.metric_name:
                sync_rates.append(metric.value)
        
        if sync_differences:
            avg_diff = np.mean(sync_differences)
            max_diff = np.max(sync_differences)
            lines.append(f"  NSE-BSE Data Synchronization:")
            lines.append(f"    Avg Record Difference: {avg_diff:.1f}")
            lines.append(f"    Max Record Difference: {max_diff:.1f}")
            
            if max_diff > 100:
                lines.append(f"    ⚠️  HIGH SYNC DIFFERENCE DETECTED!")
        
        if sync_rates:
            avg_rate = np.mean(sync_rates)
            lines.append(f"    Avg Sync Rate: {avg_rate:.1f} records/sec")
        
        lines.append("")
        return lines
    
    def _analyze_market_data_performance(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Analyze market data latency performance"""
        lines = [
            "",
            "MARKET DATA LATENCY",
            "-" * 25
        ]
        
        # Group by segment
        segment_latencies = {}
        for metric in metrics:
            if 'data_latency' in metric.metric_name and metric.details:
                segment = metric.details.get('segment', 'unknown')
                if segment not in segment_latencies:
                    segment_latencies[segment] = []
                segment_latencies[segment].append(metric.value)
        
        for segment, latencies in segment_latencies.items():
            avg_latency = np.mean(latencies) * 1000  # Convert to milliseconds
            max_latency = np.max(latencies) * 1000
            
            lines.append(f"  {segment} Segment:")
            lines.append(f"    Avg Latency: {avg_latency:.1f}ms")
            lines.append(f"    Max Latency: {max_latency:.1f}ms")
            
            if avg_latency > 1000:  # > 1 second
                lines.append(f"    ⚠️  HIGH LATENCY DETECTED!")
            
            lines.append("")
        
        return lines
    
    def generate_weekly_summary(self, week_start: Optional[datetime] = None) -> str:
        """Generate weekly performance summary"""
        if week_start is None:
            # Get start of current week (Monday)
            today = datetime.now().date()
            days_since_monday = today.weekday()
            week_start = datetime.combine(today - timedelta(days=days_since_monday), datetime.min.time())
        
        week_end = week_start + timedelta(days=7)
        
        # Get metrics for the week
        metrics = self.db.get_metrics(start_time=week_start, end_time=week_end)
        
        if not metrics:
            self.logger.warning(f"No performance metrics found for week starting {week_start.date()}")
            return ""
        
        # Generate summary
        summary_lines = [
            "=" * 80,
            f"WEEKLY PERFORMANCE SUMMARY - Week of {week_start.strftime('%B %d, %Y')}",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total Metrics Collected: {len(metrics):,}",
            ""
        ]
        
        # Calculate key performance indicators
        system_metrics = [m for m in metrics if m.category == 'system']
        api_metrics = [m for m in metrics if m.category == 'api']
        
        if system_metrics:
            cpu_metrics = [m.value for m in system_metrics if 'cpu_usage' in m.metric_name]
            if cpu_metrics:
                avg_cpu = np.mean(cpu_metrics)
                summary_lines.append(f"Average CPU Usage: {avg_cpu:.1f}%")
        
        if api_metrics:
            response_times = [m.value for m in api_metrics if 'response_time' in m.metric_name]
            if response_times:
                avg_response_time = np.mean(response_times) * 1000
                summary_lines.append(f"Average API Response Time: {avg_response_time:.1f}ms")
        
        # Save summary
        summary_filename = f"weekly_summary_{week_start.strftime('%Y%m%d')}.txt"
        summary_path = self.reports_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary_lines))
        
        self.logger.info(f"Weekly performance summary generated: {summary_path}")
        return str(summary_path)

class PerformanceMonitor:
    """Main performance monitoring orchestrator"""
    
    def __init__(self, db_path: str = 'logs/system/performance.db'):
        self.db = PerformanceDatabase(db_path)
        self.system_monitor = SystemPerformanceMonitor(self.db)
        self.app_monitor = ApplicationPerformanceMonitor(self.db)
        self.indian_market_analyzer = IndianMarketPerformanceAnalyzer(self.db)
        self.report_generator = PerformanceReportGenerator(self.db)
        self.logger = setup_logger('performance_monitor')
    
    def start(self):
        """Start all performance monitoring"""
        self.system_monitor.start_monitoring()
        self.logger.info("Performance monitoring system started")
    
    def stop(self):
        """Stop all performance monitoring"""
        self.system_monitor.stop_monitoring()
        self.logger.info("Performance monitoring system stopped")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current performance status"""
        # Get recent metrics (last 5 minutes)
        recent_time = datetime.now() - timedelta(minutes=5)
        recent_metrics = self.db.get_metrics(start_time=recent_time, limit=100)
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.system_monitor.monitoring,
            'recent_metrics_count': len(recent_metrics),
            'system_status': {},
            'alerts': []
        }
        
        # Analyze recent system metrics
        system_metrics = [m for m in recent_metrics if m.category == 'system']
        if system_metrics:
            for metric in system_metrics[-10:]:  # Last 10 system metrics
                if metric.metric_name not in status['system_status']:
                    status['system_status'][metric.metric_name] = metric.value
                
                # Check for alerts
                if 'cpu_usage' in metric.metric_name and metric.value > 80:
                    status['alerts'].append(f"High CPU usage: {metric.value:.1f}%")
                elif 'memory_usage' in metric.metric_name and metric.value > 85:
                    status['alerts'].append(f"High memory usage: {metric.value:.1f}%")
        
        return status

# Context manager for performance timing
class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: ApplicationPerformanceMonitor, operation_name: str, details: Optional[Dict[str, Any]] = None):
        self.monitor = monitor
        self.operation_name = operation_name
        self.details = details
        self.operation_id = None
    
    def __enter__(self):
        self.operation_id = self.monitor.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.operation_id:
            self.monitor.end_operation(self.operation_id, self.details)

# Decorator for automatic performance monitoring
def monitor_performance(monitor: ApplicationPerformanceMonitor, operation_name: Optional[str] = None):
    """Decorator to automatically monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with PerformanceTimer(monitor, op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage and testing
    monitor = PerformanceMonitor()
    
    try:
        # Start monitoring
        monitor.start()
        
        # Simulate some operations
        print("Testing performance monitoring...")
        
        # Test application performance monitoring
        op_id = monitor.app_monitor.start_operation("test_data_fetch")
        time.sleep(2)  # Simulate work
        monitor.app_monitor.end_operation(op_id, {'records': 1000})
        
        # Test API performance recording
        monitor.app_monitor.record_api_performance(
            "nse_equity_api", 
            response_time=1.5, 
            status_code=200, 
            data_size=1024000
        )
        
        # Test Indian market performance
        monitor.indian_market_analyzer.analyze_market_data_latency(
            "NSE_EQUITY",
            datetime.now() - timedelta(seconds=0.5),
            datetime.now()
        )
        
        # Wait a bit to collect some system metrics
        print("Collecting system metrics for 10 seconds...")
        time.sleep(10)
        
        # Generate a daily report
        print("Generating daily performance report...")
        report_path = monitor.report_generator.generate_daily_report()
        print(f"Report saved to: {report_path}")
        
        # Show current status
        status = monitor.get_current_status()
        print(f"\nCurrent Status:")
        print(f"Monitoring Active: {status['monitoring_active']}")
        print(f"Recent Metrics: {status['recent_metrics_count']}")
        if status['alerts']:
            print(f"Alerts: {', '.join(status['alerts'])}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        monitor.stop()
        print("Performance monitoring stopped.")