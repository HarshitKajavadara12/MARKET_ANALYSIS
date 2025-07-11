#!/usr/bin/env python3
"""
Market Research System v1.0 (2022)
Data Validation Script

Validates the integrity and quality of collected market data.
Checks for missing data, anomalies, and data consistency.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from typing import Dict, List, Tuple, Any
import json
import yaml

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.settings import DATA_DIR, STOCK_UNIVERSE_FILE
from utils.logging_utils import setup_logging
from utils.date_utils import get_trading_days, is_trading_day
from utils.file_utils import ensure_directory_exists

# Setup logging
logger = setup_logging('validate_data')

class DataValidator:
    """Validates market data quality and integrity"""
    
    def __init__(self):
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'stocks_validated': 0,
            'total_issues': 0,
            'critical_issues': 0,
            'warnings': 0,
            'passed_validation': [],
            'failed_validation': [],
            'issues': []
        }
        
        self.thresholds = {
            'max_daily_return': 0.20,  # 20% max daily return
            'min_volume': 1000,        # Minimum daily volume
            'max_price_gap': 0.15,     # 15% max price gap between days
            'min_trading_days': 200,   # Minimum trading days in a year
            'max_zero_volume_days': 5  # Max consecutive zero volume days
        }
    
    def load_stock_universe(self) -> List[str]:
        """Load stock symbols from universe file"""
        try:
            if os.path.exists(STOCK_UNIVERSE_FILE):
                with open(STOCK_UNIVERSE_FILE, 'r') as f:
                    universe = yaml.safe_load(f)
                return universe.get('stocks', {}).get('valid', [])
            else:
                logger.warning("Stock universe file not found, using default list")
                return [
                    'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'RELIANCE.NS',
                    'KOTAKBANK.NS', 'LT.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS'
                ]
        except Exception as e:
            logger.error(f"Error loading stock universe: {str(e)}")
            return []
    
    def validate_stock_data(self, symbol: str, period: str = '1y') -> Dict[str, Any]:
        """Validate data for a single stock"""
        logger.info(f"Validating data for {symbol}")
        
        issues = []
        warnings = []
        
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                issue = {
                    'symbol': symbol,
                    'type': 'CRITICAL',
                    'category': 'NO_DATA',
                    'message': 'No historical data available',
                    'timestamp': datetime.now().isoformat()
                }
                issues.append(issue)
                return {'symbol': symbol, 'status': 'FAILED', 'issues': issues, 'warnings': warnings}
            
            # Basic data structure validation
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                issue = {
                    'symbol': symbol,
                    'type': 'CRITICAL',
                    'category': 'MISSING_COLUMNS',
                    'message': f'Missing required columns: {missing_columns}',
                    'timestamp': datetime.now().isoformat()
                }
                issues.append(issue)
            
            # Check for missing data
            missing_data_issues = self._check_missing_data(symbol, df)
            issues.extend(missing_data_issues)
            
            # Check for anomalous prices
            price_anomalies = self._check_price_anomalies(symbol, df)
            issues.extend(price_anomalies)
            
            # Check volume anomalies
            volume_anomalies = self._check_volume_anomalies(symbol, df)
            warnings.extend(volume_anomalies)
            
            # Check data consistency
            consistency_issues = self._check_data_consistency(symbol, df)
            issues.extend(consistency_issues)
            
            # Check for sufficient trading days
            trading_days_issues = self._check_trading_days(symbol, df)
            warnings.extend(trading_days_issues)
            
            # Determine overall status
            status = 'PASSED' if not issues else 'FAILED'
            
            return {
                'symbol': symbol,
                'status': status,
                'data_points': len(df),
                'date_range': {
                    'start': df.index.min().strftime('%Y-%m-%d'),
                    'end': df.index.max().strftime('%Y-%m-%d')
                },
                'issues': issues,
                'warnings': warnings,
                'summary': {
                    'critical_issues': len([i for i in issues if i['type'] == 'CRITICAL']),
                    'minor_issues': len([i for i in issues if i['type'] == 'MINOR']),
                    'warnings': len(warnings)
                }
            }
            
        except Exception as e:
            issue = {
                'symbol': symbol,
                'type': 'CRITICAL',
                'category': 'VALIDATION_ERROR',
                'message': f'Error during validation: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
            return {'symbol': symbol, 'status': 'FAILED', 'issues': [issue], 'warnings': []}
    
    def _check_missing_data(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """Check for missing data points"""
        issues = []
        
        # Check for NaN values
        for column in ['Open', 'High', 'Low', 'Close']:
            nan_count = df[column].isna().sum()
            if nan_count > 0:
                issue = {
                    'symbol': symbol,
                    'type': 'CRITICAL',
                    'category': 'MISSING_DATA',
                    'message': f'{column} has {nan_count} missing values',
                    'timestamp': datetime.now().isoformat(),
                    'affected_dates': df[df[column].isna()].index.strftime('%Y-%m-%d').tolist()
                }
                issues.append(issue)
        
        # Check for missing trading days (gaps in data)
        expected_trading_days = get_trading_days(df.index.min(), df.index.max())
        actual_trading_days = set(df.index.date)
        missing_days = [day for day in expected_trading_days if day not in actual_trading_days]
        
        if len(missing_days) > 5:  # Allow some missing days
            issue = {
                'symbol': symbol,
                'type': 'MINOR',
                'category': 'MISSING_TRADING_DAYS',
                'message': f'{len(missing_days)} trading days missing',
                'timestamp': datetime.now().isoformat(),
                'missing_days_count': len(missing_days)
            }
            issues.append(issue)
        
        return issues
    
    def _check_price_anomalies(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """Check for price anomalies and outliers"""
        issues = []
        
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Check for extreme daily returns
        extreme_returns = df[abs(df['Daily_Return']) > self.thresholds['max_daily_return']]
        
        if not extreme_returns.empty:
            issue = {
                'symbol': symbol,
                'type': 'CRITICAL',
                'category': 'EXTREME_RETURNS',
                'message': f'{len(extreme_returns)} days with returns > {self.thresholds["max_daily_return"]*100}%',
                'timestamp': datetime.now().isoformat(),
                'extreme_dates': extreme_returns.index.strftime('%Y-%m-%d').tolist(),
                'extreme_values': extreme_returns['Daily_Return'].tolist()
            }
            issues.append(issue)
        
        # Check for price consistency (High >= Low, Close between High and Low)
        inconsistent_prices = df[
            (df['High'] < df['Low']) | 
            (df['Close'] > df['High']) | 
            (df['Close'] < df['Low']) |
            (df['Open'] > df['High']) |
            (df['Open'] < df['Low'])
        ]
        
        if not inconsistent_prices.empty:
            issue = {
                'symbol': symbol,
                'type': 'CRITICAL',
                'category': 'PRICE_INCONSISTENCY',
                'message': f'{len(inconsistent_prices)} days with inconsistent OHLC prices',
                'timestamp': datetime.now().isoformat(),
                'inconsistent_dates': inconsistent_prices.index.strftime('%Y-%m-%d').tolist()
            }
            issues.append(issue)
        
        # Check for zero or negative prices
        zero_negative_prices = df[(df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)]
        
        if not zero_negative_prices.empty:
            issue = {
                'symbol': symbol,
                'type': 'CRITICAL',
                'category': 'INVALID_PRICES',
                'message': f'{len(zero_negative_prices)} days with zero or negative prices',
                'timestamp': datetime.now().isoformat(),
                'invalid_dates': zero_negative_prices.index.strftime('%Y-%m-%d').tolist()
            }
            issues.append(issue)
        
        return issues
    
    def _check_volume_anomalies(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """Check for volume-related issues"""
        warnings = []
        
        # Check for zero volume days
        zero_volume_days = df[df['Volume'] == 0]
        
        if len(zero_volume_days) > self.thresholds['max_zero_volume_days']:
            warning = {
                'symbol': symbol,
                'type': 'WARNING',
                'category': 'ZERO_VOLUME',
                'message': f'{len(zero_volume_days)} days with zero trading volume',
                'timestamp': datetime.now().isoformat(),
                'zero_volume_dates': zero_volume_days.index.strftime('%Y-%m-%d').tolist()
            }
            warnings.append(warning)
        
        # Check for unusually low average volume
        avg_volume = df['Volume'].mean()
        if avg_volume < self.thresholds['min_volume']:
            warning = {
                'symbol': symbol,
                'type': 'WARNING',
                'category': 'LOW_VOLUME',
                'message': f'Average daily volume ({avg_volume:.0f}) below threshold ({self.thresholds["min_volume"]})',
                'timestamp': datetime.now().isoformat(),
                'average_volume': avg_volume
            }
            warnings.append(warning)
        
        return warnings
    
    def _check_data_consistency(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """Check for data consistency issues"""
        issues = []
        
        # Check for duplicate dates
        duplicate_dates = df.index.duplicated()
        if duplicate_dates.any():
            issue = {
                'symbol': symbol,
                'type': 'CRITICAL',
                'category': 'DUPLICATE_DATES',
                'message': f'{duplicate_dates.sum()} duplicate date entries found',
                'timestamp': datetime.now().isoformat()
            }
            issues.append(issue)
        
        # Check for data sorted by date
        if not df.index.is_monotonic_increasing:
            issue = {
                'symbol': symbol,
                'type': 'MINOR',
                'category': 'UNSORTED_DATA',
                'message': 'Data is not sorted chronologically',
                'timestamp': datetime.now().isoformat()
            }
            issues.append(issue)
        
        return issues
    
    def _check_trading_days(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """Check if sufficient trading days are present"""
        warnings = []
        
        # Calculate expected vs actual trading days in the last year
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        
        recent_data = df[df.index.date >= start_date]
        
        if len(recent_data) < self.thresholds['min_trading_days']:
            warning = {
                'symbol': symbol,
                'type': 'WARNING',
                'category': 'INSUFFICIENT_DATA',
                'message': f'Only {len(recent_data)} trading days in last year (expected ~{self.thresholds["min_trading_days"]})',
                'timestamp': datetime.now().isoformat(),
                'actual_days': len(recent_data)
            }
            warnings.append(warning)
        
        return warnings
    
    def validate_all_stocks(self) -> Dict[str, Any]:
        """Validate all stocks in the universe"""
        logger.info("Starting comprehensive data validation...")
        
        stock_symbols = self.load_stock_universe()
        
        if not stock_symbols:
            logger.error("No stock symbols to validate")
            return self.validation_results
        
        all_results = []
        
        for symbol in stock_symbols:
            try:
                result = self.validate_stock_data(symbol)
                all_results.append(result)
                
                # Update summary statistics
                self.validation_results['stocks_validated'] += 1
                
                if result['status'] == 'PASSED':
                    self.validation_results['passed_validation'].append(symbol)
                else:
                    self.validation_results['failed_validation'].append(symbol)
                
                # Count issues
                critical_issues = len([i for i in result.get('issues', []) if i['type'] == 'CRITICAL'])
                minor_issues = len([i for i in result.get('issues', []) if i['type'] == 'MINOR'])
                warnings = len(result.get('warnings', []))
                
                self.validation_results['critical_issues'] += critical_issues
                self.validation_results['total_issues'] += critical_issues + minor_issues
                self.validation_results['warnings'] += warnings
                
                # Add issues to global list
                self.validation_results['issues'].extend(result.get('issues', []))
                self.validation_results['issues'].extend(result.get('warnings', []))
                
                logger.info(f"Validated {symbol}: {result['status']} - "
                           f"Critical: {critical_issues}, Minor: {minor_issues}, Warnings: {warnings}")
                
            except Exception as e:
                logger.error(f"Failed to validate {symbol}: {str(e)}")
                self.validation_results['failed_validation'].append(symbol)
                
                error_issue = {
                    'symbol': symbol,
                    'type': 'CRITICAL',
                    'category': 'VALIDATION_FAILURE',
                    'message': f'Validation process failed: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
                self.validation_results['issues'].append(error_issue)
                self.validation_results['critical_issues'] += 1
                self.validation_results['total_issues'] += 1
        
        # Generate validation summary
        self.validation_results['success_rate'] = (
            len(self.validation_results['passed_validation']) / 
            self.validation_results['stocks_validated'] * 100
            if self.validation_results['stocks_validated'] > 0 else 0
        )
        
        logger.info(f"Validation completed: {self.validation_results['stocks_validated']} stocks validated, "
                   f"{len(self.validation_results['passed_validation'])} passed, "
                   f"{len(self.validation_results['failed_validation'])} failed")
        
        return self.validation_results
    
    def save_validation_report(self, output_dir: str = None) -> str:
        """Save validation results to JSON report"""
        if output_dir is None:
            output_dir = os.path.join(DATA_DIR, 'validation_reports')
        
        ensure_directory_exists(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f'validation_report_{timestamp}.json')
        
        try:
            with open(report_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {str(e)}")
            return None
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary of validation results"""
        summary = []
        summary.append("="*60)
        summary.append("MARKET DATA VALIDATION REPORT")
        summary.append("="*60)
        summary.append(f"Generated: {self.validation_results['timestamp']}")
        summary.append(f"Stocks Validated: {self.validation_results['stocks_validated']}")
        summary.append(f"Success Rate: {self.validation_results['success_rate']:.1f}%")
        summary.append("")
        
        summary.append("SUMMARY STATISTICS:")
        summary.append(f"  ✓ Passed Validation: {len(self.validation_results['passed_validation'])}")
        summary.append(f"  ✗ Failed Validation: {len(self.validation_results['failed_validation'])}")
        summary.append(f"  🔴 Critical Issues: {self.validation_results['critical_issues']}")
        summary.append(f"  🟡 Total Issues: {self.validation_results['total_issues']}")
        summary.append(f"  ⚠️  Warnings: {self.validation_results['warnings']}")
        summary.append("")
        
        if self.validation_results['passed_validation']:
            summary.append("STOCKS THAT PASSED VALIDATION:")
            for symbol in self.validation_results['passed_validation']:
                summary.append(f"  ✓ {symbol}")
            summary.append("")
        
        if self.validation_results['failed_validation']:
            summary.append("STOCKS THAT FAILED VALIDATION:")
            for symbol in self.validation_results['failed_validation']:
                summary.append(f"  ✗ {symbol}")
            summary.append("")
        
        # Group issues by category
        issue_categories = {}
        for issue in self.validation_results['issues']:
            category = issue.get('category', 'UNKNOWN')
            if category not in issue_categories:
                issue_categories[category] = []
            issue_categories[category].append(issue)
        
        if issue_categories:
            summary.append("ISSUES BY CATEGORY:")
            for category, issues in issue_categories.items():
                summary.append(f"  {category}: {len(issues)} issues")
                critical_count = len([i for i in issues if i['type'] == 'CRITICAL'])
                warning_count = len([i for i in issues if i['type'] == 'WARNING'])
                if critical_count > 0:
                    summary.append(f"    🔴 Critical: {critical_count}")
                if warning_count > 0:
                    summary.append(f"    ⚠️  Warnings: {warning_count}")
            summary.append("")
        
        summary.append("="*60)
        summary.append("END OF REPORT")
        summary.append("="*60)
        
        return "\n".join(summary)
    
    def check_data_freshness(self, symbol: str, max_age_days: int = 2) -> Dict[str, Any]:
        """Check if stock data is fresh (recent)"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period='5d')  # Get last 5 days
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'is_fresh': False,
                    'reason': 'No data available',
                    'last_update': None
                }
            
            last_date = df.index.max().date()
            today = datetime.now().date()
            days_old = (today - last_date).days
            
            # Account for weekends and holidays
            is_fresh = days_old <= max_age_days or not is_trading_day(today)
            
            return {
                'symbol': symbol,
                'is_fresh': is_fresh,
                'last_update': last_date.isoformat(),
                'days_old': days_old,
                'reason': 'Data is current' if is_fresh else f'Data is {days_old} days old'
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'is_fresh': False,
                'reason': f'Error checking freshness: {str(e)}',
                'last_update': None
            }


def main():
    """Main function to run data validation"""
    logger.info("Starting Market Data Validation System v1.0")
    
    # Initialize validator
    validator = DataValidator()
    
    try:
        # Run validation on all stocks
        results = validator.validate_all_stocks()
        
        # Save detailed JSON report
        report_file = validator.save_validation_report()
        
        # Generate and display summary
        summary = validator.generate_summary_report()
        print(summary)
        
        # Save summary to file
        if report_file:
            summary_file = report_file.replace('.json', '_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(summary)
            logger.info(f"Summary report saved to: {summary_file}")
        
        # Check data freshness for critical stocks
        logger.info("Checking data freshness for key stocks...")
        critical_stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS']
        
        for symbol in critical_stocks:
            freshness = validator.check_data_freshness(symbol)
            status = "✓" if freshness['is_fresh'] else "✗"
            logger.info(f"{status} {symbol}: {freshness['reason']}")
        
        # Exit with appropriate code
        if results['critical_issues'] > 0:
            logger.warning("Validation completed with critical issues")
            sys.exit(1)
        else:
            logger.info("Validation completed successfully")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Validation process failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()