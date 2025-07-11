#!/usr/bin/env python3
"""
Update Dependencies Script for Market Research System v1.0
Author: Independent Market Researcher
Date: January 2022
Purpose: Update Python packages, check for security vulnerabilities, and maintain system dependencies
"""

import os
import sys
import subprocess
import json
import logging
from datetime import datetime
import pkg_resources
import requests
from packaging import version

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system/dependency_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DependencyManager:
    """Manage system dependencies and updates"""
    
    def __init__(self):
        self.update_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'UNKNOWN',
            'updates_available': [],
            'security_issues': [],
            'successful_updates': [],
            'failed_updates': [],
            'recommendations': []
        }
        
        # Core dependencies for market research system
        self.core_packages = [
            'yfinance',
            'pandas',
            'numpy',
            'matplotlib',
            'ta-lib',
            'requests',
            'psutil',
            'schedule',
            'reportlab',
            'openpyxl',
            'scikit-learn'
        ]
        
        # Security-critical packages
        self.security_critical = [
            'requests',
            'urllib3',
            'cryptography',
            'pyjwt',
            'pillow'
        ]
    
    def check_current_versions(self):
        """Check current versions of installed packages"""
        logger.info("Checking current package versions...")
        
        installed_packages = {}
        
        try:
            # Get all installed packages
            installed = [d for d in pkg_resources.working_set]
            
            for package in installed:
                installed_packages[package.project_name.lower()] = package.version
            
            logger.info(f"Found {len(installed_packages)} installed packages")
            return installed_packages
            
        except Exception as e:
            logger.error(f"Failed to check current versions: {e}")
            return {}
    
    def check_available_updates(self):
        """Check for available package updates using PyPI API"""
        logger.info("Checking for available updates...")
        
        installed_packages = self.check_current_versions()
        updates_available = []
        
        for package_name in self.core_packages:
            try:
                current_version = installed_packages.get(package_name.lower())
                if not current_version:
                    logger.warning(f"Package {package_name} not installed")
                    continue
                
                # Check PyPI for latest version
                latest_version = self._get_latest_version_from_pypi(package_name)
                
                if latest_version and version.parse(latest_version) > version.parse(current_version):
                    update_info = {
                        'package': package_name,
                        'current_version': current_version,
                        'latest_version': latest_version,
                        'security_critical': package_name.lower() in [p.lower() for p in self.security_critical]
                    }
                    updates_available.append(update_info)
                    logger.info(f"Update available: {package_name} {current_version} -> {latest_version}")
                
            except Exception as e:
                logger.error(f"Failed to check updates for {package_name}: {e}")
        
        self.update_report['updates_available'] = updates_available
        return updates_available
    
    def _get_latest_version_from_pypi(self, package_name):
        """Get latest version from PyPI API"""
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['info']['version']
        except Exception as e:
            logger.error(f"Failed to get latest version for {package_name}: {e}")
        return None
    
    def check_security_vulnerabilities(self):
        """Check for known security vulnerabilities using pip-audit or safety"""
        logger.info("Checking for security vulnerabilities...")
        
        security_issues = []
        
        try:
            # Try to use pip-audit if available
            result = subprocess.run(['pip-audit', '--format=json'], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parse pip-audit output
                try:
                    audit_data = json.loads(result.stdout)
                    for vulnerability in audit_data.get('vulnerabilities', []):
                        security_issues.append({
                            'package': vulnerability.get('package'),
                            'version': vulnerability.get('version'),
                            'vulnerability_id': vulnerability.get('id'),
                            'description': vulnerability.get('description', ''),
                            'severity': vulnerability.get('severity', 'UNKNOWN')
                        })
                except json.JSONDecodeError:
                    logger.warning("Could not parse pip-audit output")
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("pip-audit not available or timed out, trying alternative method")
            
            # Alternative: Check critical packages manually
            for package in self.security_critical:
                try:
                    # Simple check for very old versions that might have known issues
                    installed_packages = self.check_current_versions()
                    current_version = installed_packages.get(package.lower())
                    
                    if current_version:
                        # Add basic version checks for known vulnerable versions
                        vulnerable_versions = self._get_known_vulnerable_versions(package)
                        if current_version in vulnerable_versions:
                            security_issues.append({
                                'package': package,
                                'version': current_version,
                                'vulnerability_id': 'KNOWN_VULNERABLE',
                                'description': f'Known vulnerable version of {package}',
                                'severity': 'HIGH'
                            })
                except Exception as e:
                    logger.error(f"Failed to check security for {package}: {e}")
        
        self.update_report['security_issues'] = security_issues
        
        if security_issues:
            logger.warning(f"Found {len(security_issues)} potential security issues")
        else:
            logger.info("No security vulnerabilities detected")
        
        return security_issues
    
    def _get_known_vulnerable_versions(self, package):
        """Get known vulnerable versions for critical packages"""
        vulnerable_versions = {
            'requests': ['2.27.0', '2.26.0'],  # Example vulnerable versions
            'urllib3': ['1.26.7', '1.26.6'],
            'pillow': ['8.3.1', '8.3.0']
        }
        return vulnerable_versions.get(package.lower(), [])
    
    def backup_requirements(self):
        """Backup current requirements before updating"""
        logger.info("Backing up current requirements...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f'backups/requirements_backup_{timestamp}.txt'
            
            os.makedirs('backups', exist_ok=True)
            
            # Generate current requirements
            result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
            if result.returncode == 0:
                with open(backup_file, 'w') as f:
                    f.write(result.stdout)
                logger.info(f"Requirements backed up to: {backup_file}")
                return backup_file
            else:
                logger.error("Failed to generate requirements backup")
                return None
                
        except Exception as e:
            logger.error(f"Failed to backup requirements: {e}")
            return None
    
    def update_packages(self, packages_to_update=None, dry_run=False):
        """Update specified packages or all available updates"""
        logger.info(f"Starting package updates (dry_run={dry_run})...")
        
        if packages_to_update is None:
            packages_to_update = [update['package'] for update in self.update_report['updates_available']]
        
        if not packages_to_update:
            logger.info("No packages to update")
            return
        
        # Backup requirements before updating
        if not dry_run:
            self.backup_requirements()
        
        successful_updates = []
        failed_updates = []
        
        for package in packages_to_update:
            try:
                logger.info(f"Updating {package}...")
                
                if dry_run:
                    logger.info(f"DRY RUN: Would update {package}")
                    successful_updates.append({
                        'package': package,
                        'status': 'DRY_RUN_SUCCESS'
                    })
                    continue
                
                # Update package
                result = subprocess.run(['pip', 'install', '--upgrade', package], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    successful_updates.append({
                        'package': package,
                        'status': 'SUCCESS',
                        'output': result.stdout
                    })
                    logger.info(f"Successfully updated {package}")
                else:
                    failed_updates.append({
                        'package': package,
                        'status': 'FAILED',
                        'error': result.stderr
                    })
                    logger.error(f"Failed to update {package}: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                failed_updates.append({
                    'package': package,
                    'status': 'TIMEOUT',
                    'error': 'Update process timed out'
                })
                logger.error(f"Update timeout for {package}")
                
            except Exception as e:
                failed_updates.append({
                    'package': package,
                    'status': 'ERROR',
                    'error': str(e)
                })
                logger.error(f"Update error for {package}: {e}")
        
        self.update_report['successful_updates'] = successful_updates
        self.update_report['failed_updates'] = failed_updates
        
        logger.info(f"Update completed: {len(successful_updates)} successful, {len(failed_updates)} failed")
    
    def update_pip_and_setuptools(self):
        """Ensure pip and setuptools are up to date"""
        logger.info("Updating pip and setuptools...")
        
        try:
            result = subprocess.run(['pip', 'install', '--upgrade', 'pip', 'setuptools'], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("Successfully updated pip and setuptools")
                return True
            else:
                logger.error(f"Failed to update pip/setuptools: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating pip/setuptools: {e}")
            return False
    
    def verify_system_integrity(self):
        """Verify system integrity after updates"""
        logger.info("Verifying system integrity...")
        
        integrity_checks = []
        
        # Test import of core packages
        for package in self.core_packages:
            try:
                if package == 'ta-lib':
                    import talib
                    integrity_checks.append({'package': package, 'status': 'OK'})
                elif package == 'yfinance':
                    import yfinance as yf
                    # Quick test
                    ticker = yf.Ticker("RELIANCE.NS")
                    integrity_checks.append({'package': package, 'status': 'OK'})
                elif package == 'pandas':
                    import pandas as pd
                    integrity_checks.append({'package': package, 'status': 'OK'})
                elif package == 'numpy':
                    import numpy as np
                    integrity_checks.append({'package': package, 'status': 'OK'})
                elif package == 'matplotlib':
                    import matplotlib.pyplot as plt
                    integrity_checks.append({'package': package, 'status': 'OK'})
                else:
                    # Generic import test
                    __import__(package.replace('-', '_'))
                    integrity_checks.append({'package': package, 'status': 'OK'})
                    
            except ImportError as e:
                integrity_checks.append({
                    'package': package, 
                    'status': 'IMPORT_ERROR',
                    'error': str(e)
                })
                logger.error(f"Import error for {package}: {e}")
            except Exception as e:
                integrity_checks.append({
                    'package': package, 
                    'status': 'TEST_ERROR',
                    'error': str(e)
                })
                logger.error(f"Test error for {package}: {e}")
        
        return integrity_checks
    
    def generate_recommendations(self):
        """Generate recommendations based on update results"""
        recommendations = []
        
        # Security recommendations
        if self.update_report['security_issues']:
            recommendations.append("URGENT: Security vulnerabilities detected - update immediately")
            for issue in self.update_report['security_issues']:
                recommendations.append(f"Update {issue['package']} to fix {issue['vulnerability_id']}")
        
        # Update recommendations
        critical_updates = [u for u in self.update_report['updates_available'] 
                          if u.get('security_critical', False)]
        if critical_updates:
            recommendations.append("Critical security updates available - prioritize these updates")
        
        # Failure recommendations
        if self.update_report['failed_updates']:
            recommendations.append("Some updates failed - check logs and retry individually")
            recommendations.append("Consider updating pip and setuptools first")
        
        # Version compatibility for Indian market data sources
        if any(pkg['package'] == 'yfinance' for pkg in self.update_report['updates_available']):
            recommendations.append("yfinance update available - improves NSE/BSE data reliability")
        
        # TA-Lib specific recommendations for Indian market analysis
        ta_lib_issues = [issue for issue in self.update_report['security_issues'] 
                        if issue['package'] == 'ta-lib']
        if ta_lib_issues:
            recommendations.append("TA-Lib has issues - critical for Indian technical analysis")
        
        self.update_report['recommendations'] = recommendations
        return recommendations
    
    def run_full_update_cycle(self, dry_run=False):
        """Run complete update cycle with all checks"""
        logger.info("Starting full dependency update cycle...")
        
        try:
            # Step 1: Update pip and setuptools first
            self.update_pip_and_setuptools()
            
            # Step 2: Check for available updates
            updates = self.check_available_updates()
            
            # Step 3: Check security vulnerabilities
            security_issues = self.check_security_vulnerabilities()
            
            # Step 4: Prioritize security-critical updates
            priority_updates = []
            regular_updates = []
            
            for update in updates:
                if update.get('security_critical', False):
                    priority_updates.append(update['package'])
                else:
                    regular_updates.append(update['package'])
            
            # Step 5: Update priority packages first
            if priority_updates:
                logger.info(f"Updating security-critical packages: {priority_updates}")
                self.update_packages(priority_updates, dry_run=dry_run)
            
            # Step 6: Update regular packages
            if regular_updates:
                logger.info(f"Updating regular packages: {regular_updates}")
                self.update_packages(regular_updates, dry_run=dry_run)
            
            # Step 7: Verify system integrity
            integrity_results = self.verify_system_integrity()
            
            # Step 8: Generate recommendations
            recommendations = self.generate_recommendations()
            
            # Step 9: Set overall status
            if self.update_report['security_issues']:
                self.update_report['status'] = 'SECURITY_ISSUES_FOUND'
            elif self.update_report['failed_updates']:
                self.update_report['status'] = 'PARTIAL_SUCCESS'
            elif self.update_report['successful_updates']:
                self.update_report['status'] = 'SUCCESS'
            else:
                self.update_report['status'] = 'NO_UPDATES_NEEDED'
            
            # Step 10: Save detailed report
            self.save_update_report()
            
            # Step 11: Test Indian market data connectivity
            self.test_indian_market_connectivity()
            
            logger.info("Full update cycle completed")
            return self.update_report
            
        except Exception as e:
            logger.error(f"Update cycle failed: {e}")
            self.update_report['status'] = 'FAILED'
            self.update_report['error'] = str(e)
            return self.update_report
    
    def test_indian_market_connectivity(self):
        """Test connectivity to Indian market data sources"""
        logger.info("Testing Indian market data connectivity...")
        
        connectivity_tests = []
        
        try:
            # Test yfinance with Indian stocks
            import yfinance as yf
            
            # Test major Indian indices and stocks
            test_symbols = [
                '^NSEI',      # Nifty 50
                '^BSESN',     # BSE Sensex
                'RELIANCE.NS', # Reliance Industries
                'TCS.NS',     # Tata Consultancy Services
                'INFY.NS'     # Infosys
            ]
            
            for symbol in test_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='5d')
                    
                    if not data.empty:
                        connectivity_tests.append({
                            'symbol': symbol,
                            'status': 'SUCCESS',
                            'last_price': data['Close'].iloc[-1] if len(data) > 0 else None
                        })
                        logger.info(f"✓ {symbol} data accessible")
                    else:
                        connectivity_tests.append({
                            'symbol': symbol,
                            'status': 'NO_DATA',
                            'error': 'Empty dataset returned'
                        })
                        logger.warning(f"✗ {symbol} - no data returned")
                        
                except Exception as e:
                    connectivity_tests.append({
                        'symbol': symbol,
                        'status': 'ERROR',
                        'error': str(e)
                    })
                    logger.error(f"✗ {symbol} - error: {e}")
            
        except ImportError:
            connectivity_tests.append({
                'symbol': 'yfinance',
                'status': 'MODULE_NOT_AVAILABLE',
                'error': 'yfinance not installed'
            })
            logger.error("yfinance module not available")
        
        self.update_report['indian_market_connectivity'] = connectivity_tests
        
        # Log summary
        successful_tests = len([t for t in connectivity_tests if t['status'] == 'SUCCESS'])
        total_tests = len(connectivity_tests)
        logger.info(f"Indian market connectivity: {successful_tests}/{total_tests} successful")
        
        return connectivity_tests
    
    def save_update_report(self):
        """Save detailed update report to JSON file"""
        try:
            os.makedirs('logs/system', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f'logs/system/dependency_update_report_{timestamp}.json'
            
            with open(report_file, 'w') as f:
                json.dump(self.update_report, f, indent=2, default=str)
            
            logger.info(f"Update report saved to: {report_file}")
            
            # Also save a human-readable summary
            summary_file = f'logs/system/dependency_update_summary_{timestamp}.txt'
            with open(summary_file, 'w') as f:
                f.write("=== Dependency Update Report ===\n")
                f.write(f"Timestamp: {self.update_report['timestamp']}\n")
                f.write(f"Status: {self.update_report['status']}\n\n")
                
                f.write(f"Updates Available: {len(self.update_report['updates_available'])}\n")
                for update in self.update_report['updates_available']:
                    f.write(f"  - {update['package']}: {update['current_version']} -> {update['latest_version']}\n")
                
                f.write(f"\nSecurity Issues: {len(self.update_report['security_issues'])}\n")
                for issue in self.update_report['security_issues']:
                    f.write(f"  - {issue['package']} v{issue['version']}: {issue['vulnerability_id']}\n")
                
                f.write(f"\nSuccessful Updates: {len(self.update_report['successful_updates'])}\n")
                f.write(f"Failed Updates: {len(self.update_report['failed_updates'])}\n")
                
                f.write("\nRecommendations:\n")
                for rec in self.update_report['recommendations']:
                    f.write(f"  - {rec}\n")
            
            logger.info(f"Update summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save update report: {e}")


def main():
    """Main function to run dependency updates"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update Market Research System Dependencies')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be updated without making changes')
    parser.add_argument('--security-only', action='store_true',
                       help='Only update security-critical packages')
    parser.add_argument('--package', type=str, nargs='+',
                       help='Update specific packages only')
    
    args = parser.parse_args()
    
    # Initialize dependency manager
    dep_manager = DependencyManager()
    
    try:
        if args.package:
            # Update specific packages
            logger.info(f"Updating specific packages: {args.package}")
            dep_manager.check_available_updates()
            dep_manager.update_packages(args.package, dry_run=args.dry_run)
            
        elif args.security_only:
            # Security updates only
            logger.info("Running security-only updates")
            dep_manager.check_available_updates()
            security_issues = dep_manager.check_security_vulnerabilities()
            
            if security_issues:
                security_packages = list(set([issue['package'] for issue in security_issues]))
                dep_manager.update_packages(security_packages, dry_run=args.dry_run)
            else:
                logger.info("No security issues found")
                
        else:
            # Full update cycle
            logger.info("Running full update cycle")
            report = dep_manager.run_full_update_cycle(dry_run=args.dry_run)
            
            # Print summary
            print("\n" + "="*50)
            print("DEPENDENCY UPDATE SUMMARY")
            print("="*50)
            print(f"Status: {report['status']}")
            print(f"Updates Available: {len(report['updates_available'])}")
            print(f"Security Issues: {len(report['security_issues'])}")
            print(f"Successful Updates: {len(report['successful_updates'])}")
            print(f"Failed Updates: {len(report['failed_updates'])}")
            
            if report['recommendations']:
                print("\nRecommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")
            
            # Indian market connectivity summary
            if 'indian_market_connectivity' in report:
                connectivity_tests = report['indian_market_connectivity']
                successful = len([t for t in connectivity_tests if t['status'] == 'SUCCESS'])
                print(f"\nIndian Market Data Connectivity: {successful}/{len(connectivity_tests)} sources working")
        
        # Test system integrity after updates
        if not args.dry_run:
            logger.info("Running post-update integrity checks...")
            integrity_results = dep_manager.verify_system_integrity()
            
            failed_imports = [r for r in integrity_results if r['status'] != 'OK']
            if failed_imports:
                logger.warning(f"System integrity issues found: {len(failed_imports)} packages have import issues")
                for failure in failed_imports:
                    logger.warning(f"  - {failure['package']}: {failure['status']}")
            else:
                logger.info("✓ All system integrity checks passed")
        
    except KeyboardInterrupt:
        logger.info("Update process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Update process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()