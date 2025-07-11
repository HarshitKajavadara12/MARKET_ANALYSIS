#!/usr/bin/env python3
"""
Market Research System v1.0 - API Connection Testing Script
Indian Stock Market Focus
Created: January 2022
Author: Independent Market Researcher

This script tests all API connections and data sources used in the system.
"""

import sys
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class APIConnectionTester:
    def __init__(self):
        self.project_root = project_root
        self.config_dir = self.project_root / "config"
        self.test_results = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for test results"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.project_root / "logs" / "application" / "api_tests.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def test_yahoo_finance_api(self):
        """Test Yahoo Finance API with Indian stocks"""
        print("\n🔍 Testing Yahoo Finance API...")
        self.logger.info("Starting Yahoo Finance API test")
        
        test_symbols = [
            'RELIANCE.NS',  # Reliance Industries
            'TCS.NS',       # Tata Consultancy Services
            'HDFCBANK.NS',  # HDFC Bank
            'INFY.NS',      # Infosys
            '^NSEI'         # Nifty 50 Index
        ]
        
        results = {}
        
        for symbol in test_symbols:
            try:
                print(f"  Testing {symbol}...")
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Test basic info
                info = ticker.info
                if info and len(info) > 1:
                    print(f"    ✅ Info: {info.get('longName', 'Unknown')}")
                    results[f"{symbol}_info"] = "SUCCESS"
                else:
                    print(f"    ❌ Info: No data available")
                    results[f"{symbol}_info"] = "FAILED"
                
                # Test historical data (last 30 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                hist_data = ticker.history(start=start_date, end=end_date)
                
                if not hist_data.empty:
                    print(f"    ✅ Historical: {len(hist_data)} records")
                    print(f"    📊 Latest Close: ₹{hist_data['Close'].iloc[-1]:.2f}")
                    results[f"{symbol}_history"] = "SUCCESS"
                else:
                    print(f"    ❌ Historical: No data available")
                    results[f"{symbol}_history"] = "FAILED"
                
                # Small delay between requests
                time.sleep(1)
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                results[f"{symbol}_error"] = str(e)
                self.logger.error(f"Yahoo Finance API error for {symbol}: {e}")
        
        self.test_results['yahoo_finance'] = results
        
        # Summary
        successful_tests = sum(1 for v in results.values() if v == "SUCCESS")
        total_tests = len([k for k in results.keys() if not k.endswith('_error')])
        
        print(f"\n📊 Yahoo Finance API Results: {successful_tests}/{total_tests} tests passed")
        self.logger.info(f"Yahoo Finance API test completed: {successful_tests}/{total_tests} passed")
        
    def test_nse_website_accessibility(self):
        """Test NSE website accessibility (basic check)"""
        print("\n🔍 Testing NSE Website Accessibility...")
        self.logger.info("Starting NSE website accessibility test")
        
        nse_urls = [
            'https://www.nseindia.com',
            'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050',
            'https://www.nseindia.com/market-data/live-equity-market'
        ]
        
        results = {}
        
        for url in nse_urls:
            try:
                print(f"  Testing {url}...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    print(f"    ✅ Status: {response.status_code}")
                    results[url] = "SUCCESS"
                else:
                    print(f"    ⚠️ Status: {response.status_code}")
                    results[url] = f"HTTP_{response.status_code}"
                
                time.sleep(2)  # Respectful delay
                
            except requests.RequestException as e:
                print(f"    ❌ Error: {str(e)}")
                results[url] = f"ERROR: {str(e)}"
                self.logger.error(f"NSE website error for {url}: {e}")
        
        self.test_results['nse_website'] = results
        
        # Summary
        successful_tests = sum(1 for v in results.values() if v == "SUCCESS")
        total_tests = len(results)
        
        print(f"\n📊 NSE Website Results: {successful_tests}/{total_tests} tests passed")
        self.logger.info(f"NSE website test completed: {successful_tests}/{total_tests} passed")
        
    def test_bse_website_accessibility(self):
        """Test BSE website accessibility (basic check)"""
        print("\n🔍 Testing BSE Website Accessibility...")
        self.logger.info("Starting BSE website accessibility test")
        
        bse_urls = [
            'https://www.bseindia.com',
            'https://api.bseindia.com/BseIndiaAPI/api/SensexData/w'
        ]
        
        results = {}
        
        for url in bse_urls:
            try:
                print(f"  Testing {url}...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    print(f"    ✅ Status: {response.status_code}")
                    results[url] = "SUCCESS"
                else:
                    print(f"    ⚠️ Status: {response.status_code}")
                    results[url] = f"HTTP_{response.status_code}"
                
                time.sleep(2)  # Respectful delay
                
            except requests.RequestException as e:
                print(f"    ❌ Error: {str(e)}")
                results[url] = f"ERROR: {str(e)}"
                self.logger.error(f"BSE website error for {url}: {e}")
        
        self.test_results['bse_website'] = results
        
        # Summary
        successful_tests = sum(1 for v in results.values() if v == "SUCCESS")
        total_tests = len(results)
        
        print(f"\n📊 BSE Website Results: {successful_tests}/{total_tests} tests passed")
        self.logger.info(f"BSE website test completed: {successful_tests}/{total_tests} passed")
        
    def test_economic_data_sources(self):
        """Test economic data sources"""
        print("\n🔍 Testing Economic Data Sources...")
        self.logger.info("Starting economic data sources test")
        
        # Test Reserve Bank of India (RBI) website
        rbi_urls = [
            'https://rbi.org.in',
            'https://www.rbi.org.in/scripts/BS_ViewBulletin.aspx'
        ]
        
        results = {}
        
        for url in rbi_urls:
            try:
                print(f"  Testing RBI: {url}...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    print(f"    ✅ Status: {response.status_code}")
                    results[f"rbi_{url.split('/')[-1]}"] = "SUCCESS"
                else:
                    print(f"    ⚠️ Status: {response.status_code}")
                    results[f"rbi_{url.split('/')[-1]}"] = f"HTTP_{response.status_code}"
                
                time.sleep(2)
                
            except requests.RequestException as e:
                print(f"    ❌ Error: {str(e)}")
                results[f"rbi_{url.split('/')[-1]}"] = f"ERROR: {str(e)}"
                self.logger.error(f"RBI website error for {url}: {e}")
        
        # Test other economic data sources
        economic_sources = [
            'https://tradingeconomics.com/india/indicators',
            'https://www.investing.com/economic-calendar/'
        ]
        
        for url in economic_sources:
            try:
                print(f"  Testing Economic Data: {url}...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    print(f"    ✅ Status: {response.status_code}")
                    source_name = url.split('/')[2].replace('www.', '').replace('.com', '').replace('.in', '')
                    results[f"{source_name}_data"] = "SUCCESS"
                else:
                    print(f"    ⚠️ Status: {response.status_code}")
                    source_name = url.split('/')[2].replace('www.', '').replace('.com', '').replace('.in', '')
                    results[f"{source_name}_data"] = f"HTTP_{response.status_code}"
                
                time.sleep(3)  # Longer delay for economic sources
                
            except requests.RequestException as e:
                print(f"    ❌ Error: {str(e)}")
                source_name = url.split('/')[2].replace('www.', '').replace('.com', '').replace('.in', '')
                results[f"{source_name}_data"] = f"ERROR: {str(e)}"
                self.logger.error(f"Economic data source error for {url}: {e}")
        
        self.test_results['economic_data'] = results
        
        # Summary
        successful_tests = sum(1 for v in results.values() if v == "SUCCESS")
        total_tests = len(results)
        
        print(f"\n📊 Economic Data Sources Results: {successful_tests}/{total_tests} tests passed")
        self.logger.info(f"Economic data sources test completed: {successful_tests}/{total_tests} passed")
        
    def test_news_sources_basic(self):
        """Test basic news sources accessibility for 2022 version"""
        print("\n🔍 Testing News Sources...")
        self.logger.info("Starting news sources test")
        
        # Basic news sources for Indian market (2022 era)
        news_sources = [
            'https://economictimes.indiatimes.com',
            'https://www.business-standard.com',
            'https://www.livemint.com',
            'https://www.moneycontrol.com',
            'https://www.cnbctv18.com'
        ]
        
        results = {}
        
        for url in news_sources:
            try:
                print(f"  Testing {url.split('/')[2]}...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    print(f"    ✅ Accessible")
                    source_name = url.split('/')[2].replace('www.', '').split('.')[0]
                    results[f"{source_name}_news"] = "SUCCESS"
                    
                    # Check if content contains market-related keywords
                    content_lower = response.text.lower()
                    market_keywords = ['nifty', 'sensex', 'stock', 'market', 'shares', 'equity']
                    
                    found_keywords = [kw for kw in market_keywords if kw in content_lower]
                    if found_keywords:
                        print(f"    📈 Market content detected: {', '.join(found_keywords[:3])}")
                    
                else:
                    print(f"    ⚠️ Status: {response.status_code}")
                    source_name = url.split('/')[2].replace('www.', '').split('.')[0]
                    results[f"{source_name}_news"] = f"HTTP_{response.status_code}"
                
                time.sleep(2)  # Respectful delay
                
            except requests.RequestException as e:
                print(f"    ❌ Error: {str(e)}")
                source_name = url.split('/')[2].replace('www.', '').split('.')[0]
                results[f"{source_name}_news"] = f"ERROR: {str(e)}"
                self.logger.error(f"News source error for {url}: {e}")
        
        self.test_results['news_sources'] = results
        
        # Summary
        successful_tests = sum(1 for v in results.values() if v == "SUCCESS")
        total_tests = len(results)
        
        print(f"\n📊 News Sources Results: {successful_tests}/{total_tests} tests passed")
        self.logger.info(f"News sources test completed: {successful_tests}/{total_tests} passed")
        
    def test_data_storage_paths(self):
        """Test if all required directories exist and are writable"""
        print("\n🔍 Testing Data Storage Paths...")
        self.logger.info("Starting data storage paths test")
        
        required_dirs = [
            self.project_root / "data" / "raw",
            self.project_root / "data" / "processed",
            self.project_root / "data" / "indian_stocks",
            self.project_root / "data" / "indices",
            self.project_root / "logs" / "application",
            self.project_root / "logs" / "data_fetch",
            self.project_root / "reports" / "daily",
            self.project_root / "reports" / "weekly",
            self.project_root / "config"
        ]
        
        results = {}
        
        for dir_path in required_dirs:
            try:
                print(f"  Testing {dir_path.relative_to(self.project_root)}...")
                
                # Check if directory exists
                if dir_path.exists():
                    print(f"    ✅ Directory exists")
                    
                    # Test write permission
                    test_file = dir_path / "test_write.tmp"
                    try:
                        test_file.write_text("test")
                        test_file.unlink()  # Delete test file
                        print(f"    ✅ Write permission OK")
                        results[str(dir_path.relative_to(self.project_root))] = "SUCCESS"
                    except Exception as e:
                        print(f"    ❌ Write permission failed: {e}")
                        results[str(dir_path.relative_to(self.project_root))] = f"WRITE_ERROR: {e}"
                        
                else:
                    print(f"    ⚠️ Directory missing - creating...")
                    dir_path.mkdir(parents=True, exist_ok=True)
                    
                    if dir_path.exists():
                        print(f"    ✅ Directory created successfully")
                        results[str(dir_path.relative_to(self.project_root))] = "CREATED"
                    else:
                        print(f"    ❌ Failed to create directory")
                        results[str(dir_path.relative_to(self.project_root))] = "CREATE_FAILED"
                        
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                results[str(dir_path.relative_to(self.project_root))] = f"ERROR: {str(e)}"
                self.logger.error(f"Directory test error for {dir_path}: {e}")
        
        self.test_results['storage_paths'] = results
        
        # Summary
        successful_tests = sum(1 for v in results.values() if v in ["SUCCESS", "CREATED"])
        total_tests = len(results)
        
        print(f"\n📊 Storage Paths Results: {successful_tests}/{total_tests} paths ready")
        self.logger.info(f"Storage paths test completed: {successful_tests}/{total_tests} ready")
        
    def test_basic_calculations(self):
        """Test basic calculation functions for technical indicators"""
        print("\n🔍 Testing Basic Calculation Functions...")
        self.logger.info("Starting basic calculations test")
        
        results = {}
        
        try:
            # Create sample data for testing
            dates = pd.date_range(start='2022-01-01', end='2022-01-30', freq='D')
            sample_prices = [100 + i + (i % 3) * 2 for i in range(len(dates))]
            
            test_data = pd.DataFrame({
                'Date': dates,
                'Close': sample_prices,
                'High': [p + 2 for p in sample_prices],
                'Low': [p - 2 for p in sample_prices],
                'Volume': [1000000 + i * 10000 for i in range(len(dates))]
            })
            
            print("  Testing Moving Average calculation...")
            # Simple Moving Average
            sma_5 = test_data['Close'].rolling(window=5).mean()
            if not sma_5.isna().all():
                print(f"    ✅ SMA(5) calculated: Last value = {sma_5.iloc[-1]:.2f}")
                results['sma_calculation'] = "SUCCESS"
            else:
                print(f"    ❌ SMA calculation failed")
                results['sma_calculation'] = "FAILED"
            
            print("  Testing RSI calculation...")
            # Basic RSI calculation
            delta = test_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if not rsi.isna().all():
                print(f"    ✅ RSI calculated: Last value = {rsi.iloc[-1]:.2f}")
                results['rsi_calculation'] = "SUCCESS"
            else:
                print(f"    ❌ RSI calculation failed")
                results['rsi_calculation'] = "FAILED"
            
            print("  Testing Bollinger Bands calculation...")
            # Bollinger Bands
            sma_20 = test_data['Close'].rolling(window=20).mean()
            std_20 = test_data['Close'].rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            
            if not upper_band.isna().all() and not lower_band.isna().all():
                print(f"    ✅ Bollinger Bands calculated")
                results['bollinger_calculation'] = "SUCCESS"
            else:
                print(f"    ❌ Bollinger Bands calculation failed")
                results['bollinger_calculation'] = "FAILED"
                
        except Exception as e:
            print(f"    ❌ Calculation error: {str(e)}")
            results['calculations_error'] = str(e)
            self.logger.error(f"Basic calculations test error: {e}")
        
        self.test_results['basic_calculations'] = results
        
        # Summary
        successful_tests = sum(1 for v in results.values() if v == "SUCCESS")
        total_tests = len([k for k in results.keys() if not k.endswith('_error')])
        
        print(f"\n📊 Basic Calculations Results: {successful_tests}/{total_tests} tests passed")
        self.logger.info(f"Basic calculations test completed: {successful_tests}/{total_tests} passed")
        
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE API CONNECTION TEST REPORT")
        print("="*60)
        print(f"🕐 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏗️ System Version: Market Research System v1.0 (2022)")
        print(f"🇮🇳 Market Focus: Indian Stock Market")
        
        total_tests = 0
        total_passed = 0
        
        for category, tests in self.test_results.items():
            print(f"\n📊 {category.upper().replace('_', ' ')}")
            print("-" * 40)
            
            category_passed = 0
            category_total = 0
            
            for test_name, result in tests.items():
                if not test_name.endswith('_error'):
                    category_total += 1
                    total_tests += 1
                    
                    if result in ["SUCCESS", "CREATED"]:
                        print(f"  ✅ {test_name}: {result}")
                        category_passed += 1
                        total_passed += 1
                    else:
                        print(f"  ❌ {test_name}: {result}")
            
            print(f"  📈 Category Score: {category_passed}/{category_total}")
        
        print("\n" + "="*60)
        print(f"🎯 OVERALL SYSTEM HEALTH: {total_passed}/{total_tests} ({(total_passed/total_tests)*100:.1f}%)")
        
        if total_passed == total_tests:
            print("🟢 STATUS: ALL SYSTEMS OPERATIONAL")
        elif total_passed >= total_tests * 0.8:
            print("🟡 STATUS: MOSTLY OPERATIONAL - MINOR ISSUES")
        elif total_passed >= total_tests * 0.6:
            print("🟠 STATUS: PARTIALLY OPERATIONAL - ATTENTION NEEDED")
        else:
            print("🔴 STATUS: SYSTEM ISSUES - IMMEDIATE ATTENTION REQUIRED")
        
        print("="*60)
        
        # Save report to file
        report_file = self.project_root / "reports" / f"api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write("Market Research System v1.0 - API Connection Test Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                for category, tests in self.test_results.items():
                    f.write(f"{category.upper().replace('_', ' ')}\n")
                    f.write("-" * 40 + "\n")
                    
                    for test_name, result in tests.items():
                        f.write(f"{test_name}: {result}\n")
                    f.write("\n")
                
                f.write(f"OVERALL SCORE: {total_passed}/{total_tests} ({(total_passed/total_tests)*100:.1f}%)\n")
            
            print(f"📄 Report saved: {report_file.relative_to(self.project_root)}")
            
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
            self.logger.error(f"Report saving error: {e}")
        
    def run_all_tests(self):
        """Run all API connection tests"""
        print("🚀 Starting Market Research System v1.0 API Connection Tests")
        print(f"🇮🇳 Focus: Indian Stock Market Data Sources")
        print(f"📅 System Era: 2022 - Basic Data Collection & Analysis")
        print("="*60)
        
        # Run all tests
        self.test_yahoo_finance_api()
        self.test_nse_website_accessibility()
        self.test_bse_website_accessibility()
        self.test_economic_data_sources()
        self.test_news_sources_basic()
        self.test_data_storage_paths()
        self.test_basic_calculations()
        
        # Generate final report
        self.generate_test_report()

def main():
    """Main function to run API connection tests"""
    tester = APIConnectionTester()
    tester.run_all_tests()
    
    print("\n🎯 Next Steps:")
    print("1. Review any failed tests and fix connectivity issues")
    print("2. Run data collection scripts if all tests pass")
    print("3. Check logs in logs/application/api_tests.log for details")
    print("4. Update config files based on working data sources")
    
if __name__ == "__main__":
    main()