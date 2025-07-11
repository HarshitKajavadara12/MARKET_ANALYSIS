"""
Functional tests for report generation workflow of Market Research System v1.0
Tests PDF report generation, email delivery, and template functionality
"""

import unittest
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from reporting.pdf_generator import PDFReportGenerator
from reporting.visualization import ChartGenerator
from reporting.report_templates import ReportTemplateManager
from reporting.email_sender import EmailReportSender
from reporting.table_generator import TableGenerator
from utils.date_utils import format_date_for_report


class TestReportGenerationWorkflow(unittest.TestCase):
    """Test complete report generation workflow"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_reports_dir = tempfile.mkdtemp()
        cls.test_templates_dir = tempfile.mkdtemp()
        cls.test_date = datetime(2022, 6, 15)
        
        # Create sample market data
        cls.sample_data = cls._create_sample_market_data()
        cls.sample_indicators = cls._create_sample_indicators()
        cls.sample_analysis = cls._create_sample_analysis()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_reports_dir):
            shutil.rmtree(cls.test_reports_dir)
        if os.path.exists(cls.test_templates_dir):
            shutil.rmtree(cls.test_templates_dir)
    
    @classmethod
    def _create_sample_market_data(cls):
        """Create sample market data for testing"""
        dates = pd.date_range(start=cls.test_date - timedelta(days=30), end=cls.test_date, freq='D')
        
        sample_data = {}
        symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        
        for symbol in symbols:
            np.random.seed(42)  # For reproducible results
            base_price = 100 if 'RELIANCE' in symbol else 3000 if 'TCS' in symbol else 1500
            
            prices = []
            for i in range(len(dates)):
                if i == 0:
                    price = base_price
                else:
                    change = np.random.normal(0, 0.02) * prices[-1]
                    price = prices[-1] + change
                prices.append(max(price, 10))
            
            data = pd.DataFrame({
                'Date': dates,
                'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(10000, 100000) for _ in range(len(dates))]
            })
            data.set_index('Date', inplace=True)
            sample_data[symbol] = data
            
        return sample_data
    
    @classmethod
    def _create_sample_indicators(cls):
        """Create sample technical indicators"""
        indicators = {}
        for symbol in cls.sample_data.keys():
            data = cls.sample_data[symbol]
            indicators[symbol] = {
                'sma_20': data['Close'].rolling(20).mean().iloc[-1],
                'sma_50': data['Close'].rolling(50).mean().iloc[-1],
                'rsi': 65.5,
                'macd': 2.3,
                'bollinger_upper': data['Close'].iloc[-1] * 1.02,
                'bollinger_lower': data['Close'].iloc[-1] * 0.98,
                'volume_sma': data['Volume'].rolling(20).mean().iloc[-1]
            }
        return indicators
    
    @classmethod
    def _create_sample_analysis(cls):
        """Create sample analysis results"""
        return {
            'summary': {
                'market_trend': 'Bullish',
                'volatility': 'Medium',
                'top_performers': ['RELIANCE.NS', 'TCS.NS'],
                'market_breadth': 75.0
            },
            'trends': {
                'short_term': 'Upward',
                'medium_term': 'Sideways',
                'long_term': 'Upward'
            },
            'volatility': {
                'current_vix': 18.5,
                'avg_volatility': 16.2,
                'volatility_trend': 'Increasing'
            },
            'sectors': {
                'technology': {'performance': 2.5, 'trend': 'Positive'},
                'banking': {'performance': 1.8, 'trend': 'Neutral'},
                'energy': {'performance': -0.5, 'trend': 'Negative'}
            }
        }
    
    def setUp(self):
        """Set up each test"""
        self.pdf_generator = PDFReportGenerator()
        self.chart_generator = ChartGenerator()
        self.template_manager = ReportTemplateManager()
        self.email_sender = EmailReportSender()
        self.table_generator = TableGenerator()
    
    def test_daily_report_generation(self):
        """Test daily report generation workflow"""
        print("Testing daily report generation...")
        
        report_path = os.path.join(self.test_reports_dir, f"daily_report_{self.test_date.strftime('%Y%m%d')}.pdf")
        
        # Generate daily report
        success = self.pdf_generator.generate_daily_report(
            market_data=self.sample_data,
            technical_data=self.sample_indicators,
            analysis_results=self.sample_analysis,
            output_path=report_path,
            report_date=self.test_date
        )
        
        self.assertTrue(success, "Daily report generation should be successful")
        self.assertTrue(os.path.exists(report_path), "Daily report file should be created")
        
        # Verify file size
        file_size = os.path.getsize(report_path)
        self.assertGreater(file_size, 5000, "Daily report should be at least 5KB")
        
        print(f"Daily report generated successfully: {report_path}")
    
    def test_weekly_report_generation(self):
        """Test weekly report generation workflow"""
        print("Testing weekly report generation...")
        
        report_path = os.path.join(self.test_reports_dir, f"weekly_report_{self.test_date.strftime('%Y%m%d')}.pdf")
        
        # Generate weekly report with additional analysis
        weekly_analysis = self.sample_analysis.copy()
        weekly_analysis['weekly_summary'] = {
            'week_performance': 3.2,
            'weekly_high': max([data['Close'].max() for data in self.sample_data.values()]),
            'weekly_low': min([data['Close'].min() for data in self.sample_data.values()]),
            'weekly_volume': sum([data['Volume'].sum() for data in self.sample_data.values()])
        }
        
        success = self.pdf_generator.generate_weekly_report(
            market_data=self.sample_data,
            technical_data=self.sample_indicators,
            analysis_results=weekly_analysis,
            output_path=report_path,
            report_date=self.test_date
        )
        
        self.assertTrue(success, "Weekly report generation should be successful")
        self.assertTrue(os.path.exists(report_path), "Weekly report file should be created")
        
        # Verify file size (weekly reports should be larger)
        file_size = os.path.getsize(report_path)
        self.assertGreater(file_size, 8000, "Weekly report should be at least 8KB")
        
        print(f"Weekly report generated successfully: {report_path}")
    
    def test_monthly_report_generation(self):
        """Test monthly report generation workflow"""
        print("Testing monthly report generation...")
        
        report_path = os.path.join(self.test_reports_dir, f"monthly_report_{self.test_date.strftime('%Y%m')}.pdf")
        
        # Generate monthly report with comprehensive analysis
        monthly_analysis = self.sample_analysis.copy()
        monthly_analysis['monthly_summary'] = {
            'month_performance': 8.5,
            'best_performing_stock': 'RELIANCE.NS',
            'worst_performing_stock': 'INFY.NS',
            'monthly_volatility': 12.5,
            'correlation_analysis': {
                'RELIANCE.NS': {'TCS.NS': 0.65, 'INFY.NS': 0.58},
                'TCS.NS': {'RELIANCE.NS': 0.65, 'INFY.NS': 0.72}
            }
        }
        
        success = self.pdf_generator.generate_monthly_report(
            market_data=self.sample_data,
            technical_data=self.sample_indicators,
            analysis_results=monthly_analysis,
            output_path=report_path,
            report_date=self.test_date
        )
        
        self.assertTrue(success, "Monthly report generation should be successful")
        self.assertTrue(os.path.exists(report_path), "Monthly report file should be created")
        
        # Verify file size (monthly reports should be largest)
        file_size = os.path.getsize(report_path)
        self.assertGreater(file_size, 15000, "Monthly report should be at least 15KB")
        
        print(f"Monthly report generated successfully: {report_path}")
    
    def test_chart_generation_workflow(self):
        """Test chart generation for reports"""
        print("Testing chart generation workflow...")
        
        charts_dir = os.path.join(self.test_reports_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Test price chart generation
        for symbol in self.sample_data.keys():
            chart_path = os.path.join(charts_dir, f"{symbol}_price_chart.png")
            success = self.chart_generator.create_price_chart(
                data=self.sample_data[symbol],
                symbol=symbol,
                output_path=chart_path,
                include_volume=True
            )
            
            self.assertTrue(success, f"Price chart generation should be successful for {symbol}")
            self.assertTrue(os.path.exists(chart_path), f"Price chart file should be created for {symbol}")
        
        # Test technical indicators chart
        tech_chart_path = os.path.join(charts_dir, "technical_indicators.png")
        success = self.chart_generator.create_technical_indicators_chart(
            data=self.sample_data['RELIANCE.NS'],
            indicators=self.sample_indicators['RELIANCE.NS'],
            output_path=tech_chart_path
        )
        
        self.assertTrue(success, "Technical indicators chart generation should be successful")
        self.assertTrue(os.path.exists(tech_chart_path), "Technical indicators chart file should be created")
        
        # Test market overview chart
        overview_chart_path = os.path.join(charts_dir, "market_overview.png")
        success = self.chart_generator.create_market_overview_chart(
            market_data=self.sample_data,
            analysis_results=self.sample_analysis,
            output_path=overview_chart_path
        )
        
        self.assertTrue(success, "Market overview chart generation should be successful")
        self.assertTrue(os.path.exists(overview_chart_path), "Market overview chart file should be created")
        
        print(f"Charts generated successfully in: {charts_dir}")
    
    def test_table_generation_workflow(self):
        """Test table generation for reports"""
        print("Testing table generation workflow...")
        
        # Test market summary table
        market_summary_table = self.table_generator.create_market_summary_table(
            market_data=self.sample_data,
            analysis_results=self.sample_analysis
        )
        
        self.assertIsInstance(market_summary_table, pd.DataFrame, "Market summary should be a DataFrame")
        self.assertGreater(len(market_summary_table), 0, "Market summary table should not be empty")
        
        # Test technical indicators table
        tech_indicators_table = self.table_generator.create_technical_indicators_table(
            technical_data=self.sample_indicators
        )
        
        self.assertIsInstance(tech_indicators_table, pd.DataFrame, "Technical indicators should be a DataFrame")
        self.assertGreater(len(tech_indicators_table), 0, "Technical indicators table should not be empty")
        
        # Test performance table
        performance_table = self.table_generator.create_performance_table(
            market_data=self.sample_data,
            period_days=30
        )
        
        self.assertIsInstance(performance_table, pd.DataFrame, "Performance table should be a DataFrame")
        self.assertGreater(len(performance_table), 0, "Performance table should not be empty")
        
        print("Tables generated successfully")
    
    def test_template_management_workflow(self):
        """Test report template management"""
        print("Testing template management workflow...")
        
        # Test template loading
        daily_template = self.template_manager.load_template('daily')
        self.assertIsNotNone(daily_template, "Daily template should be loaded")
        
        weekly_template = self.template_manager.load_template('weekly')
        self.assertIsNotNone(weekly_template, "Weekly template should be loaded")
        
        monthly_template = self.template_manager.load_template('monthly')
        self.assertIsNotNone(monthly_template, "Monthly template should be loaded")
        
        # Test template customization
        custom_template_path = os.path.join(self.test_templates_dir, 'custom_template.html')
        custom_template_content = """
        <html>
        <head><title>Custom Report</title></head>
        <body>
            <h1>Custom Market Report</h1>
            <p>Date: {{report_date}}</p>
            <p>Market Summary: {{market_summary}}</p>
        </body>
        </html>
        """
        
        with open(custom_template_path, 'w') as f:
            f.write(custom_template_content)
        
        # Test custom template loading
        custom_template = self.template_manager.load_custom_template(custom_template_path)
        self.assertIsNotNone(custom_template, "Custom template should be loaded")
        
        print("Template management workflow completed successfully")
    
    @patch('smtplib.SMTP')
    def test_email_delivery_workflow(self, mock_smtp):
        """Test email delivery workflow"""
        print("Testing email delivery workflow...")
        
        # Mock SMTP server
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server
        
        # Create a test report
        report_path = os.path.join(self.test_reports_dir, "test_email_report.pdf")
        success = self.pdf_generator.generate_daily_report(
            market_data=self.sample_data,
            technical_data=self.sample_indicators,
            analysis_results=self.sample_analysis,
            output_path=report_path,
            report_date=self.test_date
        )
        
        self.assertTrue(success, "Test report should be generated for email")
        
        # Test email configuration
        email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'test@example.com',
            'password': 'test_password',
            'recipients': ['client1@example.com', 'client2@example.com']
        }
        
        # Configure email sender
        self.email_sender.configure(email_config)
        
        # Test email sending
        email_success = self.email_sender.send_daily_report(
            report_path=report_path,
            report_date=self.test_date,
            subject=f"Daily Market Report - {format_date_for_report(self.test_date)}"
        )
        
        self.assertTrue(email_success, "Email sending should be successful")
        
        # Verify SMTP calls
        mock_smtp.assert_called_once()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()
        
        print("Email delivery workflow completed successfully")
    
    def test_indian_market_specific_features(self):
        """Test Indian market specific features in reports"""
        print("Testing Indian market specific features...")
        
        # Test NSE/BSE specific data handling
        indian_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
        
        # Test market timing validation (IST)
        market_hours = self.template_manager.get_indian_market_hours()
        self.assertIsInstance(market_hours, dict, "Market hours should be returned as dict")
        self.assertIn('open', market_hours)
        self.assertIn('close', market_hours)
        
        # Test Indian market holidays
        holidays = self.template_manager.get_indian_market_holidays(self.test_date.year)
        self.assertIsInstance(holidays, list, "Holidays should be returned as list")
        
        # Test currency formatting (INR)
        sample_price = 2500.75
        formatted_price = self.table_generator.format_indian_currency(sample_price)
        self.assertIn('₹', formatted_price, "Price should be formatted with INR symbol")
        
        # Test sector-wise analysis for Indian market
        indian_sectors = {
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS'],
            'IT': ['TCS.NS', 'INFY.NS'],
            'Energy': ['RELIANCE.NS']
        }
        
        sector_analysis = self.chart_generator.create_indian_sector_analysis(
            market_data=self.sample_data,
            sector_mapping=indian_sectors
        )
        
        self.assertIsInstance(sector_analysis, dict, "Sector analysis should return dict")
        self.assertIn('Banking', sector_analysis)
        self.assertIn('IT', sector_analysis)
        
        print("Indian market specific features tested successfully")
    
    def test_nifty_sensex_analysis(self):
        """Test NIFTY and SENSEX index analysis"""
        print("Testing NIFTY and SENSEX analysis...")
        
        # Create sample index data
        nifty_data = pd.DataFrame({
            'Date': pd.date_range(start=self.test_date - timedelta(days=30), end=self.test_date, freq='D'),
            'Close': np.random.uniform(16000, 18000, 31),
            'Volume': np.random.uniform(1000000, 5000000, 31)
        }).set_index('Date')
        
        sensex_data = pd.DataFrame({
            'Date': pd.date_range(start=self.test_date - timedelta(days=30), end=self.test_date, freq='D'),
            'Close': np.random.uniform(54000, 60000, 31),
            'Volume': np.random.uniform(500000, 2000000, 31)
        }).set_index('Date')
        
        # Test index correlation analysis
        correlation = self.chart_generator.calculate_index_correlation(nifty_data, sensex_data)
        self.assertIsInstance(correlation, float, "Correlation should be a float")
        self.assertGreaterEqual(correlation, -1, "Correlation should be >= -1")
        self.assertLessEqual(correlation, 1, "Correlation should be <= 1")
        
        # Test index performance comparison
        index_comparison = self.table_generator.create_index_comparison_table(
            nifty_data=nifty_data,
            sensex_data=sensex_data,
            period_days=30
        )
        
        self.assertIsInstance(index_comparison, pd.DataFrame, "Index comparison should be DataFrame")
        self.assertIn('NIFTY', index_comparison.columns)
        self.assertIn('SENSEX', index_comparison.columns)
        
        print("NIFTY and SENSEX analysis completed successfully")
    
    def test_regulatory_compliance_checks(self):
        """Test regulatory compliance features for Indian market"""
        print("Testing regulatory compliance checks...")
        
        # Test SEBI compliance checks
        compliance_check = self.template_manager.check_sebi_compliance(
            report_data=self.sample_analysis,
            report_type='daily'
        )
        
        self.assertIsInstance(compliance_check, dict, "Compliance check should return dict")
        self.assertIn('compliant', compliance_check)
        self.assertIn('warnings', compliance_check)
        
        # Test disclosure requirements
        disclosure_text = self.template_manager.get_indian_market_disclosures()
        self.assertIsInstance(disclosure_text, str, "Disclosure should be string")
        self.assertIn('SEBI', disclosure_text, "Disclosure should mention SEBI")
        
        # Test risk warning text
        risk_warning = self.template_manager.get_indian_risk_warning()
        self.assertIsInstance(risk_warning, str, "Risk warning should be string")
        self.assertGreater(len(risk_warning), 50, "Risk warning should be substantial")
        
        print("Regulatory compliance checks completed successfully")
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking against Indian indices"""
        print("Testing performance benchmarking...")
        
        # Create benchmark data
        benchmark_data = {
            'NIFTY50': pd.Series(np.random.uniform(16000, 18000, 30)),
            'SENSEX': pd.Series(np.random.uniform(54000, 60000, 30)),
            'NIFTY_BANK': pd.Series(np.random.uniform(35000, 40000, 30))
        }
        
        # Test individual stock vs benchmark
        for symbol in self.sample_data.keys():
            stock_data = self.sample_data[symbol]['Close']
            
            benchmark_comparison = self.chart_generator.compare_with_benchmark(
                stock_data=stock_data,
                benchmark_data=benchmark_data['NIFTY50'],
                stock_symbol=symbol,
                benchmark_name='NIFTY50'
            )
            
            self.assertIsInstance(benchmark_comparison, dict, f"Benchmark comparison should return dict for {symbol}")
            self.assertIn('outperformance', benchmark_comparison)
            self.assertIn('correlation', benchmark_comparison)
            self.assertIn('beta', benchmark_comparison)
        
        # Test portfolio vs benchmark
        portfolio_performance = self.table_generator.create_portfolio_vs_benchmark_table(
            portfolio_data=self.sample_data,
            benchmark_data=benchmark_data,
            weights={'RELIANCE.NS': 0.4, 'TCS.NS': 0.35, 'INFY.NS': 0.25}
        )
        
        self.assertIsInstance(portfolio_performance, pd.DataFrame, "Portfolio performance should be DataFrame")
        
        print("Performance benchmarking completed successfully")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        print("Testing error handling and recovery...")
        
        # Test handling of missing data
        incomplete_data = self.sample_data.copy()
        incomplete_data['TEST.NS'] = pd.DataFrame()  # Empty dataframe
        
        # Should handle gracefully without crashing
        try:
            report_path = os.path.join(self.test_reports_dir, "error_test_report.pdf")
            success = self.pdf_generator.generate_daily_report(
                market_data=incomplete_data,
                technical_data=self.sample_indicators,
                analysis_results=self.sample_analysis,
                output_path=report_path,
                report_date=self.test_date
            )
            # Should either succeed with warnings or fail gracefully
            self.assertIsInstance(success, bool, "Should return boolean result")
        except Exception as e:
            self.fail(f"Should handle missing data gracefully, but raised: {e}")
        
        # Test handling of invalid file paths
        invalid_path = "/invalid/path/report.pdf"
        try:
            success = self.pdf_generator.generate_daily_report(
                market_data=self.sample_data,
                technical_data=self.sample_indicators,
                analysis_results=self.sample_analysis,
                output_path=invalid_path,
                report_date=self.test_date
            )
            self.assertFalse(success, "Should return False for invalid path")
        except Exception as e:
            # Should either return False or raise specific exception
            self.assertIsInstance(e, (FileNotFoundError, PermissionError, OSError))
        
        # Test network failure handling for email
        with patch('smtplib.SMTP', side_effect=Exception("Network error")):
            email_success = self.email_sender.send_daily_report(
                report_path=os.path.join(self.test_reports_dir, "test_report.pdf"),
                report_date=self.test_date,
                subject="Test Report"
            )
            self.assertFalse(email_success, "Should return False on network error")
        
        print("Error handling and recovery testing completed successfully")
    
    def test_data_validation_and_quality_checks(self):
        """Test data validation and quality checks"""
        print("Testing data validation and quality checks...")
        
        # Test data completeness validation
        for symbol in self.sample_data.keys():
            data = self.sample_data[symbol]
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                self.assertIn(col, data.columns, f"{col} should be present in {symbol} data")
            
            # Check for data consistency (High >= Low, etc.)
            self.assertTrue((data['High'] >= data['Low']).all(), f"High should be >= Low for {symbol}")
            self.assertTrue((data['High'] >= data['Open']).all() or (data['High'] >= data['Close']).all(), 
                          f"High should be >= Open or Close for {symbol}")
            self.assertTrue((data['Low'] <= data['Open']).all() or (data['Low'] <= data['Close']).all(), 
                          f"Low should be <= Open or Close for {symbol}")
            
            # Check for positive volume
            self.assertTrue((data['Volume'] > 0).all(), f"Volume should be positive for {symbol}")
        
        # Test technical indicators validation
        for symbol, indicators in self.sample_indicators.items():
            # RSI should be between 0 and 100
            if 'rsi' in indicators:
                self.assertGreaterEqual(indicators['rsi'], 0, f"RSI should be >= 0 for {symbol}")
                self.assertLessEqual(indicators['rsi'], 100, f"RSI should be <= 100 for {symbol}")
            
            # Bollinger bands should make sense
            if all(key in indicators for key in ['bollinger_upper', 'bollinger_lower']):
                self.assertGreater(indicators['bollinger_upper'], indicators['bollinger_lower'], 
                                 f"Bollinger upper should be > lower for {symbol}")
        
        print("Data validation and quality checks completed successfully")
    
    def test_report_versioning_and_archival(self):
        """Test report versioning and archival system"""
        print("Testing report versioning and archival...")
        
        # Test report versioning
        base_report_path = os.path.join(self.test_reports_dir, "versioned_report.pdf")
        
        # Generate multiple versions
        for version in range(1, 4):
            versioned_analysis = self.sample_analysis.copy()
            versioned_analysis['version'] = f"v{version}"
            versioned_analysis['timestamp'] = datetime.now().isoformat()
            
            success = self.pdf_generator.generate_daily_report(
                market_data=self.sample_data,
                technical_data=self.sample_indicators,
                analysis_results=versioned_analysis,
                output_path=base_report_path.replace('.pdf', f'_v{version}.pdf'),
                report_date=self.test_date
            )
            
            self.assertTrue(success, f"Version {version} report should be generated successfully")
        
        # Test archival functionality
        archive_dir = os.path.join(self.test_reports_dir, 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        
        # Create archive metadata
        archive_metadata = {
            'archived_date': datetime.now().isoformat(),
            'reports_count': 3,
            'date_range': {
                'start': (self.test_date - timedelta(days=30)).isoformat(),
                'end': self.test_date.isoformat()
            },
            'version': '1.0'
        }
        
        metadata_path = os.path.join(archive_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(archive_metadata, f, indent=2)
        
        self.assertTrue(os.path.exists(metadata_path), "Archive metadata should be created")
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        self.assertEqual(loaded_metadata['reports_count'], 3, "Reports count should match")
        self.assertEqual(loaded_metadata['version'], '1.0', "Version should match")
        
        print("Report versioning and archival testing completed successfully")
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        print("Testing performance metrics calculation...")
        
        # Test individual stock performance metrics
        for symbol in self.sample_data.keys():
            data = self.sample_data[symbol]
            
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            # Test Sharpe ratio calculation
            sharpe_ratio = self.table_generator.calculate_sharpe_ratio(returns, risk_free_rate=0.06)
            self.assertIsInstance(sharpe_ratio, float, f"Sharpe ratio should be float for {symbol}")
            
            # Test maximum drawdown
            max_drawdown = self.table_generator.calculate_max_drawdown(data['Close'])
            self.assertIsInstance(max_drawdown, float, f"Max drawdown should be float for {symbol}")
            self.assertLessEqual(max_drawdown, 0, f"Max drawdown should be <= 0 for {symbol}")
            
            # Test volatility
            volatility = self.table_generator.calculate_volatility(returns, annualize=True)
            self.assertIsInstance(volatility, float, f"Volatility should be float for {symbol}")
            self.assertGreaterEqual(volatility, 0, f"Volatility should be >= 0 for {symbol}")
        
        # Test portfolio-level metrics
        portfolio_metrics = self.table_generator.calculate_portfolio_metrics(
            market_data=self.sample_data,
            weights={'RELIANCE.NS': 0.4, 'TCS.NS': 0.35, 'INFY.NS': 0.25},
            benchmark_return=0.12
        )
        
        self.assertIsInstance(portfolio_metrics, dict, "Portfolio metrics should be dict")
        self.assertIn('total_return', portfolio_metrics)
        self.assertIn('sharpe_ratio', portfolio_metrics)
        self.assertIn('max_drawdown', portfolio_metrics)
        self.assertIn('volatility', portfolio_metrics)
        
        print("Performance metrics calculation completed successfully")


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReportGenerationWorkflow)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MARKET RESEARCH SYSTEM v1.0 - REPORT GENERATION TESTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors))/result.testsRun)*100:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print(f"\n{'='*60}")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)