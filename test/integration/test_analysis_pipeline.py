#!/usr/bin/env python3
"""
Integration Test - Analysis Pipeline
Market Research System v1.0 - 2022
Tests complete analysis workflow

This module tests the entire analysis pipeline including:
- Technical indicator calculations
- Statistical analysis
- Correlation analysis
- Performance metrics
- Risk calculations
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from data.fetch_market_data import IndianMarketDataFetcher
from analysis.technical_indicators import TechnicalIndicators
from analysis.market_analyzer import MarketAnalyzer
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.trend_analyzer import TrendAnalyzer
from analysis.volatility_analyzer import VolatilityAnalyzer
from analysis.performance_analyzer import PerformanceAnalyzer
from utils.logging_utils import setup_logger
from . import TEST_SYMBOLS, TEST_INDICES, TEST_START_DATE


class TestAnalysisPipeline(unittest.TestCase):
    """Test complete analysis pipeline integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.logger = setup_logger('test_analysis_pipeline', 'test_analysis_pipeline.log')
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_symbols = TEST_SYMBOLS[:5]  # Use subset for faster testing
        cls.start_date = TEST_START_DATE
        cls.end_date = '2022-06-30'
        
        # Initialize components
        cls.market_fetcher = IndianMarketDataFetcher()
        cls.technical_indicators = TechnicalIndicators()
        cls.market_analyzer = MarketAnalyzer()
        cls.correlation_analyzer = CorrelationAnalyzer()
        cls.trend_analyzer = TrendAnalyzer()
        cls.volatility_analyzer = VolatilityAnalyzer()
        cls.performance_analyzer = PerformanceAnalyzer()
        
        # Fetch test data
        cls.test_data = {}
        for symbol in cls.test_symbols:
            try:
                data = cls.market_fetcher.fetch_stock_data(symbol, cls.start_date, cls.end_date)
                if data is not None and not data.empty:
                    cls.test_data[symbol] = data
                    cls.logger.info(f"Loaded test data for {symbol}: {len(data)} rows")
            except Exception as e:
                cls.logger.warning(f"Failed to load data for {symbol}: {str(e)}")
        
        cls.logger.info(f"Test environment set up with {len(cls.test_data)} symbols")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        cls.logger.info("Test environment cleaned up")
    
    def test_01_technical_indicators_pipeline(self):
        """Test technical indicators calculation pipeline"""
        self.logger.info("Testing technical indicators pipeline")
        
        for symbol, data in self.test_data.items():
            try:
                # Test moving averages
                sma_20 = self.technical_indicators.simple_moving_average(data['Close'], 20)
                ema_20 = self.technical_indicators.exponential_moving_average(data['Close'], 20)
                
                self.assertEqual(len(sma_20), len(data), f"SMA length mismatch for {symbol}")
                self.assertEqual(len(ema_20), len(data), f"EMA length mismatch for {symbol}")
                
                # Test momentum indicators
                rsi = self.technical_indicators.rsi(data['Close'], 14)
                macd_line, macd_signal, macd_histogram = self.technical_indicators.macd(data['Close'])
                
                self.assertEqual(len(rsi), len(data), f"RSI length mismatch for {symbol}")
                self.assertTrue(all(0 <= val <= 100 for val in rsi.dropna()), 
                              f"RSI values out of range for {symbol}")
                
                # Test volatility indicators
                bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(data['Close'])
                atr = self.technical_indicators.average_true_range(data['High'], data['Low'], data['Close'])
                
                self.assertTrue(all(bb_upper >= bb_middle) and all(bb_middle >= bb_lower),
                              f"Bollinger bands order incorrect for {symbol}")
                
                # Test volume indicators
                if 'Volume' in data.columns:
                    obv = self.technical_indicators.on_balance_volume(data['Close'], data['Volume'])
                    self.assertEqual(len(obv), len(data), f"OBV length mismatch for {symbol}")
                
                # Test comprehensive indicator calculation
                indicators_data = self.technical_indicators.calculate_all_indicators(data)
                self.assertGreater(len(indicators_data.columns), len(data.columns),
                                 f"Indicators not added for {symbol}")
                
                self.logger.info(f"Technical indicators pipeline completed for {symbol}")
                
            except Exception as e:
                self.fail(f"Technical indicators pipeline failed for {symbol}: {str(e)}")
    
    def test_02_market_analysis_pipeline(self):
        """Test market analysis pipeline"""
        self.logger.info("Testing market analysis pipeline")
        
        for symbol, data in self.test_data.items():
            try:
                # Add technical indicators
                analysis_data = self.technical_indicators.calculate_all_indicators(data)
                
                # Test basic statistics
                stats = self.market_analyzer.calculate_basic_statistics(analysis_data['Close'])
                
                required_stats = ['mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis']
                for stat in required_stats:
                    self.assertIn(stat, stats, f"Missing statistic {stat} for {symbol}")
                    self.assertIsNotNone(stats[stat], f"Null statistic {stat} for {symbol}")
                
                # Test price analysis
                price_analysis = self.market_analyzer.analyze_price_movements(analysis_data)
                
                self.assertIn('daily_returns', price_analysis, f"Missing daily returns for {symbol}")
                self.assertIn('volatility', price_analysis, f"Missing volatility for {symbol}")
                
                # Test support and resistance levels
                support_resistance = self.market_analyzer.find_support_resistance_levels(analysis_data)
                
                self.assertIn('support_levels', support_resistance, f"Missing support levels for {symbol}")
                self.assertIn('resistance_levels', support_resistance, f"Missing resistance levels for {symbol}")
                
                # Test pattern recognition
                patterns = self.market_analyzer.detect_chart_patterns(analysis_data)
                self.assertIsInstance(patterns, dict, f"Invalid patterns format for {symbol}")
                
                self.logger.info(f"Market analysis pipeline completed for {symbol}")
                
            except Exception as e:
                self.fail(f"Market analysis pipeline failed for {symbol}: {str(e)}")
    
    def test_03_correlation_analysis_pipeline(self):
        """Test correlation analysis pipeline"""
        self.logger.info("Testing correlation analysis pipeline")
        
        if len(self.test_data) < 2:
            self.skipTest("Need at least 2 symbols for correlation analysis")
        
        try:
            # Prepare data for correlation analysis
            price_data = {}
            return_data = {}
            
            for symbol, data in self.test_data.items():
                price_data[symbol] = data['Close']
                return_data[symbol] = data['Close'].pct_change().dropna()
            
            # Create DataFrames
            prices_df = pd.DataFrame(price_data).dropna()
            returns_df = pd.DataFrame(return_data).dropna()
            
            # Test price correlation
            price_correlation = self.correlation_analyzer.calculate_price_correlation(prices_df)
            
            self.assertEqual(price_correlation.shape, (len(self.test_data), len(self.test_data)),
                           "Price correlation matrix shape incorrect")
            
            # Check diagonal elements are 1
            np.testing.assert_array_almost_equal(np.diag(price_correlation), 1.0,
                                               decimal=6, err_msg="Diagonal correlation not 1")
            
            # Test return correlation
            return_correlation = self.correlation_analyzer.calculate_return_correlation(returns_df)
            
            self.assertEqual(return_correlation.shape, (len(self.test_data), len(self.test_data)),
                           "Return correlation matrix shape incorrect")
            
            # Test rolling correlation
            if len(self.test_data) >= 2:
                symbols = list(self.test_data.keys())[:2]
                rolling_corr = self.correlation_analyzer.calculate_rolling_correlation(
                    prices_df[symbols[0]], prices_df[symbols[1]], window=30
                )
                
                self.assertIsInstance(rolling_corr, pd.Series, "Rolling correlation not a Series")
                self.assertGreater(len(rolling_corr.dropna()), 0, "No rolling correlation values")
            
            # Test correlation clustering
            clusters = self.correlation_analyzer.cluster_by_correlation(price_correlation)
            self.assertIsInstance(clusters, dict, "Correlation clusters not a dictionary")
            
            # Test sector correlation analysis
            sector_mapping = {symbol: f"Sector_{i%3}" for i, symbol in enumerate(self.test_data.keys())}
            sector_corr = self.correlation_analyzer.analyze_sector_correlations(
                prices_df, sector_mapping
            )
            self.assertIsInstance(sector_corr, dict, "Sector correlation not a dictionary")
            
            self.logger.info("Correlation analysis pipeline completed")
            
        except Exception as e:
            self.fail(f"Correlation analysis pipeline failed: {str(e)}")
    
    def test_04_trend_analysis_pipeline(self):
        """Test trend analysis pipeline"""
        self.logger.info("Testing trend analysis pipeline")
        
        for symbol, data in self.test_data.items():
            try:
                # Test trend detection
                trend = self.trend_analyzer.detect_trend(data['Close'])
                
                self.assertIn(trend, ['uptrend', 'downtrend', 'sideways'],
                            f"Invalid trend detected for {symbol}: {trend}")
                
                # Test trend strength
                trend_strength = self.trend_analyzer.calculate_trend_strength(data['Close'])
                
                self.assertIsInstance(trend_strength, (int, float),
                                    f"Invalid trend strength type for {symbol}")
                self.assertGreaterEqual(trend_strength, 0,
                                      f"Negative trend strength for {symbol}")
                
                # Test trend lines
                trend_lines = self.trend_analyzer.calculate_trend_lines(data)
                
                self.assertIn('support_line', trend_lines, f"Missing support line for {symbol}")
                self.assertIn('resistance_line', trend_lines, f"Missing resistance line for {symbol}")
                
                # Test breakout detection
                breakouts = self.trend_analyzer.detect_breakouts(data)
                
                self.assertIsInstance(breakouts, list, f"Breakouts not a list for {symbol}")
                
                # Test trend reversal signals
                reversal_signals = self.trend_analyzer.detect_trend_reversals(data)
                
                self.assertIsInstance(reversal_signals, dict, f"Reversal signals not a dict for {symbol}")
                
                self.logger.info(f"Trend analysis pipeline completed for {symbol}")
                
            except Exception as e:
                self.fail(f"Trend analysis pipeline failed for {symbol}: {str(e)}")
    
    def test_05_volatility_analysis_pipeline(self):
        """Test volatility analysis pipeline"""
        self.logger.info("Testing volatility analysis pipeline")
        
        for symbol, data in self.test_data.items():
            try:
                # Test historical volatility
                hist_vol = self.volatility_analyzer.calculate_historical_volatility(data['Close'])
                
                self.assertIsInstance(hist_vol, (int, float),
                                    f"Invalid historical volatility type for {symbol}")
                self.assertGreater(hist_vol, 0, f"Non-positive volatility for {symbol}")
                
                # Test realized volatility
                realized_vol = self.volatility_analyzer.calculate_realized_volatility(data['Close'])
                
                self.assertIsInstance(realized_vol, (int, float),
                                    f"Invalid realized volatility type for {symbol}")
                
                # Test volatility clustering
                vol_clustering = self.volatility_analyzer.detect_volatility_clustering(data['Close'])
                
                self.assertIsInstance(vol_clustering, dict,
                                    f"Volatility clustering not a dict for {symbol}")
                
                # Test GARCH modeling (if available)
                try:
                    garch_results = self.volatility_analyzer.fit_garch_model(data['Close'])
                    if garch_results is not None:
                        self.assertIn('volatility_forecast', garch_results,
                                    f"Missing GARCH forecast for {symbol}")
                except Exception as e:
                    self.logger.warning(f"GARCH modeling failed for {symbol}: {str(e)}")
                
                # Test volatility regime detection
                vol_regimes = self.volatility_analyzer.detect_volatility_regimes(data['Close'])
                
                self.assertIsInstance(vol_regimes, dict,
                                    f"Volatility regimes not a dict for {symbol}")
                self.assertIn('high_vol_periods', vol_regimes,
                            f"Missing high volatility periods for {symbol}")
                self.assertIn('low_vol_periods', vol_regimes,
                            f"Missing low volatility periods for {symbol}")
                
                # Test volatility surface (if options data available)
                try:
                    vol_surface = self.volatility_analyzer.calculate_volatility_surface(data)
                    if vol_surface is not None:
                        self.assertIsInstance(vol_surface, pd.DataFrame,
                                            f"Invalid volatility surface for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Volatility surface calculation failed for {symbol}: {str(e)}")
                
                self.logger.info(f"Volatility analysis pipeline completed for {symbol}")
                
            except Exception as e:
                self.fail(f"Volatility analysis pipeline failed for {symbol}: {str(e)}")
    
    def test_06_performance_analysis_pipeline(self):
        """Test performance analysis pipeline"""
        self.logger.info("Testing performance analysis pipeline")
        
        for symbol, data in self.test_data.items():
            try:
                # Calculate returns for performance analysis
                returns = data['Close'].pct_change().dropna()
                
                # Test basic performance metrics
                performance_metrics = self.performance_analyzer.calculate_performance_metrics(returns)
                
                required_metrics = [
                    'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
                    'max_drawdown', 'win_rate', 'avg_win', 'avg_loss'
                ]
                
                for metric in required_metrics:
                    self.assertIn(metric, performance_metrics,
                                f"Missing performance metric {metric} for {symbol}")
                    self.assertIsNotNone(performance_metrics[metric],
                                       f"Null performance metric {metric} for {symbol}")
                
                # Test drawdown analysis
                drawdown_analysis = self.performance_analyzer.analyze_drawdowns(data['Close'])
                
                self.assertIn('max_drawdown', drawdown_analysis,
                            f"Missing max drawdown for {symbol}")
                self.assertIn('drawdown_periods', drawdown_analysis,
                            f"Missing drawdown periods for {symbol}")
                self.assertIn('recovery_periods', drawdown_analysis,
                            f"Missing recovery periods for {symbol}")
                
                # Test risk-adjusted returns
                risk_adjusted = self.performance_analyzer.calculate_risk_adjusted_returns(
                    returns, risk_free_rate=0.06  # 6% risk-free rate for India
                )
                
                self.assertIn('sharpe_ratio', risk_adjusted,
                            f"Missing Sharpe ratio for {symbol}")
                self.assertIn('sortino_ratio', risk_adjusted,
                            f"Missing Sortino ratio for {symbol}")
                
                # Test benchmark comparison (using NIFTY 50 as benchmark)
                try:
                    nifty_data = self.market_fetcher.fetch_index_data('NIFTY50', 
                                                                    self.start_date, self.end_date)
                    if nifty_data is not None and not nifty_data.empty:
                        benchmark_comparison = self.performance_analyzer.compare_with_benchmark(
                            returns, nifty_data['Close'].pct_change().dropna()
                        )
                        
                        self.assertIn('alpha', benchmark_comparison,
                                    f"Missing alpha for {symbol}")
                        self.assertIn('beta', benchmark_comparison,
                                    f"Missing beta for {symbol}")
                        self.assertIn('tracking_error', benchmark_comparison,
                                    f"Missing tracking error for {symbol}")
                        
                except Exception as e:
                    self.logger.warning(f"Benchmark comparison failed for {symbol}: {str(e)}")
                
                # Test rolling performance metrics
                rolling_metrics = self.performance_analyzer.calculate_rolling_metrics(
                    returns, window=30
                )
                
                self.assertIn('rolling_sharpe', rolling_metrics,
                            f"Missing rolling Sharpe for {symbol}")
                self.assertIn('rolling_volatility', rolling_metrics,
                            f"Missing rolling volatility for {symbol}")
                
                self.logger.info(f"Performance analysis pipeline completed for {symbol}")
                
            except Exception as e:
                self.fail(f"Performance analysis pipeline failed for {symbol}: {str(e)}")
    
    def test_07_integrated_market_analysis(self):
        """Test integrated market analysis combining all components"""
        self.logger.info("Testing integrated market analysis")
        
        try:
            # Select primary symbol for comprehensive analysis
            primary_symbol = list(self.test_data.keys())[0]
            primary_data = self.test_data[primary_symbol]
            
            # Step 1: Add technical indicators
            enhanced_data = self.technical_indicators.calculate_all_indicators(primary_data)
            
            # Step 2: Perform comprehensive market analysis
            market_analysis = self.market_analyzer.perform_comprehensive_analysis(enhanced_data)
            
            # Verify comprehensive analysis structure
            self.assertIn('basic_statistics', market_analysis,
                        "Missing basic statistics in comprehensive analysis")
            self.assertIn('price_analysis', market_analysis,
                        "Missing price analysis in comprehensive analysis")
            self.assertIn('technical_signals', market_analysis,
                        "Missing technical signals in comprehensive analysis")
            
            # Step 3: Generate trading signals
            signals = self.market_analyzer.generate_trading_signals(enhanced_data)
            
            self.assertIsInstance(signals, dict, "Trading signals not a dictionary")
            self.assertIn('current_signal', signals, "Missing current signal")
            self.assertIn('signal_strength', signals, "Missing signal strength")
            self.assertIn('confidence', signals, "Missing signal confidence")
            
            # Validate signal values
            self.assertIn(signals['current_signal'], ['BUY', 'SELL', 'HOLD'],
                        f"Invalid signal: {signals['current_signal']}")
            self.assertGreaterEqual(signals['signal_strength'], 0,
                                  "Negative signal strength")
            self.assertLessEqual(signals['signal_strength'], 100,
                               "Signal strength exceeds 100")
            
            # Step 4: Risk assessment
            risk_assessment = self.market_analyzer.assess_risk_factors(enhanced_data)
            
            self.assertIn('risk_level', risk_assessment, "Missing risk level")
            self.assertIn('risk_factors', risk_assessment, "Missing risk factors")
            self.assertIn('risk_score', risk_assessment, "Missing risk score")
            
            # Step 5: Generate market outlook  
            market_outlook = self.market_analyzer.generate_market_outlook(enhanced_data)
            
            self.assertIn('short_term_outlook', market_outlook, "Missing short-term outlook")
            self.assertIn('medium_term_outlook', market_outlook, "Missing medium-term outlook")
            self.assertIn('key_levels', market_outlook, "Missing key levels")
            
            self.logger.info("Integrated market analysis completed successfully")
            
        except Exception as e:
            self.fail(f"Integrated market analysis failed: {str(e)}")
    
    def test_08_indian_market_specific_analysis(self):
        """Test Indian market specific analysis features"""
        self.logger.info("Testing Indian market specific analysis")
        
        try:
            # Test sector-wise analysis for Indian market
            indian_sectors = {
                'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS'],
                'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS'],
                'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS']
            }
            
            sector_analysis = {}
            for sector, symbols in indian_sectors.items():
                sector_data = []
                for symbol in symbols:
                    if symbol in self.test_data:
                        sector_data.append(self.test_data[symbol]['Close'])
                
                if sector_data:
                    # Calculate sector average
                    sector_df = pd.DataFrame({f'{sector}_{i}': data for i, data in enumerate(sector_data)})
                    sector_avg = sector_df.mean(axis=1)
                    
                    # Analyze sector performance
                    sector_performance = self.performance_analyzer.calculate_performance_metrics(
                        sector_avg.pct_change().dropna()
                    )
                    
                    sector_analysis[sector] = {
                        'performance': sector_performance,
                        'trend': self.trend_analyzer.detect_trend(sector_avg),
                        'volatility': self.volatility_analyzer.calculate_historical_volatility(sector_avg)
                    }
            
            # Test FII/DII impact analysis (simulated)
            try:
                fii_impact = self.market_analyzer.analyze_fii_dii_impact(primary_data)
                if fii_impact is not None:
                    self.assertIsInstance(fii_impact, dict, "FII impact analysis not a dictionary")
            except AttributeError:
                self.logger.info("FII/DII impact analysis not implemented in basic version")
            
            # Test currency impact analysis (USD/INR)
            try:
                currency_impact = self.market_analyzer.analyze_currency_impact(primary_data)
                if currency_impact is not None:
                    self.assertIsInstance(currency_impact, dict, "Currency impact analysis not a dictionary")
            except AttributeError:
                self.logger.info("Currency impact analysis not implemented in basic version")
            
            # Test holiday and event calendar impact
            indian_market_events = [
                '2022-01-26',  # Republic Day
                '2022-03-18',  # Holi
                '2022-04-14',  # Ram Navami
                '2022-08-15',  # Independence Day
                '2022-10-05',  # Dussehra
                '2022-11-14',  # Diwali
            ]
            
            event_impact = self.market_analyzer.analyze_event_impact(
                primary_data, indian_market_events
            )
            
            self.assertIsInstance(event_impact, dict, "Event impact analysis not a dictionary")
            
            self.logger.info("Indian market specific analysis completed")
            
        except Exception as e:
            self.fail(f"Indian market specific analysis failed: {str(e)}")
    
    def test_09_data_quality_and_validation(self):
        """Test data quality and validation throughout the pipeline"""
        self.logger.info("Testing data quality and validation")
        
        for symbol, data in self.test_data.items():
            try:
                # Test data completeness
                self.assertGreater(len(data), 50, f"Insufficient data for {symbol}")
                
                # Test for required columns
                required_columns = ['Open', 'High', 'Low', 'Close']
                for col in required_columns:
                    self.assertIn(col, data.columns, f"Missing column {col} for {symbol}")
                
                # Test data consistency
                self.assertTrue(all(data['High'] >= data['Low']), 
                              f"High < Low inconsistency in {symbol}")
                self.assertTrue(all(data['High'] >= data['Open']), 
                              f"High < Open inconsistency in {symbol}")
                self.assertTrue(all(data['High'] >= data['Close']), 
                              f"High < Close inconsistency in {symbol}")
                self.assertTrue(all(data['Low'] <= data['Open']), 
                              f"Low > Open inconsistency in {symbol}")
                self.assertTrue(all(data['Low'] <= data['Close']), 
                              f"Low > Close inconsistency in {symbol}")
                
                # Test for outliers
                returns = data['Close'].pct_change().dropna()
                outlier_threshold = 3 * returns.std()
                outliers = returns[abs(returns) > outlier_threshold]
                
                outlier_percentage = len(outliers) / len(returns) * 100
                self.assertLess(outlier_percentage, 5, 
                              f"Too many outliers ({outlier_percentage:.2f}%) in {symbol}")
                
                # Test data freshness (for Version 1, we expect historical data)
                latest_date = data.index[-1]
                expected_end = pd.to_datetime(self.end_date)
                date_diff = abs((latest_date - expected_end).days)
                self.assertLess(date_diff, 10, 
                              f"Data not fresh enough for {symbol}")
                
                self.logger.info(f"Data quality validation passed for {symbol}")
                
            except Exception as e:
                self.fail(f"Data quality validation failed for {symbol}: {str(e)}")
    
    def test_10_pipeline_performance_and_timing(self):
        """Test pipeline performance and execution timing"""
        self.logger.info("Testing pipeline performance and timing")
        
        import time
        
        try:
            primary_symbol = list(self.test_data.keys())[0]
            primary_data = self.test_data[primary_symbol]
            
            # Time technical indicators calculation
            start_time = time.time()
            enhanced_data = self.technical_indicators.calculate_all_indicators(primary_data)
            technical_time = time.time() - start_time
            
            self.assertLess(technical_time, 30, 
                          f"Technical indicators calculation too slow: {technical_time:.2f}s")
            
            # Time market analysis
            start_time = time.time()
            market_analysis = self.market_analyzer.perform_comprehensive_analysis(enhanced_data)
            analysis_time = time.time() - start_time
            
            self.assertLess(analysis_time, 60, 
                          f"Market analysis too slow: {analysis_time:.2f}s")
            
            # Time correlation analysis (if multiple symbols)
            if len(self.test_data) > 1:
                price_data = {symbol: data['Close'] for symbol, data in self.test_data.items()}
                prices_df = pd.DataFrame(price_data).dropna()
                
                start_time = time.time()
                correlation_matrix = self.correlation_analyzer.calculate_price_correlation(prices_df)
                correlation_time = time.time() - start_time
                
                self.assertLess(correlation_time, 30, 
                              f"Correlation analysis too slow: {correlation_time:.2f}s")
            
            # Memory usage check
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            self.assertLess(memory_usage, 1000, 
                          f"Memory usage too high: {memory_usage:.2f}MB")
            
            self.logger.info(f"Pipeline performance test completed - Memory: {memory_usage:.2f}MB")
            
        except Exception as e:
            self.fail(f"Pipeline performance test failed: {str(e)}")


def run_analysis_pipeline_tests():
    """Run all analysis pipeline tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    # Set up test environment
    import logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run tests
    run_analysis_pipeline_tests()