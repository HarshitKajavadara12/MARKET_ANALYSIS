#!/usr/bin/env python3
"""
Sector Analysis Script for Indian Stock Market
Market Research System v1.0 (2022)

This script performs comprehensive sector analysis for Indian stock market
including sector performance, weightage analysis, and trend identification.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.fetch_market_data import IndianStockDataFetcher
from analysis.technical_indicators import TechnicalIndicators
from utils.date_utils import get_trading_days
from utils.logging_utils import setup_logger
from reporting.visualization import SectorVisualizer
from reporting.pdf_generator import SectorReportGenerator

class IndianSectorAnalyzer:
    """
    Comprehensive sector analysis for Indian stock market
    """
    
    def __init__(self, data_dir="../data", report_dir="../reports"):
        """
        Initialize sector analyzer
        
        Args:
            data_dir (str): Path to data directory
            report_dir (str): Path to reports directory
        """
        self.data_dir = data_dir
        self.report_dir = report_dir
        self.logger = setup_logger("sector_analysis")
        
        # Indian stock market sectors and representative stocks
        self.indian_sectors = {
            'Banking': {
                'stocks': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 
                          'SBIN.NS', 'INDUSINDBK.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS'],
                'index': '^NSEBANK'
            },
            'IT': {
                'stocks': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 
                          'TECHM.NS', 'LTI.NS', 'MPHASIS.NS', 'COFORGE.NS'],
                'index': 'NIFTY_IT.NS'
            },
            'Pharma': {
                'stocks': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS',
                          'BIOCON.NS', 'LUPIN.NS', 'ALKEM.NS', 'TORNTPHARM.NS'],
                'index': 'NIFTY_PHARMA.NS'
            },
            'Auto': {
                'stocks': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS',
                          'HEROMOTOCO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS'],
                'index': 'NIFTY_AUTO.NS'
            },
            'FMCG': {
                'stocks': ['HINDUNILVR.NS', 'NESTLEIND.NS', 'ITC.NS', 'BRITANNIA.NS',
                          'DABUR.NS', 'MARICO.NS', 'GODREJCP.NS', 'COLPAL.NS'],
                'index': 'NIFTY_FMCG.NS'
            },
            'Metals': {
                'stocks': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS',
                          'SAIL.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'MOIL.NS'],
                'index': 'NIFTY_METAL.NS'
            },
            'Energy': {
                'stocks': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS',
                          'HINDPETRO.NS', 'GAIL.NS', 'OIL.NS', 'MRPL.NS'],
                'index': 'NIFTY_ENERGY.NS'
            },
            'Realty': {
                'stocks': ['DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS',
                          'BRIGADE.NS', 'SOBHA.NS', 'PHOENLIX.NS', 'MAHLIFE.NS'],
                'index': 'NIFTY_REALTY.NS'
            },
            'Media': {
                'stocks': ['ZEEL.NS', 'SUNTV.NS', 'TV18BRDCST.NS', 'NETWORK18.NS',
                          'DISHTV.NS', 'TVTODAY.NS', 'JAGRAN.NS', 'DBCORP.NS'],
                'index': 'NIFTY_MEDIA.NS'
            },
            'PSU': {
                'stocks': ['NTPC.NS', 'POWERGRID.NS', 'COALINDIA.NS', 'BHEL.NS',
                          'GAIL.NS', 'SAIL.NS', 'NMDC.NS', 'RECLTD.NS'],
                'index': 'NIFTY_PSE.NS'
            }
        }
        
        # Benchmark indices
        self.benchmark_indices = {
            'NIFTY50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY_NEXT50': 'NIFTYJR.NS',
            'NIFTY_MIDCAP': 'NIFTY_MID_SELECT.NS',
            'NIFTY_SMALLCAP': 'NIFTY_SMLCP_100.NS'
        }
        
        self.data_fetcher = IndianStockDataFetcher()
        self.indicators = TechnicalIndicators()
        self.visualizer = SectorVisualizer()
        self.report_generator = SectorReportGenerator()
        
    def fetch_sector_data(self, period="1y"):
        """
        Fetch data for all sectors and stocks
        
        Args:
            period (str): Time period for data ('1y', '6mo', '3mo', '1mo')
            
        Returns:
            dict: Sector data with stock prices and indices
        """
        self.logger.info(f"Fetching sector data for period: {period}")
        sector_data = {}
        
        for sector_name, sector_info in self.indian_sectors.items():
            self.logger.info(f"Fetching data for {sector_name} sector")
            sector_data[sector_name] = {
                'stocks': {},
                'index': None
            }
            
            # Fetch individual stock data
            for stock in sector_info['stocks']:
                try:
                    stock_data = self.data_fetcher.get_stock_data(stock, period=period)
                    if stock_data is not None and not stock_data.empty:
                        sector_data[sector_name]['stocks'][stock] = stock_data
                        self.logger.debug(f"Successfully fetched data for {stock}")
                    else:
                        self.logger.warning(f"No data available for {stock}")
                except Exception as e:
                    self.logger.error(f"Error fetching data for {stock}: {str(e)}")
            
            # Fetch sector index data
            try:
                index_data = self.data_fetcher.get_stock_data(sector_info['index'], period=period)
                if index_data is not None and not index_data.empty:
                    sector_data[sector_name]['index'] = index_data
                    self.logger.debug(f"Successfully fetched index data for {sector_name}")
                else:
                    self.logger.warning(f"No index data available for {sector_name}")
            except Exception as e:
                self.logger.error(f"Error fetching index data for {sector_name}: {str(e)}")
        
        # Fetch benchmark data
        sector_data['benchmarks'] = {}
        for benchmark_name, benchmark_symbol in self.benchmark_indices.items():
            try:
                benchmark_data = self.data_fetcher.get_stock_data(benchmark_symbol, period=period)
                if benchmark_data is not None and not benchmark_data.empty:
                    sector_data['benchmarks'][benchmark_name] = benchmark_data
                    self.logger.debug(f"Successfully fetched benchmark data for {benchmark_name}")
            except Exception as e:
                self.logger.error(f"Error fetching benchmark data for {benchmark_name}: {str(e)}")
        
        return sector_data
    
    def calculate_sector_performance(self, sector_data):
        """
        Calculate performance metrics for each sector
        
        Args:
            sector_data (dict): Sector data from fetch_sector_data
            
        Returns:
            dict: Performance metrics for each sector
        """
        self.logger.info("Calculating sector performance metrics")
        performance_metrics = {}
        
        for sector_name, sector_info in sector_data.items():
            if sector_name == 'benchmarks':
                continue
                
            self.logger.info(f"Calculating performance for {sector_name}")
            
            # Calculate individual stock performance
            stock_performances = {}
            sector_returns = []
            
            for stock_symbol, stock_data in sector_info['stocks'].items():
                if stock_data is None or stock_data.empty:
                    continue
                    
                # Calculate returns
                daily_returns = stock_data['Close'].pct_change().dropna()
                
                # Performance metrics
                total_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
                volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
                
                # Risk-adjusted returns
                sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0
                
                # Maximum drawdown
                cumulative_returns = (1 + daily_returns).cumprod()
                peak = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns / peak - 1) * 100
                max_drawdown = drawdown.min()
                
                stock_performances[stock_symbol] = {
                    'total_return': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'current_price': stock_data['Close'].iloc[-1],
                    'daily_returns': daily_returns
                }
                
                sector_returns.extend(daily_returns.tolist())
            
            # Calculate sector-wide metrics
            if sector_returns:
                sector_returns = pd.Series(sector_returns)
                sector_total_return = np.mean([perf['total_return'] for perf in stock_performances.values()])
                sector_volatility = np.mean([perf['volatility'] for perf in stock_performances.values()])
                sector_sharpe = np.mean([perf['sharpe_ratio'] for perf in stock_performances.values()])
                sector_max_dd = np.mean([perf['max_drawdown'] for perf in stock_performances.values()])
            else:
                sector_total_return = 0
                sector_volatility = 0
                sector_sharpe = 0
                sector_max_dd = 0
            
            performance_metrics[sector_name] = {
                'sector_metrics': {
                    'total_return': sector_total_return,
                    'volatility': sector_volatility,
                    'sharpe_ratio': sector_sharpe,
                    'max_drawdown': sector_max_dd,
                    'stock_count': len(stock_performances)
                },
                'stock_performances': stock_performances,
                'best_performer': max(stock_performances.items(), key=lambda x: x[1]['total_return']) if stock_performances else None,
                'worst_performer': min(stock_performances.items(), key=lambda x: x[1]['total_return']) if stock_performances else None
            }
        
        return performance_metrics
    
    def analyze_sector_correlations(self, sector_data):
        """
        Analyze correlations between sectors
        
        Args:
            sector_data (dict): Sector data from fetch_sector_data
            
        Returns:
            pd.DataFrame: Correlation matrix between sectors
        """
        self.logger.info("Analyzing sector correlations")
        
        sector_returns = {}
        
        for sector_name, sector_info in sector_data.items():
            if sector_name == 'benchmarks':
                continue
                
            if 'index' in sector_info and sector_info['index'] is not None:
                # Use sector index returns
                daily_returns = sector_info['index']['Close'].pct_change().dropna()
                sector_returns[sector_name] = daily_returns
            else:
                # Calculate average sector returns from individual stocks
                all_stock_returns = []
                for stock_data in sector_info['stocks'].values():
                    if stock_data is not None and not stock_data.empty:
                        stock_returns = stock_data['Close'].pct_change().dropna()
                        all_stock_returns.append(stock_returns)
                
                if all_stock_returns:
                    # Align all returns to common dates
                    aligned_returns = pd.concat(all_stock_returns, axis=1).mean(axis=1)
                    sector_returns[sector_name] = aligned_returns
        
        # Create correlation matrix
        if sector_returns:
            returns_df = pd.DataFrame(sector_returns)
            correlation_matrix = returns_df.corr()
            return correlation_matrix
        else:
            return pd.DataFrame()
    
    def generate_sector_insights(self, performance_metrics, correlation_matrix):
        """
        Generate key insights from sector analysis
        
        Args:
            performance_metrics (dict): Performance metrics from calculate_sector_performance
            correlation_matrix (pd.DataFrame): Correlation matrix from analyze_sector_correlations
            
        Returns:
            dict: Key insights and recommendations
        """
        self.logger.info("Generating sector insights")
        
        insights = {
            'top_performers': [],
            'underperformers': [],
            'high_volatility_sectors': [],
            'defensive_sectors': [],
            'correlation_insights': [],
            'risk_opportunities': []
        }
        
        # Rank sectors by performance
        sector_returns = [(sector, metrics['sector_metrics']['total_return']) 
                         for sector, metrics in performance_metrics.items()]
        sector_returns.sort(key=lambda x: x[1], reverse=True)
        
        # Top and bottom performers
        insights['top_performers'] = sector_returns[:3]
        insights['underperformers'] = sector_returns[-3:]
        
        # Volatility analysis
        sector_volatility = [(sector, metrics['sector_metrics']['volatility']) 
                           for sector, metrics in performance_metrics.items()]
        sector_volatility.sort(key=lambda x: x[1], reverse=True)
        
        insights['high_volatility_sectors'] = sector_volatility[:3]
        insights['defensive_sectors'] = sector_volatility[-3:]
        
        # Correlation insights
        if not correlation_matrix.empty:
            # Find highly correlated sectors
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # High correlation threshold
                        high_corr_pairs.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            corr_value
                        ))
            
            insights['correlation_insights'] = high_corr_pairs
        
        # Risk-opportunity analysis
        for sector, metrics in performance_metrics.items():
            sector_metrics = metrics['sector_metrics']
            risk_return_ratio = sector_metrics['total_return'] / sector_metrics['volatility'] if sector_metrics['volatility'] != 0 else 0
            
            insights['risk_opportunities'].append({
                'sector': sector,
                'return': sector_metrics['total_return'],
                'volatility': sector_metrics['volatility'],
                'risk_return_ratio': risk_return_ratio,
                'sharpe_ratio': sector_metrics['sharpe_ratio']
            })
        
        # Sort by risk-return ratio
        insights['risk_opportunities'].sort(key=lambda x: x['risk_return_ratio'], reverse=True)
        
        return insights
    
    def save_analysis_results(self, performance_metrics, correlation_matrix, insights):
        """
        Save analysis results to files
        
        Args:
            performance_metrics (dict): Performance metrics
            correlation_matrix (pd.DataFrame): Correlation matrix
            insights (dict): Generated insights
        """
        self.logger.info("Saving analysis results")
        
        # Create output directory
        output_dir = os.path.join(self.report_dir, 'sector_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save performance metrics
        performance_file = os.path.join(output_dir, f'sector_performance_{datetime.now().strftime("%Y%m%d")}.csv')
        
        # Convert performance metrics to DataFrame
        perf_data = []
        for sector, metrics in performance_metrics.items():
            sector_metrics = metrics['sector_metrics']
            perf_data.append({
                'Sector': sector,
                'Total_Return_%': sector_metrics['total_return'],
                'Volatility_%': sector_metrics['volatility'],
                'Sharpe_Ratio': sector_metrics['sharpe_ratio'],
                'Max_Drawdown_%': sector_metrics['max_drawdown'],
                'Stock_Count': sector_metrics['stock_count']
            })
        
        performance_df = pd.DataFrame(perf_data)
        performance_df.to_csv(performance_file, index=False)
        
        # Save correlation matrix
        if not correlation_matrix.empty:
            corr_file = os.path.join(output_dir, f'sector_correlations_{datetime.now().strftime("%Y%m%d")}.csv')
            correlation_matrix.to_csv(corr_file)
        
        # Save insights as JSON
        import json
        insights_file = os.path.join(output_dir, f'sector_insights_{datetime.now().strftime("%Y%m%d")}.json')
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean insights for JSON serialization
        clean_insights = {}
        for key, value in insights.items():
            if isinstance(value, list):
                clean_insights[key] = [convert_numpy(item) for item in value]
            else:
                clean_insights[key] = convert_numpy(value)
        
        with open(insights_file, 'w') as f:
            json.dump(clean_insights, f, indent=2, default=str)
        
        self.logger.info(f"Analysis results saved to {output_dir}")
    
    def run_complete_analysis(self, period="1y", save_results=True):
        """
        Run complete sector analysis pipeline
        
        Args:
            period (str): Time period for analysis
            save_results (bool): Whether to save results to files
            
        Returns:
            dict: Complete analysis results
        """
        self.logger.info("Starting complete sector analysis")
        
        try:
            # Fetch data
            sector_data = self.fetch_sector_data(period=period)
            
            # Calculate performance metrics
            performance_metrics = self.calculate_sector_performance(sector_data)
            
            # Analyze correlations
            correlation_matrix = self.analyze_sector_correlations(sector_data)
            
            # Generate insights
            insights = self.generate_sector_insights(performance_metrics, correlation_matrix)
            
            # Save results if requested
            if save_results:
                self.save_analysis_results(performance_metrics, correlation_matrix, insights)
            
            # Create visualizations
            self.create_visualizations(performance_metrics, correlation_matrix, insights)
            
            analysis_results = {
                'performance_metrics': performance_metrics,
                'correlation_matrix': correlation_matrix,
                'insights': insights,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.logger.info("Sector analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in sector analysis: {str(e)}")
            raise
    
    def create_visualizations(self, performance_metrics, correlation_matrix, insights):
        """
        Create visualization charts for sector analysis
        
        Args:
            performance_metrics (dict): Performance metrics
            correlation_matrix (pd.DataFrame): Correlation matrix  
            insights (dict): Analysis insights
        """
        self.logger.info("Creating sector analysis visualizations")
        
        # Create output directory for charts
        chart_dir = os.path.join(self.report_dir, 'sector_analysis', 'charts')
        os.makedirs(chart_dir, exist_ok=True)
        
        # 1. Sector Performance Bar Chart
        sectors = list(performance_metrics.keys())
        returns = [performance_metrics[sector]['sector_metrics']['total_return'] for sector in sectors]
        
        plt.figure(figsize=(12, 8))
        colors = ['green' if r > 0 else 'red' for r in returns]
        bars = plt.bar(sectors, returns, color=colors, alpha=0.7)
        plt.title('Sector Performance - Total Returns (%)', fontsize=16, fontweight='bold')
        plt.xlabel('Sectors', fontsize=12)
        plt.ylabel('Total Return (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1.5),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, 'sector_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Risk-Return Scatter Plot
        plt.figure(figsize=(12, 8))
        volatilities = [performance_metrics[sector]['sector_metrics']['volatility'] for sector in sectors]
        
        scatter = plt.scatter(volatilities, returns, s=100, alpha=0.7, c=range(len(sectors)), cmap='viridis')
        
        for i, sector in enumerate(sectors):
            plt.annotate(sector, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.title('Risk-Return Analysis by Sector', fontsize=16, fontweight='bold')
        plt.xlabel('Volatility (%)', fontsize=12)
        plt.ylabel('Total Return (%)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=np.mean(volatilities), color='red', linestyle='--', alpha=0.5, label='Avg Volatility')
        plt.axhline(y=np.mean(returns), color='blue', linestyle='--', alpha=0.5, label='Avg Return')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, 'risk_return_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation Heatmap
        if not correlation_matrix.empty:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            
            plt.title('Sector Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, 'sector_correlations.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Sharpe Ratio Comparison
        sharpe_ratios = [performance_metrics[sector]['sector_metrics']['sharpe_ratio'] for sector in sectors]
        
        plt.figure(figsize=(12, 8))
        colors = ['green' if s > 0 else 'red' for s in sharpe_ratios]
        bars = plt.bar(sectors, sharpe_ratios, color=colors, alpha=0.7)
        plt.title('Sector Sharpe Ratios', fontsize=16, fontweight='bold')
        plt.xlabel('Sectors', fontsize=12)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_ratios):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height > 0 else -0.05),
                    f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, 'sharpe_ratios.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {chart_dir}")


def main():
    """
    Main execution function for sector analysis
    """
    print("=" * 60)
    print("Indian Stock Market Sector Analysis v1.0 (2022)")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = IndianSectorAnalyzer()
        
        # Run complete analysis
        print("\nStarting comprehensive sector analysis...")
        results = analyzer.run_complete_analysis(period="1y", save_results=True)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SECTOR ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Top performers
        print("\n🏆 TOP PERFORMING SECTORS:")
        for i, (sector, return_pct) in enumerate(results['insights']['top_performers'], 1):
            print(f"{i}. {sector}: {return_pct:.2f}%")
        
        # Underperformers
        print("\n📉 UNDERPERFORMING SECTORS:")
        for i, (sector, return_pct) in enumerate(results['insights']['underperformers'], 1):
            print(f"{i}. {sector}: {return_pct:.2f}%")
        
        # High volatility
        print("\n⚡ HIGH VOLATILITY SECTORS:")
        for i, (sector, volatility) in enumerate(results['insights']['high_volatility_sectors'], 1):
            print(f"{i}. {sector}: {volatility:.2f}%")
        
        # Defensive sectors
        print("\n🛡️ DEFENSIVE SECTORS (Low Volatility):")
        for i, (sector, volatility) in enumerate(results['insights']['defensive_sectors'], 1):
            print(f"{i}. {sector}: {volatility:.2f}%")
        
        # Risk-return leaders
        print("\n💎 BEST RISK-ADJUSTED RETURNS:")
        for i, opportunity in enumerate(results['insights']['risk_opportunities'][:3], 1):
            print(f"{i}. {opportunity['sector']}: Return {opportunity['return']:.2f}%, "
                  f"Volatility {opportunity['volatility']:.2f}%, "
                  f"Sharpe {opportunity['sharpe_ratio']:.2f}")
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("Reports and charts saved to reports/sector_analysis/")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error in sector analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()