"""
Table Generation Utilities for Market Research System v1.0
Generates professional tables for reports with various formatting options
Compatible with Indian stock market data and reporting requirements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableGenerator:
    """
    Professional table generator for market research reports
    Supports various table types for Indian stock market analysis
    """
    
    def __init__(self, style: str = "professional"):
        """
        Initialize table generator with styling options
        
        Args:
            style (str): Table styling - 'professional', 'minimal', 'colorful'
        """
        self.style = style
        self.default_formats = {
            'price': '₹{:.2f}',
            'percentage': '{:.2f}%',
            'volume': '{:,.0f}',
            'market_cap': '₹{:,.0f} Cr',
            'ratio': '{:.2f}',
            'currency': '₹{:,.2f}'
        }
        
    def create_stock_performance_table(self, 
                                     stock_data: pd.DataFrame,
                                     metrics: List[str] = None) -> str:
        """
        Create stock performance summary table
        
        Args:
            stock_data (pd.DataFrame): Stock data with OHLCV
            metrics (List[str]): Metrics to include in table
            
        Returns:
            str: HTML table string
        """
        try:
            if metrics is None:
                metrics = ['Symbol', 'Current_Price', 'Change', 'Change_%', 
                          'Volume', 'High_52W', 'Low_52W', 'Market_Cap']
            
            # Sample data structure for Indian stocks
            if stock_data.empty:
                stock_data = self._generate_sample_stock_data()
            
            table_html = self._create_html_table(
                data=stock_data,
                title="Stock Performance Summary",
                columns=metrics,
                formats={
                    'Current_Price': self.default_formats['price'],
                    'Change': self.default_formats['price'],
                    'Change_%': self.default_formats['percentage'],
                    'Volume': self.default_formats['volume'],
                    'High_52W': self.default_formats['price'],
                    'Low_52W': self.default_formats['price'],
                    'Market_Cap': self.default_formats['market_cap']
                }
            )
            
            logger.info("Stock performance table generated successfully")
            return table_html
            
        except Exception as e:
            logger.error(f"Error creating stock performance table: {str(e)}")
            return self._create_error_table("Stock Performance", str(e))
    
    def create_sector_analysis_table(self, 
                                   sector_data: pd.DataFrame,
                                   sort_by: str = 'Performance') -> str:
        """
        Create sector-wise analysis table
        
        Args:
            sector_data (pd.DataFrame): Sector performance data
            sort_by (str): Column to sort by
            
        Returns:
            str: HTML table string
        """
        try:
            if sector_data.empty:
                sector_data = self._generate_sample_sector_data()
            
            # Sort data
            if sort_by in sector_data.columns:
                sector_data = sector_data.sort_values(by=sort_by, ascending=False)
            
            table_html = self._create_html_table(
                data=sector_data,
                title="Sector Performance Analysis",
                columns=['Sector', 'Performance', 'Market_Cap', 'P/E_Ratio', 
                        'Volume', 'Top_Performer'],
                formats={
                    'Performance': self.default_formats['percentage'],
                    'Market_Cap': self.default_formats['market_cap'],
                    'P/E_Ratio': self.default_formats['ratio'],
                    'Volume': self.default_formats['volume']
                }
            )
            
            logger.info("Sector analysis table generated successfully")
            return table_html
            
        except Exception as e:
            logger.error(f"Error creating sector analysis table: {str(e)}")
            return self._create_error_table("Sector Analysis", str(e))
    
    def create_technical_indicators_table(self, 
                                        indicator_data: pd.DataFrame,
                                        indicators: List[str] = None) -> str:
        """
        Create technical indicators summary table
        
        Args:
            indicator_data (pd.DataFrame): Technical indicators data
            indicators (List[str]): List of indicators to include
            
        Returns:
            str: HTML table string
        """
        try:
            if indicators is None:
                indicators = ['Symbol', 'RSI', 'MACD', 'SMA_20', 'SMA_50', 
                             'Bollinger_Upper', 'Bollinger_Lower', 'Signal']
            
            if indicator_data.empty:
                indicator_data = self._generate_sample_technical_data()
            
            table_html = self._create_html_table(
                data=indicator_data,
                title="Technical Indicators Summary",
                columns=indicators,
                formats={
                    'RSI': self.default_formats['ratio'],
                    'MACD': self.default_formats['ratio'],
                    'SMA_20': self.default_formats['price'],
                    'SMA_50': self.default_formats['price'],
                    'Bollinger_Upper': self.default_formats['price'],
                    'Bollinger_Lower': self.default_formats['price']
                }
            )
            
            logger.info("Technical indicators table generated successfully")
            return table_html
            
        except Exception as e:
            logger.error(f"Error creating technical indicators table: {str(e)}")
            return self._create_error_table("Technical Indicators", str(e))
    
    def create_portfolio_summary_table(self, 
                                     portfolio_data: pd.DataFrame,
                                     total_value: float = None) -> str:
        """
        Create portfolio summary table
        
        Args:
            portfolio_data (pd.DataFrame): Portfolio holdings data
            total_value (float): Total portfolio value
            
        Returns:
            str: HTML table string
        """
        try:
            if portfolio_data.empty:
                portfolio_data = self._generate_sample_portfolio_data()
            
            if total_value is None:
                total_value = portfolio_data['Current_Value'].sum()
            
            # Calculate allocation percentages
            portfolio_data['Allocation_%'] = (
                portfolio_data['Current_Value'] / total_value * 100
            )
            
            table_html = self._create_html_table(
                data=portfolio_data,
                title=f"Portfolio Summary (Total Value: ₹{total_value:,.2f})",
                columns=['Symbol', 'Quantity', 'Avg_Price', 'Current_Price', 
                        'Current_Value', 'P&L', 'P&L_%', 'Allocation_%'],
                formats={
                    'Avg_Price': self.default_formats['price'],
                    'Current_Price': self.default_formats['price'],
                    'Current_Value': self.default_formats['currency'],
                    'P&L': self.default_formats['currency'],
                    'P&L_%': self.default_formats['percentage'],
                    'Allocation_%': self.default_formats['percentage']
                }
            )
            
            logger.info("Portfolio summary table generated successfully")
            return table_html
            
        except Exception as e:
            logger.error(f"Error creating portfolio summary table: {str(e)}")
            return self._create_error_table("Portfolio Summary", str(e))
    
    def create_market_summary_table(self, 
                                  market_data: Dict[str, Any],
                                  date: datetime = None) -> str:
        """
        Create daily market summary table
        
        Args:
            market_data (Dict): Market indices and summary data
            date (datetime): Date for the summary
            
        Returns:
            str: HTML table string
        """
        try:
            if date is None:
                date = datetime.now()
            
            if not market_data:
                market_data = self._generate_sample_market_data()
            
            # Convert dict to DataFrame
            df = pd.DataFrame(list(market_data.items()), 
                            columns=['Index', 'Value'])
            
            table_html = self._create_html_table(
                data=df,
                title=f"Market Summary - {date.strftime('%Y-%m-%d')}",
                columns=['Index', 'Value'],
                formats={'Value': self.default_formats['currency']}
            )
            
            logger.info("Market summary table generated successfully")
            return table_html
            
        except Exception as e:
            logger.error(f"Error creating market summary table: {str(e)}")
            return self._create_error_table("Market Summary", str(e))
    
    def _create_html_table(self, 
                          data: pd.DataFrame,
                          title: str,
                          columns: List[str],
                          formats: Dict[str, str] = None) -> str:
        """
        Create HTML table with professional styling
        
        Args:
            data (pd.DataFrame): Data to display
            title (str): Table title
            columns (List[str]): Columns to include
            formats (Dict[str, str]): Format strings for columns
            
        Returns:
            str: HTML table string
        """
        try:
            # Filter columns that exist in data
            available_columns = [col for col in columns if col in data.columns]
            table_data = data[available_columns].copy()
            
            # Apply formatting
            if formats:
                for col, fmt in formats.items():
                    if col in table_data.columns:
                        if pd.api.types.is_numeric_dtype(table_data[col]):
                            table_data[col] = table_data[col].apply(
                                lambda x: fmt.format(x) if pd.notna(x) else 'N/A'
                            )
            
            # Create HTML
            html_parts = [
                f'<div class="table-container">',
                f'<h3 class="table-title">{title}</h3>',
                '<table class="market-table">',
                '<thead><tr>'
            ]
            
            # Add headers
            for col in available_columns:
                html_parts.append(f'<th>{col.replace("_", " ")}</th>')
            html_parts.append('</tr></thead><tbody>')
            
            # Add data rows
            for _, row in table_data.iterrows():
                html_parts.append('<tr>')
                for col in available_columns:
                    value = row[col] if pd.notna(row[col]) else 'N/A'
                    html_parts.append(f'<td>{value}</td>')
                html_parts.append('</tr>')
            
            html_parts.extend(['</tbody></table></div>'])
            
            return '\n'.join(html_parts)
            
        except Exception as e:
            logger.error(f"Error creating HTML table: {str(e)}")
            return f'<div class="error">Error creating table: {str(e)}</div>'
    
    def _create_error_table(self, table_type: str, error_msg: str) -> str:
        """Create error message table"""
        return f'''
        <div class="error-table">
            <h3>Error: {table_type} Table</h3>
            <p>Unable to generate table: {error_msg}</p>
            <p>Please check data source and try again.</p>
        </div>
        '''
    
    def _generate_sample_stock_data(self) -> pd.DataFrame:
        """Generate sample Indian stock data for testing"""
        indian_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 
                        'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'HDFCBANK']
        
        data = []
        for stock in indian_stocks:
            base_price = np.random.uniform(100, 3000)
            change = np.random.uniform(-5, 5)
            data.append({
                'Symbol': stock,
                'Current_Price': base_price,
                'Change': change,
                'Change_%': (change / base_price) * 100,
                'Volume': np.random.randint(100000, 10000000),
                'High_52W': base_price * np.random.uniform(1.1, 1.5),
                'Low_52W': base_price * np.random.uniform(0.7, 0.9),
                'Market_Cap': np.random.uniform(10000, 500000)
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_sector_data(self) -> pd.DataFrame:
        """Generate sample sector data"""
        sectors = ['Technology', 'Banking', 'Energy', 'Pharma', 'Auto', 
                  'FMCG', 'Metals', 'Telecom', 'Realty', 'Infrastructure']
        
        data = []
        for sector in sectors:
            data.append({
                'Sector': sector,
                'Performance': np.random.uniform(-10, 15),
                'Market_Cap': np.random.uniform(50000, 1000000),
                'P/E_Ratio': np.random.uniform(10, 35),
                'Volume': np.random.randint(1000000, 50000000),
                'Top_Performer': f"Stock_{np.random.randint(1, 100)}"
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_technical_data(self) -> pd.DataFrame:
        """Generate sample technical indicators data"""
        stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
        signals = ['BUY', 'SELL', 'HOLD']
        
        data = []
        for stock in stocks:
            base_price = np.random.uniform(100, 3000)
            data.append({
                'Symbol': stock,
                'RSI': np.random.uniform(20, 80),
                'MACD': np.random.uniform(-5, 5),
                'SMA_20': base_price * np.random.uniform(0.95, 1.05),
                'SMA_50': base_price * np.random.uniform(0.90, 1.10),
                'Bollinger_Upper': base_price * 1.05,
                'Bollinger_Lower': base_price * 0.95,
                'Signal': np.random.choice(signals)
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_portfolio_data(self) -> pd.DataFrame:
        """Generate sample portfolio data"""
        stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
        
        data = []
        for stock in stocks:
            qty = np.random.randint(10, 100)
            avg_price = np.random.uniform(100, 3000)
            current_price = avg_price * np.random.uniform(0.8, 1.3)
            current_value = qty * current_price
            pnl = current_value - (qty * avg_price)
            
            data.append({
                'Symbol': stock,
                'Quantity': qty,
                'Avg_Price': avg_price,
                'Current_Price': current_price,
                'Current_Value': current_value,
                'P&L': pnl,
                'P&L_%': (pnl / (qty * avg_price)) * 100
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_market_data(self) -> Dict[str, float]:
        """Generate sample market indices data"""
        return {
            'NIFTY 50': 18500.0 + np.random.uniform(-200, 200),
            'NIFTY BANK': 42000.0 + np.random.uniform(-500, 500),
            'SENSEX': 62000.0 + np.random.uniform(-300, 300),
            'NIFTY IT': 30000.0 + np.random.uniform(-400, 400),
            'NIFTY AUTO': 15000.0 + np.random.uniform(-200, 200)
        }
    
    def get_table_styles(self) -> str:
        """
        Get CSS styles for tables
        
        Returns:
            str: CSS styles
        """
        return '''
        <style>
        .table-container {
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .table-title {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
        
        .market-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            background-color: white;
            border-radius: 6px;
            overflow: hidden;
        }
        
        .market-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            font-size: 14px;
        }
        
        .market-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
            font-size: 13px;
        }
        
        .market-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .market-table tr:hover {
            background-color: #e8f4f8;
        }
        
        .error-table {
            background-color: #ffe6e6;
            border: 1px solid #ffcccc;
            padding: 20px;
            border-radius: 8px;
            color: #cc0000;
            text-align: center;
        }
        
        .error-table h3 {
            margin-top: 0;
            color: #cc0000;
        }
        </style>
        '''

# Example usage and testing
if __name__ == "__main__":
    # Initialize table generator
    table_gen = TableGenerator(style="professional")
    
    # Test with empty data (will use sample data)
    stock_table = table_gen.create_stock_performance_table(pd.DataFrame())
    sector_table = table_gen.create_sector_analysis_table(pd.DataFrame())
    tech_table = table_gen.create_technical_indicators_table(pd.DataFrame())
    portfolio_table = table_gen.create_portfolio_summary_table(pd.DataFrame())
    market_table = table_gen.create_market_summary_table({})
    
    # Save test report
    with open('test_tables_report.html', 'w', encoding='utf-8') as f:
        f.write(table_gen.get_table_styles())
        f.write('<h1>Market Research Tables Test Report</h1>')
        f.write(stock_table)
        f.write(sector_table)
        f.write(tech_table)
        f.write(portfolio_table)
        f.write(market_table)
    
    print("Table generator test completed. Check 'test_tables_report.html' for output.")