"""
Data fetcher for financial data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Optional, Tuple


class YahooFinanceFetcher:
    """
    Fetches financial data from Yahoo Finance
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
    
    def get_income_statement(self, period: str = "quarterly") -> pd.DataFrame:
        """
        Get income statement data
        period: 'quarterly' or 'annual'
        """
        if period == "quarterly":
            return self.stock.quarterly_financials
        else:
            return self.stock.financials
    
    def get_balance_sheet(self, period: str = "quarterly") -> pd.DataFrame:
        """
        Get balance sheet data
        period: 'quarterly' or 'annual'
        """
        if period == "quarterly":
            return self.stock.quarterly_balance_sheet
        else:
            return self.stock.balance_sheet
    
    def get_cash_flow(self, period: str = "quarterly") -> pd.DataFrame:
        """
        Get cash flow statement data
        period: 'quarterly' or 'annual'
        """
        if period == "quarterly":
            return self.stock.quarterly_cashflow
        else:
            return self.stock.cashflow
    
    def get_stock_price(self) -> float:
        """
        Get current stock price
        """
        return self.stock.info.get('currentPrice', 0)
    
    def get_market_cap(self) -> float:
        """
        Get market capitalization in millions
        """
        return self.stock.info.get('marketCap', 0)
    
    def get_shares_outstanding(self) -> float:
        """
        Get shares outstanding
        """
        return self.stock.info.get('sharesOutstanding', 0)
    
    def calculate_ltm(self) -> Dict:
        """
        Calculate Last Twelve Months (LTM) metrics from quarterly data
        Sums the last 4 quarters for income statement items
        Returns dictionary with LTM values
        """
        from datetime import datetime
        
        quarterly_income = self.get_income_statement("quarterly")
        
        if quarterly_income is None or len(quarterly_income.columns) < 1:
            return {}
        
        # Get last 4 quarters (or fewer if not available)
        current_time = pd.Timestamp(datetime.now())
        valid_dates = [date for date in quarterly_income.columns if pd.Timestamp(date) <= current_time]
        
        if len(valid_dates) == 0:
            return {}
        
        last_4_quarters = valid_dates[:min(4, len(valid_dates))]
        
        ltm_data = {}
        
        # Sum these fields across quarters
        sum_fields = [
            'Net Income', 'Total Revenue', 'Operating Income', 'EBIT', 
            'Interest Expense', 'Cost of Revenue', 'Gross Profit'
        ]
        
        for field in sum_fields:
            if field in quarterly_income.index:
                try:
                    values = [quarterly_income.loc[field, date] for date in last_4_quarters if isinstance(quarterly_income.loc[field, date], (int, float)) and quarterly_income.loc[field, date] > 0]
                    if values and len(values) > 0:
                        ltm_data[field] = sum(values)
                except:
                    pass
        
        return ltm_data
    
    def extract_financial_data(self, period: str = "quarterly") -> Dict:
        """
        Extract key financial metrics from all statements
        Uses LTM (Last Twelve Months) for income statement to get most current earnings
        Balance sheet uses current period (quarterly)
        """
        from datetime import datetime
        
        # Get LTM data (last 12 months from quarterly statements)
        ltm_data = self.calculate_ltm()
        
        # Get current balance sheet (quarterly for latest positions)
        balance_sheet = self.get_balance_sheet("quarterly")
        quarterly_income = self.get_income_statement("quarterly")
        
        # Find most recent valid balance sheet date
        latest_date = None
        current_time = pd.Timestamp(datetime.now())
        
        if balance_sheet is not None:
            for date in balance_sheet.columns:
                if pd.Timestamp(date) <= current_time:
                    latest_date = date
                    break
        
        if latest_date is None and quarterly_income is not None:
            for date in quarterly_income.columns:
                if pd.Timestamp(date) <= current_time:
                    latest_date = date
                    break
        
        if latest_date is None:
            return {}
        
        try:
            # Try to get shareholders equity - try multiple possible field names
            shareholders_equity = 0
            if 'Total Stockholder Equity' in balance_sheet.index:
                shareholders_equity = balance_sheet.loc['Total Stockholder Equity', latest_date]
            elif 'Total Equity' in balance_sheet.index:
                shareholders_equity = balance_sheet.loc['Total Equity', latest_date]
            elif 'Shareholders Equity' in balance_sheet.index:
                shareholders_equity = balance_sheet.loc['Shareholders Equity', latest_date]
            elif 'Total Shareholders Equity' in balance_sheet.index:
                shareholders_equity = balance_sheet.loc['Total Shareholders Equity', latest_date]
            
            # If still 0, try to calculate from available fields
            if shareholders_equity == 0:
                # Try Assets - Liabilities
                total_assets = 0
                total_liabilities = 0
                
                if 'Total Assets' in balance_sheet.index:
                    total_assets = balance_sheet.loc['Total Assets', latest_date]
                
                if 'Total Liabilities' in balance_sheet.index:
                    total_liabilities = balance_sheet.loc['Total Liabilities', latest_date]
                
                if total_assets != 0 and total_liabilities != 0:
                    shareholders_equity = total_assets - total_liabilities
                
                # Final fallback: use market cap as proxy for shareholders equity
                # Market Cap represents the market value of equity
                market_cap = self.get_market_cap()
                if shareholders_equity == 0 and market_cap > 0:
                    shareholders_equity = market_cap
            
            # Try to get cash - try multiple possible field names
            cash = 0
            if 'Cash' in balance_sheet.index:
                cash = balance_sheet.loc['Cash', latest_date]
            elif 'Cash And Cash Equivalents' in balance_sheet.index:
                cash = balance_sheet.loc['Cash And Cash Equivalents', latest_date]
            elif 'Cash and Cash Equivalents' in balance_sheet.index:
                cash = balance_sheet.loc['Cash and Cash Equivalents', latest_date]
            
            # Try to get short-term debt - try multiple possible field names
            short_term_debt = 0
            if 'Short Long Term Debt' in balance_sheet.index:
                short_term_debt = balance_sheet.loc['Short Long Term Debt', latest_date]
            elif 'Current Portion of Long Term Debt' in balance_sheet.index:
                short_term_debt = balance_sheet.loc['Current Portion of Long Term Debt', latest_date]
            elif 'Short Term Debt' in balance_sheet.index:
                short_term_debt = balance_sheet.loc['Short Term Debt', latest_date]
            
            financial_data = {
                # Income Statement - use LTM (Last Twelve Months) for accurate ratios
                'total_revenue': ltm_data.get('Total Revenue', 0),
                'cost_of_revenue': ltm_data.get('Cost of Revenue', 0),
                'gross_profit': ltm_data.get('Gross Profit', 0),
                'operating_income': ltm_data.get('Operating Income', 0),
                'net_income': ltm_data.get('Net Income', 0),
                'interest_expense': ltm_data.get('Interest Expense', 0),
                'ebit': ltm_data.get('EBIT', 0),
                
                # Balance Sheet - use current quarter for latest positions
                'cash': cash,
                'current_assets': balance_sheet.loc['Current Assets', latest_date] if 'Current Assets' in balance_sheet.index else 0,
                'accounts_receivable': balance_sheet.loc['Accounts Receivable', latest_date] if 'Accounts Receivable' in balance_sheet.index else 0,
                'inventory': balance_sheet.loc['Inventory', latest_date] if 'Inventory' in balance_sheet.index else 0,
                'current_liabilities': balance_sheet.loc['Current Liabilities', latest_date] if 'Current Liabilities' in balance_sheet.index else 0,
                'short_term_debt': short_term_debt,
                'long_term_debt': balance_sheet.loc['Long Term Debt', latest_date] if 'Long Term Debt' in balance_sheet.index else 0,
                'total_debt': balance_sheet.loc['Total Debt', latest_date] if 'Total Debt' in balance_sheet.index else 0,
                'shareholders_equity': shareholders_equity,
                
                # Stock info
                'stock_price': self.get_stock_price(),
                'market_cap': self.get_market_cap(),
                'shares_outstanding': self.get_shares_outstanding(),
                
                # Calculated fields
                'total_capital': 0,  # Will be calculated below
                'marketable_securities': 0,  # Usually minimal
            }
            
            # Calculate total capital (equity + debt)
            financial_data['total_capital'] = financial_data['shareholders_equity'] + financial_data['total_debt']
            
            # Fallback: if EBIT not available, calculate from Operating Income
            if financial_data['ebit'] == 0 and financial_data['operating_income'] != 0:
                financial_data['ebit'] = financial_data['operating_income']
            
            return financial_data
        
        except Exception as e:
            print(f"Error extracting financial data: {str(e)}")
            return {}
    
    def get_quarterly_history(self, num_quarters: int = 4) -> pd.DataFrame:
        """
        Get historical quarterly data for comparison
        Returns a dataframe with selected metrics across quarters
        """
        income_stmt = self.get_income_statement("quarterly")
        balance_sheet = self.get_balance_sheet("quarterly")
        
        from datetime import datetime
        
        # Filter out future dates
        valid_dates = [date for date in income_stmt.columns if pd.Timestamp(date) <= pd.Timestamp(datetime.now())]
        
        if len(valid_dates) == 0:
            return pd.DataFrame()
        
        quarters = valid_dates[:num_quarters]
        
        history_data = []
        
        for date in quarters:
            try:
                data = {
                    'Date': date,
                    'Total Revenue': income_stmt.loc['Total Revenue', date] if 'Total Revenue' in income_stmt.index else 0,
                    'Net Income': income_stmt.loc['Net Income', date] if 'Net Income' in income_stmt.index else 0,
                    'Operating Income': income_stmt.loc['Operating Income', date] if 'Operating Income' in income_stmt.index else 0,
                    'EPS': income_stmt.loc['Diluted EPS', date] if 'Diluted EPS' in income_stmt.index else 0,
                }
                history_data.append(data)
            except Exception as e:
                continue
        
        # Add LTM row
        ltm_data = self.calculate_ltm()
        if ltm_data:
            ltm_row = {
                'Date': 'LTM (Last 12M)',
                'Total Revenue': ltm_data.get('Total Revenue', 0),
                'Net Income': ltm_data.get('Net Income', 0),
                'Operating Income': ltm_data.get('Operating Income', 0),
                'EPS': ltm_data.get('Net Income', 0) / self.get_shares_outstanding() if self.get_shares_outstanding() > 0 else 0,
            }
            history_data.insert(0, ltm_row)
        
        return pd.DataFrame(history_data)
