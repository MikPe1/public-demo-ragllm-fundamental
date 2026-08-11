"""
Data fetcher for financial data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Sequence


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

    @staticmethod
    def _as_number(value, default: Optional[float] = 0.0) -> Optional[float]:
        """Convert Pandas and NumPy scalar values without inventing data."""
        if value is None:
            return default

        numeric = pd.to_numeric(value, errors='coerce')
        if pd.isna(numeric):
            return default
        return float(numeric)

    @staticmethod
    def _valid_dates(statement: Optional[pd.DataFrame], limit: Optional[int] = None) -> List:
        """Return statement columns ordered from newest to oldest."""
        if statement is None or statement.empty:
            return []

        now = pd.Timestamp.now()
        valid_dates = []
        for date in statement.columns:
            try:
                parsed_date = pd.Timestamp(date)
                if parsed_date.tzinfo is not None:
                    parsed_date = parsed_date.tz_localize(None)
                if parsed_date <= now:
                    valid_dates.append((parsed_date, date))
            except (TypeError, ValueError):
                continue

        valid_dates.sort(key=lambda item: item[0], reverse=True)
        dates = [date for _, date in valid_dates]
        return dates[:limit] if limit is not None else dates

    @classmethod
    def _statement_value(
        cls,
        statement: Optional[pd.DataFrame],
        labels: Sequence[str],
        date,
        default: float = 0.0,
    ) -> float:
        """Read the first available line item from a financial statement."""
        if statement is None or statement.empty or date not in statement.columns:
            return default

        for label in labels:
            if label in statement.index:
                value = cls._as_number(statement.loc[label, date], None)
                if value is not None:
                    return value
        return default
    
    def calculate_ltm(self) -> Dict:
        """
        Calculate Last Twelve Months (LTM) metrics from quarterly data
        Sums the last 4 quarters for income statement items
        Returns dictionary with LTM values
        """
        quarterly_income = self.get_income_statement("quarterly")

        quarter_dates = self._valid_dates(quarterly_income, limit=4)
        if not quarter_dates:
            return {}

        ltm_data = {}
        sum_fields = [
            'Net Income', 'Total Revenue', 'Operating Income', 'EBIT', 
            'Interest Expense', 'Cost of Revenue', 'Gross Profit'
        ]

        for field in sum_fields:
            if field in quarterly_income.index:
                values = pd.to_numeric(
                    quarterly_income.loc[field, quarter_dates],
                    errors='coerce'
                ).dropna()
                if not values.empty:
                    ltm_data[field] = float(values.sum())

        return ltm_data

    def get_company_profile(self) -> Dict:
        """Return non-market company context available from Yahoo Finance."""
        try:
            info = self.stock.info
        except Exception:
            return {'ticker': self.ticker}

        profile_fields = {
            'shortName': 'name',
            'longBusinessSummary': 'business_summary',
            'sector': 'sector',
            'industry': 'industry',
            'country': 'country',
            'website': 'website',
            'fullTimeEmployees': 'employees',
        }
        profile = {'ticker': self.ticker}
        for source_key, target_key in profile_fields.items():
            value = info.get(source_key)
            if value not in (None, '', 'N/A'):
                profile[target_key] = value
        return profile

    def get_annual_history(self, num_years: int = 2) -> pd.DataFrame:
        """Return annual revenue, earnings, and EPS for the requested years."""
        income_statement = self.get_income_statement('annual')
        dates = self._valid_dates(income_statement, limit=num_years)
        rows = []

        for date in reversed(dates):
            rows.append({
                'Date': date,
                'Total Revenue': self._statement_value(
                    income_statement, ['Total Revenue'], date
                ),
                'Net Income': self._statement_value(
                    income_statement, ['Net Income'], date
                ),
                'Operating Income': self._statement_value(
                    income_statement, ['Operating Income'], date
                ),
                'EPS': self._statement_value(
                    income_statement, ['Diluted EPS', 'Basic EPS'], date
                ),
            })

        return pd.DataFrame(rows)
    
    def extract_financial_data(self, period: str = "quarterly") -> Dict:
        """
        Extract key financial metrics from all statements
        Uses LTM (Last Twelve Months) for income statement to get most current earnings
        Balance sheet uses current period (quarterly)
        """
        ltm_data = self.calculate_ltm()
        balance_sheet = self.get_balance_sheet("quarterly")
        quarterly_income = self.get_income_statement("quarterly")

        latest_balance_date = next(iter(self._valid_dates(balance_sheet, limit=1)), None)
        latest_income_date = next(iter(self._valid_dates(quarterly_income, limit=1)), None)
        if latest_balance_date is None and latest_income_date is None:
            return {}

        shareholders_equity = self._statement_value(
            balance_sheet,
            ['Total Stockholder Equity', 'Total Equity', 'Shareholders Equity',
             'Total Shareholders Equity'],
            latest_balance_date,
        )
        if shareholders_equity == 0:
            total_assets = self._statement_value(balance_sheet, ['Total Assets'], latest_balance_date)
            total_liabilities = self._statement_value(
                balance_sheet, ['Total Liabilities', 'Total Liabilities Net Minority Interest'],
                latest_balance_date,
            )
            if total_assets and total_liabilities:
                shareholders_equity = total_assets - total_liabilities

        cash = self._statement_value(
            balance_sheet,
            ['Cash', 'Cash And Cash Equivalents', 'Cash and Cash Equivalents',
             'Cash Cash Equivalents And Short Term Investments'],
            latest_balance_date,
        )
        short_term_debt = self._statement_value(
            balance_sheet,
            ['Short Long Term Debt', 'Current Portion of Long Term Debt', 'Short Term Debt'],
            latest_balance_date,
        )
        long_term_debt = self._statement_value(
            balance_sheet,
            ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'],
            latest_balance_date,
        )
        total_debt = self._statement_value(balance_sheet, ['Total Debt'], latest_balance_date)
        if total_debt == 0:
            total_debt = short_term_debt + long_term_debt

        financial_data = {
            'total_revenue': ltm_data.get('Total Revenue', 0),
            'cost_of_revenue': ltm_data.get('Cost of Revenue', 0),
            'gross_profit': ltm_data.get('Gross Profit', 0),
            'operating_income': ltm_data.get('Operating Income', 0),
            'net_income': ltm_data.get('Net Income', 0),
            'interest_expense': ltm_data.get('Interest Expense', 0),
            'ebit': ltm_data.get('EBIT', ltm_data.get('Operating Income', 0)),
            'cash': cash,
            'current_assets': self._statement_value(balance_sheet, ['Current Assets'], latest_balance_date),
            'accounts_receivable': self._statement_value(
                balance_sheet, ['Accounts Receivable', 'Receivables'], latest_balance_date
            ),
            'inventory': self._statement_value(balance_sheet, ['Inventory'], latest_balance_date),
            'current_liabilities': self._statement_value(
                balance_sheet, ['Current Liabilities'], latest_balance_date
            ),
            'short_term_debt': short_term_debt,
            'long_term_debt': long_term_debt,
            'total_debt': total_debt,
            'shareholders_equity': shareholders_equity,
            'stock_price': self._as_number(self.get_stock_price()),
            'market_cap': self._as_number(self.get_market_cap()),
            'shares_outstanding': self._as_number(self.get_shares_outstanding()),
            'total_capital': shareholders_equity + total_debt,
            'marketable_securities': self._statement_value(
                balance_sheet,
                ['Other Short Term Investments', 'Short Term Investments'],
                latest_balance_date,
            ),
            'period_end': str(latest_balance_date or latest_income_date),
            'source': 'Yahoo Finance company financial statements',
        }

        return financial_data
    
    def get_quarterly_history(self, num_quarters: int = 4) -> pd.DataFrame:
        """
        Get historical quarterly data for comparison
        Returns a dataframe with selected metrics across quarters
        """
        income_stmt = self.get_income_statement("quarterly")
        valid_dates = self._valid_dates(income_stmt, limit=num_quarters)
        if not valid_dates:
            return pd.DataFrame()

        history_data = []
        for date in valid_dates:
            history_data.append({
                'Date': date,
                'Total Revenue': self._statement_value(income_stmt, ['Total Revenue'], date),
                'Net Income': self._statement_value(income_stmt, ['Net Income'], date),
                'Operating Income': self._statement_value(income_stmt, ['Operating Income'], date),
                'EPS': self._statement_value(income_stmt, ['Diluted EPS', 'Basic EPS'], date),
            })

        ltm_data = self.calculate_ltm()
        if ltm_data:
            ltm_row = {
                'Date': 'LTM (Last 12M)',
                'Total Revenue': ltm_data.get('Total Revenue', 0),
                'Net Income': ltm_data.get('Net Income', 0),
                'Operating Income': ltm_data.get('Operating Income', 0),
                'EPS': ltm_data.get('Net Income', 0) / self._as_number(self.get_shares_outstanding()) if self._as_number(self.get_shares_outstanding()) > 0 else 0,
            }
            history_data.insert(0, ltm_row)

        return pd.DataFrame(history_data)
