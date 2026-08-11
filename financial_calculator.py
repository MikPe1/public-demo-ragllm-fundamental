"""
Financial metrics calculator based on income statement, balance sheet, and cash flow data
"""

import pandas as pd
from typing import Dict, Optional


class FinancialCalculator:
    """
    Calculates key financial metrics from financial statements
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.metrics = {}
    
    def calculate_eps(self, net_income: float, shares_outstanding: float) -> float:
        """
        Earnings Per Share = Net Income / Shares Outstanding
        """
        if shares_outstanding == 0 or shares_outstanding is None:
            return None
        if net_income is None or net_income == 0:
            return None
        return net_income / shares_outstanding
    
    def calculate_pe_ratio(self, stock_price: float, eps: float) -> float:
        """
        Price/Earnings Ratio = Stock Price / EPS
        """
        if eps is None or eps == 0:
            return None
        if stock_price is None or stock_price == 0:
            return None
        return stock_price / eps
    
    def calculate_roe(self, net_income: float, shareholders_equity: float) -> float:
        """
        Return on Equity = Net Income / Shareholders' Equity
        Returns as percentage
        Based on annual net income and current shareholders equity
        """
        if shareholders_equity is None or shareholders_equity <= 0:
            return None
        if net_income is None or net_income == 0:
            return None
        roe = (net_income / shareholders_equity) * 100
        # ROE rarely exceeds 500% for healthy companies, flag unrealistic values
        if roe > 500:
            return None
        return roe
    
    def calculate_debt_to_capital(self, short_term_debt: float, long_term_debt: float, 
                                  total_capital: float) -> float:
        """
        Debt-to-Capital Ratio = (Short-term Debt + Long-term Debt) / Total Capital
        Returns as percentage
        """
        if total_capital == 0 or total_capital is None:
            return None
        total_debt = short_term_debt + long_term_debt
        if total_debt is None:
            return None
        return (total_debt / total_capital) * 100
    
    def calculate_interest_coverage(self, ebit: float, interest_expense: float) -> float:
        """
        Interest Coverage Ratio = EBIT / Interest Expense
        """
        if interest_expense == 0 or interest_expense is None:
            return None
        if ebit is None or ebit == 0:
            return None
        return ebit / abs(interest_expense)
    
    def calculate_ev_to_ebit(self, market_cap: float, total_debt: float, 
                             cash: float, ebit: float) -> float:
        """
        EV/EBIT = (Market Cap + Total Debt - Cash) / EBIT
        Enterprise Value to EBIT
        """
        if ebit == 0 or ebit is None:
            return None
        if market_cap is None or market_cap == 0:
            return None
        enterprise_value = market_cap + total_debt - cash
        return enterprise_value / ebit
    
    def calculate_operating_margin(self, operating_income: float, total_revenue: float) -> float:
        """
        Operating Margin = Operating Income / Total Revenue
        Returns as percentage
        """
        if total_revenue == 0 or total_revenue is None:
            return None
        if operating_income is None or operating_income == 0:
            return None
        return (operating_income / total_revenue) * 100
    
    def calculate_quick_ratio(self, cash: float, marketable_securities: float, 
                             accounts_receivable: float, current_liabilities: float) -> float:
        """
        Quick Ratio = (Cash + Marketable Securities + Accounts Receivable) / Current Liabilities
        Acid Test - excludes inventory
        """
        if current_liabilities == 0 or current_liabilities is None:
            return None
        quick_assets = cash + marketable_securities + accounts_receivable
        if quick_assets is None or quick_assets == 0:
            return None
        return quick_assets / current_liabilities

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
        """Return a ratio only when both values are present and usable."""
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    def calculate_gross_margin(self, gross_profit: float, total_revenue: float) -> float:
        """Gross Profit / Total Revenue, returned as a percentage."""
        ratio = self._safe_divide(gross_profit, total_revenue)
        return ratio * 100 if ratio is not None else None

    def calculate_fcf_margin(self, free_cash_flow: float, total_revenue: float) -> float:
        """Free Cash Flow / Total Revenue, returned as a percentage."""
        ratio = self._safe_divide(free_cash_flow, total_revenue)
        return ratio * 100 if ratio is not None else None

    def calculate_cash_conversion(self, operating_cash_flow: float, net_income: float) -> float:
        """Operating Cash Flow / Net Income."""
        return self._safe_divide(operating_cash_flow, net_income)

    def calculate_current_ratio(self, current_assets: float, current_liabilities: float) -> float:
        """Current Assets / Current Liabilities."""
        return self._safe_divide(current_assets, current_liabilities)

    def calculate_net_debt_to_ebitda(self, total_debt: float, cash: float, ebitda: float) -> float:
        """Net Debt / EBITDA."""
        if total_debt is None or cash is None:
            return None
        return self._safe_divide(total_debt - cash, ebitda)

    def calculate_ev_to_ebitda(
        self,
        market_cap: float,
        total_debt: float,
        cash: float,
        ebitda: float,
    ) -> float:
        """Enterprise Value / EBITDA."""
        if market_cap is None or total_debt is None or cash is None:
            return None
        enterprise_value = market_cap + total_debt - cash
        return self._safe_divide(enterprise_value, ebitda)

    def calculate_fcf_yield(self, free_cash_flow: float, market_cap: float) -> float:
        """Free Cash Flow / Market Capitalization, returned as a percentage."""
        ratio = self._safe_divide(free_cash_flow, market_cap)
        return ratio * 100 if ratio is not None else None
    
    def calculate_all_metrics(self, financial_data: Dict) -> Dict:
        """
        Calculate all metrics from a dictionary of financial data
        
        Expected keys:
        - net_income
        - shares_outstanding
        - stock_price
        - shareholders_equity
        - short_term_debt
        - long_term_debt
        - total_capital
        - ebit
        - interest_expense
        - market_cap
        - total_debt
        - cash
        - operating_income
        - total_revenue
        - marketable_securities
        - accounts_receivable
        - current_liabilities
        """
        
        # Calculate EPS first, then use it for P/E
        eps = self.calculate_eps(
            financial_data.get('net_income', 0),
            financial_data.get('shares_outstanding', 0)
        )
        
        self.metrics = {
            'EPS': eps,
            'P/E Ratio': self.calculate_pe_ratio(
                financial_data.get('stock_price', 0),
                eps if eps is not None else 0
            ),
            'ROE': self.calculate_roe(
                financial_data.get('net_income', 0),
                financial_data.get('shareholders_equity', 0)
            ),
            'Debt-to-Capital': self.calculate_debt_to_capital(
                financial_data.get('short_term_debt', 0),
                financial_data.get('long_term_debt', 0),
                financial_data.get('total_capital', 0)
            ),
            'Interest Coverage': self.calculate_interest_coverage(
                financial_data.get('ebit', 0),
                financial_data.get('interest_expense', 0)
            ),
            'EV/EBIT': self.calculate_ev_to_ebit(
                financial_data.get('market_cap', 0),
                financial_data.get('total_debt', 0),
                financial_data.get('cash', 0),
                financial_data.get('ebit', 0)
            ),
            'Operating Margin': self.calculate_operating_margin(
                financial_data.get('operating_income', 0),
                financial_data.get('total_revenue', 0)
            ),
            'Gross Margin': self.calculate_gross_margin(
                financial_data.get('gross_profit'),
                financial_data.get('total_revenue')
            ),
            'Quick Ratio': self.calculate_quick_ratio(
                financial_data.get('cash', 0),
                financial_data.get('marketable_securities', 0),
                financial_data.get('accounts_receivable', 0),
                financial_data.get('current_liabilities', 0)
            ),
            'Current Ratio': self.calculate_current_ratio(
                financial_data.get('current_assets'),
                financial_data.get('current_liabilities')
            ),
            'EBITDA': financial_data.get('ebitda'),
            'Free Cash Flow': financial_data.get('free_cash_flow'),
            'FCF Margin': self.calculate_fcf_margin(
                financial_data.get('free_cash_flow'),
                financial_data.get('total_revenue')
            ),
            'Operating CF / Net Income': self.calculate_cash_conversion(
                financial_data.get('operating_cash_flow'),
                financial_data.get('net_income')
            ),
            'Net Debt / EBITDA': self.calculate_net_debt_to_ebitda(
                financial_data.get('total_debt'),
                financial_data.get('cash'),
                financial_data.get('ebitda')
            ),
            'EV/EBITDA': self.calculate_ev_to_ebitda(
                financial_data.get('market_cap'),
                financial_data.get('total_debt'),
                financial_data.get('cash'),
                financial_data.get('ebitda')
            ),
            'FCF Yield': self.calculate_fcf_yield(
                financial_data.get('free_cash_flow'),
                financial_data.get('market_cap')
            ),
            'Revenue Growth YoY': financial_data.get('revenue_growth_yoy'),
            'Net Income Growth YoY': financial_data.get('net_income_growth_yoy'),
            'Operating Income Growth YoY': financial_data.get('operating_income_growth_yoy'),
            'EPS Growth YoY': financial_data.get('eps_growth_yoy'),
        }
        
        return self.metrics
    
    def format_metrics(self) -> Dict[str, str]:
        """
        Format metrics with appropriate units and decimals
        Handle None/zero values appropriately
        """
        formatted = {}
        
        percentage_metrics = {
            'ROE',
            'Debt-to-Capital',
            'Operating Margin',
            'Gross Margin',
            'FCF Margin',
            'FCF Yield',
            'Revenue Growth YoY',
            'Net Income Growth YoY',
            'Operating Income Growth YoY',
            'EPS Growth YoY',
        }
        multiple_metrics = {
            'Interest Coverage',
            'EV/EBIT',
            'Quick Ratio',
            'Current Ratio',
            'Operating CF / Net Income',
            'Net Debt / EBITDA',
            'EV/EBITDA',
        }

        for metric_name, value in self.metrics.items():
            # Check if value is None or invalid
            if value is None or pd.isna(value):
                formatted[metric_name] = "N/A"
            elif metric_name == 'EPS':
                formatted[metric_name] = f"${value:.2f}"
            elif metric_name == 'P/E Ratio':
                if value is None or value <= 0 or value > 1000:  # Unrealistic P/E
                    formatted[metric_name] = "N/A"
                else:
                    formatted[metric_name] = f"{value:.2f}x"
            elif metric_name in percentage_metrics:
                formatted[metric_name] = f"{value:.2f}%"
            elif metric_name in multiple_metrics:
                formatted[metric_name] = f"{value:.2f}x"
            elif metric_name in {'EBITDA', 'Free Cash Flow'}:
                if abs(value) >= 1e9:
                    formatted[metric_name] = f"${value / 1e9:.2f}B"
                elif abs(value) >= 1e6:
                    formatted[metric_name] = f"${value / 1e6:.1f}M"
                else:
                    formatted[metric_name] = f"${value:,.0f}"
            else:
                formatted[metric_name] = f"{value:.2f}"
        
        return formatted
