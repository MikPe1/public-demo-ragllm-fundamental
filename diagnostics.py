"""
Financial data diagnostics - checks which data is available and what's missing
"""

def diagnose_financial_data(financial_data: dict) -> dict:
    """
    Diagnose which financial data is available and which is missing/zero
    """
    
    diagnostics = {
        'available': [],
        'missing': [],
        'zero': [],
        'negative': []
    }
    
    critical_fields = {
        'net_income': 'Net Income',
        'shares_outstanding': 'Shares Outstanding',
        'stock_price': 'Stock Price',
        'shareholders_equity': 'Shareholders Equity',
        'short_term_debt': 'Short-term Debt',
        'long_term_debt': 'Long-term Debt',
        'total_capital': 'Total Capital',
        'ebit': 'EBIT',
        'interest_expense': 'Interest Expense',
        'market_cap': 'Market Cap',
        'total_debt': 'Total Debt',
        'cash': 'Cash',
        'operating_income': 'Operating Income',
        'total_revenue': 'Total Revenue',
        'marketable_securities': 'Marketable Securities',
        'accounts_receivable': 'Accounts Receivable',
        'current_liabilities': 'Current Liabilities',
        'current_assets': 'Current Assets',
        'operating_cash_flow': 'Operating Cash Flow',
        'capital_expenditure': 'Capital Expenditure',
        'free_cash_flow': 'Free Cash Flow',
        'ebitda': 'EBITDA',
        'depreciation_and_amortization': 'Depreciation & Amortization',
    }
    
    for key, label in critical_fields.items():
        value = financial_data.get(key, None)
        
        if value is None or value == '':
            diagnostics['missing'].append(f"{label} ({key})")
        elif value == 0:
            diagnostics['zero'].append(f"{label} ({key})")
        elif value < 0:
            diagnostics['negative'].append(f"{label} ({key}): {value}")
        else:
            diagnostics['available'].append(f"{label} ({key}): {value}")
    
    return diagnostics


def print_diagnostics(diagnostics: dict) -> str:
    """
    Format diagnostics for display
    """
    output = []
    output.append("=" * 60)
    output.append("FINANCIAL DATA DIAGNOSTICS")
    output.append("=" * 60)
    
    if diagnostics['available']:
        output.append("\nAVAILABLE DATA:")
        for item in diagnostics['available']:
            output.append(f"  ✓ {item}")
    
    if diagnostics['zero']:
        output.append("\nZERO VALUES (will cause N/A):")
        for item in diagnostics['zero']:
            output.append(f"  ⚠ {item}")
    
    if diagnostics['negative']:
        output.append("\nNEGATIVE VALUES (unusual):")
        for item in diagnostics['negative']:
            output.append(f"  ⚠ {item}")
    
    if diagnostics['missing']:
        output.append("\nMISSING DATA:")
        for item in diagnostics['missing']:
            output.append(f"  ✗ {item}")
    
    output.append("=" * 60)
    
    return "\n".join(output)
