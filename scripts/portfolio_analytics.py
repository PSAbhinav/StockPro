"""
Enhanced Portfolio Analytics Module
Provides real-time portfolio tracking, P&L calculation, and performance metrics
"""

def get_real_time_portfolio_value(portfolio_manager, user_id: int, fetcher) -> dict:
    """
    Calculate real-time portfolio value using live market prices.
    
    Args:
        portfolio_manager: PortfolioManager instance
        user_id: User ID
        fetcher: AdvancedStockFetcher instance for live prices
    
    Returns:
        dict with total_value, holdings_value, cash, and breakdown per stock
    """
    try:
        holdings = portfolio_manager.get_portfolio(user_id)
        cash = portfolio_manager.get_user_balance(user_id)
        
        holdings_breakdown = []
        total_holdings_value = 0
        
        for holding in holdings:
            ticker = holding['ticker']
            quantity = holding['quantity']
            avg_price = holding['avg_price']
            
            # Get current live price
            stock_data = fetcher.get_comprehensive_data(ticker)
            current_price = stock_data.get('current_price', avg_price) if stock_data else avg_price
            
            current_value = current_price * quantity
            invested_value = avg_price * quantity
            pnl = current_value - invested_value
            pnl_percent = (pnl / invested_value * 100) if invested_value > 0 else 0
            
            holdings_breakdown.append({
                'ticker': ticker,
                'quantity': quantity,
                'avg_price': avg_price,
                'current_price': current_price,
                'invested_value': invested_value,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_percent': pnl_percent
            })
            
            total_holdings_value += current_value
        
        total_value = total_holdings_value + cash
        
        return {
            'total_value': total_value,
            'holdings_value': total_holdings_value,
            'cash': cash,
            'holdings_breakdown': holdings_breakdown,
            'holdings_count': len(holdings)
        }
    except Exception as e:
        import logging
        logging.error(f"Error calculating real-time portfolio value: {e}")
        return None


def get_sector_allocation(portfolio_manager, user_id: int, fetcher) -> list:
    """
    Calculate sector-wise allocation of portfolio.
    
    Returns list of sectors with their allocation percentages
    """
    try:
        portfolio_data = get_real_time_portfolio_value(portfolio_manager, user_id, fetcher)
        if not portfolio_data:
            return []
        
        # Sector mapping (simplified - can be enhanced with API data)
        sector_map = {
            'RELIANCE.NS': 'Energy',
            'TCS.NS': 'Technology',
            'HDFCBANK.NS': 'Finance',
            'INFY.NS': 'Technology',
            'ICICIBANK.NS': 'Finance',
            'SBIN.NS': 'Finance',
            'BHARTIARTL.NS': 'Telecom',
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer',
            'NVDA': 'Technology',
            'META': 'Technology',
            'TSLA': 'Automotive'
        }
        
        sector_values = {}
        total_value = portfolio_data['holdings_value']
        
        for holding in portfolio_data['holdings_breakdown']:
            ticker = holding['ticker']
            value = holding['current_value']
            sector = sector_map.get(ticker, 'Others')
            
            if sector in sector_values:
                sector_values[sector] += value
            else:
                sector_values[sector] = value
        
        # Convert to percentage allocation
        allocation = []
        for sector, value in sector_values.items():
            percentage = (value / total_value * 100) if total_value > 0 else 0
            allocation.append({
                'sector': sector,
                'value': value,
                'percentage': percentage
            })
        
        # Sort by value descending
        allocation.sort(key=lambda x: x['value'], reverse=True)
        
        return allocation
    except Exception as e:
        import logging
        logging.error(f"Error calculating sector allocation: {e}")
        return []


def export_portfolio_to_csv(portfolio_manager, user_id: int, fetcher, filename: str = None) -> str:
    """
    Export portfolio to CSV file.
    
    Returns path to saved CSV file
    """
    import csv
    from datetime import datetime
    import os
    
    try:
        portfolio_data = get_real_time_portfolio_value(portfolio_manager, user_id, fetcher)
        if not portfolio_data:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'portfolio_export_{timestamp}.csv'
        
        # Create exports directory if it doesn't exist
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        export_dir = os.path.join(base_dir, 'data', 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        filepath = os.path.join(export_dir, filename)
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Portfolio Export', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow([])
            writer.writerow(['Summary'])
            writer.writerow(['Total Value', f'₹{portfolio_data["total_value"]:.2f}'])
            writer.writerow(['Holdings Value', f'₹{portfolio_data["holdings_value"]:.2f}'])
            writer.writerow(['Cash', f'₹{portfolio_data["cash"]:.2f}'])
            writer.writerow([])
            
            # Holdings
            writer.writerow(['Ticker', 'Quantity', 'Avg Price', 'Current Price', 'Invested', 'Current Value', 'P&L', 'P&L %'])
            for holding in portfolio_data['holdings_breakdown']:
                writer.writerow([
                    holding['ticker'],
                    holding['quantity'],
                    f'₹{holding["avg_price"]:.2f}',
                    f'₹{holding["current_price"]:.2f}',
                    f'₹{holding["invested_value"]:.2f}',
                    f'₹{holding["current_value"]:.2f}',
                    f'₹{holding["pnl"]:.2f}',
                    f'{holding["pnl_percent"]:.2f}%'
                ])
        
        return filepath
    except Exception as e:
        import logging
        logging.error(f"Error exporting portfolio: {e}")
        return None
