"""
Reporting Module for Financial Portfolio Analysis
Generates comprehensive reports from analysis results.
"""

import pandas as pd
import json
import os
from datetime import datetime


def generate_reports(analysis_results, config):
    """
    Generate comprehensive reports from analysis results.
    
    Args:
        analysis_results: Dictionary containing analysis results
        config: Configuration module
        
    Returns:
        List of generated report file paths
    """
    report_files = []
    
    # Ensure outputs directory exists
    os.makedirs(config.OUTPUTS_PATH, exist_ok=True)
    
    # Generate text report
    text_report_path = generate_text_report(analysis_results, config)
    if text_report_path:
        report_files.append(text_report_path)
    
    # Generate JSON report
    json_report_path = generate_json_report(analysis_results, config)
    if json_report_path:
        report_files.append(json_report_path)
    
    # Generate CSV reports
    csv_reports = generate_csv_reports(analysis_results, config)
    report_files.extend(csv_reports)
    
    return report_files


def generate_text_report(analysis_results, config):
    """Generate a comprehensive text report."""
    report_path = os.path.join(config.OUTPUTS_PATH, 'portfolio_analysis_report.txt')
    
    try:
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FINANCIAL PORTFOLIO ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Period: {config.START_DATE} to {config.END_DATE}\n")
            f.write(f"Stocks Analyzed: {', '.join(config.TICKERS)}\n")
            f.write("=" * 80 + "\n\n")
            
            # Individual Stock Metrics
            f.write("INDIVIDUAL STOCK PERFORMANCE\n")
            f.write("-" * 80 + "\n\n")
            
            for ticker, metrics in analysis_results.get('individual_stocks', {}).items():
                f.write(f"Stock: {ticker}\n")
                f.write(f"  Annualized Return:     {metrics.get('annualized_return', 0)*100:.2f}%\n")
                f.write(f"  Annualized Volatility: {metrics.get('annualized_volatility', 0)*100:.2f}%\n")
                f.write(f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"  Max Drawdown:          {metrics.get('max_drawdown', 0)*100:.2f}%\n")
                f.write(f"  Cumulative Return:     {metrics.get('cumulative_return', 0)*100:.2f}%\n")
                f.write("\n")
            
            # Portfolio Strategies
            f.write("\n" + "=" * 80 + "\n")
            f.write("PORTFOLIO STRATEGIES COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            portfolios = analysis_results.get('portfolios', {})
            
            for strategy_name, metrics in portfolios.items():
                f.write(f"Strategy: {strategy_name}\n")
                f.write(f"  Annualized Return:     {metrics.get('annualized_return', 0)*100:.2f}%\n")
                f.write(f"  Annualized Volatility: {metrics.get('annualized_volatility', 0)*100:.2f}%\n")
                f.write(f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"  Max Drawdown:          {metrics.get('max_drawdown', 0)*100:.2f}%\n")
                f.write(f"  Cumulative Return:     {metrics.get('cumulative_return', 0)*100:.2f}%\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        return report_path
        
    except Exception as e:
        print(f"Error generating text report: {e}")
        return None


def generate_json_report(analysis_results, config):
    """Generate a JSON report."""
    report_path = os.path.join(config.OUTPUTS_PATH, 'portfolio_metrics.json')
    
    try:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            else:
                return obj
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'analysis_period': {
                'start': config.START_DATE,
                'end': config.END_DATE
            },
            'tickers': config.TICKERS,
            'results': convert_types(analysis_results)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_path
        
    except Exception as e:
        print(f"Error generating JSON report: {e}")
        return None


def generate_csv_reports(analysis_results, config):
    """Generate CSV reports."""
    report_files = []
    
    try:
        # Individual stocks metrics CSV
        stocks_data = []
        for ticker, metrics in analysis_results.get('individual_stocks', {}).items():
            stocks_data.append({
                'Ticker': ticker,
                'Annualized Return (%)': metrics.get('annualized_return', 0) * 100,
                'Annualized Volatility (%)': metrics.get('annualized_volatility', 0) * 100,
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
                'Cumulative Return (%)': metrics.get('cumulative_return', 0) * 100
            })
        
        if stocks_data:
            df = pd.DataFrame(stocks_data)
            csv_path = os.path.join(config.OUTPUTS_PATH, 'portfolio_metrics.csv')
            df.to_csv(csv_path, index=False)
            report_files.append(csv_path)
        
        # Daily returns CSV (if available)
        if 'daily_returns' in analysis_results:
            csv_path = os.path.join(config.OUTPUTS_PATH, 'daily_returns.csv')
            analysis_results['daily_returns'].to_csv(csv_path)
            report_files.append(csv_path)
        
        # Cumulative returns CSV (if available)
        if 'cumulative_returns' in analysis_results:
            csv_path = os.path.join(config.OUTPUTS_PATH, 'cumulative_returns.csv')
            analysis_results['cumulative_returns'].to_csv(csv_path)
            report_files.append(csv_path)
        
        return report_files
        
    except Exception as e:
        print(f"Error generating CSV reports: {e}")
        return []


if __name__ == '__main__':
    print("Reporting module loaded successfully")
