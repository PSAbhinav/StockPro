import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

def create_correlation_heatmap(correlation_matrix, output_dir):
    """Create and save correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.3f', ax=ax)
    plt.title('Correlation Matrix Between Stocks', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_cumulative_returns_individual(daily_returns, output_dir):
    """Plot individual stock cumulative returns."""
    fig, ax = plt.subplots(figsize=(14, 6))
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, col in enumerate(cumulative_returns.columns):
        ax.plot(cumulative_returns.index, cumulative_returns[col] * 100, 
                label=col, linewidth=2, color=colors[i])
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    ax.set_title('Individual Stock Cumulative Returns', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cumulative_returns_individual.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_portfolio_comparison(equal_weight_returns, inv_vol_returns, output_dir):
    """Compare cumulative returns of two portfolios."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    eq_cumulative = (1 + equal_weight_returns).cumprod() - 1
    iv_cumulative = (1 + inv_vol_returns).cumprod() - 1
    
    ax.plot(eq_cumulative.index, eq_cumulative * 100, 
            label='Equal Weight', linewidth=2.5, color='#2ca02c')
    ax.plot(iv_cumulative.index, iv_cumulative * 100, 
            label='Inverse Volatility', linewidth=2.5, color='#d62728')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    ax.set_title('Portfolio Comparison: Cumulative Returns', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'portfolio_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_risk_return_scatter(results_df, output_dir):
    """Create risk vs return scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(results_df['Annualized Volatility'] * 100, 
                        results_df['Annualized Return'] * 100,
                        c=results_df['Sharpe Ratio'], cmap='viridis', 
                        s=500, alpha=0.7, edgecolors='black', linewidth=2)
    
    for idx, row in results_df.iterrows():
        ax.text(row['Annualized Volatility'] * 100, row['Annualized Return'] * 100, 
                row['Stock Ticker'], fontsize=12, fontweight='bold', 
                ha='center', va='center')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=12)
    
    ax.set_xlabel('Annualized Volatility (Risk) %', fontsize=12)
    ax.set_ylabel('Annualized Return %', fontsize=12)
    ax.set_title('Risk vs Return Analysis for Top 5 Tech Stocks', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'risk_return_scatter.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_portfolio_weights_comparison(equal_weights, inv_vol_weights, output_dir):
    """Compare portfolio weights."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    tickers = list(equal_weights.keys())
    x = np.arange(len(tickers))
    width = 0.35
    
    eq_values = [equal_weights[t] for t in tickers]
    iv_values = [inv_vol_weights[t] for t in tickers]
    
    bars1 = ax.bar(x - width/2, eq_values, width, label='Equal Weight', 
                   color='#2ca02c', alpha=0.8)
    bars2 = ax.bar(x + width/2, iv_values, width, label='Inverse Volatility', 
                   color='#d62728', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Portfolio Weight', fontsize=12)
    ax.set_title('Portfolio Weights Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'portfolio_weights_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_candlestick(stock_data, ticker, output_dir, days=60):
    """Create candlestick chart for a stock."""
    # Get last N days
    data = stock_data.tail(days).copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Determine up and down days
    up = data[data['Close'] >= data['Open']]
    down = data[data['Close'] < data['Open']]
    
    # Plot up days (green)
    ax.bar(up.index, up['Close'] - up['Open'], bottom=up['Open'], 
           width=0.8, color='green', alpha=0.8)
    ax.vlines(up.index, up['Low'], up['High'], color='green', linewidth=0.5)
    
    # Plot down days (red)
    ax.bar(down.index, down['Open'] - down['Close'], bottom=down['Close'], 
           width=0.8, color='red', alpha=0.8)
    ax.vlines(down.index, down['Low'], down['High'], color='red', linewidth=0.5)
    
    ax.set_title(f'{ticker} - Last {days} Days', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'candlestick_{ticker}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_performance_dashboard(daily_returns, eq_returns, iv_returns, 
                              correlation_matrix, output_dir):
    """Create comprehensive performance dashboard."""
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: Equal weight portfolio cumulative returns
    ax1 = plt.subplot(2, 2, 1)
    eq_cumulative = (1 + eq_returns).cumprod() - 1
    ax1.plot(eq_cumulative.index, eq_cumulative * 100, linewidth=2.5, color='#2ca02c')
    ax1.set_title('Equal Weight Portfolio Cumulative Returns', fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Subplot 2: Inverse volatility portfolio cumulative returns
    ax2 = plt.subplot(2, 2, 2)
    iv_cumulative = (1 + iv_returns).cumprod() - 1
    ax2.plot(iv_cumulative.index, iv_cumulative * 100, linewidth=2.5, color='#d62728')
    ax2.set_title('Inverse Volatility Portfolio Cumulative Returns', fontweight='bold')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Subplot 3: All individual stocks
    ax3 = plt.subplot(2, 2, 3)
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    for col in cumulative_returns.columns:
        ax3.plot(cumulative_returns.index, cumulative_returns[col] * 100, 
                label=col, linewidth=2)
    ax3.set_title('Individual Stock Cumulative Returns', fontweight='bold')
    ax3.set_ylabel('Cumulative Return (%)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Subplot 4: Correlation heatmap
    ax4 = plt.subplot(2, 2, 4)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, fmt='.2f', ax=ax4, cbar_kws={"shrink": 0.8})
    ax4.set_title('Correlation Matrix', fontweight='bold')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax4.yaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'performance_dashboard.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_portfolio_allocation_pie(inv_vol_weights, output_dir):
    """Create pie chart for portfolio allocation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    tickers = list(inv_vol_weights.keys())
    sizes = [inv_vol_weights[t] for t in tickers]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    explode = [0.05] * len(tickers)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=tickers, autopct='%1.1f%%',
                                        colors=colors, explode=explode,
                                        shadow=True, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    ax.set_title('Portfolio Allocation (Inverse Volatility Weighted)', 
                 fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'portfolio_allocation_pie.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def plot_rolling_volatility(daily_returns, output_dir):
    """Plot rolling volatility over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 30-day rolling volatility
    rolling_30 = daily_returns.rolling(window=30).std()
    for col in rolling_30.columns:
        ax1.plot(rolling_30.index, rolling_30[col], label=col, linewidth=2)
    ax1.set_title('30-Day Rolling Volatility', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Rolling Volatility', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 90-day rolling volatility
    rolling_90 = daily_returns.rolling(window=90).std()
    for col in rolling_90.columns:
        ax2.plot(rolling_90.index, rolling_90[col], label=col, linewidth=2)
    ax2.set_title('90-Day Rolling Volatility', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Rolling Volatility', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'rolling_volatility.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()