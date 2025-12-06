# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview
Financial portfolio analysis tool that fetches stock data, analyzes portfolio performance, and generates visualizations. Built with Python using yfinance for market data, pandas for analysis, and matplotlib/seaborn for visualizations.

## Environment Setup
**Virtual Environment**: This project uses a Python virtual environment located in `venv/`

Activate the environment:
```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies (when requirements.txt is created):
```powershell
python -m pip install -r requirements.txt
```

Key packages: yfinance, pandas, numpy, matplotlib, seaborn, scipy

## Architecture

### Module Structure
- **scripts/data_fetcher.py**: Fetches financial data from Yahoo Finance API using yfinance
- **scripts/portfolio_analyzer.py**: Performs portfolio analysis (returns, risk metrics, correlations)
- **scripts/visualizations.py**: Creates charts and graphs for portfolio performance
- **scripts/main.py**: Entry point that orchestrates the data pipeline

### Data Flow
1. `data_fetcher.py` pulls stock prices and saves to `data/` (CSV files)
2. `portfolio_analyzer.py` processes data and computes metrics
3. `visualizations.py` generates charts saved to `figures/`
4. Analysis results saved to `outputs/`

### Directory Purpose
- **data/**: Raw stock price data (CSV files, git-ignored)
- **figures/**: Generated visualizations (PNG/PDF charts)
- **outputs/**: Analysis results and reports
- **notebooks/**: Jupyter notebooks for exploratory analysis
- **scripts/**: Core Python modules

## Running the Project
Execute main analysis:
```powershell
python scripts\main.py
```

Run individual modules:
```powershell
python scripts\data_fetcher.py
python scripts\portfolio_analyzer.py
python scripts\visualizations.py
```

## Development Notes
- Data files (*.csv) are git-ignored to avoid committing large datasets
- The project is in early stages - most scripts are currently empty
- When implementing yfinance calls, use appropriate date ranges and handle API rate limits
- Portfolio data should include ticker symbols, quantities, and purchase dates
