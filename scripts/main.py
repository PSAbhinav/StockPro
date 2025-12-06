"""
Financial Portfolio Analysis - Main Script
Orchestrates the complete portfolio analysis pipeline.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import traceback
from datetime import datetime

# Import configuration and modules
import config

# Setup logging
def setup_logging():
    """Configure logging for both file and console output."""
    # Create outputs directory if it doesn't exist
    os.makedirs(config.OUTPUTS_PATH, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('portfolio_analyzer')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    """Main function to orchestrate the portfolio analysis."""
    start_time = datetime.now()
    logger = setup_logging()
    
    try:
        logger.info("="*70)
        logger.info("FINANCIAL PORTFOLIO ANALYSIS - STARTING")
        logger.info("="*70)
        logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Analyzing stocks: {', '.join(config.TICKERS)}")
        logger.info(f"Period: {config.START_DATE} to {config.END_DATE}")
        
        # Import analysis modules
        logger.info("\nImporting analysis modules...")
        from data_fetcher import fetch_stock_data
        from portfolio_analyzer import analyze_portfolio
        from visualizations import generate_all_visualizations
        
        # Phase 1: Data Fetching
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: DATA FETCHING")
        logger.info("="*70)
        logger.info("Downloading stock data from Yahoo Finance...")
        
        stock_data = fetch_stock_data(
            config.TICKERS,
            config.START_DATE,
            config.END_DATE,
            config.DATA_PATH
        )
        
        if stock_data is None or stock_data.empty:
            logger.error("Failed to fetch stock data!")
            return 1
        
        logger.info(f"Successfully downloaded {len(stock_data)} days of data")
        logger.info(f"Data shape: {stock_data.shape}")
        
        # Data quality checks
        missing_values = stock_data.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in data")
        else:
            logger.info("Data quality check passed - no missing values")
        
        # Phase 2: Portfolio Analysis
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: PORTFOLIO ANALYSIS")
        logger.info("="*70)
        logger.info("Calculating returns, volatility, and risk metrics...")
        
        analysis_results = analyze_portfolio(stock_data, config)
        
        if analysis_results is None:
            logger.error("Portfolio analysis failed!")
            return 1
        
        logger.info("Portfolio analysis completed successfully")
        logger.info(f"Calculated metrics for {len(analysis_results['individual_stocks'])} stocks")
        
        # Phase 3: Visualization Generation
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: VISUALIZATION GENERATION")
        logger.info("="*70)
        logger.info("Creating charts and visualizations...")
        
        viz_count = generate_all_visualizations(
            stock_data,
            analysis_results,
            config.FIGURES_PATH,
            config
        )
        
        logger.info(f"Generated {viz_count} visualizations successfully")
        
        # Phase 4: Report Generation
        logger.info("\n" + "="*70)
        logger.info("PHASE 4: REPORT GENERATION")
        logger.info("="*70)
        logger.info("Creating comprehensive reports...")
        
        from reporting import generate_reports
        report_files = generate_reports(analysis_results, config)
        
        logger.info(f"Generated {len(report_files)} report files")
        for report_file in report_files:
            logger.info(f"  - {os.path.basename(report_file)}")
        
        # Completion
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total Duration: {duration:.2f} seconds")
        logger.info(f"\nResults saved to:")
        logger.info(f"  - Data: {config.DATA_PATH}")
        logger.info(f"  - Outputs: {config.OUTPUTS_PATH}")
        logger.info(f"  - Figures: {config.FIGURES_PATH}")
        logger.info(f"  - Logs: {config.LOG_FILE}")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("ERROR: Analysis failed!")
        logger.error("="*70)
        logger.error(f"Error message: {str(e)}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("="*70)
        return 1

if __name__ == '__main__':
    """Entry point for the script."""
    exit_code = main()
    sys.exit(exit_code)
