"""
QuantSim Data Package

This package provides data handling and sourcing capabilities for backtesting.
It includes handlers for CSV files, Yahoo Finance API, and synthetic data generation.
"""

from .base import DataHandler
from .csv_parser import CSVDataManager, load_csv_data
from .yahoo_finance import YahooFinanceDataHandler
from .synthetic_data import SyntheticDataHandler

__all__ = [
    "DataHandler",
    "CSVDataManager",
    "load_csv_data",
    "YahooFinanceDataHandler",
    "SyntheticDataHandler",
]
