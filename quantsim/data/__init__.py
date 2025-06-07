"""QuantSim Data Handling Package.

This package provides the core infrastructure for market data handling within
the QuantSim backtesting framework. It includes:

- An abstract base class `DataHandler` that defines the common interface
  for all data providers.
- Concrete implementations for various data sources:
    - `CSVDataManager`: For loading historical OHLCV data from CSV files.
    - `YahooFinanceDataHandler`: For fetching historical OHLCV data from Yahoo Finance.
    - `SyntheticDataHandler`: For generating synthetic OHLCV data for testing and
      experimentation.
- Utility functions, such as `load_csv_data` for direct loading of CSVs into DataFrames.

These components are designed to be used by the `SimulationEngine` to feed market
data to trading strategies and other modules during a backtest.
"""

from .base import DataHandler
from .csv_parser import CSVDataManager, load_csv_data
from .yahoo_finance import YahooFinanceDataHandler
from .synthetic_data import SyntheticDataHandler
