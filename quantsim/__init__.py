"""
QuantSim: Event-Driven Backtesting Framework

An event-driven backtesting and execution simulation engine for quantitative trading strategies.

Author: Yash Tripathi
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Yash Tripathi"
__email__ = "tripathiyash1004@gmail.com"
__license__ = "MIT"

# Core imports for easy access
from quantsim.core.event_queue import EventQueue
from quantsim.core.simulation_engine import SimulationEngine
from quantsim.core.events import MarketEvent, OrderEvent, FillEvent, SignalEvent

# Data handlers
from quantsim.data.yahoo_finance import YahooFinanceDataHandler
from quantsim.data.csv_parser import CSVDataManager
from quantsim.data.synthetic_data import SyntheticDataHandler

# Strategies
from quantsim.strategies.sma_crossover import SMACrossoverStrategy
from quantsim.strategies.momentum import MomentumStrategy
from quantsim.strategies.mean_reversion import MeanReversionStrategy

# Portfolio management
from quantsim.portfolio.portfolio import Portfolio
from quantsim.portfolio.position import Position

# Execution
from quantsim.execution.execution_handler import SimulatedExecutionHandler

# Indicators
from quantsim.indicators import calculate_sma, calculate_atr

__all__ = [
    # Core
    "EventQueue",
    "SimulationEngine",
    "MarketEvent",
    "OrderEvent",
    "FillEvent",
    "SignalEvent",
    # Data handlers
    "YahooFinanceDataHandler",
    "CSVDataManager",
    "SyntheticDataHandler",
    # Strategies
    "SMACrossoverStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    # Portfolio
    "Portfolio",
    "Position",
    # Execution
    "SimulatedExecutionHandler",
    # Indicators
    "calculate_sma",
    "calculate_atr",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
