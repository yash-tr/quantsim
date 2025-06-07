"""
This module contains the strategy classes for the QuantSim framework.
"""

from .base import Strategy
from .sma_crossover import SMACrossoverStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .pairs_trading import PairsTradingStrategy
from .simple_ml import SimpleMLStrategy

__all__ = [
    "Strategy",
    "SMACrossoverStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "PairsTradingStrategy",
    "SimpleMLStrategy",
]
