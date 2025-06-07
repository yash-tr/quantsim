"""QuantSim Indicators Package.

This package provides common technical indicator calculation functions
designed to operate on pandas Series (typically of price data). These
indicators can be used by trading strategies for signal generation.

Currently implemented indicators:
- Average True Range (ATR) via `calculate_atr`
- Simple Moving Average (SMA) via `calculate_sma`

Future indicators (e.g., EMA, RSI, MACD) would be added here and
exported for easy access.
"""

from .atr import calculate_atr
from .sma import calculate_sma

# Define what is publicly available when 'from quantsim.indicators import *' is used
__all__ = [
    "calculate_atr",
    "calculate_sma",
]

# Example for future expansion:
# from .ema import calculate_ema
# __all__.append('calculate_ema')
