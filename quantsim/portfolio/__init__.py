"""QuantSim Portfolio Management Package.

This package contains components crucial for tracking and evaluating trading
performance within the QuantSim backtesting framework.

Key components include:
-   `Position`: Represents a holding in a single financial instrument, detailing
    quantity, average price, and current market value.
-   `Portfolio`: The central class for managing the overall trading account. It
    handles cash, multiple `Position` objects, processes `FillEvent`s to update
    state, tracks equity over time, and calculates a comprehensive suite of
    performance metrics and trade statistics.
-   `TradeLog`: Reconstructs round-trip trades from individual fill events,
    providing a detailed history of trading activity and enabling the calculation
    of trade-specific statistics.
-   `Trade` (dataclass) and `DetailedFill` (NamedTuple): Data structures used by
    `TradeLog` to store information about trades and their constituent fills.

These components work together to provide a clear picture of a strategy's
behavior and profitability.
"""

from .position import Position
from .portfolio import Portfolio
from .trade_log import TradeLog, Trade, DetailedFill

__all__ = [
    "Position",
    "Portfolio",
    "TradeLog",
    "Trade",
    "DetailedFill",
]
