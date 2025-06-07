"""QuantSim Execution Handling Package.

This package is responsible for simulating the process of order execution
in a financial market. It defines the interfaces and provides concrete
implementations for:

-   **Execution Handlers**: Components that take `OrderEvent`s and generate
    `FillEvent`s, simulating how orders are processed by a broker or exchange.
    Includes `SimulatedExecutionHandler` which models various execution complexities.
-   **Slippage Models**: Used by the `SimulatedExecutionHandler` to adjust fill
    prices from the reference market price, simulating market impact or price
    movement between order generation and execution. Examples include
    `PercentageSlippage` and `ATRSlippage`.
-   **Commission Models**: Used by the `SimulatedExecutionHandler` to calculate
    brokerage commissions on trades. Examples include `FixedCommission` and
    `PerShareCommission`.

The main goal is to provide a realistic yet configurable simulation of
trade execution to make backtest results more robust.
"""

from .execution_handler import (
    ExecutionHandler,
    SimulatedExecutionHandler,
    CommissionModel,  # Defined in execution_handler.py
    FixedCommission,
    PerShareCommission,
)
from .slippage import SlippageModel, PercentageSlippage, ATRSlippage

__all__ = [
    "ExecutionHandler",
    "SimulatedExecutionHandler",
    "CommissionModel",
    "FixedCommission",
    "PerShareCommission",
    "SlippageModel",
    "PercentageSlippage",
    "ATRSlippage",
]
