"""
Defines position sizing logic for portfolio management.
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from quantsim.core.events import SignalEvent

if TYPE_CHECKING:
    from quantsim.portfolio.portfolio import Portfolio


class PositionSizer(ABC):
    """
    Abstract base class for position sizers.
    """

    @abstractmethod
    def size_order(
        self, portfolio: "Portfolio", signal: SignalEvent
    ) -> Optional[float]:
        """
        Calculates the size of an order.
        Returns None if the order should not be placed.
        """
        raise NotImplementedError("Should implement size_order()")


class FixedQuantitySizer(PositionSizer):
    """
    Sizes orders to a fixed quantity.
    """

    def __init__(self, quantity: float = 100.0):
        self.quantity = quantity

    def size_order(
        self, portfolio: "Portfolio", signal: SignalEvent
    ) -> Optional[float]:
        return self.quantity


class RiskPercentageSizer(PositionSizer):
    """
    Sizes orders based on a percentage of portfolio equity and risk per trade.
    """

    def __init__(self, risk_per_trade_pct: float = 0.02, stop_loss_pct: float = 0.05):
        if not 0 < risk_per_trade_pct < 1:
            raise ValueError("risk_per_trade_pct must be between 0 and 1.")
        if not 0 < stop_loss_pct < 1:
            raise ValueError("stop_loss_pct must be between 0 and 1.")

        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct

    def size_order(
        self, portfolio: "Portfolio", signal: SignalEvent
    ) -> Optional[float]:
        """
        Calculates order size based on risk percentage.
        """
        portfolio_value = portfolio.get_equity()
        risk_amount = portfolio_value * self.risk_per_trade_pct

        last_close = portfolio.get_last_close_price(signal.symbol)
        if last_close is None:
            return None

        stop_price = (
            last_close * (1 - self.stop_loss_pct)
            if signal.direction == "LONG"
            else last_close * (1 + self.stop_loss_pct)
        )
        risk_per_share = abs(last_close - stop_price)

        if risk_per_share <= 1e-9:
            return None

        quantity = risk_amount / risk_per_share
        return quantity if quantity > 0 else None
