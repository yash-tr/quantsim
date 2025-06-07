"""
Position management for individual asset holdings.
"""

import pandas as pd
from typing import Optional


class Position:
    """Represents a single trading position in a portfolio.

    Tracks quantity, average price, and current market value for an asset.
    """

    def __init__(
        self,
        symbol: str,
        quantity: float,
        average_price: float,
        last_price: Optional[float] = None,
    ):
        """Initialize a position.

        Args:
            symbol: The asset symbol
            quantity: Number of shares/units (positive for long, negative for short)
            average_price: Average entry price
            last_price: Current market price
        """
        self.symbol = symbol
        self.quantity = quantity
        self.average_price = average_price
        self.avg_price = average_price  # Alias for compatibility
        self.last_price = last_price if last_price is not None else average_price
        self.last_update_time: Optional[pd.Timestamp] = None

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.last_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position."""
        return self.quantity * self.average_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return (self.quantity * self.last_price) - (self.quantity * self.average_price)

    def update_last_price(
        self, price: float, timestamp: Optional[pd.Timestamp] = None
    ) -> None:
        """Update the position's last price."""
        self.last_price = price
        self.last_update_time = timestamp

    def transact(self, quantity: float, price: float) -> None:
        """Add to or reduce the position.

        Args:
            quantity: Change in quantity (positive for buy, negative for sell)
            price: Transaction price
        """
        if self.quantity == 0:
            # Opening new position
            self.quantity = quantity
            self.average_price = price
            self.avg_price = price
        elif (self.quantity > 0 and quantity > 0) or (
            self.quantity < 0 and quantity < 0
        ):
            # Adding to existing position
            total_cost = (self.quantity * self.average_price) + (quantity * price)
            self.quantity += quantity
            if self.quantity != 0:
                self.average_price = total_cost / self.quantity
                self.avg_price = self.average_price
        else:
            # Reducing or flipping position
            new_quantity = self.quantity + quantity

            if abs(new_quantity) < 1e-9:
                # Position exactly closed
                self.quantity = 0.0
                self.average_price = 0.0
                self.avg_price = 0.0
            elif (self.quantity > 0 and new_quantity > 0) or (
                self.quantity < 0 and new_quantity < 0
            ):
                # Reducing position but staying same direction - keep average price
                self.quantity = new_quantity
                # average_price stays the same
            else:
                # Position flipped - new average price is the transaction price for the flipped portion
                self.quantity = new_quantity
                self.average_price = price
                self.avg_price = price

        self.last_price = price

    def __repr__(self) -> str:
        return f"Position(symbol='{self.symbol}', quantity={self.quantity}, average_price={self.average_price}, last_price={self.last_price})"
