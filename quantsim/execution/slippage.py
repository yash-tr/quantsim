"""Defines slippage models for simulated order execution.

This module provides an abstract base class `SlippageModel` and concrete
implementations such as `PercentageSlippage` and `ATRSlippage`. These models
are used by the `SimulatedExecutionHandler` to adjust fill prices from the
reference market price, simulating market impact or adverse price movement.
"""

from abc import ABC, abstractmethod
from typing import Optional

# Forward declaration for type hinting
if "quantsim.core.events" not in __import__("sys").modules:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from quantsim.core.events import OrderEvent, MarketEvent


class SlippageModel(ABC):
    """Abstract base class for all slippage models.

    Custom slippage models should inherit from this class and implement the
    `calculate_slippage` method. This design allows for easy extension of
    slippage simulation logic.

    To create a custom slippage model:
    1. Inherit from this `SlippageModel` class.
    2. Implement the `calculate_slippage` method.
    3. The `calculate_slippage` method must accept `order_event` (OrderEvent) and
       `market_price` (float: the price before slippage), and `market_event` (Optional[MarketEvent]).
    4. Use `market_event` to access any additional data your model might need. For example,
       if your model requires the current Average True Range (ATR), the `OrderEvent`
       should be augmented to carry this `current_atr` value. The `SimulatedExecutionHandler`
       will then pass `atr_value=order_event.current_atr` when calling your model.
       Your implementation would then retrieve it via `atr_value = market_event.atr_value`.
    5. The method should return the calculated fill price (float) after applying slippage.
    """

    @abstractmethod
    def calculate_slippage(
        self,
        order_event: "OrderEvent",
        market_price: float,
        market_event: Optional["MarketEvent"] = None,
    ) -> float:
        """Calculates the fill price after applying slippage.

        Args:
            order_event (OrderEvent): The order being executed.
            market_price (float): The ideal execution price before slippage
                                  (e.g., based on bid/ask or limit price).
            market_event (Optional[MarketEvent]): The market data event that
                triggered the execution. Can be used for context like bid-ask spread.

        Returns:
            float: The calculated fill price after applying slippage.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Should implement calculate_slippage()")


class PercentageSlippage(SlippageModel):
    """Applies a fixed percentage-based slippage to the market price.

    For BUY orders, slippage increases the price. For SELL orders, it decreases the price.

    Attributes:
        slippage_rate (float): The slippage rate (e.g., 0.001 for 0.1%). Must be >= 0 and < 1.
    """

    def __init__(self, slippage_rate: float = 0.001):
        """Initializes the PercentageSlippage model.

        Args:
            slippage_rate (float, optional): The slippage rate to apply.
                E.g., 0.001 means 0.1% slippage. Defaults to 0.001.

        Raises:
            ValueError: If `slippage_rate` is negative or greater than or equal to 1.
        """
        if not 0 <= slippage_rate < 1:
            raise ValueError(
                "Slippage rate must be between 0.0 (inclusive) and 1.0 (exclusive)."
            )
        self.slippage_rate: float = slippage_rate

    def calculate_slippage(
        self,
        order_event: "OrderEvent",
        market_price: float,
        market_event: Optional["MarketEvent"] = None,
    ) -> float:
        """Calculates fill price with percentage slippage.

        Args:
            order_event (OrderEvent): The order being executed.
            market_price (float): Reference market price.
            market_event (Optional[MarketEvent]): Ignored by this model.

        Returns:
            float: Fill price after slippage.
        """
        if order_event.direction == "BUY":
            return market_price * (1 + self.slippage_rate)
        elif order_event.direction == "SELL":
            return market_price * (1 - self.slippage_rate)
        # This case should ideally be prevented by validation of order_direction before calling.
        # print(f"Warning: PercentageSlippage received unrecognized order_direction '{order_event.direction}'. Applying no slippage.")
        return market_price


class ATRSlippage(SlippageModel):
    """Applies slippage based on a multiple of the Average True Range (ATR).

    This model uses the ATR value from the `OrderEvent` to calculate slippage.
    For BUY orders, slippage increases the price; for SELL orders, it decreases it.

    Attributes:
        atr_multiplier (float): The multiplier for the ATR value. Must be non-negative.
    """

    def __init__(self, atr_multiplier: float = 0.5):
        """Initializes the ATRSlippage model.

        Args:
            atr_multiplier (float, optional): Multiplier for the ATR value. Defaults to 0.5.

        Raises:
            ValueError: If `atr_multiplier` is negative.
        """
        if atr_multiplier < 0:
            raise ValueError("ATR multiplier cannot be negative.")
        self.atr_multiplier: float = atr_multiplier

    def calculate_slippage(
        self,
        order_event: "OrderEvent",
        market_price: float,
        market_event: Optional["MarketEvent"] = None,
    ) -> float:
        """Calculates fill price with ATR-based slippage.

        Args:
            order_event (OrderEvent): The order, which should contain `current_atr`.
            market_price (float): Reference market price.
            market_event (Optional[MarketEvent]): Ignored by this model.

        Returns:
            float: Fill price after slippage.
        """
        atr_value = order_event.current_atr

        if atr_value is not None and atr_value > 0:
            slippage_amount = atr_value * self.atr_multiplier
            if order_event.direction == "BUY":
                return market_price + slippage_amount
            elif order_event.direction == "SELL":
                return market_price - slippage_amount
        # No ATR value provided or not positive, so no ATR-based slippage.
        # print("ATRSlippage: ATR value not available or not positive. Applying zero slippage for ATR component.")
        return market_price
