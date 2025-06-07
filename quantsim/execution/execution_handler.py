"""Handles order execution simulation, including slippage, commission, and latency.

This module defines the `ExecutionHandler` abstract base class for processing
`OrderEvent`s and generating `FillEvent`s. A concrete implementation,
`SimulatedExecutionHandler`, is provided, which models various aspects of
real-world trading such as order types (Market, Limit, Stop), slippage,
commission, fill latency, and partial fills.
"""

import datetime
from abc import ABC, abstractmethod
from typing import Optional, Any

from quantsim.core.events import OrderEvent, FillEvent, MarketEvent
from quantsim.core.event_queue import EventQueue
from .slippage import SlippageModel, PercentageSlippage

# TODO: Consider moving CommissionModel and its implementations to a separate commission.py


class CommissionModel(ABC):
    """Abstract base class for all commission models.

    Custom commission models should inherit from this class and implement the
    `calculate_commission` method. This allows for flexible commission structures
    to be integrated into the simulation.

    To create a custom commission model:
    1. Inherit from this `CommissionModel` class.
    2. Implement the `calculate_commission` method.
    3. The `calculate_commission` method must accept `quantity` (float: the number
       of shares/contracts traded, always positive) and `fill_price` (float: the
       price at which the trade was executed).
    4. The method should return the calculated commission amount (float), which is
       always a positive value representing the cost.
    """

    @abstractmethod
    def calculate_commission(self, quantity: float, fill_price: float) -> float:
        """Calculates the commission for a trade.

        Args:
            quantity (float): The absolute quantity of shares/contracts traded.
            fill_price (float): The price at which the shares/contracts were traded.

        Returns:
            float: The calculated commission amount (should be non-negative).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Should implement calculate_commission()")


class FixedCommission(CommissionModel):
    """A simple commission model that charges a fixed amount per trade.

    Attributes:
        commission_per_trade (float): The fixed commission amount for each trade.
                                     Must be non-negative.
    """

    def __init__(self, commission_per_trade: float = 1.0):
        """Initializes the FixedCommission model.

        Args:
            commission_per_trade (float, optional): The fixed commission amount
                to charge per trade. Defaults to 1.0.

        Raises:
            ValueError: If `commission_per_trade` is negative.
        """
        if commission_per_trade < 0:
            raise ValueError("Commission per trade cannot be negative.")
        self.commission_per_trade: float = commission_per_trade

    def calculate_commission(self, quantity: float, fill_price: float) -> float:
        """Calculates the fixed commission for a trade.

        Args:
            quantity (float): The quantity of shares traded (ignored by this model).
            fill_price (float): The price at which shares were traded (ignored by this model).

        Returns:
            float: The fixed commission amount.
        """
        return self.commission_per_trade


class PerShareCommission(CommissionModel):
    """A commission model that charges a fixed amount per share/contract traded.

    This model can also include a minimum commission per trade.

    Attributes:
        commission_per_share (float): The commission charged for each share traded.
                                     Must be non-negative.
        min_commission (float): The minimum commission amount per trade.
                               Must be non-negative.
    """

    def __init__(
        self, commission_per_share: float = 0.005, min_commission: float = 1.0
    ):
        """Initializes the PerShareCommission model.

        Args:
            commission_per_share (float, optional): Commission per share. Defaults to 0.005.
            min_commission (float, optional): Minimum commission per trade. Defaults to 1.0.

        Raises:
            ValueError: If `commission_per_share` or `min_commission` are negative.
        """
        if commission_per_share < 0:
            raise ValueError("Commission per share cannot be negative.")
        if min_commission < 0:
            raise ValueError("Minimum commission cannot be negative.")
        self.commission_per_share: float = commission_per_share
        self.min_commission: float = min_commission

    def calculate_commission(self, quantity: float, fill_price: float) -> float:
        """Calculates commission based on shares traded, subject to a minimum.

        Args:
            quantity (float): The absolute quantity of shares traded.
            fill_price (float): The price at which shares were traded (ignored by this model).

        Returns:
            float: The calculated commission, subject to the minimum.
        """
        commission = abs(quantity) * self.commission_per_share
        return max(commission, self.min_commission)


class ExecutionHandler(ABC):
    """Abstract base class for all execution handlers.

    The role of an ExecutionHandler is to take `OrderEvent`s from the event queue
    and simulate their execution in the market, ultimately producing `FillEvent`s
    which are then placed back onto the event queue.

    Attributes:
        event_queue (EventQueue): The central event queue for the system.
    """

    def __init__(self, event_queue: EventQueue):
        """Initializes the ExecutionHandler.

        Args:
            event_queue (EventQueue): The system's event queue where FillEvents will be placed.
        """
        self.event_queue: EventQueue = event_queue

    @abstractmethod
    def execute_order(self, event: OrderEvent) -> None:
        """Processes an OrderEvent and should generate a FillEvent if filled.

        Concrete implementations will define how orders are treated (e.g., market,
        limit, stop), how slippage and commission are applied, and if there's
        any latency or partial fill logic.

        Args:
            event (OrderEvent): The order to be executed.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    """Simulates order execution, applying slippage, commission, and latency.

    This handler processes `OrderEvent`s for Market ('MKT'), Limit ('LMT'), and
    Stop ('STP') orders. It uses configurable models for slippage and commission.
    It also simulates execution latency and partial fills based on per-bar limits.
    Limit and Stop orders are treated as "Immediate Or Cancel" (IOC) against the
    current bar's reference price.

    Attributes:
        slippage_model (SlippageModel): The model used to simulate slippage.
        commission_model (CommissionModel): The model used to calculate commissions.
        latency_duration (datetime.timedelta): Simulated delay for fill events.
        max_fill_pct_per_bar (float): Max percentage of an order fillable in one bar.
        max_fill_qty_per_bar (float): Max absolute quantity fillable in one bar.
        volume_limit_pct_per_bar (float): Max percentage of bar volume fillable in one bar.
        market_data_provider (Optional[Any]): Optional provider for live market prices
            (not fully implemented for use in this version, relies on OrderEvent.reference_price).
    """

    def __init__(
        self,
        event_queue: EventQueue,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        latency_ms: int = 0,
        max_fill_pct_per_bar: float = 1.0,
        max_fill_qty_per_bar: float = float("inf"),
        volume_limit_pct_per_bar: float = 1.0,
        market_data_provider: Optional[Any] = None,
    ):
        """Initializes the SimulatedExecutionHandler.

        Args:
            event_queue (EventQueue): The system's event queue.
            slippage_model (Optional[SlippageModel], optional): Model for slippage.
                Defaults to `PercentageSlippage(0.001)`.
            commission_model (Optional[CommissionModel], optional): Model for commission.
                Defaults to `FixedCommission(1.0)`.
            latency_ms (int, optional): Simulated latency in milliseconds. Defaults to 0.
            max_fill_pct_per_bar (float, optional): Maximum percentage (0.0 to 1.0)
                of an order that can be filled in one bar. Defaults to 1.0 (100%).
            max_fill_qty_per_bar (float, optional): Maximum absolute quantity of an
                order that can be filled in one bar. Defaults to `float('inf')`.
            volume_limit_pct_per_bar (float, optional): Maximum percentage of bar volume
                fillable in one bar. Defaults to 1.0 (100%).
            market_data_provider (Optional[Any], optional): A provider for current market
                prices if `OrderEvent.reference_price` is not sufficient. Defaults to None.

        Raises:
            ValueError: If `max_fill_pct_per_bar`, `max_fill_qty_per_bar`, or `volume_limit_pct_per_bar` are invalid.
        """
        super().__init__(event_queue)
        self.slippage_model: SlippageModel = (
            slippage_model if slippage_model is not None else PercentageSlippage()
        )
        self.commission_model: CommissionModel = (
            commission_model if commission_model is not None else FixedCommission()
        )
        self.latency_duration: datetime.timedelta = datetime.timedelta(
            milliseconds=latency_ms
        )

        if not (0.0 <= max_fill_pct_per_bar <= 1.0):
            raise ValueError("max_fill_pct_per_bar must be between 0.0 and 1.0.")
        if max_fill_qty_per_bar < 0:
            raise ValueError("max_fill_qty_per_bar cannot be negative.")
        if not (0.0 <= volume_limit_pct_per_bar <= 1.0):
            raise ValueError("volume_limit_pct_per_bar must be between 0.0 and 1.0.")

        self.max_fill_pct_per_bar: float = max_fill_pct_per_bar
        self.max_fill_qty_per_bar: float = max_fill_qty_per_bar
        self.volume_limit_pct_per_bar: float = volume_limit_pct_per_bar

        self.market_data_provider: Optional[Any] = market_data_provider

        init_messages = []
        if latency_ms > 0:
            init_messages.append(f"latency: {latency_ms}ms")
        if max_fill_pct_per_bar < 1.0:
            init_messages.append(f"max_fill_pct: {max_fill_pct_per_bar*100:.1f}%")
        if max_fill_qty_per_bar != float("inf"):
            init_messages.append(f"max_fill_qty: {max_fill_qty_per_bar}")
        if volume_limit_pct_per_bar < 1.0:
            init_messages.append(
                f"volume_limit_pct: {volume_limit_pct_per_bar*100:.1f}%"
            )
        if init_messages:
            print(
                f"SimulatedExecutionHandler initialized with ({', '.join(init_messages)})."
            )

    def execute_order(
        self, event: OrderEvent, market_event: Optional["MarketEvent"] = None
    ) -> None:
        """Simulates executing an order, potentially creating a FillEvent.
        Handles Market, Limit, and Stop orders. Applies configured slippage,
        commission, latency, and partial fill logic. Limit and Stop orders are
        treated as IOC based on `event.reference_price`.
        Args:
            event (OrderEvent): The order to be executed.
            market_event (Optional[MarketEvent]): The market event that triggered this execution.
                                                  Used for accessing bid/ask prices.
        """
        if event.type != "ORDER":
            return
        if not market_event:
            return

        if event.symbol != market_event.symbol:
            return

        requested_quantity = event.quantity
        fill_price = 0.0
        order_triggers = False

        # For market orders, the execution price is based on the bid/ask spread
        if event.order_type == "MKT":
            order_triggers = True
            fill_price = (
                market_event.ask_price
                if event.direction == "BUY"
                else market_event.bid_price
            )

        # For limit orders
        elif event.order_type == "LMT":
            if hasattr(event, "limit_price") and event.limit_price is not None:
                if event.direction == "BUY" and market_event.close <= event.limit_price:
                    order_triggers = True
                    fill_price = min(event.limit_price, market_event.ask_price)
                elif (
                    event.direction == "SELL"
                    and market_event.close >= event.limit_price
                ):
                    order_triggers = True
                    fill_price = max(event.limit_price, market_event.bid_price)

        # For stop orders
        elif event.order_type == "STP":
            if hasattr(event, "stop_price") and event.stop_price is not None:
                if event.direction == "BUY" and market_event.close >= event.stop_price:
                    order_triggers = True
                    fill_price = (
                        market_event.ask_price
                    )  # Triggered, becomes a market order
                elif (
                    event.direction == "SELL" and market_event.close <= event.stop_price
                ):
                    order_triggers = True
                    fill_price = (
                        market_event.bid_price
                    )  # Triggered, becomes a market order

        if not order_triggers:
            return  # Order did not meet conditions to execute on this bar

        if fill_price is None or fill_price == 0.0:
            return

        # --- Slippage Calculation ---
        final_fill_price = self._calculate_slippage(event, fill_price, market_event)

        # --- Partial Fill Logic ---
        fillable_by_pct = round(requested_quantity * self.max_fill_pct_per_bar)
        fillable_by_qty = self.max_fill_qty_per_bar

        market_volume = market_event.volume if market_event else 0
        fillable_by_volume = (
            round(market_volume * self.volume_limit_pct_per_bar)
            if market_volume > 0
            else float("inf")
        )

        actual_fill_quantity = min(
            requested_quantity, fillable_by_pct, fillable_by_qty, fillable_by_volume
        )

        if actual_fill_quantity <= 1e-9:
            return

        # --- Commission Calculation ---
        commission = self._calculate_commission(actual_fill_quantity, final_fill_price)

        # --- Create and Queue Fill Event ---
        base_timestamp = market_event.timestamp
        fill_event_timestamp = base_timestamp + self.latency_duration
        if fill_event_timestamp <= base_timestamp:
            fill_event_timestamp = base_timestamp + datetime.timedelta(microseconds=1)

        fill_event = FillEvent(
            symbol=event.symbol,
            quantity=actual_fill_quantity,
            direction=event.direction,
            fill_price=final_fill_price,
            commission=commission,
            exchange="SIMULATED",
            timestamp=fill_event_timestamp,
            order_id=event.order_id,
        )
        self.event_queue.put_event(fill_event)

        remaining_quantity = requested_quantity - actual_fill_quantity
        if remaining_quantity > 1e-9:
            print(
                f"SimulatedExecutionHandler: Order {event.order_id} for {event.symbol} partially filled. "
                f"Filled: {actual_fill_quantity}, Remainder: {remaining_quantity} (Note: Remainder is currently ignored)."
            )

    def _calculate_slippage(
        self,
        order_event: OrderEvent,
        market_price: float,
        market_event: Optional["MarketEvent"] = None,
    ) -> float:
        """Applies the configured slippage model to the market price.
        Args:
            order_event (OrderEvent): The original order event.
            market_price (float): The ideal execution price before slippage.
            market_event (Optional[MarketEvent]): The market event, used for bid/ask spread.
        Returns:
            float: The execution price after applying slippage.
        """
        # The slippage model now gets the market_event for more context
        return self.slippage_model.calculate_slippage(
            order_event, market_price, market_event
        )

    def _calculate_commission(self, quantity: float, fill_price: float) -> float:
        """Calculates commission using the configured commission model.

        Args:
            quantity (float): The absolute quantity of shares filled.
            fill_price (float): The price at which shares were filled.

        Returns:
            float: The calculated commission amount. Returns 0.0 if no model.
        """
        if self.commission_model:
            return self.commission_model.calculate_commission(
                abs(quantity), fill_price
            )  # Ensure positive quantity
        return 0.0
