"""Implements a basic Momentum trading strategy.

This strategy identifies momentum by comparing the current price to a price
from a specified number of periods ago. It generates BUY signals for positive
momentum and SELL signals for negative momentum.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from quantsim.strategies.base import Strategy
from quantsim.core.events import (
    MarketEvent,
    FillEvent,
    SignalEvent,
)  # Added SignalEvent for on_signal
from quantsim.core.event_queue import EventQueue
from quantsim.indicators import calculate_atr


class MomentumStrategy(Strategy):
    """A momentum strategy based on historical price changes.

    This strategy enters a LONG position if the price has increased over a defined
    `momentum_window`, and a SHORT position if the price has decreased. It aims to
    capitalize on prevailing trends. It can optionally use limit orders for entry
    and place stop-loss orders upon fill.

    Attributes:
        momentum_window (int): The lookback period for calculating momentum.
        order_quantity (float): The fixed quantity of shares for orders.
        limit_order_offset_pct (float): Percentage offset for limit orders.
        stop_loss_pct (float): Percentage offset for stop-loss orders.
        atr_period (int): Period for ATR calculation (if used for orders/stops).
        prices (Dict[str, List[float]]): Stores recent close prices for each symbol.
        current_positions (Dict[str, Optional[str]]): Tracks the strategy's perceived
            current position ('LONG', 'SHORT', None) for each symbol.
        last_order_direction (Dict[str, Optional[str]]): Tracks the last order
            direction to prevent duplicate orders for the same continuous signal.
        atr_series (Dict[str, Optional[pd.Series]]): Pre-calculated ATR series per symbol.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        symbols: List[str],
        momentum_window: int = 20,
        order_quantity: float = 100.0,
        data_handler: Optional[Any] = None,
        atr_period: int = 0,
        limit_order_offset_pct: float = 0.0,
        stop_loss_pct: float = 0.0,
        **kwargs: Any,
    ):
        """Initializes the MomentumStrategy.

        Args:
            event_queue (EventQueue): The system's event queue.
            symbols (List[str]): A list of symbols this strategy will operate on.
            momentum_window (int, optional): Lookback window for momentum calculation.
                Defaults to 20.
            order_quantity (float, optional): Fixed quantity for orders. Defaults to 100.0.
            data_handler (Optional[Any], optional): Provides access to historical data.
                Used for pre-calculating ATR if `atr_period` > 0. Can be a pandas
                DataFrame (for single symbol context typically) or a DataHandler instance.
                Defaults to None.
            atr_period (int, optional): Period for ATR calculation. If > 0, ATR will be
                calculated and included in `OrderEvent`s. Defaults to 0.
            limit_order_offset_pct (float, optional): Percentage offset from current
                price for limit orders. If 0, market orders are used. Defaults to 0.0.
            stop_loss_pct (float, optional): Percentage offset from fill price for
                stop-loss orders. If 0, no stop-loss orders are placed. Defaults to 0.0.
            **kwargs (Any): Additional keyword arguments passed to the `Strategy` base class.
        """
        super().__init__(event_queue, symbols, data_handler=data_handler, **kwargs)
        self.momentum_window: int = momentum_window
        self.order_quantity: float = order_quantity
        # self.data_handler is set in base class
        self.atr_period: int = atr_period
        self.limit_order_offset_pct: float = limit_order_offset_pct
        self.stop_loss_pct: float = stop_loss_pct

        self.prices: Dict[str, List[float]] = {sym: [] for sym in self.symbols}
        self.current_positions: Dict[str, Optional[str]] = {
            sym: None for sym in self.symbols
        }
        self.last_order_direction: Dict[str, Optional[str]] = {
            sym: None for sym in self.symbols
        }
        self.atr_series: Dict[str, Optional[pd.Series]] = {
            sym: None for sym in self.symbols
        }

        if self.atr_period > 0 and self.data_handler is not None:
            for sym_for_atr in self.symbols:
                try:
                    symbol_df: Optional[pd.DataFrame] = None
                    if isinstance(
                        self.data_handler, pd.DataFrame
                    ):  # Single DataFrame for all symbols (common in tests)
                        symbol_df = self.data_handler
                    elif hasattr(
                        self.data_handler, "get_historical_data"
                    ):  # Full DataHandler object
                        symbol_df = self.data_handler.get_historical_data(sym_for_atr)

                    if symbol_df is not None and not symbol_df.empty:
                        if all(
                            col in symbol_df.columns for col in ["High", "Low", "Close"]
                        ):
                            self.atr_series[sym_for_atr] = calculate_atr(
                                symbol_df["High"],
                                symbol_df["Low"],
                                symbol_df["Close"],
                                self.atr_period,
                            )
                        else:
                            print(
                                f"Warning: MomentumStrategy ({sym_for_atr}) ATR calc: HLC columns missing for ATR."
                            )
                    elif symbol_df is None and not isinstance(
                        self.data_handler, pd.DataFrame
                    ):
                        print(
                            f"MomentumStrategy ({sym_for_atr}): No data from data_handler for ATR pre-calc."
                        )
                except Exception as e:
                    print(
                        f"MomentumStrategy ({sym_for_atr}): Error pre-calculating ATR: {e}"
                    )

        print(
            f"MomentumStrategy initialized for {self.symbols}, window: {self.momentum_window}, ATR: {self.atr_period if self.atr_period > 0 else 'N/A'}."
        )

    def on_market_data(self, event: MarketEvent) -> None:
        """Handles new market data to calculate momentum and generate orders.

        Calculates momentum as `(current_price / price_n_periods_ago) - 1`.
        Generates a BUY order for positive momentum if not already long,
        and a SELL order for negative momentum if not already short.
        Orders can be Market or Limit type.

        Args:
            event (MarketEvent): The market data event.
        """
        if event.symbol not in self.symbols:
            return

        self.prices[event.symbol].append(event.close)
        # Trim price list to save memory, keep enough for momentum window + a small buffer
        max_len = self.momentum_window + 5
        if len(self.prices[event.symbol]) > max_len:
            self.prices[event.symbol] = self.prices[event.symbol][-max_len:]

        if len(self.prices[event.symbol]) < self.momentum_window:
            return

        price_now = self.prices[event.symbol][-1]
        price_then = self.prices[event.symbol][
            -self.momentum_window
        ]  # Price N periods ago
        momentum_return = (price_now / price_then) - 1 if price_then != 0 else 0.0

        current_atr_val: Optional[float] = None
        symbol_atr_s = self.atr_series.get(event.symbol)  # Get Series for symbol
        if symbol_atr_s is not None:
            current_atr_val = symbol_atr_s.get(
                event.timestamp
            )  # Get ATR value for specific timestamp
            if pd.isna(current_atr_val):
                current_atr_val = None  # Ensure None if pandas returns NaN

        order_type: str = "MKT"
        limit_price: Optional[float] = None
        order_direction: Optional[str] = None

        if momentum_return > 0 and self.last_order_direction.get(event.symbol) != "BUY":
            order_direction = "BUY"
            if self.limit_order_offset_pct > 0:
                order_type = "LMT"
                limit_price = event.close * (1 - self.limit_order_offset_pct)
        elif (
            momentum_return < 0
            and self.last_order_direction.get(event.symbol) != "SELL"
        ):
            order_direction = "SELL"
            if self.limit_order_offset_pct > 0:
                order_type = "LMT"
                limit_price = event.close * (1 + self.limit_order_offset_pct)

        if order_direction:
            # TODO: Consider cancelling active stop-loss for this symbol before placing new main order
            self._generate_order(
                symbol=event.symbol,
                order_type=order_type,
                direction=order_direction,
                quantity=self.order_quantity,
                timestamp=event.timestamp,
                reference_price=event.close,
                limit_price=limit_price,
                current_atr=current_atr_val,
            )
            self.last_order_direction[event.symbol] = order_direction
            print(
                f"{event.timestamp} MomentumStrategy ({event.symbol}): Momentum {momentum_return:.2%}. Generated {order_direction} {order_type} order. RefPx: {event.close:.2f}"
                + (f", LimPx: {str(limit_price) if limit_price is not None else 'N/A'}")
            )

    def on_fill(self, event: FillEvent) -> None:
        """Handles `FillEvent` to update position state and place stop-loss orders.

        Args:
            event (FillEvent): The fill event confirming trade execution.
        """
        if event.symbol not in self.symbols:
            return

        is_opening_trade = (
            False  # True if this fill opens a new position or changes direction
        )
        current_intended_pos_before_fill = self.current_positions.get(event.symbol)

        if event.direction == "BUY":
            if current_intended_pos_before_fill != "LONG":
                is_opening_trade = True
            self.current_positions[event.symbol] = "LONG"
        elif event.direction == "SELL":
            if current_intended_pos_before_fill != "SHORT":
                is_opening_trade = True
            self.current_positions[event.symbol] = "SHORT"

        print(
            f"MomentumStrategy ({event.symbol}): Fill {event.direction} {event.quantity} @ {event.fill_price}. Intended Pos: {self.current_positions.get(event.symbol)}"
        )

        # Place stop-loss if it's an opening/flipping trade that established the current position
        if self.stop_loss_pct > 0 and is_opening_trade:
            stop_price: Optional[float] = None
            stop_direction: Optional[str] = None

            if self.current_positions.get(event.symbol) == "LONG":
                stop_price = event.fill_price * (1 - self.stop_loss_pct)
                stop_direction = "SELL"
            elif self.current_positions.get(event.symbol) == "SHORT":
                stop_price = event.fill_price * (1 + self.stop_loss_pct)
                stop_direction = "BUY"

            if stop_price is not None and stop_direction is not None:
                current_atr_val: Optional[float] = None
                symbol_atr_s = self.atr_series.get(event.symbol)
                if symbol_atr_s is not None:
                    current_atr_val = symbol_atr_s.get(event.timestamp)
                    if pd.isna(current_atr_val):
                        current_atr_val = None

                # TODO: Implement robust stop-loss management (e.g., cancel previous SL order).
                print(
                    f"MomentumStrategy ({event.symbol}): Issuing STP {stop_direction}, Qty: {event.quantity}, StopPx: {stop_price:.2f}"
                )
                self._generate_order(
                    symbol=event.symbol,
                    order_type="STP",
                    quantity=event.quantity,
                    direction=stop_direction,
                    timestamp=event.timestamp,
                    reference_price=event.fill_price,  # Base SL on actual fill price
                    stop_price=stop_price,
                    current_atr=current_atr_val,
                )

    def on_signal(self, event: SignalEvent) -> None:
        """Handles external `SignalEvent`s. (Not used by this strategy).

        Args:
            event (SignalEvent): The signal event.
        """
        pass  # This strategy generates its own orders based on market data.
