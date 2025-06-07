"""Implements a basic Mean Reversion trading strategy.

This strategy assumes that prices will tend to revert to a historical mean,
typically represented by a Simple Moving Average (SMA). It generates BUY signals
when the price falls significantly below the SMA and SELL signals when the price
rises significantly above it.
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
from quantsim.indicators import calculate_sma, calculate_atr


class MeanReversionStrategy(Strategy):
    """A mean reversion strategy based on deviation from a Simple Moving Average.

    This strategy calculates an SMA for each symbol. When the price deviates
    below the SMA by more than a specified threshold, a LONG position is initiated.
    When the price deviates above the SMA by more than the threshold, a SHORT
    position is initiated. It includes basic logic to exit positions when the
    price reverts partially towards the SMA. It can also use limit orders for
    entry and place stop-loss orders.

    Attributes:
        sma_window (int): The lookback period for the SMA calculation.
        reversion_threshold (float): The percentage deviation from SMA to trigger a trade.
        order_quantity (float): The fixed quantity of shares for orders.
        limit_order_offset_pct (float): Percentage offset for limit orders.
        stop_loss_pct (float): Percentage for stop-loss orders.
        atr_period (int): Period for ATR calculation (for `current_atr` in orders).
        current_prices (Dict[str, Optional[float]]): Stores the latest close price per symbol.
        sma_series (Dict[str, Optional[pd.Series]]): Pre-calculated SMA series per symbol.
        atr_series (Dict[str, Optional[pd.Series]]): Pre-calculated ATR series per symbol.
        current_positions (Dict[str, Optional[str]]): Tracks perceived current position.
        last_order_direction (Dict[str, Optional[str]]): Tracks last order type to avoid
            duplicates and manage exits.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        symbols: List[str],
        sma_window: int = 20,
        reversion_threshold: float = 0.02,
        order_quantity: float = 100.0,
        data_handler: Optional[Any] = None,
        atr_period: int = 0,
        limit_order_offset_pct: float = 0.0,
        stop_loss_pct: float = 0.0,
        **kwargs: Any,
    ):
        """Initializes the MeanReversionStrategy.

        Args:
            event_queue (EventQueue): The system's event queue.
            symbols (List[str]): Symbols this strategy will operate on.
            sma_window (int, optional): Lookback period for SMA. Defaults to 20.
            reversion_threshold (float, optional): Percentage deviation from SMA
                to trigger a trade (e.g., 0.02 for 2%). Defaults to 0.02.
            order_quantity (float, optional): Fixed quantity for orders. Defaults to 100.0.
            data_handler (Optional[Any], optional): Provides historical data for
                pre-calculating indicators. Can be a DataFrame or DataHandler instance.
                Defaults to None.
            atr_period (int, optional): Period for ATR calculation. Defaults to 0.
            limit_order_offset_pct (float, optional): Offset for limit orders. Defaults to 0.0.
            stop_loss_pct (float, optional): Offset for stop-loss orders. Defaults to 0.0.
            **kwargs (Any): Additional arguments for the base `Strategy` class.
        """
        super().__init__(event_queue, symbols, data_handler=data_handler, **kwargs)
        self.sma_window: int = sma_window
        self.reversion_threshold: float = reversion_threshold
        self.order_quantity: float = order_quantity
        # self.data_handler is set in base class
        self.atr_period: int = atr_period
        self.limit_order_offset_pct: float = limit_order_offset_pct
        self.stop_loss_pct: float = stop_loss_pct

        self.current_prices: Dict[str, Optional[float]] = {
            sym: None for sym in self.symbols
        }
        self.sma_series: Dict[str, Optional[pd.Series]] = {
            sym: None for sym in self.symbols
        }
        self.atr_series: Dict[str, Optional[pd.Series]] = {
            sym: None for sym in self.symbols
        }

        self.current_positions: Dict[str, Optional[str]] = {
            sym: None for sym in self.symbols
        }
        self.last_order_direction: Dict[str, Optional[str]] = {
            sym: None for sym in self.symbols
        }

        # Pre-calculate indicators
        if self.data_handler is not None:
            for sym in self.symbols:
                try:
                    symbol_df: Optional[pd.DataFrame] = None
                    if isinstance(
                        self.data_handler, pd.DataFrame
                    ):  # Single DF from CLI or test
                        # For single DataFrame, use it for all symbols (common in tests)
                        symbol_df = self.data_handler
                    elif hasattr(
                        self.data_handler, "get_historical_data"
                    ):  # Full DataHandler
                        symbol_df = self.data_handler.get_historical_data(sym)

                    if symbol_df is not None and not symbol_df.empty:
                        if "Close" in symbol_df.columns:
                            self.sma_series[sym] = calculate_sma(
                                symbol_df["Close"], self.sma_window
                            )
                        else:
                            print(
                                f"Warning: MeanReversionStrategy ({sym}) SMA calc: 'Close' column missing."
                            )

                        if self.atr_period > 0:
                            if all(
                                col in symbol_df.columns
                                for col in ["High", "Low", "Close"]
                            ):
                                self.atr_series[sym] = calculate_atr(
                                    symbol_df["High"],
                                    symbol_df["Low"],
                                    symbol_df["Close"],
                                    self.atr_period,
                                )
                            else:
                                print(
                                    f"Warning: MeanReversionStrategy ({sym}) ATR calc: HLC columns missing."
                                )
                    elif symbol_df is None and not isinstance(
                        self.data_handler, pd.DataFrame
                    ):
                        print(
                            f"MeanReversionStrategy ({sym}): No data from data_handler for pre-calc."
                        )
                except Exception as e:
                    print(
                        f"MeanReversionStrategy ({sym}): Error pre-calculating indicators: {e}"
                    )

        print(
            f"MeanReversionStrategy initialized for {self.symbols}, SMA: {self.sma_window}, Threshold: {self.reversion_threshold:.2%}, ATR: {self.atr_period if self.atr_period > 0 else 'N/A'}"
        )

    def on_market_data(self, event: MarketEvent) -> None:
        """Handles new market data to check for mean reversion signals and generate orders.

        Calculates deviation from SMA. If price is significantly below SMA (by `reversion_threshold`),
        a BUY order is generated. If significantly above, a SELL order is generated.
        Includes basic logic to exit positions if price reverts towards the mean.

        Args:
            event (MarketEvent): The market data event.
        """
        if event.symbol not in self.symbols:
            return

        self.current_prices[event.symbol] = event.close

        current_sma: Optional[float] = None
        symbol_sma_s = self.sma_series.get(event.symbol)
        if symbol_sma_s is not None:
            current_sma = symbol_sma_s.get(event.timestamp)
            if pd.isna(current_sma):
                current_sma = None

        if current_sma is None:
            return  # Not enough data for SMA or SMA series not available

        current_atr_val: Optional[float] = None
        symbol_atr_s = self.atr_series.get(event.symbol)
        if symbol_atr_s is not None:
            current_atr_val = symbol_atr_s.get(event.timestamp)
            if pd.isna(current_atr_val):
                current_atr_val = None

        deviation = (
            (event.close - current_sma) / current_sma if current_sma != 0 else 0.0
        )

        order_type: str = "MKT"
        limit_price: Optional[float] = None
        order_direction: Optional[str] = None

        current_pos = self.current_positions.get(event.symbol)
        last_dir = self.last_order_direction.get(event.symbol)

        # Entry Logic
        if (
            deviation < -self.reversion_threshold and last_dir != "BUY"
        ):  # Avoid re-buying if already in buy mode
            order_direction = "BUY"
            if self.limit_order_offset_pct > 0:
                order_type = "LMT"
                limit_price = event.close * (1 - self.limit_order_offset_pct)
        elif (
            deviation > self.reversion_threshold and last_dir != "SELL"
        ):  # Avoid re-selling if already in sell mode
            order_direction = "SELL"
            if self.limit_order_offset_pct > 0:
                order_type = "LMT"
                limit_price = event.close * (1 + self.limit_order_offset_pct)

        # Basic Exit Logic: If position exists and deviation crosses towards zero (e.g. half threshold)
        elif (
            current_pos == "LONG"
            and deviation >= -self.reversion_threshold / 2
            and last_dir != "SELL_EXIT"
        ):
            order_direction = "SELL"
            order_type = "MKT"
            limit_price = None
            self.last_order_direction[event.symbol] = "SELL_EXIT"
        elif (
            current_pos == "SHORT"
            and deviation <= self.reversion_threshold / 2
            and last_dir != "BUY_EXIT"
        ):
            order_direction = "BUY"
            order_type = "MKT"
            limit_price = None
            self.last_order_direction[event.symbol] = "BUY_EXIT"

        if order_direction:
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
            # Update last_order_direction only if it's not an exit, to prevent immediate re-entry on same bar after exit signal
            if not (last_dir == "SELL_EXIT" or last_dir == "BUY_EXIT"):
                self.last_order_direction[event.symbol] = order_direction

            print(
                f"{event.timestamp} MeanReversionStrategy ({event.symbol}): Dev: {deviation:.2%}. Generated {order_direction} {order_type} order. RefPx: {event.close:.2f}"
                + (f", LimPx: {str(limit_price) if limit_price is not None else 'N/A'}")
            )

    def on_fill(self, event: FillEvent) -> None:
        """Handles `FillEvent` to update position state and place stop-loss orders.

        Args:
            event (FillEvent): The fill event confirming trade execution.
        """
        if event.symbol not in self.symbols:
            return

        is_opening_trade = False
        current_intended_pos_before_fill = self.current_positions.get(event.symbol)
        last_order_dir_before_fill = self.last_order_direction.get(event.symbol)

        if event.direction == "BUY":
            if last_order_dir_before_fill == "BUY_EXIT":  # This was an exit of a short
                self.current_positions[event.symbol] = None
                self.last_order_direction[event.symbol] = None
            else:
                if current_intended_pos_before_fill != "LONG":
                    is_opening_trade = True
                self.current_positions[event.symbol] = "LONG"
                # self.last_order_direction[event.symbol] = 'BUY' # Set by on_market_data
        elif event.direction == "SELL":
            if last_order_dir_before_fill == "SELL_EXIT":  # This was an exit of a long
                self.current_positions[event.symbol] = None
                self.last_order_direction[event.symbol] = None
            else:
                if current_intended_pos_before_fill != "SHORT":
                    is_opening_trade = True
                self.current_positions[event.symbol] = "SHORT"
                # self.last_order_direction[event.symbol] = 'SELL' # Set by on_market_data

        print(
            f"MeanReversionStrategy ({event.symbol}): Fill {event.direction} {event.quantity} @ {event.fill_price}. Intended Pos: {self.current_positions.get(event.symbol)}"
        )

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

                print(
                    f"MeanReversionStrategy ({event.symbol}): Issuing STP {stop_direction}, Qty: {event.quantity}, StopPx: {stop_price:.2f}"
                )
                self._generate_order(
                    symbol=event.symbol,
                    order_type="STP",
                    quantity=event.quantity,
                    direction=stop_direction,
                    timestamp=event.timestamp,
                    reference_price=event.fill_price,
                    stop_price=stop_price,
                    current_atr=current_atr_val,
                )

    def on_signal(self, event: SignalEvent) -> None:  # From ABC
        """Handles external `SignalEvent`s. (Not used by this strategy).

        Args:
            event (SignalEvent): The signal event.
        """
        pass
