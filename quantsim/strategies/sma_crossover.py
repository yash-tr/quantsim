"""Implements the Simple Moving Average (SMA) Crossover trading strategy.

This strategy generates trading signals based on the crossover of two SMAs
of different lookback periods (a short-window SMA and a long-window SMA).
A BUY signal is generated when the short SMA crosses above the long SMA.
A SELL signal is generated when the short SMA crosses below the long SMA.
"""

import pandas as pd
from typing import List, Optional, Dict, Any  # For type hints

from quantsim.core.events import (
    MarketEvent,
    FillEvent,
    SignalEvent,
)  # SignalEvent for on_signal
from quantsim.core.event_queue import EventQueue
from quantsim.strategies.base import Strategy
from quantsim.indicators import calculate_sma, calculate_atr


class SMACrossoverStrategy(Strategy):
    """A trading strategy based on SMA crossovers.

    This strategy can pre-calculate SMA and ATR series if a suitable DataFrame
    is provided as `data_handler` during initialization. Otherwise, it can
    calculate SMAs on-the-fly for each market event (less efficient).
    It generates Market or Limit orders and can optionally place Stop-Loss orders.

    Attributes:
        symbol (str): The primary symbol this strategy instance trades.
        short_window (int): Lookback period for the short SMA.
        long_window (int): Lookback period for the long SMA.
        order_quantity (float): Quantity for generated orders.
        limit_order_offset_pct (float): Percentage offset for limit orders from current price.
                                      If 0, market orders are used.
        stop_loss_pct (float): Percentage for stop-loss orders from fill price.
                               If 0, no stop-loss orders are placed.
        atr_period (int): Period for ATR calculation, used if ATR slippage is active
                          or if stop-losses are ATR-based (not currently implemented).
        prices (Dict[str, List[float]]): Stores recent close prices for on-the-fly SMA calculation fallback.
        current_position (Dict[str, Optional[str]]): Tracks the strategy's perceived current position
                                                 ('LONG', 'SHORT', or None) for the symbol.
        last_order_direction (Dict[str, Optional[str]]): Tracks the last order direction to avoid duplicates.
        short_sma_series (Optional[pd.Series]): Pre-calculated short SMA series.
        long_sma_series (Optional[pd.Series]): Pre-calculated long SMA series.
        atr_series (Optional[pd.Series]): Pre-calculated ATR series (for the primary symbol).
    """

    def __init__(
        self,
        event_queue: EventQueue,
        symbols: List[str],
        short_window: int,
        long_window: int,
        order_quantity: float = 100.0,
        limit_order_offset_pct: float = 0.0,
        stop_loss_pct: float = 0.0,
        atr_period: int = 0,
        data_handler: Optional[Any] = None,  # pd.DataFrame or DataHandler ABC
        **kwargs: Any,
    ):
        """Initializes the SMACrossoverStrategy.

        Args:
            event_queue (EventQueue): The system's event queue.
            symbols (List[str]): List of symbols. This strategy primarily uses the first symbol
                                 for its core logic and pre-calculation if a single DataFrame
                                 is passed as `data_handler`.
            short_window (int): Lookback period for the short SMA.
            long_window (int): Lookback period for the long SMA.
            order_quantity (float, optional): Fixed quantity for orders. Defaults to 100.0.
            limit_order_offset_pct (float, optional): Percentage offset for limit orders.
                If 0, market orders are used. Defaults to 0.0.
            stop_loss_pct (float, optional): Percentage for stop-loss orders from fill price.
                If 0, no stop-loss orders. Defaults to 0.0.
            atr_period (int, optional): Period for ATR calculation. Used for `current_atr`
                in `OrderEvent`s. Defaults to 0 (no ATR calculation).
            data_handler (Optional[Any], optional): Provides historical data.
                If a pandas DataFrame, it's used for pre-calculating indicators for the
                primary symbol. If a DataHandler instance, it might be used by base or future features.
                Defaults to None.
            **kwargs (Any): Additional keyword arguments passed to the base `Strategy` class.

        Raises:
            ValueError: If `symbols` list is empty or `short_window` >= `long_window`.
        """
        if not symbols:
            raise ValueError("Symbols list cannot be empty for SMACrossoverStrategy.")
        # The strategy_id is now set in the base class if not provided.
        # We can customize it here if needed, e.g. by passing it to super()
        custom_strategy_id = kwargs.pop(
            "strategy_id", f"SMACrossover_{symbols[0]}_{short_window}_{long_window}"
        )
        super().__init__(
            event_queue,
            symbols,
            strategy_id=custom_strategy_id,
            data_handler=data_handler,
            **kwargs,
        )

        self.symbol: str = symbols[
            0
        ]  # This strategy focuses on a single symbol from the list
        self.short_window: int = short_window
        self.long_window: int = long_window
        self.order_quantity: float = order_quantity
        self.limit_order_offset_pct: float = limit_order_offset_pct
        self.stop_loss_pct: float = stop_loss_pct
        self.atr_period: int = atr_period

        self.short_sma_series: Optional[pd.Series] = None
        self.long_sma_series: Optional[pd.Series] = None
        self.atr_series: Optional[pd.Series] = None  # Only for self.symbol

        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window.")

        self.prices: Dict[str, List[float]] = {
            s: [] for s in self.symbols
        }  # Store prices for all symbols strategy is aware of
        self.current_position: Dict[str, Optional[str]] = {
            s: None for s in self.symbols
        }
        self.last_order_direction: Dict[str, Optional[str]] = {
            s: None for s in self.symbols
        }

        # Pre-calculation if data_handler is a DataFrame (typically for the primary symbol)
        # The data_handler in Strategy base is Any. Here we check if it's DataFrame for this specific strategy's pre-calc.
        if self.data_handler is not None and isinstance(
            self.data_handler, pd.DataFrame
        ):
            df_for_indicators = self.data_handler  # Assume it's for self.symbol

            if "Close" in df_for_indicators.columns:
                print(f"{self.strategy_id}: Pre-calculating SMAs for {self.symbol}...")
                self.short_sma_series = calculate_sma(
                    df_for_indicators["Close"], self.short_window
                )
                self.long_sma_series = calculate_sma(
                    df_for_indicators["Close"], self.long_window
                )
            else:
                print(
                    f"Warning: {self.strategy_id} for {self.symbol} - 'Close' column missing for pre-calculating SMAs."
                )

            if self.atr_period > 0:
                if all(
                    col in df_for_indicators.columns for col in ["High", "Low", "Close"]
                ):
                    print(
                        f"{self.strategy_id}: Pre-calculating ATR with period {self.atr_period} for {self.symbol}..."
                    )
                    # This ATR series is specific to self.symbol (the first symbol)
                    self.atr_series = calculate_atr(
                        df_for_indicators["High"],
                        df_for_indicators["Low"],
                        df_for_indicators["Close"],
                        self.atr_period,
                    )
                else:
                    print(
                        f"Warning: {self.strategy_id} for {self.symbol} wants ATR but 'High', 'Low', or 'Close' columns missing."
                    )
        elif self.data_handler is not None:
            # If data_handler is not a DataFrame, but a DataHandler instance,
            # _preload_historical_data might be used to populate self.prices for on-the-fly.
            # However, SMACrossover primarily uses pre-calculation if DF is given.
            # self._preload_historical_data(self.data_handler) # Not used if pre-calc is primary
            pass

        # Note: _preload_historical_data was for a different pattern of price handling.
        # This strategy now prioritizes pre-calculated series from a DataFrame if provided.
        # If not, it falls back to on-the-fly SMA calculation using the self.prices list.
        # self.min_data_points is implicitly handled by NaN values in SMA series or length of self.prices.

    def _preload_historical_data(self, data_source: Any) -> None:
        """(Primarily for fallback) Preloads historical close prices for on-the-fly SMA.

        This method is intended for scenarios where indicator series are not pre-calculated
        (e.g., `data_handler` in `__init__` was not a suitable DataFrame for `self.symbol`).
        It populates `self.prices[self.symbol]` list.

        Args:
            data_source (Any): Can be a pandas DataFrame or a DataHandler instance.
        """
        if isinstance(data_source, pd.DataFrame):
            if "Close" in data_source.columns:
                self.prices[self.symbol] = data_source["Close"].tolist()
        elif hasattr(data_source, "get_historical_data"):  # Check if it's a DataHandler
            hist_df = data_source.get_historical_data(self.symbol)
            if not hist_df.empty and "Close" in hist_df.columns:
                self.prices[self.symbol] = hist_df["Close"].tolist()

        if not self.prices[self.symbol]:
            print(
                f"Warning: {self.strategy_id} could not preload prices for {self.symbol} for on-the-fly SMA fallback."
            )

    def on_market_data(self, event: MarketEvent) -> None:
        """Handles new market data to check for SMA crossovers and generate orders.

        If pre-calculated SMA series are available, they are used. Otherwise,
        SMAs are calculated on-the-fly from recent close prices. Orders are
        generated based on crossover signals and existing position/order state.

        Args:
            event (MarketEvent): The market data event.
        """
        if event.symbol != self.symbol:  # This strategy instance focuses on one symbol
            return

        self.prices[self.symbol].append(
            event.close
        )  # Always append for potential fallback calc
        max_len = self.long_window + 5  # Keep a buffer
        if len(self.prices[self.symbol]) > max_len:
            self.prices[self.symbol] = self.prices[self.symbol][-max_len:]

        short_sma: Optional[float] = None
        long_sma: Optional[float] = None

        if self.short_sma_series is not None and self.long_sma_series is not None:
            short_sma = self.short_sma_series.get(event.timestamp)
            long_sma = self.long_sma_series.get(event.timestamp)

        # Fallback or if pre-calculated SMAs are NaN (e.g., at start of series)
        if pd.isna(short_sma) or pd.isna(long_sma):
            if len(self.prices[self.symbol]) >= self.long_window:
                price_series = pd.Series(self.prices[self.symbol])
                short_sma = calculate_sma(price_series, self.short_window).iloc[-1]
                long_sma = calculate_sma(price_series, self.long_window).iloc[-1]
            else:
                return  # Not enough data even for on-the-fly

        if pd.isna(short_sma) or pd.isna(long_sma):
            return  # Still no valid SMA

        current_atr_val: Optional[float] = None
        if self.atr_series is not None:  # atr_series is for self.symbol
            current_atr_val = self.atr_series.get(event.timestamp)
            if pd.isna(current_atr_val):
                current_atr_val = None

        order_type: str = "MKT"
        limit_price: Optional[float] = None
        order_direction: Optional[str] = None

        # Entry signals
        if short_sma > long_sma and self.last_order_direction.get(self.symbol) != "BUY":
            order_direction = "BUY"
            if self.limit_order_offset_pct > 0:
                order_type = "LMT"
                limit_price = event.close * (1 - self.limit_order_offset_pct)
        elif (
            short_sma < long_sma
            and self.last_order_direction.get(self.symbol) != "SELL"
        ):
            order_direction = "SELL"
            if self.limit_order_offset_pct > 0:
                order_type = "LMT"
                limit_price = event.close * (1 + self.limit_order_offset_pct)
        # Exit signals (reversal of position due to crossover)
        elif (
            short_sma < long_sma
            and self.last_order_direction.get(self.symbol) == "BUY"
            and self.current_position.get(self.symbol) != "SHORT"
        ) or (
            short_sma > long_sma
            and self.last_order_direction.get(self.symbol) == "SELL"
            and self.current_position.get(self.symbol) != "LONG"
        ):
            self.last_order_direction[self.symbol] = (
                None  # Reset if aligned with position or neutral
            )

        if order_direction:
            self._generate_order(
                symbol=self.symbol,
                order_type=order_type,
                direction=order_direction,
                quantity=self.order_quantity,
                timestamp=event.timestamp,
                reference_price=event.close,
                limit_price=limit_price,
                current_atr=current_atr_val,
            )
            self.last_order_direction[self.symbol] = order_direction
            # print(f"{event.timestamp} {self.strategy_id}: {order_direction} {order_type} for {self.symbol}")

    def on_signal(self, event: SignalEvent) -> None:
        """Handles external `SignalEvent`s. (Not used by this strategy)."""
        pass

    def on_fill(self, event: FillEvent) -> None:
        """Handles `FillEvent` to update position state and place stop-loss orders.

        Args:
            event (FillEvent): The fill event confirming trade execution.
        """
        if event.symbol != self.symbol:
            return

        is_opening_trade = False
        if event.direction == "BUY":
            if self.current_position.get(self.symbol) != "LONG":
                is_opening_trade = True
            self.current_position[self.symbol] = "LONG"
            self.last_order_direction[self.symbol] = "BUY"
        elif event.direction == "SELL":
            if self.current_position.get(self.symbol) != "SHORT":
                is_opening_trade = True
            self.current_position[self.symbol] = "SHORT"
            self.last_order_direction[self.symbol] = "SELL"

        # Simple check: if fill quantity is less than typical order_quantity, it might be a partial close from SL
        # More robust: check if fill closes the position (would need portfolio feedback or more state)
        if (
            event.quantity < self.order_quantity and not is_opening_trade
        ):  # Potentially a stop-loss fill
            # If a stop-loss was filled, reset current_position and last_order_direction
            # This is simplified; actual position quantity is needed from portfolio.
            # For now, if a fill that is not an opening trade has different qty, assume it might be a SL closing out.
            # A more robust way is to check order_id if SL order_ids are tracked.
            # print(f"{self.strategy_id}: Possible SL fill for {event.symbol}. Resetting position state.")
            # self.current_position[self.symbol] = None
            # self.last_order_direction[self.symbol] = None # Ready for new signals
            pass

        if self.stop_loss_pct > 0 and is_opening_trade:
            stop_price: Optional[float] = None
            stop_direction: Optional[str] = None
            reference_price_for_stop = event.fill_price

            if self.current_position.get(self.symbol) == "LONG":
                stop_price = reference_price_for_stop * (1 - self.stop_loss_pct)
                stop_direction = "SELL"
            elif self.current_position.get(self.symbol) == "SHORT":
                stop_price = reference_price_for_stop * (1 + self.stop_loss_pct)
                stop_direction = "BUY"

            if stop_price is not None and stop_direction is not None:
                current_atr_val_for_sl: Optional[float] = None
                if self.atr_series is not None:  # self.atr_series is for self.symbol
                    current_atr_val_for_sl = self.atr_series.get(event.timestamp)
                    if pd.isna(current_atr_val_for_sl):
                        current_atr_val_for_sl = None

                # print(f"{self.strategy_id}: Issuing STP {stop_direction} for {self.symbol} @ {stop_price:.2f}")
                self._generate_order(
                    symbol=self.symbol,
                    order_type="STP",
                    quantity=abs(event.quantity),
                    direction=stop_direction,
                    timestamp=event.timestamp,
                    reference_price=reference_price_for_stop,
                    stop_price=stop_price,
                    current_atr=current_atr_val_for_sl,
                )
