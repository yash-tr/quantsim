"""
Core simulation engine that orchestrates the backtesting process.
"""

from typing import Any, TYPE_CHECKING, Dict, Callable

from .event_queue import EventQueue
from .events import MarketEvent, OrderEvent, FillEvent, SignalEvent
from quantsim.data.base import DataHandler
from quantsim.strategies.base import Strategy
from quantsim.execution.execution_handler import ExecutionHandler

if TYPE_CHECKING:
    from quantsim.portfolio.portfolio import Portfolio


class SimulationEngine:
    """Orchestrates event-driven backtesting simulations.

    This engine iterates through market data provided by a DataHandler,
    places MarketEvents onto an EventQueue, and then dispatches events from
    the queue to the appropriate handlers (Strategy, Portfolio, ExecutionHandler).

    Attributes:
        event_queue (EventQueue): The central queue for all system events.
        data_handler (DataHandler): The source of market data for the simulation.
        strategy (Strategy): The trading strategy being backtested.
        portfolio (Portfolio): Manages holdings, cash, and performance.
        execution_handler (ExecutionHandler): Simulates order execution.
        event_dispatch (Dict[str, Callable[[Event], None]]): Maps event types to handler methods.
        latest_market_event (Dict[str, MarketEvent]): Tracks the latest market event for each symbol.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        data_handler: DataHandler,
        strategy: Strategy,
        portfolio: "Portfolio",
        execution_handler: ExecutionHandler,
    ):
        """Initializes the SimulationEngine.

        Args:
            event_queue (EventQueue): The central event queue for the system.
            data_handler (DataHandler): An iterable data source that yields tuples of
                                        (timestamp, symbol, ohlcv_data_dict).
            strategy (Strategy): The trading strategy instance.
            portfolio (Portfolio): The portfolio manager instance.
            execution_handler (ExecutionHandler): The execution handler instance.
        """
        self.event_queue: EventQueue = event_queue
        self.data_handler: DataHandler = data_handler
        self.strategy: Strategy = strategy
        self.portfolio: "Portfolio" = portfolio
        self.execution_handler: ExecutionHandler = execution_handler
        self.latest_market_event: Dict[str, MarketEvent] = {}

        self.event_dispatch: Dict[str, Callable[[Any], None]] = {
            "MARKET": self._handle_market_event,
            "SIGNAL": self._handle_signal_event,
            "ORDER": self._handle_order_event,
            "FILL": self._handle_fill_event,
        }
        print("SimulationEngine initialized.")

    def _handle_market_event(self, event: MarketEvent) -> None:
        """Handles MarketEvents by passing them to the strategy and portfolio.

        Args:
            event (MarketEvent): The market data event to process.
        """
        self.strategy.on_market_data(event)
        self.portfolio.on_market_data(event)
        self.latest_market_event[event.symbol] = event

    def _handle_signal_event(self, event: SignalEvent) -> None:
        """Handles SignalEvents by passing them to the portfolio.

        Args:
            event (SignalEvent): The signal event to process.
        """
        self.portfolio.on_signal(event)

    def _handle_order_event(self, event: OrderEvent) -> None:
        """Handles OrderEvents by passing them to the execution handler.

        Args:
            event (OrderEvent): The order event to process.
        """
        market_event_for_order = self.latest_market_event.get(event.symbol)
        self.execution_handler.execute_order(event, market_event_for_order)

    def _handle_fill_event(self, event: FillEvent) -> None:
        """Handles FillEvents by passing them to the strategy and portfolio.

        Args:
            event (FillEvent): The fill event to process.
        """
        self.strategy.on_fill(event)  # Strategy might update its state based on fills
        self.portfolio.on_fill(event)

    def run(self) -> None:
        """Runs the main simulation event loop.

        The loop proceeds as follows:
        1. Iterates through the `data_handler`. Each item from the data handler
           is expected to be a bar of market data for a specific symbol at a specific timestamp.
           Format: `(timestamp, symbol, ohlcv_data_dict)`
           where `ohlcv_data_dict` is a dictionary like {'Open': ..., 'Close': ..., ...}.
        2. For each data bar, a `MarketEvent` is created and put onto the `event_queue`.
        3. An inner loop processes events from the `event_queue` until it's empty for that bar:
           - Retrieves an event.
           - Dispatches the event to the appropriate handler method (_handle_market_event,
             _handle_order_event, _handle_fill_event) based on `event.type`.
        4. This continues until the `data_handler` is exhausted (i.e., all historical data is processed).
        5. Finally, it calls the portfolio to calculate and print performance metrics and a summary.
        """
        print("\n--- Starting Backtest Simulation ---")

        while self.data_handler.continue_backtest:
            # 1. Get the next market data bar
            data_bar_info = next(self.data_handler, None)
            if data_bar_info is None:
                # End of data stream
                break

            # 2. Create and queue the MarketEvent
            timestamp, symbol, ohlcv_data_dict = data_bar_info

            # Include bid/ask if available in the data source
            bid_price = ohlcv_data_dict.get("Bid", ohlcv_data_dict["Close"])
            ask_price = ohlcv_data_dict.get("Ask", ohlcv_data_dict["Close"])

            market_event = MarketEvent(
                symbol=symbol,
                timestamp=timestamp,
                open_price=ohlcv_data_dict["Open"],
                high_price=ohlcv_data_dict["High"],
                low_price=ohlcv_data_dict["Low"],
                close_price=ohlcv_data_dict["Close"],
                volume=int(ohlcv_data_dict["Volume"]),
                bid_price=bid_price,
                ask_price=ask_price,
            )
            self.event_queue.put_event(market_event)

            # 3. Process events until the queue is empty for this time tick
            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_event()
                    if event is None:
                        break
                except Exception:
                    break

                handler_method = self.event_dispatch.get(event.type)
                if handler_method:
                    handler_method(event)
                else:
                    print(
                        f"Warning: SimulationEngine - No handler registered for event type {event.type}"
                    )

        print("\n--- Backtest Simulation Finished ---")

        self.portfolio.calculate_performance_metrics()
        self.portfolio.print_final_summary()
