"""
Implements a Pairs Trading strategy based on cointegration and z-score.
This strategy identifies two cointegrated assets and trades on the divergence
and convergence of their price spread.
"""

import pandas as pd
from typing import List, Any

try:
    from statsmodels.tsa.stattools import coint

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    coint = None

from quantsim.core.events import MarketEvent, FillEvent, OrderEvent
from quantsim.core.event_queue import EventQueue
from quantsim.strategies.base import Strategy


class PairsTradingStrategy(Strategy):
    """
    A strategy that trades pairs of assets based on cointegration.
    It calculates the z-score of the spread and trades when it crosses
    predefined thresholds.

    Note: This strategy requires the 'statsmodels' package for cointegration testing.
    Install with: pip install statsmodels
    """

    def __init__(
        self,
        event_queue: EventQueue,
        symbols: List[str],
        lookback_period: int = 252,
        z_score_threshold: float = 2.0,
        order_quantity: float = 100.0,
        **kwargs: Any,
    ):
        """
        Initializes the PairsTradingStrategy.
        Args:
            event_queue (EventQueue): The system's event queue.
            symbols (List[str]): List of two symbols for pairs trading.
            lookback_period (int): The period for spread calculation.
            z_score_threshold (float): The threshold for generating signals.
            order_quantity (float): The quantity for generated orders.
        """
        if len(symbols) != 2:
            raise ValueError("PairsTradingStrategy requires exactly two symbols.")

        if not HAS_STATSMODELS:
            raise ImportError(
                "PairsTradingStrategy requires 'statsmodels' package for cointegration testing. "
                "Install with: pip install statsmodels"
            )

        super().__init__(event_queue, symbols, **kwargs)
        self.symbol1 = self.symbols[0]
        self.symbol2 = self.symbols[1]
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.order_quantity = order_quantity

        self.prices1 = []
        self.prices2 = []
        self.spread = []
        self.in_long = False
        self.in_short = False

    def on_market_data(self, event: MarketEvent) -> None:
        """
        Handles new market data to check for trading opportunities.
        """
        if event.symbol == self.symbol1:
            self.prices1.append(event.close)
        elif event.symbol == self.symbol2:
            self.prices2.append(event.close)
        else:
            return

        if (
            len(self.prices1) > self.lookback_period
            and len(self.prices2) > self.lookback_period
        ):
            # Ensure we have data for both symbols at this timestamp
            if len(self.prices1) != len(self.prices2):
                # Simple synchronization: wait for the next tick
                return

            prices1_series = pd.Series(self.prices1[-self.lookback_period :])
            prices2_series = pd.Series(self.prices2[-self.lookback_period :])

            # Cointegration test
            score, pvalue, _ = coint(prices1_series, prices2_series)
            if pvalue > 0.05:
                # Not cointegrated, do not trade
                return

            # Calculate spread and z-score
            spread = prices1_series - prices2_series
            z_score = (spread.iloc[-1] - spread.mean()) / spread.std()

            if z_score > self.z_score_threshold and not self.in_short:
                # Short the spread (Sell symbol1, Buy symbol2)
                self.event_queue.put_event(
                    OrderEvent(self.symbol1, "MKT", self.order_quantity, "SELL")
                )
                self.event_queue.put_event(
                    OrderEvent(self.symbol2, "MKT", self.order_quantity, "BUY")
                )
                self.in_short = True
                self.in_long = False

            elif z_score < -self.z_score_threshold and not self.in_long:
                # Long the spread (Buy symbol1, Sell symbol2)
                self.event_queue.put_event(
                    OrderEvent(self.symbol1, "MKT", self.order_quantity, "BUY")
                )
                self.event_queue.put_event(
                    OrderEvent(self.symbol2, "MKT", self.order_quantity, "SELL")
                )
                self.in_long = True
                self.in_short = False

            elif abs(z_score) < 0.5 and (self.in_long or self.in_short):
                # Close position
                self.event_queue.put_event(
                    OrderEvent(
                        self.symbol1,
                        "MKT",
                        self.order_quantity,
                        "SELL" if self.in_long else "BUY",
                    )
                )
                self.event_queue.put_event(
                    OrderEvent(
                        self.symbol2,
                        "MKT",
                        self.order_quantity,
                        "BUY" if self.in_long else "SELL",
                    )
                )
                self.in_long = False
                self.in_short = False

    def on_fill(self, event: FillEvent) -> None:
        """
        Handles fill events to update strategy state.
        """
        # Can be used to track position more accurately
        pass
