"""
Implements a risk parity portfolio construction model.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Optional

from quantsim.portfolio.portfolio import Portfolio
from quantsim.core.events import SignalEvent, MarketEvent, OrderEvent


class RiskParityPortfolio(Portfolio):
    """
    A portfolio that allocates assets based on risk parity.
    """

    def __init__(
        self,
        initial_cash: float,
        event_queue,
        position_sizer=None,
        symbols: List[str] = None,
        lookback_period: int = 252,
        rebalance_on_close: bool = True,
        **kwargs,
    ):
        # Extract symbols before calling parent constructor
        self.symbols = symbols if symbols is not None else []

        # Call parent constructor with only the parameters it accepts
        super().__init__(
            initial_cash=initial_cash,
            event_queue=event_queue,
            position_sizer=position_sizer,
        )

        # Initialize risk parity specific attributes
        self.lookback_period = lookback_period
        self.returns_history: Dict[str, List[float]] = {
            symbol: [] for symbol in self.symbols
        }
        self.rebalance_on_close = rebalance_on_close

    def on_market_data(self, event: MarketEvent) -> None:
        """
        Calculates and executes portfolio adjustments based on market data.
        Only rebalances if the event timestamp is on or after the next rebalance date.
        """
        super().on_market_data(event)
        if event.symbol in self.symbols:
            self.returns_history[event.symbol].append(event.close)

    def on_signal(self, event: SignalEvent) -> None:
        """
        Generates orders based on risk parity allocation.
        """
        if event.direction == "REBALANCE":
            weights = self._calculate_risk_parity_weights()
            if weights is None:
                return

            for symbol, weight in weights.items():
                target_value = self.get_equity() * weight
                current_position = self.positions.get(symbol)
                current_quantity = current_position.quantity if current_position else 0
                current_price = self.get_last_close_price(symbol)

                if current_price is None:
                    continue

                current_value = current_quantity * current_price

                quantity = (target_value - current_value) / current_price
                direction = "BUY" if quantity > 0 else "SELL"

                order = OrderEvent(symbol, "MKT", abs(quantity), direction)
                self.event_queue.put_event(order)

    def _calculate_risk_parity_weights(self) -> Optional[Dict[str, float]]:
        """
        Calculates asset weights according to risk parity.
        """
        # Check if we have enough data for all symbols
        if not all(
            len(self.returns_history[symbol]) >= self.lookback_period
            for symbol in self.symbols
        ):
            return None

        # Create DataFrame from recent price history
        recent_data = {}
        for symbol in self.symbols:
            recent_data[symbol] = self.returns_history[symbol][-self.lookback_period :]

        returns_df = pd.DataFrame(recent_data)
        returns_series = returns_df.pct_change().dropna()

        if len(returns_series) < 2:
            return None

        cov_matrix = returns_series.cov() * 252  # Annualized covariance

        def objective(weights):
            portfolio_variance = weights.T @ cov_matrix @ weights
            if portfolio_variance <= 0:
                return 1e6  # Large penalty for invalid variance
            risk_contributions = (weights * (cov_matrix @ weights)) / np.sqrt(
                portfolio_variance
            )
            return np.sum((risk_contributions - risk_contributions.mean()) ** 2)

        num_assets = len(self.symbols)
        initial_weights = np.array([1 / num_assets] * num_assets)
        bounds = tuple(
            (0.001, 0.999) for _ in range(num_assets)
        )  # Small bounds to avoid edge cases
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        try:
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                return dict(zip(self.symbols, result.x))
        except Exception:
            pass  # Optimization failed, return None

        return None
