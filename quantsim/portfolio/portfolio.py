"""
Portfolio management for backtesting.
Handles positions, cash tracking, and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import FillEvent, MarketEvent, SignalEvent, OrderEvent
from .position import Position
from .trade_log import TradeLog
from .position_sizer import PositionSizer, FixedQuantitySizer


class Portfolio:
    """
    Manages trading positions, cash, and performance metrics for a backtest.
    """

    def __init__(
        self,
        initial_cash: float,
        event_queue: EventQueue,
        position_sizer: Optional[PositionSizer] = None,
    ):
        """Initialize the portfolio with starting cash and event queue."""
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.event_queue = event_queue
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.current_total_value = initial_cash
        self.trade_logger = TradeLog()
        self.position_sizer = (
            position_sizer if position_sizer is not None else FixedQuantitySizer()
        )
        self.last_prices: Dict[str, float] = {}

        # Risk management
        self.drawdown_series: Optional[pd.Series] = None
        self._peak_value = initial_cash
        self._daily_returns: Optional[pd.Series] = None

    def on_fill(self, event: FillEvent) -> None:
        """
        Updates the portfolio state upon receiving a FillEvent.
        This includes updating cash, position details, and logging trades.
        """
        # Update cash - correct logic for buy/sell
        if event.direction == "BUY":
            self.cash -= (event.quantity * event.fill_price) + event.commission
        else:  # SELL
            self.cash += (event.quantity * event.fill_price) - event.commission

        # Update or create position
        if event.symbol not in self.positions:
            position_quantity = (
                event.quantity if event.direction == "BUY" else -event.quantity
            )
            self.positions[event.symbol] = Position(
                symbol=event.symbol,
                quantity=position_quantity,
                average_price=event.fill_price,
                last_price=event.fill_price,
            )
        else:
            transaction_quantity = (
                event.quantity if event.direction == "BUY" else -event.quantity
            )
            self.positions[event.symbol].transact(
                quantity=transaction_quantity, price=event.fill_price
            )

        # Log the trade fill
        self.trade_logger.add_fill(event)

        # If a position is closed, remove it
        if self.positions[event.symbol].quantity == 0:
            del self.positions[event.symbol]

    def on_market_data(self, event: MarketEvent) -> None:
        """
        Updates the value of open positions based on new market data.
        """
        if event.symbol in self.positions:
            self.positions[event.symbol].update_last_price(event.close)

        self.last_prices[event.symbol] = event.close
        self._update_unrealized_pnl(event.timestamp)

    def on_signal(self, event: SignalEvent) -> None:
        """
        Handles a SignalEvent to generate an OrderEvent.
        """
        quantity = self.position_sizer.size_order(self, event)
        if quantity is None or quantity <= 0:
            return

        order = OrderEvent(
            symbol=event.symbol,
            order_type="MKT",
            quantity=quantity,
            direction=event.direction,
        )
        self.event_queue.put_event(order)

    def _update_unrealized_pnl(self, timestamp: pd.Timestamp) -> None:
        """
        Calculates and records the current total value of the portfolio.
        """
        total_value = self.cash
        for pos in self.positions.values():
            total_value += pos.market_value

        self.current_total_value = total_value
        self.equity_curve.append((timestamp, total_value))

        # Update peak value for drawdown calculation
        if total_value > self._peak_value:
            self._peak_value = total_value

    def calculate_daily_returns(self) -> None:
        """Calculates daily returns from the equity curve."""
        if not self.equity_curve:
            self._daily_returns = pd.Series(dtype=float)
            return

        equity_df = pd.DataFrame(
            self.equity_curve, columns=["Timestamp", "PortfolioValue"]
        )
        equity_df = equity_df.set_index("Timestamp")
        self._daily_returns = (
            equity_df["PortfolioValue"].resample("D").last().pct_change().dropna()
        )

    def get_equity(self) -> float:
        """Returns the current total value of the portfolio."""
        return self.current_total_value

    def get_last_close_price(self, symbol: str) -> Optional[float]:
        """Returns the last known closing price for a symbol."""
        return self.last_prices.get(symbol)

    @property
    def holdings(self):
        """Alias for positions for backward compatibility."""
        return self.positions

    @property
    def current_holdings_value(self) -> float:
        """Total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def metrics(self) -> Dict:
        """Calculate and return portfolio performance metrics."""
        # Return cached metrics if available, otherwise calculate with default
        if hasattr(self, "_cached_metrics"):
            return self._cached_metrics
        return self._calculate_metrics()

    def get_metrics(self, risk_free_rate: float = 0.02) -> Dict:
        """Get portfolio metrics with a specific risk-free rate."""
        return self._calculate_metrics(risk_free_rate)

    def _calculate_metrics(self, risk_free_rate: float = 0.02) -> Dict:
        """Calculate and return portfolio performance metrics with specified risk-free rate."""
        try:
            if self._daily_returns is None:
                self.calculate_daily_returns()

            returns = (
                self._daily_returns
                if self._daily_returns is not None
                else pd.Series(dtype=float)
            )

            if len(self.equity_curve) < 1:
                return {"error": "Insufficient equity data", "total_return_pct": 0.0}

            if len(self.equity_curve) == 1:
                # Single point equity curve
                total_return = (
                    self.current_total_value - self.initial_cash
                ) / self.initial_cash
                return {
                    "error": "Insufficient equity data",
                    "total_return_pct": total_return * 100,
                    "total_trades": 0,
                    "realized_pnl": 0.0,
                    "max_drawdown_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "cagr_pct": 0.0,
                    "annualized_volatility_pct": 0.0,
                }

            # Convert equity curve to returns series
            equity_values = [value for timestamp, value in self.equity_curve]
            returns = pd.Series(equity_values).pct_change().dropna()

            # Basic calculations
            total_return = (
                self.current_total_value - self.initial_cash
            ) / self.initial_cash

            # Calculate max drawdown
            equity_series = pd.Series(
                equity_values, index=[t for t, v in self.equity_curve]
            )
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()
            self.drawdown_series = drawdown  # Store for reporting

            # Calculate annualized metrics (assuming daily returns)
            if len(returns) > 0 and returns.std() > 0:
                mean_daily_return = returns.mean()
                daily_volatility = returns.std()

                # Convert annual risk-free rate to daily
                daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1

                # Annualized metrics (252 trading days per year)
                annualized_volatility = daily_volatility * (252**0.5)

                # Sharpe ratio using daily calculation then annualized
                sharpe_ratio = (
                    (mean_daily_return - daily_risk_free_rate)
                    / daily_volatility
                    * (252**0.5)
                    if daily_volatility > 0
                    else 0
                )

                # CAGR calculation using actual calendar time difference
                start_date = self.equity_curve[0][0]
                end_date = self.equity_curve[-1][0]
                time_delta_years = (end_date - start_date).days / 365.25

                if time_delta_years == 0:
                    # If same day or very short period, use sum of returns approach
                    equity_df = pd.DataFrame(
                        self.equity_curve, columns=["Timestamp", "PortfolioValue"]
                    ).set_index("Timestamp")
                    returns_for_cagr = equity_df["PortfolioValue"].pct_change().dropna()
                    cagr = returns_for_cagr.sum() if len(returns_for_cagr) > 0 else 0
                else:
                    cagr = (self.current_total_value / self.initial_cash) ** (
                        1.0 / time_delta_years
                    ) - 1
            else:
                annualized_volatility = 0
                sharpe_ratio = (
                    float("nan") if len(returns) > 0 and returns.std() == 0 else 0
                )
                cagr = 0

            # Simple metrics that don't require time series
            completed_trades = self.trade_logger.get_completed_trades()
            total_trades = len(completed_trades)

            # Base metrics that always exist
            base_metrics = {
                "total_return_pct": total_return * 100,
                "total_trades": total_trades,
                "realized_pnl": (
                    sum(t.net_pnl for t in completed_trades)
                    if completed_trades
                    else 0.0
                ),
                "max_drawdown_pct": max_drawdown * 100,
                "sharpe_ratio": sharpe_ratio,
                "cagr_pct": cagr * 100,
                "annualized_volatility_pct": annualized_volatility * 100,
            }

            if total_trades == 0:
                base_metrics["error"] = "No completed trades for detailed metrics"
                return base_metrics

            # Trade-based metrics
            winning_trades = [t for t in completed_trades if t.net_pnl > 0]
            losing_trades = [t for t in completed_trades if t.net_pnl < 0]

            win_rate = (
                len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            )

            gross_profit = (
                sum(t.net_pnl for t in winning_trades) if winning_trades else 0
            )
            gross_loss = sum(t.net_pnl for t in losing_trades) if losing_trades else 0

            base_metrics.update(
                {
                    "num_winning_trades": len(winning_trades),
                    "num_losing_trades": len(losing_trades),
                    "win_rate_pct": win_rate,
                    "gross_profit": gross_profit,
                    "gross_loss": gross_loss,
                    "profit_factor": (
                        abs(gross_profit / gross_loss)
                        if gross_loss != 0
                        else float("inf")
                    ),
                    "avg_win_pnl": (
                        gross_profit / len(winning_trades) if winning_trades else 0
                    ),
                    "avg_loss_pnl": (
                        gross_loss / len(losing_trades) if losing_trades else 0
                    ),
                }
            )

            # Add new risk metrics
            sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate)
            calmar_ratio = self.calculate_calmar_ratio(returns, risk_free_rate)
            var = self.calculate_var(returns)
            cvar = self.calculate_cvar(returns)

            base_metrics.update(
                {
                    "sortino_ratio": sortino_ratio,
                    "calmar_ratio": calmar_ratio,
                    "value_at_risk_95": var,
                    "cond_value_at_risk_95": cvar,
                }
            )

            return base_metrics

        except Exception as e:
            return {"error": f"Error calculating metrics: {str(e)}"}

    def calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float
    ) -> float:
        """Calculates the Sortino ratio."""
        if returns.empty:
            return 0.0
        target_return = (1 + risk_free_rate) ** (1 / 252) - 1
        downside_returns = returns[returns < target_return]
        downside_deviation = downside_returns.std()

        if downside_deviation == 0 or pd.isna(downside_deviation):
            return 0.0

        return (returns.mean() - target_return) / downside_deviation * np.sqrt(252)

    def calculate_calmar_ratio(
        self, returns: pd.Series, risk_free_rate: float
    ) -> float:
        """Calculates the Calmar ratio."""
        if not hasattr(self, "_cached_metrics") or self._cached_metrics is None:
            # Temporarily calculate basic metrics if not available
            temp_metrics = self._calculate_metrics(risk_free_rate)
            cagr = temp_metrics.get("cagr_pct", 0) / 100
            max_drawdown = abs(temp_metrics.get("max_drawdown_pct", 0) / 100)
        else:
            cagr = self.metrics.get("cagr_pct", 0) / 100
            max_drawdown = abs(self.metrics.get("max_drawdown_pct", 0) / 100)

        if max_drawdown == 0:
            return 0.0

        return (cagr - risk_free_rate) / max_drawdown

    def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculates the Value at Risk (VaR)."""
        if returns.empty:
            return 0.0
        return returns.quantile(1 - confidence_level)

    def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculates the Conditional Value at Risk (CVaR)."""
        if returns.empty:
            return 0.0
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    def calculate_performance_metrics(self, risk_free_rate: float = 0.02) -> None:
        """Calculate and print performance metrics."""
        # Ensure daily returns are calculated before metrics
        self.calculate_daily_returns()

        self._cached_metrics = self._calculate_metrics(risk_free_rate)
        metrics = self._cached_metrics

        if "error" in metrics:
            print("Error calculating metrics: {}".format(metrics["error"]))
            return

        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            if key != "error":
                formatted_key = key.replace("_pct", " (%)").replace("_", " ").title()
                print("{}: {}".format(formatted_key, value))

    def export_trade_log(
        self, filepath: str, include_open_trades: bool = False
    ) -> None:
        """Export trade log to CSV."""
        self.trade_logger.to_csv(filepath, include_open_trades)

    def print_final_summary(self) -> None:
        """Print a final summary of the portfolio performance."""
        print("\n=== Final Portfolio Summary ===")
        print(f"Initial Cash: ${self.initial_cash:,.2f}")
        print(f"Final Cash: ${self.cash:,.2f}")
        print(f"Total Portfolio Value: ${self.current_total_value:,.2f}")
        print(f"Total Return: ${self.current_total_value - self.initial_cash:,.2f}")
        print(
            f"Total Return %: {((self.current_total_value - self.initial_cash) / self.initial_cash) * 100:.2f}%"
        )

        if self.positions:
            print("\nOpen Positions:")
            for symbol, position in self.positions.items():
                print(
                    f"  {symbol}: {position.quantity:.2f} shares @ ${position.last_price:.2f} (Unrealized P&L: ${position.unrealized_pnl:.2f})"
                )

        completed_trades = self.trade_logger.get_completed_trades()
        if completed_trades:
            print(f"Completed Trades: {len(completed_trades)}")
            total_pnl = sum(trade.net_pnl for trade in completed_trades)
            print(f"Total Realized P&L: ${total_pnl:.2f}")
