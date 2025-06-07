"""
Generates backtest reports in Markdown format.
"""

import os
import pandas as pd
import numpy as np  # For np.inf formatting
import datetime
from typing import List, Dict, Optional
from .plotter import plot_equity_curve, plot_drawdown_series
from quantsim.portfolio.trade_log import Trade  # Changed from FillRecord to Trade


class ReportGenerator:
    """
    Generates a Markdown report summarizing backtest results, including
    performance metrics and an equity curve plot.
    """

    def __init__(
        self,
        portfolio_metrics: Dict,
        equity_curve: pd.DataFrame,
        completed_trades: List[Trade],  # Changed from fills_log
        output_dir: str,
        strategy_name: str,
        symbol: str,
        initial_capital: float,
        drawdown_series: Optional[pd.Series] = None,
    ):
        """
        Initializes the ReportGenerator.

        Args:
            portfolio_metrics (Dict): Dictionary of calculated performance metrics.
            equity_curve (pd.DataFrame): DataFrame of the portfolio's equity curve.
                                         Index should be Timestamp, must contain 'PortfolioValue' column.
            fills_log (List[FillRecord]): List of FillRecord named tuples.
            output_dir (str): Directory to save the report and associated images.
            strategy_name (str): Name of the strategy used.
            symbol (str): Symbol traded.
            initial_capital (float): The initial capital for the backtest.
        """
        self.portfolio_metrics = portfolio_metrics
        self.equity_curve = equity_curve
        self.completed_trades = completed_trades  # Store completed trades
        self.output_dir = output_dir
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.drawdown_series = drawdown_series

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"ReportGenerator initialized. Output directory: {self.output_dir}")

    def generate_report(self, filename_prefix: str = "backtest_report"):
        """
        Generates and saves the Markdown report.

        Args:
            filename_prefix (str, optional): Prefix for the report filename.
                                             Defaults to "backtest_report".
        """
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = (
            f"{filename_prefix}_{self.symbol}_{self.strategy_name}_{timestamp_str}.md"
        )
        report_filepath = os.path.join(self.output_dir, report_filename)

        # --- Equity Curve ---
        equity_img_filename = (
            f"equity_curve_{self.symbol}_{self.strategy_name}_{timestamp_str}.png"
        )
        equity_curve_img_path = os.path.join(self.output_dir, equity_img_filename)
        print(f"Generating equity curve plot: {equity_curve_img_path}")
        plot_equity_curve(
            self.equity_curve,
            equity_curve_img_path,
            title=f"Equity Curve - {self.strategy_name} on {self.symbol}",
            initial_capital=self.initial_capital,
        )

        # Start building Markdown content
        md_content = []
        md_content.append(f"# Backtest Report: {self.strategy_name} on {self.symbol}\n")
        md_content.append(
            f"Date Generated: {timestamp_str}\n"
        )  # Use consistent timestamp

        md_content.append("## Equity Curve\n")
        md_content.append(f"![Equity Curve]({equity_img_filename})\n")

        # --- Drawdown Plot ---
        if self.drawdown_series is not None and not self.drawdown_series.empty:
            drawdown_img_filename = f"drawdown_series_{self.symbol}_{self.strategy_name}_{timestamp_str}.png"
            drawdown_plot_img_path = os.path.join(
                self.output_dir, drawdown_img_filename
            )
            print(f"Generating drawdown plot: {drawdown_plot_img_path}")
            plot_drawdown_series(
                self.drawdown_series,
                drawdown_plot_img_path,
                title=f"Portfolio Drawdown - {self.strategy_name} on {self.symbol}",
            )
            md_content.append("\n## Drawdown\n")
            md_content.append(f"![Portfolio Drawdown]({drawdown_img_filename})\n")
        else:
            md_content.append("\n## Drawdown\n")
            md_content.append("Drawdown series data not available or empty.\n")

        md_content.append("\n## Performance Metrics\n")
        if self.portfolio_metrics and "error" not in self.portfolio_metrics:
            metric_display_order = [
                "total_return_pct",
                "cagr_pct",
                "annualized_volatility_pct",
                "sharpe_ratio",
                "max_drawdown_pct",
                "realized_pnl",
                "total_trades",
                "num_winning_trades",
                "num_losing_trades",
                "win_rate_pct",
                "gross_profit",
                "gross_loss",
                "profit_factor",
                "avg_win_pnl",
                "avg_loss_pnl",
                "avg_trade_duration_seconds",
            ]
            for key in metric_display_order:
                if key in self.portfolio_metrics:
                    value = self.portfolio_metrics[key]
                    formatted_key = (
                        key.replace("_pct", " (%)").replace("_", " ").title()
                    )
                    if isinstance(value, (float, np.floating)) and not pd.isna(value):
                        if value == np.inf:
                            formatted_value = "inf"
                        elif value == -np.inf:
                            formatted_value = "-inf"
                        else:
                            formatted_value = f"{value:,.2f}"
                        md_content.append(f"- **{formatted_key}:** {formatted_value}")
                    else:
                        md_content.append(f"- **{formatted_key}:** {str(value)}")
            md_content.append("\n")
        elif "error" in self.portfolio_metrics:
            md_content.append(
                f"Metrics calculation error: {self.portfolio_metrics['error']}\n"
            )
        else:
            md_content.append("No metrics available.\n")

        md_content.append("## Trade Log Summary\n")  # Changed from Fill Log
        md_content.append(f"Total Completed Trades: {len(self.completed_trades)}\n")
        # The CLI now names the trade log with symbol and strategy.
        # We can construct that name here if needed, or just refer to it generally.
        md_content.append(
            f"Detailed trade log saved to: '{self.symbol}_{self.strategy_name}_trades.csv' (in the same output directory).\n"
        )

        # Optional: Add a table of top N trades to the report
        # md_content.append("\n### Sample Trades (Top 5 by Net PnL)\n")
        # if self.completed_trades:
        #     sorted_trades = sorted(self.completed_trades, key=lambda t: t.net_pnl, reverse=True)
        #     md_content.append("| Symbol | Direction | Entry Time | Exit Time | Net PnL |")
        #     md_content.append("|--------|-----------|------------|-----------|---------|")
        #     for trade in sorted_trades[:5]:
        #         entry_t = trade.entry_timestamp.strftime('%Y-%m-%d %H:%M')
        #         exit_t = trade.exit_timestamp.strftime('%Y-%m-%d %H:%M') if trade.exit_timestamp else "N/A"
        #         md_content.append(f"| {trade.symbol} | {trade.direction} | {entry_t} | {exit_t} | {trade.net_pnl:.2f} |")
        #     md_content.append("\n")
        # else:
        #     md_content.append("No completed trades to display sample.\n")

        # Write to Markdown file
        try:
            with open(report_filepath, "w") as f:
                for line in md_content:
                    f.write(line + "\n")  # Ensure each item is on a new line in MD
            print(f"Markdown report generated: {report_filepath}")
        except IOError as e:
            print(f"Error writing Markdown report: {e}")
