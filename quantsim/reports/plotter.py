"""
Provides plotting functionalities for backtest reports.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from typing import Optional

matplotlib.use("Agg")  # Use non-interactive backend for saving files


def plot_equity_curve(
    equity_curve_df: pd.DataFrame,
    output_path: str,
    title: str = "Portfolio Equity Curve",
    initial_capital: Optional[float] = None,
):
    """
    Generates and saves a plot of the equity curve.

    Args:
        equity_curve_df (pd.DataFrame): DataFrame with 'Timestamp' as index and 'PortfolioValue' as a column.
        output_path (str): The full path to save the plot image (e.g., 'output/equity_curve.png').
        title (str, optional): The title of the plot. Defaults to "Equity Curve".
        initial_capital (float | None, optional): If provided, a horizontal line will be drawn.
    """
    if not isinstance(equity_curve_df, pd.DataFrame) or equity_curve_df.empty:
        print("Plotter: Equity curve data is empty or not a DataFrame. Cannot plot.")
        return

    if "PortfolioValue" not in equity_curve_df.columns:
        print(
            "Plotter: 'PortfolioValue' column not found in equity_curve_df. Cannot plot."
        )
        return

    plt.figure(figsize=(12, 7))
    plt.plot(
        equity_curve_df.index,
        equity_curve_df["PortfolioValue"],
        label="Portfolio Value",
        color="blue",
    )

    if initial_capital is not None:
        plt.axhline(
            y=initial_capital,
            color="grey",
            linestyle="--",
            label=f"Initial Capital (${initial_capital:,.0f})",
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")

    # Format Y-axis to have commas for thousands
    formatter = mticker.FormatStrFormatter("$ {:,.0f}")
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xticks(rotation=45)
    plt.grid(True, which="major", linestyle="--", alpha=0.7)
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.legend()

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Plotter: Equity curve plot saved to {output_path}")
    except Exception as e:
        print(f"Plotter: Error saving equity curve plot - {e}")
    finally:
        plt.close()  # Close the plot figure to free memory


def plot_drawdown_series(
    drawdown_series: pd.Series, output_path: str, title: str = "Portfolio Drawdown"
):
    """
    Generates and saves a plot of the drawdown series.

    Args:
        drawdown_series (pd.Series): Pandas Series with Timestamp index and drawdown
                                     percentage as values (typically negative or zero).
        output_path (str): The full path to save the plot image.
        title (str, optional): The title of the plot. Defaults to "Portfolio Drawdown".
    """
    if not isinstance(drawdown_series, pd.Series) or drawdown_series.empty:
        print("Plotter: Drawdown series is empty or not a Series. Cannot plot.")
        return

    plt.figure(figsize=(12, 7))
    # Multiply by 100 to display as percentages
    plt.plot(
        drawdown_series.index, drawdown_series * 100, label="Drawdown", color="red"
    )
    # Using fill_between to shade the drawdown area
    plt.fill_between(
        drawdown_series.index,
        drawdown_series * 100,
        0,
        color="red",
        alpha=0.3,
        interpolate=True,
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")

    # Format Y-axis as percentage
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    plt.xticks(rotation=45)
    plt.grid(True, which="major", linestyle="--", alpha=0.7)
    plt.tight_layout()
    # plt.legend() # Legend might be redundant if only one line/area

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Plotter: Drawdown plot saved to {output_path}")
    except Exception as e:
        print(f"Plotter: Error saving drawdown plot - {e}")
    finally:
        plt.close()
