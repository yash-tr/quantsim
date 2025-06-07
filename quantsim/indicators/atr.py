"""Provides calculation for Average True Range (ATR).

ATR is a technical analysis volatility indicator originally developed by J. Welles Wilder Jr.
This module contains the `calculate_atr` function to compute ATR values from
high, low, and close price series.
"""

import pandas as pd
import numpy as np


def calculate_atr(
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculates the Average True Range (ATR).

    The True Range (TR) is defined as the greatest of the following:
    - Current High less the current Low
    - Absolute value of the current High less the previous Close
    - Absolute value of the current Low less the previous Close

    ATR is then calculated as a Wilder's Smoothing Moving Average of TR.
    The first ATR value is a simple average of the TR for the initial `period`.
    Subsequent values are calculated as:
    `ATR = (PreviousATR * (period - 1) + CurrentTR) / period`

    Args:
        high_prices (pd.Series): A pandas Series of high prices.
        low_prices (pd.Series): A pandas Series of low prices.
        close_prices (pd.Series): A pandas Series of close prices.
                                  The index of all three series must align.
        period (int, optional): The lookback period for ATR calculation.
                                Defaults to 14.

    Returns:
        pd.Series: A pandas Series containing the ATR values, with the same index
                   as the input price series. Values before the first full period
                   will be `NaN`. Returns an empty Series with the original index
                   if input data length is less than the period.

    Raises:
        ValueError: If inputs are not pandas Series or have mismatched lengths,
                    or if `period` is not a positive integer.
    """
    if (
        not isinstance(high_prices, pd.Series)
        or not isinstance(low_prices, pd.Series)
        or not isinstance(close_prices, pd.Series)
    ):
        raise ValueError("Inputs (high, low, close) must be pandas Series.")

    if not (len(high_prices) == len(low_prices) == len(close_prices)):
        raise ValueError("Input Series (high, low, close) must have the same length.")

    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer.")

    if len(high_prices) < period:
        # Not enough data to calculate any ATR value for the given period
        return pd.Series(dtype=np.float64, index=high_prices.index)

    # Calculate True Range (TR)
    tr1 = high_prices - low_prices
    prev_close = close_prices.shift(1)
    tr2 = (high_prices - prev_close).abs()
    tr3 = (low_prices - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    # First TR is High - Low as there's no previous close for Wilder's original definition.
    # However, yfinance and others might use H-L for first TR point if prev_close is NaN.
    # pd.concat and .max handles NaNs from prev_close.shift(1) correctly for tr2, tr3.
    # For the very first point, if prev_close is NaN, tr2 & tr3 will be NaN. max of (H-L, NaN, NaN) is H-L.
    # So, explicit override for true_range.iloc[0] might not be strictly needed if data starts clean.
    # However, to be absolutely sure for the first data point (index 0):
    if len(high_prices) > 0:  # Ensure there's at least one data point
        true_range.iloc[0] = high_prices.iloc[0] - low_prices.iloc[0]

    # Calculate ATR using Wilder's Smoothing (which is a specific type of EMA)
    # atr = true_range.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    # The above EWM is close but Wilder's has a specific startup.
    # Manual calculation for Wilder's:
    atr = pd.Series(index=high_prices.index, dtype=np.float64)

    # First ATR value is the SMA of the first 'period' TR values
    if len(true_range) >= period:
        atr.iloc[period - 1] = true_range.iloc[:period].mean()
    else:  # Should have been caught by earlier length check, but defensive
        return pd.Series(dtype=np.float64, index=high_prices.index)

    # Subsequent ATR values using Wilder's smoothing formula
    for i in range(period, len(high_prices)):
        if pd.isna(atr.iloc[i - 1]):  # Should not happen if period-1 was calculated
            # This might occur if true_range had NaNs that propagated into first ATR calc.
            # Fallback for robustness, though ideally input data is clean.
            atr.iloc[i] = (
                true_range.iloc[i - period + 1 : i + 1].mean()
                if len(true_range.iloc[i - period + 1 : i + 1].dropna()) == period
                else np.nan
            )
        else:
            atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + true_range.iloc[i]) / period

    return atr
