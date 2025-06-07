"""
Simple Moving Average (SMA) calculation functions.

This module provides the implementation for calculating Simple Moving Averages
on time series data, specifically stock price data.
"""

import pandas as pd


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculates the Simple Moving Average (SMA) for a given price series.

    The SMA is calculated as the rolling mean of the prices over the specified period.
    This implementation uses `min_periods=period` in the rolling calculation,
    meaning that SMA values will only be computed for windows where enough
    data is available to fill the entire period. Earlier points in the series
    will have `NaN` values.

    Args:
        prices (pd.Series): A pandas Series of prices (e.g., close prices).
            The Series should have a DatetimeIndex if time-series properties are important.
        period (int): The lookback period for the SMA calculation. Must be a
                      positive integer.

    Returns:
        pd.Series: A pandas Series containing the calculated SMA values, aligned
                   with the index of the input `prices` Series. Returns a Series
                   of NaNs with the original index if input data length is less
                   than `period`.

    Raises:
        ValueError: If `prices` is not a pandas Series or if `period` is not
                    a positive integer.
    """
    if not isinstance(prices, pd.Series):
        raise ValueError("Input 'prices' must be a pandas Series.")
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Input 'period' must be a positive integer.")

    # rolling().mean() will naturally produce NaNs at the start if min_periods=period
    # and there isn't enough data for a full window.
    # If len(prices) < period, all values will be NaN.
    return prices.rolling(window=period, min_periods=period).mean()
