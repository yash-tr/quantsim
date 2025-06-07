"""Generates synthetic OHLCV market data for backtesting.

This module provides `SyntheticDataHandler`, an implementation of `DataHandler`
that creates artificial market data using a Geometric Brownian Motion (GBM)
model for close prices. Open, High, and Low prices are then heuristically
derived from the generated Close prices. This handler is useful for testing
strategies in reproducible environments or when external data is unavailable.
"""

import pandas as pd
import numpy as np

# from datetime import datetime, timedelta # timedelta not directly used but good for context
from typing import List, Tuple, Dict, Any, Iterator, Optional, Union
from quantsim.data.base import DataHandler


class SyntheticDataHandler(DataHandler):
    """Generates synthetic OHLCV data using Geometric Brownian Motion.

    This data handler creates artificial time series data for specified symbols
    over a given date range and frequency. Close prices are modeled using GBM,
    and OHL prices are derived based on these Close prices. Volume is randomized.
    It supports reproducibility through a random seed and can generate data for
    multiple symbols, yielding bars chronologically for backtesting.

    Attributes:
        start_date (pd.Timestamp): The start date for data generation.
        end_date (pd.Timestamp): The end date for data generation.
        initial_price (float): The starting price for the GBM model for each symbol.
        drift (float): The drift parameter (per period) for the GBM model.
        volatility (float): The volatility parameter (per period) for the GBM model.
        data_frequency (str): The frequency of data generation (pandas offset string).
        seed (Optional[int]): Random seed for reproducibility.
        symbol_data (Dict[str, pd.DataFrame]): Stores generated OHLCV DataFrames per symbol.
            Each DataFrame is indexed by 'Timestamp' and has 'Open', 'High', 'Low',
            'Close', 'Volume' columns.
        _combined_data_for_iter (Optional[pd.DataFrame]): Internal DataFrame used by the
            iterator, containing merged and sorted data for all symbols. Populated when
            the iterator is first prepared.
        _iter_current_idx (int): Current index for the iterator over combined data. More accurately,
            it tracks how many items have been yielded by the most recent call to
            `_prepare_data_iterator`'s generator if `__next__` were implemented here.
            Given `__next__` is in ABC, this is used by `continue_backtest` after exhaustion.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_frequency: str = "B",
        initial_price: float = 100.0,
        drift_per_period: float = 0.0001,
        volatility_per_period: float = 0.01,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initializes the SyntheticDataHandler.

        Args:
            symbols (List[str]): A list of symbols for which to generate data.
            start_date (str): The start date for data generation (e.g., "YYYY-MM-DD").
            end_date (str): The end date for data generation (e.g., "YYYY-MM-DD").
            data_frequency (str, optional): The frequency of the generated data,
                as a pandas offset string (e.g., "D" for daily, "B" for business day,
                "H" for hourly). Defaults to "B".
            initial_price (float, optional): The starting price for each symbol's
                Close price series. Defaults to 100.0.
            drift_per_period (float, optional): The drift term for the GBM model,
                applied per period according to `data_frequency`. Defaults to 0.0001.
            volatility_per_period (float, optional): The volatility term for the GBM
                model, applied per period. Defaults to 0.01.
            seed (Optional[int], optional): A random seed for NumPy to ensure
                reproducibility. If None, randomness will vary. Defaults to None.
            **kwargs (Any): Additional keyword arguments passed to the base `DataHandler`.
        """
        super().__init__(symbols, **kwargs)
        self.start_date: pd.Timestamp = pd.Timestamp(start_date)
        self.end_date: pd.Timestamp = pd.Timestamp(end_date)
        self.initial_price: float = initial_price
        self.drift: float = drift_per_period
        self.volatility: float = volatility_per_period
        self.data_frequency: str = data_frequency
        self.seed: Optional[int] = seed

        if self.seed is not None:
            np.random.seed(self.seed)

        self.symbol_data: Dict[str, pd.DataFrame] = self._generate_all_symbol_data()

        self._combined_data_for_iter: Optional[pd.DataFrame] = None
        self._iter_current_idx: int = (
            0  # Tracks if iterator has been exhausted for continue_backtest
        )

        if not any(not df.empty for df in self.symbol_data.values()):
            print(
                f"Warning: SyntheticDataHandler generated no data for symbols: {self.symbols}"
            )

    def _generate_ohlcv_for_one_symbol(
        self, date_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Generates OHLCV data for a single symbol using GBM for Close prices.

        Args:
            date_index (pd.DatetimeIndex): The DatetimeIndex for which to generate data.

        Returns:
            pd.DataFrame: A DataFrame containing 'Open', 'High', 'Low', 'Close', 'Volume'
                          columns, indexed by `date_index`.
        """
        num_periods = len(date_index)
        if num_periods == 0:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        prices = np.zeros(num_periods)
        prices[0] = self.initial_price
        for t in range(1, num_periods):
            prices[t] = prices[t - 1] * np.exp(
                (self.drift - 0.5 * self.volatility**2)
                + self.volatility * np.random.normal()
            )

        df = pd.DataFrame(index=date_index)
        df["Close"] = prices
        df["Open"] = df["Close"].shift(1).fillna(self.initial_price)

        price_range_factor = self.volatility * 0.5
        # Generate random positive offsets for High and Low based on Close price and volatility
        # Ensure these offsets are positive to correctly calculate High and Low
        high_offsets = np.abs(
            np.random.normal(
                0, price_range_factor * df["Close"].mean(), size=num_periods
            )
        )
        low_offsets = np.abs(
            np.random.normal(
                0, price_range_factor * df["Close"].mean(), size=num_periods
            )
        )

        df["High"] = np.maximum(df["Open"], df["Close"]) + high_offsets
        df["Low"] = np.minimum(df["Open"], df["Close"]) - low_offsets

        # Ensure High is the highest and Low is the lowest of Open, High, Low, Close
        df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)

        df["Volume"] = np.random.randint(100000, 1000000, size=num_periods)

        return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    def _generate_all_symbol_data(self) -> Dict[str, pd.DataFrame]:
        """Generates and stores OHLCV data for all configured symbols.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping each symbol to its
                                     generated OHLCV DataFrame.
        """
        all_data: Dict[str, pd.DataFrame] = {}
        try:
            date_index = pd.date_range(
                start=self.start_date, end=self.end_date, freq=self.data_frequency
            )
            if date_index.empty and self.start_date <= self.end_date:
                date_index = pd.DatetimeIndex([self.start_date])
        except ValueError as e:
            print(
                f"Error generating date range for SyntheticData: {e}. Defaulting to single start_date point."
            )
            date_index = pd.DatetimeIndex([self.start_date])

        if date_index.empty:
            print(
                "Warning: SyntheticDataHandler generated an empty date index. No data will be produced."
            )
            return {
                sym: pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
                for sym in self.symbols
            }

        for symbol in self.symbols:
            if self.seed is not None:
                symbol_hash_component = sum(ord(c) for c in symbol)
                np.random.seed(self.seed + symbol_hash_component)
            all_data[symbol] = self._generate_ohlcv_for_one_symbol(date_index.copy())
        return all_data

    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Retrieves historical OHLCV data for a symbol from generated data.
        (Implementation as per DataHandler ABC)
        """
        if symbol not in self.symbol_data or self.symbol_data[symbol].empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = self.symbol_data[symbol].copy()
        df.index = pd.to_datetime(df.index)

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        return df

    def get_latest_bar(
        self, symbol: str
    ) -> Optional[Tuple[pd.Timestamp, Dict[str, Any]]]:
        """Returns the last generated data bar for the specified symbol.
        (Implementation as per DataHandler ABC)
        """
        if symbol not in self.symbol_data or self.symbol_data[symbol].empty:
            return None
        latest = self.symbol_data[symbol].iloc[-1]
        return latest.name, latest.to_dict()  # type: ignore

    def get_latest_bars(
        self, symbol: str, n: int = 1
    ) -> Optional[List[Tuple[pd.Timestamp, Dict[str, Any]]]]:
        """Returns the N latest generated data bars for the specified symbol.
        (Implementation as per DataHandler ABC)
        """
        if symbol not in self.symbol_data or self.symbol_data[symbol].empty or n < 0:
            return None
        if n == 0:
            return []

        df_symbol = self.symbol_data[symbol]
        n_to_fetch = min(n, len(df_symbol))
        if n_to_fetch == 0:
            return []
        return [
            (idx, row.to_dict()) for idx, row in df_symbol.iloc[-n_to_fetch:].iterrows()
        ]

    def _prepare_data_iterator(
        self,
    ) -> Iterator[Tuple[pd.Timestamp, str, Dict[str, Any]]]:
        """Prepares a chronological iterator for all symbols' generated data.
        (Implementation as per DataHandler ABC)
        """
        all_dfs_for_iter: List[pd.DataFrame] = []
        for symbol_iter, df_iter in self.symbol_data.items():
            if not df_iter.empty:
                temp_df = df_iter.copy()
                temp_df["Symbol"] = symbol_iter
                all_dfs_for_iter.append(temp_df)

        if not all_dfs_for_iter:
            self._combined_data_for_iter = pd.DataFrame()
        else:
            self._combined_data_for_iter = pd.concat(all_dfs_for_iter)
            if not isinstance(self._combined_data_for_iter.index, pd.DatetimeIndex):
                print(
                    "Warning: Synthetic combined DataFrame index is not DatetimeIndex."
                )
            self._combined_data_for_iter = self._combined_data_for_iter.sort_index(
                kind="mergesort"
            )

        self._iter_current_idx = 0

        if (
            self._combined_data_for_iter is not None
            and not self._combined_data_for_iter.empty
        ):
            for timestamp, row in self._combined_data_for_iter.iterrows():
                symbol_val = row["Symbol"]
                ohlcv_dict = {
                    "Open": row["Open"],
                    "High": row["High"],
                    "Low": row["Low"],
                    "Close": row["Close"],
                    "Volume": int(row["Volume"]),
                }
                self._iter_current_idx += 1
                yield timestamp, symbol_val, ohlcv_dict

    def __iter__(self) -> Iterator[Tuple[pd.Timestamp, str, Dict[str, Any]]]:
        """Returns an iterator for chronological data access."""
        self._iterator = self._prepare_data_iterator()
        return self._iterator

    def _on_iterator_exhausted(self) -> None:
        """Updates internal state when iterator is exhausted."""
        if self._combined_data_for_iter is not None:
            # Mark that iteration is complete relative to the known data length
            self._iter_current_idx = len(self._combined_data_for_iter)

    @property
    def continue_backtest(self) -> bool:
        """True if there are more bars to process; False otherwise."""
        if self._combined_data_for_iter is None:
            # Check if there's any data to iterate at all
            return any(not df.empty for df in self.symbol_data.values())

        # If iterator exists, this property's accuracy depends on whether __next__ has been exhausted.
        # The _iter_current_idx is updated by _on_iterator_exhausted when StopIteration occurs.
        # So, this check reflects if the *previous* iteration completed fully.
        # For a live feed, this logic would be different (e.g. always True or based on connection).
        # For historical/synthetic, it's true if we haven't *finished* a full iteration pass.
        if self._combined_data_for_iter is not None:
            return self._iter_current_idx < len(self._combined_data_for_iter)
        return False  # No combined data means nothing to iterate
