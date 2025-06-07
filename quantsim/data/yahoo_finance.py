"""Data handler for fetching historical market data from Yahoo Finance.

This module provides `YahooFinanceDataHandler`, a concrete implementation
of `DataHandler` that retrieves OHLCV data from Yahoo Finance using the
`yfinance` library. It can handle multiple symbols and prepares data for
chronological iteration by the simulation engine.
"""

import pandas as pd
import yfinance as yf
from typing import List, Tuple, Dict, Any, Iterator, Optional, Union  # Added Union
from .base import DataHandler
import time


class YahooFinanceDataHandler(DataHandler):
    """Fetches and provides historical market data from Yahoo Finance.

    This handler uses the `yfinance` library to download OHLCV data for a list
    of specified symbols over a given date range and interval. It standardizes
    column names and provides data through the `DataHandler` interface, including
    an iterator that yields bars chronologically across all symbols.

    Attributes:
        start_date (str): The start date for data fetching (YYYY-MM-DD).
        end_date (str): The end date for data fetching (YYYY-MM-DD).
        interval (str): The data interval (e.g., "1d", "1h").
        symbol_data (Dict[str, pd.DataFrame]): A dictionary mapping symbols to their
            respective DataFrames of OHLCV data. DataFrames have a 'Timestamp' index.
        _combined_data_for_iter (Optional[pd.DataFrame]): Internal DataFrame used by the
            iterator, containing merged and sorted data for all symbols.
        _iter_current_idx (int): Current index for the iterator.
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        **kwargs: Any,
    ):
        """Initializes the YahooFinanceDataHandler.

        Args:
            symbols (List[str]): List of stock ticker symbols to fetch data for.
            start_date (str): Start date for data fetching in 'YYYY-MM-DD' format.
            end_date (str): End date for data fetching in 'YYYY-MM-DD' format.
            interval (str, optional): Data interval (e.g., "1d", "1h", "1wk", "1mo").
                Defaults to "1d".
            **kwargs (Any): Additional keyword arguments passed to the base `DataHandler`.
        """
        super().__init__(symbols=symbols, **kwargs)
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.interval: str = interval

        self.symbol_data: Dict[str, pd.DataFrame] = self._fetch_all_symbol_data()

        self._combined_data_for_iter: Optional[pd.DataFrame] = None
        self._iter_current_idx: int = 0

        if not any(not df.empty for df in self.symbol_data.values()):
            print(
                f"Warning: YahooFinanceDataHandler failed to fetch any data for symbols: {self.symbols} "
                f"in range {self.start_date} to {self.end_date} with interval {self.interval}."
            )

    def _fetch_one_symbol(self, symbol: str) -> pd.DataFrame:
        """Fetches and processes OHLCV data for a single symbol from Yahoo Finance.

        Standardizes column names ('Open', 'High', 'Low', 'Close', 'Volume') and
        uses 'Adj Close' as 'Close'. Sets a 'Timestamp' index.

        Args:
            symbol (str): The ticker symbol to fetch.

        Returns:
            pd.DataFrame: A DataFrame with processed OHLCV data, indexed by 'Timestamp'.
                          Returns an empty DataFrame if fetching fails or no data is found.
        """
        try:
            print(
                f"YahooFinanceDataHandler: Fetching data for {symbol} from {self.start_date} to {self.end_date} ({self.interval})"
            )
            data_df = yf.download(
                tickers=symbol,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False,  # Suppress yfinance download progress bar
                show_errors=True,
            )
            if data_df.empty:
                print(
                    f"YahooFinanceDataHandler: No data returned by yfinance for symbol {symbol}."
                )
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

            rename_map = {
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "OriginalClose",
                "Adj Close": "Close",
                "Volume": "Volume",
            }
            data_df = data_df.rename(columns=rename_map)

            # If 'Close' (from 'Adj Close') is not present, try to use 'OriginalClose'
            if "Close" not in data_df.columns and "OriginalClose" in data_df.columns:
                data_df["Close"] = data_df["OriginalClose"]

            standard_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in standard_cols:
                if col not in data_df.columns:
                    if col == "Volume":
                        data_df[col] = 0  # Default volume to 0 instead of NaN
                    else:
                        # For price columns, use Close price as default if available
                        if "Close" in data_df.columns:
                            data_df[col] = data_df["Close"]
                        else:
                            data_df[col] = float("nan")

            data_df = data_df[standard_cols]
            data_df.index.name = (
                "Timestamp"  # yfinance index is usually DatetimeIndex already
            )

            # Only drop rows where Close price is NaN (most critical column)
            return data_df.dropna(subset=["Close"])

        except Exception as e:
            print(
                f"YahooFinanceDataHandler: Error fetching or processing data for {symbol}: {e}"
            )
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def _fetch_all_symbol_data(self) -> Dict[str, pd.DataFrame]:
        """Fetches and processes data for all symbols specified in `self.symbols`.

        Iterates through each symbol, calls `_fetch_one_symbol`, and stores the
        resulting DataFrame in `self.symbol_data`. Includes a small delay
        between API calls.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping symbols to their DataFrames.
        """
        all_data: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            all_data[symbol] = self._fetch_one_symbol(symbol)
            if len(self.symbols) > 1:  # Add delay only if fetching multiple symbols
                time.sleep(0.1)
        return all_data

    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Retrieves historical OHLCV data for a specific symbol from the fetched data.

        Args:
            symbol (str): The symbol for which data is requested.
            start_date (Optional[Union[str, pd.Timestamp]], optional): Filters data from this date.
            end_date (Optional[Union[str, pd.Timestamp]], optional): Filters data up to this date.
            **kwargs (Any): Ignored by this handler.

        Returns:
            pd.DataFrame: DataFrame of historical data, filtered by date if specified.
                          Returns an empty DataFrame if symbol not found or no data in range.
        """
        if symbol not in self.symbol_data or self.symbol_data[symbol].empty:
            return pd.DataFrame()

        df = self.symbol_data[symbol]
        df.index = (
            pd.to_datetime(df.index.values, utc=True).tz_convert(None)
            if df.index.tz is not None
            else pd.to_datetime(df.index.values)
        )
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        return df.copy()

    def get_latest_bar(
        self, symbol: str
    ) -> Optional[Tuple[pd.Timestamp, Dict[str, Any]]]:
        """Returns the last data bar for the specified symbol from the fetched data.

        Args:
            symbol (str): The symbol for which to get the latest bar.

        Returns:
            Optional[Tuple[pd.Timestamp, Dict[str, Any]]]:
                `(timestamp, ohlcv_dict)` or `None`.
        """
        if symbol not in self.symbol_data or self.symbol_data[symbol].empty:
            return None
        latest = self.symbol_data[symbol].iloc[-1]
        return latest.name, latest.to_dict()  # type: ignore

    def get_latest_bars(
        self, symbol: str, n: int = 1
    ) -> Optional[List[Tuple[pd.Timestamp, Dict[str, Any]]]]:
        """Returns the N latest data bars for the specified symbol.

        Args:
            symbol (str): The symbol for which to get the bars.
            n (int, optional): Number of bars to return. Defaults to 1.

        Returns:
            Optional[List[Tuple[pd.Timestamp, Dict[str, Any]]]]: List of bars or `None`.
                Returns empty list if `n=0`.
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
        """Prepares a chronological iterator for all symbols' data.
        (Implementation as per DataHandler ABC)
        """
        all_dfs: List[pd.DataFrame] = []
        for symbol_iter, df_iter in self.symbol_data.items():
            if not df_iter.empty:
                temp_df = df_iter.copy()
                temp_df["Symbol"] = symbol_iter
                all_dfs.append(temp_df)

        if not all_dfs:
            self._combined_data_for_iter = pd.DataFrame()  # No data to iterate
        else:
            self._combined_data_for_iter = pd.concat(all_dfs)
            # Ensure index is a proper, sorted DatetimeIndex
            if not isinstance(self._combined_data_for_iter.index, pd.DatetimeIndex):
                self._combined_data_for_iter.index = pd.to_datetime(
                    self._combined_data_for_iter.index.values
                )
            self._combined_data_for_iter = self._combined_data_for_iter.sort_index(
                kind="mergesort"
            )

        self._iter_current_idx = 0

        # Return generator that yields the data
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
        return self

    def __next__(self) -> Tuple[pd.Timestamp, str, Dict[str, Any]]:
        """Get the next item from the iterator."""
        try:
            return next(self._iterator)
        except StopIteration:
            self._on_iterator_exhausted()
            raise

    def _on_iterator_exhausted(self) -> None:
        """Updates state when the iterator is exhausted."""
        if self._combined_data_for_iter is not None:
            self._iter_current_idx = len(self._combined_data_for_iter)

    @property
    def continue_backtest(self) -> bool:
        """True if there are more bars to process; False otherwise."""
        if self._combined_data_for_iter is None:
            # Iterator not prepared: True if any symbol has data that *could* be iterated.
            return any(not df.empty for df in self.symbol_data.values())
        # Iterator prepared: True if current index is less than total length.
        return self._iter_current_idx < len(self._combined_data_for_iter)
