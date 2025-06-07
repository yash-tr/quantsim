"""Abstract base class for data handlers in the QuantSim system.

This module defines the `DataHandler` abstract base class that all data handlers
must implement. It provides a common interface for accessing market data, whether
from CSV files, live feeds, or other sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd


class DataHandler(ABC):
    """Abstract base class for all data handlers.

    A DataHandler is responsible for providing market data to the system.
    It must implement methods for retrieving the latest bar, historical data,
    and iterating through the data chronologically.

    Attributes:
        symbols (List[str]): List of symbols this handler provides data for.
    """

    def __init__(self, symbols: List[str], **kwargs: Any):
        """Initializes the DataHandler.

        Args:
            symbols (List[str]): List of symbols to handle.
            **kwargs: Additional configuration parameters.
        """
        self.symbols = symbols

    @abstractmethod
    def get_latest_bar(
        self, symbol: str
    ) -> Optional[Tuple[pd.Timestamp, Dict[str, Any]]]:
        """Returns the most recent bar for a symbol.

        Args:
            symbol (str): The symbol to get data for.

        Returns:
            Optional[Tuple[pd.Timestamp, Dict[str, Any]]]: A tuple of (timestamp, bar_data)
                if data is available, None otherwise. The bar_data dictionary should contain
                at least 'open', 'high', 'low', 'close', and 'volume' keys.
        """
        raise NotImplementedError("Should implement get_latest_bar()")

    @abstractmethod
    def get_latest_bars(
        self, symbol: str, n: int = 1
    ) -> Optional[List[Tuple[pd.Timestamp, Dict[str, Any]]]]:
        """Returns the n most recent bars for a symbol.

        Args:
            symbol (str): The symbol to get data for.
            n (int, optional): Number of bars to return. Defaults to 1.

        Returns:
            Optional[List[Tuple[pd.Timestamp, Dict[str, Any]]]]: A list of (timestamp, bar_data)
                tuples if data is available, None otherwise. The bar_data dictionary should
                contain at least 'open', 'high', 'low', 'close', and 'volume' keys.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Returns historical data for a symbol.

        Args:
            symbol (str): The symbol to get data for.
            start_date (Optional[Union[str, pd.Timestamp]], optional): Start date for data.
                Defaults to None (earliest available).
            end_date (Optional[Union[str, pd.Timestamp]], optional): End date for data.
                Defaults to None (latest available).
            **kwargs: Additional parameters for data retrieval.

        Returns:
            pd.DataFrame: DataFrame containing historical data with columns for
                timestamp, open, high, low, close, and volume.
        """
        raise NotImplementedError("Should implement get_historical_data()")

    @abstractmethod
    def _prepare_data_iterator(
        self,
    ) -> Iterator[Tuple[pd.Timestamp, str, Dict[str, Any]]]:
        """Prepares an iterator for chronological data access.

        This method should be called before using the iterator protocol
        (__iter__ and __next__). It sets up the internal state needed
        for iteration.

        Returns:
            Iterator[Tuple[pd.Timestamp, str, Dict[str, Any]]]: An iterator that yields
                (timestamp, symbol, bar_data) tuples in chronological order.
        """
        raise NotImplementedError("Should implement _prepare_data_iterator()")

    def __iter__(self) -> Iterator[Tuple[pd.Timestamp, str, Dict[str, Any]]]:
        """Returns an iterator for chronological data access.

        This method should be called after _prepare_data_iterator() to
        start iterating through the data.

        Returns:
            Iterator[Tuple[pd.Timestamp, str, Dict[str, Any]]]: An iterator that yields
                (timestamp, symbol, bar_data) tuples in chronological order.
        """
        return self._prepare_data_iterator()

    def __next__(self) -> Tuple[pd.Timestamp, str, Dict[str, Any]]:
        """Returns the next data point in chronological order.

        This method is called by the iterator protocol to get the next
        data point. It should be used after calling __iter__().

        Returns:
            Tuple[pd.Timestamp, str, Dict[str, Any]]: A tuple of (timestamp, symbol, bar_data).

        Raises:
            StopIteration: When there is no more data to iterate through.
        """
        try:
            return next(self._iterator)
        except StopIteration:
            self._on_iterator_exhausted()
            raise

    def _on_iterator_exhausted(self) -> None:
        """Called when the data iterator is exhausted.

        This method can be overridden by subclasses to perform cleanup
        or other actions when all data has been iterated through.
        """
        pass

    @property
    @abstractmethod
    def continue_backtest(self) -> bool:
        """Indicates whether there is more data to process.

        Returns:
            bool: True if there is more data to process, False otherwise.
        """
        raise NotImplementedError("Should implement continue_backtest()")
