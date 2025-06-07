"""Handles loading and processing of market data from CSV files.

This module provides `CSVDataManager`, a concrete implementation of `DataHandler`
designed to read market data for a single symbol from a specified CSV file.
It also includes a utility function `load_csv_data` for direct DataFrame loading.
"""

import pandas as pd
from typing import List, Tuple, Dict, Any, Iterator, Optional, Union  # Added Union
from .base import DataHandler


class CSVDataManager(DataHandler):
    """Manages market data for a single symbol loaded from a CSV file.

    This class implements the `DataHandler` interface to provide historical
    OHLCV data from a CSV source. It expects a CSV file where rows represent
    time-series data for a single financial instrument. It handles customizable
    column naming and date parsing.

    Attributes:
        symbol (str): The symbol this instance manages data for.
        csv_file_path (str): Absolute path to the CSV file.
        date_column_in_csv (str): The original name of the date/timestamp column in the CSV.
        column_map (Dict[str, str]): Effective map used to rename CSV columns to standard names.
        data_frame (pd.DataFrame): Processed DataFrame with 'Timestamp' index and
                                   standardized OHLCV columns.
        _iter_current_idx (int): Tracks the current row index for the iterator.
    """

    def __init__(
        self,
        symbol: str,
        csv_file_path: str,
        date_column: str = "Date",
        column_map: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """Initializes the CSVDataManager.

        Args:
            symbol (str): The symbol this data handler is for (e.g., 'AAPL').
            csv_file_path (str): Full path to the CSV file.
            date_column (str, optional): The name of the column in the CSV that contains
                date/timestamp information. Defaults to 'Date'.
            column_map (Optional[Dict[str, str]], optional): A dictionary to map column names
                from the CSV file to standard internal names. Expected standard names are
                'Open', 'High', 'Low', 'Close', 'Volume'. The `date_column` should typically
                be mapped to 'Timestamp' if its original name is different and not 'Date'.
                Example: `{'TransactionDate': 'Timestamp', 'PriceOpen': 'Open', ...}`.
                If None, defaults are assumed (e.g., 'Open' maps to 'Open', `date_column` to 'Timestamp').
            **kwargs (Any): Additional keyword arguments passed to the `DataHandler` base class.
        """
        super().__init__(symbols=[symbol], **kwargs)
        self.symbol: str = symbol
        self.csv_file_path: str = csv_file_path
        self.date_column_in_csv: str = date_column

        standard_names_map = {
            "Timestamp": "Timestamp",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }

        # Effective column map: User map overrides defaults. Date column handled carefully.
        effective_map = {}
        if column_map:  # User-provided map
            effective_map.update(column_map)

        # Ensure date_column_in_csv is set to be parsed as 'Timestamp' if not already explicitly mapped by user
        if (
            self.date_column_in_csv not in effective_map
            and self.date_column_in_csv != "Timestamp"
        ):
            effective_map[self.date_column_in_csv] = "Timestamp"

        # Apply default mappings for standard OHLCV if not covered by user's map
        for std_key, std_val in standard_names_map.items():
            if (
                std_key not in effective_map and std_key not in effective_map.values()
            ):  # if neither key nor value is user-mapped
                # Check if std_key (e.g. "Open") exists as a column name in CSV to map from
                # This part is tricky without reading CSV headers first.
                # Simpler: rely on user_column_map to be comprehensive or CSV to have standard names.
                # For now, the _load_and_process_data will handle missing standard columns.
                pass

        self.column_map: Dict[str, str] = effective_map  # Store the map used

        self.data_frame: pd.DataFrame = self._load_and_process_data()
        self._iter_current_idx: int = 0

    def _load_and_process_data(self) -> pd.DataFrame:
        """Loads data from the CSV, standardizes columns, and sets DatetimeIndex.

        This internal method reads the CSV file specified in `csv_file_path`.
        It renames columns according to `self.column_map`, converts the specified
        date column to datetime objects, and sets it as a 'Timestamp' index.
        It ensures the presence of standard columns ('Open', 'High', 'Low', 'Close', 'Volume'),
        filling missing ones with NaN.

        Returns:
            pd.DataFrame: A processed DataFrame with a 'Timestamp' index and
                          standardized OHLCV columns. Returns an empty DataFrame on error.

        Raises:
            ValueError: If the date column cannot be found or parsed.
        """
        try:
            df = pd.read_csv(self.csv_file_path)

            # Use a copy of self.column_map for renaming to avoid altering the instance dict
            current_rename_map = self.column_map.copy()

            # Identify the column to be parsed as Timestamp.
            # It's either the value of self.date_column_in_csv if it's in current_rename_map keys
            # and maps to 'Timestamp', or it's 'Timestamp' itself if directly present or mapped.
            # Or it's self.date_column_in_csv if no specific mapping makes it 'Timestamp'.

            col_to_parse_as_date = self.date_column_in_csv
            if (
                self.date_column_in_csv in current_rename_map
                and current_rename_map[self.date_column_in_csv] == "Timestamp"
            ):
                # If "OriginalDateCol" is mapped to "Timestamp", then after rename, "Timestamp" is the one to parse.
                # But we need to parse the *original* column.
                # So, if date_column_in_csv is key in map, and its value is 'Timestamp', then date_parse_col is date_column_in_csv.
                pass  # col_to_parse_as_date is already correct
            elif (
                "Timestamp" in df.columns and self.date_column_in_csv not in df.columns
            ):
                # If 'Timestamp' column already exists and original date_column_in_csv doesn't, assume 'Timestamp' is the date col.
                col_to_parse_as_date = "Timestamp"

            # Rename columns that are present in the DataFrame
            df = df.rename(
                columns={k: v for k, v in current_rename_map.items() if k in df.columns}
            )

            # Convert the identified date column to datetime and set as index
            if col_to_parse_as_date in df.columns:
                df["Timestamp"] = pd.to_datetime(df[col_to_parse_as_date])
                df = df.set_index("Timestamp")
                # Drop the original date column if it was different from 'Timestamp' AND was not the index name itself
                if (
                    col_to_parse_as_date != "Timestamp"
                    and col_to_parse_as_date in df.columns
                ):
                    df = df.drop(columns=[col_to_parse_as_date], errors="ignore")
            elif "Timestamp" in df.columns and not isinstance(
                df.index, pd.DatetimeIndex
            ):  # If 'Timestamp' column exists (not index yet)
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df = df.set_index("Timestamp")
            elif not isinstance(
                df.index, pd.DatetimeIndex
            ):  # If index is not datetime but no obvious date column found
                raise ValueError(
                    f"Date column '{self.date_column_in_csv}' (or 'Timestamp') not found or index not DatetimeIndex."
                )

            df = df.sort_index()

            final_cols_no_ts = ["Open", "High", "Low", "Close", "Volume"]
            for col in final_cols_no_ts:
                if col not in df.columns:
                    print(
                        f"Warning: Standard column '{col}' for {self.symbol} not found after processing. Filling with NaN."
                    )
                    df[col] = float("nan")

            return df[final_cols_no_ts]

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file_path}' not found for {self.symbol}.")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        except Exception as e:
            print(
                f"Error loading/processing CSV for {self.symbol} from '{self.csv_file_path}': {e}"
            )
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Retrieves historical OHLCV data for the managed symbol.

        Args:
            symbol (str): The symbol for which data is requested. Must match `self.symbol`.
            start_date (Optional[Union[str, pd.Timestamp]], optional): Filters data from this date.
            end_date (Optional[Union[str, pd.Timestamp]], optional): Filters data up to this date.
            **kwargs (Any): Ignored by this data handler.

        Returns:
            pd.DataFrame: A DataFrame with 'Timestamp' index and OHLCV columns.
                          Returns an empty DataFrame if requested symbol doesn't match
                          or if data filtering results in no data.
        """
        if symbol != self.symbol:
            print(
                f"Warning: CSVDataManager for {self.symbol} received request for {symbol}. No data returned."
            )
            return pd.DataFrame()

        df_to_return = self.data_frame
        if start_date:
            df_to_return = df_to_return[
                df_to_return.index >= pd.to_datetime(start_date)
            ]
        if end_date:
            df_to_return = df_to_return[df_to_return.index <= pd.to_datetime(end_date)]
        return df_to_return.copy()

    def get_latest_bar(
        self, symbol: str
    ) -> Optional[Tuple[pd.Timestamp, Dict[str, Any]]]:
        """Returns the last data bar from the loaded CSV file for the managed symbol.

        Args:
            symbol (str): The symbol for which data is requested. Must match `self.symbol`.

        Returns:
            Optional[Tuple[pd.Timestamp, Dict[str, Any]]]:
                A tuple `(timestamp, ohlcv_dict)` for the last bar, or `None`.
        """
        if symbol != self.symbol or self.data_frame.empty:
            return None
        latest = self.data_frame.iloc[-1]
        return latest.name, latest.to_dict()  # type: ignore

    def get_latest_bars(
        self, symbol: str, n: int = 1
    ) -> Optional[List[Tuple[pd.Timestamp, Dict[str, Any]]]]:
        """Returns the N latest data bars for the managed symbol.

        Args:
            symbol (str): The symbol for which data is requested. Must match `self.symbol`.
            n (int, optional): The number of bars to return. Defaults to 1.

        Returns:
            Optional[List[Tuple[pd.Timestamp, Dict[str, Any]]]]:
                A list of `(timestamp, ohlcv_dict)` tuples, or `None` if no data.
                Returns an empty list if `n=0` and data exists.
        """
        if symbol != self.symbol or self.data_frame.empty or n < 0:
            return None  # n<0 is invalid
        if n == 0:
            return []

        n_to_fetch = min(n, len(self.data_frame))
        if n_to_fetch == 0:
            return []

        return [
            (idx, row.to_dict())
            for idx, row in self.data_frame.iloc[-n_to_fetch:].iterrows()
        ]

    def _prepare_data_iterator(
        self,
    ) -> Iterator[Tuple[pd.Timestamp, str, Dict[str, Any]]]:
        """Prepares an iterator yielding data bars for the managed symbol.

        Yields:
            Iterator[Tuple[pd.Timestamp, str, Dict[str, Any]]]:
                `(timestamp, symbol, ohlcv_dict)` for each row.
        """
        self._iter_current_idx = 0  # Reset for each new iteration via __iter__
        for timestamp, row in self.data_frame.iterrows():
            self._iter_current_idx += 1
            ohlcv_dict = row.to_dict()
            ohlcv_dict["Volume"] = int(
                ohlcv_dict.get("Volume", 0)
            )  # Ensure Volume is int
            yield timestamp, self.symbol, ohlcv_dict

    def __iter__(self) -> Iterator[Tuple[pd.Timestamp, str, Dict[str, Any]]]:
        """Returns an iterator for chronological data access."""
        self._iterator = self._prepare_data_iterator()
        return self._iterator

    def _on_iterator_exhausted(self) -> None:
        """Updates internal state when iterator is exhausted."""
        self._iter_current_idx = len(self.data_frame)

    @property
    def continue_backtest(self) -> bool:
        """True if there are more bars to process; False otherwise.

        Relies on the internal iterator's current index compared to total data length.
        The `SimulationEngine` primarily uses `StopIteration` from the iterator itself.
        """
        return not self.data_frame.empty and self._iter_current_idx < len(
            self.data_frame
        )


def load_csv_data(
    symbol: str,
    csv_file_path: str,
    date_column: str = "Date",
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Utility function to load market data from a CSV file into a pandas DataFrame.

    This function instantiates a `CSVDataManager` and uses its `get_historical_data`
    method to return the processed DataFrame.

    Args:
        symbol (str): The symbol the data represents.
        csv_file_path (str): Path to the CSV file.
        date_column (str, optional): Name of the date column in the CSV. Defaults to 'Date'.
        column_map (Optional[Dict[str, str]], optional): Mapping for CSV column names to
            standard names ('Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp').
            Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the processed market data, indexed by 'Timestamp'.
                      Returns an empty DataFrame if loading fails.
    """
    manager = CSVDataManager(
        symbol=symbol,
        csv_file_path=csv_file_path,
        date_column=date_column,
        column_map=column_map,
    )
    return manager.get_historical_data(symbol)  # This returns a copy already
