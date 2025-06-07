"""
Generates features from market data for use in machine learning models.
"""

import pandas as pd
import numpy as np


class FeatureGenerator:
    """
    Generates a variety of technical features from OHLCV market data.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
                                 Must also contain a 'symbol' column for multi-asset dataframes.
        """
        self.data = data.copy()
        if "symbol" not in self.data.columns:
            self.data["symbol"] = "default"  # Assign a default symbol if not present

    def _add_returns(self):
        self.data["returns"] = self.data.groupby("symbol")["close"].pct_change()

    def _add_log_returns(self):
        if "returns" not in self.data.columns:
            self._add_returns()
        self.data["log_returns"] = np.log(1 + self.data["returns"])

    def _add_volatility(self, window=21):
        if "log_returns" not in self.data.columns:
            self._add_log_returns()
        self.data["volatility"] = self.data.groupby("symbol")["log_returns"].transform(
            lambda x: x.rolling(window=window).std()
        )

    def _add_momentum(self, window=21):
        if "returns" not in self.data.columns:
            self._add_returns()
        self.data["momentum"] = self.data.groupby("symbol")["returns"].transform(
            lambda x: x.rolling(window=window).mean()
        )

    def _add_rsi(self, window=14):
        if "returns" not in self.data.columns:
            self._add_returns()

        def rsi_calc(group):
            delta = group.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        self.data["rsi"] = self.data.groupby("symbol")["close"].transform(rsi_calc)

    def _add_macd(self, fast_window=12, slow_window=26, signal_window=9):
        def macd_calc(group):
            fast_ema = group.ewm(span=fast_window, adjust=False).mean()
            slow_ema = group.ewm(span=slow_window, adjust=False).mean()
            return fast_ema - slow_ema

        self.data["macd"] = self.data.groupby("symbol")["close"].transform(macd_calc)
        self.data["macd_signal"] = self.data.groupby("symbol")["macd"].transform(
            lambda x: x.ewm(span=signal_window, adjust=False).mean()
        )

    def _add_lagged_returns(self, lags=5):
        if "log_returns" not in self.data.columns:
            self._add_log_returns()
        for lag in range(1, lags + 1):
            self.data[f"lag_{lag}"] = self.data.groupby("symbol")["log_returns"].shift(
                lag
            )

    def generate_features(self, feature_list: list) -> pd.DataFrame:
        """
        Generates the specified list of features.

        Args:
            feature_list (list): A list of strings specifying which features to generate.
                                 e.g., ['returns', 'volatility', 'rsi']

        Returns:
            pd.DataFrame: A DataFrame with the original data and the new feature columns.
        """
        feature_map = {
            "returns": self._add_returns,
            "log_returns": self._add_log_returns,
            "volatility": self._add_volatility,
            "momentum": self._add_momentum,
            "rsi": self._add_rsi,
            "macd": self._add_macd,
            "lagged_returns": self._add_lagged_returns,
        }

        for feature_name in feature_list:
            if feature_name in feature_map:
                feature_map[feature_name]()
            else:
                # Support for direct column names like 'lag_1', 'macd_signal'
                if feature_name not in self.data.columns:
                    print(
                        f"Warning: Feature '{feature_name}' is not a known generation method or existing column."
                    )

        return self.data
