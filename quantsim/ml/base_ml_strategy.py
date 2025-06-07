"""
Base class for machine learning-based trading strategies.
"""

from quantsim.strategies.base import Strategy
from quantsim.core.events import MarketEvent
from quantsim.core.event_queue import EventQueue
from typing import List, Any
import pandas as pd
import numpy as np

# Optional ML dependencies
try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class BaseMLStrategy(Strategy):
    """
    Base class for strategies that use a machine learning model to generate signals.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        symbols: List[str],
        model_path: str,
        feature_lags: int = 5,
        **kwargs: Any,
    ):
        """
        Initializes the BaseMLStrategy.
        Args:
            event_queue (EventQueue): The system's event queue.
            symbols (List[str]): List of symbols for trading.
            model_path (str): Path to the pre-trained machine learning model.
            feature_lags (int): Number of lags for feature generation.
        """
        super().__init__(event_queue, symbols, **kwargs)
        self.model = self.load_model(model_path)
        self.feature_lags = feature_lags
        self.prices = {symbol: [] for symbol in self.symbols}
        self.min_data_points = feature_lags + 1

    def load_model(self, model_path: str) -> Any:
        """
        Loads the machine learning model.
        """
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def on_market_data(self, event: MarketEvent) -> None:
        """
        Handles new market data, generates features, and makes predictions.
        """
        if event.symbol not in self.symbols:
            return

        self.prices[event.symbol].append(event.close)

        if len(self.prices[event.symbol]) < self.min_data_points:
            return

        # Generate features
        price_series = pd.Series(self.prices[event.symbol])
        features = self.generate_features(price_series)

        # Make prediction
        prediction = self.model.predict(features)
        self.generate_signal(event, prediction)

    def generate_features(self, prices: pd.Series) -> np.ndarray:
        """
        Generates features for the model.
        """
        # This should be implemented by the specific ML strategy
        raise NotImplementedError("Should implement generate_features()")

    def generate_signal(self, event: MarketEvent, prediction: Any) -> None:
        """
        Generates a trading signal based on the model's prediction.
        """
        # This should be implemented by the specific ML strategy
        raise NotImplementedError("Should implement generate_signal()")
