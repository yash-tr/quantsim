"""
A simple machine learning-based trading strategy.
"""

from quantsim.ml.base_ml_strategy import BaseMLStrategy
from quantsim.core.events import MarketEvent, OrderEvent
from quantsim.ml.feature_generator import FeatureGenerator
from typing import Any


class SimpleMLStrategy(BaseMLStrategy):
    """
    A simple machine learning strategy that loads a pre-trained model
    and generates trading signals based on its predictions.
    """

    def generate_features(self, data):
        """
        Generates features for the ML model.
        This is a placeholder and should be customized.
        """
        feature_generator = FeatureGenerator(data)
        feature_list = [
            "returns",
            "volatility",
            "momentum",
            "rsi",
            "macd",
            "lagged_returns",
        ]
        features_df = feature_generator.generate_features(feature_list)
        return features_df.dropna()

    def generate_signal(self, event: MarketEvent, prediction: Any) -> None:
        """
        Generates a trading signal based on the model's prediction.
        Prediction[0] > 0.5 -> Buy
        Prediction[0] < 0.5 -> Sell
        """
        if prediction[0][0] > 0.5:
            self.event_queue.put_event(OrderEvent(event.symbol, "MKT", 100, "BUY"))
        elif prediction[0][0] < 0.5:
            self.event_queue.put_event(OrderEvent(event.symbol, "MKT", 100, "SELL"))
