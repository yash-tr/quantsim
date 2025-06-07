"""
Orchestrates the training of machine learning models for use in trading strategies.
"""

import pandas as pd
import joblib
import os

# Optional ML dependencies
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from quantsim.data import YahooFinanceDataHandler
from quantsim.ml.feature_generator import FeatureGenerator


class ModelTrainer:
    """
    Handles the end-to-end process of training, evaluating, and saving an ML model.

    Requires optional ML dependencies (install with: pip install quantsim[ml])
    """

    def __init__(
        self,
        symbols: list,
        start_date: str,
        end_date: str,
        model_type: str,
        features: list,
        target_lag: int,
        output_path: str,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for ML training. "
                "Install with: pip install quantsim[ml]"
            )

        if model_type == "simple_nn" and not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for neural network training. "
                "Install with: pip install quantsim[ml]"
            )

        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.model_type = model_type
        self.feature_list = features
        self.target_lag = target_lag
        self.output_path = output_path
        self.scaler_path = os.path.splitext(output_path)[0] + "_scaler.joblib"
        self.model = None
        self.scaler = StandardScaler()

    def _prepare_data(self) -> tuple:
        """
        Fetches data, generates features, and prepares the final dataset for training.
        """
        print("Fetching and preparing data...")
        data_handler = YahooFinanceDataHandler(
            symbols=self.symbols, start_date=self.start_date, end_date=self.end_date
        )

        all_data = []
        for symbol in self.symbols:
            hist_data = data_handler.get_historical_data(symbol)
            if hist_data is not None and not hist_data.empty:
                all_data.append(hist_data)

        if not all_data:
            raise ValueError(
                "No historical data could be fetched for the given symbols."
            )

        data = pd.concat(all_data)

        feature_gen = FeatureGenerator(data)
        features_df = feature_gen.generate_features(self.feature_list)

        # Define target variable: 1 if future price is higher, 0 otherwise
        features_df["future_return"] = (
            features_df.groupby("symbol")["close"].shift(self.target_lag).pct_change()
        )
        features_df["target"] = (features_df["future_return"] > 0).astype(int)

        # Drop rows with NaN values resulting from feature generation
        features_df.dropna(inplace=True)

        if features_df.empty:
            raise ValueError(
                "Feature generation resulted in an empty DataFrame. Check data and feature parameters."
            )

        X = features_df[self.feature_list]
        y = features_df["target"]

        return X, y

    def train(self):
        """
        Trains the specified machine learning model.
        """
        X, y = self._prepare_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training a {self.model_type} model...")

        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42)
            self.model.fit(X_train_scaled, y_train)
        elif self.model_type == "svc":
            self.model = SVC(probability=True, random_state=42)
            self.model.fit(X_train_scaled, y_train)
        elif self.model_type == "simple_nn":
            self.model = self._build_simple_nn(X_train_scaled.shape[1])
            self.model.fit(
                X_train_scaled,
                y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=1,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        print("\nModel training complete. Evaluating...")
        self._evaluate(X_test_scaled, y_test)

    def _build_simple_nn(self, input_dim):
        """Builds a simple TensorFlow/Keras neural network."""
        model = Sequential(
            [
                Dense(64, activation="relu", input_dim=input_dim),
                Dropout(0.5),
                Dense(32, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def _evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test set and prints a classification report.
        """
        if self.model_type in ["logistic_regression", "svc"]:
            y_pred = self.model.predict(X_test)
        else:  # simple_nn
            y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def save_model(self):
        """
        Saves the trained model and the feature scaler to disk.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        print(f"Saving model to {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        if self.model_type in ["logistic_regression", "svc"]:
            joblib.dump(self.model, self.output_path)
        else:  # simple_nn
            self.model.save(self.output_path)

        # Save the scaler
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Saving scaler to {self.scaler_path}")

    def run_training_pipeline(self):
        """
        Executes the full training pipeline: data prep, training, and saving.
        """
        self.train()
        self.save_model()
        print("\n--- Training Pipeline Finished ---")
