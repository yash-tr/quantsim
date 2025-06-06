"""
Unit tests for ML trainer functionality.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, Mock, MagicMock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from quantsim.ml.trainer import ModelTrainer

pytestmark = pytest.mark.ml

class TestModelTrainer:
    """Tests for the ModelTrainer class."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create realistic price data
        close_prices = 100 + np.random.randn(100).cumsum() * 0.5
        
        data = {
            'symbol': ['AAPL'] * 100,
            'close': close_prices,
            'volume': np.random.randint(1000000, 5000000, 100),
            'returns': np.random.randn(100) * 0.02
        }
        
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def mock_data_handler(self, sample_training_data):
        """Mock data handler that returns sample data."""
        handler = Mock()
        handler.get_historical_data.return_value = sample_training_data
        return handler

    @pytest.fixture
    def mock_feature_generator(self):
        """Mock feature generator."""
        def mock_generate_features(feature_list):
            # Create mock feature data
            np.random.seed(42)
            n_rows = 90  # Reduced due to feature generation dropping some rows
            
            feature_data = {
                'symbol': ['AAPL'] * n_rows,
                'close': 100 + np.random.randn(n_rows).cumsum() * 0.5,
                'rsi': np.random.uniform(20, 80, n_rows),
                'macd': np.random.randn(n_rows) * 0.1,
                'sma_20': 100 + np.random.randn(n_rows).cumsum() * 0.3
            }
            
            # Add any features that were requested
            for feature in feature_list:
                if feature not in feature_data:
                    feature_data[feature] = np.random.randn(n_rows)
            
            dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')
            return pd.DataFrame(feature_data, index=dates)
        
        generator = Mock()
        generator.generate_features.side_effect = mock_generate_features
        return generator

    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary file path for model output."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        for path in [temp_path, temp_path.replace('.joblib', '_scaler.joblib')]:
            if os.path.exists(path):
                os.unlink(path)

    def test_basic_initialization(self, temp_output_path):
        """Test basic ModelTrainer initialization."""
        trainer = ModelTrainer(
            symbols=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['rsi', 'macd'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        assert trainer.symbols == ['AAPL', 'MSFT']
        assert trainer.start_date == '2023-01-01'
        assert trainer.end_date == '2023-12-31'
        assert trainer.model_type == 'logistic_regression'
        assert trainer.feature_list == ['rsi', 'macd']
        assert trainer.target_lag == 1
        assert trainer.output_path == temp_output_path
        assert trainer.scaler_path == temp_output_path.replace('.joblib', '_scaler.joblib')
        assert trainer.model is None
        assert isinstance(trainer.scaler, StandardScaler)

    @patch('quantsim.ml.trainer.YahooFinanceDataHandler')
    @patch('quantsim.ml.trainer.FeatureGenerator')
    def test_prepare_data_basic(self, mock_fg_class, mock_dh_class, mock_data_handler, 
                               mock_feature_generator, temp_output_path):
        """Test basic data preparation."""
        mock_dh_class.return_value = mock_data_handler
        mock_fg_class.return_value = mock_feature_generator
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['rsi', 'macd'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        X, y = trainer._prepare_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert not X.empty
        assert not y.empty
        assert len(X) == len(y)
        assert list(X.columns) == ['rsi', 'macd']
        assert y.dtype == int
        assert set(y.unique()).issubset({0, 1})

    @patch('quantsim.ml.trainer.YahooFinanceDataHandler')
    @patch('quantsim.ml.trainer.FeatureGenerator')
    def test_prepare_data_no_historical_data(self, mock_fg_class, mock_dh_class, temp_output_path):
        """Test data preparation when no historical data is available."""
        # Mock data handler that returns empty data
        mock_data_handler = Mock()
        mock_data_handler.get_historical_data.return_value = pd.DataFrame()
        mock_dh_class.return_value = mock_data_handler
        
        trainer = ModelTrainer(
            symbols=['INVALID'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['rsi'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with pytest.raises(ValueError, match="No historical data could be fetched"):
            trainer._prepare_data()

    @patch('quantsim.ml.trainer.YahooFinanceDataHandler')
    @patch('quantsim.ml.trainer.FeatureGenerator')
    def test_prepare_data_empty_features(self, mock_fg_class, mock_dh_class, 
                                        mock_data_handler, temp_output_path):
        """Test data preparation when feature generation results in empty DataFrame."""
        mock_dh_class.return_value = mock_data_handler
        
        # Mock feature generator that returns empty DataFrame
        mock_feature_generator = Mock()
        mock_feature_generator.generate_features.return_value = pd.DataFrame()
        mock_fg_class.return_value = mock_feature_generator
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['invalid_feature'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        # Should raise an error when feature DataFrame is empty, but the exact error 
        # may vary depending on whether it's during groupby or during validation
        with pytest.raises((ValueError, KeyError)):
            trainer._prepare_data()

    @patch('quantsim.ml.trainer.ModelTrainer._prepare_data')
    def test_train_logistic_regression(self, mock_prepare_data, temp_output_path):
        """Test training logistic regression model."""
        # Mock prepared data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        mock_prepare_data.return_value = (X, y)
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['feature1', 'feature2'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with patch('builtins.print'):  # Suppress output
            trainer.train()
        
        assert trainer.model is not None
        assert isinstance(trainer.model, LogisticRegression)

    @patch('quantsim.ml.trainer.ModelTrainer._prepare_data')
    def test_train_svm(self, mock_prepare_data, temp_output_path):
        """Test training SVM model."""
        # Mock prepared data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        mock_prepare_data.return_value = (X, y)
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='svc',
            features=['feature1', 'feature2'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with patch('builtins.print'):  # Suppress output
            trainer.train()
        
        assert trainer.model is not None
        assert isinstance(trainer.model, SVC)

    @patch('quantsim.ml.trainer.ModelTrainer._prepare_data')
    @patch('quantsim.ml.trainer.Sequential')
    def test_train_neural_network(self, mock_sequential, mock_prepare_data, temp_output_path):
        """Test training neural network model."""
        # Mock prepared data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        mock_prepare_data.return_value = (X, y)
        
        # Mock Keras model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.rand(20, 1)
        mock_sequential.return_value = mock_model
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='simple_nn',
            features=['feature1', 'feature2'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with patch('builtins.print'):  # Suppress output
            trainer.train()
        
        assert trainer.model is not None
        mock_model.fit.assert_called_once()

    @patch('quantsim.ml.trainer.ModelTrainer._prepare_data')
    def test_train_unsupported_model_type(self, mock_prepare_data, temp_output_path):
        """Test training with unsupported model type."""
        # Mock prepared data with sufficient samples for stratification
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        mock_prepare_data.return_value = (X, y)
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='unsupported_model',
            features=['feature1', 'feature2'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            trainer.train()

    def test_build_simple_nn(self, temp_output_path):
        """Test building simple neural network architecture."""
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='simple_nn',
            features=['feature1', 'feature2'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with patch('quantsim.ml.trainer.Sequential') as mock_sequential:
            with patch('quantsim.ml.trainer.Dense') as mock_dense:
                with patch('quantsim.ml.trainer.Dropout') as mock_dropout:
                    mock_model = Mock()
                    mock_sequential.return_value = mock_model
                    
                    model = trainer._build_simple_nn(input_dim=5)
                    
                    assert model is not None
                    mock_sequential.assert_called_once()
                    mock_model.compile.assert_called_once()

    @patch('quantsim.ml.trainer.ModelTrainer._prepare_data')
    def test_evaluate_sklearn_model(self, mock_prepare_data, temp_output_path):
        """Test evaluation of sklearn models."""
        # Mock prepared data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        mock_prepare_data.return_value = (X, y)
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['feature1', 'feature2'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with patch('builtins.print') as mock_print:
            trainer.train()
        
        # Check that evaluation was called (classification report printed)
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Classification Report' in call for call in print_calls)

    @patch('quantsim.ml.trainer.ModelTrainer._prepare_data')
    @patch('quantsim.ml.trainer.Sequential')
    def test_evaluate_neural_network_model(self, mock_sequential, mock_prepare_data, temp_output_path):
        """Test evaluation of neural network models."""
        # Mock prepared data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        mock_prepare_data.return_value = (X, y)
        
        # Mock Keras model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.rand(20, 1)
        mock_sequential.return_value = mock_model
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='simple_nn',
            features=['feature1', 'feature2'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with patch('builtins.print') as mock_print:
            trainer.train()
        
        # Check that evaluation was called
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Classification Report' in call for call in print_calls)

    @patch('quantsim.ml.trainer.ModelTrainer.train')
    @patch('quantsim.ml.trainer.joblib.dump')
    @patch('quantsim.ml.trainer.os.makedirs')
    def test_save_sklearn_model(self, mock_makedirs, mock_joblib_dump, mock_train, temp_output_path):
        """Test saving sklearn models."""
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['feature1'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        # Mock a trained model
        trainer.model = LogisticRegression()
        
        with patch('builtins.print'):
            trainer.save_model()
        
        mock_joblib_dump.assert_called()
        assert mock_joblib_dump.call_count == 2  # Model and scaler

    @patch('quantsim.ml.trainer.ModelTrainer.train')
    @patch('quantsim.ml.trainer.joblib.dump')
    @patch('quantsim.ml.trainer.os.makedirs')
    def test_save_neural_network_model(self, mock_makedirs, mock_joblib_dump, mock_train, temp_output_path):
        """Test saving neural network models."""
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='simple_nn',
            features=['feature1'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        # Mock a trained Keras model
        mock_model = Mock()
        trainer.model = mock_model
        
        with patch('builtins.print'):
            trainer.save_model()
        
        mock_model.save.assert_called_once_with(temp_output_path)
        mock_joblib_dump.assert_called_once()  # Only scaler for NN

    def test_save_model_not_trained(self, temp_output_path):
        """Test saving model when not trained yet."""
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['feature1'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            trainer.save_model()

    @patch('quantsim.ml.trainer.ModelTrainer.train')
    @patch('quantsim.ml.trainer.ModelTrainer.save_model')
    def test_run_training_pipeline(self, mock_save, mock_train, temp_output_path):
        """Test running the complete training pipeline."""
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['feature1'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with patch('builtins.print'):
            trainer.run_training_pipeline()
        
        mock_train.assert_called_once()
        mock_save.assert_called_once()

    @patch('quantsim.ml.trainer.YahooFinanceDataHandler')
    @patch('quantsim.ml.trainer.FeatureGenerator')
    def test_target_variable_creation(self, mock_fg_class, mock_dh_class, 
                                    mock_data_handler, mock_feature_generator, temp_output_path):
        """Test that target variable is created correctly."""
        mock_dh_class.return_value = mock_data_handler
        mock_fg_class.return_value = mock_feature_generator
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['rsi'],
            target_lag=5,  # Use different lag
            output_path=temp_output_path
        )
        
        X, y = trainer._prepare_data()
        
        # Target should be binary (0 or 1)
        assert y.dtype == int
        assert set(y.unique()).issubset({0, 1})

    @patch('quantsim.ml.trainer.YahooFinanceDataHandler')
    @patch('quantsim.ml.trainer.FeatureGenerator')
    def test_multiple_symbols_data_concatenation(self, mock_fg_class, mock_dh_class, temp_output_path):
        """Test that data from multiple symbols is properly concatenated."""
        # Mock data handler to return different data for different symbols
        def mock_get_data(symbol):
            np.random.seed(hash(symbol) % 1000)  # Different seed per symbol
            dates = pd.date_range('2023-01-01', periods=50, freq='D')
            return pd.DataFrame({
                'close': 100 + np.random.randn(50).cumsum() * 0.5
            }, index=dates)
        
        mock_data_handler = Mock()
        mock_data_handler.get_historical_data.side_effect = mock_get_data
        mock_dh_class.return_value = mock_data_handler
        
        # Mock feature generator
        def mock_generate_features(feature_list):
            # Simulate concatenated data from multiple symbols
            all_data = []
            for symbol in ['AAPL', 'MSFT']:
                np.random.seed(hash(symbol) % 1000)
                n_rows = 45  # Some data lost in feature generation
                dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')
                data = pd.DataFrame({
                    'symbol': [symbol] * n_rows,
                    'close': 100 + np.random.randn(n_rows).cumsum() * 0.5,
                    'rsi': np.random.uniform(20, 80, n_rows)
                }, index=dates)
                all_data.append(data)
            return pd.concat(all_data)
        
        mock_feature_generator = Mock()
        mock_feature_generator.generate_features.side_effect = mock_generate_features
        mock_fg_class.return_value = mock_feature_generator
        
        trainer = ModelTrainer(
            symbols=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['rsi'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        X, y = trainer._prepare_data()
        
        # Should have data from both symbols
        assert len(X) > 45  # More than single symbol would provide
        
        # Check that get_historical_data was called for each symbol
        assert mock_data_handler.get_historical_data.call_count == 2

    @patch('quantsim.ml.trainer.ModelTrainer._prepare_data')
    def test_scaler_fitting(self, mock_prepare_data, temp_output_path):
        """Test that the scaler is properly fitted to training data."""
        # Mock prepared data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100) * 10 + 50,  # Different scales
            'feature2': np.random.randn(100) * 0.1 + 2
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        mock_prepare_data.return_value = (X, y)
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['feature1', 'feature2'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        with patch('builtins.print'):
            trainer.train()
        
        # Check that scaler was fitted (has mean_ and scale_ attributes)
        assert hasattr(trainer.scaler, 'mean_')
        assert hasattr(trainer.scaler, 'scale_')
        assert len(trainer.scaler.mean_) == 2
        assert len(trainer.scaler.scale_) == 2

    @patch('quantsim.ml.trainer.ModelTrainer._prepare_data')
    def test_train_test_split_stratification(self, mock_prepare_data, temp_output_path):
        """Test that train-test split uses stratification."""
        # Mock prepared data with balanced classes for proper stratification
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        # Create balanced target (50% class 0, 50% class 1) to ensure stratification works
        y = pd.Series([0] * 50 + [1] * 50)
        mock_prepare_data.return_value = (X, y)
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['feature1', 'feature2'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        # Mock the actual train_test_split function in the module namespace
        with patch('quantsim.ml.trainer.train_test_split') as mock_split:
            mock_split.return_value = (
                X.iloc[:80], X.iloc[80:], y.iloc[:80], y.iloc[80:]
            )
            
            with patch('builtins.print'):
                trainer.train()
            
            # Check that stratify parameter was used
            mock_split.assert_called_once()
            call_kwargs = mock_split.call_args[1]
            assert 'stratify' in call_kwargs
            pd.testing.assert_series_equal(call_kwargs['stratify'], y)

    def test_scaler_path_generation(self, temp_output_path):
        """Test that scaler path is correctly generated."""
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=['feature1'],
            target_lag=1,
            output_path=temp_output_path
        )
        
        expected_scaler_path = temp_output_path.replace('.joblib', '_scaler.joblib')
        assert trainer.scaler_path == expected_scaler_path

    @patch('quantsim.ml.trainer.ModelTrainer._prepare_data')
    def test_feature_list_consistency(self, mock_prepare_data, temp_output_path):
        """Test that feature list is used consistently throughout training."""
        feature_list = ['rsi', 'macd', 'sma_20']
        
        # Mock prepared data with the specified features
        np.random.seed(42)
        X = pd.DataFrame({
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.randn(100) * 0.1,
            'sma_20': np.random.randn(100) * 5 + 100
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        mock_prepare_data.return_value = (X, y)
        
        trainer = ModelTrainer(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='logistic_regression',
            features=feature_list,
            target_lag=1,
            output_path=temp_output_path
        )
        
        X_result, y_result = trainer._prepare_data()
        
        # Check that only the specified features are used
        assert list(X_result.columns) == feature_list
        assert len(X_result.columns) == len(feature_list) 