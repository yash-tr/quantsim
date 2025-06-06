"""
Unit tests for YahooFinanceDataHandler.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock

from quantsim.data.yahoo_finance import YahooFinanceDataHandler

pytestmark = pytest.mark.data

class TestYahooFinanceDataHandler:
    """Tests for the YahooFinanceDataHandler class."""

    @pytest.fixture
    def mock_yf_download_success(self):
        """Mock successful yfinance download."""
        def mock_download(*args, **kwargs):
            # Create sample OHLCV data
            dates = pd.date_range('2023-01-01', periods=5, freq='D')
            data = {
                'Open': [100.0, 102.0, 104.0, 103.0, 105.0],
                'High': [102.0, 105.0, 106.0, 105.0, 107.0],
                'Low': [99.0, 101.0, 103.0, 102.0, 104.0],
                'Close': [101.5, 104.2, 105.1, 104.8, 106.3],
                'Adj Close': [101.0, 104.0, 105.0, 104.5, 106.0],
                'Volume': [10000, 12000, 11000, 13000, 14000]
            }
            return pd.DataFrame(data, index=dates)
        
        with patch('quantsim.data.yahoo_finance.yf.download', side_effect=mock_download):
            yield mock_download

    @pytest.fixture
    def mock_yf_download_empty(self):
        """Mock empty yfinance download."""
        def mock_download(*args, **kwargs):
            return pd.DataFrame()
        
        with patch('quantsim.data.yahoo_finance.yf.download', side_effect=mock_download):
            yield mock_download

    @pytest.fixture
    def mock_yf_download_error(self):
        """Mock yfinance download with error."""
        def mock_download(*args, **kwargs):
            raise Exception("Network error")
        
        with patch('quantsim.data.yahoo_finance.yf.download', side_effect=mock_download):
            yield mock_download

    @pytest.fixture
    def mock_yf_download_missing_adj_close(self):
        """Mock yfinance download without Adj Close column."""
        def mock_download(*args, **kwargs):
            dates = pd.date_range('2023-01-01', periods=3, freq='D')
            data = {
                'Open': [100.0, 102.0, 104.0],
                'High': [102.0, 105.0, 106.0],
                'Low': [99.0, 101.0, 103.0],
                'Close': [101.5, 104.2, 105.1],
                'Volume': [10000, 12000, 11000]
                # No 'Adj Close'
            }
            return pd.DataFrame(data, index=dates)
        
        with patch('quantsim.data.yahoo_finance.yf.download', side_effect=mock_download):
            yield mock_download

    def test_basic_initialization_single_symbol(self, mock_yf_download_success):
        """Test basic initialization with single symbol."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            interval='1d'
        )
        
        assert handler.symbols == ['AAPL']
        assert handler.start_date == '2023-01-01'
        assert handler.end_date == '2023-01-05'
        assert handler.interval == '1d'
        assert 'AAPL' in handler.symbol_data
        assert not handler.symbol_data['AAPL'].empty
        assert len(handler.symbol_data['AAPL']) == 5

    def test_basic_initialization_multiple_symbols(self, mock_yf_download_success):
        """Test basic initialization with multiple symbols."""
        with patch('quantsim.data.yahoo_finance.time.sleep'):  # Mock sleep to speed up tests
            handler = YahooFinanceDataHandler(
                symbols=['AAPL', 'MSFT', 'GOOGL'],
                start_date='2023-01-01',
                end_date='2023-01-05'
            )
            
            assert len(handler.symbols) == 3
            assert 'AAPL' in handler.symbol_data
            assert 'MSFT' in handler.symbol_data
            assert 'GOOGL' in handler.symbol_data
            
            for symbol in handler.symbols:
                assert not handler.symbol_data[symbol].empty
                assert list(handler.symbol_data[symbol].columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

    def test_column_standardization(self, mock_yf_download_success):
        """Test that columns are properly standardized."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        df = handler.symbol_data['AAPL']
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert list(df.columns) == expected_columns
        
        # Check that Adj Close was used as Close
        assert df['Close'].iloc[0] == 101.0  # Adj Close value
        assert df.index.name == 'Timestamp'

    def test_missing_adj_close_fallback(self, mock_yf_download_missing_adj_close):
        """Test fallback when Adj Close is missing."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-03'
        )
        
        df = handler.symbol_data['AAPL']
        # Should use original Close when Adj Close is missing
        assert df['Close'].iloc[0] == 101.5

    def test_empty_download_handling(self, mock_yf_download_empty):
        """Test handling of empty download result."""
        with patch('builtins.print') as mock_print:
            handler = YahooFinanceDataHandler(
                symbols=['INVALID'],
                start_date='2023-01-01',
                end_date='2023-01-05'
            )
            
            assert handler.symbol_data['INVALID'].empty
            mock_print.assert_called()

    def test_download_error_handling(self, mock_yf_download_error):
        """Test handling of download errors."""
        with patch('builtins.print') as mock_print:
            handler = YahooFinanceDataHandler(
                symbols=['AAPL'],
                start_date='2023-01-01',
                end_date='2023-01-05'
            )
            
            assert handler.symbol_data['AAPL'].empty
            mock_print.assert_called()

    def test_get_historical_data_basic(self, mock_yf_download_success):
        """Test basic historical data retrieval."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        data = handler.get_historical_data('AAPL')
        assert not data.empty
        assert len(data) == 5
        assert list(data.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

    def test_get_historical_data_missing_symbol(self, mock_yf_download_success):
        """Test historical data retrieval for missing symbol."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        data = handler.get_historical_data('MSFT')
        assert data.empty

    def test_get_historical_data_date_filtering(self, mock_yf_download_success):
        """Test historical data retrieval with date filtering."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        # Filter by start date
        data = handler.get_historical_data('AAPL', start_date='2023-01-03')
        assert len(data) == 3
        
        # Filter by end date
        data = handler.get_historical_data('AAPL', end_date='2023-01-03')
        assert len(data) == 3
        
        # Filter by both
        data = handler.get_historical_data('AAPL', start_date='2023-01-02', end_date='2023-01-04')
        assert len(data) == 3

    def test_get_latest_bar(self, mock_yf_download_success):
        """Test getting the latest bar."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        result = handler.get_latest_bar('AAPL')
        assert result is not None
        timestamp, data_dict = result
        assert isinstance(timestamp, pd.Timestamp)
        assert data_dict['Close'] == 106.0
        assert data_dict['Volume'] == 14000

    def test_get_latest_bar_missing_symbol(self, mock_yf_download_success):
        """Test getting latest bar for missing symbol."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        result = handler.get_latest_bar('MSFT')
        assert result is None

    def test_get_latest_bars(self, mock_yf_download_success):
        """Test getting multiple latest bars."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        result = handler.get_latest_bars('AAPL', n=3)
        assert result is not None
        assert len(result) == 3
        
        # Check chronological order
        timestamps = [bar[0] for bar in result]
        assert timestamps == sorted(timestamps)

    def test_get_latest_bars_edge_cases(self, mock_yf_download_success):
        """Test edge cases for getting latest bars."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        # Request more bars than available
        result = handler.get_latest_bars('AAPL', n=10)
        assert result is not None
        assert len(result) == 5
        
        # Request zero bars
        result = handler.get_latest_bars('AAPL', n=0)
        assert result == []
        
        # Request negative bars
        result = handler.get_latest_bars('AAPL', n=-1)
        assert result is None

    def test_data_iterator_single_symbol(self, mock_yf_download_success):
        """Test data iterator for single symbol."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        bars = list(handler)
        assert len(bars) == 5
        
        for timestamp, symbol, data in bars:
            assert isinstance(timestamp, pd.Timestamp)
            assert symbol == 'AAPL'
            assert isinstance(data, dict)
            assert 'Open' in data
            assert 'Close' in data
            assert 'Volume' in data

    def test_data_iterator_multiple_symbols(self, mock_yf_download_success):
        """Test data iterator for multiple symbols."""
        with patch('quantsim.data.yahoo_finance.time.sleep'):
            handler = YahooFinanceDataHandler(
                symbols=['AAPL', 'MSFT'],
                start_date='2023-01-01',
                end_date='2023-01-05'
            )
            
            bars = list(handler)
            assert len(bars) == 10  # 5 bars * 2 symbols
            
            # Check that bars are chronologically sorted
            timestamps = [bar[0] for bar in bars]
            assert timestamps == sorted(timestamps)
            
            # Check that both symbols appear
            symbols = {bar[1] for bar in bars}
            assert symbols == {'AAPL', 'MSFT'}

    def test_continue_backtest_property(self, mock_yf_download_success):
        """Test the continue_backtest property."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        # Should start as True
        assert handler.continue_backtest is True
        
        # Exhaust the iterator
        list(handler)
        
        # Should be False after exhaustion
        assert handler.continue_backtest is False

    def test_timezone_handling(self, mock_yf_download_success):
        """Test timezone handling in data."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        # Test that timezone-aware index is handled correctly
        data = handler.get_historical_data('AAPL')
        
        # The method should handle timezone conversion
        assert isinstance(data.index, pd.DatetimeIndex)

    def test_missing_volume_default_to_zero(self):
        """Test that missing volume defaults to zero."""
        def mock_download_no_volume(*args, **kwargs):
            dates = pd.date_range('2023-01-01', periods=2, freq='D')
            data = {
                'Open': [100.0, 102.0],
                'High': [102.0, 105.0],
                'Low': [99.0, 101.0],
                'Close': [101.5, 104.2],
                'Adj Close': [101.0, 104.0]
                # No Volume column
            }
            return pd.DataFrame(data, index=dates)
        
        with patch('quantsim.data.yahoo_finance.yf.download', side_effect=mock_download_no_volume):
            handler = YahooFinanceDataHandler(
                symbols=['AAPL'],
                start_date='2023-01-01',
                end_date='2023-01-02'
            )
            
            df = handler.symbol_data['AAPL']
            assert 'Volume' in df.columns
            assert df['Volume'].iloc[0] == 0

    def test_missing_price_columns_fallback(self):
        """Test fallback for missing price columns."""
        def mock_download_minimal(*args, **kwargs):
            dates = pd.date_range('2023-01-01', periods=2, freq='D')
            data = {
                'Close': [101.5, 104.2],
                'Adj Close': [101.0, 104.0]
                # Missing Open, High, Low
            }
            return pd.DataFrame(data, index=dates)
        
        with patch('quantsim.data.yahoo_finance.yf.download', side_effect=mock_download_minimal):
            handler = YahooFinanceDataHandler(
                symbols=['AAPL'],
                start_date='2023-01-01',
                end_date='2023-01-02'
            )
            
            df = handler.symbol_data['AAPL']
            # Missing price columns should be filled with Close price (from Adj Close)
            assert df['Open'].iloc[0] == 101.0  # Uses Adj Close as fallback
            assert df['High'].iloc[0] == 101.0
            assert df['Low'].iloc[0] == 101.0
            assert df['Close'].iloc[0] == 101.0  # This is Adj Close
            assert df['Volume'].iloc[0] == 0  # Missing volume defaults to 0

    def test_dropna_only_close_prices(self):
        """Test that only rows with NaN Close prices are dropped."""
        def mock_download_with_nan(*args, **kwargs):
            dates = pd.date_range('2023-01-01', periods=3, freq='D')
            data = {
                'Open': [100.0, np.nan, 104.0],
                'High': [102.0, 105.0, 106.0],
                'Low': [99.0, 101.0, 103.0],
                'Close': [101.5, 104.2, np.nan],  # NaN in Close on last row
                'Adj Close': [101.0, 104.0, np.nan],
                'Volume': [10000, 12000, 11000]
            }
            return pd.DataFrame(data, index=dates)
        
        with patch('quantsim.data.yahoo_finance.yf.download', side_effect=mock_download_with_nan):
            handler = YahooFinanceDataHandler(
                symbols=['AAPL'],
                start_date='2023-01-01',
                end_date='2023-01-03'
            )
            
            df = handler.symbol_data['AAPL']
            # Should drop only the row with NaN Close price
            assert len(df) == 2
            assert not df['Close'].isna().any()
            # Since missing price columns are filled with Close, first row should have filled Open value
            assert df['Open'].iloc[0] == 101.0  # First row Open was NaN, filled with Close (Adj Close)

    def test_different_intervals(self, mock_yf_download_success):
        """Test different data intervals."""
        handler = YahooFinanceDataHandler(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            interval='1h'
        )
        
        assert handler.interval == '1h'
        # The mock will still return daily data, but interval should be stored correctly
        assert not handler.symbol_data['AAPL'].empty

    def test_warning_on_no_data_for_all_symbols(self, mock_yf_download_empty):
        """Test warning when no data is fetched for any symbol."""
        with patch('builtins.print') as mock_print:
            handler = YahooFinanceDataHandler(
                symbols=['INVALID1', 'INVALID2'],
                start_date='2023-01-01',
                end_date='2023-01-05'
            )
            
            # Should print warning about no data fetched
            warning_calls = [call for call in mock_print.call_args_list 
                           if 'Warning' in str(call) and 'failed to fetch any data' in str(call)]
            assert len(warning_calls) > 0
