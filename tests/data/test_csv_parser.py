"""
Unit tests for quantsim.data.csv_parser.CSVDataManager
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
from quantsim.data.csv_parser import CSVDataManager, load_csv_data
from quantsim.data.base import DataHandler
from unittest.mock import patch, Mock

pytestmark = pytest.mark.data

@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    csv_content = (
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-01,100,102,99,101,1000\n"
        "2023-01-02,101,103,100,102,1200\n"
        "2023-01-03,102,102,98,99,1500\n"
        "2023-01-04,99,100,97,98,1100\n"
        "2023-01-05,98,101,98,100,1300\n"
    )
    csv_file = tmp_path / "sample_data.csv"
    csv_file.write_text(csv_content)
    return csv_file

@pytest.fixture
def sample_custom_cols_csv_path(tmp_path: Path) -> Path:
    csv_content = (
        "time,opening,top,bottom,closing,vol\n"
        "2023-01-01,100,102,99,101,1000\n"
        "2023-01-02,101,103,100,102,1200\n"
    )
    csv_file = tmp_path / "sample_custom_cols.csv"
    csv_file.write_text(csv_content)
    return csv_file

@pytest.fixture
def malformed_csv_path(tmp_path: Path) -> Path:
    csv_content = (
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-01,100,102,99,101,1000\n"
        "2023-01-02,not_a_number,103,100,102,1200\n"
    )
    csv_file = tmp_path / "malformed_data.csv"
    csv_file.write_text(csv_content)
    return csv_file

@pytest.fixture
def missing_columns_csv_path(tmp_path: Path) -> Path:
    csv_content = (
        "Date,Open,Close\n"
        "2023-01-01,100,101\n"
        "2023-01-02,101,102\n"
    )
    csv_file = tmp_path / "missing_columns.csv"
    csv_file.write_text(csv_content)
    return csv_file

@pytest.fixture
def timestamp_column_csv_path(tmp_path: Path) -> Path:
    csv_content = (
        "Timestamp,Open,High,Low,Close,Volume\n"
        "2023-01-01,100,102,99,101,1000\n"
        "2023-01-02,101,103,100,102,1200\n"
    )
    csv_file = tmp_path / "timestamp_column.csv"
    csv_file.write_text(csv_content)
    return csv_file

@pytest.fixture
def no_date_column_csv_path(tmp_path: Path) -> Path:
    csv_content = (
        "Open,High,Low,Close,Volume\n"
        "100,102,99,101,1000\n"
        "101,103,100,102,1200\n"
    )
    csv_file = tmp_path / "no_date_column.csv"
    csv_file.write_text(csv_content)
    return csv_file

@pytest.fixture
def unsorted_csv_path(tmp_path: Path) -> Path:
    csv_content = (
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-03,102,102,98,99,1500\n"
        "2023-01-01,100,102,99,101,1000\n"
        "2023-01-02,101,103,100,102,1200\n"
    )
    csv_file = tmp_path / "unsorted_data.csv"
    csv_file.write_text(csv_content)
    return csv_file

class TestCSVDataManager:
    """Tests for the CSVDataManager class."""

    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file with sample data."""
        csv_content = """Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,98.0,102.0,10000
2023-01-02,102.0,107.0,101.0,106.0,12000
2023-01-03,106.0,110.0,104.0,108.0,11000
2023-01-04,108.0,112.0,106.0,110.0,13000
2023-01-05,110.0,115.0,109.0,113.0,14000"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def custom_column_csv_file(self):
        """Create a CSV file with custom column names."""
        csv_content = """TransactionDate,PriceOpen,PriceHigh,PriceLow,PriceClose,VolumeTraded
2023-01-01,100.0,105.0,98.0,102.0,10000
2023-01-02,102.0,107.0,101.0,106.0,12000
2023-01-03,106.0,110.0,104.0,108.0,11000"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def missing_columns_csv_file(self):
        """Create a CSV file with missing standard columns."""
        csv_content = """Date,Close,Volume
2023-01-01,102.0,10000
2023-01-02,106.0,12000
2023-01-03,108.0,11000"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_csv_data_manager_basic_initialization(self, sample_csv_file):
        """Test basic initialization with standard CSV format."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        assert manager.symbol == 'AAPL'
        assert manager.csv_file_path == sample_csv_file
        assert manager.date_column_in_csv == 'Date'
        assert not manager.data_frame.empty
        assert len(manager.data_frame) == 5
        assert list(manager.data_frame.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert isinstance(manager.data_frame.index, pd.DatetimeIndex)

    def test_csv_data_manager_custom_columns(self, custom_column_csv_file):
        """Test initialization with custom column mapping."""
        column_map = {
            'TransactionDate': 'Timestamp',
            'PriceOpen': 'Open',
            'PriceHigh': 'High',
            'PriceLow': 'Low',
            'PriceClose': 'Close',
            'VolumeTraded': 'Volume'
        }
        
        manager = CSVDataManager(
            symbol='MSFT',
            csv_file_path=custom_column_csv_file,
            date_column='TransactionDate',
            column_map=column_map
        )
        
        assert manager.symbol == 'MSFT'
        assert manager.date_column_in_csv == 'TransactionDate'
        assert not manager.data_frame.empty
        assert len(manager.data_frame) == 3
        assert isinstance(manager.data_frame.index, pd.DatetimeIndex)

    def test_csv_data_manager_missing_columns(self, missing_columns_csv_file):
        """Test handling of missing standard columns."""
        with patch('builtins.print') as mock_print:
            manager = CSVDataManager(symbol='TEST', csv_file_path=missing_columns_csv_file)
            
            # Should have filled missing columns with NaN
            assert not manager.data_frame.empty
            assert 'Open' in manager.data_frame.columns
            assert 'High' in manager.data_frame.columns
            assert 'Low' in manager.data_frame.columns
            assert 'Close' in manager.data_frame.columns
            assert 'Volume' in manager.data_frame.columns
            
            # Should have printed warnings for missing columns
            mock_print.assert_called()

    def test_csv_data_manager_file_not_found(self):
        """Test handling of non-existent CSV file."""
        with patch('builtins.print') as mock_print:
            manager = CSVDataManager(symbol='TEST', csv_file_path='/non/existent/file.csv')
            
            assert manager.data_frame.empty
            mock_print.assert_called()

    def test_get_historical_data_basic(self, sample_csv_file):
        """Test basic historical data retrieval."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        data = manager.get_historical_data('AAPL')
        assert not data.empty
        assert len(data) == 5
        assert list(data.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

    def test_get_historical_data_wrong_symbol(self, sample_csv_file):
        """Test historical data retrieval for wrong symbol."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        with patch('builtins.print') as mock_print:
            data = manager.get_historical_data('MSFT')
            assert data.empty
            mock_print.assert_called()

    def test_get_historical_data_date_filtering(self, sample_csv_file):
        """Test historical data retrieval with date filtering."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        # Filter by start date
        data = manager.get_historical_data('AAPL', start_date='2023-01-03')
        assert len(data) == 3  # Should include 3rd, 4th, 5th rows
        
        # Filter by end date
        data = manager.get_historical_data('AAPL', end_date='2023-01-03')
        assert len(data) == 3  # Should include 1st, 2nd, 3rd rows
        
        # Filter by both
        data = manager.get_historical_data('AAPL', start_date='2023-01-02', end_date='2023-01-04')
        assert len(data) == 3  # Should include 2nd, 3rd, 4th rows

    def test_get_latest_bar(self, sample_csv_file):
        """Test getting the latest bar."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        result = manager.get_latest_bar('AAPL')
        assert result is not None
        timestamp, data_dict = result
        assert isinstance(timestamp, pd.Timestamp)
        assert data_dict['Close'] == 113.0
        assert data_dict['Volume'] == 14000

    def test_get_latest_bar_wrong_symbol(self, sample_csv_file):
        """Test getting latest bar for wrong symbol."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        result = manager.get_latest_bar('MSFT')
        assert result is None

    def test_get_latest_bars(self, sample_csv_file):
        """Test getting multiple latest bars."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        # Get last 3 bars
        result = manager.get_latest_bars('AAPL', n=3)
        assert result is not None
        assert len(result) == 3
        
        # Check that they're in chronological order
        timestamps = [bar[0] for bar in result]
        assert timestamps == sorted(timestamps)

    def test_get_latest_bars_edge_cases(self, sample_csv_file):
        """Test edge cases for getting latest bars."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        # Request more bars than available
        result = manager.get_latest_bars('AAPL', n=10)
        assert result is not None
        assert len(result) == 5  # Should return all available
        
        # Request zero bars
        result = manager.get_latest_bars('AAPL', n=0)
        assert result == []
        
        # Request negative bars
        result = manager.get_latest_bars('AAPL', n=-1)
        assert result is None

    def test_data_iterator(self, sample_csv_file):
        """Test the data iterator functionality."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        # Test iterator
        bars = list(manager)
        assert len(bars) == 5
        
        # Check format of each bar
        for timestamp, symbol, data in bars:
            assert isinstance(timestamp, pd.Timestamp)
            assert symbol == 'AAPL'
            assert isinstance(data, dict)
            assert 'Open' in data
            assert 'Close' in data
            assert 'Volume' in data

    def test_continue_backtest_property(self, sample_csv_file):
        """Test the continue_backtest property."""
        manager = CSVDataManager(symbol='AAPL', csv_file_path=sample_csv_file)
        
        # Should start as True
        assert manager.continue_backtest is True
        
        # Exhaust the iterator
        list(manager)
        
        # Should be False after exhaustion
        assert manager.continue_backtest is False

class TestLoadCSVDataUtility:
    """Tests for the load_csv_data utility function."""

    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file with sample data."""
        csv_content = """Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,98.0,102.0,10000
2023-01-02,102.0,107.0,101.0,106.0,12000"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_load_csv_data_basic(self, sample_csv_file):
        """Test basic CSV data loading utility."""
        df = load_csv_data(symbol='TEST', csv_file_path=sample_csv_file)
        
        assert not df.empty
        assert len(df) == 2
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_csv_data_with_column_map(self, sample_csv_file):
        """Test CSV data loading with custom column mapping."""
        # This should work even with standard columns since the utility is flexible
        column_map = {'Date': 'Timestamp'}
        df = load_csv_data(symbol='TEST', csv_file_path=sample_csv_file, column_map=column_map)
        
        assert not df.empty
        assert len(df) == 2

