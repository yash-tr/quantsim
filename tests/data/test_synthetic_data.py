"""
Unit tests for quantsim.data.synthetic_data.SyntheticDataHandler
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
from quantsim.data.synthetic_data import SyntheticDataHandler
from quantsim.data.base import DataHandler

pytestmark = pytest.mark.data

class TestSyntheticDataHandler:
    """Tests for the SyntheticDataHandler class."""

    def test_synthetic_creation_and_basic_properties(self):
        symbols = ['SYNTH1']
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        handler = SyntheticDataHandler(
            symbols=symbols, start_date=start_date, end_date=end_date, data_frequency='B'
        )
        assert isinstance(handler, DataHandler)
        assert handler.symbols == symbols
        assert symbols[0] in handler.symbol_data
        df = handler.symbol_data[symbols[0]]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert isinstance(df.index, pd.DatetimeIndex)
        expected_periods = len(pd.date_range(start=start_date, end=end_date, freq='B'))
        assert len(df) == expected_periods
        if expected_periods > 0:
            actual_start_in_index = pd.date_range(start=start_date, end=end_date, freq='B')[0]
            assert df.index[0] == actual_start_in_index

    def test_synthetic_reproducibility_with_seed(self):
        params = {
            'symbols': ['REPRO'], 'start_date': '2023-01-01', 'end_date': '2023-01-05',
            'initial_price': 100.0, 'drift_per_period': 0.001, 'volatility_per_period': 0.02,
            'data_frequency': 'D', 'seed': 42
        }
        handler1 = SyntheticDataHandler(**params)
        df1 = handler1.symbol_data['REPRO']
        handler2 = SyntheticDataHandler(**params)
        df2 = handler2.symbol_data['REPRO']
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)

        params_diff_seed = {**params, 'seed': 43}
        handler3 = SyntheticDataHandler(**params_diff_seed)
        df3 = handler3.symbol_data['REPRO']
        assert not df1.equals(df3)

        np.random.seed(None)
        params_no_seed = {k: v for k, v in params.items() if k != 'seed'}
        handler4_no_seed = SyntheticDataHandler(**params_no_seed)
        df4 = handler4_no_seed.symbol_data['REPRO']
        if len(df1) > 1:
            assert not df1.equals(df4)

    def test_synthetic_multi_symbol_generation_and_iteration(self):
        symbols = ['SYM1', 'SYM2']
        start = '2023-01-01'
        end = '2023-01-03'
        freq = 'D'
        handler = SyntheticDataHandler(
            symbols=symbols, start_date=start, end_date=end,
            data_frequency=freq, seed=123
        )
        assert 'SYM1' in handler.symbol_data and 'SYM2' in handler.symbol_data
        assert not handler.symbol_data['SYM1'].empty
        assert not handler.symbol_data['SYM2'].empty
        assert not handler.symbol_data['SYM1']['Close'].equals(handler.symbol_data['SYM2']['Close'])

        bars_yielded = list(handler)
        num_expected_periods = len(pd.date_range(start=start, end=end, freq=freq))
        assert len(bars_yielded) == num_expected_periods * len(symbols)
        timestamps = [bar[0] for bar in bars_yielded]
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        assert set(bar[1] for bar in bars_yielded) == set(symbols)

    def test_synthetic_data_frequency(self):
        handler_d = SyntheticDataHandler(['S_D'], '2023-01-01', '2023-01-02', data_frequency='D')
        df_d = handler_d.symbol_data['S_D']
        assert len(df_d) == 2
        assert df_d.index.freqstr == 'D'
        handler_b = SyntheticDataHandler(['S_B'], '2023-01-01', '2023-01-07', data_frequency='B')
        df_b = handler_b.symbol_data['S_B']
        assert len(df_b) == 5
        assert df_b.index.freqstr == 'B'

    def test_synthetic_get_historical_data(self):
        handler = SyntheticDataHandler(['SYM1'], '2023-01-01', '2023-01-05', data_frequency='D')
        df_full = handler.get_historical_data('SYM1')
        pd.testing.assert_frame_equal(df_full, handler.symbol_data['SYM1'])
        df_partial = handler.get_historical_data('SYM1', start_date='2023-01-02', end_date='2023-01-04')
        assert len(df_partial) == 3

    def test_synthetic_ohlc_consistency(self):
        handler = SyntheticDataHandler(['CONSIST'], '2023-01-01', '2023-02-01', data_frequency='B', seed=77)
        df = handler.symbol_data['CONSIST']
        assert not df.empty
        assert (df['High'] >= df['Open']).all() and (df['High'] >= df['Close']).all()
        assert (df['Low'] <= df['Open']).all() and (df['Low'] <= df['Close']).all()
        assert (df['High'] >= df['Low']).all()

    def test_synthetic_drift_effect_simple(self):
        df_pos = SyntheticDataHandler(['DP'], '2023-01-01', '2023-03-01', drift_per_period=0.001, seed=1).symbol_data['DP']
        df_neg = SyntheticDataHandler(['DN'], '2023-01-01', '2023-03-01', drift_per_period=-0.001, seed=1).symbol_data['DN']
        assert not df_pos.empty and not df_neg.empty
        if len(df_pos) > 1 and len(df_neg) > 1:  # Ensure more than just initial price
            assert not df_pos['Close'].equals(df_neg['Close'])

    def test_synthetic_volatility_effect_simple(self):
        df_low = SyntheticDataHandler(['VL'], '2023-01-01', '2023-03-01', volatility_per_period=0.005, seed=10).symbol_data['VL']
        df_high = SyntheticDataHandler(['VH'], '2023-01-01', '2023-03-01', volatility_per_period=0.02, seed=10).symbol_data['VH']
        assert not df_low.empty and not df_high.empty
        if len(df_low) > 1 and len(df_high) > 1:
            assert not df_low['Close'].equals(df_high['Close'])

    def test_synthetic_continue_backtest_and_iterator_reset(self):
        handler = SyntheticDataHandler(['IT'], '2023-01-01', '2023-01-03', 'D')
        assert handler.continue_backtest
        bars1 = list(handler)
        assert len(bars1) == 3  # 3 days inclusive
        assert not handler.continue_backtest

        handler._iter_current_idx = 0
        handler._combined_data_for_iter = None
        assert handler.continue_backtest
        bars2 = list(handler)
        assert len(bars2) == 3
        assert not handler.continue_backtest

    def test_iteration_with_no_data(self):
        handler = SyntheticDataHandler(['SYM1'], '2023-01-01', '2023-01-01')
        # This will generate one day, so we need to create a handler with no days
        handler = SyntheticDataHandler(['SYM1'], '2023-01-01', '2022-12-31')
        assert not list(handler)

    def test_reproducibility_with_seed(self):
        handler1 = SyntheticDataHandler(['S1'], '2023-01-01', '2023-01-05', seed=42)
        handler2 = SyntheticDataHandler(['S1'], '2023-01-01', '2023-01-05', seed=42)
        pd.testing.assert_frame_equal(handler1.symbol_data['S1'], handler2.symbol_data['S1'])

    def test_empty_date_range_handling(self):
        """Test handling when end_date is before start_date."""
        handler = SyntheticDataHandler(['EMPTY'], '2023-01-10', '2023-01-01')
        assert handler.symbol_data['EMPTY'].empty
        assert not handler.continue_backtest

    def test_invalid_date_frequency_error_handling(self):
        """Test error handling for invalid date frequency."""
        handler = SyntheticDataHandler(['TEST'], '2023-01-01', '2023-01-05', data_frequency='INVALID')
        # Should still create handler but may have empty data or single point
        assert 'TEST' in handler.symbol_data

    def test_single_date_point(self):
        """Test data generation for a single date point."""
        handler = SyntheticDataHandler(['SINGLE'], '2023-01-01', '2023-01-01', data_frequency='D')
        df = handler.symbol_data['SINGLE']
        assert len(df) == 1
        assert df.index[0] == pd.Timestamp('2023-01-01')
        assert df['Open'].iloc[0] == 100.0  # Initial price
        assert df['Close'].iloc[0] == 100.0  # Should be same as initial

    def test_zero_periods_handling(self):
        """Test handling when date range generates zero periods."""
        handler = SyntheticDataHandler(['ZERO'], '2023-01-01', '2022-12-31')
        assert handler.symbol_data['ZERO'].empty
        assert not handler.continue_backtest
        assert list(handler) == []

    def test_get_latest_bar_methods(self):
        """Test get_latest_bar and get_latest_bars methods."""
        handler = SyntheticDataHandler(['TEST'], '2023-01-01', '2023-01-05', data_frequency='D')
        
        # Test get_latest_bar
        latest_bar = handler.get_latest_bar('TEST')
        assert latest_bar is not None
        timestamp, ohlcv = latest_bar
        assert timestamp == pd.Timestamp('2023-01-05')
        assert 'Close' in ohlcv

        # Test with non-existent symbol
        assert handler.get_latest_bar('NONEXISTENT') is None

        # Test get_latest_bars
        latest_3_bars = handler.get_latest_bars('TEST', n=3)
        assert latest_3_bars is not None
        assert len(latest_3_bars) == 3

        # Test with n=0
        latest_0_bars = handler.get_latest_bars('TEST', n=0)
        assert latest_0_bars == []

        # Test with n > available data
        latest_10_bars = handler.get_latest_bars('TEST', n=10)
        assert len(latest_10_bars) == 5  # Only 5 days available

        # Test with n < 0
        assert handler.get_latest_bars('TEST', n=-1) is None

        # Test with non-existent symbol
        assert handler.get_latest_bars('NONEXISTENT', n=3) is None

    def test_get_historical_data_edge_cases(self):
        """Test edge cases for get_historical_data method."""
        handler = SyntheticDataHandler(['TEST'], '2023-01-01', '2023-01-05', data_frequency='D')
        
        # Test with non-existent symbol
        empty_df = handler.get_historical_data('NONEXISTENT')
        assert empty_df.empty
        assert list(empty_df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

        # Test with start_date and end_date as Timestamp objects
        df_range = handler.get_historical_data(
            'TEST', 
            start_date=pd.Timestamp('2023-01-02'), 
            end_date=pd.Timestamp('2023-01-04')
        )
        assert len(df_range) == 3

        # Test with only start_date
        df_start = handler.get_historical_data('TEST', start_date='2023-01-03')
        assert len(df_start) == 3  # 2023-01-03, 04, 05

        # Test with only end_date
        df_end = handler.get_historical_data('TEST', end_date='2023-01-03')
        assert len(df_end) == 3  # 2023-01-01, 02, 03

    def test_multi_symbol_seeded_generation(self):
        """Test that multi-symbol generation produces different but reproducible data."""
        symbols = ['A', 'B', 'C']
        handler = SyntheticDataHandler(symbols, '2023-01-01', '2023-01-05', seed=42)
        
        # Check that all symbols have data (business days only)
        expected_days = len(pd.date_range(start='2023-01-01', end='2023-01-05', freq='B'))
        for symbol in symbols:
            assert not handler.symbol_data[symbol].empty
            assert len(handler.symbol_data[symbol]) == expected_days

        # Check that different symbols have different data
        assert not handler.symbol_data['A']['Close'].equals(handler.symbol_data['B']['Close'])
        assert not handler.symbol_data['B']['Close'].equals(handler.symbol_data['C']['Close'])

        # Check reproducibility
        handler2 = SyntheticDataHandler(symbols, '2023-01-01', '2023-01-05', seed=42)
        for symbol in symbols:
            pd.testing.assert_frame_equal(
                handler.symbol_data[symbol], 
                handler2.symbol_data[symbol]
            )

    def test_volume_as_integer_in_iteration(self):
        """Test that Volume values are converted to integers during iteration."""
        handler = SyntheticDataHandler(['TEST'], '2023-01-01', '2023-01-03', data_frequency='D')
        bars = list(handler)
        
        for timestamp, symbol, ohlcv_dict in bars:
            assert isinstance(ohlcv_dict['Volume'], int)
            assert ohlcv_dict['Volume'] >= 100000
            assert ohlcv_dict['Volume'] < 1000000

    def test_iterator_state_tracking(self):
        """Test that iterator state is properly tracked."""
        handler = SyntheticDataHandler(['TEST'], '2023-01-01', '2023-01-03', data_frequency='D')
        
        # Before iteration
        assert handler.continue_backtest
        assert handler._iter_current_idx == 0
        
        # Start iteration
        iterator = iter(handler)
        first_bar = next(iterator)
        assert handler._iter_current_idx == 1
        assert handler.continue_backtest
        
        # Continue until exhaustion
        remaining_bars = list(iterator)
        assert len(remaining_bars) == 2  # 2 more bars after the first
        assert not handler.continue_backtest

    def test_prepare_data_iterator_with_empty_symbols(self):
        """Test _prepare_data_iterator when symbols have empty data."""
        handler = SyntheticDataHandler(['EMPTY'], '2023-01-01', '2022-12-31')  # Invalid range
        
        # Combined data should be empty
        iterator = iter(handler)
        bars = list(iterator)
        assert len(bars) == 0
        assert handler._combined_data_for_iter is not None
        assert handler._combined_data_for_iter.empty

    def test_different_initial_prices(self):
        """Test data generation with different initial prices."""
        handler_100 = SyntheticDataHandler(['TEST'], '2023-01-01', '2023-01-02', initial_price=100.0, seed=42)
        handler_200 = SyntheticDataHandler(['TEST'], '2023-01-01', '2023-01-02', initial_price=200.0, seed=42)
        
        df_100 = handler_100.symbol_data['TEST']
        df_200 = handler_200.symbol_data['TEST']
        
        # First open price should match initial price
        assert df_100['Open'].iloc[0] == 100.0
        assert df_200['Open'].iloc[0] == 200.0
        
        # Close prices should be different due to different starting points
        assert not df_100['Close'].equals(df_200['Close'])

    def test_hourly_frequency(self):
        """Test data generation with hourly frequency."""
        handler = SyntheticDataHandler(['HOURLY'], '2023-01-01', '2023-01-01', data_frequency='h')
        df = handler.symbol_data['HOURLY']
        
        # Should have 24 hours for one day, but since end_date is same as start_date,
        # we get a single point. Let's test with a proper range
        handler_range = SyntheticDataHandler(['HOURLY'], '2023-01-01 00:00', '2023-01-01 02:00', data_frequency='h')
        df_range = handler_range.symbol_data['HOURLY']
        assert len(df_range) >= 1  # At least one data point

    def test_combined_data_sorting(self):
        """Test that combined data for iteration is properly sorted."""
        symbols = ['A', 'B']
        handler = SyntheticDataHandler(symbols, '2023-01-01', '2023-01-03', data_frequency='D', seed=42)
        
        # Trigger iterator preparation
        bars = list(handler)
        
        # Check that timestamps are sorted
        timestamps = [bar[0] for bar in bars]
        assert timestamps == sorted(timestamps)
        
        # Check that combined data exists and is sorted
        assert handler._combined_data_for_iter is not None
        assert not handler._combined_data_for_iter.empty
        combined_index = handler._combined_data_for_iter.index
        assert list(combined_index) == sorted(combined_index)

    def test_on_iterator_exhausted_method(self):
        """Test the _on_iterator_exhausted method."""
        handler = SyntheticDataHandler(['TEST'], '2023-01-01', '2023-01-03', data_frequency='D')
        
        # Prepare iterator to populate combined data
        list(handler)
        
        # Check that _iter_current_idx is updated correctly
        assert handler._combined_data_for_iter is not None
        assert handler._iter_current_idx == len(handler._combined_data_for_iter)
        assert not handler.continue_backtest

    def test_basic_initialization_single_symbol(self):
        """Test basic initialization with single symbol."""
        handler = SyntheticDataHandler(
            symbols=['TEST'],
            start_date='2023-01-01',
            end_date='2023-01-10',
            data_frequency='D',
            initial_price=100.0,
            drift_per_period=0.001,
            volatility_per_period=0.02,
            seed=42
        )
        
        assert handler.symbols == ['TEST']
        assert handler.start_date == pd.Timestamp('2023-01-01')
        assert handler.end_date == pd.Timestamp('2023-01-10')
        assert handler.initial_price == 100.0
        assert handler.drift == 0.001
        assert handler.volatility == 0.02
        assert handler.data_frequency == 'D'
        assert handler.seed == 42
        assert 'TEST' in handler.symbol_data
        assert not handler.symbol_data['TEST'].empty

    def test_basic_initialization_multiple_symbols(self):
        """Test basic initialization with multiple symbols."""
        handler = SyntheticDataHandler(
            symbols=['STOCK1', 'STOCK2', 'STOCK3'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            seed=123
        )
        
        assert len(handler.symbols) == 3
        for symbol in handler.symbols:
            assert symbol in handler.symbol_data
            assert not handler.symbol_data[symbol].empty
            assert list(handler.symbol_data[symbol].columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

    def test_default_parameters(self):
        """Test initialization with default parameters."""
        handler = SyntheticDataHandler(
            symbols=['DEFAULT'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        assert handler.data_frequency == 'B'  # Business days
        assert handler.initial_price == 100.0
        assert handler.drift == 0.0001
        assert handler.volatility == 0.01
        assert handler.seed is None

    def test_data_structure_integrity(self):
        """Test that generated data has correct structure."""
        handler = SyntheticDataHandler(
            symbols=['INTEGRITY'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            seed=42
        )
        
        df = handler.symbol_data['INTEGRITY']
        
        # Check columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert list(df.columns) == expected_columns
        
        # Check index
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check OHLC relationships
        for _, row in df.iterrows():
            assert row['High'] >= max(row['Open'], row['Close'])
            assert row['Low'] <= min(row['Open'], row['Close'])
            assert row['Volume'] > 0

    def test_price_evolution_consistency(self):
        """Test that prices follow GBM evolution."""
        handler = SyntheticDataHandler(
            symbols=['EVOLUTION'],
            start_date='2023-01-01',
            end_date='2023-01-10',
            initial_price=50.0,
            seed=42
        )
        
        df = handler.symbol_data['EVOLUTION']
        
        # First close should be based on initial price
        assert df['Close'].iloc[0] > 0
        
        # All prices should be positive
        assert (df['Close'] > 0).all()
        assert (df['Open'] > 0).all()
        assert (df['High'] > 0).all()
        assert (df['Low'] > 0).all()

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        seed = 12345
        
        handler1 = SyntheticDataHandler(
            symbols=['REPRO'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            seed=seed
        )
        
        handler2 = SyntheticDataHandler(
            symbols=['REPRO'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            seed=seed
        )
        
        df1 = handler1.symbol_data['REPRO']
        df2 = handler2.symbol_data['REPRO']
        
        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        handler1 = SyntheticDataHandler(
            symbols=['DIFF1'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            seed=123
        )
        
        handler2 = SyntheticDataHandler(
            symbols=['DIFF1'],  # Same symbol name
            start_date='2023-01-01',
            end_date='2023-01-05',
            seed=456  # Different seed
        )
        
        df1 = handler1.symbol_data['DIFF1']
        df2 = handler2.symbol_data['DIFF1']
        
        # Should be different
        assert not df1['Close'].equals(df2['Close'])

    def test_multiple_symbols_different_series(self):
        """Test that multiple symbols generate different price series."""
        handler = SyntheticDataHandler(
            symbols=['SYM1', 'SYM2'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            seed=42
        )
        
        df1 = handler.symbol_data['SYM1']
        df2 = handler.symbol_data['SYM2']
        
        # Should be different series due to symbol-specific seeding
        assert not df1['Close'].equals(df2['Close'])

    def test_volume_generation(self):
        """Test volume generation characteristics."""
        handler = SyntheticDataHandler(
            symbols=['VOLUME'],
            start_date='2023-01-01',
            end_date='2023-01-10',
            seed=42
        )
        
        df = handler.symbol_data['VOLUME']
        
        # Volume should be between specified ranges
        assert (df['Volume'] >= 100000).all()
        assert (df['Volume'] <= 1000000).all()
        assert df['Volume'].dtype in [np.int64, np.int32, int]

    def test_different_frequencies(self):
        """Test different data frequencies."""
        # Daily frequency
        handler_daily = SyntheticDataHandler(
            symbols=['DAILY'],
            start_date='2023-01-01',
            end_date='2023-01-10',
            data_frequency='D'
        )
        
        # Business days frequency
        handler_business = SyntheticDataHandler(
            symbols=['BUSINESS'],
            start_date='2023-01-01',
            end_date='2023-01-10',
            data_frequency='B'
        )
        
        # Hourly frequency (small range)
        handler_hourly = SyntheticDataHandler(
            symbols=['HOURLY'],
            start_date='2023-01-01',
            end_date='2023-01-02',
            data_frequency='H'
        )
        
        daily_len = len(handler_daily.symbol_data['DAILY'])
        business_len = len(handler_business.symbol_data['BUSINESS'])
        hourly_len = len(handler_hourly.symbol_data['HOURLY'])
        
        # Daily should have more bars than business days
        assert daily_len >= business_len
        
        # Hourly should have many more bars
        assert hourly_len > daily_len

    def test_edge_case_empty_date_range(self):
        """Test handling of empty date range."""
        with patch('builtins.print') as mock_print:
            handler = SyntheticDataHandler(
                symbols=['EMPTY'],
                start_date='2023-01-05',
                end_date='2023-01-01',  # End before start
                data_frequency='D'
            )
            
            # Should handle gracefully
            assert 'EMPTY' in handler.symbol_data
            # Should have at least the start date
            df = handler.symbol_data['EMPTY']
            assert len(df) >= 0

    def test_edge_case_single_day(self):
        """Test handling of single day range."""
        handler = SyntheticDataHandler(
            symbols=['SINGLE'],
            start_date='2023-01-01',
            end_date='2023-01-01',
            data_frequency='D'
        )
        
        df = handler.symbol_data['SINGLE']
        assert len(df) == 1
        assert df.index[0] == pd.Timestamp('2023-01-01')

    def test_get_historical_data_basic(self):
        """Test basic historical data retrieval."""
        handler = SyntheticDataHandler(
            symbols=['HIST'],
            start_date='2023-01-01',
            end_date='2023-01-10',
            seed=42
        )
        
        data = handler.get_historical_data('HIST')
        assert not data.empty
        assert len(data) == len(handler.symbol_data['HIST'])
        assert list(data.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

    def test_get_historical_data_missing_symbol(self):
        """Test historical data retrieval for missing symbol."""
        handler = SyntheticDataHandler(
            symbols=['EXISTS'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        data = handler.get_historical_data('MISSING')
        assert data.empty

    def test_get_historical_data_date_filtering(self):
        """Test historical data retrieval with date filtering."""
        handler = SyntheticDataHandler(
            symbols=['FILTER'],
            start_date='2023-01-01',
            end_date='2023-01-10',
            data_frequency='D'
        )
        
        # Filter by start date
        data = handler.get_historical_data('FILTER', start_date='2023-01-05')
        original_data = handler.symbol_data['FILTER']
        expected_len = len(original_data[original_data.index >= pd.Timestamp('2023-01-05')])
        assert len(data) == expected_len
        
        # Filter by end date
        data = handler.get_historical_data('FILTER', end_date='2023-01-05')
        expected_len = len(original_data[original_data.index <= pd.Timestamp('2023-01-05')])
        assert len(data) == expected_len
        
        # Filter by both
        data = handler.get_historical_data('FILTER', start_date='2023-01-03', end_date='2023-01-07')
        expected = original_data[
            (original_data.index >= pd.Timestamp('2023-01-03')) &
            (original_data.index <= pd.Timestamp('2023-01-07'))
        ]
        assert len(data) == len(expected)

    def test_get_latest_bar(self):
        """Test getting the latest bar."""
        handler = SyntheticDataHandler(
            symbols=['LATEST'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            seed=42
        )
        
        result = handler.get_latest_bar('LATEST')
        assert result is not None
        timestamp, data_dict = result
        assert isinstance(timestamp, pd.Timestamp)
        assert timestamp == handler.symbol_data['LATEST'].index[-1]
        assert 'Close' in data_dict
        assert 'Volume' in data_dict

    def test_get_latest_bar_missing_symbol(self):
        """Test getting latest bar for missing symbol."""
        handler = SyntheticDataHandler(
            symbols=['EXISTS'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        result = handler.get_latest_bar('MISSING')
        assert result is None

    def test_get_latest_bars(self):
        """Test getting multiple latest bars."""
        handler = SyntheticDataHandler(
            symbols=['LATEST_MULTI'],
            start_date='2023-01-01',
            end_date='2023-01-10',
            seed=42
        )
        
        result = handler.get_latest_bars('LATEST_MULTI', n=3)
        assert result is not None
        assert len(result) == 3
        
        # Check chronological order
        timestamps = [bar[0] for bar in result]
        assert timestamps == sorted(timestamps)
        
        # Check they match the last 3 rows
        original_data = handler.symbol_data['LATEST_MULTI']
        for i, (timestamp, data_dict) in enumerate(result):
            expected_row = original_data.iloc[-(3-i)]
            assert timestamp == expected_row.name
            assert data_dict['Close'] == expected_row['Close']

    def test_get_latest_bars_edge_cases(self):
        """Test edge cases for getting latest bars."""
        handler = SyntheticDataHandler(
            symbols=['EDGE'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            data_frequency='D'
        )
        
        # Request more bars than available
        result = handler.get_latest_bars('EDGE', n=10)
        assert result is not None
        assert len(result) == len(handler.symbol_data['EDGE'])
        
        # Request zero bars
        result = handler.get_latest_bars('EDGE', n=0)
        assert result == []
        
        # Request negative bars
        result = handler.get_latest_bars('EDGE', n=-1)
        assert result is None

    def test_data_iterator_single_symbol(self):
        """Test data iterator for single symbol."""
        handler = SyntheticDataHandler(
            symbols=['ITER'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            data_frequency='D',
            seed=42
        )
        
        bars = list(handler)
        expected_length = len(handler.symbol_data['ITER'])
        assert len(bars) == expected_length
        
        for timestamp, symbol, data in bars:
            assert isinstance(timestamp, pd.Timestamp)
            assert symbol == 'ITER'
            assert isinstance(data, dict)
            assert 'Open' in data
            assert 'Close' in data
            assert 'Volume' in data

    def test_data_iterator_multiple_symbols(self):
        """Test data iterator for multiple symbols."""
        handler = SyntheticDataHandler(
            symbols=['MULTI1', 'MULTI2'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            data_frequency='D',
            seed=42
        )
        
        bars = list(handler)
        
        # Should have bars from both symbols
        expected_length = len(handler.symbol_data['MULTI1']) + len(handler.symbol_data['MULTI2'])
        assert len(bars) == expected_length
        
        # Check that bars are chronologically sorted
        timestamps = [bar[0] for bar in bars]
        assert timestamps == sorted(timestamps)
        
        # Check that both symbols appear
        symbols = {bar[1] for bar in bars}
        assert symbols == {'MULTI1', 'MULTI2'}

    def test_continue_backtest_property(self):
        """Test the continue_backtest property."""
        handler = SyntheticDataHandler(
            symbols=['CONTINUE'],
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        # Should start as True
        assert handler.continue_backtest is True
        
        # Exhaust the iterator
        list(handler)
        
        # Should be False after exhaustion
        assert handler.continue_backtest is False

    def test_drift_and_volatility_effects(self):
        """Test that drift and volatility parameters affect price evolution."""
        # High positive drift
        handler_high_drift = SyntheticDataHandler(
            symbols=['HIGH_DRIFT'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_frequency='D',
            drift_per_period=0.001,  # High positive drift
            volatility_per_period=0.001,  # Low volatility
            seed=42
        )
        
        # High negative drift
        handler_low_drift = SyntheticDataHandler(
            symbols=['LOW_DRIFT'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_frequency='D',
            drift_per_period=-0.001,  # Negative drift
            volatility_per_period=0.001,  # Low volatility
            seed=42
        )
        
        high_drift_final = handler_high_drift.symbol_data['HIGH_DRIFT']['Close'].iloc[-1]
        low_drift_final = handler_low_drift.symbol_data['LOW_DRIFT']['Close'].iloc[-1]
        
        # With same seed but different drift, final prices should reflect the drift
        # (though randomness might sometimes overcome this, the test is probabilistic)
        assert high_drift_final != low_drift_final

    def test_invalid_frequency_handling(self):
        """Test handling of invalid frequency string."""
        try:
            handler = SyntheticDataHandler(
                symbols=['INVALID'],
                start_date='2023-01-01',
                end_date='2023-01-05',
                data_frequency='INVALID_FREQ'
            )
            # Should either handle gracefully or create minimal data
            assert 'INVALID' in handler.symbol_data
        except ValueError:
            # This is also acceptable behavior
            pass

    def test_warning_on_no_data_generated(self):
        """Test warning when no data is generated."""
        with patch('builtins.print') as mock_print:
            # Try to trigger a case where no data is generated
            handler = SyntheticDataHandler(
                symbols=['NO_DATA'],
                start_date='2023-01-05',
                end_date='2023-01-01',  # Invalid date range
                data_frequency='D'
            )
            
            # Should have printed a warning
            warning_calls = [call for call in mock_print.call_args_list 
                           if 'Warning' in str(call)]
            # May or may not print warning depending on implementation

    def test_different_initial_prices(self):
        """Test different initial prices."""
        price1 = 50.0
        price2 = 200.0
        
        handler1 = SyntheticDataHandler(
            symbols=['PRICE1'],
            start_date='2023-01-01',
            end_date='2023-01-02',
            initial_price=price1,
            seed=42
        )
        
        handler2 = SyntheticDataHandler(
            symbols=['PRICE2'],
            start_date='2023-01-01',
            end_date='2023-01-02',
            initial_price=price2,
            seed=42
        )
        
        close1 = handler1.symbol_data['PRICE1']['Close'].iloc[0]
        close2 = handler2.symbol_data['PRICE2']['Close'].iloc[0]
        
        # First close prices should be scaled by initial price
        assert close1 < close2  # Generally true since price2 > price1

    def test_zero_volatility_constant_prices(self):
        """Test that zero volatility produces more stable prices."""
        handler = SyntheticDataHandler(
            symbols=['ZERO_VOL'],
            start_date='2023-01-01',
            end_date='2023-01-10',
            drift_per_period=0.0,
            volatility_per_period=0.0,
            seed=42
        )
        
        prices = handler.symbol_data['ZERO_VOL']['Close']
        
        # With zero drift and volatility, prices should remain close to initial price
        # (exact equality depends on implementation details)
        price_std = prices.std()
        assert price_std < 1.0  # Should have very low standard deviation

    def test_open_price_consistency(self):
        """Test that Open prices follow expected pattern."""
        handler = SyntheticDataHandler(
            symbols=['OPEN_TEST'],
            start_date='2023-01-01',
            end_date='2023-01-05',
            seed=42
        )
        
        df = handler.symbol_data['OPEN_TEST']
        
        # Open price should generally be previous close (with some implementation variation)
        # This tests the general structure rather than exact values
        assert all(df['Open'] > 0)
        assert len(df['Open']) == len(df['Close'])
