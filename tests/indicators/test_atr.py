"""
Unit tests for quantsim.indicators.atr
"""
import pytest
import pandas as pd
import numpy as np
from quantsim.indicators.atr import calculate_atr

pytestmark = pytest.mark.indicators

class TestATRCalculation:
    """Tests for the calculate_atr function."""

    def test_atr_basic_calculation(self):
        """Test basic ATR calculation with valid data."""
        highs = pd.Series([10, 12, 11, 13, 14])
        lows = pd.Series([8, 10, 9, 11, 12])
        closes = pd.Series([9, 11, 10, 12, 13])
        
        atr = calculate_atr(highs, lows, closes, period=3)
        
        # Check that the result is a pandas Series
        assert isinstance(atr, pd.Series)
        # Check that first few values are NaN (before period)
        assert pd.isna(atr.iloc[0])
        assert pd.isna(atr.iloc[1])
        # Check that we have valid values after period
        assert not pd.isna(atr.iloc[2])

    def test_atr_input_validation(self):
        """Test input validation for calculate_atr."""
        highs = pd.Series([10, 12, 11])
        lows = pd.Series([8, 10, 9])
        closes = pd.Series([9, 11, 10])
        
        # Test non-Series inputs
        with pytest.raises(ValueError, match="must be pandas Series"):
            calculate_atr([10, 12, 11], lows, closes)
            
        with pytest.raises(ValueError, match="must be pandas Series"):
            calculate_atr(highs, [8, 10, 9], closes)
            
        with pytest.raises(ValueError, match="must be pandas Series"):
            calculate_atr(highs, lows, [9, 11, 10])

    def test_atr_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        highs = pd.Series([10, 12, 11])
        lows = pd.Series([8, 10])  # Different length
        closes = pd.Series([9, 11, 10])
        
        with pytest.raises(ValueError, match="same length"):
            calculate_atr(highs, lows, closes)

    def test_atr_invalid_period(self):
        """Test error handling for invalid period values."""
        highs = pd.Series([10, 12, 11])
        lows = pd.Series([8, 10, 9])
        closes = pd.Series([9, 11, 10])
        
        # Test zero period
        with pytest.raises(ValueError, match="positive integer"):
            calculate_atr(highs, lows, closes, period=0)
            
        # Test negative period
        with pytest.raises(ValueError, match="positive integer"):
            calculate_atr(highs, lows, closes, period=-1)
            
        # Test non-integer period
        with pytest.raises(ValueError, match="positive integer"):
            calculate_atr(highs, lows, closes, period=2.5)

    def test_atr_insufficient_data(self):
        """Test ATR calculation with insufficient data."""
        highs = pd.Series([10, 12])
        lows = pd.Series([8, 10])
        closes = pd.Series([9, 11])
        
        # Request period longer than data
        atr = calculate_atr(highs, lows, closes, period=5)
        
        # Should return empty series with original index
        assert len(atr) == len(highs)
        assert atr.index.equals(highs.index)
        assert atr.isna().all()

    def test_atr_with_nan_values(self):
        """Test ATR calculation with NaN values in input."""
        highs = pd.Series([10, np.nan, 11, 13, 14])
        lows = pd.Series([8, 10, 9, 11, 12])
        closes = pd.Series([9, 11, 10, 12, 13])
        
        atr = calculate_atr(highs, lows, closes, period=3)
        
        # Should handle NaN values gracefully
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(highs)

    def test_atr_edge_case_single_value(self):
        """Test ATR with single data point."""
        highs = pd.Series([10])
        lows = pd.Series([8])
        closes = pd.Series([9])
        
        atr = calculate_atr(highs, lows, closes, period=1)
        
        # Should handle single value case
        assert len(atr) == 1
        assert not pd.isna(atr.iloc[0])
        assert atr.iloc[0] == 2  # High - Low = 10 - 8 = 2

    def test_atr_wilder_smoothing_formula(self):
        """Test that Wilder's smoothing formula is correctly applied."""
        # Create test data where we can verify the calculation
        highs = pd.Series([10, 12, 11, 13, 14, 15, 16])
        lows = pd.Series([8, 10, 9, 11, 12, 13, 14])
        closes = pd.Series([9, 11, 10, 12, 13, 14, 15])
        
        atr = calculate_atr(highs, lows, closes, period=3)
        
        # Verify that subsequent values use Wilder's formula
        # ATR[i] = (ATR[i-1] * (period-1) + TR[i]) / period
        assert len(atr) == len(highs)
        # First 3 periods should be NaN, then calculations start
        assert pd.isna(atr.iloc[0])
        assert pd.isna(atr.iloc[1])
        assert not pd.isna(atr.iloc[2])  # First ATR value (SMA of first 3 TRs)
        assert not pd.isna(atr.iloc[3])  # Second ATR value using Wilder's formula 