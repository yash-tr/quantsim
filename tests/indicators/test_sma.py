"""
Unit tests for quantsim.indicators.sma
"""
import pytest
import pandas as pd
import numpy as np
from quantsim.indicators.sma import calculate_sma

pytestmark = pytest.mark.indicators

class TestSMACalculation:
    """Tests for the calculate_sma function."""

    def test_sma_basic_calculation(self):
        """Test basic SMA calculation with valid data."""
        prices = pd.Series([10, 12, 14, 16, 18, 20])
        
        sma = calculate_sma(prices, period=3)
        
        # Check that the result is a pandas Series
        assert isinstance(sma, pd.Series)
        # Check that first two values are NaN (before period)
        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])
        # Check calculated values
        assert sma.iloc[2] == (10 + 12 + 14) / 3  # 12.0
        assert sma.iloc[3] == (12 + 14 + 16) / 3  # 14.0

    def test_sma_input_validation(self):
        """Test input validation for calculate_sma."""
        # Test non-Series input
        with pytest.raises(ValueError, match="must be a pandas Series"):
            calculate_sma([10, 12, 14], period=3)

    def test_sma_invalid_period(self):
        """Test error handling for invalid period values."""
        prices = pd.Series([10, 12, 14, 16])
        
        # Test zero period
        with pytest.raises(ValueError, match="positive integer"):
            calculate_sma(prices, period=0)
            
        # Test negative period
        with pytest.raises(ValueError, match="positive integer"):
            calculate_sma(prices, period=-1)
            
        # Test non-integer period
        with pytest.raises(ValueError, match="positive integer"):
            calculate_sma(prices, period=2.5)

    def test_sma_insufficient_data(self):
        """Test SMA calculation with insufficient data."""
        prices = pd.Series([10, 12])
        
        # Request period longer than data
        sma = calculate_sma(prices, period=5)
        
        # Should return series with all NaN values
        assert len(sma) == len(prices)
        assert sma.isna().all()

    def test_sma_with_nan_values(self):
        """Test SMA calculation with NaN values in input."""
        prices = pd.Series([10, np.nan, 14, 16, 18])
        
        sma = calculate_sma(prices, period=3)
        
        # Should handle NaN values gracefully
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(prices)

    def test_sma_edge_case_single_value(self):
        """Test SMA with single data point."""
        prices = pd.Series([10])
        
        sma = calculate_sma(prices, period=1)
        
        # Should return the single value
        assert len(sma) == 1
        assert sma.iloc[0] == 10.0

    def test_sma_period_equals_length(self):
        """Test SMA when period equals data length."""
        prices = pd.Series([10, 12, 14])
        
        sma = calculate_sma(prices, period=3)
        
        # First two should be NaN, last should be average
        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])
        assert sma.iloc[2] == (10 + 12 + 14) / 3

    def test_sma_period_equals_length(self):
        """Test SMA when period equals data length."""
        prices = pd.Series([10, 12, 14])
        
        sma = calculate_sma(prices, period=3)
        
        # First two should be NaN, last should be average
        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])
        assert sma.iloc[2] == (10 + 12 + 14) / 3 