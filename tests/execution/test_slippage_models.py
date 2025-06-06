"""
Unit tests for slippage models in quantsim.execution.slippage.
"""
import pytest
import math
from quantsim.execution.slippage import PercentageSlippage, ATRSlippage
from quantsim.core.events import OrderEvent

pytestmark = pytest.mark.execution

class TestPercentageSlippage:
    """Tests for the PercentageSlippage model."""

    def test_percentage_slippage_buy(self):
        model = PercentageSlippage(slippage_rate=0.01)
        market_price = 100.0
        order = OrderEvent('TEST', 'MKT', 100, 'BUY')
        fill_price = model.calculate_slippage(order, market_price)
        assert math.isclose(fill_price, 101.0)

    def test_percentage_slippage_sell(self):
        model = PercentageSlippage(slippage_rate=0.01)
        market_price = 100.0
        order = OrderEvent('TEST', 'MKT', 100, 'SELL')
        fill_price = model.calculate_slippage(order, market_price)
        assert math.isclose(fill_price, 99.0)

    def test_percentage_slippage_zero_rate(self):
        model = PercentageSlippage(slippage_rate=0.0)
        market_price = 123.45
        buy_order = OrderEvent('TEST', 'MKT', 100, 'BUY')
        sell_order = OrderEvent('TEST', 'MKT', 100, 'SELL')
        fill_price_buy = model.calculate_slippage(buy_order, market_price)
        fill_price_sell = model.calculate_slippage(sell_order, market_price)
        assert math.isclose(fill_price_buy, market_price)
        assert math.isclose(fill_price_sell, market_price)

    def test_percentage_slippage_invalid_rate(self):
        with pytest.raises(ValueError):
            PercentageSlippage(slippage_rate=-0.1)
        with pytest.raises(ValueError):
            PercentageSlippage(slippage_rate=1.0)

class TestATRSlippage:
    """Tests for the ATRSlippage model."""

    def test_atr_slippage_buy(self):
        model = ATRSlippage(atr_multiplier=0.5)
        market_price = 100.0
        atr_value = 2.0
        order = OrderEvent('TEST', 'MKT', 100, 'BUY', current_atr=atr_value)
        fill_price = model.calculate_slippage(order, market_price)
        assert math.isclose(fill_price, 100.0 + (2.0 * 0.5))

    def test_atr_slippage_sell(self):
        model = ATRSlippage(atr_multiplier=0.5)
        market_price = 100.0
        atr_value = 2.0
        order = OrderEvent('TEST', 'MKT', 100, 'SELL', current_atr=atr_value)
        fill_price = model.calculate_slippage(order, market_price)
        assert math.isclose(fill_price, 100.0 - (2.0 * 0.5))

    def test_atr_slippage_zero_multiplier(self):
        model = ATRSlippage(atr_multiplier=0.0)
        market_price = 123.45
        atr_value = 2.0
        buy_order = OrderEvent('TEST', 'MKT', 100, 'BUY', current_atr=atr_value)
        sell_order = OrderEvent('TEST', 'MKT', 100, 'SELL', current_atr=atr_value)
        fill_price_buy = model.calculate_slippage(buy_order, market_price)
        fill_price_sell = model.calculate_slippage(sell_order, market_price)
        assert math.isclose(fill_price_buy, market_price)
        assert math.isclose(fill_price_sell, market_price)

    def test_atr_slippage_none_atr(self):
        model = ATRSlippage(atr_multiplier=0.5)
        market_price = 100.0
        order = OrderEvent('TEST', 'MKT', 100, 'BUY', current_atr=None)
        fill_price = model.calculate_slippage(order, market_price)
        assert math.isclose(fill_price, market_price)

    def test_atr_slippage_zero_atr(self):
        model = ATRSlippage(atr_multiplier=0.5)
        market_price = 100.0
        order = OrderEvent('TEST', 'MKT', 100, 'BUY', current_atr=0.0)
        fill_price = model.calculate_slippage(order, market_price)
        assert math.isclose(fill_price, market_price)

    def test_atr_slippage_negative_atr(self):
        model = ATRSlippage(atr_multiplier=0.5)
        market_price = 100.0
        order = OrderEvent('TEST', 'MKT', 100, 'BUY', current_atr=-1.0)
        fill_price = model.calculate_slippage(order, market_price)
        assert math.isclose(fill_price, market_price)

    def test_atr_slippage_invalid_multiplier(self):
        with pytest.raises(ValueError):
            ATRSlippage(atr_multiplier=-0.1)

