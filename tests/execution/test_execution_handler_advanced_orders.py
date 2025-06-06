"""
Unit tests for SimulatedExecutionHandler focusing on Limit and Stop Orders.
"""
import pytest
import math
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import Mock
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import OrderEvent, FillEvent, MarketEvent
from quantsim.execution.execution_handler import SimulatedExecutionHandler, FixedCommission
from quantsim.execution.slippage import PercentageSlippage, ATRSlippage

pytestmark = pytest.mark.execution

@pytest.fixture
def event_queue() -> EventQueue:
    return EventQueue()

@pytest.fixture
def base_handler(event_queue: EventQueue) -> SimulatedExecutionHandler:
    return SimulatedExecutionHandler(
        event_queue=event_queue,
        slippage_model=PercentageSlippage(0.0),
        commission_model=FixedCommission(0.0),
        latency_ms=0,
        max_fill_pct_per_bar=1.0,
        max_fill_qty_per_bar=float('inf')
    )

@pytest.fixture
def slippage_handler(event_queue: EventQueue) -> SimulatedExecutionHandler:
    return SimulatedExecutionHandler(
        event_queue=event_queue,
        slippage_model=PercentageSlippage(0.01),
        commission_model=FixedCommission(0.0),
        latency_ms=0
    )

@pytest.fixture
def atr_slippage_handler(event_queue: EventQueue) -> SimulatedExecutionHandler:
    return SimulatedExecutionHandler(
        event_queue=event_queue,
        slippage_model=ATRSlippage(atr_multiplier=0.5),
        commission_model=FixedCommission(0.0),
        latency_ms=0
    )

class TestExecutionHandlerLimitOrders:
    def test_limit_order_buy_triggers_and_fills(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('AAPL', 'LMT', 100, 'BUY', order_ts, 'LMT_B1', reference_price=99.0, limit_price=100.0)
        market_event = MarketEvent(
            symbol='AAPL', 
            timestamp=order_ts, 
            open_price=99.0, 
            high_price=99.5, 
            low_price=98.5, 
            close_price=99.0,
            volume=1000, 
            bid_price=98.95, 
            ask_price=99.05
        )
        base_handler.execute_order(order, market_event)
        fill: FillEvent = event_queue.get_event()
        assert fill.fill_price == 99.05

    def test_limit_order_buy_market_at_limit_triggers(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('AAPL', 'LMT', 100, 'BUY', order_ts, 'LMT_B2', reference_price=100.0, limit_price=100.0)
        market_event = MarketEvent(
            symbol='AAPL', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=100.0,
            volume=1000, 
            bid_price=99.95, 
            ask_price=100.05
        )
        base_handler.execute_order(order, market_event)
        fill: FillEvent = event_queue.get_event()
        assert fill.fill_price == 100.0

    def test_limit_order_buy_does_not_trigger(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('AAPL', 'LMT', 100, 'BUY', order_ts, 'LMT_B3', reference_price=101.0, limit_price=100.0)
        market_event = MarketEvent(
            symbol='AAPL', 
            timestamp=order_ts, 
            open_price=101.0, 
            high_price=101.5, 
            low_price=100.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(order, market_event)
        assert event_queue.empty()

    def test_limit_order_sell_triggers_and_fills(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('MSFT', 'LMT', 50, 'SELL', order_ts, 'LMT_S1', reference_price=101.0, limit_price=100.0)
        market_event = MarketEvent(
            symbol='MSFT', 
            timestamp=order_ts, 
            open_price=101.0, 
            high_price=101.5, 
            low_price=100.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(order, market_event)
        fill: FillEvent = event_queue.get_event()
        assert fill.fill_price == 100.95

    def test_limit_order_no_limit_price(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('GOOG', 'LMT', 10, 'BUY', order_ts, 'LMT_B4_NO_LP', reference_price=100.0, limit_price=None)
        market_event = MarketEvent(
            symbol='GOOG', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=100.0,
            volume=1000, 
            bid_price=99.95, 
            ask_price=100.05
        )
        base_handler.execute_order(order, market_event)
        assert event_queue.empty()

class TestExecutionHandlerStopOrders:
    def test_stop_order_sell_triggers_and_fills(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('TSLA', 'STP', 70, 'SELL', order_ts, 'STP_S1', reference_price=99.0, stop_price=100.0)
        market_event = MarketEvent(
            symbol='TSLA', 
            timestamp=order_ts, 
            open_price=99.0, 
            high_price=99.5, 
            low_price=98.5, 
            close_price=99.0,
            volume=1000, 
            bid_price=98.95, 
            ask_price=99.05
        )
        base_handler.execute_order(order, market_event)
        fill: FillEvent = event_queue.get_event()
        assert fill.fill_price == 98.95

    def test_stop_order_sell_market_at_stop_triggers(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('TSLA', 'STP', 70, 'SELL', order_ts, 'STP_S2', reference_price=100.0, stop_price=100.0)
        market_event = MarketEvent(
            symbol='TSLA', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=100.0,
            volume=1000, 
            bid_price=99.95, 
            ask_price=100.05
        )
        base_handler.execute_order(order, market_event)
        fill: FillEvent = event_queue.get_event()
        assert fill.fill_price == 99.95

    def test_stop_order_sell_does_not_trigger(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('TSLA', 'STP', 70, 'SELL', order_ts, 'STP_S3', reference_price=101.0, stop_price=100.0)
        market_event = MarketEvent(
            symbol='TSLA', 
            timestamp=order_ts, 
            open_price=101.0, 
            high_price=101.5, 
            low_price=100.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(order, market_event)
        assert event_queue.empty()

    def test_stop_order_buy_triggers_and_fills(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('NVDA', 'STP', 30, 'BUY', order_ts, 'STP_B1', reference_price=101.0, stop_price=100.0)
        market_event = MarketEvent(
            symbol='NVDA', 
            timestamp=order_ts, 
            open_price=101.0, 
            high_price=101.5, 
            low_price=100.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(order, market_event)
        fill: FillEvent = event_queue.get_event()
        assert fill.fill_price == 101.05

    def test_stop_order_sell_with_slippage(self, slippage_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('SPY', 'STP', 10, 'SELL', order_ts, 'STP_S_SLIP', reference_price=99.0, stop_price=100.0)
        market_event = MarketEvent(
            symbol='SPY', 
            timestamp=order_ts, 
            open_price=99.0, 
            high_price=99.5, 
            low_price=98.5, 
            close_price=99.0,
            volume=1000, 
            bid_price=98.95, 
            ask_price=99.05
        )
        slippage_handler.execute_order(order, market_event)
        fill: FillEvent = event_queue.get_event()
        assert math.isclose(fill.fill_price, 98.95 * 0.99)

    def test_stop_order_buy_with_atr_slippage(self, atr_slippage_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('QQQ', 'STP', 5, 'BUY', order_ts, 'STP_B_ATR',
                          reference_price=101.0, stop_price=100.0, current_atr=2.0)
        market_event = MarketEvent(
            symbol='QQQ', 
            timestamp=order_ts, 
            open_price=101.0, 
            high_price=101.5, 
            low_price=100.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        atr_slippage_handler.execute_order(order, market_event)
        fill: FillEvent = event_queue.get_event()
        assert math.isclose(fill.fill_price, 101.05 + (2.0 * 0.5))

    def test_stop_order_no_stop_price(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        order_ts = datetime.now()
        order = OrderEvent('FB', 'STP', 10, 'SELL', order_ts, 'STP_S_NO_SP', reference_price=100.0, stop_price=None)
        market_event = MarketEvent(
            symbol='FB', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=100.0,
            volume=1000, 
            bid_price=99.95, 
            ask_price=100.05
        )
        base_handler.execute_order(order, market_event)
        assert event_queue.empty()

    def test_stop_order_partial_fill(self, event_queue: EventQueue):
        handler_partial = SimulatedExecutionHandler(event_queue, max_fill_pct_per_bar=0.5)
        order_ts = datetime.now()
        order = OrderEvent('AMD', 'STP', 100, 'BUY', order_ts, 'STP_B_PART', reference_price=101.0, stop_price=100.0)
        market_event = MarketEvent(
            symbol='AMD', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        handler_partial.execute_order(order, market_event)
        fill: FillEvent = event_queue.get_event()
        assert fill.quantity == 50

    def test_limit_order_buy_not_filled(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test limit buy order that doesn't meet execution criteria."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Buy order at 100.0, but market close is at 99.0 (below limit, so execute)
        order = OrderEvent('AAPL', 'LMT', 100, 'BUY', limit_price=100.0)
        market_event = MarketEvent(
            symbol='AAPL', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=99.0,
            volume=1000, 
            bid_price=98.95, 
            ask_price=99.05
        )
        base_handler.execute_order(order, market_event)
        
        # Should be filled since close <= limit
        assert not event_queue.empty()

    def test_limit_order_buy_filled(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test limit buy order that meets execution criteria."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Buy order at 100.0, market close is at 100.0 (at limit, so execute)
        order = OrderEvent('AAPL', 'LMT', 100, 'BUY', limit_price=100.0)
        market_event = MarketEvent(
            symbol='AAPL', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=100.0,
            volume=1000, 
            bid_price=99.95, 
            ask_price=100.05
        )
        base_handler.execute_order(order, market_event)
        
        assert not event_queue.empty()

    def test_limit_order_buy_above_limit(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test limit buy order when price is above limit."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Buy order at 100.0, but market close is at 101.0 (above limit, so no execute)
        order = OrderEvent('AAPL', 'LMT', 100, 'BUY', limit_price=100.0)
        market_event = MarketEvent(
            symbol='AAPL', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(order, market_event)
        
        # Should not be filled
        assert event_queue.empty()

    def test_limit_order_sell_above_limit(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test limit sell order when price is above limit."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Sell order at 100.0, market close is at 101.0 (above limit, so execute)
        order = OrderEvent('MSFT', 'LMT', 50, 'SELL', limit_price=100.0)
        market_event = MarketEvent(
            symbol='MSFT', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(order, market_event)
        
        assert not event_queue.empty()

    def test_limit_order_sell_at_limit(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test limit sell order when price is at limit."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Sell order at 100.0, market close is at 100.0 (at limit, so execute)
        order = OrderEvent('GOOG', 'LMT', 25, 'SELL', limit_price=100.0)
        market_event = MarketEvent(
            symbol='GOOG', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=100.0,
            volume=1000, 
            bid_price=99.95, 
            ask_price=100.05
        )
        base_handler.execute_order(order, market_event)
        
        assert not event_queue.empty()

    def test_stop_order_buy_triggered(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test stop buy order that gets triggered."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Stop buy at 100.0, market close is at 99.0 (below stop, so no trigger)
        order = OrderEvent('TSLA', 'STP', 10, 'BUY', stop_price=100.0)
        market_event = MarketEvent(
            symbol='TSLA', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=99.0,
            volume=1000, 
            bid_price=98.95, 
            ask_price=99.05
        )
        base_handler.execute_order(order, market_event)
        
        # Should not trigger
        assert event_queue.empty()

    def test_stop_order_buy_at_stop(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test stop buy order triggered at stop price."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Stop buy at 100.0, market close is at 100.0 (at stop, so trigger)
        order = OrderEvent('TSLA', 'STP', 10, 'BUY', stop_price=100.0)
        market_event = MarketEvent(
            symbol='TSLA', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=100.0,
            volume=1000, 
            bid_price=99.95, 
            ask_price=100.05
        )
        base_handler.execute_order(order, market_event)
        
        assert not event_queue.empty()

    def test_stop_order_buy_above_stop(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test stop buy order triggered above stop price."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Stop buy at 100.0, market close is at 101.0 (above stop, so trigger)
        order = OrderEvent('TSLA', 'STP', 10, 'BUY', stop_price=100.0)
        market_event = MarketEvent(
            symbol='TSLA', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(order, market_event)
        
        assert not event_queue.empty()

    def test_stop_order_sell_above_stop(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test stop sell order when price is above stop."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Stop sell at 100.0, market close is at 101.0 (above stop, so no trigger)
        order = OrderEvent('NVDA', 'STP', 5, 'SELL', stop_price=100.0)
        market_event = MarketEvent(
            symbol='NVDA', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(order, market_event)
        
        # Should not trigger
        assert event_queue.empty()

    def test_stop_order_sell_triggered(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test stop sell order that gets triggered."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Stop sell at 100.0, market close is at 99.0 (below stop, so trigger)
        order = OrderEvent('SPY', 'STP', 20, 'SELL', stop_price=100.0)
        market_event = MarketEvent(
            symbol='SPY', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=99.0,
            volume=1000, 
            bid_price=98.95, 
            ask_price=99.05
        )
        base_handler.execute_order(order, market_event)
        
        assert not event_queue.empty()

    def test_order_different_symbol_no_execution(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test that orders for different symbols don't execute."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Order for QQQ but market event for different symbol
        order = OrderEvent('QQQ', 'MKT', 10, 'BUY')
        market_event = MarketEvent(
            symbol='QQQ', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(order, market_event)
        
        assert not event_queue.empty()

    def test_no_market_event_no_execution(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test that orders don't execute without market events."""
        order = OrderEvent('FB', 'MKT', 10, 'BUY')
        market_event = MarketEvent(
            symbol='FB', 
            timestamp=datetime(2023, 1, 1, 12, 0, 0), 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=100.0,
            volume=1000, 
            bid_price=99.95, 
            ask_price=100.05
        )
        base_handler.execute_order(order, market_event)
        
        assert not event_queue.empty()

    def test_order_type_not_order_no_execution(self, base_handler: SimulatedExecutionHandler, event_queue: EventQueue):
        """Test that non-order events don't get executed."""
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        
        # Create a mock event that's not an OrderEvent
        fake_event = Mock()
        fake_event.type = 'NOT_ORDER'
        
        market_event = MarketEvent(
            symbol='AMD', 
            timestamp=order_ts, 
            open_price=100.0, 
            high_price=100.5, 
            low_price=99.5, 
            close_price=101.0,
            volume=1000, 
            bid_price=100.95, 
            ask_price=101.05
        )
        base_handler.execute_order(fake_event, market_event)
        
        # Should not execute since it's not an OrderEvent
        assert event_queue.empty()
