"""
Unit tests for SimulatedExecutionHandler focusing on Market Orders.
"""
import pytest
from datetime import datetime, timedelta
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import OrderEvent, FillEvent, MarketEvent
from quantsim.execution.execution_handler import SimulatedExecutionHandler, FixedCommission
from quantsim.execution.slippage import PercentageSlippage

pytestmark = pytest.mark.execution

@pytest.fixture
def basic_event_queue() -> EventQueue:
    return EventQueue()

@pytest.fixture
def basic_handler(basic_event_queue: EventQueue) -> SimulatedExecutionHandler:
    return SimulatedExecutionHandler(
        event_queue=basic_event_queue,
        slippage_model=PercentageSlippage(0.0),
        commission_model=FixedCommission(0.0),
        latency_ms=0,
        max_fill_pct_per_bar=1.0,
        max_fill_qty_per_bar=float('inf')
    )

@pytest.fixture
def slippage_handler(basic_event_queue: EventQueue) -> SimulatedExecutionHandler:
    return SimulatedExecutionHandler(
        event_queue=basic_event_queue,
        slippage_model=PercentageSlippage(0.01),
        commission_model=FixedCommission(0.0),
        latency_ms=0
    )

@pytest.fixture
def commission_handler(basic_event_queue: EventQueue) -> SimulatedExecutionHandler:
    return SimulatedExecutionHandler(
        event_queue=basic_event_queue,
        slippage_model=PercentageSlippage(0.0),
        commission_model=FixedCommission(1.50),
        latency_ms=0
    )

class TestExecutionHandlerMarketOrders:
    def test_market_order_buy_execution_simple(self, basic_handler: SimulatedExecutionHandler, basic_event_queue: EventQueue):
        ts = datetime(2023, 1, 1, 10, 0, 0)
        order = OrderEvent('AAPL', 'MKT', 100, 'BUY')
        market_event = MarketEvent(
            symbol='AAPL', 
            timestamp=ts, 
            open_price=150.0, 
            high_price=150.5, 
            low_price=149.5, 
            close_price=150.00, 
            volume=1000, 
            bid_price=149.95, 
            ask_price=150.05
        )
        basic_handler.execute_order(order, market_event)
        fill_event: FillEvent = basic_event_queue.get_event()
        assert fill_event.fill_price == 150.05
        assert fill_event.commission == 0.0
        assert fill_event.timestamp == ts + timedelta(microseconds=1)

    def test_market_order_sell_execution_simple(self, basic_handler: SimulatedExecutionHandler, basic_event_queue: EventQueue):
        ts = datetime(2023, 1, 1, 10, 5, 0)
        order = OrderEvent('MSFT', 'MKT', 50, 'SELL')
        market_event = MarketEvent(
            symbol='MSFT', 
            timestamp=ts, 
            open_price=250.0, 
            high_price=250.5, 
            low_price=249.5, 
            close_price=250.00, 
            volume=1000, 
            bid_price=249.95, 
            ask_price=250.05
        )
        basic_handler.execute_order(order, market_event)
        fill_event: FillEvent = basic_event_queue.get_event()
        assert fill_event.fill_price == 249.95
        assert fill_event.commission == 0.0
        assert fill_event.timestamp == ts + timedelta(microseconds=1)

    def test_market_order_buy_with_slippage(self, slippage_handler: SimulatedExecutionHandler, basic_event_queue: EventQueue):
        ts = datetime.now()
        order = OrderEvent('SPY', 'MKT', 10, 'BUY')
        market_event = MarketEvent(
            symbol='SPY', 
            timestamp=ts, 
            open_price=300.0, 
            high_price=300.5, 
            low_price=299.5, 
            close_price=300.00, 
            volume=1000, 
            bid_price=299.90, 
            ask_price=300.10
        )
        slippage_handler.execute_order(order, market_event)
        fill_event: FillEvent = basic_event_queue.get_event()
        assert fill_event.fill_price == pytest.approx(300.10 * 1.01)

    def test_market_order_sell_with_slippage(self, slippage_handler: SimulatedExecutionHandler, basic_event_queue: EventQueue):
        ts = datetime.now()
        order = OrderEvent('QQQ', 'MKT', 20, 'SELL')
        market_event = MarketEvent(
            symbol='QQQ', 
            timestamp=ts, 
            open_price=350.0, 
            high_price=350.5, 
            low_price=349.5, 
            close_price=350.00, 
            volume=1000, 
            bid_price=349.90, 
            ask_price=350.10
        )
        slippage_handler.execute_order(order, market_event)
        fill_event: FillEvent = basic_event_queue.get_event()
        assert fill_event.fill_price == pytest.approx(349.90 * 0.99)

    def test_market_order_with_commission(self, commission_handler: SimulatedExecutionHandler, basic_event_queue: EventQueue):
        ts = datetime.now()
        order = OrderEvent('IWM', 'MKT', 5, 'BUY')
        market_event = MarketEvent(
            symbol='IWM', 
            timestamp=ts, 
            open_price=180.0, 
            high_price=180.5, 
            low_price=179.5, 
            close_price=180.00, 
            volume=1000, 
            bid_price=179.95, 
            ask_price=180.05
        )
        commission_handler.execute_order(order, market_event)
        fill_event: FillEvent = basic_event_queue.get_event()
        assert fill_event.commission == 1.50

    def test_market_order_latency(self, basic_event_queue: EventQueue):
        handler = SimulatedExecutionHandler(basic_event_queue, latency_ms=150)
        order_ts = datetime(2023, 1, 1, 12, 0, 0)
        order = OrderEvent('DIA', 'MKT', 2, 'SELL')
        market_event = MarketEvent(
            symbol='DIA', 
            timestamp=order_ts, 
            open_price=320.0, 
            high_price=320.5, 
            low_price=319.5, 
            close_price=320.00, 
            volume=1000, 
            bid_price=319.95, 
            ask_price=320.05
        )
        handler.execute_order(order, market_event)
        fill_event: FillEvent = basic_event_queue.get_event()
        assert fill_event.timestamp == order_ts + timedelta(milliseconds=150)

    def test_market_order_no_market_event(self, basic_handler: SimulatedExecutionHandler, basic_event_queue: EventQueue):
        order = OrderEvent('TEST', 'MKT', 100, 'BUY')
        basic_handler.execute_order(order, None)
        assert basic_event_queue.empty()

