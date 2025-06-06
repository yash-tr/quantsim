"""
Unit tests for the event classes in quantsim.core.events.
"""
import pytest
from datetime import datetime, timedelta
from quantsim.core.events import Event, MarketEvent, SignalEvent, OrderEvent, FillEvent

TIME_EPSILON_MS = 100

@pytest.mark.core
class TestEventSystem:
    """Tests for the core event classes."""

    def test_event_creation(self):
        """Test basic Event creation."""
        event = Event()
        assert event.type == 'EVENT'
        assert isinstance(event.timestamp, datetime)
        assert (datetime.utcnow() - event.timestamp) < timedelta(milliseconds=TIME_EPSILON_MS)

        custom_ts = datetime(2023, 1, 1, 12, 0, 0)
        event_with_ts = Event(timestamp=custom_ts)
        assert event_with_ts.timestamp == custom_ts

    def test_market_event_creation(self):
        """Test MarketEvent creation and attribute assignment."""
        ts = datetime(2023, 10, 1, 10, 0, 0)
        event = MarketEvent(
            symbol='AAPL', timestamp=ts, open_price=150.0, high_price=152.5,
            low_price=149.8, close_price=151.75, volume=100000
        )
        assert event.type == 'MARKET'
        assert event.symbol == 'AAPL'
        assert event.timestamp == ts
        assert event.open == 150.0
        assert event.high == 152.5
        assert event.low == 149.8
        assert event.close == 151.75
        assert event.volume == 100000
        rep = repr(event)
        assert "MarketEvent" in rep
        assert "AAPL" in rep
        assert "151.75" in rep

    def test_signal_event_creation(self):
        """Test SignalEvent creation and attribute assignment."""
        ts = datetime(2023, 10, 1, 10, 5, 0)
        event = SignalEvent(
            symbol='MSFT', direction='LONG', strength=0.85, timestamp=ts,
            strategy_id='SMA_Cross_MSFT_10_20'
        )
        assert event.type == 'SIGNAL'
        assert event.symbol == 'MSFT'
        assert event.timestamp == ts
        assert event.direction == 'LONG'
        assert event.strength == 0.85
        assert event.strategy_id == 'SMA_Cross_MSFT_10_20'
        rep = repr(event)
        assert "SignalEvent" in rep
        assert "MSFT" in rep
        assert "LONG" in rep

    def test_order_event_creation(self):
        """Test OrderEvent creation and attribute assignment."""
        ts = datetime(2023, 10, 1, 10, 10, 0)
        event = OrderEvent(
            symbol='GOOG', order_type='LMT', quantity=50, direction='SELL',
            timestamp=ts, order_id='ORDER123', reference_price=2500.50,
            limit_price=2505.00, stop_price=None, current_atr=15.5
        )
        assert event.type == 'ORDER'
        assert event.symbol == 'GOOG'
        assert event.timestamp == ts
        assert event.order_type == 'LMT'
        assert event.quantity == 50
        assert event.direction == 'SELL'
        assert event.order_id == 'ORDER123'
        assert event.reference_price == 2500.50
        assert event.limit_price == 2505.00
        assert event.stop_price is None
        assert event.current_atr == 15.5
        rep = repr(event)
        assert "OrderEvent" in rep
        assert "GOOG" in rep
        assert "LMT" in rep
        assert "ORDER123" in rep

        with pytest.raises(ValueError, match="Order quantity must be positive."):
            OrderEvent(symbol='XYZ', order_type='MKT', quantity=0, direction='BUY', timestamp=ts)
        with pytest.raises(ValueError, match="Order quantity must be positive."):
            OrderEvent(symbol='XYZ', order_type='MKT', quantity=-10, direction='BUY', timestamp=ts)

    def test_fill_event_creation(self):
        """Test FillEvent creation and attribute assignment."""
        ts = datetime(2023, 10, 1, 10, 15, 0)
        event = FillEvent(
            symbol='TSLA', quantity=20, direction='BUY', fill_price=250.75,
            commission=5.00, exchange='SIM_EXCH', timestamp=ts, order_id='ORDERXYZ'
        )
        assert event.type == 'FILL'
        assert event.symbol == 'TSLA'
        assert event.timestamp == ts
        assert event.order_id == 'ORDERXYZ'
        assert event.quantity == 20
        assert event.direction == 'BUY'
        assert event.fill_price == 250.75
        assert event.commission == 5.00
        assert event.exchange == 'SIM_EXCH'
        rep = repr(event)
        assert "FillEvent" in rep
        assert "TSLA" in rep
        assert "250.75" in rep

        with pytest.raises(ValueError, match="Fill quantity must be positive."):
            FillEvent(symbol='ABC', quantity=0, direction='SELL', fill_price=100.0, timestamp=ts)
        with pytest.raises(ValueError, match="Fill quantity must be positive."):
            FillEvent(symbol='ABC', quantity=-5, direction='SELL', fill_price=100.0, timestamp=ts)

