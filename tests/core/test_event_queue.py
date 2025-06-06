#Unit tests for the EventQueue class in quantsim.core.event_queue.

import pytest
from queue import Empty
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import MarketEvent
from datetime import datetime, timedelta

@pytest.mark.core
class TestEventQueue:
    """Tests for the EventQueue."""

    def test_event_queue_initialization(self):
        """Test if the event queue initializes correctly and is empty."""
        eq = EventQueue()
        assert eq.empty(), "New event queue should be empty."

    def test_event_queue_put_and_get(self):
        """Test putting an event into the queue and getting it back."""
        eq = EventQueue()
        ts = datetime.now()
        market_event = MarketEvent(
            symbol="AAPL", 
            timestamp=ts, 
            open_price=150.0, 
            high_price=151.0, 
            low_price=149.0, 
            close_price=150.5, 
            volume=10000
        )

        eq.put_event(market_event)
        assert not eq.empty(), "Queue should not be empty after putting an event."

        retrieved_event = eq.get_event()
        assert retrieved_event is market_event, "Retrieved event should be the same as the one put in."
        assert eq.empty(), "Queue should be empty after getting the event."

    def test_event_queue_multiple_puts_and_gets(self):
        """Test FIFO behavior with multiple events."""
        eq = EventQueue()
        ts1 = datetime.now()
        ts2 = datetime.now() + timedelta(microseconds=10)

        event1 = MarketEvent(
            symbol="AAPL", 
            timestamp=ts1, 
            open_price=150.0, 
            high_price=151.0, 
            low_price=149.0, 
            close_price=150.5, 
            volume=10000
        )
        event2 = MarketEvent(
            symbol="MSFT", 
            timestamp=ts2, 
            open_price=250.0, 
            high_price=251.0, 
            low_price=249.0, 
            close_price=250.5, 
            volume=5000
        )

        eq.put_event(event1)
        eq.put_event(event2)
        assert not eq.empty()

        retrieved1 = eq.get_event()
        assert retrieved1 is event1, "First event retrieved should be event1."
        assert not eq.empty()

        retrieved2 = eq.get_event()
        assert retrieved2 is event2, "Second event retrieved should be event2."
        assert eq.empty()

    def test_event_queue_get_event_behavior_on_empty(self):
        """Tests behavior related to an empty queue."""
        eq = EventQueue()
        assert eq.empty()

        ts = datetime.now()
        event = MarketEvent(
            symbol="GOOG", 
            timestamp=ts, 
            open_price=2000.0, 
            high_price=2001.0, 
            low_price=1999.0, 
            close_price=2000.0, 
            volume=100
        )
        eq.put_event(event)
        assert not eq.empty()
        retrieved_event = eq.get_event()
        assert retrieved_event is event
        assert eq.empty()
