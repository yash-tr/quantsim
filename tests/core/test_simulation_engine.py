"""
Unit tests for the SimulationEngine.
"""
import pytest
from unittest.mock import Mock, MagicMock
from quantsim.core.simulation_engine import SimulationEngine
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from quantsim.data.base import DataHandler

@pytest.fixture
def mock_data_handler():
    """Fixture for a mock DataHandler."""
    data_handler = MagicMock(spec=DataHandler)
    market_events = [
        (Mock(), 'AAPL', {'Open': 150.0, 'High': 152.0, 'Low': 149.0, 'Close': 151.0, 'Volume': 100000}),
        (Mock(), 'AAPL', {'Open': 151.0, 'High': 153.0, 'Low': 150.0, 'Close': 152.0, 'Volume': 120000}),
    ]
    
    # Create a proper iterator
    data_handler.__iter__.return_value = iter(market_events)
    data_handler.continue_backtest = True
    
    # Track iteration state
    data_handler._current_iter = iter(market_events)
    
    def _next_side_effect():
        try:
            return next(data_handler._current_iter)
        except StopIteration:
            data_handler.continue_backtest = False
            raise
    
    data_handler.__next__.side_effect = _next_side_effect
    return data_handler

@pytest.fixture
def mock_strategy():
    """Fixture for a mock Strategy."""
    strategy = Mock()
    return strategy

@pytest.fixture
def mock_portfolio():
    """Fixture for a mock Portfolio."""
    portfolio = Mock()
    return portfolio

@pytest.fixture
def mock_execution_handler():
    """Fixture for a mock ExecutionHandler."""
    execution = Mock()
    return execution

@pytest.fixture
def event_queue():
    """Fixture for an EventQueue."""
    return EventQueue()

class TestSimulationEngine:
    """Tests for the SimulationEngine."""

    def test_simulation_engine_run(self, event_queue, mock_data_handler, mock_strategy, mock_portfolio, mock_execution_handler):
        """Test the main run loop of the simulation engine."""
        engine = SimulationEngine(
            event_queue,
            data_handler=mock_data_handler,
            strategy=mock_strategy,
            portfolio=mock_portfolio,
            execution_handler=mock_execution_handler
        )
        engine.run()

        # Check that the main components were called for each market event
        assert mock_strategy.on_market_data.call_count == 2
        assert mock_portfolio.on_market_data.call_count == 2

    def test_event_handling(self, event_queue, mock_strategy, mock_portfolio, mock_execution_handler):
        """Test the handling of different event types."""
        engine = SimulationEngine(
            event_queue,
            data_handler=Mock(spec=DataHandler),
            strategy=mock_strategy,
            portfolio=mock_portfolio,
            execution_handler=mock_execution_handler
        )

        signal_event = SignalEvent(symbol='AAPL', direction='LONG', strength=1.0, timestamp=Mock())
        order_event = OrderEvent(symbol='AAPL', order_type='MKT', quantity=100, direction='BUY', timestamp=Mock())
        fill_event = FillEvent(symbol='AAPL', quantity=100, direction='BUY', fill_price=150.0, commission=1.0, timestamp=Mock())

        engine._handle_signal_event(signal_event)
        engine._handle_order_event(order_event)
        engine._handle_fill_event(fill_event)
        
        mock_portfolio.on_signal.assert_called_once_with(signal_event)
        mock_execution_handler.execute_order.assert_called_once_with(order_event, None)
        mock_portfolio.on_fill.assert_called_once_with(fill_event)
        mock_strategy.on_fill.assert_called_once_with(fill_event)

    def test_simulation_engine_stops_on_empty_queue(self, event_queue, mock_strategy, mock_portfolio, mock_execution_handler):
        empty_data_handler = Mock(spec=DataHandler)
        empty_data_handler.continue_backtest = False
        engine = SimulationEngine(
            event_queue,
            data_handler=empty_data_handler,
            strategy=mock_strategy,
            portfolio=mock_portfolio,
            execution_handler=mock_execution_handler
        )
        engine.run()
        assert event_queue.empty()

    def test_simulation_engine_handles_stop_iteration(self, event_queue, mock_data_handler, mock_strategy, mock_portfolio, mock_execution_handler):
        """Test that the engine handles StopIteration from the data handler."""
        engine = SimulationEngine(
            event_queue,
            data_handler=mock_data_handler,
            strategy=mock_strategy,
            portfolio=mock_portfolio,
            execution_handler=mock_execution_handler
        )
        engine.run()
        # The test passes if run() completes without error
        assert not mock_data_handler.continue_backtest

    def test_simulation_with_no_data(self, event_queue, mock_strategy, mock_portfolio, mock_execution_handler):
        """Test that the engine runs with no data."""
        no_data_handler = MagicMock(spec=DataHandler)
        no_data_handler.continue_backtest = False
        no_data_handler.__iter__.return_value = iter([])
        
        def _next_side_effect():
            raise StopIteration
        
        no_data_handler.__next__.side_effect = _next_side_effect
        
        engine = SimulationEngine(
            event_queue, 
            data_handler=no_data_handler, 
            strategy=mock_strategy, 
            portfolio=mock_portfolio, 
            execution_handler=mock_execution_handler
        )
        engine.run()
        assert event_queue.empty() 