"""
Unit tests for core portfolio calculations (Position and Portfolio basics).
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from quantsim.portfolio.portfolio import Position, Portfolio
from quantsim.core.events import FillEvent, MarketEvent
from quantsim.core.event_queue import EventQueue

pytestmark = pytest.mark.portfolio

class TestPosition:
    """Tests for the Position class."""

    def test_position_creation_long(self):
        pos = Position(symbol='AAPL', quantity=100, average_price=150.0)
        assert pos.symbol == 'AAPL'
        assert pos.quantity == 100
        assert pos.average_price == 150.0
        assert pos.last_price == 150.0
        assert pos.cost_basis == 100 * 150.0
        assert pos.market_value == 100 * 150.0
        assert pos.unrealized_pnl == 0.0

    def test_position_creation_short(self):
        pos = Position(symbol='MSFT', quantity=-50, average_price=200.0)
        assert pos.quantity == -50
        assert pos.average_price == 200.0
        assert pos.cost_basis == -50 * 200.0
        assert pos.market_value == -50 * 200.0
        assert pos.unrealized_pnl == 0.0

    def test_position_update_last_price(self):
        pos = Position('AAPL', 100, 150.0)
        ts = datetime.now()
        pos.update_last_price(155.0, timestamp=ts)
        assert pos.last_price == 155.0
        assert pos.market_value == 100 * 155.0
        assert pos.unrealized_pnl == (100 * 155.0) - (100 * 150.0)
        assert pos.last_update_time == ts

    def test_position_pnl_long(self):
        pos = Position('SPY', 10, 300.0)
        pos.update_last_price(310.0)
        assert pos.unrealized_pnl == (310.0 - 300.0) * 10

        pos.update_last_price(290.0)
        assert pos.unrealized_pnl == (290.0 - 300.0) * 10

    def test_position_pnl_short(self):
        pos = Position('TSLA', -5, 250.0)
        pos.update_last_price(240.0)
        assert pos.unrealized_pnl == (pos.quantity * 240.0) - (pos.quantity * 250.0)
        assert pos.unrealized_pnl == 50.0

        pos.update_last_price(260.0)
        assert pos.unrealized_pnl == (pos.quantity * 260.0) - (pos.quantity * 250.0)
        assert pos.unrealized_pnl == -50.0

@pytest.fixture
def empty_portfolio() -> Portfolio:
    """Returns an empty portfolio with 100k initial cash."""
    return Portfolio(initial_cash=100000.0, event_queue=EventQueue())

class TestPortfolioCore:
    """Tests for core Portfolio operations."""

    def test_portfolio_creation(self, empty_portfolio: Portfolio):
        pf = empty_portfolio
        assert pf.initial_cash == 100000.0
        assert pf.cash == 100000.0
        assert pf.current_holdings_value == 0.0
        assert pf.current_total_value == 100000.0
        assert not pf.holdings

    def test_portfolio_on_fill_buy_new_position(self, empty_portfolio: Portfolio):
        pf = empty_portfolio
        fill_ts = datetime.now()
        buy_event = FillEvent(
            symbol='AAPL', quantity=100, direction='BUY', fill_price=150.0,
            commission=10.0, timestamp=fill_ts, order_id='B1'
        )
        pf.on_fill(buy_event)
        expected_cash = 100000.0 - (100 * 150.0) - 10.0
        assert pf.cash == expected_cash
        # ... (rest of assertions)

    def test_portfolio_on_fill_sell_short_new_position(self, empty_portfolio: Portfolio):
        pf = empty_portfolio
        fill_ts = datetime.now()
        sell_event = FillEvent(
            symbol='MSFT', quantity=50, direction='SELL', fill_price=200.0,
            commission=5.0, timestamp=fill_ts, order_id='S1'
        )
        pf.on_fill(sell_event)
        expected_cash = 100000.0 + (50 * 200.0) - 5.0
        assert pf.cash == expected_cash
        # ... (rest of assertions)

    def test_portfolio_on_fill_add_to_existing_long(self, empty_portfolio: Portfolio):
        pf = empty_portfolio
        pf.on_fill(FillEvent('SPY', 10, 'BUY', 300.0, 1.0, timestamp=datetime.now()))
        pf.on_fill(FillEvent('SPY', 15, 'BUY', 310.0, 1.5, timestamp=datetime.now()))
        spy_pos = pf.holdings['SPY']
        assert spy_pos.quantity == 25
        expected_avg_price = ((10 * 300.0) + (15 * 310.0)) / 25.0
        assert spy_pos.average_price == pytest.approx(expected_avg_price)
        # ... (cash assertion)

    def test_portfolio_on_fill_partially_close_long(self, empty_portfolio: Portfolio):
        pf = empty_portfolio
        pf.on_fill(FillEvent('AAPL', 100, 'BUY', 150.0, 10.0, timestamp=datetime.now()))
        pf.on_fill(FillEvent('AAPL', 30, 'SELL', 155.0, 3.0, timestamp=datetime.now()))
        aapl_pos = pf.holdings['AAPL']
        assert aapl_pos.quantity == 70
        assert aapl_pos.average_price == 150.0
        # ... (cash assertion)

    def test_portfolio_on_fill_fully_close_long(self, empty_portfolio: Portfolio):
        pf = empty_portfolio
        pf.on_fill(FillEvent('AAPL', 100, 'BUY', 150.0, 10.0, timestamp=datetime.now()))
        pf.on_fill(FillEvent('AAPL', 100, 'SELL', 160.0, 10.0, timestamp=datetime.now()))
        assert 'AAPL' not in pf.holdings
        # ... (cash assertion)

    def test_portfolio_on_fill_flip_long_to_short(self, empty_portfolio: Portfolio):
        pf = empty_portfolio
        pf.on_fill(FillEvent('TSLA', 50, 'BUY', 200.0, 5.0, timestamp=datetime.now()))
        pf.on_fill(FillEvent('TSLA', 80, 'SELL', 210.0, 8.0, timestamp=datetime.now()))
        tsla_pos = pf.holdings['TSLA']
        assert tsla_pos.quantity == -30
        assert tsla_pos.average_price == 210.0
        # ... (cash assertion)

    def test_portfolio_on_market_data_updates_holding_value(self, empty_portfolio: Portfolio):
        pf = empty_portfolio
        pf.on_fill(FillEvent('GOOG', 10, 'BUY', 1000.0, 10.0, timestamp=datetime(2023,1,1)))
        initial_total_value = pf.current_total_value
        market_ev = MarketEvent(
            symbol='GOOG', 
            timestamp=datetime(2023,1,2),
            open_price=1005, 
            high_price=1010, 
            low_price=1000,
            close_price=1008.0, 
            volume=100
        )
        pf.on_market_data(market_ev)
        assert pf.holdings['GOOG'].last_price == 1008.0
        # ... (other assertions)

        # Update market data to set position value
        market_event = MarketEvent(
            symbol='AAPL',
            timestamp=pd.Timestamp('2023-01-02'),
            open_price=151.0, 
            high_price=152.0, 
            low_price=150.0, 
            close_price=151.0, 
            volume=100000
        )
        empty_portfolio.on_market_data(market_event)
        
        assert empty_portfolio.current_holdings_value > 0

    def test_portfolio_initial_total_value(self, empty_portfolio: Portfolio):
        assert empty_portfolio.current_total_value == 100000.0

    def test_portfolio_calculate_performance_metrics_no_trades(self, empty_portfolio):
        """Test performance metrics calculation with no trades."""
        empty_portfolio.calculate_performance_metrics()
        # Should handle gracefully and not crash

    def test_portfolio_export_trade_log(self, empty_portfolio, tmp_path):
        """Test export trade log functionality."""
        filepath = tmp_path / "test_trades.csv"
        empty_portfolio.export_trade_log(str(filepath))
        
        # With no trades, no file is created (as shown by the output "No trades to export")
        # This is expected behavior, so we just check it doesn't crash
        # Let's add a trade first to test actual export
        
        # Add a trade
        fill_event = FillEvent(
            symbol='AAPL',
            quantity=100,
            direction='BUY',
            fill_price=150.0,
            commission=1.0,
            timestamp=pd.Timestamp('2023-01-01'),
            order_id='test_order_1'
        )
        empty_portfolio.on_fill(fill_event)
        
        # Close the position to create a completed trade
        close_fill = FillEvent(
            symbol='AAPL',
            quantity=100,
            direction='SELL',
            fill_price=155.0,
            commission=1.0,
            timestamp=pd.Timestamp('2023-01-02'),
            order_id='test_order_2'
        )
        empty_portfolio.on_fill(close_fill)
        
        # Now export should create a file
        empty_portfolio.export_trade_log(str(filepath))
        assert filepath.exists()

    def test_portfolio_print_final_summary_with_positions(self, empty_portfolio):
        """Test final summary printing with open positions."""
        # Add a position by simulating a fill
        fill_event = FillEvent(
            symbol='AAPL',
            quantity=100,
            direction='BUY',
            fill_price=150.0,
            commission=1.0,
            timestamp=pd.Timestamp('2023-01-01'),
            order_id='test_order_1'
        )
        empty_portfolio.on_fill(fill_event)
        
        # This should not crash
        empty_portfolio.print_final_summary()

    def test_portfolio_get_metrics_with_custom_risk_free_rate(self, empty_portfolio):
        """Test getting metrics with custom risk-free rate."""
        metrics = empty_portfolio.get_metrics(risk_free_rate=0.05)
        assert isinstance(metrics, dict)
        assert 'total_return_pct' in metrics

    def test_portfolio_properties(self, empty_portfolio):
        """Test portfolio properties."""
        # Test holdings alias
        assert empty_portfolio.holdings == empty_portfolio.positions
        
        # Test current_holdings_value property
        assert empty_portfolio.current_holdings_value == 0.0
        
        # Add a position and test again
        fill_event = FillEvent(
            symbol='AAPL',
            quantity=100,
            direction='BUY',
            fill_price=150.0,
            commission=1.0,
            timestamp=pd.Timestamp('2023-01-01'),
            order_id='test_order_1'
        )
        empty_portfolio.on_fill(fill_event)
        
        # Update market data to set position value
        market_event = MarketEvent(
            symbol='AAPL',
            timestamp=pd.Timestamp('2023-01-02'),
            open_price=151.0, 
            high_price=152.0, 
            low_price=150.0, 
            close_price=151.0, 
            volume=100000
        )
        empty_portfolio.on_market_data(market_event)
        
        assert empty_portfolio.current_holdings_value > 0

    def test_portfolio_metrics_edge_cases(self, empty_portfolio):
        """Test portfolio metrics with edge cases."""
        # Test with single equity point
        empty_portfolio.equity_curve = [(pd.Timestamp('2023-01-01'), 100000)]
        empty_portfolio.current_total_value = 100000
        
        metrics = empty_portfolio.get_metrics()
        assert 'total_return_pct' in metrics
        
        # Test with no equity data
        empty_portfolio.equity_curve = []
        metrics = empty_portfolio.get_metrics()
        assert 'error' in metrics

