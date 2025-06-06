"""
Unit tests for quantsim.strategies.momentum.MomentumStrategy
"""
import pytest
import pandas as pd
from datetime import datetime
import math
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import MarketEvent, OrderEvent, FillEvent
from quantsim.strategies.momentum import MomentumStrategy

pytestmark = pytest.mark.strategies

@pytest.fixture
def basic_event_queue() -> EventQueue:
    return EventQueue()

@pytest.fixture
def momentum_sample_data_df() -> pd.DataFrame:
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                            '2023-01-05', '2023-01-06', '2023-01-07'])
    data = {'Open':  [10, 11, 12, 10, 11, 12, 10],
            'High':  [10, 11, 12, 12, 13, 12, 10],
            'Low':   [10, 11, 12, 10, 11, 10,  8],
            'Close': [10, 11, 12, 12, 13, 10,  9],
            'Volume':[100,100,100,150,150,150,150]}
    return pd.DataFrame(data, index=pd.Index(dates, name="Timestamp"))

@pytest.fixture
def momentum_strategy_defaults(basic_event_queue: EventQueue, momentum_sample_data_df: pd.DataFrame) -> MomentumStrategy:
    return MomentumStrategy(event_queue=basic_event_queue, symbols=['TEST'], momentum_window=3,
                            order_quantity=10, data_handler=momentum_sample_data_df, atr_period=0)

class TestMomentumStrategy:
    def test_momentum_creation(self, momentum_strategy_defaults: MomentumStrategy):
        strat = momentum_strategy_defaults
        assert strat.symbols == ['TEST']
        assert strat.momentum_window == 3

    def test_momentum_atr_precalculation(self, basic_event_queue: EventQueue, momentum_sample_data_df: pd.DataFrame):
        strat_with_atr = MomentumStrategy(basic_event_queue, ['TEST'], momentum_window=3, atr_period=3,
                                          data_handler=momentum_sample_data_df)
        assert strat_with_atr.atr_series['TEST'] is not None

    def test_momentum_buy_signal_and_order(self, momentum_strategy_defaults: MomentumStrategy, basic_event_queue: EventQueue, momentum_sample_data_df: pd.DataFrame):
        strat = momentum_strategy_defaults
        target_ts_buy = pd.Timestamp('2023-01-04')
        for _, row in momentum_sample_data_df.iterrows():
            ts = row.name
            market_event = MarketEvent(
                symbol=strat.symbols[0], 
                timestamp=ts, 
                open_price=row['Open'], 
                high_price=row['High'], 
                low_price=row['Low'], 
                close_price=row['Close'], 
                volume=int(row['Volume'])
            )
            strat.on_market_data(market_event)
            if ts == target_ts_buy: break
        order_event: OrderEvent = basic_event_queue.get_event()
        assert order_event.direction == 'BUY' and order_event.reference_price == 12

    def test_momentum_sell_signal_and_order(self, momentum_strategy_defaults: MomentumStrategy, basic_event_queue: EventQueue, momentum_sample_data_df: pd.DataFrame):
        strat = momentum_strategy_defaults
        
        # First, process enough data to establish initial signals, then clear the queue
        for _, row in momentum_sample_data_df.iterrows():
            if row.name <= pd.Timestamp('2023-01-05'):
                ts = row.name
                market_event = MarketEvent(
                    symbol=strat.symbols[0], 
                    timestamp=ts, 
                    open_price=row['Open'], 
                    high_price=row['High'], 
                    low_price=row['Low'], 
                    close_price=row['Close'], 
                    volume=int(row['Volume'])
                )
                strat.on_market_data(market_event)
        
        # Clear any previous orders and set position state
        while not basic_event_queue.empty():
            basic_event_queue.get_event()
        strat.last_order_direction[strat.symbols[0]] = 'BUY'
        strat.current_positions[strat.symbols[0]] = 'LONG'
        
        # Now process the data that should generate a SELL signal (2023-01-06 with close=10)
        target_ts_sell = pd.Timestamp('2023-01-06')
        row_target = momentum_sample_data_df.loc[target_ts_sell]
        ts = row_target.name
        market_event = MarketEvent(
            symbol=strat.symbols[0], 
            timestamp=ts, 
            open_price=row_target['Open'], 
            high_price=row_target['High'], 
            low_price=row_target['Low'], 
            close_price=row_target['Close'], 
            volume=int(row_target['Volume'])
        )
        strat.on_market_data(market_event)
        
        order_event: OrderEvent = basic_event_queue.get_event()
        assert order_event.direction == 'SELL' and order_event.reference_price == 10

    def test_momentum_limit_order_generation(self, basic_event_queue: EventQueue, momentum_sample_data_df: pd.DataFrame):
        strat_lmt = MomentumStrategy(basic_event_queue, ['TESTLMT'], momentum_window=3, order_quantity=5,
                                     limit_order_offset_pct=0.01, data_handler=momentum_sample_data_df)
        target_ts_buy = pd.Timestamp('2023-01-04')
        for _, row in momentum_sample_data_df.iterrows():
            if row.name > target_ts_buy: break
            ts = row.name
            market_event = MarketEvent(
                symbol=strat_lmt.symbols[0], 
                timestamp=ts, 
                open_price=row['Open'], 
                high_price=row['High'], 
                low_price=row['Low'], 
                close_price=row['Close'], 
                volume=int(row['Volume'])
            )
            strat_lmt.on_market_data(market_event)
        order_event: OrderEvent = basic_event_queue.get_event()
        assert order_event.order_type == 'LMT' and math.isclose(order_event.limit_price, 12 * 0.99)

    def test_momentum_stop_loss_generation(self, momentum_sample_data_df):
        eq = EventQueue()
        strat_sl = MomentumStrategy(
            event_queue=eq,
            symbols=['TESTSL'], 
            momentum_window=3, 
            order_quantity=20,
            stop_loss_pct=0.05, 
            atr_period=3,
            data_handler=momentum_sample_data_df
        )
        
        # Feed market data to build momentum
        for _, row in momentum_sample_data_df.iterrows():
            if row.name <= pd.Timestamp('2023-01-04'):
                ts = row.name
                event = MarketEvent(
                    symbol='TESTSL', 
                    timestamp=ts, 
                    open_price=row['Open'], 
                    high_price=row['High'], 
                    low_price=row['Low'], 
                    close_price=row['Close'], 
                    volume=row['Volume']
                )
                strat_sl.on_market_data(event)

        # Clear any market data generated orders
        while not eq.empty():
            eq.get_event()

        buy_fill = FillEvent(
            symbol=strat_sl.symbols[0], 
            quantity=20, 
            direction='BUY', 
            fill_price=12.0, 
            commission=1.0,
            timestamp=pd.Timestamp('2023-01-04'),
            order_id='B_FILL'
        )
        strat_sl.current_positions[strat_sl.symbols[0]] = None
        strat_sl.on_fill(buy_fill)
        
        # Check if stop loss order was generated
        assert not eq.empty(), "Expected stop loss order to be generated"
        sl_order_event: OrderEvent = eq.get_event()
        assert sl_order_event.order_type == 'STP' and math.isclose(sl_order_event.stop_price, 12.0 * 0.95)
        assert sl_order_event.current_atr is not None

    def test_momentum_no_signal_if_window_not_met(self, momentum_strategy_defaults: MomentumStrategy, basic_event_queue: EventQueue, momentum_sample_data_df: pd.DataFrame):
        strat = momentum_strategy_defaults
        for _, row in momentum_sample_data_df.head(2).iterrows(): # Window is 3
            ts = row.name
            market_event = MarketEvent(
                symbol=strat.symbols[0], 
                timestamp=ts, 
                open_price=row['Open'], 
                high_price=row['High'], 
                low_price=row['Low'], 
                close_price=row['Close'], 
                volume=int(row['Volume'])
            )
            strat.on_market_data(market_event)
        assert basic_event_queue.empty()

