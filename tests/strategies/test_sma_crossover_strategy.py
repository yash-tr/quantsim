"""
Unit tests for quantsim.strategies.sma_crossover.SMACrossoverStrategy
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import math
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import MarketEvent, OrderEvent, FillEvent
from quantsim.strategies.sma_crossover import SMACrossoverStrategy
from quantsim.indicators import calculate_sma, calculate_atr

pytestmark = pytest.mark.strategies

@pytest.fixture
def basic_event_queue_sma() -> EventQueue:
    return EventQueue()

@pytest.fixture
def sma_sample_market_data_df() -> pd.DataFrame:
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                            '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
                            '2023-01-09', '2023-01-10'])
    data = {'Open':  [10,11,12,13,14,17,18,19,18,17], 'High': [11,12,13,14,18,18,19,20,19,18],
            'Low':   [9,10,11,12,13,16,17,18,17,16], 'Close': [11,12,13,14,17,18,19,18,17,16],
            'Volume':[100,100,100,100,150,100,100,150,100,100]}
    return pd.DataFrame(data, index=pd.Index(dates, name="Timestamp"))

@pytest.fixture
def strategy_mkt_defaults_sma(basic_event_queue_sma: EventQueue, sma_sample_market_data_df: pd.DataFrame) -> SMACrossoverStrategy:
    return SMACrossoverStrategy(
        event_queue=basic_event_queue_sma, symbols=['TEST_SMA'], short_window=3, long_window=5,
        order_quantity=10, data_handler=sma_sample_market_data_df)

class TestSMACrossoverStrategy:
    def test_strategy_creation_and_sma_precalculation(self, strategy_mkt_defaults_sma: SMACrossoverStrategy):
        strat = strategy_mkt_defaults_sma
        assert strat.symbol == 'TEST_SMA'
        assert strat.short_sma_series is not None and strat.long_sma_series is not None
        assert math.isclose(strat.short_sma_series.loc[pd.Timestamp('2023-01-03')], (11+12+13)/3)

    def test_strategy_atr_precalculation(self, basic_event_queue_sma: EventQueue, sma_sample_market_data_df: pd.DataFrame):
        strat_atr = SMACrossoverStrategy(basic_event_queue_sma, ['TEST_SMA_ATR'], short_window=3, long_window=5,
                                         atr_period=3, data_handler=sma_sample_market_data_df)
        assert strat_atr.atr_series is not None and not strat_atr.atr_series.empty

    def test_strategy_buy_signal_market_order(self, strategy_mkt_defaults_sma: SMACrossoverStrategy, basic_event_queue_sma: EventQueue, sma_sample_market_data_df: pd.DataFrame):
        strat = strategy_mkt_defaults_sma
        for _, row in sma_sample_market_data_df.iterrows():
            if row.name > pd.Timestamp('2023-01-05'): break
            strat.on_market_data(MarketEvent(
                symbol=strat.symbol, 
                timestamp=row.name,
                open_price=row['Open'], 
                high_price=row['High'], 
                low_price=row['Low'], 
                close_price=row['Close'], 
                volume=int(row['Volume'])
            ))
        order: OrderEvent = basic_event_queue_sma.get_event()
        assert order.direction == 'BUY' and order.reference_price == 17

    def test_strategy_sell_signal_market_order(self, strategy_mkt_defaults_sma: SMACrossoverStrategy, basic_event_queue_sma: EventQueue, sma_sample_market_data_df: pd.DataFrame):
        strat = strategy_mkt_defaults_sma
        
        # First, process enough data to establish a BUY position, then clear the queue
        for _, row in sma_sample_market_data_df.iterrows():
            if row.name <= pd.Timestamp('2023-01-07'):  # Process to get initial BUY signal
                strat.on_market_data(MarketEvent(
                    symbol=strat.symbol, 
                    timestamp=row.name,
                    open_price=row['Open'], 
                    high_price=row['High'], 
                    low_price=row['Low'], 
                    close_price=row['Close'], 
                    volume=int(row['Volume'])
                ))
        
        # Clear any previous orders and set position state manually
        while not basic_event_queue_sma.empty():
            basic_event_queue_sma.get_event()
        strat.last_order_direction[strat.symbol] = 'BUY'
        strat.current_position[strat.symbol] = 'LONG'
        
        # Now process to 2023-01-10 where SMA3 < SMA5 for actual crossover
        for _, row in sma_sample_market_data_df.loc[pd.Timestamp('2023-01-08'):pd.Timestamp('2023-01-10')].iterrows():
            strat.on_market_data(MarketEvent(
                symbol=strat.symbol, 
                timestamp=row.name,
                open_price=row['Open'], 
                high_price=row['High'], 
                low_price=row['Low'], 
                close_price=row['Close'], 
                volume=int(row['Volume'])
            ))
        
        final_order = None
        while not basic_event_queue_sma.empty(): 
            final_order = basic_event_queue_sma.get_event()
        
        assert final_order and final_order.direction == 'SELL' and final_order.reference_price == 16

    def test_strategy_limit_order_generation(self, basic_event_queue_sma: EventQueue, sma_sample_market_data_df: pd.DataFrame):
        strat_lmt = SMACrossoverStrategy(basic_event_queue_sma, ['TESTLMT_SMA'], short_window=3, long_window=5,
                                         limit_order_offset_pct=0.01, data_handler=sma_sample_market_data_df)
        for _, row in sma_sample_market_data_df.loc[:pd.Timestamp('2023-01-05')].iterrows():
            strat_lmt.on_market_data(MarketEvent(
                symbol=strat_lmt.symbol, 
                timestamp=row.name,
                open_price=row['Open'], 
                high_price=row['High'], 
                low_price=row['Low'], 
                close_price=row['Close'], 
                volume=int(row['Volume'])
            ))
        order: OrderEvent = basic_event_queue_sma.get_event()
        assert order.order_type == 'LMT' and math.isclose(order.limit_price, 17 * 0.99)

    def test_strategy_stop_loss_generation_on_fill(self, basic_event_queue_sma: EventQueue, sma_sample_market_data_df: pd.DataFrame):
        strat_sl = SMACrossoverStrategy(basic_event_queue_sma, ['TESTSL_SMA'], short_window=3, long_window=5,
                                        stop_loss_pct=0.05, data_handler=sma_sample_market_data_df, atr_period=3)
        fill = FillEvent(
            symbol=strat_sl.symbol, 
            quantity=10, 
            direction='BUY', 
            fill_price=17.0, 
            commission=1.0,
            timestamp=pd.Timestamp('2023-01-05'),
            order_id='B_FILL'
        )
        strat_sl.current_position[strat_sl.symbol] = None
        strat_sl.on_fill(fill)
        sl_order: OrderEvent = basic_event_queue_sma.get_event()
        assert sl_order.order_type == 'STP' and math.isclose(sl_order.stop_price, 17.0 * 0.95)
        assert sl_order.current_atr is not None

    def test_strategy_no_redundant_signals(self, strategy_mkt_defaults_sma: SMACrossoverStrategy, basic_event_queue_sma: EventQueue, sma_sample_market_data_df: pd.DataFrame):
        strat = strategy_mkt_defaults_sma
        for _, row in sma_sample_market_data_df.loc[:pd.Timestamp('2023-01-05')].iterrows():
            strat.on_market_data(MarketEvent(
                symbol=strat.symbol, 
                timestamp=row.name,
                open_price=row['Open'], 
                high_price=row['High'], 
                low_price=row['Low'], 
                close_price=row['Close'], 
                volume=int(row['Volume'])
            ))
        _ = basic_event_queue_sma.get_event() # Consume first order
        ts_next = pd.Timestamp('2023-01-06')
        row_next = sma_sample_market_data_df.loc[ts_next]
        strat.on_market_data(MarketEvent(
            symbol=strat.symbol, 
            timestamp=ts_next, 
            open_price=row_next['Open'], 
            high_price=row_next['High'], 
            low_price=row_next['Low'], 
            close_price=row_next['Close'], 
            volume=int(row_next['Volume'])
        ))
        assert basic_event_queue_sma.empty()

    def test_strategy_handles_missing_indicator_values_gracefully(self, basic_event_queue_sma: EventQueue):
        short_df = pd.DataFrame({'Close': [10]}, index=pd.Index([pd.Timestamp('2023-01-01')], name="Timestamp"))
        strat = SMACrossoverStrategy(basic_event_queue_sma, ['SHORTTEST_SMA'], short_window=3, long_window=5, data_handler=short_df)
        strat.on_market_data(MarketEvent(
            symbol=strat.symbol, 
            timestamp=pd.Timestamp('2023-01-01'), 
            open_price=10,
            high_price=10,
            low_price=10,
            close_price=10,
            volume=100
        ))
        assert basic_event_queue_sma.empty()
        if strat.short_sma_series is not None: assert pd.isna(strat.short_sma_series.iloc[0])

