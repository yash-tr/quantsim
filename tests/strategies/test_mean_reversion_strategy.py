"""
Unit tests for quantsim.strategies.mean_reversion.MeanReversionStrategy
"""
import pytest
import pandas as pd
from datetime import datetime
import math
from quantsim.core.event_queue import EventQueue
from quantsim.core.events import MarketEvent, OrderEvent, FillEvent
from quantsim.strategies.mean_reversion import MeanReversionStrategy
from quantsim.indicators import calculate_sma, calculate_atr

pytestmark = pytest.mark.strategies

@pytest.fixture
def basic_event_queue_mr() -> EventQueue:
    return EventQueue()

@pytest.fixture
def mr_sample_data_df() -> pd.DataFrame:
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
                            '2023-01-05', '2023-01-06', '2023-01-07'])
    data = {'Open':  [10, 11, 12, 11,  10, 11, 12.5],
            'High':  [10, 11, 12, 11,  11, 12, 13],
            'Low':   [10, 11, 12, 9.5, 10, 11, 12.5],
            'Close': [10, 11, 12, 9.5, 11, 12, 13],
            'Volume':[100,100,100,150,150,150,150]}
    return pd.DataFrame(data, index=pd.Index(dates, name="Timestamp"))

@pytest.fixture
def mr_strategy_defaults(basic_event_queue_mr: EventQueue, mr_sample_data_df: pd.DataFrame) -> MeanReversionStrategy:
    return MeanReversionStrategy(
        event_queue=basic_event_queue_mr, symbols=['TESTMR'], sma_window=3,
        reversion_threshold=0.10, order_quantity=5, data_handler=mr_sample_data_df, atr_period=0
    )

class TestMeanReversionStrategy:
    def test_mr_creation_and_sma_atr_precalculation(self, mr_strategy_defaults: MeanReversionStrategy, basic_event_queue_mr: EventQueue, mr_sample_data_df: pd.DataFrame):
        strat = mr_strategy_defaults
        assert strat.symbols == ['TESTMR']
        assert strat.sma_series['TESTMR'] is not None
        assert math.isclose(strat.sma_series['TESTMR'].loc[pd.Timestamp('2023-01-03')], 11.0)

        strat_with_atr = MeanReversionStrategy(
            event_queue=basic_event_queue_mr, 
            symbols=['TEST'], 
            sma_window=3, 
            atr_period=3, 
            data_handler=mr_sample_data_df
        )
        assert strat_with_atr.atr_series['TEST'] is not None

    def test_mr_buy_signal_price_below_threshold(self, mr_strategy_defaults: MeanReversionStrategy, basic_event_queue_mr: EventQueue, mr_sample_data_df: pd.DataFrame):
        strat = mr_strategy_defaults
        target_ts_buy = pd.Timestamp('2023-01-04')
        for _, row in mr_sample_data_df.iterrows():
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

        order_event: OrderEvent = basic_event_queue_mr.get_event()
        assert order_event.direction == 'BUY' and order_event.reference_price == 9.5

    def test_mr_sell_signal_price_above_threshold(self, mr_strategy_defaults: MeanReversionStrategy, basic_event_queue_mr: EventQueue, mr_sample_data_df: pd.DataFrame):
        strat = mr_strategy_defaults
        strat.last_order_direction[strat.symbols[0]] = 'BUY'; strat.current_positions[strat.symbols[0]] = 'LONG'
        target_ts_sell = pd.Timestamp('2023-01-07')
        for _, row in mr_sample_data_df.loc[:target_ts_sell].iterrows():
            ts = row.name
            strat.on_market_data(MarketEvent(
                symbol=strat.symbols[0], 
                timestamp=ts, 
                open_price=row['Open'], 
                high_price=row['High'], 
                low_price=row['Low'], 
                close_price=row['Close'], 
                volume=int(row['Volume'])
            ))

        final_order = None
        while not basic_event_queue_mr.empty(): final_order = basic_event_queue_mr.get_event()
        assert final_order is not None and final_order.direction == 'SELL' and final_order.reference_price == 13

    def test_mr_no_signal_within_threshold(self, mr_strategy_defaults: MeanReversionStrategy, basic_event_queue_mr: EventQueue, mr_sample_data_df: pd.DataFrame):
        strat = mr_strategy_defaults
        for _, row in mr_sample_data_df.loc[:pd.Timestamp('2023-01-04')].iterrows(): # Process up to BUY
            ts = row.name
            strat.on_market_data(MarketEvent(
                symbol=strat.symbols[0], 
                timestamp=ts, 
                open_price=row['Open'], 
                high_price=row['High'], 
                low_price=row['Low'], 
                close_price=row['Close'], 
                volume=int(row['Volume'])
            ))
        while not basic_event_queue_mr.empty(): basic_event_queue_mr.get_event() # Clear BUY

        strat.last_order_direction[strat.symbols[0]] = 'BUY'; strat.current_positions[strat.symbols[0]] = 'LONG'

        ts_target = pd.Timestamp('2023-01-05') # Price is 11, SMA is 10.833, Dev is 0.015 (within 0.10 threshold but should trigger exit)
        row_target = mr_sample_data_df.loc[ts_target]
        strat.on_market_data(MarketEvent(
            symbol=strat.symbols[0], 
            timestamp=ts_target, 
            open_price=row_target['Open'],
            high_price=row_target['High'],
            low_price=row_target['Low'],
            close_price=row_target['Close'],
            volume=int(row_target['Volume'])
        ))

        assert not basic_event_queue_mr.empty() # Exit order should be generated
        exit_order: OrderEvent = basic_event_queue_mr.get_event()
        assert exit_order.direction == 'SELL'

    def test_mr_stop_loss_generation(self, mr_sample_data_df):
        eq = EventQueue()
        strat_sl = MeanReversionStrategy(
            event_queue=eq, 
            symbols=['TESTSL_MR'], 
            sma_window=3, 
            reversion_threshold=0.10, 
            atr_period=3,
            stop_loss_pct=0.03,
            data_handler=mr_sample_data_df
        )
        
        # Feed some market data first
        for _, row in mr_sample_data_df.iterrows():
            if row.name <= pd.Timestamp('2023-01-04'):
                ts = row.name
                event = MarketEvent(
                    symbol='TESTSL_MR', 
                    timestamp=ts, 
                    open_price=row['Open'], 
                    high_price=row['High'], 
                    low_price=row['Low'], 
                    close_price=row['Close'], 
                    volume=int(row['Volume'])
                )
                strat_sl.on_market_data(event)

        # Clear any market data generated orders
        while not eq.empty():
            eq.get_event()

        buy_fill = FillEvent(
            symbol=strat_sl.symbols[0], 
            quantity=10, 
            direction='BUY', 
            fill_price=9.5, 
            commission=0.5,
            timestamp=pd.Timestamp('2023-01-04'),
            order_id='BUY_FILL_MR'
        )
        strat_sl.current_positions[strat_sl.symbols[0]] = None
        strat_sl.on_fill(buy_fill)

        # Check if stop loss order was generated
        assert not eq.empty(), "Expected stop loss order to be generated"
        sl_order_event: OrderEvent = eq.get_event()
        assert sl_order_event.order_type == 'STP' and sl_order_event.direction == 'SELL'
        assert math.isclose(sl_order_event.stop_price, 9.5 * (1 - 0.03))
        assert sl_order_event.current_atr is not None

