"""
Unit tests for advanced portfolio metrics calculations.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quantsim.portfolio.portfolio import Portfolio
from quantsim.core.event_queue import EventQueue

pytestmark = pytest.mark.portfolio

@pytest.fixture
def portfolio_for_metrics() -> Portfolio:
    portfolio = Portfolio(initial_cash=100000.0, event_queue=EventQueue())
    timestamps = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02'),
                  pd.Timestamp('2023-01-03'), pd.Timestamp('2023-01-04'),
                  pd.Timestamp('2023-01-05'), pd.Timestamp('2023-01-06')]
    values = [101000.0, 102000.0, 103000.0, 100000.0, 101000.0, 101500.0]
    portfolio.equity_curve = list(zip(timestamps, values))
    portfolio.current_total_value = values[-1]
    return portfolio

class TestPortfolioMetrics:
    def test_calculate_total_return(self, portfolio_for_metrics: Portfolio):
        pf = portfolio_for_metrics
        pf.calculate_performance_metrics()
        expected_return = (101500.0 / 100000.0 - 1) * 100
        assert pf.metrics['total_return_pct'] == pytest.approx(expected_return)

    def test_calculate_max_drawdown(self, portfolio_for_metrics: Portfolio):
        pf = portfolio_for_metrics
        pf.calculate_performance_metrics()
        expected_max_drawdown = (100000.0 - 103000.0) / 103000.0 * 100
        assert pf.metrics['max_drawdown_pct'] == pytest.approx(expected_max_drawdown)

        pf_up = Portfolio(initial_cash=100000.0, event_queue=EventQueue())
        pf_up.equity_curve = [(pd.Timestamp('2023-01-01'), 100000.0),
                              (pd.Timestamp('2023-01-02'), 101000.0)]
        if pf_up.equity_curve: pf_up.current_total_value = pf_up.equity_curve[-1][1]
        else: pf_up.current_total_value = pf_up.initial_cash
        pf_up.calculate_performance_metrics()
        assert pf_up.metrics.get('max_drawdown_pct', 0.0) == pytest.approx(0.0)

    def test_calculate_sharpe_ratio(self, portfolio_for_metrics: Portfolio):
        pf = portfolio_for_metrics
        risk_free_rate = 0.01
        pf.calculate_performance_metrics(risk_free_rate=risk_free_rate)
        equity_df = pd.DataFrame(pf.equity_curve, columns=['Timestamp', 'PortfolioValue'])
        equity_df = equity_df.set_index(pd.to_datetime(equity_df['Timestamp'])).sort_index().drop_duplicates()
        returns = equity_df['PortfolioValue'].pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0: expected_sharpe = np.nan
        else:
            mean_daily_return = returns.mean()
            std_daily_return = returns.std()
            daily_rfr = (1 + risk_free_rate)**(1/252) - 1
            expected_sharpe = (mean_daily_return - daily_rfr) / std_daily_return * np.sqrt(252)
        assert pf.metrics['sharpe_ratio'] == pytest.approx(expected_sharpe, nan_ok=True)

    def test_calculate_cagr(self, portfolio_for_metrics: Portfolio):
        pf = portfolio_for_metrics
        pf.calculate_performance_metrics()
        final_value = pf.equity_curve[-1][1]
        initial_value = pf.initial_cash
        start_date = pf.equity_curve[0][0]
        end_date = pf.equity_curve[-1][0]
        time_delta_years = (end_date - start_date).days / 365.25
        if time_delta_years == 0:
             equity_df = pd.DataFrame(pf.equity_curve, columns=['Timestamp', 'PortfolioValue']).set_index('Timestamp')
             returns_for_cagr = equity_df['PortfolioValue'].pct_change().dropna()
             expected_cagr = returns_for_cagr.sum() * 100
        else: expected_cagr = ((final_value / initial_value)**(1.0 / time_delta_years) - 1) * 100
        assert pf.metrics['cagr_pct'] == pytest.approx(expected_cagr)

    def test_calculate_volatility(self, portfolio_for_metrics: Portfolio):
        pf = portfolio_for_metrics
        pf.calculate_performance_metrics()
        equity_df = pd.DataFrame(pf.equity_curve, columns=['Timestamp', 'PortfolioValue'])
        equity_df = equity_df.set_index(pd.to_datetime(equity_df['Timestamp'])).sort_index().drop_duplicates()
        returns = equity_df['PortfolioValue'].pct_change().dropna()
        expected_volatility = returns.std() * np.sqrt(252) * 100
        assert pf.metrics['annualized_volatility_pct'] == pytest.approx(expected_volatility)

    def test_metrics_with_empty_equity_curve(self):
        pf = Portfolio(initial_cash=100000.0, event_queue=EventQueue())
        pf.current_total_value = pf.initial_cash
        pf.calculate_performance_metrics()
        assert pf.metrics.get('error') == "Insufficient equity data"
        assert pf.metrics.get('total_return_pct') == pytest.approx(0.0)

    def test_metrics_with_single_point_equity_curve(self):
        pf = Portfolio(initial_cash=100000.0, event_queue=EventQueue())
        pf.equity_curve = [(pd.Timestamp('2023-01-01'), 100500.0)]
        pf.current_total_value = 100500.0
        pf.calculate_performance_metrics()
        assert pf.metrics.get('error') == "Insufficient equity data"
        assert pf.metrics.get('total_return_pct') == pytest.approx(0.5)

    def test_metrics_with_flat_equity_curve(self):
        pf = Portfolio(initial_cash=100000.0, event_queue=EventQueue())
        timestamps = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03')]
        values = [101000.0, 101000.0, 101000.0]
        pf.equity_curve = list(zip(timestamps, values))
        pf.current_total_value = 101000.0
        pf.calculate_performance_metrics(risk_free_rate=0.01)
        assert pf.metrics['total_return_pct'] == pytest.approx(1.0)
        assert pf.metrics['annualized_volatility_pct'] == pytest.approx(0.0)
        assert pd.isna(pf.metrics['sharpe_ratio'])
        assert pf.metrics['max_drawdown_pct'] == pytest.approx(0.0)

