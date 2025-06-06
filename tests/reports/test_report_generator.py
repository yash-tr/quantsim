"""
Unit tests for the ReportGenerator and plotting functions.
"""
import os
import pytest
import pandas as pd
from datetime import datetime, timedelta

from quantsim.reports.report_generator import ReportGenerator
from quantsim.reports.plotter import plot_equity_curve, plot_drawdown_series
from quantsim.portfolio.trade_log import Trade
from quantsim.core.events import FillEvent
from quantsim.portfolio.trade_log import TradeLog

@pytest.fixture
def sample_metrics():
    """Sample performance metrics for testing."""
    return {
        'total_return_pct': 15.5,
        'cagr_pct': 5.2,
        'sharpe_ratio': 1.2,
        'max_drawdown_pct': -8.3,
        'realized_pnl': 15500.0,
    }

@pytest.fixture
def sample_equity_curve():
    """Sample equity curve DataFrame."""
    dates = pd.to_datetime([datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)])
    values = [100000 + i * 1000 for i in range(10)]
    return pd.DataFrame({'PortfolioValue': values}, index=dates)

@pytest.fixture
def sample_drawdown_series():
    """Sample drawdown Series."""
    dates = pd.to_datetime([datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)])
    drawdowns = [0, -0.01, -0.005, -0.02, -0.01, 0, -0.03, -0.02, -0.01, 0]
    return pd.Series(drawdowns, index=dates)

@pytest.fixture
def sample_trades():
    """Sample list of completed Trade objects."""
    trade_log = TradeLog()
    
    # First trade
    fill1_entry = FillEvent(symbol='AAPL', timestamp=datetime(2023, 1, 2), direction='BUY', quantity=10, fill_price=150.0, commission=5.0)
    fill1_exit = FillEvent(symbol='AAPL', timestamp=datetime(2023, 1, 5), direction='SELL', quantity=10, fill_price=155.0, commission=5.0)
    trade_log.process_fill(fill1_entry)
    trade_log.process_fill(fill1_exit)

    # Second trade
    fill2_entry = FillEvent(symbol='AAPL', timestamp=datetime(2023, 1, 8), direction='BUY', quantity=10, fill_price=152.0, commission=5.0)
    fill2_exit = FillEvent(symbol='AAPL', timestamp=datetime(2023, 1, 10), direction='SELL', quantity=10, fill_price=160.0, commission=5.0)
    trade_log.process_fill(fill2_entry)
    trade_log.process_fill(fill2_exit)
    
    return trade_log.get_completed_trades()

@pytest.fixture
def report_generator(tmpdir, sample_metrics, sample_equity_curve, sample_drawdown_series, sample_trades):
    """Fixture to create a ReportGenerator instance."""
    output_dir = str(tmpdir)
    return ReportGenerator(
        portfolio_metrics=sample_metrics,
        equity_curve=sample_equity_curve,
        drawdown_series=sample_drawdown_series,
        completed_trades=sample_trades,
        output_dir=output_dir,
        strategy_name="TestStrategy",
        symbol="AAPL",
        initial_capital=100000.0
    )

class TestReportGenerator:
    """Tests for the ReportGenerator class."""

    def test_report_generator_initialization(self, report_generator):
        assert report_generator.strategy_name == "TestStrategy"
        assert report_generator.symbol == "AAPL"
        assert len(report_generator.completed_trades) == 2

    def test_generate_report(self, report_generator, mocker):
        """Test the main report generation function."""
        mock_plot_equity = mocker.patch('quantsim.reports.report_generator.plot_equity_curve')
        mock_plot_drawdown = mocker.patch('quantsim.reports.report_generator.plot_drawdown_series')

        report_generator.generate_report()

        mock_plot_equity.assert_called_once()
        mock_plot_drawdown.assert_called_once()

        report_files = os.listdir(report_generator.output_dir)
        md_files = [f for f in report_files if f.endswith('.md')]
        assert len(md_files) == 1

        md_file_path = os.path.join(report_generator.output_dir, md_files[0])
        with open(md_file_path, 'r') as f:
            content = f.read()
            assert "# Backtest Report: TestStrategy on AAPL" in content
            assert "## Performance Metrics" in content
            assert "- **Total Return (%):** 15.50" in content
            assert "## Equity Curve" in content
            assert "![Equity Curve](equity_curve_AAPL_TestStrategy_" in content
            assert "## Drawdown" in content
            assert "![Portfolio Drawdown](drawdown_series_AAPL_TestStrategy_" in content
            assert "## Trade Log Summary" in content

    def test_generate_report_no_drawdown(self, report_generator, mocker):
        """Test report generation when drawdown series is not available."""
        mock_plot_equity = mocker.patch('quantsim.reports.report_generator.plot_equity_curve')
        mock_plot_drawdown = mocker.patch('quantsim.reports.report_generator.plot_drawdown_series')
        report_generator.drawdown_series = None

        report_generator.generate_report()

        mock_plot_equity.assert_called_once()
        mock_plot_drawdown.assert_not_called()
        
        report_files = os.listdir(report_generator.output_dir)
        md_files = [f for f in report_files if f.endswith('.md')]
        md_file_path = os.path.join(report_generator.output_dir, md_files[0])
        with open(md_file_path, 'r') as f:
            content = f.read()
            assert "Drawdown series data not available or empty." in content

    def test_generate_report_no_metrics(self, report_generator, mocker):
        """Test report generation when metrics are not available."""
        mocker.patch('quantsim.reports.report_generator.plot_equity_curve')
        report_generator.portfolio_metrics = {}

        report_generator.generate_report()
        
        report_files = os.listdir(report_generator.output_dir)
        md_files = [f for f in report_files if f.endswith('.md')]
        md_file_path = os.path.join(report_generator.output_dir, md_files[0])
        with open(md_file_path, 'r') as f:
            content = f.read()
            assert "No metrics available." in content

class TestPlotterFunctions:
    """Tests for plotting functions in plotter.py."""

    def test_plot_equity_curve(self, tmpdir, sample_equity_curve):
        """Test the equity curve plotting function."""
        output_path = os.path.join(str(tmpdir), 'equity.png')
        plot_equity_curve(sample_equity_curve, output_path, initial_capital=100000.0)
        assert os.path.exists(output_path)

    def test_plot_equity_curve_no_data(self, tmpdir):
        """Test equity curve plotting with no data."""
        output_path = os.path.join(str(tmpdir), 'equity.png')
        plot_equity_curve(pd.DataFrame(), output_path)
        assert not os.path.exists(output_path)

    def test_plot_drawdown_series(self, tmpdir, sample_drawdown_series):
        """Test the drawdown series plotting function."""
        output_path = os.path.join(str(tmpdir), 'drawdown.png')
        plot_drawdown_series(sample_drawdown_series, output_path)
        assert os.path.exists(output_path)

    def test_plot_drawdown_series_no_data(self, tmpdir):
        """Test drawdown series plotting with no data."""
        output_path = os.path.join(str(tmpdir), 'drawdown.png')
        plot_drawdown_series(pd.Series(dtype=float), output_path)
        assert not os.path.exists(output_path) 