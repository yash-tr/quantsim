"""
Command Line Interface for QuantSim backtester.
Supports single runs and batch processing from YAML configuration.
"""

import argparse
import datetime
import os
import pandas as pd
import yaml  # For batch processing
from argparse import Namespace
from typing import Any, Optional, List
import traceback

from quantsim.core.event_queue import EventQueue
from quantsim.core.simulation_engine import SimulationEngine
from quantsim.data import (
    CSVDataManager,
    YahooFinanceDataHandler,
    SyntheticDataHandler,
    DataHandler,
)
from quantsim.strategies.base import Strategy
from quantsim.strategies import (
    SMACrossoverStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    PairsTradingStrategy,
    SimpleMLStrategy,
)
from quantsim.portfolio.portfolio import Portfolio
from quantsim.portfolio.risk_parity_portfolio import RiskParityPortfolio
from quantsim.portfolio.position_sizer import (
    PositionSizer,
    FixedQuantitySizer,
    RiskPercentageSizer,
)
from quantsim.execution.execution_handler import (
    SimulatedExecutionHandler,
    FixedCommission,
)
from quantsim.execution.slippage import PercentageSlippage, ATRSlippage, SlippageModel
from quantsim.reports.report_generator import ReportGenerator
from quantsim.ml.trainer import ModelTrainer


def create_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QuantSim Backtesting Engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_source_group = parser.add_argument_group("Data Source Options")
    data_source_group.add_argument(
        "--data-source",
        type=str,
        default="csv",
        choices=["csv", "yahoo", "synthetic"],
        help="Source of market data.",
    )

    csv_group = parser.add_argument_group("CSV Data Source Specific Options")
    csv_group.add_argument(
        "--symbol",
        type=str,
        help="Primary symbol for trading (used with CSV data source as primary symbol for data loading and naming).",
    )
    csv_group.add_argument(
        "--csv-file-path",
        type=str,
        help="Full path to the CSV data file for the --symbol.",
    )
    csv_group.add_argument(
        "--csv-date-col",
        type=str,
        default="Date",
        help="Name of the date column in the CSV file.",
    )

    yahoo_group = parser.add_argument_group(
        "Yahoo Finance Data Source Specific Options"
    )
    yahoo_group.add_argument(
        "--yf-symbols",
        type=str,
        nargs="+",
        help="List of symbols to download from Yahoo Finance (e.g., AAPL MSFT).",
    )
    yahoo_group.add_argument(
        "--yf-start-date",
        type=str,
        help="Start date for Yahoo Finance download (YYYY-MM-DD).",
    )
    yahoo_group.add_argument(
        "--yf-end-date",
        type=str,
        help="End date for Yahoo Finance download (YYYY-MM-DD).",
    )
    yahoo_group.add_argument(
        "--yf-interval", type=str, default="1d", help="Interval for Yahoo Finance data."
    )

    synth_group = parser.add_argument_group("Synthetic Data Source Options")
    synth_group.add_argument(
        "--synth-symbols",
        type=str,
        nargs="+",
        default=["SYNTH1"],
        help="List of symbols for synthetic data.",
    )
    synth_group.add_argument(
        "--synth-start-date",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD).",
    )
    synth_group.add_argument(
        "--synth-end-date",
        type=str,
        default="2023-12-31",
        help="End date (YYYY-MM-DD).",
    )
    synth_group.add_argument(
        "--synth-initial-price", type=float, default=100.0, help="Initial price."
    )
    synth_group.add_argument(
        "--synth-drift", type=float, default=0.0001, help="Drift per period."
    )
    synth_group.add_argument(
        "--synth-volatility", type=float, default=0.01, help="Volatility per period."
    )
    synth_group.add_argument(
        "--synth-frequency",
        type=str,
        default="B",
        help="Data frequency (pandas offset string).",
    )
    synth_group.add_argument(
        "--synth-seed", type=int, default=None, help="Random seed (optional)."
    )

    general_group = parser.add_argument_group("General Backtest Options")
    general_group.add_argument(
        "--strategy",
        type=str,
        choices=[
            "sma_crossover",
            "momentum",
            "mean_reversion",
            "pairs_trading",
            "simple_ml",
        ],
        help="Strategy name (required for single run).",
    )
    general_group.add_argument(
        "--initial-capital", type=float, default=100000.0, help="Initial capital."
    )
    general_group.add_argument(
        "--output-dir",
        type=str,
        default="backtest_results",
        help="Base output directory for all results.",
    )

    portfolio_group = parser.add_argument_group("Portfolio and Position Sizing Options")
    portfolio_group.add_argument(
        "--portfolio-type",
        type=str,
        default="default",
        choices=["default", "risk_parity"],
        help="Type of portfolio to use.",
    )
    portfolio_group.add_argument(
        "--lookback-period",
        type=int,
        default=252,
        help="Lookback period for portfolio optimizations (e.g., risk_parity).",
    )
    portfolio_group.add_argument(
        "--position-sizer",
        type=str,
        default="fixed",
        choices=["fixed", "risk_percentage"],
        help="Position sizing strategy.",
    )
    portfolio_group.add_argument(
        "--order-quantity",
        type=float,
        default=100.0,
        help="Fixed order quantity (for 'fixed' position sizer).",
    )
    portfolio_group.add_argument(
        "--risk-per-trade-pct",
        type=float,
        default=0.02,
        help="Portfolio equity percentage to risk per trade (for 'risk_percentage' sizer).",
    )
    portfolio_group.add_argument(
        "--stop-loss-pct-sizer",
        type=float,
        default=0.05,
        help="Stop loss percentage used for position size calculation (for 'risk_percentage' sizer).",
    )

    strategy_params_group = parser.add_argument_group("Strategy Specific Parameters")
    # SMA Crossover
    strategy_params_group.add_argument(
        "--short-window", type=int, default=10, help="Short SMA window (sma_crossover)."
    )
    strategy_params_group.add_argument(
        "--long-window", type=int, default=20, help="Long SMA window (sma_crossover)."
    )
    # Momentum
    strategy_params_group.add_argument(
        "--momentum-window", type=int, default=20, help="Momentum window (momentum)."
    )
    # Mean Reversion
    strategy_params_group.add_argument(
        "--mr-sma-window", type=int, default=20, help="SMA window (mean_reversion)."
    )
    strategy_params_group.add_argument(
        "--mr-threshold",
        type=float,
        default=0.02,
        help="Reversion threshold (mean_reversion).",
    )
    # Pairs Trading
    strategy_params_group.add_argument(
        "--pairs-lookback",
        type=int,
        default=60,
        help="Lookback window for cointegration test (pairs_trading).",
    )
    strategy_params_group.add_argument(
        "--pairs-zscore-entry",
        type=float,
        default=2.0,
        help="Z-score to enter a trade (pairs_trading).",
    )
    strategy_params_group.add_argument(
        "--pairs-zscore-exit",
        type=float,
        default=0.5,
        help="Z-score to exit a trade (pairs_trading).",
    )
    # Machine Learning
    strategy_params_group.add_argument(
        "--ml-model-path",
        type=str,
        help="Path to the trained ML model file (simple_ml).",
    )
    # Common
    strategy_params_group.add_argument(
        "--limit-offset-pct",
        type=float,
        default=0.0,
        help="Limit order offset percentage.",
    )
    strategy_params_group.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.0,
        help="Stop-loss percentage for orders.",
    )

    exec_group = parser.add_argument_group("Execution Simulator Options")
    exec_group.add_argument(
        "--slippage-type",
        type=str,
        default="percentage",
        choices=["percentage", "atr"],
        help="Slippage model.",
    )
    exec_group.add_argument(
        "--percentage-slippage-rate",
        type=float,
        default=0.001,
        help="Rate for percentage slippage.",
    )
    exec_group.add_argument(
        "--atr-multiplier",
        type=float,
        default=0.5,
        help="ATR multiplier for ATR slippage.",
    )
    exec_group.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR calculation period (strategy & slippage).",
    )
    exec_group.add_argument(
        "--latency-ms", type=int, default=0, help="Order latency in milliseconds."
    )
    exec_group.add_argument(
        "--partial-fill-max-pct",
        type=float,
        default=1.0,
        help="Max order fill percentage per bar.",
    )
    exec_group.add_argument(
        "--partial-fill-max-qty",
        type=float,
        default=float("inf"),
        help="Max order fill quantity per bar.",
    )
    exec_group.add_argument(
        "--commission-fixed",
        type=float,
        default=1.0,
        help="Fixed commission per trade.",
    )

    return parser


def run_backtest(args: Namespace):
    run_name_cli = getattr(args, "name", "Single_Run")
    current_run_output_dir = os.path.abspath(
        args.output_dir
    )  # output_dir is now pre-set for this run

    print(f"\n--- Starting Backtest: {run_name_cli} ---")
    print(f"Outputting to: {current_run_output_dir}")
    # ... (rest of run_backtest logic: data_handler, strategy, portfolio, exec_handler setup) ...
    event_queue = EventQueue()
    data_handler: DataHandler
    symbols_list: List[str] = []

    if args.data_source.lower() == "csv":
        if not args.symbol or not args.csv_file_path:
            print("Error: CSV: --symbol and --csv-file-path required.")
            return
        csv_path = os.path.abspath(args.csv_file_path)
        print(f"Symbol (CSV): {args.symbol}, Path: {csv_path}")
        data_handler = CSVDataManager(
            symbol=args.symbol, csv_file_path=csv_path, date_column=args.csv_date_col
        )
        symbols_list = [args.symbol]
    elif args.data_source.lower() == "yahoo":
        if not args.yf_symbols or not args.yf_start_date or not args.yf_end_date:
            print(
                "Error: Yahoo: --yf-symbols, --yf-start-date, --yf-end-date required."
            )
            return
        print(
            f"Symbols (Yahoo): {', '.join(args.yf_symbols)}, Period: {args.yf_start_date}-{args.yf_end_date}, Interval: {args.yf_interval}"
        )
        data_handler = YahooFinanceDataHandler(
            symbols=args.yf_symbols,
            start_date=args.yf_start_date,
            end_date=args.yf_end_date,
            interval=args.yf_interval,
        )
        symbols_list = args.yf_symbols
    elif args.data_source.lower() == "synthetic":
        if not args.synth_symbols:
            print("Error: Synthetic: --synth-symbols required.")
            return
        print(
            f"Symbols (Synthetic): {', '.join(args.synth_symbols)}, Period: {args.synth_start_date}-{args.synth_end_date}"
        )
        data_handler = SyntheticDataHandler(
            symbols=args.synth_symbols,
            start_date=args.synth_start_date,
            end_date=args.synth_end_date,
            initial_price=args.synth_initial_price,
            drift_per_period=args.synth_drift,
            volatility_per_period=args.synth_volatility,
            data_frequency=args.synth_frequency,
            seed=args.synth_seed,
        )
        symbols_list = args.synth_symbols
    else:
        print(f"Error: Unknown data source '{args.data_source}'.")
        return

    primary_symbol_for_naming = symbols_list[0] if symbols_list else "UNKNOWN"

    data_for_strategy_init: Any
    if (
        args.strategy.lower() in ["sma_crossover"]
        and primary_symbol_for_naming != "UNKNOWN"
    ):
        data_for_strategy_init = data_handler.get_historical_data(
            primary_symbol_for_naming
        )
        if data_for_strategy_init.empty:
            print(
                f"Warning: No hist data for {primary_symbol_for_naming} for {args.strategy} pre-calc."
            )
    else:
        data_for_strategy_init = data_handler

    print(
        f"\nStrategy: {args.strategy} for {primary_symbol_for_naming if len(symbols_list)==1 else symbols_list}"
    )
    strategy_common_params = {
        "event_queue": event_queue,
        "symbols": symbols_list,
        "limit_order_offset_pct": args.limit_offset_pct,
        "stop_loss_pct": args.stop_loss_pct,
        "atr_period": args.atr_period,
    }
    strategy_object: Optional[Strategy] = None
    if args.strategy.lower() == "sma_crossover":
        strategy_object = SMACrossoverStrategy(
            **strategy_common_params,
            short_window=args.short_window,
            long_window=args.long_window,
            data_handler=data_for_strategy_init,
        )
    elif args.strategy.lower() == "momentum":
        strategy_object = MomentumStrategy(
            **strategy_common_params,
            momentum_window=args.momentum_window,
            data_handler=data_for_strategy_init,
        )
    elif args.strategy.lower() == "mean_reversion":
        strategy_object = MeanReversionStrategy(
            **strategy_common_params,
            sma_window=args.mr_sma_window,
            reversion_threshold=args.mr_threshold,
            data_handler=data_for_strategy_init,
        )
    elif args.strategy.lower() == "pairs_trading":
        if len(symbols_list) != 2:
            print("Error: PairsTradingStrategy requires exactly two symbols.")
            return
        strategy_object = PairsTradingStrategy(
            **strategy_common_params,
            data_handler=data_handler,
            lookback_period=args.pairs_lookback,
            z_score_entry_threshold=args.pairs_zscore_entry,
            z_score_exit_threshold=args.pairs_zscore_exit,
        )
    elif args.strategy.lower() == "simple_ml":
        if not args.ml_model_path:
            print("Error: SimpleMLStrategy requires --ml-model-path.")
            return
        strategy_object = SimpleMLStrategy(
            **strategy_common_params,
            model_path=args.ml_model_path,
            data_handler=data_handler,
        )
    else:
        print(f"Error: Strategy '{args.strategy}' not supported.")
        return

    position_sizer: PositionSizer
    if args.position_sizer == "fixed":
        position_sizer = FixedQuantitySizer(quantity=args.order_quantity)
    elif args.position_sizer == "risk_percentage":
        position_sizer = RiskPercentageSizer(
            risk_per_trade_pct=args.risk_per_trade_pct,
            stop_loss_pct=args.stop_loss_pct_sizer,
        )
    else:
        print(
            f"Error: Position sizer '{args.position_sizer}' not supported. Defaulting to FixedQuantitySizer."
        )
        position_sizer = FixedQuantitySizer(quantity=args.order_quantity)

    portfolio: Portfolio
    if args.portfolio_type == "default":
        portfolio = Portfolio(
            initial_cash=args.initial_capital,
            event_queue=event_queue,
            position_sizer=position_sizer,
        )
    elif args.portfolio_type == "risk_parity":
        portfolio = RiskParityPortfolio(
            initial_cash=args.initial_capital,
            event_queue=event_queue,
            position_sizer=position_sizer,
            symbols=symbols_list,
            lookback_period=args.lookback_period,
        )
    else:
        print(
            f"Error: Portfolio type '{args.portfolio_type}' not supported. Defaulting to standard Portfolio."
        )
        portfolio = Portfolio(
            initial_cash=args.initial_capital,
            event_queue=event_queue,
            position_sizer=position_sizer,
        )

    slippage_model_instance: SlippageModel
    if args.slippage_type.lower() == "atr":
        slippage_model_instance = ATRSlippage(atr_multiplier=args.atr_multiplier)
    else:
        slippage_model_instance = PercentageSlippage(
            slippage_rate=args.percentage_slippage_rate
        )
    commission_model_instance = FixedCommission(
        commission_per_trade=args.commission_fixed
    )

    execution_handler = SimulatedExecutionHandler(
        event_queue=event_queue,
        slippage_model=slippage_model_instance,
        commission_model=commission_model_instance,
        latency_ms=args.latency_ms,
        max_fill_pct_per_bar=args.partial_fill_max_pct,
        max_fill_qty_per_bar=args.partial_fill_max_qty,
    )

    print("--- Backtest Setup Complete & Running ---")
    simulation_engine = SimulationEngine(
        event_queue=event_queue,
        data_handler=data_handler,
        strategy=strategy_object,
        portfolio=portfolio,
        execution_handler=execution_handler,
    )
    simulation_engine.run()

    if not os.path.exists(current_run_output_dir):
        os.makedirs(current_run_output_dir)
        print(f"Created run output directory: {current_run_output_dir}")

    base_filename_for_output = f"{primary_symbol_for_naming}_{args.strategy}_{run_name_cli.replace(' ', '_').replace('/', '_')}"
    trades_log_path = os.path.join(
        current_run_output_dir, f"{base_filename_for_output}_trades.csv"
    )
    portfolio.export_trade_log(trades_log_path, include_open_trades=True)

    print("\nGenerating backtest report...")
    metrics = portfolio.metrics
    equity_curve_df = pd.DataFrame(columns=["PortfolioValue"])
    equity_curve_df.index.name = "Timestamp"
    if portfolio.equity_curve:
        equity_curve_df_temp = pd.DataFrame(
            portfolio.equity_curve, columns=["Timestamp", "PortfolioValue"]
        )
        if not equity_curve_df_temp.empty:
            equity_curve_df = (
                equity_curve_df_temp.set_index("Timestamp")
                .sort_index()
                .drop_duplicates()
            )

    report_gen = ReportGenerator(
        portfolio_metrics=metrics,
        equity_curve=equity_curve_df,
        completed_trades=portfolio.trade_logger.get_completed_trades(),
        output_dir=current_run_output_dir,
        strategy_name=args.strategy,
        symbol=primary_symbol_for_naming,
        initial_capital=portfolio.initial_cash,
        drawdown_series=portfolio.drawdown_series,
    )
    report_gen.generate_report(filename_prefix=base_filename_for_output)
    print(f"Backtest report, plots, and trade log saved in: {current_run_output_dir}")


def run_training(args: Namespace):
    """Orchestrates the model training process based on CLI arguments."""
    print("--- Starting Model Training ---")
    trainer = ModelTrainer(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        model_type=args.model_type,
        features=args.features,
        target_lag=args.target_lag,
        output_path=args.output_path,
    )
    trainer.run_training_pipeline()


def run_batch_mode(batch_config_file_path: str, main_parser: argparse.ArgumentParser):
    print(f"--- Starting Batch Processing from: {batch_config_file_path} ---")
    abs_batch_config_path = os.path.abspath(batch_config_file_path)
    base_yaml_dir = os.path.dirname(abs_batch_config_path)

    try:
        with open(abs_batch_config_path, "r") as f:
            batch_configs = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading/parsing YAML '{abs_batch_config_path}': {e}")
        return

    if not isinstance(batch_configs, list):
        print("Error: Batch YAML must be a list of configurations.")
        return

    default_args_ns = main_parser.parse_args([])

    for i, run_cfg_dict in enumerate(batch_configs):
        run_name = run_cfg_dict.get("name", f"BatchRun_{i+1}")
        print(f"\n>>> Starting Batch Item: {run_name} ({i+1}/{len(batch_configs)}) <<<")

        current_run_ns = Namespace(**vars(default_args_ns))
        for key, value in run_cfg_dict.items():
            if hasattr(current_run_ns, key):
                setattr(current_run_ns, key, value)
            elif key != "name":
                print(
                    f"Warning for '{run_name}': Unknown param '{key}' in config. Ignoring."
                )

        if not getattr(current_run_ns, "strategy", None):
            print(f"Error for '{run_name}': 'strategy' missing. Skipping.")
            continue

        if (
            "csv_file_path" in run_cfg_dict
            and current_run_ns.csv_file_path
            and not os.path.isabs(current_run_ns.csv_file_path)
        ):
            current_run_ns.csv_file_path = os.path.normpath(
                os.path.join(base_yaml_dir, current_run_ns.csv_file_path)
            )

        # Set output_dir for this specific batch run
        # If 'output_dir' key exists in YAML for this run, use it (making it absolute).
        # Otherwise, create a subfolder under the main --output-dir (from CLI or its default 'backtest_results').
        if (
            "output_dir" in run_cfg_dict
        ):  # User specified an output_dir for this specific batch item
            current_run_ns.output_dir = os.path.abspath(current_run_ns.output_dir)
        else:  # Create a sub-directory based on run_name under the general output_dir
            current_run_ns.output_dir = os.path.abspath(
                os.path.join(
                    default_args_ns.output_dir,
                    run_name.replace(" ", "_").replace("/", "_"),
                )
            )

        setattr(current_run_ns, "name", run_name)

        try:
            run_backtest(current_run_ns)
            print(f">>> Batch Item '{run_name}' Completed Successfully <<<")
        except Exception as e:
            print(f"!!! Batch Item '{run_name}' Failed: {e} !!!")
            traceback.print_exc()

    print("\n--- Batch Processing Finished ---")


def main():
    parser = create_main_parser()
    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Available commands. If no command is given, a single run is attempted based on provided arguments.",
    )

    # Trainer command
    train_parser = subparsers.add_parser(
        "trainer",
        help="Train a machine learning model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        required=True,
        help="List of symbols to use for training data (e.g., AAPL MSFT).",
    )
    train_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for training data (YYYY-MM-DD).",
    )
    train_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date for training data (YYYY-MM-DD).",
    )
    train_parser.add_argument(
        "--model-type",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "svc", "simple_nn"],
        help="Type of model to train.",
    )
    train_parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["returns", "volatility", "momentum", "rsi", "macd"],
        help="List of features to use for training.",
    )
    train_parser.add_argument(
        "--target-lag",
        type=int,
        default=-1,
        help="The future period to target for prediction (e.g., -1 for next bar).",
    )
    train_parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the trained model file (e.g., models/my_model.joblib).",
    )
    train_parser.set_defaults(func=run_training)

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run multiple backtests from a YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    batch_parser.add_argument(
        "config_file", type=str, help="Path to the batch configuration YAML file."
    )
    batch_parser.set_defaults(
        func=lambda args: run_batch_mode(args.config_file, parser)
    )

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    elif args.command is None:  # Default to single run backtest
        if not args.strategy:
            parser.print_help()
            print(
                "\nError: For a single run, --strategy is required. Or use 'batch <config_file>'."
            )
            return
        missing_args_single_run = []
        if args.data_source.lower() == "csv":
            if not args.symbol:
                missing_args_single_run.append("--symbol (for CSV)")
            if not args.csv_file_path:
                missing_args_single_run.append("--csv-file-path")
        elif args.data_source.lower() == "yahoo":
            if not args.yf_symbols:
                missing_args_single_run.append("--yf-symbols")
            if not args.yf_start_date:
                missing_args_single_run.append("--yf-start-date")
            if not args.yf_end_date:
                missing_args_single_run.append("--yf-end-date")

        if missing_args_single_run:
            parser.error(
                f"Missing required arguments for single run with data source '{args.data_source}': {', '.join(missing_args_single_run)}"
            )

        # For single run, output_dir from args is used directly (or its default 'backtest_results')
        # If user wants a named subfolder for a single run, they can specify output_dir as such.
        # Or we can add a --run-name flag for single runs too. For now, it's simpler.
        if not hasattr(args, "name"):  # Ensure 'name' attribute exists for run_backtest
            run_name = f"{args.symbol or args.synth_symbols[0] if args.data_source=='synthetic' else (args.yf_symbols[0] if args.data_source=='yahoo' else 'DefaultSymbol')}_{args.strategy}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
            setattr(args, "name", run_name)
            # For single runs, make output_dir specific if not already a deep path
            if args.output_dir == "backtest_results":  # Default output_dir
                args.output_dir = os.path.join(args.output_dir, run_name)

        run_backtest(args)
    else:
        parser.print_help()


def main_cli_entry():
    """
    Entry point for the `quantsim` command-line script.
    This function is called when the user types `quantsim` in the terminal
    after the package is installed.
    """
    main()
