# Quick Start Guide

Get up and running with QuantSim in under 10 minutes! This guide will walk you through creating and running your first quantitative trading strategy.

## Prerequisites

Make sure you have QuantSim installed:

```bash
pip install quantsim
```

## Your First Strategy

Let's create a simple SMA (Simple Moving Average) crossover strategy that buys when the short-term average crosses above the long-term average.

### Step 1: Basic Setup

```python
import quantsim as qs
from datetime import datetime

# Create a simulation engine
engine = qs.SimulationEngine(
    data_source='yahoo',           # Use Yahoo Finance data
    symbols=['AAPL'],              # Trade Apple stock
    start_date='2022-01-01',       # Start date
    end_date='2023-01-01',         # End date
    strategy='sma_crossover',      # Strategy type
    initial_capital=100000,        # Starting with $100k
    short_window=10,               # 10-day moving average
    long_window=30                 # 30-day moving average
)

# Run the backtest
results = engine.run()

# Display results
print(f"📊 Backtest Results for AAPL")
print(f"💰 Total Return: {results.total_return:.2%}")
print(f"📈 Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"📉 Max Drawdown: {results.max_drawdown:.2%}")
print(f"🎯 Win Rate: {results.win_rate:.2%}")
```

### Step 2: Run and Analyze

When you run this code, you'll see output like:

```
📊 Backtest Results for AAPL
💰 Total Return: 15.23%
📈 Sharpe Ratio: 1.24
📉 Max Drawdown: -8.45%
🎯 Win Rate: 62.50%
```

### Step 3: Generate Reports

Create detailed reports with charts:

```python
# Generate comprehensive report
report = engine.generate_report()
report.save('my_first_backtest.md')

# Create equity curve chart
chart = engine.plot_equity_curve()
chart.show()
```

## Command Line Usage

You can also run strategies from the command line:

```bash
# Basic backtest
quantsim run --strategy sma_crossover --symbol AAPL --start 2022-01-01 --end 2023-01-01

# Multiple symbols
quantsim run --strategy momentum --symbols AAPL GOOGL MSFT --start 2022-01-01

# Custom parameters
quantsim run --strategy sma_crossover --symbol SPY --short-window 5 --long-window 20
```

## Configuration Files

For complex setups, use YAML configuration:

=== "strategy_config.yaml"
    ```yaml
    name: "My SMA Strategy"
    strategy: "sma_crossover"
    data_source: "yahoo"
    symbols: ["AAPL", "GOOGL", "MSFT"]
    start_date: "2022-01-01"
    end_date: "2023-01-01"
    initial_capital: 100000
    
    parameters:
      short_window: 10
      long_window: 30
      
    risk_management:
      max_position_size: 0.2
      stop_loss: 0.05
    ```

=== "Run Configuration"
    ```bash
    quantsim batch strategy_config.yaml
    ```

## Multiple Strategies Comparison

Compare different strategies easily:

```python
import quantsim as qs

# Define strategies to compare
strategies = [
    {
        'name': 'SMA Crossover',
        'strategy': 'sma_crossover',
        'short_window': 10,
        'long_window': 30
    },
    {
        'name': 'Momentum',
        'strategy': 'momentum',
        'lookback': 20,
        'threshold': 0.02
    },
    {
        'name': 'Mean Reversion',
        'strategy': 'mean_reversion',
        'window': 20,
        'z_threshold': 2.0
    }
]

# Run comparison
comparison = qs.compare_strategies(
    strategies=strategies,
    symbols=['AAPL'],
    start_date='2022-01-01',
    end_date='2023-01-01',
    initial_capital=100000
)

# Display comparison table
print(comparison.summary_table())
```

## Real-Time Data Integration

For live trading simulation:

```python
# Connect to real-time data (paper trading)
engine = qs.SimulationEngine(
    data_source='live',  # Real-time data
    symbols=['AAPL'],
    strategy='sma_crossover',
    mode='paper',        # Paper trading mode
    initial_capital=100000
)

# Start live simulation
engine.run_live()
```

## Next Steps

Now that you've run your first strategy, explore more advanced features:

### 🚀 Immediate Next Steps
1. **[Try Different Strategies](../examples/momentum.md)** - Explore momentum and mean reversion
2. **[Add Multiple Assets](../user-guide/portfolio.md)** - Build a diversified portfolio  
3. **[Custom Risk Management](../user-guide/risk-management.md)** - Add stop losses and position sizing

### 📚 Learn More
4. **[Understanding Concepts](concepts.md)** - Core framework concepts
5. **[Data Sources](../user-guide/data-sources.md)** - Work with different data types
6. **[Performance Metrics](../user-guide/metrics.md)** - Understand the results

### 🛠️ Advanced Topics
7. **[Create Custom Strategies](../examples/custom-strategy.md)** - Build your own trading logic
8. **[ML Integration](../examples/ml-integration.md)** - Add machine learning
9. **[API Reference](../api/core.md)** - Dive deep into the code

## Common Patterns

Here are some common usage patterns to get you started:

### Pattern 1: Portfolio Backtesting
```python
# Multi-asset portfolio
engine = qs.SimulationEngine(
    symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
    strategy='equal_weight_rebalancing',
    rebalance_frequency='monthly'
)
```

### Pattern 2: Parameter Optimization
```python
# Test different parameter combinations
results = qs.optimize_parameters(
    strategy='sma_crossover',
    symbol='AAPL',
    parameters={
        'short_window': [5, 10, 15, 20],
        'long_window': [20, 30, 40, 50]
    }
)
```

### Pattern 3: Risk-Adjusted Backtesting
```python
# Add risk management
engine = qs.SimulationEngine(
    strategy='momentum',
    risk_manager=qs.RiskManager(
        max_drawdown=0.10,      # Max 10% drawdown
        position_size=0.05,     # Max 5% per position
        var_limit=0.02          # 2% Value at Risk
    )
)
```

!!! tip "Pro Tips"
    - Start with simple strategies and gradually add complexity
    - Always validate results with out-of-sample testing
    - Use paper trading before live implementation
    - Monitor performance metrics beyond just returns

Ready to build more sophisticated strategies? Check out our [examples](../examples/sma-crossover.md) section! 