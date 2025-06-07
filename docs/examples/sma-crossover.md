# SMA Crossover Strategy

The Simple Moving Average (SMA) crossover is one of the most popular and straightforward trading strategies. This example demonstrates how to implement and backtest an SMA crossover strategy using QuantSim.

## Strategy Overview

The SMA crossover strategy generates trading signals based on the intersection of two moving averages:

- **Short-term SMA**: Faster-moving average (e.g., 10-day)
- **Long-term SMA**: Slower-moving average (e.g., 30-day)

### Trading Rules

1. **Buy Signal**: When short SMA crosses above long SMA
2. **Sell Signal**: When short SMA crosses below long SMA

## Implementation

### Basic Implementation

```python
import quantsim as qs
import pandas as pd
import matplotlib.pyplot as plt

# Configure the strategy
config = {
    'strategy': 'sma_crossover',
    'data_source': 'yahoo',
    'symbols': ['AAPL'],
    'start_date': '2022-01-01',
    'end_date': '2023-12-31',
    'initial_capital': 100000,
    'short_window': 10,
    'long_window': 30
}

# Create and run the simulation
engine = qs.SimulationEngine(**config)
results = engine.run()

# Display results
print("📊 SMA Crossover Strategy Results")
print("=" * 40)
print(f"Total Return: {results.total_return:.2%}")
print(f"Annual Return: {results.annual_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Total Trades: {results.total_trades}")
print(f"Win Rate: {results.win_rate:.2%}")
```

### Advanced Implementation with Custom Parameters

```python
import quantsim as qs
from quantsim.strategies import SMACrossoverStrategy
from quantsim.core.events import EventQueue

# Create event queue
event_queue = EventQueue()

# Create custom SMA strategy with additional parameters
strategy = SMACrossoverStrategy(
    event_queue=event_queue,
    symbols=['AAPL'],
    short_window=5,      # Faster crossover
    long_window=20,      # Shorter long window
    min_volume=1000000,  # Minimum volume filter
    stop_loss=0.05,      # 5% stop loss
    take_profit=0.15     # 15% take profit
)

# Advanced configuration
engine = qs.SimulationEngine(
    strategy=strategy,
    data_source='yahoo',
    symbols=['AAPL'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=100000,
    commission=0.001,    # 0.1% commission
    slippage=0.0005     # 0.05% slippage
)

results = engine.run()
```

## Parameter Optimization

Find the optimal SMA parameters:

```python
# Parameter optimization
optimization_results = qs.optimize_strategy(
    strategy_class=SMACrossoverStrategy,
    symbols=['AAPL'],
    start_date='2020-01-01',
    end_date='2022-12-31',  # Training period
    parameters={
        'short_window': range(5, 25, 5),
        'long_window': range(20, 100, 10)
    },
    metric='sharpe_ratio',
    n_jobs=4  # Parallel optimization
)

# Display best parameters
best_params = optimization_results.best_params
print(f"Best Parameters: {best_params}")
print(f"Best Sharpe Ratio: {optimization_results.best_score:.3f}")

# Test on out-of-sample data
oos_engine = qs.SimulationEngine(
    strategy='sma_crossover',
    symbols=['AAPL'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    **best_params
)

oos_results = oos_engine.run()
print(f"Out-of-sample Sharpe: {oos_results.sharpe_ratio:.3f}")
```

## Multi-Asset Implementation

Apply the strategy to multiple assets:

```python
# Multi-asset SMA crossover
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

engine = qs.SimulationEngine(
    strategy='sma_crossover',
    data_source='yahoo',
    symbols=symbols,
    start_date='2022-01-01',
    end_date='2023-12-31',
    initial_capital=500000,  # Larger capital for multiple assets
    short_window=10,
    long_window=30,
    position_sizing='equal_weight'  # Equal weight allocation
)

results = engine.run()

# Analyze per-symbol performance
for symbol in symbols:
    symbol_results = results.get_symbol_results(symbol)
    print(f"{symbol}: {symbol_results.total_return:.2%} return")
```

## Visualization and Analysis

### Equity Curve

```python
# Plot equity curve
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Equity curve
results.plot_equity_curve(ax=ax1)
ax1.set_title('Portfolio Equity Curve')
ax1.grid(True, alpha=0.3)

# Drawdown
results.plot_drawdown(ax=ax2)
ax2.set_title('Drawdown')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Signal Analysis

```python
# Analyze trading signals
signals_df = results.get_signals_dataframe()

print("\n📈 Signal Analysis")
print("-" * 30)
print(f"Total Buy Signals: {len(signals_df[signals_df['signal'] == 'BUY'])}")
print(f"Total Sell Signals: {len(signals_df[signals_df['signal'] == 'SELL'])}")

# Plot price with signals
plt.figure(figsize=(14, 8))
price_data = results.get_price_data('AAPL')

plt.plot(price_data.index, price_data['close'], label='AAPL Price', alpha=0.7)
plt.plot(price_data.index, price_data['sma_short'], label='SMA 10', alpha=0.8)
plt.plot(price_data.index, price_data['sma_long'], label='SMA 30', alpha=0.8)

# Mark buy/sell signals
buy_signals = signals_df[signals_df['signal'] == 'BUY']
sell_signals = signals_df[signals_df['signal'] == 'SELL']

plt.scatter(buy_signals.index, buy_signals['price'], 
           color='green', marker='^', s=100, label='Buy Signal')
plt.scatter(sell_signals.index, sell_signals['price'], 
           color='red', marker='v', s=100, label='Sell Signal')

plt.title('SMA Crossover Strategy - AAPL')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Performance Metrics

Detailed performance analysis:

```python
# Comprehensive performance metrics
metrics = results.calculate_metrics()

print("\n📊 Performance Metrics")
print("=" * 50)
print(f"Total Return:        {metrics['total_return']:.2%}")
print(f"Annual Return:       {metrics['annual_return']:.2%}")
print(f"Volatility:          {metrics['volatility']:.2%}")
print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
print(f"Sortino Ratio:       {metrics['sortino_ratio']:.3f}")
print(f"Calmar Ratio:        {metrics['calmar_ratio']:.3f}")
print(f"Max Drawdown:        {metrics['max_drawdown']:.2%}")
print(f"Max Drawdown Days:   {metrics['max_dd_duration']} days")
print(f"VaR (95%):          {metrics['var_95']:.2%}")
print(f"CVaR (95%):         {metrics['cvar_95']:.2%}")

print(f"\n📈 Trading Statistics")
print("-" * 30)
print(f"Total Trades:        {metrics['total_trades']}")
print(f"Win Rate:            {metrics['win_rate']:.2%}")
print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
print(f"Avg Trade Return:    {metrics['avg_trade_return']:.2%}")
print(f"Avg Win:             {metrics['avg_win']:.2%}")
print(f"Avg Loss:            {metrics['avg_loss']:.2%}")
print(f"Best Trade:          {metrics['best_trade']:.2%}")
print(f"Worst Trade:         {metrics['worst_trade']:.2%}")
```

## Risk Management Enhancements

Add risk management to the strategy:

```python
from quantsim.risk import RiskManager

# Create risk manager
risk_manager = RiskManager(
    max_position_size=0.1,      # Max 10% per position
    max_portfolio_risk=0.15,    # Max 15% portfolio risk
    stop_loss=0.05,             # 5% stop loss
    take_profit=0.20,           # 20% take profit
    max_drawdown=0.10           # Max 10% drawdown
)

# Run strategy with risk management
engine = qs.SimulationEngine(
    strategy='sma_crossover',
    symbols=['AAPL'],
    start_date='2022-01-01',
    end_date='2023-12-31',
    initial_capital=100000,
    short_window=10,
    long_window=30,
    risk_manager=risk_manager
)

results = engine.run()
```

## Common Variations

### 1. Triple SMA Crossover

```python
# Three moving averages for additional confirmation
engine = qs.SimulationEngine(
    strategy='triple_sma_crossover',
    symbols=['SPY'],
    fast_window=5,
    medium_window=15,
    slow_window=30
)
```

### 2. Exponential Moving Average

```python
# Use EMA instead of SMA for more responsive signals
engine = qs.SimulationEngine(
    strategy='ema_crossover',
    symbols=['QQQ'],
    short_window=12,
    long_window=26
)
```

### 3. Volume-Weighted Crossover

```python
# Include volume confirmation
engine = qs.SimulationEngine(
    strategy='volume_weighted_sma',
    symbols=['SPY'],
    short_window=10,
    long_window=30,
    volume_threshold=1.5  # 1.5x average volume
)
```

## Best Practices

!!! tip "Optimization Tips"
    - Use out-of-sample testing to validate parameters
    - Consider transaction costs and slippage
    - Test across different market regimes
    - Combine with other indicators for confirmation

!!! warning "Common Pitfalls"
    - Over-optimization on historical data
    - Ignoring transaction costs
    - Not accounting for market regime changes
    - Using too short or too long moving averages

## Next Steps

Ready to explore more? Here are some suggestions:

1. **[Quick Start Guide](../getting-started/quickstart.md)** - Learn more QuantSim basics
2. **[Installation Guide](../getting-started/installation.md)** - Advanced installation options
3. **[Contributing](../development/contributing.md)** - Help improve QuantSim
4. **[GitHub Repository](https://github.com/yash-tr/quantsim)** - Explore the source code 