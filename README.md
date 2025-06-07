# 🚀 QuantSim: Professional Event-Driven Backtesting Framework

[![Tests](https://github.com/yash-tr/quantsim/workflows/Test%20Suite/badge.svg)](https://github.com/yash-tr/quantsim/actions)
[![PyPI](https://img.shields.io/pypi/v/quantsim.svg)](https://pypi.org/project/quantsim/)
[![Python](https://img.shields.io/pypi/pyversions/quantsim.svg)](https://pypi.org/project/quantsim/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](#testing)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://yash-tr.github.io/quantsim/)

**QuantSim** is a modern, event-driven backtesting framework for quantitative trading strategies. Built with Python 3.9+, it provides institutional-grade simulation capabilities with a focus on performance, accuracy, and extensibility.

## ✨ **Why QuantSim?**

- 🏗️ **Event-Driven Architecture**: Realistic simulation that processes market events chronologically
- 📊 **Multiple Data Sources**: Yahoo Finance, CSV files, synthetic data generation
- ⚡ **High Performance**: Optimized for speed with comprehensive caching and vectorized operations
- 🧪 **Battle-Tested**: 178 unit tests with 95%+ coverage ensuring reliability
- 🔧 **Highly Extensible**: Plugin architecture for strategies, indicators, and execution models
- 📈 **Professional Reporting**: Rich markdown reports with equity curves and performance metrics
- 🤖 **ML Integration**: Optional machine learning components for advanced strategies
- 🛡️ **Production Ready**: Comprehensive error handling, logging, and validation

---

## 📦 **Quick Installation**

### From PyPI (Recommended)
```bash
# Core package
pip install quantsim

# With ML capabilities
pip install quantsim[ml]

# With pairs trading (requires statsmodels)
pip install quantsim[pairs]

# Full installation with all features
pip install quantsim[ml,pairs]
```

### For Development
```bash
git clone https://github.com/yash-tr/quantsim.git
cd quantsim
pip install -e .[dev]
```

---

## 🚀 **Quick Start**

### 1. Simple Strategy Backtest
```python
import quantsim as qs

# Create and run a simple SMA crossover strategy
engine = qs.SimulationEngine(
    data_source='yahoo',
    symbols=['AAPL'],
    start_date='2022-01-01',
    end_date='2023-01-01',
    strategy='sma_crossover',
    initial_capital=100000
)

results = engine.run()
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

### 2. Command Line Interface
```bash
# Run a single backtest
quantsim run --strategy sma_crossover --symbol AAPL --start 2022-01-01 --end 2023-01-01

# Batch backtesting from YAML config
quantsim batch my_strategies.yaml

# Get help
quantsim --help
```

### 3. Custom Strategy Development
```python
from quantsim.strategies import Strategy
from quantsim.core.events import OrderEvent

class MyStrategy(Strategy):
    def __init__(self, symbols, **kwargs):
        super().__init__(symbols, **kwargs)
        self.window = kwargs.get('window', 20)
    
    def on_market_event(self, event):
        # Your strategy logic here
        if self.should_buy(event.symbol):
            order = OrderEvent(
                symbol=event.symbol,
                order_type='MKT',
                quantity=100,
                direction='BUY'
            )
            self.event_queue.put(order)
    
    def should_buy(self, symbol):
        # Implement your buy logic
        return True
```

---

## 🏗️ **Core Features**

### **Event-Driven Simulation Engine**
- **Realistic Order Processing**: Market, limit, and stop orders with configurable slippage
- **Portfolio Management**: Real-time P&L tracking, risk metrics, and position management
- **Execution Simulation**: Latency modeling, partial fills, and commission structures

### **Built-in Strategies**
- **SMA Crossover**: Moving average crossover with customizable windows
- **Momentum**: Trend-following strategy with momentum indicators
- **Mean Reversion**: Statistical arbitrage based on price deviations
- **Pairs Trading**: Cointegration-based pairs trading (requires `statsmodels`)
- **ML Strategies**: Integration with scikit-learn and TensorFlow

### **Data Sources**
- **Yahoo Finance**: Automatic data fetching with symbol validation
- **CSV Files**: Flexible parser supporting multiple formats
- **Synthetic Data**: Configurable data generation for testing

### **Advanced Analytics**
- **Performance Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, VaR
- **Trade Analysis**: Win rate, profit factor, average trade duration
- **Risk Metrics**: Beta, alpha, tracking error, information ratio
- **Visualizations**: Equity curves, drawdown plots, trade distributions

---

## 📊 **Performance Metrics**

QuantSim calculates comprehensive performance metrics:

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative return over the backtest period |
| **CAGR** | Compound Annual Growth Rate |
| **Sharpe Ratio** | Risk-adjusted return measure |
| **Sortino Ratio** | Downside deviation-adjusted returns |
| **Maximum Drawdown** | Largest peak-to-trough decline |
| **Calmar Ratio** | CAGR divided by maximum drawdown |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Ratio of gross profits to gross losses |

---

## 🛠️ **Architecture Overview**

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│  Data Handler   │───▶│ Event Queue  │───▶│  Strategy   │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │                      │
                              ▼                      ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Portfolio     │◀───│   Engine     │◀───│   Orders    │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │
                              ▼
                    ┌──────────────┐
                    │  Execution   │
                    │   Handler    │
                    └──────────────┘
```

---

## 📈 **Example Strategies**

### SMA Crossover Strategy
```python
from quantsim import SMACrossoverStrategy

strategy = SMACrossoverStrategy(
    symbols=['AAPL', 'GOOGL'],
    short_window=10,
    long_window=30,
    event_queue=event_queue
)
```

### Custom ML Strategy
```python
from quantsim.strategies.ml import MLStrategy
from sklearn.ensemble import RandomForestClassifier

strategy = MLStrategy(
    symbols=['SPY'],
    model=RandomForestClassifier(),
    features=['sma_10', 'rsi_14', 'macd'],
    lookback_window=60
)
```

---

## 🧪 **Testing & Quality**

QuantSim maintains high code quality standards:

- **178 Unit Tests** with 95%+ coverage
- **Multi-platform Testing** (Ubuntu, Windows, macOS)
- **Python 3.9+ Support** across versions
- **Automated CI/CD** with GitHub Actions
- **Code Quality Checks** (Black, Flake8, MyPy)
- **Security Scanning** (Bandit, Safety)

```bash
# Run tests locally
pytest tests/ -v --cov=quantsim

# Generate coverage report
pytest --cov=quantsim --cov-report=html
```

---

## 📚 **Documentation & Examples**

### **Available Resources**
- 📖 **[API Documentation](https://yash-tr.github.io/quantsim/)** - Complete API reference
- 📓 **[Jupyter Notebooks](notebooks/)** - Interactive examples and tutorials
- 🔧 **[Configuration Guide](PYPI_GITHUB_SETUP.md)** - Setup and configuration
- 🤝 **[Contributing Guide](CONTRIBUTING.md)** - How to contribute

### **Example Configurations**
```yaml
# sample_batch_config.yaml
strategies:
  - name: "SPY_SMA_Crossover"
    strategy: "sma_crossover"
    data_source: "yahoo"
    symbols: ["SPY"]
    start_date: "2020-01-01"
    end_date: "2023-01-01"
    short_window: 10
    long_window: 30
    initial_capital: 100000
```

---

## 🚀 **Advanced Usage**

### Batch Processing
```bash
# Run multiple strategies from YAML config
quantsim batch strategies.yaml --output-dir results/

# Parallel execution
quantsim batch strategies.yaml --parallel --workers 4
```

### Custom Indicators
```python
from quantsim.indicators import Indicator

class RSI(Indicator):
    def __init__(self, period=14):
        self.period = period
    
    def calculate(self, prices):
        # RSI calculation logic
        return rsi_values
```

### Risk Management
```python
from quantsim.risk import RiskManager

risk_manager = RiskManager(
    max_position_size=0.1,  # 10% max position
    max_drawdown=0.05,      # 5% max drawdown
    var_limit=0.02          # 2% VaR limit
)
```

---

## 🌟 **Competitive Advantages**

| Feature | QuantSim | Zipline | Backtrader | FreqTrade |
|---------|----------|---------|------------|-----------|
| **Modern Python** | ✅ 3.9+ | ❌ 3.6+ | ✅ 3.7+ | ✅ 3.8+ |
| **Event-Driven** | ✅ | ✅ | ❌ | ✅ |
| **ML Integration** | ✅ | ❌ | ❌ | ✅ |
| **Multi-Asset** | ✅ | ✅ | ✅ | ❌ |
| **Real-time Ready** | ✅ | ❌ | ✅ | ✅ |
| **Professional Reports** | ✅ | ❌ | ❌ | ✅ |

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Quick Contribution Steps**
1. 🍴 Fork the repository
2. 🔄 Clone your fork: `git clone https://github.com/yourusername/quantsim.git`
3. 🌿 Create a branch: `git checkout -b feature/amazing-feature`
4. ✨ Make your changes and add tests
5. ✅ Run tests: `pytest tests/`
6. 📝 Commit: `git commit -m "Add amazing feature"`
7. 🚀 Push: `git push origin feature/amazing-feature`
8. 🔄 Create a Pull Request

### **Development Setup**
```bash
git clone https://github.com/yash-tr/quantsim.git
cd quantsim
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .[dev]
pre-commit install  # Optional: setup pre-commit hooks
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Financial Data**: Powered by [yfinance](https://github.com/ranaroussi/yfinance)
- **Numerical Computing**: Built on [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/)
- **Visualization**: Charts generated with [Matplotlib](https://matplotlib.org/)
- **Testing**: Quality assured with [pytest](https://pytest.org/)

---

## 📞 **Support & Community**

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yash-tr/quantsim/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yash-tr/quantsim/discussions)
- 📧 **Email**: tripathiyash1004@gmail.com
- 💬 **Community**: Join our discussions for tips, strategies, and support

---

## 🗺️ **Roadmap**

### **Version 0.2.0** (Coming Soon)
- 🔄 Real-time trading integration
- 📊 Advanced portfolio optimization
- 🌐 WebSocket data feeds
- 📱 Interactive dashboard

### **Version 0.3.0** (Future)
- 🤖 AutoML strategy generation
- ☁️ Cloud deployment options
- 📈 Options and derivatives support
- 🔗 Crypto exchange integration

---

**Ready to transform your trading strategies? Install QuantSim today and start building professional backtests in minutes!**

```bash
pip install quantsim
```

⭐ **Star us on GitHub if QuantSim helps your trading!** ⭐