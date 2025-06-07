# Installation Guide

Get QuantSim up and running on your system in just a few minutes.

## Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 4GB RAM recommended for backtesting
- **Storage**: 1GB free space for data and results

## Installation Options

### Option 1: Core Package (Recommended)

For most users, the core package provides all essential features:

```bash
pip install quantsim
```

This includes:
- Event-driven backtesting engine
- Built-in strategies (SMA, Momentum, Mean Reversion)
- Yahoo Finance data integration
- CSV data support
- Performance analytics and reporting

### Option 2: With Machine Learning

If you plan to use ML-enhanced strategies:

```bash
pip install quantsim[ml]
```

Additional features:
- Scikit-learn integration
- TensorFlow support
- ML-based strategy examples
- Feature engineering utilities

### Option 3: With Pairs Trading

For statistical arbitrage and pairs trading:

```bash
pip install quantsim[pairs]
```

Additional features:
- Cointegration testing
- Pairs trading strategies
- Statistical analysis tools

### Option 4: Full Installation

Get all features in one package:

```bash
pip install quantsim[ml,pairs]
```

### Option 5: Development Installation

For contributors or advanced users:

```bash
git clone https://github.com/yash-tr/quantsim.git
cd quantsim
pip install -e .[dev,ml,pairs]
```

## Verification

Verify your installation:

=== "Python"
    ```python
    import quantsim
    print(f"QuantSim version: {quantsim.__version__}")
    
    # Test basic functionality
    engine = quantsim.SimulationEngine()
    print("✅ QuantSim installed successfully!")
    ```

=== "Command Line"
    ```bash
    quantsim --version
    quantsim --help
    ```

## Troubleshooting

### Common Issues

!!! warning "ImportError: No module named 'statsmodels'"
    If you're using pairs trading strategies, install the pairs extra:
    ```bash
    pip install quantsim[pairs]
    ```

!!! warning "Permission denied errors"
    On some systems, use `--user` flag:
    ```bash
    pip install --user quantsim
    ```

!!! warning "SSL Certificate errors"
    For corporate networks:
    ```bash
    pip install --trusted-host pypi.org --trusted-host pypi.python.org quantsim
    ```

### Platform-Specific Notes

=== "Windows"
    - Ensure Python is added to PATH
    - Consider using Anaconda for easier package management
    - Visual C++ build tools may be required for some dependencies

=== "macOS"
    - Xcode command line tools may be required
    - Consider using Homebrew to install Python
    ```bash
    brew install python
    pip install quantsim
    ```

=== "Linux"
    - Most distributions work out of the box
    - Ensure `python3-dev` is installed for source builds
    ```bash
    sudo apt-get install python3-dev  # Ubuntu/Debian
    sudo yum install python3-devel    # CentOS/RHEL
    ```

## Updating

Keep QuantSim up to date:

```bash
pip install --upgrade quantsim
```

## What's Next?

Once installed, continue with:

1. **[Quick Start Guide](quickstart.md)** - Build your first strategy
2. **[SMA Crossover Example](../examples/sma-crossover.md)** - See a real strategy in action
3. **[Contributing](../development/contributing.md)** - Help improve QuantSim 