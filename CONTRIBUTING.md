# 🤝 Contributing to QuantSim

First off, **thank you** for considering contributing to QuantSim! 🎉 Your help makes this project better for everyone in the quantitative trading community.

We welcome contributions in all forms:
- 🐛 Bug reports and fixes
- ✨ New features and enhancements  
- 📚 Documentation improvements
- 🧪 Test coverage improvements
- 💡 Ideas and discussions

---

## 🚀 **Quick Start for Contributors**

### **1. Fork & Setup**
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/quantsim.git
cd quantsim

# Set up upstream remote
git remote add upstream https://github.com/yash-tr/quantsim.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .[dev,ml]
```

### **2. Create Your Branch**
```bash
# Always start from the latest main
git checkout main
git pull upstream main

# Create your feature branch
git checkout -b feature/amazing-new-feature
# or: git checkout -b fix/important-bug-fix
```

### **3. Make Your Changes**
- Write your code following our [style guidelines](#coding-style)
- Add tests for new functionality
- Update documentation if needed
- Ensure all tests pass

### **4. Submit Your Contribution**
```bash
# Run tests and quality checks
pytest tests/ -v
black quantsim/ tests/  # Format code
flake8 quantsim/ tests/  # Check style

# Commit with a clear message
git add .
git commit -m "feat: add amazing new feature"

# Push and create PR
git push origin feature/amazing-new-feature
```

---

## 🐛 **Reporting Bugs**

Found a bug? Help us fix it!

1. **🔍 Search existing issues** first: [GitHub Issues](https://github.com/yash-tr/quantsim/issues)
2. **📝 Use our bug report template** when creating a new issue
3. **🔄 Provide reproduction steps** - we can't fix what we can't reproduce!
4. **📋 Include environment details** (OS, Python version, QuantSim version)

**Quick Bug Report Checklist:**
- [ ] Clear title and description
- [ ] Steps to reproduce
- [ ] Expected vs actual behavior
- [ ] Environment information
- [ ] Error logs or screenshots

---

## 💡 **Suggesting Features**

Have an idea to make QuantSim better?

1. **💭 Check existing feature requests** in [GitHub Issues](https://github.com/yash-tr/quantsim/issues)
2. **📝 Use our feature request template**
3. **🎯 Explain the use case** - what problem does this solve?
4. **💻 Provide example usage** if possible

---

## 🛠️ **Code Contributions**

### **Setting Up Development Environment**

```bash
# Clone and setup
git clone https://github.com/yash-tr/quantsim.git
cd quantsim

# Install with all development dependencies
pip install -e .[dev,ml]

# Verify installation
pytest tests/ -v
quantsim --help
```

### **Development Workflow**

1. **🍴 Fork** the repository
2. **🔄 Clone** your fork locally  
3. **🌿 Create** a feature branch
4. **✨ Develop** your changes
5. **🧪 Test** thoroughly
6. **📝 Document** your changes
7. **🚀 Submit** a pull request

### **Code Quality Standards**

We maintain high code quality through:

- **📏 Style**: PEP 8 compliance (enforced by `black` and `flake8`)
- **🧪 Testing**: Maintain/improve test coverage
- **📚 Documentation**: Clear docstrings and comments
- **🔍 Type Hints**: Use type hints for better code clarity
- **⚡ Performance**: Consider performance implications

### **Testing Your Changes**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=quantsim --cov-report=html

# Run specific test file
pytest tests/test_simulation_engine.py -v

# Run tests for your changes only
pytest tests/ -k "test_your_feature" -v
```

### **Code Style Guidelines**

**🎨 Formatting:**
```bash
# Auto-format your code
black quantsim/ tests/

# Check style compliance
flake8 quantsim/ tests/ --max-line-length=88
```

**📝 Docstring Style:**
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe ratio for a return series.
    
    Args:
        returns: Series of portfolio returns
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Sharpe ratio value
        
    Raises:
        ValueError: If returns series is empty
        
    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe Ratio: {sharpe:.3f}")
    """
```

**🏗️ Type Hints:**
```python
from typing import List, Dict, Optional, Union
import pandas as pd

def process_data(
    symbols: List[str],
    start_date: str,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Process market data for given symbols."""
```

---

## 🧪 **Testing Guidelines**

### **Writing Tests**

- **📍 Location**: Mirror the package structure in `tests/`
- **🎯 Coverage**: Aim for >90% coverage for new code
- **🔧 Types**: Unit tests, integration tests, property-based tests
- **📝 Naming**: Descriptive test names explaining what's being tested

**Example Test Structure:**
```python
# tests/strategies/test_my_strategy.py
import pytest
import pandas as pd
from quantsim.strategies.my_strategy import MyStrategy

class TestMyStrategy:
    """Test suite for MyStrategy."""
    
    def test_initialization(self):
        """Test strategy initializes correctly."""
        strategy = MyStrategy(['AAPL'], window=20)
        assert strategy.window == 20
        assert 'AAPL' in strategy.symbols
    
    def test_signal_generation(self):
        """Test strategy generates correct signals."""
        # Test implementation
        pass
    
    @pytest.mark.parametrize("window,expected", [
        (10, True),
        (50, False),
    ])
    def test_different_windows(self, window, expected):
        """Test strategy behavior with different windows."""
        # Parametrized test implementation
        pass
```

### **Test Data**

- Use **synthetic data** for reproducible tests
- Keep test data **small and focused**
- **Mock external dependencies** (Yahoo Finance, etc.)

---

## 📚 **Documentation**

### **Code Documentation**

- **📖 Docstrings**: All public functions, classes, and methods
- **💬 Comments**: Explain complex logic and business rules
- **📋 Type Hints**: Help users understand expected inputs/outputs

### **User Documentation**

- **📓 README updates**: For new features or usage changes
- **📝 Examples**: Add Jupyter notebook examples
- **🔧 Configuration**: Document new configuration options

---

## 🏷️ **Commit Message Guidelines**

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(strategies): add RSI-based mean reversion strategy
fix(portfolio): correct position sizing calculation
docs(readme): update installation instructions
test(execution): add tests for partial fill simulation
```

---

## 🔄 **Pull Request Process**

### **Before Submitting**

- [ ] ✅ All tests pass locally
- [ ] 📏 Code formatted with `black`
- [ ] 🔍 No linting errors (`flake8`)
- [ ] 📚 Documentation updated
- [ ] 🧪 Tests added for new functionality
- [ ] 📝 Clear commit messages

### **PR Guidelines**

1. **📝 Use our PR template**
2. **🔗 Link related issues**
3. **📋 Fill out all sections**
4. **🧪 Ensure CI passes**
5. **💬 Respond to review feedback**

### **Review Process**

- **🔍 Automated checks** must pass (tests, style, security)
- **👥 Code review** by maintainers
- **💬 Discussion** and potential changes
- **✅ Approval** and merge

---

## 🌟 **Recognition**

Contributors are recognized in several ways:

- **📜 Contributors list** in README
- **🏷️ GitHub contributor status**
- **📣 Social media acknowledgment**
- **🎁 Special recognition** for significant contributions

---

## 🆘 **Getting Help**

Need help contributing?

- **💬 GitHub Discussions**: [Start a discussion](https://github.com/yash-tr/quantsim/discussions)
- **🐛 GitHub Issues**: Ask questions with the `question` label
- **📧 Email**: tripathiyash1004@gmail.com

---

## 📋 **Contributor Checklist**

Before your first contribution:

- [ ] 👥 Join our community discussions
- [ ] 📖 Read this contributing guide
- [ ] 🍴 Fork the repository
- [ ] 🔧 Set up development environment
- [ ] 🧪 Run tests to ensure everything works
- [ ] 🌿 Create your first branch
- [ ] ✨ Make a small change to get familiar
- [ ] 🚀 Submit your first PR!

---

## 🎯 **What to Contribute**

Not sure where to start? Look for:

- **🏷️ `good first issue`** labels for beginners
- **🆘 `help wanted`** labels for desired features
- **🐛 Open bugs** that need fixing
- **📚 Documentation** improvements
- **🧪 Test coverage** gaps
- **⚡ Performance** optimizations

---

**Thank you for making QuantSim better! Every contribution, no matter how small, is valuable to our community.** 🙏

**Happy Coding!** 🚀
```