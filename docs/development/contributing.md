# Contributing to QuantSim

Thank you for your interest in contributing to QuantSim! 🎉

This page provides a quick overview of how to contribute. For the complete contributing guide, please see our main [CONTRIBUTING.md](https://github.com/yash-tr/quantsim/blob/main/CONTRIBUTING.md) file.

## Quick Start for Contributors

### 1. Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/quantsim.git
cd quantsim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev,ml,pairs]
```

### 2. Make Your Changes

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and add tests
# ... your development work ...

# Run tests
pytest tests/ -v

# Format code
black quantsim/ tests/
```

### 3. Submit Pull Request

```bash
# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push to your fork
git push origin feature/your-feature-name

# Create PR on GitHub
```

## What Can You Contribute?

- 🐛 **Bug fixes** - Help us squash bugs
- ✨ **New features** - Add new strategies, indicators, or functionality
- 📚 **Documentation** - Improve guides, examples, and API docs
- 🧪 **Tests** - Increase test coverage
- 🎨 **Examples** - Add new strategy examples or notebooks
- 🔧 **Performance** - Optimize existing code

## Development Guidelines

- Write tests for new functionality
- Follow PEP 8 style guidelines
- Add docstrings to public methods
- Update documentation when needed
- Ensure all tests pass before submitting

## Getting Help

- 💬 **[GitHub Discussions](https://github.com/yash-tr/quantsim/discussions)** - Ask questions
- 🐛 **[GitHub Issues](https://github.com/yash-tr/quantsim/issues)** - Report bugs
- 📧 **Email**: tripathiyash1004@gmail.com

## Recognition

All contributors are recognized in our [contributors list](https://github.com/yash-tr/quantsim/graphs/contributors) and in project documentation.

---

**Ready to contribute?** Check out our [good first issue](https://github.com/yash-tr/quantsim/labels/good%20first%20issue) labels to get started!

For the complete contributing guide with detailed instructions, coding standards, and workflow details, see: **[CONTRIBUTING.md](https://github.com/yash-tr/quantsim/blob/main/CONTRIBUTING.md)** 