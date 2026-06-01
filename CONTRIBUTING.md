# Contributing to Betting Intelligence

First off, thanks for taking the time to contribute! 

## Code of Conduct

This project is committed to providing a welcoming environment. Be respectful, constructive, and professional.

## How to Contribute

### 1. Reporting Issues

- Check existing issues before creating a new one
- Use a clear, descriptive title
- Include steps to reproduce, expected behavior, and actual behavior
- Include Python version, OS, and package versions (`pip list`)

### 2. Suggesting Features

- Open an issue with the label `enhancement`
- Explain why the feature would be useful
- Include examples of how it would work

### 3. Pull Requests

1. **Fork** the repo and create your branch from `main`
2. **Install** dev dependencies (see platform-specific instructions below)
3. **Make your changes** — keep them focused and minimal
4. **Write tests** for any new functionality
5. **Run tests** (see testing section)
6. **Lint**: `make lint` (Linux/macOS) or `ruff check src/ tests/` (Windows)
7. **Commit** with a clear message
8. **Push** to your fork and open a PR

### PR Guidelines

- One feature per PR
- Keep changes small and focused
- Update documentation if needed
- Add tests for new functionality
- Ensure all tests pass before submitting
- Reference the issue number if applicable

## Development Setup

### Linux / macOS

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/betting-intelligence.git
cd betting-intelligence

# Install with dev and dashboard dependencies
make dev

# Copy and configure environment
cp .env.example .env

# Run tests
make test

# Start the dashboard
make run-dashboard
```

### Windows

```powershell
# Clone the repo
git clone https://github.com/YOUR_USERNAME/betting-intelligence.git
cd betting-intelligence

# Run the Windows bootstrap installer (recommended)
powershell -ExecutionPolicy Bypass -File install.ps1

# Or manually:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"

# Copy and configure environment
Copy-Item .env.example .env

# Initialize database
python scripts/init_db.py

# Run tests
pytest -v -m "not slow"

# Start the dashboard
streamlit run dashboard/app.py
```

> **Windows Requirements:**
> - Python 3.10+ (add to PATH during installation)
> - Some packages need a C compiler. If you see build errors, install
>   [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> - PowerShell 5.1+ (included with Windows 10/11)

## Project Structure

```
betting-intelligence/
├── src/betting_intel/       # Main package
│   ├── betting/             # Betting simulation & edge detection
│   ├── cli/                 # CLI commands (Click)
│   ├── config/              # Configuration & settings
│   ├── data/                # Data loading, features, small leagues
│   ├── models/              # ML models (XGBoost, Linear, Ensemble)
│   └── recommendations/     # Bet recommendation engine
├── dashboard/               # Streamlit dashboard
├── tests/                   # Test suite
├── data/                    # Data directory (gitignored)
├── output/                  # Output directory (gitignored)
└── docs/                    # Documentation
```

## Testing

### Linux / macOS

```bash
# Run all tests with coverage
make test

# Run fast tests only
make test-fast
```

### Windows

```powershell
# Run all tests with coverage
pytest -v --cov=src/betting_intel --cov-report=term-missing

# Run fast tests only
pytest -v -m "not slow"
```

## Code Style

- Follow PEP 8
- Use type hints everywhere
- Write docstrings for public APIs
- Use `ruff` for linting and formatting

### Running Linters

```bash
# Linux / macOS
make lint
make format

# Windows
ruff check src/ tests/
ruff format --check src/ tests/
ruff format src/ tests/   # auto-fix formatting
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
