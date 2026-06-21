# Contributing to adAI

Thank you for your interest in contributing to adAI! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
5. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Format code: `black src/adAI tests`
5. Check formatting: `flake8 src/adAI tests`
6. Commit and push your changes
7. Submit a pull request

## Code Style

- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Add type hints where possible

## Testing

All pull requests should include tests. Run tests with:

```bash
pytest
```

## Documentation

Update documentation for any new features or changes to the API.

## Questions?

Feel free to open an issue for any questions or suggestions!
