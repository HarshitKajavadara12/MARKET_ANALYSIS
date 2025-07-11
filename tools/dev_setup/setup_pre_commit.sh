#!/bin/bash

# Market Research System v1.0 - Pre-commit Hooks Setup
# Created: January 2022
# Purpose: Ensure code quality and consistency

set -e

echo "🔗 Setting up Pre-commit Hooks for Market Research System"
echo "========================================================"

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
fi

# Install pre-commit hooks
echo "⚙️ Installing pre-commit hooks..."
pre-commit install

# Install commit message hook
pre-commit install --hook-type commit-msg

# Create .pre-commit-config.yaml if it doesn't exist
if [ ! -f ".pre-commit-config.yaml" ]; then
    echo "📝 Creating .pre-commit-config.yaml..."
    cat > .pre-commit-config.yaml << 'EOF'
# Market Research System - Pre-commit Configuration
# See https://pre-commit.com for more information

repos:
  # Code formatting
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.8
        args: [--line-length=88]

  # Import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: [--profile=black]

  # Linting
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  # Security checks
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.4
    hooks:
      - id: bandit
        args: [-r, ., -f, json, -o, reports/security_report.json]
        exclude: ^tests/

  # Dockerfile linting
  - repo: https://github.com/hadolint/hadolint
    rev: v2.8.0
    hooks:
      - id: hadolint-docker

  # YAML formatting
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.1.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements
      - id: requirements-txt-fixer

  # Python docstring formatting
  - repo: https://github.com/pycqa/pydocstyle
    rev: 6.1.1
    hooks:
      - id: pydocstyle
        args: [--convention=google]

  # Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.931
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]

  # Jupyter notebook formatting
  - repo: https://github.com/nbQA-dev/nbQA
    rev: 1.3.1
    hooks:
      - id: nbqa-black
        additional_dependencies: [black==22.3.0]
      - id: nbqa-isort
        additional_dependencies: [isort==5.10.1]

  # Shell script linting
  - repo: https://github.com/shellcheck-py/shellcheck-py
    rev: v0.8.0.4
    hooks:
      - id: shellcheck

  # SQL formatting
  - repo: https://github.com/sqlfluff/sqlfluff
    rev: 0.12.0
    hooks:
      - id: sqlfluff-lint
        args: [--dialect=sqlite]
      - id: sqlfluff-fix
        args: [--dialect=sqlite]

  # Documentation checks
  - repo: https://github.com/pycqa/doc8
    rev: 0.10.1
    hooks:
      - id: doc8
        args: [--max-line-length=88]

  # Commit message formatting
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v2.21.2
    hooks:
      - id: commitizen
        stages: [commit-msg]

# Configuration for specific tools
exclude: |
  (?x)^(
      data/.*|
      logs/.*|
      reports/.*|
      venv/.*|
      \.git/.*|
      \.pytest_cache/.*|
      __pycache__/.*
  )$
EOF
    echo "✅ .pre-commit-config.yaml created"
fi

# Create .flake8 configuration if it doesn't exist
if [ ! -f ".flake8" ]; then
    echo "📝 Creating .flake8 configuration..."
    cat > .flake8 << 'EOF'
[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501
exclude = 
    .git,
    __pycache__,
    venv,
    data,
    logs,
    reports,
    .pytest_cache
per-file-ignores =
    __init__.py:F401
    tests/*:S101
max-complexity = 10
EOF
    echo "✅ .flake8 configuration created"
fi

# Create .pylintrc if it doesn't exist
if [ ! -f ".pylintrc" ]; then
    echo "📝 Creating .pylintrc configuration..."
    cat > .pylintrc << 'EOF'
[MASTER]
extension-pkg-whitelist=numpy,pandas,yfinance,talib

[FORMAT]
max-line-length=88

[MESSAGES CONTROL]
disable=C0114,C0115,C0116,R0903,R0913,W0613,C0103

[DESIGN]
max-args=10
max-locals=25
max-returns=10
max-branches=15
max-statements=60

[SIMILARITIES]
min-similarity-lines=6
ignore-comments=yes
ignore-docstrings=yes
EOF
    echo "✅ .pylintrc configuration created"
fi

# Create pyproject.toml for black and isort if it doesn't exist
if [ ! -f "pyproject.toml" ]; then
    echo "📝 Creating pyproject.toml configuration..."
    cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "market-research-system"
version = "1.0.0"
description = "Independent Market Research System for Indian Stock Market"
authors = [
    {name = "Independent Market Researcher", email = "researcher@example.com"}
]
readme = "README.md"
license = {text = "MIT"}
keywords = ["finance", "stock market", "technical analysis", "india"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Office/Business :: Financial :: Investment",
]
requires-python = ">=3.8"
dependencies = [
    "pandas>=1.4.0",
    "numpy>=1.21.0",
    "yfinance>=0.1.87",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.6.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.8.0",
    "requests>=2.27.0",
    "beautifulsoup4>=4.10.0",
    "python-dotenv>=0.19.0",
    "pyyaml>=6.0",
    "sqlalchemy>=1.4.0",
    "schedule>=1.1.0",
    "reportlab>=3.6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.3.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.931",
    "pre-commit>=2.17.0",
    "bandit>=1.7.0",
    "safety>=1.10.0",
]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | venv
  | _build
  | buck-out
  | build
  | dist
  | data
  | logs
  | reports
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 88
skip_glob = ["data/*", "logs/*", "reports/*", "venv/*"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true
exclude = [
    "data/",
    "logs/",
    "reports/",
    "venv/",
]

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --cov=src --cov-report=html --cov-report=term-missing"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "setup.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
EOF
    echo "✅ pyproject.toml configuration created"
fi

# Run pre-commit on all files to test setup
echo "🧪 Testing pre-commit setup on sample files..."
pre-commit run --all-files || echo "⚠️ Some pre-commit checks failed - this is normal for initial setup"

echo ""
echo "✅ Pre-commit hooks setup completed!"
echo "======================================="
echo ""
echo "📋 What was installed:"
echo "  • Code formatting (Black, isort)"
echo "  • Linting (flake8, pylint)"
echo "  • Security checks (bandit)"
echo "  • Type checking (mypy)"
echo "  • Documentation checks (pydocstyle, doc8)"
echo "  • Jupyter notebook formatting"
echo "  • Shell script linting"
echo "  • SQL formatting"
echo "  • YAML/JSON validation"
echo ""
echo "🎯 Pre-commit will now run automatically on every commit"
echo "🔧 To run manually: pre-commit run --all-files"
echo "⚙️ To update hooks: pre-commit autoupdate"
echo ""
echo "📚 Configuration files created:"
echo "  • .pre-commit-config.yaml"
echo "  • .flake8"
echo "  • .pylintrc"
echo "  • pyproject.toml"