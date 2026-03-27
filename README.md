# Grokking Bayes

Companion code for the *Grokking Bayes* book — chapter-by-chapter [Marimo](https://marimo.io) notebooks covering Bayesian inference from first principles to Bayesian neural networks.

---

## Prerequisites — Install uv

This project uses [uv](https://docs.astral.sh/uv/) for environment management. Install it once on your machine:

**Linux / macOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify with `uv --version`.

---

## Setup

```bash
git clone <repo-url>
cd Grokking-Bayes

# Install all dependencies into an isolated .venv
uv sync

# Open a notebook for editing (browser UI)
uv run marimo edit "ch02/01 - Tea vs. coffee.py"

# Or run as a read-only app
uv run marimo run "ch02/01 - Tea vs. coffee.py"
```

---

## Running tests

Notebooks are executed end-to-end as automated tests using `marimo export html`, which runs each notebook headlessly and fails on any cell error:

```bash
# Run all notebooks
uv run pytest -v

# Run a subset
uv run pytest -v -k "ch06 or ch07"
```

A 1-hour timeout per notebook is configured in `pyproject.toml` to account for MCMC sampling time. The data-download notebook (`ch09/00.1 - Mona Loa download.py`) is excluded from the test suite as it requires network access.

---

## Project structure

| Chapter | Topic |
|---|---|
| ch02 | Discrete Bayesian inference |
| ch03 | Continuous inference — Beta, Gaussian, GPs |
| ch04 | Model checking — prior/posterior predictive |
| ch05 | MCMC — Metropolis-Hastings, NUTS |
| ch06 | Variational Inference |
| ch07 | Hierarchical models |
| ch08 | Mixture models — finite & Dirichlet Process |
| ch09 | Time series — Kalman filtering, CO2 forecasting |
| ch10 | Bayesian neural networks |
| ch11 | Hypothesis testing & Bayes factors |
| ch12 | Decision making under uncertainty |
| ch13 | Active learning — binary search |
