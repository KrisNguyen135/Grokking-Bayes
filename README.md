# Grokking Bayes

Companion code for the *Grokking Bayes* book — chapter-by-chapter Jupyter notebooks covering Bayesian inference from first principles to Bayesian neural networks.

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

# Register the Jupyter kernel
uv run python -m ipykernel install --user --name grokking-bayes --display-name "Grokking Bayes (Python 3.10)"

# Launch JupyterLab
uv run jupyter lab
```

Select the **Grokking Bayes (Python 3.10)** kernel when opening any notebook.

---

## Running tests

Notebooks can be executed top-to-bottom as automated tests using [nbmake](https://github.com/treebeardtech/nbmake):

```bash
uv run pytest --nbmake ch06/ ch07/ ch08/ ch10/ -v
```

A 1-hour timeout per notebook is configured in `pyproject.toml` to account for MCMC sampling time.

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
