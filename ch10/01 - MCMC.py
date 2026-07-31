import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import os
    import warnings
    os.environ['PYTENSOR_FLAGS'] = 'mode=NUMBA'
    warnings.filterwarnings("ignore", message="PyTensor could not link to a BLAS")
    import pytensor
    import numba
    warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2
    return np, plt


@app.cell
def _(np, plt):
    np.random.seed(0)

    N = 120
    x = np.linspace(-3, 3, N).reshape(-1, 1)
    y_true = np.sin(x)

    sigma_true = 0.2
    y = y_true + np.random.normal(0, sigma_true, size=y_true.shape)

    plt.scatter(x, y, alpha=0.6)
    plt.plot(x, y_true, color="black", linewidth=2)
    plt.title("Sine Data with Homoscedastic Noise")
    plt.show()
    return sigma_true, x, y


@app.cell
def _():
    import pymc as pm
    import pytensor.tensor as pt

    return pm, pt


@app.cell
def _(np, x, y):
    x_data = x.astype("float32")
    y_data = y.flatten().astype("float32")

    n_hidden = 20

    x_test = np.linspace(-5, 5, 101).reshape(-1, 1)
    y_test = np.sin(x_test)
    return n_hidden, x_data, x_test, y_data, y_test


@app.cell
def _(n_hidden, plt, pm, pt, sigma_true, x, x_data, x_test, y, y_data, y_test):
    with pm.Model() as bnn:
        _x_shared = pm.Data("x_shared", x_data)

        # First layer
        _w1 = pm.Normal("w1", 0, 1, shape=(1, n_hidden))
        _b1 = pm.Normal("b1", 0, 1, shape=(n_hidden,))

        # Output layer
        _w2 = pm.Normal("w2", 0, 1, shape=(n_hidden,))
        _b2 = pm.Normal("b2", 0, 1)

        # Forward pass
        _hidden = pt.tanh(pt.dot(_x_shared, _w1) + _b1)
        _mu = pt.dot(_hidden, _w2) + _b2

        # Likelihood — fixed, known observation noise
        _y_obs = pm.Normal("y_obs", mu=_mu, sigma=sigma_true, observed=y_data, shape=_x_shared.shape[0])

        _trace = pm.sample(2000, tune=2000, target_accept=0.95)

        bnn.set_data("x_shared", x_test.astype("float32"))
        _posterior_pred = pm.sample_posterior_predictive(_trace)

    _y_samples = _posterior_pred.posterior_predictive["y_obs"].data

    _pred_mean = _y_samples.mean(axis=(0, 1))
    _pred_std = _y_samples.std(axis=(0, 1))

    plt.scatter(x, y, alpha=0.4, label="Observed data")
    plt.plot(x_test, y_test, color="black", linewidth=2, label="True function")

    plt.plot(x_test, _pred_mean, label="Predictive mean")
    plt.fill_between(
        x_test.flatten(),
        _pred_mean - 2*_pred_std,
        _pred_mean + 2*_pred_std,
        alpha=0.3,
        label="95% predictive interval"
    )

    plt.legend()
    plt.title("Bayesian Neural Network (MCMC)")
    plt.show()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Unknown noise
    """)
    return


@app.cell
def _(n_hidden, plt, pm, pt, x, x_data, x_test, y, y_data, y_test):
    with pm.Model() as unknown_noise_bnn:
        _x_shared = pm.Data("x_shared", x_data)

        # First layer
        _w1 = pm.Normal("w1", 0, 1, shape=(1, n_hidden))
        _b1 = pm.Normal("b1", 0, 1, shape=(n_hidden,))

        # Output layer
        _w2 = pm.Normal("w2", 0, 1, shape=(n_hidden,))
        _b2 = pm.Normal("b2", 0, 1)

        # Unknown noise
        _sigma = pm.HalfNormal("sigma", 1.0)

        # Forward pass
        _hidden = pt.tanh(pt.dot(_x_shared, _w1) + _b1)
        _mu = pt.dot(_hidden, _w2) + _b2

        # Likelihood — unknown observation noise
        _y_obs = pm.Normal("y_obs", mu=_mu, sigma=_sigma, observed=y_data, shape=_x_shared.shape[0])

        _trace = pm.sample(2000, tune=2000, target_accept=0.95)

        unknown_noise_bnn.set_data("x_shared", x_test.astype("float32"))
        _posterior_pred = pm.sample_posterior_predictive(_trace)

    _y_samples = _posterior_pred.posterior_predictive["y_obs"].data

    _pred_mean = _y_samples.mean(axis=(0, 1))
    _pred_std = _y_samples.std(axis=(0, 1))

    plt.scatter(x, y, alpha=0.4, label="Observed data")
    plt.plot(x_test, y_test, color="black", linewidth=2, label="True function")

    plt.plot(x_test, _pred_mean, label="Predictive mean")
    plt.fill_between(
        x_test.flatten(),
        _pred_mean - 2*_pred_std,
        _pred_mean + 2*_pred_std,
        alpha=0.3,
        label="95% predictive interval"
    )

    plt.legend()
    plt.title("Bayesian Neural Network (MCMC) with unknown noise")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
