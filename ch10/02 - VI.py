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
def _(np):
    np.random.seed(0)

    N = 120
    x = np.linspace(-3, 3, N).reshape(-1, 1)
    y_true = np.sin(x)

    sigma_true = 0.2
    y = y_true + np.random.normal(0, sigma_true, size=y_true.shape)
    return x, y


@app.cell
def _():
    import pymc as pm
    import pytensor.tensor as pt

    return pm, pt


@app.cell
def _(x, y):
    x_data = x.astype("float32")
    y_data = y.flatten().astype("float32")

    n_hidden = 20
    return n_hidden, x_data, y_data


@app.cell
def _(n_hidden, pm, pt, x_data, y_data):
    with pm.Model() as vi_bnn:
        x_shared = pm.Data("x_shared", x_data)

        # First layer
        w1 = pm.Normal("w1", 0, 1, shape=(1, n_hidden))
        b1 = pm.Normal("b1", 0, 1, shape=(n_hidden,))

        # Output layer
        w2 = pm.Normal("w2", 0, 1, shape=(n_hidden,))
        b2 = pm.Normal("b2", 0, 1)

        # # Global observation noise
        sigma = pm.HalfNormal("sigma", 1.0)

        # Forward pass
        hidden = pt.tanh(pt.dot(x_shared, w1) + b1)
        mu = pt.dot(hidden, w2) + b2

        # Likelihood
        # y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_true, observed=y_data, shape=x_shared.shape[0])
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data, shape=x_shared.shape[0])

        approx = pm.fit(
            n=200_000,
            obj_n_mc=10,
            obj_optimizer=pm.adam(learning_rate=1e-3),
        )
    return approx, vi_bnn


@app.cell
def _(approx, plt):
    plt.plot(approx.hist)
    plt.title("ELBO Convergence")
    plt.show()
    return


@app.cell
def _(approx, plt):
    plt.plot(approx.hist)
    plt.title("ELBO Convergence")
    plt.ylim(0, 200)
    plt.xticks(rotation=90)
    plt.show()
    return


@app.cell
def _(np):
    x_test = np.linspace(-5, 5, 101).reshape(-1, 1)
    y_test = np.sin(x_test)
    return x_test, y_test


@app.cell
def _(approx, pm, vi_bnn, x_test):
    with vi_bnn:
        vi_bnn.set_data("x_shared", x_test.astype("float32"))
        trace_vi = approx.sample(2000)
        posterior_pred = pm.sample_posterior_predictive(trace_vi)
    return (posterior_pred,)


@app.cell
def _(posterior_pred):
    y_samples = posterior_pred.posterior_predictive["y_obs"].data

    pred_mean = y_samples.mean(axis=(0, 1))
    pred_std = y_samples.std(axis=(0, 1))
    return pred_mean, pred_std


@app.cell
def _(plt, pred_mean, pred_std, x, x_test, y, y_test):
    plt.scatter(x, y, alpha=0.4, label="Observed data")
    plt.plot(x_test, y_test, color="black", linewidth=2, label="True function")

    plt.plot(x_test, pred_mean, label="Predictive mean")
    plt.fill_between(
        x_test.flatten(),
        pred_mean - 2*pred_std,
        pred_mean + 2*pred_std,
        alpha=0.3,
        label="95% predictive interval"
    )

    plt.legend()
    plt.title("Bayesian Neural Network (ADVI)")
    plt.show()
    return


@app.cell
def _(plt, pred_mean, pred_std, x, x_test, y, y_test):
    plt.scatter(x, y, alpha=0.4, label="Observed data")
    plt.plot(x_test, y_test, color="black", linewidth=2, label="True function")

    plt.plot(x_test, pred_mean, label="Predictive mean")
    plt.fill_between(
        x_test.flatten(),
        pred_mean - 2*pred_std,
        pred_mean + 2*pred_std,
        alpha=0.3,
        label="95% predictive interval"
    )

    plt.legend()
    plt.title("Bayesian Neural Network (ADVI)")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
