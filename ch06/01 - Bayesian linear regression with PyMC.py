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
    import time
    import numpy as np

    import pymc as pm
    import arviz as az

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
    return az, np, plt, pm


@app.cell
def _(np):
    sigma_noise = 20000
    N = 20

    np.random.seed(0)
    x = np.random.uniform(500, 2500, N)
    true_w = 200
    noise = np.random.normal(0, sigma_noise, size=N)
    y = x * true_w + noise
    return sigma_noise, x, y


@app.cell
def _(plt, x, y):
    plt.scatter(x, y)
    plt.show()
    return


@app.cell
def _(pm, sigma_noise, x, y):
    with pm.Model() as _model:
        m = pm.Normal('m', mu=150, sigma=100)
        _mu = m * x
        pm.Normal('obs', mu=_mu, sigma=sigma_noise, observed=y)
        mcmc_samples = pm.sample(random_seed=0)
    return (mcmc_samples,)


@app.cell
def _(az, mcmc_samples, plt):
    az.plot_posterior(mcmc_samples)
    plt.show()
    return


@app.cell
def _(pm, sigma_noise, x, y):
    with pm.Model() as _model:
        w = pm.Normal('w', mu=150, sigma=100)
        _mu = w * x
        pm.Normal('obs', mu=_mu, sigma=sigma_noise, observed=y)
        vi_approx = pm.fit(n=200000)
        vi_samples = vi_approx.sample(1000, random_seed=0)
    return vi_approx, vi_samples


@app.cell
def _(az, plt, vi_samples):
    az.plot_posterior(vi_samples)
    plt.show()
    return


@app.cell
def _(plt, vi_approx):
    plt.plot(vi_approx.hist)
    plt.xticks(rotation=90)
    plt.xlabel("gradient descent step")
    plt.ylabel("ELBO")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
