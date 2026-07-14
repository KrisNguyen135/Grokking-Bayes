import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


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
    from scipy.stats import lognorm

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
    return az, lognorm, np, plt, pm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise: Bayesian Linear Regression with PyMC and ArviZ

    In chapter 3 we derived the posterior over the slope $m$ analytically using
    the conjugate Normal-Normal update. Here we reimplement the same model using
    PyMC (NUTS sampling) and validate it with ArviZ's diagnostics and posterior
    predictive checks.

    As an extra step we replace the Normal prior with a **LogNormal prior** to
    guarantee that the slope is non-negative — a natural constraint for house
    prices per square foot.
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)
    x = np.linspace(500, 2500, 20)   # square footage
    true_w = 200                      # dollars per sqft
    sigma_noise = 20000
    noise = np.random.normal(0, sigma_noise, size=x.shape)
    y = true_w * x + noise
    return noise, sigma_noise, true_w, x, y


@app.cell
def _(plt, x, y):
    plt.scatter(x, y)
    plt.xlabel('Square footage')
    plt.ylabel('Price ($)')
    plt.title('House price data')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 1: Normal prior (conjugate)

    We place a Normal prior on the slope $m$:

    $$m \sim \mathcal{N}(\mu=150,\; \sigma=100)$$

    This is the same conjugate prior family used in chapter 3, but now PyMC's
    NUTS sampler approximates the posterior instead of computing it in closed form.
    The two approaches should agree closely.
    """)
    return


@app.cell
def _(pm, sigma_noise, x, y):
    with pm.Model() as normal_model:
        _m = pm.Normal('m', mu=150, sigma=100)
        _mu = _m * x
        pm.Normal('obs', mu=_mu, sigma=sigma_noise, observed=y)
        normal_samples = pm.sample(random_seed=0)
    return normal_model, normal_samples


@app.cell
def _(az, normal_samples):
    az.summary(normal_samples, var_names=['m'])


@app.cell
def _(az, normal_samples, plt):
    az.plot_trace(normal_samples, var_names=['m'])
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(az, normal_samples, plt):
    az.plot_posterior(normal_samples, var_names=['m'])
    plt.show()
    return


@app.cell
def _(normal_model, normal_samples, pm):
    with normal_model:
        normal_ppc = pm.sample_posterior_predictive(normal_samples, random_seed=0)
    return (normal_ppc,)


@app.cell
def _(az, normal_ppc, plt):
    az.plot_ppc(normal_ppc)
    plt.show()
    return


@app.cell
def _(normal_samples, np, plt, x, y):
    _posterior_m = normal_samples.posterior['m'].values.flatten()
    _xs = np.linspace(500, 2500, 101)
    _y_mean = _xs * _posterior_m.mean()
    _y_std = _xs * _posterior_m.std()

    plt.scatter(x, y, label='Data', zorder=5)
    plt.plot(_xs, _y_mean, c='red', label='Posterior mean')
    plt.fill_between(
        _xs,
        _y_mean - 2 * _y_std,
        _y_mean + 2 * _y_std,
        alpha=0.3, color='red', label='95% credible band'
    )
    plt.xlabel('Square footage')
    plt.ylabel('Price ($)')
    plt.title('Normal prior — posterior predictive')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 2: LogNormal prior (non-conjugate)

    A Normal prior on $m$ assigns probability mass to negative slopes — implying
    that larger houses are cheaper, which makes no physical sense.

    A **LogNormal prior** fixes this: if $\log(m)$ follows a Normal distribution
    then $m$ is always positive. This is a non-conjugate prior, so there is no
    closed-form posterior — exactly the setting where MCMC earns its place.

    $$m \sim \text{LogNormal}(\mu_{\log}=\log(150),\; \sigma_{\log}=1.0)$$
    """)
    return


@app.cell
def _(lognorm, np, plt):
    _m_vals = np.linspace(0, 800, 500)
    _pdf = lognorm.pdf(_m_vals, s=1.0, scale=150.0)

    plt.fill_between(_m_vals, 0, _pdf, alpha=0.6, edgecolor='k')
    plt.title('LogNormal prior on slope $m$')
    plt.xlabel('m (dollars per square foot)')
    plt.ylabel('Density')
    plt.show()
    return


@app.cell
def _(np, pm, sigma_noise, x, y):
    with pm.Model() as lognormal_model:
        _m = pm.LogNormal('m', mu=np.log(150.0), sigma=1.0)
        _mu = _m * x
        pm.Normal('obs', mu=_mu, sigma=sigma_noise, observed=y)
        lognormal_samples = pm.sample(random_seed=0)
    return lognormal_model, lognormal_samples


@app.cell
def _(az, lognormal_samples):
    az.summary(lognormal_samples, var_names=['m'])


@app.cell
def _(az, lognormal_samples, plt):
    az.plot_trace(lognormal_samples, var_names=['m'])
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(lognormal_model, lognormal_samples, pm):
    with lognormal_model:
        lognormal_ppc = pm.sample_posterior_predictive(lognormal_samples, random_seed=0)
    return (lognormal_ppc,)


@app.cell
def _(az, lognormal_ppc, plt):
    az.plot_ppc(lognormal_ppc)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing the two posteriors

    Both models recover a similar posterior for $m$ because the data strongly
    constrains the slope — likelihood dominates the prior. The key difference is
    support: the Normal prior assigns probability mass to negative slopes, while
    the LogNormal prior does not. When the data are informative the choice of
    prior matters less; when the data are sparse it matters a great deal.
    """)
    return


@app.cell
def _(az, lognormal_samples, normal_samples, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    az.plot_posterior(normal_samples, var_names=['m'], ax=_axes[0])
    az.plot_posterior(lognormal_samples, var_names=['m'], ax=_axes[1])
    _axes[0].set_title('Normal prior')
    _axes[1].set_title('LogNormal prior')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
