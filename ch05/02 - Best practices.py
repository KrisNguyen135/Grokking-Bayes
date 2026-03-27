import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
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
    return az, plt, pm


@app.cell
def _():
    prior_a = 1.
    prior_b = 1.

    # observed data
    k = 3  # number of tea drinkers
    n = 5  # number of people surveyed
    return k, n, prior_a, prior_b


@app.cell
def _(k, n, pm, prior_a, prior_b):
    with pm.Model() as model:
        p = pm.Beta("p", alpha=prior_a, beta=prior_b)
        likelihood = pm.Binomial("k", n=n, p=p, observed=k)

        samples = pm.sample(random_seed=0)
    return model, samples


@app.cell
def _(az, samples):
    az.summary(samples, var_names=["p"])
    return


@app.cell
def _(az, plt, samples):
    az.plot_trace(samples, var_names=["p"])
    plt.show()
    return


@app.cell
def _(az, model, plt, pm, samples):
    with model:
        ppc = pm.sample_posterior_predictive(samples, var_names=["k"])

    az.plot_ppc(ppc)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
