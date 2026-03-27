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
    import pandas as pd

    import pymc as pm
    import arviz as az

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2
    return az, np, pd, plt, pm


@app.cell
def _(pd):
    from pathlib import Path
    df = pd.read_csv(Path(__file__).parent / "faithful.csv", index_col=0)
    waiting_times = df["waiting"].values
    df["standardized_waiting"] = (
        (waiting_times - waiting_times.mean()) 
        / waiting_times.std()
    )
    df
    return (df,)


@app.cell
def _(df, plt):
    (_fig, _ax) = plt.subplots(figsize=(8, 6))
    _ax.hist(df['standardized_waiting'], color='C0', lw=0, alpha=0.5)
    # n_bins = 20
    _ax.set_xlabel('Standardized waiting time between eruptions')
    _ax.set_ylabel('Number of eruptions')
    plt.show()
    return


@app.cell
def _(df, np, pm):
    with pm.Model(coords={'cluster': range(2)}) as model:
        _mu = pm.Normal('mu', mu=0, sigma=1, transform=pm.distributions.transforms.ordered, initval=[-1, 1], dims='cluster')
        _sigma = pm.HalfNormal('sigma', sigma=1, dims='cluster')
        _w = pm.Dirichlet('w', np.ones(2), dims='cluster')
        _obs = pm.NormalMixture('obs', w=_w, mu=_mu, sigma=_sigma, observed=df['standardized_waiting'].values)
        samples = pm.sample(random_state=0)
    return model, samples


@app.cell
def _(az, plt, samples):
    az.plot_trace(samples)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(az, plt, samples):
    az.plot_trace(samples)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(model, pm, samples):
    with model:
        ppc_samples = pm.sample_posterior_predictive(samples)
    return (ppc_samples,)


@app.cell
def _(az, plt, ppc_samples):
    az.plot_ppc(ppc_samples)
    plt.show()
    return


@app.cell
def _():
    from scipy.stats import norm
    from xarray_einstats.stats import XrContinuousRV

    return XrContinuousRV, norm


@app.cell
def _(XrContinuousRV, norm, np, samples):
    xs = np.linspace(-2.5, 2.5, 101)
    posterior = samples.posterior
    pdf_components = (
        XrContinuousRV(norm, posterior["mu"], posterior["sigma"])
        .pdf(xs)
        * posterior["w"]
    )
    pdf = pdf_components.sum("cluster")
    return pdf, pdf_components


@app.cell
def _(df, pdf, pdf_components, plt):
    (_fig, _ax) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    _ax[0].hist(df['standardized_waiting'])
    # empirical histogram
    _ax[0].set(title='Data', xlabel='Standardized waiting time', ylabel='Frequency')
    (pdf_components / pdf).mean(dim=['chain', 'draw']).plot.line(hue='cluster', ax=_ax[1])
    _ax[1].set(title='Group membership', xlabel='Standardized waiting time', ylabel='Probability')
    # # pdf
    # pdf_components.mean(dim=["chain", "draw"]).sum("cluster").plot.line(ax=ax[1])
    # ax[1].set(title="PDF", xlabel="Standardized waiting time", ylabel="Probability\ndensity")
    plt.tight_layout()
    # plot group membership probabilities
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dirichlet process mixture
    """)
    return


@app.cell
def _():
    K = 10
    return (K,)


@app.cell
def _(pm):
    def stick_breaking(beta):
        portion_remaining = pm.math.concatenate([[1], pm.math.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining

    return (stick_breaking,)


@app.cell
def _(K, df, np, pm, stick_breaking):
    with pm.Model(coords={'cluster': np.arange(K)}) as dp_model:
        beta = pm.Beta('beta', alpha=1, beta=1, dims='cluster')  # Stick-breaking construction
        _w = pm.Deterministic('w', stick_breaking(beta), dims='cluster')
        _mu = pm.Normal('mu', mu=0, sigma=1, dims='cluster')
        _sigma = pm.HalfNormal('sigma', sigma=1, dims='cluster')
        _obs = pm.NormalMixture('obs', w=_w, mu=_mu, sigma=_sigma, observed=df['standardized_waiting'].values)  # Component means and scales
        vi_approx = pm.fit(n=50000)  # initval=np.linspace(-1, 1, K).tolist(),  # Mixture likelihood
    return dp_model, vi_approx


@app.cell
def _(plt, vi_approx):
    plt.plot(vi_approx.hist)
    plt.show()
    return


@app.cell
def _(dp_model, vi_approx):
    with dp_model:
        vi_samples = vi_approx.sample(1000, random_seed=0)
    return (vi_samples,)


@app.cell
def _(az, plt, vi_samples):
    az.plot_trace(vi_samples)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(K, np, plt, vi_samples):
    plot_w = np.arange(K) + 1

    plt.bar(plot_w, vi_samples.posterior["w"].mean(("chain", "draw")), width=1.0, edgecolor="k")

    # plt.xlim(0.5, K)
    plt.xlabel("Component")

    plt.ylabel("Posterior expected mixture weight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
