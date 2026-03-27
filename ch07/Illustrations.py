import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
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
    return np, plt


@app.cell
def _(np):
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
    return sigma, y


@app.cell
def _():
    from scipy.stats import gaussian_kde

    return


@app.cell
def _(np, sigma, y):
    np.random.seed(0)
    _n_samples = 10000
    data = {}
    for (i, (mu, sd)) in enumerate(zip(y, sigma), start=1):
        data[f'School {i}'] = np.random.normal(mu, sd, _n_samples)
    return (data,)


@app.cell
def _(data, plt):
    plt.violinplot([data[k] for k in data], showmeans=True)
    plt.xticks(
        [i + 1 for i in range(8)],
        [k for k in data],
        rotation=45,
    )
    plt.ylabel("Score increase")
    plt.show()
    return


@app.cell
def _():
    from scipy.stats import norm, halfnorm

    return halfnorm, norm


@app.cell
def _(halfnorm, np, plt):
    _xs = np.linspace(-3, 3, 101)
    plt.plot(_xs, halfnorm.pdf(_xs, 0, 1), label='half-normal density')
    # plt.plot(
    #     xs,
    #     norm.pdf(xs, 0, 1),
    #     linestyle="--",
    #     label="normal density"
    # )
    # plt.legend()
    plt.show()
    return


@app.cell
def _():
    from scipy.stats import gamma, beta
    from sklearn.mixture import GaussianMixture

    return GaussianMixture, gamma


@app.cell
def _(gamma, norm, np):
    # --- Target distribution: skewed Gamma ---
    x = np.linspace(0, 20, 500)
    target_pdf = gamma(a=5, scale=1).pdf(x)

    # --- Helper: Gaussian mixture PDF ---
    def gaussian_mixture_pdf(x, weights, mus, sigmas):
        pdf = np.zeros_like(x)
        for (w, mu, sigma) in zip(weights, mus, sigmas):
            pdf = pdf + w * norm.pdf(x, mu, sigma)
        return pdf

    return


@app.cell
def _(GaussianMixture, np, plt):
    components_list = [1, 10, 1000]
    (_fig, _axes) = plt.subplots(1, len(components_list), figsize=(16, 4), sharey=True)
    np.random.seed(42)
    n = 10000
    data_1 = np.linspace(0, 1, n).reshape(-1, 1)
    _xs = np.linspace(0, 1, 500).reshape(-1, 1)
    for (k_i, k) in enumerate(components_list):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(data_1)
        logprob = gmm.score_samples(_xs)
        density = np.exp(logprob)
        _axes[k_i].hist(data_1, bins=60, density=True, alpha=0.5)
        _axes[k_i].plot(_xs, density)
        _axes[k_i].set_title(f'Gaussian Mixture with {k} Component(s)')
        _axes[k_i].set_xlabel('x')
    _axes[0].set_ylabel('Density')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, plt):
    def stick_breaking(alpha, n_components=20, n_samples=5, random_state=None):
        """
        Generate stick-breaking weights for a Dirichlet Process.
    
        Parameters
        ----------
        alpha : float
            Concentration parameter of the DP.
        n_components : int
            Number of components to approximate (truncation level).
        n_samples : int
            Number of different stick-breaking draws to visualize.
        random_state : int, optional
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(random_state)
        weights_all = []
        for _ in range(n_samples):
            betas = rng.beta(1, alpha, size=n_components)
            remaining_stick = np.concatenate(([1.0], np.cumprod(1 - betas[:-1])))
            weights = betas * remaining_stick
            weights_all.append(weights)
        return np.array(weights_all)
    alphas = [0.1, 1, 10]
    # Parameters for visualization
    n_components = 20
    _n_samples = 5
    (_fig, _axes) = plt.subplots(1, len(alphas), figsize=(15, 4), sharey=True)
    for (ax, alpha) in zip(_axes, alphas):
        weights_samples = stick_breaking(alpha, n_components, _n_samples, random_state=42)
        for weights in weights_samples:
            ax.plot(range(1, n_components + 1), weights, marker='o', alpha=0.7)
        ax.set_title(f'alpha = {alpha}')
        ax.set_xlabel('Component index')
        ax.set_ylabel('Weight')
        ax.set_ylim(0, 1)
    _fig.suptitle('Stick-breaking weights for different alpha values', fontsize=14)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
