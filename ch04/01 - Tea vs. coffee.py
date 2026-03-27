import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    from scipy.stats import beta, norm, binom

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
    return beta, binom, norm, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prior predictive check
    """)
    return


@app.cell
def _(np):
    # prior
    ps = np.linspace(0, 1, 101)

    # prior_a = 1
    # prior_b = 1

    prior_a = 0.01
    prior_b = 10
    return prior_a, prior_b


@app.cell
def _(beta, binom, plt, prior_a, prior_b):
    _seed = 0
    prior_p_samples = beta.rvs(prior_a, prior_b, size=10000, random_state=_seed)
    prior_k_samples = binom.rvs(100, prior_p_samples, random_state=_seed)
    plt.figure(figsize=(8, 5))
    plt.hist(prior_k_samples, color='b', density=True)
    plt.xlabel('$k_i$')
    plt.ylabel('Density')
    plt.title('Histogram of prior samples of $k$')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Posterior predictive check
    """)
    return


@app.cell
def _(prior_a, prior_b):
    # observed data
    k = 3  # number of tea drinkers
    n = 5  # number of people surveyed

    # posterior
    post_a = prior_a + k
    post_b = prior_b + (n - k)

    post_a, post_b
    return k, n, post_a, post_b


@app.cell
def _(beta, binom, plt, post_a, post_b):
    _seed = 0
    post_p_samples = beta.rvs(post_a, post_b, size=10000, random_state=_seed)
    post_k_samples = binom.rvs(5, post_p_samples, random_state=_seed)
    plt.figure(figsize=(8, 5))
    plt.hist(post_k_samples, color='b', density=True)
    plt.xlabel('$k_i$')
    plt.ylabel('Density')
    plt.title('Histogram of post samples of $k$')
    plt.show()
    return (post_k_samples,)


@app.cell
def _(k, post_k_samples):
    (post_k_samples == k).sum()
    return


@app.cell
def _(k, post_k_samples):
    (post_k_samples > k).mean(), (post_k_samples < k).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayes factor
    """)
    return


@app.cell
def _():
    from scipy.special import comb, beta as beta_fn

    return beta_fn, comb


@app.cell
def _(beta_fn, comb):
    def marginal_likelihood(k, n, a, b):
        return comb(n, k) * beta_fn(a + k, b + n - k) / beta_fn(a, b)

    return (marginal_likelihood,)


@app.cell
def _(k, marginal_likelihood, n):
    (
        marginal_likelihood(k, n, 1, 1),
        marginal_likelihood(k, n, 4, 3),
        marginal_likelihood(k, n, 61, 41)
    )
    return


@app.cell
def _(k, marginal_likelihood, n):
    (
        marginal_likelihood(k, n, 2, 4)
        / marginal_likelihood(k, n, 1, 1)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # illustrations
    """)
    return


@app.cell
def _(beta, norm, np, plt, prior_a, prior_b):
    fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharey="row")

    xs = np.linspace(0, 1, 101)
    prior_pdf = beta.pdf(xs, prior_a, prior_b)
    prior_samples = beta.rvs(prior_a, prior_b, size=1000, random_state=0)

    ax[0, 0].fill_between(xs, prior_pdf, color="blue", edgecolor="k", alpha=0.6)
    ax[0, 1].hist(prior_samples, density=True, color="blue")

    ax[0, 0].set_xlabel(r'$p$')
    ax[0, 1].set_xlabel(r'$p$')

    ax[0, 0].set_ylabel("Beta(1, 1)")
    ax[1, 0].set_ylabel(r"$\mathcal{N}~(100, 100^2)$")
    ax[0, 0].set_title("True PDF")
    ax[0, 1].set_title("Sample histogram")

    mu = 100
    sigma2 = 100**2

    mus = np.linspace(-300, 500, 101)
    prior_pdf = norm.pdf(mus, mu, np.sqrt(sigma2))
    prior_samples = norm.rvs(mu, np.sqrt(sigma2), size=1000, random_state=0)

    ax[1, 0].fill_between(mus, prior_pdf, color="blue", edgecolor="k", alpha=0.6)
    ax[1, 1].hist(prior_samples, density=True, color="blue")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
