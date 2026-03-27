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
    from scipy.stats import binom, binomtest, beta
    from scipy.special import comb, betaln

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
    return beta, betaln, binomtest, comb, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hypothesis testing
    """)
    return


@app.cell
def _(betaln, comb, np):
    def log_marginal_likelihood_beta(k, n, a, b):
        return np.log(comb(n, k)) + betaln(a + k, b + n - k) - betaln(a, b)


    def log_marginal_likelihood_constant(k, n, p):
        return np.log(comb(n, k)) + k * np.log(p) + (n - k) * np.log(1 - p)

    return log_marginal_likelihood_beta, log_marginal_likelihood_constant


@app.cell
def _():
    baseline = 0.10
    n = 500
    k = 63
    return baseline, k, n


@app.cell
def _(baseline, binomtest, k, n):
    _test = binomtest(k, n, baseline, alternative='greater')
    _p_value = _test.pvalue
    _p_value
    return


@app.cell
def _(
    baseline,
    k,
    log_marginal_likelihood_beta,
    log_marginal_likelihood_constant,
    n,
    np,
):
    _bf = np.exp(log_marginal_likelihood_constant(k, n, baseline) - log_marginal_likelihood_beta(k, n, 1, 1))
    _bf  # - log_marginal_likelihood_beta(k, n, 1000, 9000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Accumulating evidence
    """)
    return


@app.cell
def _():
    experiments = [
        (13, 100),
        (12, 100),
        (10, 100),
        (16, 100),
        (6, 50),
        (6, 50),
    ]

    n_experiments = len(experiments)
    return experiments, n_experiments


@app.cell
def _(
    baseline,
    binomtest,
    experiments,
    log_marginal_likelihood_beta,
    log_marginal_likelihood_constant,
    np,
):
    running_a = 1
    running_b = 1
    running_n = 0
    running_k = 0
    p_values = []
    bfs = []
    for (i, (k_1, n_1)) in enumerate(experiments):
        print(running_a, running_b)
        running_n = running_n + n_1
        running_k = running_k + k_1
        _test = binomtest(k_1, n_1, baseline, alternative='greater')
        _p_value = _test.pvalue
        _bf = np.exp(log_marginal_likelihood_constant(running_k, running_n, baseline) - log_marginal_likelihood_beta(k_1, n_1, running_a, running_b))
        p_values.append(_p_value)
        bfs.append(_bf)
        running_a = running_a + k_1
        running_b = running_b + (n_1 - k_1)
    (running_a, running_b)
    return bfs, k_1, n_1, p_values


@app.cell
def _(bfs, experiments, n_experiments, p_values, plt):
    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    axes[0].plot(p_values)
    axes[0].axhline(0.05, linestyle="--", c="r")
    axes[0].set_ylabel(r"$p$-value")
    axes[0].set_xticks(
        [i for i in range(n_experiments)],
        [f"Initial\n{experiments[0]}"] + [f"Rep #{i + 1}\n{experiments[i + 1]}" for i in range(n_experiments - 1)],
        rotation=45,
    )

    axes[1].plot(bfs)
    axes[1].set_ylabel("Bayes factor")
    axes[1].set_xticks(
        [i for i in range(n_experiments)],
        [f"Initial\n{experiments[0]}"] + [f"Rep #{i + 1}\n{experiments[i + 1]}" for i in range(n_experiments - 1)],
        rotation=45,
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Working with posteriors
    """)
    return


@app.cell
def _(k_1, n_1):
    prior_a = 1
    prior_b = 1
    a = prior_a + k_1
    b = prior_b + n_1 - k_1
    return a, b


@app.cell
def _(a, b, beta):
    1 - beta(a, b).cdf(0.1)
    return


@app.cell
def _(a, b, beta):
    beta(a, b).cdf(0.05)
    return


if __name__ == "__main__":
    app.run()
