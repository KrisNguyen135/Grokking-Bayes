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
    from scipy.stats import beta

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
    return beta, np, plt


@app.cell
def _(beta, np):
    # prior
    ps = np.linspace(0, 1, 101)
    prior_a = 1
    prior_b = 1
    prior = beta.pdf(ps, prior_a, prior_b)

    # observed data
    k = 3  # number of tea drinkers
    n = 5  # number of people surveyed
    return k, n, prior, prior_a, prior_b, ps


@app.cell
def _(beta, k, n, prior_a, prior_b, ps):
    # posterior
    post_a = prior_a + k
    post_b = prior_b + (n - k)
    posterior = beta.pdf(ps, post_a, post_b)
    return post_a, post_b, posterior


@app.cell
def _(post_a, post_b):
    post_a, post_b
    return


@app.cell
def _(plt, posterior, prior, ps):
    width = 0.1

    plt.figure(figsize=(8, 5))

    plt.fill_between(
        ps, 
        0, 
        prior, 
        color="blue", 
        alpha=0.6, 
        label="Prior Beta(1, 1)", 
        edgecolor="k",
    )
    plt.fill_between(
        ps, 
        posterior, 
        color='red', 
        alpha=0.6, 
        label='Posterior Beta(4, 3)', 
        hatch="//", 
        edgecolor="k",
    )

    plt.xlabel(r'$p$')
    plt.ylabel('Probability density')
    plt.title('Model of tea preference')
    # plt.xticks(ps)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(beta, post_a, post_b):
    1 - beta.cdf(0.8, post_a, post_b)
    return


@app.cell
def _(beta, post_a, post_b):
    beta.cdf(0.4, post_a, post_b)- beta.cdf(0.2, post_a, post_b)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Illustrations
    """)
    return


@app.cell
def _(beta, plt, posterior, ps):
    plt.figure(figsize=(8, 5))

    # plt.fill_between(ps, 0, prior, color="blue", label="Prior", edgecolor="k")
    plt.fill_between(
        ps, 
        beta(2, 4).pdf(ps), 
        color='blue', 
        # label='Posterior'
    )
    plt.fill_between(ps, posterior, color='red', alpha=0., hatch="//", edgecolor="k")
    plt.axvline(0.6, c="k", linewidth=5, label="Data")

    plt.axis("off")
    # plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, posterior, ps):
    plt.figure(figsize=(8, 5))

    # plt.fill_between(ps, 0, prior, color="blue", label="Prior", edgecolor="k")
    plt.fill_between(ps, posterior, color='red', alpha=0.6, label='Posterior')
    plt.fill_between(ps, posterior, color='red', alpha=0., label='Posterior', hatch="//", edgecolor="k")

    plt.axvline(0.7, linestyle="--", c="k")
    plt.fill_between(ps[ps <= 0.705], posterior[ps <= 0.705], color='red', label='Posterior', hatch="//", edgecolor="k")

    plt.axis("off")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, posterior, ps):
    plt.figure(figsize=(8, 5))

    # plt.fill_between(ps, 0, prior, color="blue", label="Prior", edgecolor="k")
    plt.fill_between(ps, posterior, color='red', alpha=0.6, label='Posterior')
    plt.fill_between(ps, posterior, color='red', alpha=0., label='Posterior', hatch="//", edgecolor="k")

    plt.axvline(0.7, linestyle="--", c="k")
    plt.axvline(0.4, linestyle="--", c="k")
    plt.fill_between(ps[(ps <= 0.705) * (ps >= 0.4)], posterior[(ps <= 0.705) * (ps >= 0.4)], color='red', label='Posterior', hatch="//", edgecolor="k")

    plt.axis("off")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(beta, k, n, plt, ps):
    prior_hyperparams = [(1, 1), (0.5, 0.5), (4, 3), (2, 5)]
    descriptions = ['Uniform prior\nBeta(1, 1)', 'Bimodal prior\nBeta(0.5, 0.5)', 'Prior Beta(4, 3)\nagreeing with data', 'Prior Beta(2, 5)\ndisagreeing with data']
    (fig, ax) = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    for (i, (prior_a_1, prior_b_1)) in enumerate(prior_hyperparams):
        this_ax = ax[i // 2][i % 2]
        post_a_1 = prior_a_1 + k
        post_b_1 = prior_b_1 + (n - k)
        prior_1 = beta.pdf(ps, prior_a_1, prior_b_1)
        posterior_1 = beta.pdf(ps, post_a_1, post_b_1)
        this_ax.fill_between(ps, 0, prior_1, color='blue', alpha=0.6, label='Prior', edgecolor='k')
        this_ax.fill_between(ps, posterior_1, color='red', alpha=0.6, label='Posterior', hatch='//', edgecolor='k')
        this_ax.set_title(descriptions[i])
    ax[1, 1].legend()
    ax[0, 0].set_ylabel('Probability density')
    ax[1, 0].set_ylabel('Probability density')
    ax[1, 0].set_xlabel('$p$')
    ax[1, 1].set_xlabel('$p$')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
