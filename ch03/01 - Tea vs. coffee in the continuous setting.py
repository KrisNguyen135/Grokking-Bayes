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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise

    Adjust the prior hyperparameters $\alpha$ and $\beta$ to explore different prior
    shapes and their effect on the posterior. Try uniform, bimodal, or strongly
    opinionated priors.

    Then increase the sample size $n$ to observe how more data leads to a sharper,
    more concentrated posterior whose peak becomes narrow and tall.

    The **observed proportion** $\hat{p}$ controls what fraction of the $n$ trials
    were successes: the integer count is $k = \text{round}(\hat{p} \times n)$,
    so $k$ is always bounded between $0$ and $n$.
    """)
    return


@app.cell
def _(mo):
    prior_a_slider = mo.ui.slider(start=0.1, stop=10.0, step=0.1, value=1.0, label="Prior α")
    prior_b_slider = mo.ui.slider(start=0.1, stop=10.0, step=0.1, value=1.0, label="Prior β")
    n_slider = mo.ui.slider(start=1, stop=200, step=1, value=5, label="Sample size n")
    p_hat_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.01, value=0.6, label="Observed proportion p̂")
    mo.output.replace(mo.vstack([
        mo.hstack([prior_a_slider, prior_b_slider], justify="start"),
        mo.hstack([n_slider, p_hat_slider], justify="start"),
    ]))
    return n_slider, p_hat_slider, prior_a_slider, prior_b_slider


@app.cell
def _(beta, n_slider, p_hat_slider, plt, prior_a_slider, prior_b_slider, ps):
    _prior_a = prior_a_slider.value
    _prior_b = prior_b_slider.value
    _n = n_slider.value
    _k = round(p_hat_slider.value * _n)

    _prior = beta.pdf(ps, _prior_a, _prior_b)
    _post_a = _prior_a + _k
    _post_b = _prior_b + (_n - _k)
    _posterior = beta.pdf(ps, _post_a, _post_b)

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.fill_between(ps, 0, _prior, color="blue", alpha=0.6,
                     label=f"Prior Beta({_prior_a:.1f}, {_prior_b:.1f})", edgecolor="k")
    _ax.fill_between(ps, _posterior, color='red', alpha=0.6,
                     label=f"Posterior Beta({_post_a:.1f}, {_post_b:.1f})",
                     hatch="//", edgecolor="k")
    _ax.set_xlabel(r'$p$')
    _ax.set_ylabel('Probability density')
    _ax.set_title(f'n={_n}, k={_k} (p̂={p_hat_slider.value:.2f})')
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
