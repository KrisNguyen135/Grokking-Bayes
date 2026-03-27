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
    from scipy.special import comb

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
    return comb, np, plt


@app.cell
def _(np):
    # prior
    ps = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    prior = np.ones_like(ps) / len(ps)

    # observed data
    k = 3  # number of tea drinkers
    n = 5  # number of people surveyed
    return k, n, prior, ps


@app.cell
def _(comb, k, n, prior, ps):
    # likelihood
    binom_coeff = comb(n, k)
    likelihood = binom_coeff * (ps ** k) * ((1 - ps) ** (n - k))

    # unnormalized posterior
    unnormalized_posterior = prior * likelihood

    # normalize the posterior
    posterior = unnormalized_posterior / unnormalized_posterior.sum()
    return (posterior,)


@app.cell
def _(plt, posterior, prior, ps):
    width = 0.1

    plt.figure(figsize=(8, 5))

    plt.bar(ps, prior, width=width, color='blue', alpha=0.6, label='Prior', align='center', edgecolor="k")
    plt.bar(ps, posterior, width=width, color='red', alpha=0.6, label='Posterior', align='center', edgecolor="k", hatch="//")

    plt.xlabel(r'$p$')
    plt.ylabel('Probability')
    plt.title('Model of tea preference')
    plt.xticks(ps)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Code to generate illustrations
    """)
    return


@app.cell
def _(plt, posterior, ps):
    width_1 = 0.1
    plt.figure(figsize=(8, 5))
    plt.bar(ps, posterior, width=width_1, color='#FF0000', alpha=1, label='Prior', align='center', edgecolor='k')
    plt.bar(ps, posterior, width=width_1, color='red', alpha=0.0, label='Posterior', align='center')
    # plt.bar(ps, prior, width=width, color='#0000FF', alpha=1, label='Prior', align='center', edgecolor="k")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import matplotlib as mpl

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    return


@app.cell
def _(plt, posterior, ps):
    width_2 = 0.1
    plt.figure(figsize=(8, 5))
    plt.bar(ps, posterior, width=width_2, color='red', alpha=0.6, label='Posterior', align='center', edgecolor='k', hatch='//')
    plt.xlabel('$p$')
    # plt.bar(ps, prior, width=width, color='blue', alpha=0.6, label='Prior', align='center', edgecolor="k")
    plt.ylabel('Probability')
    plt.xticks(ps)
    plt.tight_layout()
    # plt.title('Model of tea preference')
    # plt.legend()
    plt.show()
    return (width_2,)


@app.cell
def _(ps):
    ps
    return


@app.cell
def _(posterior, ps):
    posterior[ps >= 0.8].sum()
    return


@app.cell
def _(posterior, ps):
    posterior[(ps <= 0.4) * (ps >= 0.2)].sum()
    return


@app.cell
def _(np, posterior, ps):
    posterior[np.isclose(ps, 0.6)]
    return


@app.cell
def _(comb, ps):
    def compute_posterior(prior, n, k):
        # likelihood
        binom_coeff = comb(n, k)
        likelihood = binom_coeff * (ps ** k) * ((1 - ps) ** (n - k))
    
        # unnormalized posterior
        unnormalized_posterior = prior * likelihood
    
        # normalize the posterior
        return unnormalized_posterior / unnormalized_posterior.sum()

    return (compute_posterior,)


@app.cell
def _(compute_posterior, plt, prior, ps, width_2):
    data = [(5, 3), (10, 6), (50, 30), (100, 60)]
    (_fig, _ax) = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
    for (_i, (n_1, k_1)) in enumerate(data):
        _this_ax = _ax[_i // 2][_i % 2]
        posterior_1 = compute_posterior(prior, n_1, k_1)
        _this_ax.bar(ps, prior, width=width_2, color='blue', alpha=0.6, label='Prior', align='center', edgecolor='k')
        _this_ax.bar(ps, posterior_1, width=width_2, color='red', alpha=0.6, label='Posterior', align='center', edgecolor='k', hatch='//')
        _this_ax.set_title(f'{k_1} out of {n_1} people prefer tea\n' + '$\\Pr(p = 0.6 \\mid D) \\approx$' + f'{posterior_1[6]:.2f}')
    _ax[1, 1].legend()
    _ax[0, 0].set_ylabel('Probability')
    _ax[1, 0].set_ylabel('Probability')
    _ax[1, 0].set_xlabel('$p$')
    _ax[1, 1].set_xlabel('$p$')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(compute_posterior, np, plt, ps, width_2):
    priors = [np.ones_like(ps) / len(ps), np.array([5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5]) / np.array([5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5]).sum(), np.array([0.08] * 6 + [0.2] + [0.08] * 4), np.array([0.2] * 5 + [0] * 6)]
    descriptions = ['Uniform prior', 'Bimodal prior', 'Prior agreeing with data', 'Limited prior']
    (_fig, _ax) = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    for (_i, prior_1) in enumerate(priors):
        _this_ax = _ax[_i // 2][_i % 2]
        posterior_2 = compute_posterior(prior_1, 100, 60)
        _this_ax.bar(ps, prior_1, width=width_2, color='blue', alpha=0.6, label='Prior', align='center', edgecolor='k')
        _this_ax.bar(ps, posterior_2, width=width_2, color='red', alpha=0.6, label='Posterior', align='center', edgecolor='k', hatch='//')
        _this_ax.set_title(descriptions[_i])
    _ax[1, 1].legend()
    _ax[0, 0].set_ylabel('Probability')
    _ax[1, 0].set_ylabel('Probability')
    _ax[1, 0].set_xlabel('$p$')
    _ax[1, 1].set_xlabel('$p$')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
