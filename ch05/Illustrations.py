import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from scipy.stats import beta

    import pymc as pm

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
    return beta, np, plt, pm


@app.cell
def _():
    prior_a = 1.
    prior_b = 1.

    # observed data
    k = 3  # number of tea drinkers
    n = 5  # number of people surveyed

    # posterior
    post_a = prior_a + k
    post_b = prior_b + (n - k)

    post_a, post_b
    return k, n, post_a, post_b, prior_a, prior_b


@app.cell
def _(k, n, pm, prior_a, prior_b):
    with pm.Model() as model:
        p = pm.Beta("p", alpha=prior_a, beta=prior_b)
        likelihood = pm.Binomial("k", n=n, p=p, observed=k)

        post_samples = pm.sample(draws=10_000, chains=1, random_seed=0)
    return (post_samples,)


@app.cell
def _(np, post_samples):
    from pathlib import Path
    post_samples_1 = post_samples['posterior'].p.data.flatten()
    np.save(Path(__file__).parent / 'p_post_samples.npy', post_samples_1)
    return


@app.cell
def _(np):
    from pathlib import Path
    post_samples_2 = np.load(Path(__file__).parent / 'p_post_samples.npy')
    return (post_samples_2,)


@app.cell
def _(plt, post_samples_2):
    plt.hist(post_samples_2, color='b', density=True, alpha=0.6, label='Histogram of samples')
    plt.axis('off')
    # xs = np.linspace(0, 1, 101)
    # plt.plot(xs, beta.pdf(xs, post_a, post_b), c="k", linewidth=5, label="True PDF")
    # plt.legend()
    plt.show()
    return


@app.cell
def _(beta, post_a, post_b, post_samples_2):
    (beta.mean(post_a, post_b), post_samples_2.mean())
    return


@app.cell
def _(beta, np, post_a, post_b, post_samples_2):
    (beta.interval(0.95, post_a, post_b), (np.quantile(post_samples_2, 0.025), np.quantile(post_samples_2, 0.975)))
    return


@app.cell
def _(beta, post_a, post_b, post_samples_2):
    (beta.cdf(0.7, post_a, post_b), (post_samples_2 < 0.7).mean())
    return


@app.cell
def _(beta, np, plt, post_a, post_b):
    iid_samples = beta.rvs(post_a, post_b, size=10000, random_state=0)
    from pathlib import Path
    post_samples_3 = np.load(Path(__file__).parent / 'p_post_samples.npy')
    (fig, axes) = plt.subplots(2, 2, figsize=(10, 6), sharey='row', sharex=True)
    ax = axes[0, 0]
    ax.set_title('Ideal: i.i.d. samples')
    ax.hist(iid_samples, color='b', alpha=0.6, density=True)
    _xs = np.linspace(0, 1, 101)
    ax.plot(_xs, beta.pdf(_xs, post_a, post_b), c='k', linewidth=5)
    iid_samples = iid_samples[-100:]
    ax = axes[1, 0]
    np.random.seed(0)
    y = np.random.uniform(0, 1, size=len(iid_samples))
    ax.scatter(iid_samples, y, alpha=0.7, s=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.set_xlabel('$p$')
    ax.set_ylabel('')
    ax = axes[0, 1]
    ax.set_title('MCMC: chain samples')
    ax.hist(post_samples_3, color='r', alpha=0.6, density=True)
    ax.plot(_xs, beta.pdf(_xs, post_a, post_b), c='k', linewidth=5)
    post_samples_3 = post_samples_3[-min(len(post_samples_3), 100):]
    ax = axes[1, 1]
    _y_chain = np.linspace(0, 1, len(post_samples_3))
    ax.plot(post_samples_3, _y_chain, '-o', markersize=4, alpha=0.8, lw=0.5, color='r')
    ax.set_xlim(0, 1)
    ax.set_xlabel('$p$')
    ax.set_yticks([])
    ax.set_ylabel('')
    for _i in range(len(post_samples_3) - 1):
        ax.annotate('', xy=(post_samples_3[_i + 1], _y_chain[_i + 1]), xytext=(post_samples_3[_i], _y_chain[_i]), arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.6))
    axes[0, 0].set_ylabel('Density')
    axes[1, 0].set_ylabel('Order of generation')
    plt.tight_layout()
    plt.show()
    return (post_samples_3,)


@app.cell
def _(beta, np, plt, post_a, post_b, post_samples_3):
    post_samples_4 = post_samples_3[-min(len(post_samples_3), 100):]
    _y_chain = np.linspace(0, 1, len(post_samples_4))
    plt.plot(post_samples_4, _y_chain, '-o', markersize=4, alpha=0.8, lw=0.5, color='r')
    for _i in range(len(post_samples_4) - 1):
        plt.annotate('', xy=(post_samples_4[_i + 1], _y_chain[_i + 1]), xytext=(post_samples_4[_i], _y_chain[_i]), arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.6))
    _xs = np.linspace(0, 1, 101)
    plt.plot(_xs, beta.pdf(_xs, post_a, post_b) / 2, alpha=0.3, c='k', linewidth=5, label='Target distribution')
    plt.xlim(0, 1)
    plt.legend()
    plt.axis('off')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
