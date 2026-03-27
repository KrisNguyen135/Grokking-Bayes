import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from scipy.stats import beta
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
    return beta, comb, np, plt


@app.cell
def _():
    # observed data
    k = 3  # number of tea drinkers
    n = 5  # number of people surveyed
    return k, n


@app.cell
def _(comb, k, n):
    def unnormalized_posterior(p):
        if p <= 0 or p >= 1:
            return 0

        binom_coeff = comb(n, k)
        return binom_coeff * (p ** k) * ((1 - p) ** (n - k))

    return (unnormalized_posterior,)


@app.cell
def _(np, unnormalized_posterior):
    # Run sampler
    np.random.seed(0)

    initial = 0.1
    n_samples = 10_000
    proposal_stddev = 0.05

    samples = np.empty(n_samples)
    samples[0] = initial
    current_post = unnormalized_posterior(initial)

    for t in range(1, n_samples):
        proposal = np.random.normal(samples[t-1], proposal_stddev)
        proposal_post = unnormalized_posterior(proposal)

        # Acceptance ratio
        r = proposal_post / current_post
        if np.random.rand() < r:
            samples[t] = proposal
            current_post = proposal_post
        else:
            samples[t] = samples[t-1]

    # Discard burn-in
    # burn_in = 1000
    post_samples = samples#[burn_in:]
    return (post_samples,)


@app.cell
def _(plt, post_samples):
    # Plot trace and histogram
    (_fig, _axes) = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(post_samples, alpha=0.6)
    _axes[0].set_title('Trace plot (after burn-in)')
    _axes[0].set_xlabel('Iteration')
    _axes[0].set_ylabel('theta')
    _axes[1].hist(post_samples, density=True, alpha=0.6)
    _axes[1].set_title('Posterior histogram from MCMC samples')
    _axes[1].set_xlabel('theta')
    _axes[1].set_ylabel('Density')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(beta, np, plt, post_samples):
    burn_in = 1000
    xs = np.linspace(0, 1, 101)
    (_fig, _axes) = plt.subplots(1, 3, figsize=(15, 4))
    _axes[0].plot(post_samples, alpha=0.6)
    # Plot trace and histogram
    _axes[0].set_title('Trace plot')
    _axes[0].set_xlabel('Iteration')
    _axes[0].set_ylabel('theta')
    _axes[1].hist(post_samples[:burn_in], density=True, alpha=0.6)
    _axes[1].set_title('Histogram of burn-in samples')
    _axes[1].plot(xs, beta.pdf(xs, 4, 3), c='k')
    _axes[1].set_xlabel('theta')
    _axes[1].set_ylabel('Density')
    _axes[2].hist(post_samples[burn_in:], density=True, alpha=0.6)
    _axes[2].set_title('Histogram of samples without burn-in')
    _axes[2].plot(xs, beta.pdf(xs, 4, 3), c='k')
    _axes[2].set_xlabel('theta')
    _axes[2].set_ylabel('Density')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
