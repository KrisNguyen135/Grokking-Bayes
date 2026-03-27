import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from scipy.stats import norm

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
    return norm, np, plt


@app.cell
def _():
    prior_mean = 0.0
    prior_var = 10.0

    obs = 5.0
    obs_var = 1.0
    return obs, obs_var, prior_mean, prior_var


@app.cell
def _(obs, obs_var, prior_mean, prior_var):
    K = prior_var / (prior_var + obs_var)
    post_mean = prior_mean + K * (obs - prior_mean)
    post_var = (1 - K) * prior_var
    return post_mean, post_var


@app.cell
def _(norm, np, post_mean, post_var, prior_mean, prior_var):
    xs = np.linspace(-5, 10, 101)
    prior_pdf = norm.pdf(xs, prior_mean, np.sqrt(prior_var))
    post_pdf = norm.pdf(xs, post_mean, np.sqrt(post_var))
    return post_pdf, prior_pdf, xs


@app.cell
def _(obs, plt, post_pdf, prior_pdf, xs):
    plt.plot(xs, prior_pdf, linestyle="--", label="Prior")
    plt.fill_between(xs, prior_pdf, 0, alpha=0.3,)

    plt.plot(xs, post_pdf, label="Posterior")
    plt.fill_between(xs, post_pdf, 0, alpha=0.3)

    plt.axvline(obs, linestyle=":", c="k", label="Observation")
    plt.legend()
    plt.axis("off")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
