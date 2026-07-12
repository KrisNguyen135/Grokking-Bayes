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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prior predictive checks
    """)
    return


@app.cell
def _():
    mu = 150
    sigma2 = 50**2

    sigma2_noise = 20000**2
    return mu, sigma2, sigma2_noise


@app.cell
def _(mu, norm, np, sigma2):
    seed = 0
    x = np.linspace(500, 2500, 10)  # square footage
    _noise = norm.rvs(0, 20000, size=10000, random_state=seed)
    prior_m_samples = norm.rvs(mu, np.sqrt(sigma2), size=10000, random_state=seed)
    prior_y_samples = prior_m_samples.reshape(-1, 1) * x.reshape(1, -1) + _noise.reshape(-1, 1)
    prior_y_samples.shape
    return prior_y_samples, seed, x


@app.cell
def _(plt, prior_y_samples, x):
    plt.violinplot(
        prior_y_samples, 
        positions=x, 
        widths=150,
        showmeans=True,
    )
    plt.xlabel("Square feet")
    plt.ylabel("Price")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Posterior predictive checks
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)
    x_1 = np.linspace(500, 2500, 20)
    true_w = 200
    _noise = np.random.normal(0, 20000, size=x_1.shape)
    y = true_w * x_1 + _noise
    return x_1, y


@app.cell
def _(mu, np, sigma2, sigma2_noise, x_1, y):
    post_sigma2 = 1 / (1 / sigma2 + np.sum(x_1 ** 2) / sigma2_noise)
    post_mu = post_sigma2 * (mu / sigma2 + np.sum(x_1 * y) / sigma2_noise)
    return post_mu, post_sigma2


@app.cell
def _(norm, np, post_mu, post_sigma2, seed, x_1):
    _noise = norm.rvs(0, 20000, size=10000, random_state=seed + 1)
    post_m_samples = norm.rvs(post_mu, np.sqrt(post_sigma2), size=10000, random_state=seed)
    post_y_samples = post_m_samples.reshape(-1, 1) * x_1.reshape(1, -1) + _noise.reshape(-1, 1)
    return (post_y_samples,)


@app.cell
def _(np, plt, post_mu, post_sigma2, sigma2_noise, x_1, y):
    xs = np.linspace(500, 2500, 20)
    y_mean = xs * post_mu
    y_var = xs ** 2 * post_sigma2 + sigma2_noise
    y_stddev = np.sqrt(y_var)
    plt.scatter(x_1, y, label='Data')
    plt.plot(xs, y_mean, c='r', label='Mean predictions')
    plt.fill_between(xs, y_mean + 2 * y_stddev, y_mean - 2 * y_stddev, color='r', alpha=0.3, label='95% credible intervals')
    plt.xlabel('Square foot')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    return y_mean, y_stddev


@app.cell
def _(y, y_mean, y_stddev):
    (
        (y < y_mean + 2 * y_stddev) 
        * (y > y_mean - 2 * y_stddev)
    ).mean()
    return


@app.cell
def _(plt, y, y_mean, y_stddev):
    plt.hist((y - y_mean) / y_stddev, density=True)
    plt.show()
    return


@app.cell
def _(plt, post_y_samples, x_1, y):
    plt.violinplot(post_y_samples, positions=x_1, widths=150, showmeans=True)
    plt.scatter(x_1, y, label='Data', marker='X', c='k', s=100)
    plt.xlabel('Square feet')
    plt.ylabel('Price')
    plt.legend()
    plt.show()  # label="Posterior predictive samples",
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise: misspecified (quadratic) model
    """)
    return


@app.cell
def _(np, x_1):
    np.random.seed(0)
    _noise = np.random.normal(0, 20000, size=x_1.shape)
    y_quad = 5 * x_1 ** 2 + 200 * x_1 + _noise
    return (y_quad,)


@app.cell
def _(mu, np, sigma2, sigma2_noise, x_1, y_quad):
    post_sigma2_quad = 1 / (1 / sigma2 + np.sum(x_1 ** 2) / sigma2_noise)
    post_mu_quad = post_sigma2_quad * (mu / sigma2 + np.sum(x_1 * y_quad) / sigma2_noise)
    return post_mu_quad, post_sigma2_quad


@app.cell
def _(norm, np, post_mu_quad, post_sigma2_quad, seed, x_1):
    _noise = norm.rvs(0, 20000, size=10000, random_state=seed + 1)
    post_m_samples_quad = norm.rvs(post_mu_quad, np.sqrt(post_sigma2_quad), size=10000, random_state=seed)
    post_y_samples_quad = post_m_samples_quad.reshape(-1, 1) * x_1.reshape(1, -1) + _noise.reshape(-1, 1)
    return (post_y_samples_quad,)


@app.cell
def _(np, plt, post_mu_quad, post_sigma2_quad, sigma2_noise, x_1, y_quad):
    y_mean_quad = x_1 * post_mu_quad
    y_var_quad = x_1 ** 2 * post_sigma2_quad + sigma2_noise
    y_stddev_quad = np.sqrt(y_var_quad)
    plt.scatter(x_1, y_quad, label='Data')
    plt.plot(x_1, y_mean_quad, c='r', label='Mean predictions')
    plt.fill_between(x_1, y_mean_quad + 2 * y_stddev_quad, y_mean_quad - 2 * y_stddev_quad, color='r', alpha=0.3, label='95% credible intervals')
    plt.xlabel('Square foot')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    return y_mean_quad, y_stddev_quad


@app.cell
def _(y_quad, y_mean_quad, y_stddev_quad):
    (
        (y_quad < y_mean_quad + 2 * y_stddev_quad)
        * (y_quad > y_mean_quad - 2 * y_stddev_quad)
    ).mean()
    return


@app.cell
def _(plt, y_quad, y_mean_quad, y_stddev_quad):
    plt.hist((y_quad - y_mean_quad) / y_stddev_quad, density=True)
    plt.show()
    return


@app.cell
def _(plt, post_y_samples_quad, x_1, y_quad):
    plt.violinplot(post_y_samples_quad, positions=x_1, widths=150, showmeans=True)
    plt.scatter(x_1, y_quad, label='Data', marker='X', c='k', s=100)
    plt.xlabel('Square feet')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
