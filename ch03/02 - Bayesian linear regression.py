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
    mu = 100
    sigma2 = 100**2

    sigma2_noise = 20000**2
    return mu, sigma2, sigma2_noise


@app.cell
def _(np):
    np.random.seed(0)
    x = np.linspace(500, 2500, 20)  # square footage
    true_w = 200  # dollars per sqft
    noise = np.random.normal(0, 20000, size=x.shape)
    y = true_w * x + noise
    return x, y


@app.cell
def _(mu, norm, np, plt, sigma2):
    ms = np.linspace(-300, 500, 101)
    prior_pdf = norm.pdf(ms, mu, np.sqrt(sigma2))

    plt.fill_between(ms, 0, prior_pdf, color="blue", alpha=0.6, edgecolor="k")
    plt.title("Prior distribution over price-per-sqft (slope)")
    plt.xlabel("m (dollars per square foot)")
    plt.ylabel("Density")
    plt.show()
    return ms, prior_pdf


@app.cell
def _(mu, np, sigma2, sigma2_noise, x, y):
    post_sigma2 = 1 / (1 / sigma2 + (np.sum(x**2) / sigma2_noise))
    post_mu = post_sigma2 * (mu / sigma2 + np.sum(x * y) / sigma2_noise)
    return post_mu, post_sigma2


@app.cell
def _(ms, norm, np, plt, post_mu, post_sigma2, prior_pdf):
    _post_pdf = norm.pdf(ms, post_mu, np.sqrt(post_sigma2))
    plt.fill_between(ms, 0, prior_pdf, color='blue', alpha=0.6, edgecolor='k', label='Prior')
    plt.fill_between(ms, 0, _post_pdf, color='red', alpha=0.6, edgecolor='k', label='Posterior')
    plt.title('Belief over price-per-sqft (slope)')
    plt.xlabel('m (dollars per square foot)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    return


@app.cell
def _(norm, np, plt, post_mu, post_sigma2):
    # Plot
    w_vals = np.linspace(50, 350, 200)
    _post_pdf = norm.pdf(w_vals, post_mu, np.sqrt(post_sigma2))
    plt.plot(w_vals, _post_pdf)
    plt.title('Posterior distribution over price-per-sqft (slope)')
    plt.xlabel('w (dollars per square foot)')
    plt.ylabel('Density')
    plt.show()
    return


@app.cell
def _(np, plt, post_mu, x, y):
    _xs = np.linspace(500, 2500, 101)
    _ys = _xs * post_mu
    _error = 0
    for (_i, (_x_i, _y_i)) in enumerate(zip(x, y)):
        _pred_y_i = _x_i * post_mu
        plt.vlines(_x_i, _pred_y_i, _y_i, linestyle='--', color='k', label='Error' if _i == 0 else None)
        _error += (_pred_y_i - _y_i) ** 2
    _error /= len(x)
    plt.scatter(x, y, label='Data')
    plt.plot(_xs, _ys, c='r', label='Best-fit line')
    plt.xlabel('Square foot')
    plt.ylabel('Price')
    plt.title(f'Total error: {_error / 10 ** 8:.1f}' + 'e+08')
    plt.legend()
    plt.show()
    return


@app.cell
def _(np, plt, post_mu, post_sigma2, sigma2_noise, x, y):
    _xs = np.linspace(500, 2500, 101)
    y_mean = _xs * post_mu
    y_var = _xs ** 2 * post_sigma2 + sigma2_noise
    y_stddev = np.sqrt(y_var)
    plt.scatter(x, y, label='Data')
    # error = 0
    # for i, (x_i, y_i) in enumerate(zip(x, y)):
    #     pred_y_i = x_i * post_mu
    #     plt.vlines(
    #         x_i, 
    #         pred_y_i, 
    #         y_i, 
    #         linestyle="--", 
    #         color="k", 
    #         label="Error" if i == 0 else None,
    #     )
    #     error += (pred_y_i - y_i) ** 2
    # error /= len(x)
    plt.plot(_xs, y_mean, c='r', label='Mean predictions')
    plt.fill_between(_xs, y_mean + 2 * y_stddev, y_mean - 2 * y_stddev, color='r', alpha=0.3, label='95% credible intervals')
    plt.xlabel('Square foot')
    plt.ylabel('Price')
    plt.legend()
    # plt.title(f"Total error: {error / 10**8:.1f}" + "e+08")
    plt.show()
    return


@app.cell
def _(np, plt, x, y):
    # slope = 206
    slope = 100
    _xs = np.linspace(500, 2500, 101)
    _ys = _xs * slope
    _error = 0
    for (_i, (_x_i, _y_i)) in enumerate(zip(x, y)):
        _pred_y_i = _x_i * slope
        plt.vlines(_x_i, _pred_y_i, _y_i, linestyle='--', color='k', label='Error' if _i == 0 else None)
        _error += (_pred_y_i - _y_i) ** 2
    _error /= len(x)
    plt.scatter(x, y, label='Data')
    plt.plot(_xs, _ys, c='r', label='Trial line')
    plt.xlabel('Square foot')
    plt.ylabel('Price')
    plt.title(f'Total error: {_error / 10 ** 8:.1f}' + 'e+08')
    plt.legend()
    plt.ylim(28239.56387729646, 535600.0)
    print(plt.ylim())
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(ms, plt, prior_pdf):
    plt.fill_between(ms, 0, prior_pdf, color="green", alpha=0.6, edgecolor="k")
    # plt.fill_between(ms, 0, norm.pdf(ms, post_mu, np.sqrt(post_sigma2)), color="red", alpha=0.6, edgecolor="k")
    plt.axis("off")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
