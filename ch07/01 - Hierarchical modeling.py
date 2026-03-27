import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import os
    import warnings
    os.environ['PYTENSOR_FLAGS'] = 'mode=NUMBA'
    warnings.filterwarnings("ignore", message="PyTensor could not link to a BLAS")
    import pytensor
    import numba
    warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)
    import numpy as np

    import pymc as pm
    import arviz as az

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.2
    return az, np, plt, pm, sns


@app.cell
def _(np):
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
    return sigma, y


@app.cell
def _(pm, sigma, y):
    with pm.Model() as pooled:
        _theta = pm.Normal('theta', 0, sigma=1000000.0)
        _obs = pm.Normal('obs', _theta, sigma=sigma, observed=y)
        trace_p = pm.sample(random_state=0)
    return (trace_p,)


@app.cell
def _(az, plt, trace_p):
    az.plot_trace(trace_p)
    plt.show()
    return


@app.cell
def _(trace_p):
    trace_p.posterior["theta"].values.mean()
    return


@app.cell
def _(pm, sigma, y):
    with pm.Model() as hierarchical:
        mu = pm.Normal('mu', 0, sigma=100)
        tau = pm.HalfNormal('tau', 100)
        _theta = pm.Normal('theta', mu=mu, sigma=tau, shape=len(y))
        _obs = pm.Normal('obs', _theta, sigma=sigma, observed=y)
        trace_h = pm.sample(draws=10000, target_accept=0.9, random_state=0)
    return (trace_h,)


@app.cell
def _(az, plt, trace_h):
    az.plot_trace(trace_h)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(trace_h):
    (
        (trace_h.posterior["tau"].values > 25).mean(),
        (trace_h.posterior["tau"].values < 10).mean(),
    )
    return


@app.cell
def _(np, trace_h):
    tau_bins = np.linspace(0, 20, 11)
    tau_bin_indices = np.digitize(trace_h.posterior["tau"], bins=tau_bins)
    # tau_bin_indices
    return tau_bin_indices, tau_bins


@app.cell
def _(np, tau_bin_indices, tau_bins, trace_h, y):
    theta_means = np.zeros((len(tau_bins), len(y)))
    theta_stddevs = np.zeros((len(tau_bins), len(y)))
    for _i in range(len(tau_bins)):
        theta_given_tau = trace_h.posterior['theta'].values[tau_bin_indices == _i + 1]
        theta_means[_i, :] = theta_given_tau.mean(axis=0)
        theta_stddevs[_i, :] = theta_given_tau.std(axis=0)
    return theta_means, theta_stddevs


@app.cell
def _(plt, tau_bins, theta_means, y):
    for _j in range(len(y)):
        plt.plot(tau_bins, theta_means[:, _j], color='k')
    plt.xlabel('tau')
    plt.ylabel('Estimated effect conditioned on tau')
    plt.show()  # yerr=theta_stddevs[:, j],  # label=f"School number {j + 1}",  # fmt="o-",  # cmap="Set1",
    return


@app.cell
def _(plt, sns, tau_bins, theta_means, y):
    with sns.color_palette('Set1', n_colors=8):
        for _j in range(len(y)):
            plt.plot(tau_bins, theta_means[:, _j], label=f'School number {_j + 1}')
    plt.legend(ncol=4, bbox_to_anchor=(1.4, 1.3))
    plt.show()  # yerr=theta_stddevs[:, j],  # fmt="o-",  # cmap="Set1",
    return


@app.cell
def _(plt, tau_bins, theta_stddevs, y):
    for _j in range(len(y)):
        plt.plot(tau_bins, theta_stddevs[:, _j], label=f'School number {_j + 1}', color='k')
    plt.xlabel('tau')
    plt.ylabel('Effect standard error conditioned on tau')
    plt.show()  # yerr=theta_stddevs[:, j],  # fmt="o-",  # cmap="Set1",
    return


@app.cell
def _(plt, sns, tau_bins, theta_stddevs, y):
    with sns.color_palette('Set1', n_colors=8):
        for _j in range(len(y)):
            plt.plot(tau_bins, theta_stddevs[:, _j], label=f'School number {_j + 1}')
    plt.legend(ncol=4, bbox_to_anchor=(1.4, 1.3))
    plt.show()  # yerr=theta_stddevs[:, j],  # fmt="o-",  # cmap="Set1",
    return


@app.cell
def _(trace_h):
    post_thetas = trace_h.posterior["theta"].values
    return (post_thetas,)


@app.cell
def _(post_thetas):
    for _i in range(8):
        print(_i, (post_thetas.argmax(axis=-1) == _i).mean())
    return


@app.cell
def _(post_thetas):
    (post_thetas[..., 0] > post_thetas[..., 2]).mean()
    return


@app.cell
def _(post_thetas):
    (post_thetas[..., 0] > 28).mean()
    return


@app.cell
def _(plt, post_thetas, trace_p, y):
    for _i in range(8):
        plt.plot([y[_i], post_thetas[..., _i].mean(), trace_p.posterior['theta'].values.mean()], '-o', c='k')
    plt.xticks([0, 1, 2], ['separate', 'hierarchical\n(partial pooling)', 'pooling'])
    plt.xlim(-0.25, 2.25)
    plt.ylabel('Estimated effects')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
