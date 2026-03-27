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
    import time
    import numpy as np

    import pymc as pm
    import arviz as az

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
    return np, plt, pm, time


@app.cell
def _(np, pm, time):
    def simulate_problem(N, D):
        sigma_noise = 20000

        def build_model(X, y=None):
            with pm.Model() as model:
                w = pm.Normal('w', mu=np.array([150.0] + [0.0] * (D - 1)), sigma=np.array([100.0] * D))
                mu = pm.math.dot(X, w)
                pm.Normal('obs', mu=mu, sigma=sigma_noise, observed=y)
            return model
        np.random.seed(0)
        x = np.hstack([np.random.uniform(500, 2500, N).reshape(-1, 1), np.random.uniform(0, 100, size=(N, D - 1))])
        true_w = np.array([200] + [0] * (D - 1))
        noise = np.random.normal(0, sigma_noise, size=N)
        y = x @ true_w + noise
        t0 = time.perf_counter()
        with build_model(x, y):
            mcmc_samples = pm.sample(random_seed=0)
        return time.perf_counter() - t0

    return (simulate_problem,)


@app.cell
def _(simulate_problem):
    # N = 20
    _N = 1000
    _D = 1
    # D = 50
    simulate_problem(_N, _D)
    return


@app.cell
def _(simulate_problem):
    _N = 1000
    _D = 50
    simulate_problem(_N, _D)
    return


@app.cell
def _(simulate_problem):
    # N = 10_000
    _N = 1000
    _D = 100
    # D = 50
    simulate_problem(_N, _D)
    return


@app.cell
def _(np, plt, pm, time):
    def simulate_problem_vi(N, D, num_steps):
        sigma_noise = 20000

        def build_model(X, y=None):
            if y is None:
                X = pm.Minibatch(X, batch_size=50)
            else:
                (X, y) = pm.Minibatch(X, y, batch_size=50)
            with pm.Model() as model:
                w = pm.Normal('w', mu=np.array([150.0] + [0.0] * (D - 1)), sigma=np.array([100.0] * D))
                mu = pm.math.dot(X, w)
                pm.Normal('obs', mu=mu, sigma=sigma_noise, observed=y)
            return model
        np.random.seed(0)
        x = np.hstack([np.random.uniform(500, 2500, N).reshape(-1, 1), np.random.uniform(0, 100, size=(N, D - 1))])
        true_w = np.array([200] + [0] * (D - 1))
        noise = np.random.normal(0, sigma_noise, size=N)
        y = x @ true_w + noise
        t0 = time.perf_counter()
        with build_model(x, y):
            vi_approx = pm.fit(n=num_steps)
            vi_samples = vi_approx.sample(4000, random_seed=0)
        plt.plot(vi_approx.hist)
        plt.show()
        return time.perf_counter() - t0

    return (simulate_problem_vi,)


@app.cell
def _(simulate_problem_vi):
    _N = 20
    _D = 1
    simulate_problem_vi(_N, _D, 200000)
    return


@app.cell
def _(simulate_problem_vi):
    _N = 1000
    _D = 50
    simulate_problem_vi(_N, _D, 200000)
    return


@app.cell
def _(simulate_problem_vi):
    _N = 10000
    _D = 100
    simulate_problem_vi(_N, _D, 200000)
    return


if __name__ == "__main__":
    app.run()
