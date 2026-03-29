import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
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
    return np, plt


@app.cell
def _(np):
    def entropy(p):
        p = np.array(p)
        p = p[p > 0]  # avoid log(0)
        return -np.sum(p * np.log2(p))

    return (entropy,)


@app.cell
def _(entropy, np, plt):
    n = 10

    # 1. Uniform
    uniform = np.ones(n) / n

    # 2. Moderately peaked
    moderate = np.array([0.4] + [0.6/(n-1)]*(n-1))

    # 3. Highly concentrated
    peaked = np.array([0.9] + [0.1/(n-1)]*(n-1))

    distributions = [uniform, moderate, peaked]
    titles = ["Uniform", "Moderately Peaked", "Highly Concentrated"]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ax, p, title in zip(axes, distributions, titles):
        x = np.arange(len(p))

        ax.bar(x, p)
        H = entropy(p)

        ax.set_title(f"{title}\n" + r"$H$" + f" = {H:.2f} bits")
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, max(p) * 1.2)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np):
    from scipy.stats import beta

    def differential_entropy(dist, x):
        pdf = dist.pdf(x)
        pdf = np.clip(pdf, 1e-12, None)  # avoid log(0)
        dx = x[1] - x[0]
        return -np.sum(pdf * np.log2(pdf)) * dx  # bits

    return beta, differential_entropy


@app.cell
def _(beta, differential_entropy, np, plt):
    # Grid for evaluation
    _x = np.linspace(0, 1, 1000)

    # Define three Beta distributions
    dists = [
        ("Uniform Beta(1,1)", beta(1, 1)),
        ("Moderate Beta(2,2)", beta(2, 2)),
        ("Concentrated Beta(10,10)", beta(10, 10)),
    ]

    # Plot
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for _ax, (label, dist) in zip(_axes, dists):
        pdf = dist.pdf(_x)
        diff_ent = differential_entropy(dist, _x)

        _ax.plot(_x, pdf)
        _ax.set_title(f"{label}\n" + r"$H$" + f" = {diff_ent:.2f} bits")
        _ax.set_xlabel("x")
        _ax.set_ylabel("Density")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, plt):
    _n = 100

    _uniform = np.ones(_n) / _n
    _x = np.arange(_n) + 1

    plt.bar(_x, _uniform)
    plt.ylim(0, 0.02)
    plt.show()
    return


@app.cell
def _(np, plt):
    _n = 50

    _uniform = np.ones(_n) / _n
    _x = np.arange(_n) + 1

    plt.bar(_x, _uniform)
    plt.plot(np.arange(100) + 1, [0] * 100, alpha=0)
    # plt.xlim(-5, 106)
    plt.ylim(0, 0.04)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
