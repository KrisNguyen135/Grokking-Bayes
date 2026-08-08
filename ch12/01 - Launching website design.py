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
def _(np):
    n = 500
    k = 63

    a = 1
    b = 1

    baseline = 0.1
    N = 10_000
    V = 100  # fixed value matching the text
    C = 20_000


    def utility_launch(r, V):
        return N * r * V - C

    def utility_no_launch(r, V):
        return np.ones_like(r) * N * baseline * V

    return a, b, k, n, utility_launch, utility_no_launch, V


@app.cell
def _(V, a, b, beta, k, n, np, utility_launch, utility_no_launch):
    r_grid = np.linspace(0, 1, 101)

    launch_utility_grid = utility_launch(r_grid, V)
    no_launch_utility_grid = utility_no_launch(r_grid, V)

    posterior_pdf = beta.pdf(r_grid, a + k, b + n - k)
    return launch_utility_grid, no_launch_utility_grid, posterior_pdf, r_grid


@app.cell
def _(launch_utility_grid, no_launch_utility_grid, plt, posterior_pdf, r_grid):
    utility_color = "tab:blue"
    posterior_color = "tab:orange"

    fig, ax1 = plt.subplots()

    ax1.plot(r_grid, launch_utility_grid, color=utility_color, label="Utility (launch)")
    ax1.axhline(no_launch_utility_grid[0], linestyle=':', color=utility_color, alpha=0.5, label="Utility (no launch)")
    ax1.set_xlabel("Conversion rate r")
    ax1.set_ylabel("Utility ($)", color=utility_color)
    ax1.tick_params(axis='y', labelcolor=utility_color)

    # Posterior axis (right)
    ax2 = ax1.twinx()
    ax2.plot(r_grid, posterior_pdf, linestyle='--', color=posterior_color, label="Density")
    ax2.set_ylabel("Density", color=posterior_color)
    ax2.tick_params(axis='y', labelcolor=posterior_color)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    plt.title("Utility and PDF as functions of conversion rate")
    plt.show()
    return


@app.cell
def _(a, b, k, n, np):
    n_samples = 100_000

    np.random.seed(0)
    r_samples = np.random.beta(a + k, b + n - k, size=n_samples)
    return (r_samples,)


@app.cell
def _(V, r_samples, utility_launch, utility_no_launch):
    launch_utility_samples = utility_launch(r_samples, V)
    no_launch_utility_samples = utility_no_launch(r_samples, V)

    launch_expected_utility = launch_utility_samples.mean()
    no_launch_expected_utility = no_launch_utility_samples.mean()

    (
        launch_expected_utility,
        no_launch_expected_utility,
        launch_expected_utility - no_launch_expected_utility
    )
    return (launch_utility_samples,)


@app.cell
def _(launch_utility_samples, no_launch_utility_grid, plt):
    # Plot distribution of utility
    plt.hist(launch_utility_samples, bins=100, alpha=0.6)
    plt.axvline(no_launch_utility_grid[0], linestyle='--', c="r", label="No launch utility")
    plt.title("Distribution of Utility for Launch Decision")
    plt.legend()
    plt.xlabel("Utility ($)")
    plt.ylabel("Frequency")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("## Explore: how does the value per conversion affect the decision?")
    return


@app.cell
def _(mo):
    V_explore = mo.ui.slider(start=50, stop=150, step=10, value=100, debounce=True, label="Value per conversion V ($)")
    V_explore
    return (V_explore,)


@app.cell
def _(V_explore, a, b, beta, k, n, np, plt, utility_launch, utility_no_launch):
    r_grid_explore = np.linspace(0, 1, 101)
    launch_utility_explore = utility_launch(r_grid_explore, V_explore.value)
    no_launch_utility_explore = utility_no_launch(r_grid_explore, V_explore.value)
    posterior_pdf_explore = beta.pdf(r_grid_explore, a + k, b + n - k)

    # Pin the utility axis to the range spanned across the whole slider (not
    # just the current value), so moving the slider visibly reshapes the
    # curves instead of only relabeling an autoscaled axis.
    _bounds = [
        utility_launch(r_grid_explore, V_explore.start),
        utility_launch(r_grid_explore, V_explore.stop),
        utility_no_launch(r_grid_explore, V_explore.start),
        utility_no_launch(r_grid_explore, V_explore.stop),
    ]
    _y_min = min(_b.min() for _b in _bounds)
    _y_max = max(_b.max() for _b in _bounds)
    _y_margin = 0.05 * (_y_max - _y_min)

    utility_color_explore, posterior_color_explore = "tab:blue", "tab:orange"
    fig2, ax1_explore = plt.subplots()
    ax1_explore.plot(r_grid_explore, launch_utility_explore, color=utility_color_explore, label="Utility (launch)")
    ax1_explore.axhline(no_launch_utility_explore[0], linestyle=':', color=utility_color_explore, alpha=0.5, label="Utility (no launch)")
    ax1_explore.set_xlabel("Conversion rate r")
    ax1_explore.set_ylabel("Utility ($)", color=utility_color_explore)
    ax1_explore.tick_params(axis='y', labelcolor=utility_color_explore)
    ax1_explore.set_ylim(_y_min - _y_margin, _y_max + _y_margin)

    ax2_explore = ax1_explore.twinx()
    ax2_explore.plot(r_grid_explore, posterior_pdf_explore, linestyle='--', color=posterior_color_explore, label="Density")
    ax2_explore.set_ylabel("Density", color=posterior_color_explore)
    ax2_explore.tick_params(axis='y', labelcolor=posterior_color_explore)

    lines_1_explore, labels_1_explore = ax1_explore.get_legend_handles_labels()
    lines_2_explore, labels_2_explore = ax2_explore.get_legend_handles_labels()
    ax1_explore.legend(lines_1_explore + lines_2_explore, labels_1_explore + labels_2_explore, loc="center right")
    plt.title(f"Utility and PDF as functions of r (V = ${V_explore.value})")
    plt.show()
    return


@app.cell
def _(V_explore, mo, r_samples, utility_launch, utility_no_launch):
    launch_eu = utility_launch(r_samples, V_explore.value).mean()
    no_launch_eu = utility_no_launch(r_samples, V_explore.value).mean()
    mo.md(
        f"""
        With V = ${V_explore.value}:

        - Expected utility (launch): **{launch_eu:,.0f}**
        - Expected utility (no launch): **{no_launch_eu:,.0f}**
        - Difference: **{launch_eu - no_launch_eu:,.0f}**
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
