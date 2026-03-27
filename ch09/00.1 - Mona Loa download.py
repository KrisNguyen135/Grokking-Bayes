import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml

    co2 = fetch_openml(data_id=41187, as_frame=True)
    co2.frame.head()
    return co2, np, plt


@app.cell
def _(co2):
    data = co2.frame.groupby("year")["co2"].mean().values
    data
    return (data,)


@app.cell
def _(data, plt):
    plt.plot(data);
    return


@app.cell
def _(data, np):
    from pathlib import Path
    np.save(Path(__file__).parent / "co2.npy", data)
    return


if __name__ == "__main__":
    app.run()
