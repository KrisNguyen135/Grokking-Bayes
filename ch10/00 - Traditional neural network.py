import marimo

__generated_with = "0.21.1"
app = marimo.App()


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
    np.random.seed(0)

    N = 120
    x = np.linspace(-3, 3, N).reshape(-1, 1)
    y_true = np.sin(x)

    sigma_true = 0.2
    y = y_true + np.random.normal(0, sigma_true, size=y_true.shape)
    return x, y


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from tqdm import tqdm

    return nn, torch, tqdm


@app.cell
def _(nn, torch):
    class NN(nn.Module):
        def __init__(self, n_hidden):
            super().__init__()
            self.fc1 = nn.Linear(1, n_hidden)
            self.fc2 = nn.Linear(n_hidden, 1)
    
        def forward(self, x):
            hidden = torch.tanh(self.fc1(x))
            mu = self.fc2(hidden)
            return mu

    return (NN,)


@app.cell
def _(x, y):
    x_data = x.astype("float32")
    y_data = y.flatten().astype("float32")

    n_hidden = 20
    return n_hidden, x_data, y_data


@app.cell
def _(NN, n_hidden, nn, torch, tqdm, x_data, y_data):
    model = NN(n_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Convert your data to torch tensors
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    # Training loop
    losses = []

    model.train()
    for epoch in tqdm(range(50_000)):
        optimizer.zero_grad()
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses, model


@app.cell
def _(losses, plt):
    plt.plot(losses)
    plt.title("MSE Loss Convergence")
    plt.show()
    return


@app.cell
def _(losses, plt):
    plt.plot(losses)
    plt.title("MSE Loss Convergence")
    plt.ylim(0, 0.1)
    plt.show()
    return


@app.cell
def _(torch):
    x_test = torch.linspace(-5, 5, 101).reshape(-1, 1)
    y_test = torch.sin(x_test)
    return x_test, y_test


@app.cell
def _(model, torch, x_test):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
    return (predictions,)


@app.cell
def _(plt, predictions, x, x_test, y, y_test):
    plt.scatter(x, y, alpha=0.4, label="Observed data")
    plt.plot(x_test, y_test, color="black", linewidth=2, label="True function")

    plt.plot(x_test, predictions, label="Predictions")

    plt.legend()
    plt.title("Traditional Neural Network")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
