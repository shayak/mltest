import torch
from torch import nn
import matplotlib.pyplot as plt

import os

x = os.getcwd()
print(x)

from pytorch.models import linear_regression


# plot
def plot_predictions(X_train, y_train, X_test, y_test, predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, y_train, c="b", s=4, label="Training data")
    plt.scatter(X_test, y_test, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(X_test, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})


def plot_loss(losses):
    x, y = zip(*enumerate(losses))
    plt.figure(figsize=(10, 7))
    plt.scatter(torch.tensor(x), torch.tensor(y), c="b", s=4, label="Loss curve")
    plt.legend(prop={"size": 14})


def setup_data(config):
    X = torch.arange(config['start'], config['end'], config['step']).unsqueeze(dim=1)
    y = config['weight'] * X + config['bias']

    # train/test split
    tt_split = int(0.8 * len(X))
    X_train, y_train = X[:tt_split], y[:tt_split]
    X_test, y_test = X[tt_split:], y[tt_split:]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def init():
    print(torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    config = {
        'weight': 0.7,
        'bias': 0.3,
        'start': 0,
        'end': 1,
        'step': 0.02,
        'device': device
    }

    data = setup_data(config)

    model = linear_regression.LinearRegressionModel()

    # put both model and data on same device
    model.to(config['device'])
    data['X_train'] = data['X_train'].to(device)
    data['y_train'] = data['y_train'].to(device)
    data['X_test'] = data['X_test'].to(device)
    data['y_test'] = data['y_test'].to(device)

    return model, data


def train(model, data):
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    torch.manual_seed(42)
    epochs = 1000

    for epoch in range(epochs):
        model.train()  # training mode

        y_pred = model(data['X_train'])

        loss = loss_fn(y_pred, data['y_train'])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # test
        model.eval()  # eval mode
        with torch.inference_mode():
            test_pred = model(data['X_test'])
            test_loss = loss_fn(test_pred, data['y_test'])

        if epoch % 100 == 0:
            print(f'Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss}')


def run():
    model, data = init()
    train(model, data)


if __name__ == "__main__":
    run()
