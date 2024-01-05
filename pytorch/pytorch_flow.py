import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt

from pathlib import Path

torch.__version__

weight = 0.7
bias = 0.3

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

start, end, step = 0, 1, 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
losses = []

# train / test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# plot_predictions(X_train, y_train, X_test, y_test, None)

# pytorch model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
            requires_grad=True,
            dtype=torch.float
        ))
        self.bias = nn.Parameter(torch.randn(1,
            requires_grad=True,
            dtype=torch.float
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

# torch.manual_seed(42)

model = LinearRegressionModel()

# make prediction using torch.inference_mode()
# try predicting y_test from X_test
with torch.inference_mode():
  y_preds = model(X_test)

# plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# loss function
loss_fn = nn.L1Loss()

# optimizer
optimizer = optim.SGD(params=model.parameters(), lr=0.001)


def training_step():
    # this is batch GD...for stochastic GD need an inner loop traversing each input data point

    model.train()  # set to training mode

    # forward pass
    y_pred = model(X_train)

    # calculate loss
    loss = loss_fn(y_pred, y_train)
    # print(f'Loss: {loss}')

    # optimizer zero grad (reset per epoch, otherwise there's some kind of accumulative effect)
    optimizer.zero_grad()

    # backprop
    loss.backward()

    # step. perform grad descent and update weights
    optimizer.step()

    return loss

epochs = 1  # one loop through the full training data

### training loop (batch gradient descent...not stochastic (individual))
def training_loop(threshold=0.001):
    epoch = 0
    loss = 1
    while abs(loss) > threshold:
        epoch += 1
        loss = training_step()
        losses.append(loss)
        print(f'epoch: {epoch}, loss:{loss}')

training_loop(0.001)

plot_loss(losses)

### testing
model.eval()
with torch.inference_mode():
    y_preds_new = model(X_test)

    # calculate test loss
    test_loss = loss_fn(y_preds_new, y_test)

    # plot preds
    plot_predictions(X_train, y_train, X_test, y_test, y_preds_new)

### save/load models. 3 main methods:
### 1. torch.save() - save pytorch obj in python's pickle format
### 2. torch.load() - load
### 3. torch.nn.Module.load_state_dict() - load a model's saved state dictionary

# create model's directory
MODEL_PATH = Path('pytorch/models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = 'pytorch_flow_model.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME  # pathlib syntax

# save
# torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

# load
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model_preds = loaded_model(X_test)
