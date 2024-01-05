from torch import nn 

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features=1, out_features=1):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.linear_layer(x)
