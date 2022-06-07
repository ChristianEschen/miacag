import torch
from torch import nn


class debug_3d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # hidden_dim = out_dim
        # self.dimension = dimension
        # self.hidden_dim = hidden_dim
        self.in_dim = in_dim

        self.layer1 = nn.Sequential(
            torch.nn.Conv3d(3, self.in_dim, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
                    nn.Linear(self.in_dim, 1),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.layer1(x)
       # x = x.mean(dim=(-3, -2, -1))
      #  x = self.layer2(x)
        return x
