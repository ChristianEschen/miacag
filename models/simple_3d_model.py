import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, 3)
        # self.pool = nn.MaxPool3d(2, 2, 2)
        # self.conv2 = nn.Conv3d(16, 32, 2)
        # self.poo2 = nn.MaxPool3d(2, 2, 2)
        self.fc1 = nn.Linear(16*2 * 510 * 510, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 16*2 * 510 * 510)
        x = F.relu(self.fc1(x))
        return x
