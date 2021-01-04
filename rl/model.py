import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    """DNN to approximate State Action.
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int
                 ) -> None:
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)