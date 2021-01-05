import torch.nn as nn


class MLP(nn.Module):
    """DNN to approximate State Action.
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int
                 ) -> None:
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)