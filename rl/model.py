import torch.nn as nn


class MLP(nn.Module):
    """DNN to approximate State Action.
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int
                 ) -> None:
        super(MLP, self).__init__()
        l1 = nn.Linear(n_states, 256)
        l2 = nn.Linear(256, 128)
        l3 = nn.Linear(128, 64)
        l4 = nn.Linear(64, n_actions)
        nn.init.xavier_uniform_(l1.weight)
        nn.init.xavier_uniform_(l2.weight)
        nn.init.xavier_uniform_(l3.weight)
        nn.init.xavier_uniform_(l4.weight)
        self.net = nn.Sequential(
            l1,
            nn.ReLU(),
            l2,
            nn.ReLU(),
            l3,
            nn.ReLU(),
            l4,
        )

    def forward(self, x):
        return self.net(x)