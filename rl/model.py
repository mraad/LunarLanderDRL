import torch
import torch.nn as nn


class LinearRelu(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(inp_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class MLP(nn.Module):
    """DNN to approximate State -> Action.
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int
                 ) -> None:
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            LinearRelu(n_states, 256),
            LinearRelu(256, 128),
            LinearRelu(128, 64),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
