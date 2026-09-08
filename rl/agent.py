import numpy as np
import torch
from torch import nn


class Agent:
    """Epsilon-greedy policy over a Q-network."""

    def __init__(self,
                 net: nn.Module,
                 n_actions: int,
                 device: torch.device
                 ) -> None:
        self.net = net
        self.n_actions = n_actions
        self.device = device
        self.eps = 1.0

    @torch.no_grad()
    def __call__(self, np_state: np.ndarray) -> int:
        if np.random.random() < self.eps:
            return int(np.random.randint(0, self.n_actions))
        pt_state = torch.as_tensor(np_state, dtype=torch.float32, device=self.device)
        return int(self.net(pt_state).argmax().item())
