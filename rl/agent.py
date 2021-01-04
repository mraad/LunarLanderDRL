import numpy as np
import torch
from torch import nn


class Agent:
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
    def __call__(self, np_state: np.array) -> int:
        if np.random.random() < self.eps:
            action = np.random.randint(0, self.n_actions)
        else:
            pt_state = torch.tensor(np_state, device=self.device)
            actions = self.net(pt_state)
            action = torch.argmax(actions).item()

        return action