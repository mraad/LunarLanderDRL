import os
import torch


class Checkpoint:
    def __init__(self, path: str) -> None:
        self.max_reward = 0.0
        self.path = path
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def checkpoint(self,
                   net: torch.nn.Module,
                   reward: float) -> None:
        if reward > self.max_reward:
            self.max_reward = reward
            torch.save(net.state_dict(), self.path)