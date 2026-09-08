import os

import torch


class Checkpoint:
    """Keeps the weights of the best scoring network seen so far."""

    def __init__(self,
                 path: str,
                 min_reward: float = 0.0
                 ) -> None:
        self.max_reward = min_reward
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def checkpoint(self,
                   net: torch.nn.Module,
                   reward: float) -> bool:
        """Save the weights if ``reward`` beats the best so far. Returns whether it saved."""
        if reward <= self.max_reward:
            return False
        self.max_reward = reward
        torch.save(net.state_dict(), self.path)
        return True
