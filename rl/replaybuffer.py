from collections import deque
from typing import List, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self,
                 capacity: int,
                 batch_size: int
                 ) -> None:
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self,
               state: np.array,
               action: int,
               reward: float,
               terminal: int,
               next_state: np.array) -> None:
        self.buffer.append((state, action, reward, terminal, next_state))

    def sample(self) -> Tuple[List[np.array], List[int], List[float], List[int], List[np.array]]:
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, terminals, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return states, actions, rewards, terminals, next_states