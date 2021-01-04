import random
from collections import deque
from typing import List, Tuple

import numpy as np


class Memory:
    def __init__(self,
                 mem_size: int,
                 batch_size: int
                 ) -> None:
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=mem_size)

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
        sample = random.sample(self.buffer, self.batch_size)
        states = []
        actions = []
        rewards = []
        terminals = []
        next_states = []
        for state, action, reward, terminal, next_state in sample:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminal)
            next_states.append(next_state)

        return states, actions, rewards, terminals, next_states