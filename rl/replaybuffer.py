from collections import deque
from typing import List, Tuple

import numpy as np

from .sumtree import SumTree


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
               terminal: bool,
               next_state: np.array
               ) -> None:
        self.buffer.append((state, action, reward, terminal, next_state))

    def sample(self) -> Tuple[List[np.array], List[int], List[float], List[bool], List[np.array]]:
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, terminals, next_states = zip(
                *[self.buffer[idx] for idx in indices]
        )
        return states, actions, rewards, terminals, next_states


class PEReplayBuffer:
    def __init__(self,
                 capacity: int,
                 batch_size: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 eps: float = 0.001
                 ) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.tree = SumTree(capacity=capacity)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.states = [object] * batch_size
        self.actions = [0] * batch_size
        self.rewards = [0.0] * batch_size
        self.terminals = [False] * batch_size
        self.next_states = [object] * batch_size
        self.indices = [0] * batch_size
        self.priorities = [0.0] * batch_size

    @property
    def sum_priorities(self) -> float:
        return self.tree.sum_priorities

    @property
    def num_entries(self) -> int:
        return self.tree.num_entries

    # @property
    # def priority_factor(self) -> float:
    #     return self.tree.num_entries / self.tree.sum_priorities

    # def _err_to_priority(self, error: float) -> float:
    #     return (np.abs(error) + self.eps) ** self.alpha

    def append(self,
               state: np.array,
               action: int,
               reward: float,
               terminal: bool,
               next_state: np.array,
               error: float = 1000.0
               ) -> None:
        priority = (error + self.eps) ** self.alpha
        data = (state, action, reward, terminal, next_state)
        self.tree.add(priority, data)

    def sample(self) -> Tuple:
        segment = self.tree.sum_priorities / self.batch_size
        priorities = np.random.uniform(size=self.batch_size) * segment
        priorities += np.arange(self.batch_size, dtype=np.float) * segment
        priorities = np.clip(priorities, 0.0, max(self.tree.sum_priorities - 1e-6, 0.0))
        # print(priorities)
        for i, p in enumerate(priorities):
            index, priority, (state, action, reward, terminal, next_state) = self.tree.get(p)
            self.indices[i] = index
            self.priorities[i] = priority  # self.tree.n_entries * priority / self.tree.total
            self.states[i] = state
            self.actions[i] = action
            self.rewards[i] = reward
            self.terminals[i] = terminal
            self.next_states[i] = next_state

        # weights = self.priorities.add(1e-6).power(-self.beta)
        # weights = np.power(self.priorities + 1e-6, -self.beta)
        # weights /= weights.max()

        return self.states, self.actions, self.rewards, self.terminals, self.next_states, self.indices, self.priorities

    # def update(self, index: int, error: float) -> None:
    #     priority = self._err_to_priority(error)
    #     self.tree.update(index, priority)

    def update_priorities(self, indices: np.array, priorities: np.array) -> None:
        for index, priority in zip(indices, priorities):
            # self.update(index, error)
            self.tree.update(index, priority)