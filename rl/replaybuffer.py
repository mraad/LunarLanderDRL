from typing import Tuple

import numpy as np

from .sumtree import SumTree


class ReplayBuffer:
    """Fixed capacity ring buffer of transitions held in flat numpy arrays.

    Storing columns rather than tuples keeps a sampled batch contiguous, so it can be
    handed to ``torch.from_numpy`` without a per-element copy.
    """

    def __init__(self,
                 capacity: int,
                 batch_size: int,
                 n_states: int
                 ) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.states = np.zeros((capacity, n_states), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=bool)
        self.next_states = np.zeros((capacity, n_states), dtype=np.float32)
        # Discount to apply to the bootstrapped value of next_state. gamma for a plain
        # one-step transition, gamma ** k once k rewards have been folded into one.
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def append(self,
               state: np.ndarray,
               action: int,
               reward: float,
               terminal: bool,
               next_state: np.ndarray,
               discount: float
               ) -> int:
        """Store a transition, overwriting the oldest one once full. Returns its slot."""
        slot = self.pos
        self.states[slot] = state
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.terminals[slot] = terminal
        self.next_states[slot] = next_state
        self.discounts[slot] = discount
        self.pos = (slot + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return slot

    def _gather(self, indices: np.ndarray) -> Tuple[np.ndarray, ...]:
        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.terminals[indices],
                self.next_states[indices],
                self.discounts[indices])

    def sample(self) -> Tuple[np.ndarray, ...]:
        return self._gather(np.random.randint(0, self.size, self.batch_size))


class PEReplayBuffer(ReplayBuffer):
    """Prioritized experience replay: transitions are drawn proportionally to ``|TD error|``.

    Sampling is stratified over ``batch_size`` equal priority mass segments, and the
    resulting bias is corrected by importance sampling weights annealed through ``beta``.
    """

    def __init__(self,
                 capacity: int,
                 batch_size: int,
                 n_states: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 eps: float = 0.001
                 ) -> None:
        super().__init__(capacity, batch_size, n_states)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.max_priority = 1.0

    @property
    def sum_priorities(self) -> float:
        return self.tree.sum_priorities

    def append(self,
               state: np.ndarray,
               action: int,
               reward: float,
               terminal: bool,
               next_state: np.ndarray,
               discount: float
               ) -> int:
        # A fresh transition has no TD error yet, so give it the highest priority seen so
        # far: it is guaranteed to be replayed at least once, then re-priced from its error.
        slot = super().append(state, action, reward, terminal, next_state, discount)
        self.tree.update(slot, self.max_priority)
        return slot

    def sample(self) -> Tuple[np.ndarray, ...]:
        total = self.tree.sum_priorities
        segment = total / self.batch_size
        offsets = (np.random.uniform(size=self.batch_size) + np.arange(self.batch_size)) * segment

        # Slots past `size` are unwritten and carry zero priority; clamping guards the
        # rounding case where the walk still lands on one.
        retrieve, priority, last = self.tree.retrieve, self.tree.priority, self.size - 1
        slots = [min(retrieve(offset), last) for offset in offsets.tolist()]
        priorities = np.fromiter((priority(slot) for slot in slots), np.float64, self.batch_size)
        indices = np.asarray(slots, dtype=np.int64)

        probabilities = priorities / total
        weights = (self.size * probabilities + 1e-6) ** -self.beta
        weights /= weights.max()

        return (*self._gather(indices), indices, weights.astype(np.float32))

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        priorities = (np.abs(errors) + self.eps) ** self.alpha
        self.max_priority = max(self.max_priority, float(priorities.max()))
        update = self.tree.update
        for index, priority in zip(indices.tolist(), priorities.tolist()):
            update(index, priority)
