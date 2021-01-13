from typing import Tuple

import numpy as np


class SumTree:
    """Binary tree structure where the value of the parent is the sum of its children.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree_len = 2 * capacity - 1
        self.tree = np.zeros(self.tree_len)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.num_entries = 0

    @property
    def sum_priorities(self) -> float:
        return self.tree[0]

    def _propagate(self, index: int, change: float) -> None:
        """Update change to the root node.
        """
        parent = (index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, index: int, s: float) -> int:
        """Find sample on _leaf_ node.
        """
        left = 2 * index + 1
        right = left + 1

        if left >= self.tree_len:
            return index

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def update(self, index: int, priority: float) -> None:
        """Update priority.
        """
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def add(self, priority: float, data: Tuple) -> None:
        """Store priority and data.
        """
        index = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(index, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.num_entries < self.capacity:
            self.num_entries += 1

    def get(self, s: float) -> Tuple[int, float, Tuple]:
        """Get priority and data.
        """
        tree_index = self._retrieve(0, s)
        data_index = tree_index - self.capacity + 1
        return tree_index, self.tree[tree_index], self.data[data_index]