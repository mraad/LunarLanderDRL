from typing import Tuple

import numpy as np


class SumTree:
    # Binary tree structure where the value of the parent is the sum of its children.
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree_len = 2 * capacity - 1
        self.tree = np.zeros(self.tree_len)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    # update to the root node
    def _propagate(self, index: int, change: float) -> None:
        parent = (index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, index: int, s: float) -> int:
        left = 2 * index + 1
        right = left + 1

        if left >= self.tree_len:
            return index

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    # store priority and data
    def add(self, p: float, data: Tuple) -> None:
        index = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(index, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, index: int, p: float) -> None:
        change = p - self.tree[index]
        self.tree[index] = p
        self._propagate(index, change)

    # get priority and data
    def get(self, s: float) -> Tuple[int, float, Tuple]:
        tree_index = self._retrieve(0, s)
        data_index = tree_index - self.capacity + 1
        return tree_index, self.tree[tree_index], self.data[data_index]