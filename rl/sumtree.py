from typing import List


class SumTree:
    """Binary tree structure where the value of the parent is the sum of its children.

    Only priorities are stored: leaf ``i`` lives at ``tree[capacity - 1 + i]`` and maps
    one-to-one onto slot ``i`` of the replay buffer that owns the tree.

    The nodes are a plain Python list rather than a numpy array. Every access here is a
    scalar on a root-to-leaf walk, and list indexing avoids numpy's scalar boxing, which
    measures ~2.5x faster for both ``update`` and ``retrieve``.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.n_nodes = 2 * capacity - 1
        self.tree: List[float] = [0.0] * self.n_nodes

    @property
    def sum_priorities(self) -> float:
        return self.tree[0]

    def priority(self, leaf: int) -> float:
        return self.tree[leaf + self.capacity - 1]

    def update(self, leaf: int, priority: float) -> None:
        """Set the priority of a leaf and propagate the change up to the root."""
        tree = self.tree
        index = leaf + self.capacity - 1
        change = priority - tree[index]
        while True:
            tree[index] += change
            if index == 0:
                return
            index = (index - 1) // 2

    def retrieve(self, s: float) -> int:
        """Return the leaf whose cumulative priority interval contains ``s``."""
        tree = self.tree
        n_nodes = self.n_nodes
        index = 0
        while True:
            left = 2 * index + 1
            if left >= n_nodes:
                return index - self.capacity + 1
            left_sum = tree[left]
            if s <= left_sum:
                index = left
            else:
                s -= left_sum
                index = left + 1
