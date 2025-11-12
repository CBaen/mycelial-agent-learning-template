"""
Sum Tree Data Structure

Implements a binary sum tree for efficient O(log N) prioritized sampling.

The sum tree is a complete binary tree where:
- Leaf nodes store priorities of experiences
- Internal nodes store sum of their children's priorities
- Root node stores total sum of all priorities

This enables:
- O(log N) sampling: Sample proportional to priority
- O(log N) updates: Update priority and propagate changes
- O(1) total: Get sum of all priorities

Visual Structure:
```
         Root (sum: 42)
        /              \
   Left (24)        Right (18)
   /    \            /      \
 L1(10) L2(14)   R1(8)   R2(10)
  (exp)  (exp)   (exp)   (exp)
```

Based on: Schaul et al. (2016) "Prioritized Experience Replay"
"""

import numpy as np
from typing import Tuple, Any


class SumTree:
    """
    Binary sum tree for efficient prioritized sampling.

    Uses a complete binary tree stored in an array:
    - Parent of node i is at (i-1)//2
    - Left child of node i is at 2*i + 1
    - Right child of node i is at 2*i + 2

    Tree structure in array:
    [root, left_child, right_child, left_left, left_right, right_left, right_right, ...]

    Capacity: Maximum number of leaf nodes (experiences)
    Tree size: 2*capacity - 1 (all nodes including internal)
    """

    def __init__(self, capacity: int):
        """
        Initialize sum tree with given capacity.

        Args:
            capacity: Maximum number of leaf nodes (experiences to store)
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity

        # Binary tree stored as array
        # Size: 2*capacity - 1 (capacity leaves + capacity-1 internal nodes)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

        # Data stored at leaf nodes (experiences or indices)
        self.data = np.zeros(capacity, dtype=object)

        # Current write position (circular buffer)
        self.write_idx = 0

        # Number of entries currently stored
        self.n_entries = 0

    def _propagate(self, tree_idx: int, change: float):
        """
        Propagate priority change up the tree to root.

        When a leaf priority changes, all ancestor nodes must be updated.
        This maintains the invariant that each node = sum of its children.

        Args:
            tree_idx: Index in tree array where change occurred
            change: Amount of priority change (can be positive or negative)
        """
        # Stop if already at root
        if tree_idx == 0:
            return

        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change

        # Recursively propagate to root
        if parent_idx != 0:
            self._propagate(parent_idx, change)

    def _retrieve(self, tree_idx: int, value: float) -> int:
        """
        Retrieve leaf index by sampling value.

        Given a value in [0, total_priority), traverse tree to find
        the leaf node whose cumulative priority interval contains the value.

        Algorithm:
        1. Start at node
        2. If value <= left_child_sum: go left
        3. Else: subtract left_child_sum from value, go right
        4. Repeat until reaching leaf

        Args:
            tree_idx: Current node index (start at root = 0)
            value: Sample value in [0, total_priority)

        Returns:
            Leaf node index in tree array
        """
        left_child_idx = 2 * tree_idx + 1
        right_child_idx = left_child_idx + 1

        # Base case: reached leaf node
        if left_child_idx >= len(self.tree):
            return tree_idx

        # Recursive case: traverse tree
        left_priority = self.tree[left_child_idx]

        # Special case: if left child is zero, go right
        if left_priority == 0:
            return self._retrieve(right_child_idx, value)

        if value < left_priority:
            # Value in left subtree
            return self._retrieve(left_child_idx, value)
        else:
            # Value in right subtree (adjust value)
            return self._retrieve(right_child_idx, value - left_priority)

    def total(self) -> float:
        """
        Get sum of all priorities (root node value).

        Returns:
            Total sum of all priorities in tree
        """
        return self.tree[0]

    def add(self, priority: float, data: Any):
        """
        Add new experience with given priority.

        Adds to next position in circular buffer, overwriting old data if full.

        Args:
            priority: Priority value (must be >= 0)
            data: Experience data to store
        """
        if priority < 0:
            raise ValueError(f"Priority must be non-negative, got {priority}")

        # Store data at next write position
        data_idx = self.write_idx
        self.data[data_idx] = data

        # Update tree with new priority
        self.update(data_idx, priority)

        # Advance write pointer (circular)
        self.write_idx = (self.write_idx + 1) % self.capacity

        # Track number of entries
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, data_idx: int, priority: float):
        """
        Update priority of experience at given data index.

        Args:
            data_idx: Index in data array (0 to capacity-1)
            priority: New priority value (must be >= 0)
        """
        if priority < 0:
            raise ValueError(f"Priority must be non-negative, got {priority}")

        if data_idx < 0 or data_idx >= self.capacity:
            raise IndexError(f"Data index {data_idx} out of range [0, {self.capacity})")

        # Convert data index to tree index (leaf nodes start at capacity-1)
        tree_idx = data_idx + self.capacity - 1

        # Calculate change in priority
        change = priority - self.tree[tree_idx]

        # Update leaf node
        self.tree[tree_idx] = priority

        # Propagate change up to root
        if change != 0:
            self._propagate(tree_idx, change)

    def get(self, value: float) -> Tuple[int, float, Any]:
        """
        Get experience by sampling with given value.

        Given a value in [0, total_priority), find the experience whose
        priority interval contains this value.

        Args:
            value: Sample value in [0, total_priority)

        Returns:
            Tuple of (data_idx, priority, data)
            - data_idx: Index in data array
            - priority: Priority of the experience
            - data: The stored experience data

        Raises:
            ValueError: If value is out of valid range
        """
        if value < 0 or value > self.total():
            raise ValueError(f"Value {value} out of range [0, {self.total()})")

        # Find leaf node
        tree_idx = self._retrieve(0, value)

        # Convert tree index back to data index
        data_idx = tree_idx - self.capacity + 1

        # Return tuple
        return data_idx, self.tree[tree_idx], self.data[data_idx]

    def get_priority(self, data_idx: int) -> float:
        """
        Get priority of experience at given data index.

        Args:
            data_idx: Index in data array

        Returns:
            Priority value
        """
        if data_idx < 0 or data_idx >= self.capacity:
            raise IndexError(f"Data index {data_idx} out of range [0, {self.capacity})")

        tree_idx = data_idx + self.capacity - 1
        return self.tree[tree_idx]

    def get_max_priority(self) -> float:
        """
        Get maximum priority in tree.

        Useful for initializing new experiences with competitive priority.

        Returns:
            Maximum priority value (1.0 if tree is empty)
        """
        if self.n_entries == 0:
            return 1.0

        # Check all leaf nodes
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries

        return np.max(self.tree[leaf_start:leaf_end])

    def __len__(self) -> int:
        """Get number of entries currently stored."""
        return self.n_entries

    def __repr__(self) -> str:
        return f"SumTree(capacity={self.capacity}, entries={self.n_entries}, total={self.total():.2f})"

    def validate(self) -> bool:
        """
        Validate tree invariants (for testing).

        Checks that each internal node equals sum of its children.

        Returns:
            True if tree is valid

        Raises:
            AssertionError: If tree invariants are violated
        """
        # Check all internal nodes
        for i in range(self.capacity - 1):
            left_child = 2 * i + 1
            right_child = 2 * i + 2

            if right_child < len(self.tree):
                expected_sum = self.tree[left_child] + self.tree[right_child]
                actual_sum = self.tree[i]

                if not np.isclose(expected_sum, actual_sum, rtol=1e-5):
                    raise AssertionError(
                        f"Node {i}: expected {expected_sum:.6f}, got {actual_sum:.6f}"
                    )

        return True
