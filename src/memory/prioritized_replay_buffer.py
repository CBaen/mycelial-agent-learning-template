"""
Prioritized Experience Replay Buffer

Implements prioritized experience replay using a sum tree for efficient sampling.

Key Features:
- Priority-based sampling: P(i) ∝ priority_i^α
- Importance sampling correction: w_i = (N * P(i))^(-β)
- O(log N) sampling and updates via sum tree
- Configurable prioritization strength (α) and bias correction (β)

Algorithm:
1. Store experiences with priority (initially max priority)
2. Sample batch proportional to priority^α
3. Compute importance weights (N * P(i))^(-β) to correct bias
4. Update priorities based on TD errors after learning

Based on: Schaul et al. (2016) "Prioritized Experience Replay", ICLR
Paper: https://arxiv.org/abs/1511.05952

Performance: 2-3x faster convergence in DQN on Atari games
"""

import numpy as np
from typing import Tuple, List, Any, Optional
from .sum_tree import SumTree


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer with sum-tree sampling.

    Experiences are sampled with probability proportional to their priority:
        P(i) = priority_i^α / Σ_k priority_k^α

    Importance sampling weights correct for sampling bias:
        w_i = (N * P(i))^(-β) / max_w

    Parameters:
        α (alpha): Prioritization exponent
            - 0: uniform sampling (no prioritization)
            - 1: fully prioritized (greedy sampling)
            - Typical: 0.6

        β (beta): Importance sampling correction exponent
            - 0: no correction (biased)
            - 1: full correction (unbiased)
            - Typical: start at 0.4, anneal to 1.0

        ε (epsilon): Small constant added to priorities for numerical stability
            - Ensures all experiences have non-zero probability
            - Typical: 1e-6
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0=uniform, 1=greedy)
            beta: Importance sampling correction (0=none, 1=full)
            epsilon: Small constant for numerical stability
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        if not 0 <= beta <= 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        # Sum tree for efficient prioritized sampling
        self.tree = SumTree(capacity)

        # Experience storage (separate from tree)
        self.experiences = []

        # Current write position
        self.position = 0

        # Number of experiences stored
        self.size = 0

        # Track max priority for new experiences
        self.max_priority = 1.0

        # Statistics
        self.total_added = 0
        self.total_sampled = 0

    def add(self, experience: Any, priority: Optional[float] = None):
        """
        Add experience with given priority.

        If priority is None, uses current max priority (optimistic initialization).

        Args:
            experience: Experience tuple (state, action, reward, next_state, done, ...)
            priority: Priority value (optional, defaults to max_priority)
        """
        # Use max priority if not specified
        if priority is None:
            priority = self.max_priority

        # Add epsilon for numerical stability
        priority = max(priority + self.epsilon, self.epsilon)

        # Apply alpha exponent
        priority_alpha = priority ** self.alpha

        # Add to tree
        self.tree.add(priority_alpha, self.position)

        # Store experience
        if self.size < self.capacity:
            self.experiences.append(experience)
        else:
            self.experiences[self.position] = experience

        # Update tracking
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

        # Update max priority
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int, beta: Optional[float] = None) -> Tuple[List[Any], List[int], np.ndarray]:
        """
        Sample batch of experiences using prioritized sampling.

        Sampling algorithm:
        1. Divide total priority into batch_size segments
        2. Sample uniformly from each segment
        3. Retrieve experience from tree using sampled value
        4. Compute importance sampling weights

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling correction (overrides self.beta if provided)

        Returns:
            Tuple of (experiences, indices, weights)
            - experiences: List of sampled experiences
            - indices: List of data indices in buffer
            - weights: Importance sampling weights (normalized)

        Raises:
            ValueError: If batch_size > size (not enough experiences)
        """
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} from buffer of size {self.size}")

        if beta is None:
            beta = self.beta

        batch = []
        indices = []
        priorities = []

        # Total priority sum
        total = self.tree.total()

        if total == 0:
            raise ValueError("Tree has zero total priority")

        # Stratified sampling: divide into segments
        segment = total / batch_size

        for i in range(batch_size):
            # Sample uniformly from segment i
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)

            # Retrieve experience from tree
            data_idx, priority, tree_data_idx = self.tree.get(value)

            # Get experience
            experience = self.experiences[data_idx]

            batch.append(experience)
            indices.append(data_idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities)
        probs = priorities / total

        # w_i = (N * P(i))^(-β)
        weights = (self.size * probs) ** (-beta)

        # Normalize by max weight
        weights /= weights.max()

        # Update statistics
        self.total_sampled += batch_size

        return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Update priorities for experiences at given indices.

        Typically called after learning with TD errors:
            priorities = |TD_errors| + ε

        Args:
            indices: List of data indices to update
            priorities: New priority values (before alpha exponent)

        Raises:
            ValueError: If indices and priorities have different lengths
        """
        if len(indices) != len(priorities):
            raise ValueError(
                f"Indices ({len(indices)}) and priorities ({len(priorities)}) must have same length"
            )

        for idx, priority in zip(indices, priorities):
            # Add epsilon and apply alpha exponent
            priority = max(priority + self.epsilon, self.epsilon)
            priority_alpha = priority ** self.alpha

            # Update in tree
            self.tree.update(idx, priority_alpha)

            # Update max priority
            self.max_priority = max(self.max_priority, priority)

    def update_beta(self, beta: float):
        """
        Update importance sampling correction exponent.

        Typically annealed from 0.4 to 1.0 during training.

        Args:
            beta: New beta value (0 to 1)
        """
        if not 0 <= beta <= 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")
        self.beta = beta

    def get_max_priority(self) -> float:
        """
        Get current maximum priority.

        Useful for initializing new experiences.

        Returns:
            Maximum priority value
        """
        return self.max_priority

    def get_priority(self, index: int) -> float:
        """
        Get priority of experience at given index.

        Args:
            index: Data index

        Returns:
            Priority value (after alpha exponent)
        """
        return self.tree.get_priority(index)

    def __len__(self) -> int:
        """Get number of experiences currently stored."""
        return self.size

    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer(capacity={self.capacity}, size={self.size}, "
            f"alpha={self.alpha}, beta={self.beta}, max_priority={self.max_priority:.2f})"
        )

    def get_statistics(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with statistics:
            - capacity: Maximum capacity
            - size: Current size
            - utilization: Size / capacity
            - total_added: Total experiences added
            - total_sampled: Total experiences sampled
            - max_priority: Current max priority
            - alpha: Prioritization exponent
            - beta: Importance sampling correction
        """
        return {
            'capacity': self.capacity,
            'size': self.size,
            'utilization': self.size / self.capacity if self.capacity > 0 else 0,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'max_priority': self.max_priority,
            'alpha': self.alpha,
            'beta': self.beta,
            'tree_total': self.tree.total(),
        }

    def clear(self):
        """Clear all experiences from buffer."""
        self.tree = SumTree(self.capacity)
        self.experiences = []
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def validate(self) -> bool:
        """
        Validate buffer invariants (for testing).

        Checks:
        - Tree is valid
        - Size matches tree entries
        - All priorities are non-negative

        Returns:
            True if valid

        Raises:
            AssertionError: If invariants violated
        """
        # Validate tree
        self.tree.validate()

        # Check size
        assert self.size == self.tree.n_entries, \
            f"Size mismatch: buffer={self.size}, tree={self.tree.n_entries}"

        # Check experience count
        assert len(self.experiences) >= self.size, \
            f"Not enough experiences: {len(self.experiences)} < {self.size}"

        # Check priorities
        for i in range(self.size):
            priority = self.tree.get_priority(i)
            assert priority >= 0, f"Negative priority at {i}: {priority}"

        return True
