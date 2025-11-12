"""
Episodic Memory with Prioritized Experience Replay

Implements hippocampus-inspired episodic memory for reinforcement learning agents.

Key Features:
- Prioritized experience replay (PER) based on TD-error
- Memory consolidation for offline learning ("sleep" phases)
- Semantic indexing via Vector DB (optional)
- Statistics tracking and analysis

Biological Inspiration:
- Hippocampus: Stores episodic memories (specific events)
- Memory replay: Experiences replayed during sleep for consolidation
- Prioritization: Important events remembered better
- Semantic organization: Retrieve by similarity, not just recency

Based on:
- Schaul et al. (2016): Prioritized Experience Replay
- Stickgold (2005): Memory consolidation during sleep
- Lake et al. (2017): Semantic memory organization

Performance Targets:
- 2-5x data efficiency improvement
- 100K+ experience capacity
- <10ms retrieval time
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from .prioritized_replay_buffer import PrioritizedReplayBuffer


@dataclass
class Experience:
    """
    Single experience tuple.

    Contains all information needed for learning and replay.
    """
    state: np.ndarray
    action: Any  # int for discrete, np.ndarray for continuous
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any]
    timestamp: float

    def to_tuple(self) -> tuple:
        """Convert to tuple for compatibility."""
        return (self.state, self.action, self.reward, self.next_state, self.done)


class EpisodicMemory:
    """
    Episodic memory with prioritized experience replay.

    Stores agent experiences and enables prioritized replay for efficient learning.
    Integrates with Vector DB for semantic retrieval (optional).

    Example:
        >>> memory = EpisodicMemory(capacity=100000, alpha=0.6, beta=0.4)
        >>>
        >>> # Store experience
        >>> memory.store(state, action, reward, next_state, done)
        >>>
        >>> # Sample batch for learning
        >>> batch, indices, weights = memory.sample(batch_size=32)
        >>>
        >>> # Update priorities after learning
        >>> td_errors = agent.compute_td_errors(batch)
        >>> memory.update_priorities(indices, td_errors)
        >>>
        >>> # Consolidate during idle time
        >>> memory.consolidate(agent, num_steps=100)
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        use_semantic_index: bool = False,
        semantic_retriever=None,
    ):
        """
        Initialize episodic memory.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0=uniform, 1=greedy)
            beta: Importance sampling correction (0=none, 1=full)
            epsilon: Small constant for numerical stability
            use_semantic_index: Whether to use Vector DB for semantic retrieval
            semantic_retriever: Optional SemanticRetriever instance
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.use_semantic_index = use_semantic_index

        # Prioritized replay buffer (core storage)
        self.buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon
        )

        # Semantic retriever (optional)
        self.semantic_retriever = semantic_retriever
        if use_semantic_index and semantic_retriever is None:
            # Will be set later via set_semantic_retriever()
            pass

        # Statistics
        self.total_stored = 0
        self.total_sampled = 0
        self.total_consolidated = 0

        # Consolidation tracking
        self.last_consolidation_time = time.time()
        self.consolidation_history = []

    def store(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None
    ):
        """
        Store experience in episodic memory.

        If priority is not provided, uses max priority (optimistic initialization).

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            info: Additional information
            priority: Optional priority (defaults to max_priority)
        """
        # Create experience object
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
            timestamp=time.time()
        )

        # Add to replay buffer
        self.buffer.add(experience, priority=priority)

        # Add to semantic index
        if self.use_semantic_index and self.semantic_retriever is not None:
            embedding = self._compute_experience_embedding(experience)
            self.semantic_retriever.add(experience, embedding)

        self.total_stored += 1

    def sample(
        self,
        batch_size: int = 32,
        beta: Optional[float] = None
    ) -> Tuple[List[Experience], List[int], np.ndarray]:
        """
        Sample batch of experiences using prioritized sampling.

        Returns experiences, indices, and importance sampling weights.

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling correction (overrides self.beta if provided)

        Returns:
            Tuple of (experiences, indices, weights)
            - experiences: List of Experience objects
            - indices: List of buffer indices
            - weights: Importance sampling weights (normalized)

        Raises:
            ValueError: If batch_size > size or buffer is empty
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty memory")

        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer with {len(self.buffer)} experiences"
            )

        # Sample from buffer
        batch, indices, weights = self.buffer.sample(batch_size, beta=beta)

        self.total_sampled += batch_size

        return batch, indices, weights

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on TD errors.

        Priority = |TD_error| + ε

        Args:
            indices: List of buffer indices to update
            td_errors: TD errors for each experience
        """
        # Convert TD errors to priorities
        priorities = np.abs(td_errors)

        # Update in buffer
        self.buffer.update_priorities(indices, priorities)

    def consolidate(self, agent, num_steps: int = 100) -> Dict[str, Any]:
        """
        Perform memory consolidation (offline learning).

        Inspired by sleep-based memory consolidation in biological systems.
        Replays high-priority experiences to strengthen learning.

        Args:
            agent: Agent with learn_from_batch(batch, weights) method
            num_steps: Number of consolidation steps

        Returns:
            Dictionary with consolidation statistics
        """
        if len(self.buffer) == 0:
            return {
                'status': 'skipped',
                'reason': 'empty_buffer'
            }

        start_time = time.time()
        consolidation_losses = []

        for step in range(num_steps):
            # Sample high-priority batch
            batch, indices, weights = self.sample(batch_size=min(32, len(self.buffer)))

            # Learn from batch
            td_errors, loss = agent.learn_from_batch(batch, weights)

            # Update priorities
            self.update_priorities(indices, td_errors)

            consolidation_losses.append(loss)

        elapsed = time.time() - start_time

        # Track consolidation
        result = {
            'status': 'success',
            'steps': num_steps,
            'elapsed': elapsed,
            'mean_loss': np.mean(consolidation_losses),
            'final_loss': consolidation_losses[-1],
            'timestamp': time.time()
        }

        self.consolidation_history.append(result)
        self.total_consolidated += num_steps
        self.last_consolidation_time = time.time()

        return result

    def get_similar_experiences(
        self,
        state: np.ndarray,
        k: int = 5
    ) -> List[Experience]:
        """
        Retrieve k most similar experiences by state similarity.

        Requires semantic_retriever to be set.

        Args:
            state: Query state
            k: Number of similar experiences to retrieve

        Returns:
            List of similar experiences

        Raises:
            ValueError: If semantic retrieval not enabled
        """
        if not self.use_semantic_index or self.semantic_retriever is None:
            raise ValueError("Semantic retrieval not enabled")

        return self.semantic_retriever.search_by_state(state, k=k)

    def update_beta(self, beta: float):
        """
        Update importance sampling correction exponent.

        Typically annealed from 0.4 to 1.0 during training.

        Args:
            beta: New beta value (0 to 1)
        """
        self.beta = beta
        self.buffer.update_beta(beta)

    def set_semantic_retriever(self, retriever):
        """
        Set semantic retriever for similarity-based queries.

        Args:
            retriever: SemanticRetriever instance
        """
        self.semantic_retriever = retriever
        self.use_semantic_index = True

    def clear(self):
        """Clear all memories."""
        self.buffer.clear()
        if self.use_semantic_index and self.semantic_retriever is not None:
            self.semantic_retriever.clear()

        self.total_stored = 0
        self.total_sampled = 0
        self.total_consolidated = 0
        self.consolidation_history = []

    def __len__(self) -> int:
        """Get number of experiences currently stored."""
        return len(self.buffer)

    def __repr__(self) -> str:
        return (
            f"EpisodicMemory(capacity={self.capacity}, size={len(self)}, "
            f"alpha={self.alpha}, beta={self.beta})"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with statistics:
            - capacity: Maximum capacity
            - size: Current size
            - utilization: Fraction of capacity used
            - total_stored: Total experiences ever stored
            - total_sampled: Total samples drawn
            - total_consolidated: Total consolidation steps
            - buffer_stats: Detailed buffer statistics
            - consolidation_stats: Consolidation history
        """
        buffer_stats = self.buffer.get_statistics()

        consolidation_stats = {}
        if self.consolidation_history:
            consolidation_stats = {
                'total_consolidations': len(self.consolidation_history),
                'total_steps': sum(c['steps'] for c in self.consolidation_history),
                'mean_loss': np.mean([c['mean_loss'] for c in self.consolidation_history]),
                'last_consolidation': self.consolidation_history[-1],
                'time_since_last': time.time() - self.last_consolidation_time
            }

        return {
            'capacity': self.capacity,
            'size': len(self),
            'utilization': len(self) / self.capacity if self.capacity > 0 else 0,
            'total_stored': self.total_stored,
            'total_sampled': self.total_sampled,
            'total_consolidated': self.total_consolidated,
            'alpha': self.alpha,
            'beta': self.beta,
            'use_semantic_index': self.use_semantic_index,
            'buffer': buffer_stats,
            'consolidation': consolidation_stats
        }

    def _compute_experience_embedding(self, experience: Experience) -> np.ndarray:
        """
        Compute embedding for experience (for semantic indexing).

        Combines state, action, and reward features.
        Can be overridden for custom embedding logic.

        Args:
            experience: Experience to embed

        Returns:
            Embedding vector
        """
        # Simple concatenation (can be improved with learned embeddings)
        state_features = experience.state.flatten()[:64]  # Limit size
        action_features = np.array([experience.action] if np.isscalar(experience.action) else experience.action).flatten()[:8]
        reward_features = np.array([experience.reward, int(experience.done)])

        # Pad if needed
        state_features = np.pad(state_features, (0, max(0, 64 - len(state_features))))[:64]
        action_features = np.pad(action_features, (0, max(0, 8 - len(action_features))))[:8]

        # Concatenate
        embedding = np.concatenate([state_features, action_features, reward_features])

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def validate(self) -> bool:
        """
        Validate memory invariants (for testing).

        Checks:
        - Buffer is valid
        - Statistics are consistent
        - No negative values

        Returns:
            True if valid

        Raises:
            AssertionError: If invariants violated
        """
        # Validate buffer
        self.buffer.validate()

        # Check sizes
        assert len(self) >= 0, "Negative size"
        assert len(self) <= self.capacity, f"Size {len(self)} exceeds capacity {self.capacity}"

        # Check statistics
        assert self.total_stored >= len(self), "total_stored < current size"
        assert self.total_sampled >= 0, "Negative total_sampled"
        assert self.total_consolidated >= 0, "Negative total_consolidated"

        # Check parameters
        assert 0 <= self.alpha <= 1, f"Alpha {self.alpha} out of range"
        assert 0 <= self.beta <= 1, f"Beta {self.beta} out of range"
        assert self.epsilon > 0, f"Epsilon {self.epsilon} must be positive"

        return True
