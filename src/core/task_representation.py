"""
Task Representation for Transfer Learning (Big Rock 8)

Provides task embedding, similarity computation, and task matching
for intelligent knowledge transfer across related tasks.

Author: MAE Development Team
Date: 2025-11-12
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import hashlib
import json
from collections import defaultdict
import threading


@dataclass
class TaskDescriptor:
    """
    Comprehensive descriptor for a learning task.

    Captures all relevant characteristics needed to:
    1. Compute task similarity
    2. Enable knowledge transfer
    3. Track learning progress

    Attributes:
        task_id: Unique task identifier
        task_type: Type of task (classification, control, optimization, etc.)
        state_dim: Dimensionality of state space
        action_dim: Dimensionality of action space
        reward_range: Min and max possible rewards
        episode_length: Average episode length
        difficulty: Estimated difficulty [0, 1]
        metadata: Additional task-specific information
        state_space_signature: Statistical summary of state distributions
        reward_signature: Reward distribution characteristics
        dynamics_signature: State transition characteristics
    """
    task_id: str
    task_type: str
    state_dim: int
    action_dim: int
    reward_range: Tuple[float, float] = (-1.0, 1.0)
    episode_length: int = 100
    difficulty: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed signatures (set via compute_signatures method)
    state_space_signature: Optional[np.ndarray] = None
    reward_signature: Optional[np.ndarray] = None
    dynamics_signature: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate task descriptor attributes"""
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")

        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")

        if not 0 <= self.difficulty <= 1:
            raise ValueError(f"difficulty must be in [0, 1], got {self.difficulty}")

        if self.reward_range[0] > self.reward_range[1]:
            raise ValueError(f"Invalid reward_range: {self.reward_range}")

    def compute_signatures(
        self,
        state_samples: Optional[List[np.ndarray]] = None,
        reward_samples: Optional[List[float]] = None,
        transition_samples: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ):
        """
        Compute statistical signatures from sample data.

        Args:
            state_samples: List of state vectors
            reward_samples: List of reward values
            transition_samples: List of (state, next_state) pairs
        """
        # State space signature: mean, std, min, max per dimension
        if state_samples and len(state_samples) > 0:
            states = np.array(state_samples)
            self.state_space_signature = np.concatenate([
                np.mean(states, axis=0),
                np.std(states, axis=0),
                np.min(states, axis=0),
                np.max(states, axis=0)
            ])
        else:
            # Default: uniform distribution
            self.state_space_signature = np.zeros(self.state_dim * 4)

        # Reward signature: mean, std, min, max, skewness
        if reward_samples and len(reward_samples) > 0:
            rewards = np.array(reward_samples)
            from scipy import stats
            self.reward_signature = np.array([
                np.mean(rewards),
                np.std(rewards),
                np.min(rewards),
                np.max(rewards),
                stats.skew(rewards) if len(rewards) > 2 else 0.0
            ])
        else:
            # Default: centered in reward range
            mid_reward = (self.reward_range[0] + self.reward_range[1]) / 2
            self.reward_signature = np.array([mid_reward, 1.0, self.reward_range[0], self.reward_range[1], 0.0])

        # Dynamics signature: state change statistics
        if transition_samples and len(transition_samples) > 0:
            deltas = [next_s - s for s, next_s in transition_samples]
            deltas_array = np.array(deltas)
            self.dynamics_signature = np.concatenate([
                np.mean(deltas_array, axis=0),
                np.std(deltas_array, axis=0)
            ])
        else:
            # Default: no dynamics knowledge
            self.dynamics_signature = np.zeros(self.state_dim * 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task descriptor to dictionary"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'reward_range': self.reward_range,
            'episode_length': self.episode_length,
            'difficulty': self.difficulty,
            'metadata': self.metadata,
            'state_space_signature': self.state_space_signature.tolist() if self.state_space_signature is not None else None,
            'reward_signature': self.reward_signature.tolist() if self.reward_signature is not None else None,
            'dynamics_signature': self.dynamics_signature.tolist() if self.dynamics_signature is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskDescriptor':
        """Deserialize task descriptor from dictionary"""
        task = cls(
            task_id=data['task_id'],
            task_type=data['task_type'],
            state_dim=data['state_dim'],
            action_dim=data['action_dim'],
            reward_range=tuple(data['reward_range']),
            episode_length=data['episode_length'],
            difficulty=data['difficulty'],
            metadata=data.get('metadata', {})
        )

        if data.get('state_space_signature'):
            task.state_space_signature = np.array(data['state_space_signature'])
        if data.get('reward_signature'):
            task.reward_signature = np.array(data['reward_signature'])
        if data.get('dynamics_signature'):
            task.dynamics_signature = np.array(data['dynamics_signature'])

        return task


class TaskEmbedding:
    """
    Embed tasks into fixed-size vector space for similarity comparison.

    Combines hand-crafted features with learned representations to create
    a comprehensive task embedding that captures task characteristics.
    """

    def __init__(self, embedding_dim: int = 128):
        """
        Initialize task embedding encoder.

        Args:
            embedding_dim: Dimension of task embeddings
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.embedding_dim = embedding_dim

        # Feature dimensions
        self.basic_feature_dim = 6  # state_dim, action_dim, reward_range, episode_length, difficulty
        self.signature_weight = 0.7  # Weight for signature features vs basic features

        # Cache for computed embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.lock = threading.RLock()

    def encode(self, task: TaskDescriptor) -> np.ndarray:
        """
        Encode task into embedding vector.

        Args:
            task: Task descriptor to encode

        Returns:
            Fixed-size embedding vector (normalized to unit length)
        """
        with self.lock:
            # Check cache
            if task.task_id in self.embedding_cache:
                return self.embedding_cache[task.task_id]

            # Compute embedding
            embedding = self._compute_embedding(task)

            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Cache result
            self.embedding_cache[task.task_id] = embedding

            return embedding

    def _compute_embedding(self, task: TaskDescriptor) -> np.ndarray:
        """
        Compute task embedding from features.

        Strategy:
        1. Normalize basic features (dims, difficulty, etc.)
        2. Concatenate signatures (state, reward, dynamics)
        3. Hash task type to categorical encoding
        4. Combine all features with learned weights
        """
        # Basic features (normalized)
        basic_features = self._encode_basic_features(task)

        # Task type encoding (one-hot or hash-based)
        type_features = self._encode_task_type(task.task_type)

        # Signature features
        signature_features = self._encode_signatures(task)

        # Combine all features
        # Allocate embedding dimensions
        basic_len = len(basic_features)
        type_len = len(type_features)
        sig_len = len(signature_features)

        # Fit everything into embedding_dim
        embedding = np.zeros(self.embedding_dim)

        # Fill in order: basic, type, signature (truncate or pad as needed)
        idx = 0
        for i in range(min(basic_len, self.embedding_dim - idx)):
            embedding[idx] = basic_features[i]
            idx += 1

        for i in range(min(type_len, self.embedding_dim - idx)):
            embedding[idx] = type_features[i]
            idx += 1

        for i in range(min(sig_len, self.embedding_dim - idx)):
            embedding[idx] = signature_features[i]
            idx += 1

        return embedding

    def _encode_basic_features(self, task: TaskDescriptor) -> np.ndarray:
        """
        Encode basic task features with normalization.

        Features:
        - log(state_dim), log(action_dim)
        - normalized reward range
        - log(episode_length)
        - difficulty
        """
        features = np.array([
            np.log10(max(1, task.state_dim)),
            np.log10(max(1, task.action_dim)),
            task.reward_range[0] / 100.0,  # Normalize assuming typical range [-100, 100]
            task.reward_range[1] / 100.0,
            np.log10(max(1, task.episode_length)) / 3.0,  # Normalize to ~[0, 1]
            task.difficulty
        ])

        # Pad to desired dimension
        return features

    def _encode_task_type(self, task_type: str) -> np.ndarray:
        """
        Encode task type using hash-based embedding.

        Uses deterministic hashing to create consistent embeddings
        for the same task type.
        """
        # Hash task type to integer
        type_hash = int(hashlib.md5(task_type.encode()).hexdigest(), 16)

        # Use hash as seed for deterministic random embedding
        rng = np.random.RandomState(type_hash % (2**32))
        embedding = rng.randn(16)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _encode_signatures(self, task: TaskDescriptor) -> np.ndarray:
        """
        Encode task signatures (state space, reward, dynamics).

        Concatenates all signature vectors with normalization.
        """
        signatures = []

        if task.state_space_signature is not None:
            # Normalize state signature
            sig = task.state_space_signature.copy()
            sig = sig / (np.linalg.norm(sig) + 1e-8)
            signatures.append(sig)

        if task.reward_signature is not None:
            sig = task.reward_signature.copy()
            sig = sig / (np.linalg.norm(sig) + 1e-8)
            signatures.append(sig)

        if task.dynamics_signature is not None:
            sig = task.dynamics_signature.copy()
            sig = sig / (np.linalg.norm(sig) + 1e-8)
            signatures.append(sig)

        if signatures:
            combined = np.concatenate(signatures)
            return combined
        else:
            # No signatures available, return zeros
            return np.zeros(10)

    def similarity(self, task1: TaskDescriptor, task2: TaskDescriptor) -> float:
        """
        Compute cosine similarity between two tasks.

        Args:
            task1: First task
            task2: Second task

        Returns:
            Similarity score in [-1, 1] (typically [0, 1] for meaningful tasks)
        """
        emb1 = self.encode(task1)
        emb2 = self.encode(task2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def batch_encode(self, tasks: List[TaskDescriptor]) -> np.ndarray:
        """
        Encode multiple tasks in batch.

        Args:
            tasks: List of task descriptors

        Returns:
            Array of shape (n_tasks, embedding_dim)
        """
        embeddings = []
        for task in tasks:
            embeddings.append(self.encode(task))

        return np.array(embeddings)

    def clear_cache(self):
        """Clear embedding cache"""
        with self.lock:
            self.embedding_cache.clear()

    def get_cache_size(self) -> int:
        """Get number of cached embeddings"""
        with self.lock:
            return len(self.embedding_cache)


class TaskSimilarityMatrix:
    """
    Maintain pairwise similarity scores between tasks.

    Enables fast nearest-neighbor task retrieval for transfer learning.
    Uses efficient caching and indexing strategies.
    """

    def __init__(self, task_embedding: Optional[TaskEmbedding] = None):
        """
        Initialize task similarity matrix.

        Args:
            task_embedding: TaskEmbedding instance (creates default if None)
        """
        self.task_embedding = task_embedding or TaskEmbedding()

        # Task storage
        self.tasks: Dict[str, TaskDescriptor] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

        # Similarity cache: (task_id1, task_id2) -> similarity
        self.similarity_cache: Dict[Tuple[str, str], float] = {}

        # Inverse index: task_type -> set of task_ids
        self.type_index: Dict[str, Set[str]] = defaultdict(set)

        # Performance tracking: task_id -> performance metrics
        self.performance_data: Dict[str, Dict[str, float]] = {}

        # Thread safety
        self.lock = threading.RLock()

    def add_task(
        self,
        task: TaskDescriptor,
        performance_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Add task to similarity matrix.

        Args:
            task: Task descriptor
            performance_metrics: Optional performance data (avg_reward, convergence_time, etc.)
        """
        with self.lock:
            # Encode task
            embedding = self.task_embedding.encode(task)

            # Store task and embedding
            self.tasks[task.task_id] = task
            self.embeddings[task.task_id] = embedding

            # Update type index
            self.type_index[task.task_type].add(task.task_id)

            # Store performance metrics
            if performance_metrics:
                self.performance_data[task.task_id] = performance_metrics

            # Compute similarities with existing tasks
            for other_id in self.tasks:
                if other_id != task.task_id:
                    sim = self._compute_similarity_from_embeddings(
                        embedding,
                        self.embeddings[other_id]
                    )
                    # Cache both directions
                    self.similarity_cache[(task.task_id, other_id)] = sim
                    self.similarity_cache[(other_id, task.task_id)] = sim

    def remove_task(self, task_id: str) -> bool:
        """
        Remove task from matrix.

        Args:
            task_id: Task identifier to remove

        Returns:
            True if task was removed, False if not found
        """
        with self.lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]

            # Remove from storage
            del self.tasks[task_id]
            del self.embeddings[task_id]

            # Remove from type index
            self.type_index[task.task_type].discard(task_id)

            # Remove from performance data
            if task_id in self.performance_data:
                del self.performance_data[task_id]

            # Clear similarity cache entries
            keys_to_remove = [
                key for key in self.similarity_cache
                if task_id in key
            ]
            for key in keys_to_remove:
                del self.similarity_cache[key]

            return True

    def get_similarity(self, task_id1: str, task_id2: str) -> Optional[float]:
        """
        Get cached similarity between two tasks.

        Args:
            task_id1: First task ID
            task_id2: Second task ID

        Returns:
            Similarity score, or None if tasks not found
        """
        with self.lock:
            if task_id1 not in self.tasks or task_id2 not in self.tasks:
                return None

            # Check cache
            cache_key = (task_id1, task_id2)
            if cache_key in self.similarity_cache:
                return self.similarity_cache[cache_key]

            # Compute and cache
            sim = self._compute_similarity_from_embeddings(
                self.embeddings[task_id1],
                self.embeddings[task_id2]
            )
            self.similarity_cache[cache_key] = sim
            self.similarity_cache[(task_id2, task_id1)] = sim

            return sim

    def find_similar_tasks(
        self,
        query_task: TaskDescriptor,
        k: int = 5,
        min_similarity: float = 0.5,
        same_type_only: bool = False,
        exclude_task_ids: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar tasks to query task.

        Args:
            query_task: Task to find similar tasks for
            k: Number of similar tasks to return
            min_similarity: Minimum similarity threshold
            same_type_only: Only return tasks of same type
            exclude_task_ids: Set of task IDs to exclude

        Returns:
            List of (task_id, similarity) tuples, sorted by similarity descending
        """
        with self.lock:
            if not self.tasks:
                return []

            # Get query embedding
            query_emb = self.task_embedding.encode(query_task)

            # Filter candidates
            candidates = set(self.tasks.keys())

            if same_type_only:
                candidates &= self.type_index.get(query_task.task_type, set())

            if exclude_task_ids:
                candidates -= exclude_task_ids

            if query_task.task_id in candidates:
                candidates.remove(query_task.task_id)

            # Compute similarities
            similarities = []
            for task_id in candidates:
                sim = self._compute_similarity_from_embeddings(
                    query_emb,
                    self.embeddings[task_id]
                )

                if sim >= min_similarity:
                    similarities.append((task_id, sim))

            # Sort by similarity descending
            similarities.sort(key=lambda x: x[1], reverse=True)

            return similarities[:k]

    def find_similar_tasks_with_performance(
        self,
        query_task: TaskDescriptor,
        k: int = 5,
        min_similarity: float = 0.5,
        min_performance: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Find similar tasks that also meet performance criteria.

        Args:
            query_task: Task to find similar tasks for
            k: Number of tasks to return
            min_similarity: Minimum similarity threshold
            min_performance: Minimum average reward/performance

        Returns:
            List of (task_id, similarity, performance_metrics) tuples
        """
        with self.lock:
            # Find similar tasks
            similar = self.find_similar_tasks(query_task, k=k*2, min_similarity=min_similarity)

            # Filter by performance
            results = []
            for task_id, sim in similar:
                perf = self.performance_data.get(task_id, {})
                avg_reward = perf.get('avg_reward', 0.0)

                if avg_reward >= min_performance:
                    results.append((task_id, sim, perf))

                if len(results) >= k:
                    break

            return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the similarity matrix.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            if not self.tasks:
                return {
                    'num_tasks': 0,
                    'num_similarities_cached': 0,
                    'task_types': {},
                    'avg_similarity': 0.0
                }

            # Count tasks by type
            type_counts = {
                task_type: len(task_ids)
                for task_type, task_ids in self.type_index.items()
            }

            # Average similarity
            if self.similarity_cache:
                avg_sim = np.mean(list(self.similarity_cache.values()))
            else:
                avg_sim = 0.0

            return {
                'num_tasks': len(self.tasks),
                'num_similarities_cached': len(self.similarity_cache),
                'task_types': type_counts,
                'avg_similarity': float(avg_sim),
                'embedding_dim': self.task_embedding.embedding_dim
            }

    def _compute_similarity_from_embeddings(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """Compute cosine similarity from embeddings"""
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def clear_cache(self):
        """Clear similarity cache"""
        with self.lock:
            self.similarity_cache.clear()

    def __len__(self) -> int:
        """Return number of tasks"""
        return len(self.tasks)

    def __contains__(self, task_id: str) -> bool:
        """Check if task is in matrix"""
        return task_id in self.tasks

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"TaskSimilarityMatrix(tasks={stats['num_tasks']}, "
            f"types={len(stats['task_types'])}, "
            f"cached_similarities={stats['num_similarities_cached']})"
        )


# Convenience functions

def create_task_descriptor(
    task_id: str,
    task_type: str,
    state_dim: int,
    action_dim: int,
    **kwargs
) -> TaskDescriptor:
    """
    Convenience function to create a task descriptor.

    Args:
        task_id: Unique task identifier
        task_type: Type of task
        state_dim: State space dimensionality
        action_dim: Action space dimensionality
        **kwargs: Additional TaskDescriptor parameters

    Returns:
        TaskDescriptor instance
    """
    return TaskDescriptor(
        task_id=task_id,
        task_type=task_type,
        state_dim=state_dim,
        action_dim=action_dim,
        **kwargs
    )


def compute_task_similarity(
    task1: TaskDescriptor,
    task2: TaskDescriptor,
    embedding: Optional[TaskEmbedding] = None
) -> float:
    """
    Convenience function to compute similarity between two tasks.

    Args:
        task1: First task
        task2: Second task
        embedding: Optional TaskEmbedding instance

    Returns:
        Similarity score
    """
    encoder = embedding or TaskEmbedding()
    return encoder.similarity(task1, task2)
